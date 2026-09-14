# 02 — Kernel Anatomy

## vLLM's own derivation

`mla_attention.py` carries a long comment block deriving both formulations. It
is the clearest statement of MLA we have found, and worth reproducing in its own
notation (`N` heads, `P` = `qk_nope_head_dim`, `R` = `qk_rope_head_dim`,
`Lkv` = `kv_lora_rank`, `V` = `v_head_dim`).

### "Compute Friendly" — `forward_mha`

```
k_nope = (kv_c @ W_UK.view(Lkv, N * P)).view(Skv, N, P)
v      = (kv_c @ W_UV.view(Lkv, N * V)).view(Skv, N, V)

// MHA with QK headdim = P + R, V headdim = V
sdpa_o = sdpa(cat([q_nope, q_pe]), cat([k_nope, k_pe.expand(N)]), v)
return sdpa_o @ W_O
```

Decompresses the latent to full per-head K and V. Used for **prefill**.

### "Data-Movement Friendly" — `forward_mqa`

```
ql_nope = einsum("snh,lnh->snl", q_nope, W_UK)      # <-- runtime, not fused

// MQA with QK headdim = Lkv + R, V headdim = Lkv
sdpa_o = sdpa(cat([ql_nope, q_pe]), cat([kv_c, k_pe]), kv_c)

o = einsum("snl,lnv->snv", sdpa_o.reshape(-1, N, Lkv), W_UV)
return o.view(-1, N * V) @ W_O
```

With vLLM's own comment on the trade-off:

> "NOTE: this is less compute-friendly since Lkv > P but is more data-movement
> friendly since its MQA vs MHA"

**This is used for decode, and it is exactly our chosen approach.** Note `W_UK`
and `W_UV` appear as *operands* of runtime einsums, never pre-multiplied into
`W_UQ` or `W_O`.

### The MQA reframing

The decode attention is a **multi-query** attention: a single KV "head" of width
576 (QK) / 512 (V), against `N=16` query heads. FlashMLA uses identical language:

> "MQA stands for Multi-Query Attention mode (i.e. `head_dim_k` = 576 ... with
> `head_dim_v` = 512), while MHA stands for ... `head_dim_k` = 192 / 128 with
> `head_dim_v` = 128."

This matters more than it first appears. The 16 heads are **not** 16 independent
problems — they are the **M dimension of a GEMM sharing one KV read**. That is
the property that makes MLA cheap, and the property that decides our tiling.

## The decode kernel, in three phases

Both vLLM paths and Fleet's own `kv_cache_update_mi300.cuh` use the same
decomposition:

| Phase | Work | Cost |
|---|---|---|
| **A** — prepare | RoPE on `q_pe` and new `k_pe`; append `c_KV`/`k_pe` to cache; stage Q | ~3.8K cycles (Fleet's comment) |
| **B** — attend | split-KV: each block computes partial `o_acc` + `lse` over its slice | the bulk |
| **C** — merge | rescale and combine partials across splits | small |

Our MLA Chiplet-task should follow the same split. Phase A is cheap and
wavefront-scoped; phase B is the Chiplet-task proper; phase C is a CU-task.

## Phase B tiling, from the Triton grouped kernel

`_decode_grouped_att_m_fwd` is the path taken when `kv_group_num != 1`, i.e. MLA.

```python
if Lk == 576:                 # our case
    BLOCK_DMODEL = 512        # the latent
    BLOCK_DPE   = 64          # the decoupled RoPE slice
BLOCK_DV = 512
BLOCK    = 32                 # KV tile along the sequence
BLOCK_H  = 16                 # query heads per block
grid = (batch, cdiv(head_num, min(BLOCK_H, kv_group_num)), NUM_KV_SPLITS)
```

Four things to take:

1. **`nope` and `pe` are separate tiles**, `512` and `64` — not one 576 operand.
   Their score contributions are computed separately and summed.
2. **`BLOCK_H = 16`** — a block covers 16 query heads, reading the KV slice once
   for all of them.
3. **`BLOCK = 32`** KV positions per inner iteration on this path (the
   non-grouped path uses `BLOCK = 64 if not is_hip_ else 8` — ROCm prefers much
   smaller KV tiles).
4. The accumulator is `tl.zeros([BLOCK_DV])` in **FP32**, with `lse` stored
   alongside at offset `Lv` in the partial buffer.

## ROCm-specific tuning

```python
extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
```

- `matrix_instr_nonkdim: 16` — use **MFMA 16×16** instructions.
- `waves_per_eu: 1` — **one wave per execution unit**, the same occupancy Fleet
  reports for its megakernel. So a tuned MLA decode kernel already runs at the
  occupancy we were worried about, and does so deliberately.

## Split-count heuristic

```python
_MIN_WORK_PER_SPLIT = 512
_SPLIT_OCCUPANCY_MULTIPLIER = 2

ideal_splits = next_power_of_2(max(1, max_seq_len // _MIN_WORK_PER_SPLIT))
max_splits   = sm_count * _SPLIT_OCCUPANCY_MULTIPLIER
num_kv_splits = min(ideal_splits, max_splits)
```

At `max_seq_len = 1024` this gives **2 splits**. Combined with `BLOCK_H = 16`
and `head_num = 16`, the grid is `(1, 1, 2)` — **two workgroups**.

That is correct for vLLM's target regime, where `batch` supplies parallelism.
It is pathological at batch 1, and it is why we cannot simply adopt their
numbers. See `04-our-kernel-spec.md`.
