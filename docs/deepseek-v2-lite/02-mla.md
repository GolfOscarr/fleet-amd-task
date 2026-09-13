# 02 — Multi-head Latent Attention

This file decides the KV-cache layout, which the task requires the design spec
to define.

## Notation

| Symbol | Value | Meaning |
|---|---|---|
| `H` | 2048 | hidden size |
| `n_h` | 16 | attention heads |
| `d_c` | 512 | `kv_lora_rank`, latent width |
| `d_nope` | 128 | non-positional Q/K head dim |
| `d_rope` | 64 | RoPE head dim (decoupled) |
| `d_v` | 128 | value head dim |
| `d_qk` | 192 | `d_nope + d_rope` |
| `S` | 1024 | context length |

## The reference forward (what HF actually does)

From `sources/modeling_deepseek.py`, `DeepseekV2Attention.forward`:

```python
q = self.q_proj(hidden_states)                      # [B,T,3072]  (q_lora_rank is None)
q = q.view(B, T, 16, 192).transpose(1, 2)
q_nope, q_pe = split(q, [128, 64], dim=-1)

compressed_kv = self.kv_a_proj_with_mqa(hidden_states)     # [B,T,576]
compressed_kv, k_pe = split(compressed_kv, [512, 64], dim=-1)
k_pe = k_pe.view(B, T, 1, 64).transpose(1, 2)              # ONE head, shared

kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv))    # [B,T,4096]
kv = kv.view(B, T, 16, 256).transpose(1, 2)
k_nope, value_states = split(kv, [128, 128], dim=-1)

q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

query_states = concat(q_nope, q_pe)        # [B,16,T,192]
key_states   = concat(k_nope, k_pe)        # [B,16,T,192]   k_pe broadcast to 16 heads
attn = softmax(query_states @ key_states^T * softmax_scale) @ value_states
out  = self.o_proj(attn.reshape(B, T, 2048))
```

Four things to take from this:

1. **`kv_a_layernorm` applies to the 512 latent only.** `k_pe` is split off
   *before* the norm and never normalized.
2. **`k_pe` has a single head** and is broadcast across all 16 — that is the
   "MQA" in `kv_a_proj_with_mqa`, and the "decoupled" in decoupled RoPE.
3. **RoPE is applied to `q_pe` and `k_pe` only** — the 64-wide slices. The
   128-wide `nope` halves never see RoPE. This is what makes absorption
   possible at all.
4. **The reference caches `key_states` and `value_states`** — i.e. the
   *decompressed* tensors, `[B,16,S,192]` and `[B,16,S,128]`. It does **not**
   cache the latent.

## Naive vs absorbed

The scores decompose because RoPE only touches the `pe` halves:

```
q·k = q_nope·k_nope + q_pe·k_pe
```

and `k_nope = W_UK · c_KV` where `c_KV` is the 512-wide latent. So

```
q_nope · k_nope = q_nope · (W_UK · c_KV) = (q_nope · W_UK) · c_KV
                                            \-----------/
                                         absorb into the Q path
```

Likewise `W_UV` folds into `o_proj`: `out = W_O · (W_UV · c_KV weighted) =
(W_O · W_UV) · (c_KV weighted)`. Both halves of `kv_b_proj` `[4096,512]` —
the `k_nope` half and the `v` half — disappear into `q_proj` and `o_proj`.

There are **two ways to realize this**, and they differ enormously in cost:

- **Materialized fusion**: precompute `W_UQ' = W_UQ @ W_UK` and
  `W_O' = W_O @ blockdiag(W_UV)` at load time, so `kv_b_proj` disappears.
- **Runtime reassociation**: keep the original weights, and simply apply
  `W_UK` to `q_nope` (and `W_UV` to the attention output) as separate small
  per-head products each step.

| | Naive (HF) | Materialized fusion | **Runtime reassociation** |
|---|---|---|---|
| Cached per token per layer | 5120 elem | 576 elem | **576 elem** |
| Cache bytes (BF16) | 10,240 B | 1,152 B | **1,152 B** |
| Cache at S=1024, 27 layers | 270 MB | 30.4 MB | **30.4 MB** |
| Attention weights per layer | 26.25 MB | **70.25 MB** | **26.25 MB** |
| Attention traffic/layer/token | 36.25 MB | 71.38 MB | **27.38 MB** |
| × 27 layers | 978.75 MB | 1,927.12 MB | **739.12 MB** |
| `kv_b_proj` at decode | executed | folded away | read, used as `W_UK`/`W_UV` |
| Score matmul | `[16,1,192] × [16,192,S]` | `[16,1,512] × [512,S]` + `[1,64] × [64,S]` | same as fusion |
| Matches reference exactly? | yes | algebraically, not bitwise | algebraically, not bitwise |

**Materialized fusion is nearly 2× worse than doing nothing**: fusing grows the
Q projection from 12 MB to 36 MB and the output projection from 8 MB to 32 MB,
because both expand from the 128-wide per-head dimension to the 512-wide latent.
That costs 44 MB/layer to save 8.9 MB/layer of cache traffic at S=1024.

**Runtime reassociation gets the cache reduction for free** — identical weight
traffic to naive, 8.9× less cache traffic, at the price of 32 tiny GEMVs per
layer that are irrelevant in a memory-bound regime. Derivation in
`05-weights.md`.

**8.89× less cache traffic**, obtainable at zero weight-traffic cost via runtime
reassociation. At S=1024 this is worth 240 MB/token — about 5% of end-to-end
traffic (`07-roofline.md`), not the dominant win the 8.89× ratio suggests. It
becomes decisive at long context.

### Why this matters on MI300X specifically

30.4 MB of latent cache against **32 MB of aggregate L2** (8 XCDs × 4 MB) —
see `../mi300x/01-architecture.md`. The entire KV cache for the full 1,024-token
context nearly fits in on-die L2. The naive 270 MB cannot.

That is the Fleet thesis made concrete for this model: if we shard the latent
cache across chiplets and pin each shard to its XCD's L2, the attention step can
in principle run out of L2 rather than HBM. Whether it *does* depends on
displacement by the 4.6 GB/token of weight traffic streaming through the same
cache — which is the central open question of the design, not a settled result.
See `07-roofline.md` and Q3 in `99-open-questions.md`.

## Decision

**Use the absorbed form via runtime reassociation, with a latent KV cache.**
Do **not** materialize fused `W_UQ'` / `W_O'` weights. Layout:

```
kv_cache[layer][token] = { c_KV : bf16[512],  k_pe : bf16[64] }   // 576 elem, 1152 B
```

Store as two separate contiguous arrays per layer (`c_KV[S][512]` and
`k_pe[S][64]`) rather than interleaved — the two are consumed by different
matmuls with different shapes, and separating them keeps both reads fully
coalesced.

### The prefill→decode interface

The task states that if the reference prefill produces an incompatible cache we
must convert it, document the decision, and exclude the one-time conversion from
measured decode latency. That applies here exactly:

- **HF reference prefill produces** `key_states [16,1024,192]`, `value_states
  [16,1024,128]` per layer — decompressed.
- **Fleet decode needs** `c_KV [1024,512]`, `k_pe [1024,64]` per layer.

Two ways to bridge:

1. **Re-derive the latent from the prompt** (preferred): run
   `kv_a_proj_with_mqa` over the 1,024 prompt hidden states, split, apply RoPE
   to `k_pe`, and store. This is exact — it is what the model computes anyway —
   and needs no inversion.
2. **Invert the decompression** from the HF cache. `W_UK` is `[16·128, 512]`,
   a tall matrix, so this is a least-squares solve, not an exact inverse.
   Avoid it.

Option 1 requires the prefill to expose per-layer hidden states, which HF does
via `output_hidden_states=True`. **Document this as the interface**, and time it
outside the measured window.

Also note: `k_pe` must be stored **post-RoPE**, matching the reference, which
applies RoPE before the cache update. Storing pre-RoPE would require re-rotating
every cached position at every step.

## Numerical caveats

- The reference **upcasts the softmax to FP32** (`dtype=torch.float32`) and casts
  back to BF16. Our kernel must do the same or accept a documented difference.
- Absorption changes the order of operations, so results are algebraically
  equivalent but **not bitwise identical** to the reference. The tolerance for
  this is a decision, not an accident — see `08-correctness.md`.
- Because we reassociate at runtime rather than fusing weights, there is **no
  load-time weight product to validate** — a meaningful correctness risk that
  the materialized variant would have introduced, avoided for free.
- The per-head `q_nope[h] @ W_UK[h]` product accumulates in a different order
  than the reference's `kv_b_proj` decompression, so BF16 rounding differs.
  Accumulate these small products in FP32. See `08-correctness.md`.
- The `softmax_scale` is **0.1147213867929261**, not `192^-0.5`. See
  `01-config.md`.
