# 04 — Our MLA Decode Kernel Spec

The deliverable. Shapes from `../deepseek-v2-lite/01-config.md`, task model from
`../fleet/02-task-model.md`, bandwidth and prefetch from
`../mi300x/07-achievable-bandwidth.md`, tiling from `03-design-choices.md`.

## Fixed parameters

| Symbol | Value | Source |
|---|---|---|
| `N` query heads | 16 | `num_attention_heads` |
| `Lkv` latent | 512 | `kv_lora_rank` |
| `R` RoPE slice | 64 | `qk_rope_head_dim` |
| `P` nope head dim | 128 | `qk_nope_head_dim` |
| `V` value head dim | 128 | `v_head_dim` |
| `S` context | 1024 | task spec |
| `softmax_scale` | **0.1147213867929261** | `192^-0.5 × mscale²`, YaRN |
| dtype | BF16 in, **FP32 accumulate** | reference |

## Memory layout

```c
// per layer, contiguous, no paging
bf16 c_KV [S][512];     // 1,048,576 B   post kv_a_layernorm
bf16 k_pe [S][ 64];     //   131,072 B   post-RoPE
                        // total 1,152 KiB = 1.125 MiB per layer
```

Split arrays, not interleaved: the two are consumed by differently shaped
matmuls (`BLOCK_DMODEL=512` vs `BLOCK_DPE=64`), and separating them keeps both
reads fully coalesced.

## Three phases

### Phase A — prepare (wavefront-task, ~4K cycles)

```
q      = q_proj(h)                      [16, 192]  -> split 128 nope + 64 pe
c_new, k_pe_new = kv_a_proj(h)          [512], [64]
c_new  = kv_a_layernorm(c_new)          // latent ONLY, not k_pe
RoPE(q_pe, k_pe_new)                    // 64-wide slices only
append c_new -> c_KV[S], k_pe_new -> k_pe[S]
ql_nope[h] = q_nope[h] @ W_UK[h]        // 16 x ([1,128] x [128,512]), FP32 acc
                                        // -> write head-major so q is contiguous
q_mqa = [ql_nope | q_pe]                [16, 576]  logically; kept as two arrays
```

`W_UK` is `kv_b_proj[:, :128, :]` reshaped `[16, 128, 512]` — a **view**, no copy.
`W_UV` is `kv_b_proj[:, 128:, :]` as `[16, 512, 128]`.

### Phase B — attend (Chiplet-task, split-KV)

Each of `P_split` blocks owns a contiguous slice of the sequence and computes,
over all 16 heads:

```
for tile in slice, step BLOCK_N = 32:
    s  = ql_nope [16,512] · c_KV[tile] ^T     // MFMA 16x16x16, K=512
    s += q_pe    [16, 64] · k_pe[tile] ^T     // MFMA, K=64
    s *= softmax_scale
    online-softmax update (running max m, sumexp l) in FP32
    acc += p · c_KV[tile]                     // V headdim = Lkv = 512
emit o_acc [16, 512] FP32 and lse [16] FP32
```

Note the V operand is `c_KV` itself — in MQA form the value *is* the latent.

**`BLOCK_H` must be 16.** All 16 heads share one read of the KV slice; that is
the MQA property and the whole reason MLA is cheap. Splitting heads across blocks
multiplies the cache read by `16 / BLOCK_H` — see the trade-off below.

### Phase C — merge (CU-task)

Standard running-max rescale across `P_split` partials:

```
m   = max(m_a, m_b)
l   = l_a·exp(m_a−m) + l_b·exp(m_b−m)
acc = acc_a·exp(m_a−m) + acc_b·exp(m_b−m)
```

then `o[h] = (acc[h] / l[h]) @ W_UV[h]` giving `[16,128]`, flattened to `[2048]`
for `o_proj`. All rescaling in FP32.

## Split-KV sizing — the decision

Cache read is `1.125 MiB × (16 / BLOCK_H)`. Partial traffic is
`2 × blocks × BLOCK_H × (512+1) × 4 B` (written once, read once at merge).

| `BLOCK_H` | `P_split` | blocks | cache read | partials | total | % of layer |
|---|---|---|---|---|---|---|
| 16 | 2 (vLLM default) | 2 | 1,152 K | 128 K | 1,280 K | 0.79% |
| 16 | 8 (our old plan) | 8 | 1,152 K | 513 K | 1,665 K | 1.03% |
| **16** | **32** | **32** | 1,152 K | 2,052 K | 3,204 K | **1.97%** |
| 16 | 64 | 64 | 1,152 K | 4,104 K | 5,256 K | 3.24% |
| 8 | 32 | 64 | 2,304 K | 2,052 K | 4,356 K | 2.68% |
| 4 | 32 | 128 | 4,608 K | 2,052 K | 6,660 K | 4.10% |

Reading across: at equal block count, **splitting the sequence is cheaper than
splitting heads** (`BLOCK_H=16, P=64` gives 64 blocks for 5,256 K;
`BLOCK_H=8, P=32` gives 64 blocks for 4,356 K — but only because it halves the
splits; push both to 128 blocks and head-splitting loses).

**Choose `BLOCK_H = 16`, `P_split = 32`**, i.e. **4 splits per XCD, 32 positions
each**, aligning with `BLOCK_N = 32` so each split is one KV tile. 32 blocks is
11% of the 296 worker CUs; attention is only 0.71% of layer traffic, so it does
not need the whole machine.

`P_split = 64` (8 per XCD, 16 positions each) is the alternative if measurement
shows attention on the critical path. Sweep it — it is one constant.

## Why not more splits

Naively more blocks look better, but partial traffic grows linearly: `P=296`
would cost 19.5 MB of partials against 1.125 MiB of actual data — **17× write
amplification**, and the merge becomes the bottleneck. The knee is around 32–64.

## Resource budget

| Item | Estimate |
|---|---|
| `o_acc` FP32, 16×512 | 32,768 B — **must live in LDS**, not registers |
| `lse`, `m`, `l` | 16 × 3 × 4 B = 192 B |
| `ql_nope` 16×512 BF16 | 16 KiB — LDS or re-read |
| Per-wave prefetch depth | **4–8** `dwordx4` loads = 16–32 VGPRs (`../mi300x/07`) |
| LDS per workgroup | ~48 KiB of the 64 KiB budget |
| Target occupancy | 1 wave/SIMD (matches vLLM's `waves_per_eu: 1`) |

The 32 KiB FP32 accumulator is the binding resource and is the reason
`BLOCK_H=16` with `Lkv=512` needs LDS rather than registers: 512 FP32 per head
across 16 heads cannot sit in a 512-VGPR budget.

## Instruction selection

- Score and AV matmuls: **`V_MFMA_F32_16X16X16_BF16`** — M=16 fills the tile.
- `q_nope @ W_UK` in phase A: 16 × `[1,128]×[128,512]`, M=1 — **VALU dot-product
  with DPP reduction**, or a padded MFMA; measure (Q2 below).
- Loads: `global_load_dwordx4`, unrolled 4–8 deep.
- Cache policy: KV-cache loads **cached**; weight loads `sc1=1 nt=1`
  (`../acceleration/03-kernel-craft.md`).

## Fleet integration

```
task 7  MLA attend   Chiplet-task x8,  tile_idx -> (xcd, local_split)
task 8  MLA merge    CU-task x1,       reads P_split partials
```

Phase A folds into task 6 (cache append) as a wavefront-task. The partial buffer
is `[P_split][16][513]` FP32 = 32 KiB × `P_split` = **1 MiB at P=32**, allocated
once per layer in the workspace, not per token.

Dependency edges 6→7 and 7→8 cross chiplets, so they carry the agent-scope
release/acquire from `../mi300x/03-memory-model.md`: `buffer_wbl2 sc1` before
publishing partials, `buffer_inv sc1` in the merge before reading them.

## Validation order

1. Phase A alone vs reference `q_nope @ W_UK` — boundary B2/B3.
2. Phase B with `P_split = 1` (no split) vs reference scores — boundary B5.
3. Phase B with `P_split = 32` + phase C vs `P_split = 1` — isolates the merge.
4. Full task vs reference attention output — boundary B6.

Thresholds from `../deepseek-v2-lite/08-correctness.md`.
