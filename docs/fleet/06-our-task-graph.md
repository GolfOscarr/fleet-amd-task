# 06 — DeepSeek-V2-Lite Layer 1 as a Fleet Task Graph

This is where the three prior doc sets converge. Nodes come from
`../deepseek-v2-lite/04-tensor-flow.md`; placement from
`../mi300x/02-chiplet-dispatch.md`; edge semantics from
`../mi300x/03-memory-model.md` and `03-runtime.md`.

Configuration: batch 1, S=1024 context, BF16, SPX + NPS1, MI300X (8 XCDs × 38
CUs; 8 schedulers, 296 workers, 37 per chiplet).

## Layer 1 (MoE) task graph

| # | Task | Scope | Count | Weight bytes | Waits on | Existing kernel? |
|---|---|---|---|---|---|---|
| 1 | `input_layernorm` (RMSNorm) | CU-task | 1 | 4 KB | residual | `rmsnorm_mi300` yes |
| 2 | `q_proj` `[3072,2048]` | Chiplet-task | 8 | 12 MB | 1 | `gang_linear_mi300` yes |
| 3 | `kv_a_proj_with_mqa` `[576,2048]` | Chiplet-task | 8 | 2.25 MB | 1 | `gang_linear_mi300` yes |
| 4 | `kv_a_layernorm` (latent only) | Wavefront-task | 1 | 1 KB | 3 | `rmsnorm_mi300` yes |
| 5 | RoPE on `q_pe`, `k_pe` | Wavefront-task | 1 | — | 2, 3 | `rotary_embedding_mi300` yes |
| 6 | latent KV-cache append | Wavefront-task | 1 | — | 4, 5 | `kv_cache_update_mi300` ~ |
| 7 | **MLA scores + softmax + weighted sum, split-KV** | Chiplet-task | 8 | 1.125 MB cache | 6 | **NEW** |
| 8 | **split-KV merge** (rescale + combine partials) | CU-task | 1 | — | 7 | `gang_attention_merge_mi300` ~ |
| 9 | `o_proj` + residual `[2048,2048]` | Chiplet-task | 8 | 8 MB | 8 | `gang_linear_mi300` yes |
| 10 | `post_attention_layernorm` | CU-task | 1 | 4 KB | 9 | `rmsnorm_mi300` yes |
| 11 | router `mlp.gate` `[64,2048]`, **FP32** | CU-task | 1 | 256 KB | 10 | `moe_topk_softmax_mi300` yes |
| 12 | softmax + top-6 (**FP32**, unnormalized) | CU-task | 1 | — | 11 | same yes |
| 13 | shared-expert W13 `[5632,2048]` + SiLU | Chiplet-task | 8 | 22 MB | **10 only** | `gang_rmsnorm_linear` / `silu_mul_linear` yes |
| 14 | shared-expert `down_proj` `[2048,2816]` | Chiplet-task | 8 | 11 MB | 13 | `gang_linear_mi300` yes |
| 15 | routed-expert W13, 6 of 64, + SiLU | Chiplet-task | 8 | 66 MB | 12 | `gang_moe_linear_mi300` yes |
| 16 | routed-expert `down_proj`, 6 of 64 | Chiplet-task | 8 | 33 MB | 15 | `moe_linear_mi300` yes |
| 17 | weighted sum of 6 + shared + residual | Chiplet-task | 8 | — | 14, 16 | `moe_mul_sum_add_mi300` yes |

**Task count = 80 per MoE layer.** Against ~30–40 *kernel launches* per layer in
an eager reference, and against Fleet's 543 tasks/layer for Qwen3-8B (which has
bigger GEMMs and therefore more tiles per Chiplet-task, but the same 8-per-GEMM
structure).

Per-token: 26 MoE layers (80 each) + layer 0 dense (~45) + head (~10)
≈ **2,135 tasks, 1 kernel launch**.

## The two critical paths

```
   (10) post_attn_norm
        ├──────────────► (13) shared W13 ──► (14) shared down ──┐
        │                     33 MB, router-independent          │
        └── (11) router ─► (12) top-6 ─► (15) routed W13 ─► (16) routed down ─┤
                                              99 MB, router-dependent         │
                                                                    (17) combine
```

The shared-expert branch (33 MB) is issuable the instant task 10 completes and
overlaps the entire routing latency. The routed branch (99 MB) cannot start
until top-6 resolves. **That single edge — 12→15 — is the only true
data-dependent stall in the layer**, and it is the thing most worth optimizing.

## Chiplet placement

| Task | Placement | Rationale |
|---|---|---|
| Dense projections (2, 3, 9, 13, 14) | **N-split across 8 XCDs** | Paper: N-split dominates at bs=1–16; no cross-XCD reduction |
| MLA attention (7) | **split-KV, `P_split`=32: 4 splits per XCD, 32 positions each** | 16 heads share one KV read (MQA), so `BLOCK_H`=16 and parallelism comes from splits alone. Sizing in `../mla-decode/04-our-kernel-spec.md` |
| Routed experts (15, 16) | round-robin expert→XCD per `gang_moe_linear_mi300` | Inherited; **but top-6 over 8 XCDs leaves 2 idle** — see below |
| `lm_head` + argmax | N-split over 102,400 rows, 8 partial argmaxes, reduce | `argmax_mi300` exists |

### Two placement concerns specific to us

**1. Expert count vs chiplet count.** Fleet's MoE dispatch assigns
`ae_idx = xcd_id + expert_local_idx * 8`. With `num_experts_per_tok = 6` and 8
XCDs, one round covers 6 experts on 6 chiplets and **2 chiplets idle** — 25% of
the machine unused during the largest single traffic item in the layer (99 MB).
Alternatives: split each expert's N dimension across all 8 XCDs (more uniform,
smaller per-XCD tiles), or give 2 chiplets a second slice of the largest experts.
Worth measuring; this is a genuine adaptation question Fleet's dense evaluation
never had to face.

**2. The latent KV cache is small enough to pin.** Per layer it is 1.125 MiB
(`../deepseek-v2-lite/07-roofline.md`) against 4 MiB per XCD. Under split-KV at
`P_split`=32, each XCD holds 4 splits × 32 positions × 576 × 2 B = **144 KiB** —
trivially resident. Combined with Fleet's cache policy (weights streaming `sc1=1 nt=1`,
activations `NT=1`), the KV slice should survive the weight stream. This is the
one place our model has a locality opportunity Qwen3-8B did not, and it is worth
an explicit experiment.

Note this **inverts Fleet's cache policy rationale**: they mark weights
streaming to protect *other weights* (cooperative M-tile sharing); we would mark
weights streaming to protect the *KV cache*. Same mechanism, different
beneficiary — and ours works at bs=1 where theirs does not.

## Edge semantics

Per `03-runtime.md`, each edge is an event, and the level used depends on where
producer and consumer sit:

- **Within a chiplet** (e.g. 15→16 when both land on the same XCD): per-XCD
  counter, device-scope atomic resolving in local L2, **no fence**.
- **Across chiplets** (e.g. 7→8 merge, 17→next layer): last worker per XCD
  issues one `buffer_wbl2`-bearing fence and a GPU-scope atomic on the global
  counter.

With 80 tasks/layer and 296 workers, naive per-worker global signaling would be
80 × 296 = **23,680** fences per layer; two-level counting makes it 80 × 8 =
**640**. Those land almost exactly on the repo's own note — "Reduces
`buffer_wbl2` from ~24K to ~300" — which is a satisfying independent check that
we have understood the mechanism correctly.

## Milestone mapping

| Milestone | Tasks | @5.3 theo | @4.3 meas | @3.66 consv |
|---|---|---|---|---|
| Single operator (e.g. `q_proj` Chiplet-task) | 8 | ~2.3 µs | 2.9 µs | 3.4 µs |
| **Layer 1 complete (required)** | **80** | **31.6 µs** | **38.9 µs** | **45.7 µs** |
| 4 consecutive MoE layers | 320 | 126 µs | 156 µs | 183 µs |
| All 27 layers + head | ~2,135 | 931 µs | 1,148 µs | 1,348 µs |

Rooflines from `../deepseek-v2-lite/07-roofline.md`; bandwidth band from
`../mi300x/07-achievable-bandwidth.md`. The theoretical column is a floor, not
a target.
