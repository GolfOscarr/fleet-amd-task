# 02 - Fleet task graph

The graph in the runtime's own terms: a straight-line list of `mpk.*_layer`
calls, each an operator with a task grid, linked to its predecessor by a
shared tensor (`docs/fleet/03-runtime.md`, "The dependency model is a
chain"). Per-op counts are from `sources/graph_counts.py`. The API and its
constraints are recorded in `docs/fleet/04-repo-map.md`.

## The builder, as it will be written

A stand-alone script (`00-decisions.md` D17). Tensor names are the keys the
generated `kernel_0.cu` binds; the same names appear in `04-memory-plan.md`.

```python
mpk = mi.PersistentKernel(mode="online", world_size=1, mpi_rank=0,
                          num_workers=296, num_local_schedulers=8, num_remote_schedulers=0,
                          max_seq_length=1056, max_num_batched_requests=1,
                          max_num_batched_tokens=1, max_num_pages=1, page_size=1056,
                          meta_tensors={...ten tensors...}, profiler_tensor=None,
                          trace_name="", spec_decode_config=None,
                          use_cutlass_kernel=False, eos_token_id=-1)   # all required, no defaults

mpk.embed_layer(input=tok_dummy, weight=W_embed, output=x_res, grid_dim=(1,1,1),
                block_dim=(256,1,1), input_source=0)   # variant: sc1 load of tokens[step];
                                                      # tok_dummy satisfies the two-input registration
for l in range(27):
    mpk.rmsnorm_layer(x_res, w_norm1[l], h, grid_dim=(1,1,1), block_dim=(256,1,1))
    mpk.gang_linear_layer(h, W_qkva[l], qkva, tile_n=24, output_stride=3648)
    mpk.mla_prep_layer(qkva, w_kv_norm[l], W_uk[l], cos, sin,                   # NEW
                       c_kv[l], k_pe[l], ql_nope, q_pe, block_dim=(256,1,1))
    mpk.mla_attend_layer(ql_nope, q_pe, c_kv[l], k_pe[l], partials,            # NEW, gang
                         softmax_scale=0.1147213867929261, split=32)
    mpk.mla_merge_uv_layer(partials, W_uv[l], attn, block_dim=(256,1,1))       # NEW, gang
    mpk.gang_linear_with_residual_layer(attn, W_o[l], x_res, x_res, tile_n=32, output_stride=2048)
    mpk.rmsnorm_layer(x_res, w_norm2[l], h, grid_dim=(1,1,1), block_dim=(256,1,1))
    if l == 0:
        mpk.gang_linear_silu_layer(h, W_gu_shuffled, act, tile_n=64, output_stride=11264)
        mpk.gang_linear_with_residual_layer(act, W_down_pad, x_res, x_res, tile_n=32, output_stride=2048)
    else:
        mpk.moe_router_layer(h, W_gate[l], (topk_w, routing, mask, logits_router), block_dim=(256,1,1))   # NEW
        mpk.gang_moe_w13_linear_layer(h, W13[l], routing, mask, mid)
        mpk.moe_silu_mul_layer(mid, act8, grid_dim=(1,8,1), block_dim=(256,1,1))
        mpk.gang_moe_w2_linear_layer(act8, W2[l], routing, mask, out8)
        mpk.moe_mul_sum_add_layer(out8, topk_w, x_res, x_res, grid_dim=(1,8,1), block_dim=(256,1,1))
mpk.rmsnorm_layer(x_res, w_final_norm, h, grid_dim=(1,1,1), block_dim=(256,1,1))
mpk.gang_linear_layer(h, W_lm, logits, tile_n=64, output_stride=102400)
mpk.argmax_partial_layer(logits, (amax_v, amax_i), grid_dim=(50,1,1), block_dim=(256,1,1))
mpk.argmax_reduce_layer((amax_v, amax_i), tok_out, grid_dim=(1,1,1), block_dim=(256,1,1),
                        output_to_tokens=True)   # variant: new kwarg -> param; kernel writes tokens[step+1];
                                                 # tok_out [1,1] satisfies the output assert
```

Two things this list shows. The residual `x_res` is written in place by the
residual-carrying ops, exactly as the Qwen3 demo does. And every op reads
something the previous op wrote, which is the linking rule; where the natural
graph would have had two consumers of one tensor (`q_proj` and `kv_a_proj`;
router and shared experts) the design fused them (`00-decisions.md` D6, D9).

## Every operator

`Event` is what the runtime creates at the op's output boundary: the number
of events is the gcd of the producer's and consumer's partitions of the shared
tensor, and each event fires when all its producer tasks have signalled.

### Prologue and attention (every layer)

| # | Op | Kernel | Status | Tasks | Tiles per task | Scope | Output partition | Event to next op |
|---|---|---|---|---|---|---|---|---|
| P1 | embed | `embedding` | variant | 1 | - | wavefront | none | 1 event, 1 trigger |
| A1 | `input_layernorm` | `rmsnorm` | reuse | 1 | - | CU | none | 1 event, 1 trigger |
| A2 | `qkv_a_proj` | `gang_linear_mi300` | reuse | 8 | 19 (tile_n 24) | Chiplet | columns by XCD | 1 event, 8 triggers |
| A3 | `mla_prep` | `mla_prep_mi300` | **new** | 1 | - | CU | none | 1 event, 1 trigger |
| A4 | `mla_attend` | `mla_attend_mi300` | **new** | 8 | 5 splits | Chiplet | partials by XCD | 1 event, 8 triggers |
| A5 | `mla_merge_uv` | `mla_merge_uv_mi300` | **new** | 8 | 2 heads | Chiplet | columns by XCD | 1 event, 8 triggers |
| A6 | `o_proj` + residual | `gang_linear_res_mi300` | reuse | 8 | 8 (tile_n 32) | Chiplet | columns by XCD | 1 event, 8 triggers |
| A7 | `post_attention_layernorm` | `rmsnorm` | reuse | 1 | - | CU | none | 1 event, 1 trigger |

### Dense MLP (layer 0)

| # | Op | Kernel | Status | Tasks | Tiles per task | Scope | Event to next op |
|---|---|---|---|---|---|---|---|
| D1 | `gate_up` + SiLU, width 11264 | `gang_linear_silu_mi300` | reuse | 8 | 22 | Chiplet | 1 event, 8 triggers |
| D2 | `down` + residual, K 11264 | `gang_linear_res_mi300` | reuse | 8 | 8 | Chiplet | 1 event, 8 triggers |

### MoE MLP (layers 1-26)

| # | Op | Kernel | Status | Tasks | Tiles per task | Scope | Event to next op |
|---|---|---|---|---|---|---|---|
| M1 | router, top-6 + forced 64, 65 | `moe_router_mi300` | **new** | 1 | - | CU | 1 event, 1 trigger |
| M2 | W13, 8 active experts | `gang_moe_w13_linear_mi300` | reuse | 8 | 44 | Chiplet | 1 event, 8 triggers |
| M3 | SiLU * mul | `moe_silu_mul` | reuse | 8 | - | CU | 1 event, 8 triggers |
| M4 | W2, 8 active experts | `gang_moe_w2_linear_mi300` | reuse | 8 | 32 | Chiplet | 1 event, 8 triggers |
| M5 | weighted sum + residual | `moe_mul_sum_add_mi300` | reuse | 8 | - | CU | 1 event, 8 triggers |

### Head

| # | Op | Kernel | Status | Tasks | Tiles per task | Scope | Event to next op |
|---|---|---|---|---|---|---|---|
| H1 | `model.norm` | `rmsnorm` | reuse | 1 | - | CU | 1 event, 1 trigger |
| H2 | `lm_head` | `gang_linear_mi300` | reuse | 8 | 200 | Chiplet | **2 events**, 4 triggers each (gcd(8, 50) = 2) |
| H3 | `argmax_partial` | `argmax_partial` | reuse | 50 | - | CU | 1 event, 50 triggers |
| H4 | `argmax_reduce` -> `tokens[step+1]` | `argmax_reduce` | variant | 1 | - | wavefront | `EVENT_END_OF_TASK_GRAPH`, 1 trigger |

## Per iteration

| | |
|---|---|
| Operators | 326 (1 + 27 x 7 + 2 + 26 x 5 + 4) |
| Events | 327 (one per op boundary, two after `lm_head`) |
| Tasks | 1,880; 6.35 per worker |
| Ops by status | **217 reuse**, 2 variant, **107 new** |
| Bytes by status | reuse 4,565.0 MiB (96.9%), new 144.9 MiB (3.1%) |

This is the "Fleet-native operations versus remaining fallbacks" metric stated
before implementation: 96.9% of the bytes move through kernels Fleet ships,
and there are no host fallbacks anywhere in the loop. The new kernels are
the MLA path (three per layer) and the router (one per MoE layer).

## Placement

After the prelaunch rewrite the task at graph position `p` is dispatched by
the scheduler that owns worker `(p - 2) mod 296`, i.e. it runs on that
worker's XCD: a non-gang task goes to that XCD's workers round-robin (a
counter that persists across iterations), a gang task is broadcast to the
first `min(tiles, 37)` workers of that XCD (`docs/fleet/03-runtime.md`). With
workers placed round-robin (`w` on XCD `w mod 8`), the XCD is `(p - 2) mod 8`,
eight consecutive gang tasks land on eight distinct XCDs, and which XCD
serves which `bid.x` rotates with the position.
The kernels do not care: each reads its slice from `bid.x`-partitioned
pointers, and the MoE kernels choose their expert from the hardware XCD id.

| Op | What an XCD serves |
|---|---|
| gang linears | 1/8 of the output columns; tiles spread over its 37 workers |
| `mla_attend` | the task with `bid.x = b` serves splits `{b, b+8, b+16, b+24, b+32}` (the last only for `b = 0`) and runs on XCD `(p + b - 2) mod 8` for the op at position `p`: a fixed permutation, the same every iteration; 5 workers busy per XCD, 33 in all |
| `mla_merge_uv` | the task with `bid.x = b` serves heads `{2b, 2b + 1}`, on XCD `(p + b - 2) mod 8`; 2 workers per XCD |
| `gang_moe_w13`, `gang_moe_w2` | active expert `mask[x]`: with 8 active, one expert per XCD; 44 (W13) or 32 (W2) tiles over 37 workers |
| 1-task ops | XCD `(p - 2) mod 8`, on whichever of its workers the round-robin counter names |

The mapping is static across iterations because the graph is identical every
iteration, so an XCD serves the same split indices and head pair every time.
Expert-to-XCD assignment is by the order of `mask`, which the routing task
writes; the design makes that order the top-k order followed by the two
forced experts, so XCDs 6 and 7 always serve the shared halves.

## The new tasks

Each follows the eight-place recipe in `docs/fleet/04-repo-map.md`. Worker
contract: 256 threads, 57 KiB dynamic LDS, `task_desc->input_ptrs[i]` and
`output_ptrs[i]` as `void*`, `tile_idx` for gang kernels, and any runtime
state through `runtime_config` (`step` in particular).

### `mla_prep_mi300` (CU-task, 1 per layer)

| | |
|---|---|
| Inputs | `qkva [1, 3648]` BF16; `w_kv_norm [512]`; `W_uk [16, 128, 512]` BF16 (a view of `kv_b_proj`); `cos`, `sin` `[1056, 64]` |
| Outputs | `c_kv[l][step, :]`, `k_pe[l][step, :]` (cache rows); `ql_nope [16, 512]` BF16; `q_pe [16, 64]` BF16 |
| Reads `step` | yes: cache row index and RoPE position |
| Work | `kv_a_layernorm` over 512 (FP32, eps 1e-6); RoPE on `k_pe_raw` and on 16 x `q_pe`; 16 products `[1,128] x [128,512]` with FP32 accumulation, 2 MiB of `W_uk` read once |
| Resources | LDS: `q [16, 192]` + `c [512]` staged, under 8 KiB; the `W_uk` read is the only bandwidth item: 2 MiB by one workgroup |
| Note | one workgroup streaming 2 MiB is the slowest element of the attention path (about 2 MiB at a single CU's rate); if it shows on the critical path the `W_uk` product moves into `mla_attend` phase A as in the kernel spec |

### `mla_attend_mi300` (Chiplet-task, gang, 8 x 5 tiles)

| | |
|---|---|
| Inputs | `ql_nope [16, 512]`, `q_pe [16, 64]`, `c_kv[l] [1056, 512]`, `k_pe[l] [1056, 64]`; `softmax_scale`; `step` |
| Outputs | `partials [33][16][513]` FP32: `o_acc [16, 512]`, `lse [16]` per split |
| Tile decode | `split = xcd_local_tile * 8 + bid.x` for tile `t` in `[0, 5)`; positions `[32 split, 32 split + 32)` clipped to `step + 1`; tiles with `split >= 33` return without writing (the buffer has 33 slots); a split with `32 split > step` writes `lse = -inf` and returns |
| Work per split | scores `[16, 32]` = `ql_nope . c_kv[tile]^T` (K = 512) + `q_pe . k_pe[tile]^T` (K = 64), times `softmax_scale`; running max and sum in FP32; `acc [16, 512] += p . c_kv[tile]` |
| Instruction | MFMA `16x16x16` BF16 with FP32 accumulate: M = 16 heads fills the tile |
| Resources | LDS: `acc [16, 512]` FP32 = 32 KiB, `c_kv` tile `[32, 512]` BF16 = 32 KiB double-buffered would exceed 57 KiB, so single-buffered with register prefetch of the next tile; VGPR: prefetch depth 4-8 `dwordx4` loads per wave |
| Preferred implementation | CK `BlockFmhaFwdSplitKVPipelineNWarpSShuffleQRKSVS` at `(kM0 = 16, kQKHeaddim = 576, kN1 = 512)` if the machine's `ck_tile` provides it (`00-decisions.md` D12); then `q` is `[ql_nope | q_pe]` `[16, 576]` and `k`, `v` are `[c_kv | k_pe]` and `c_kv` views |
| Fallback | the kernel in `docs/mla-decode/04-our-kernel-spec.md`, phase B |

### `mla_merge_uv_mi300` (Chiplet-task, gang, 8 x 2 tiles)

| | |
|---|---|
| Inputs | `partials`; `W_uv [16, 128, 512]` BF16 (a view of `kv_b_proj`, `04-memory-plan.md`); `step` |
| Outputs | `attn [1, 2048]` BF16, columns `[128 h, 128 h + 128)` for head `h` |
| Tile decode | `h = 2 bid.x + t` for tile `t` in `{0, 1}` |
| Work | over the `ceil((step + 1) / 32)` live splits: `m = max m_j`, `l = sum l_j exp(m_j - m)`, `acc = sum acc_j exp(m_j - m)`; `o = acc / l` `[512]`; `attn[h] = o @ W_uv[h]^T` `[128]`, FP32 accumulation, 128 KiB of `W_uv` per tile |
| Resources | LDS under 4 KiB; 33 x 513 FP32 partial reads per head = 66 KiB |
| Reuse | the rescale is `merge_splitkv_ck_fmha`'s inner loop (`docs/fleet/99-open-questions.md` Q8) |

### `moe_router_mi300` (CU-task, 1 per MoE layer)

| | |
|---|---|
| Inputs | `h [1, 2048]` BF16; `W_gate [64, 2048]` BF16 |
| Outputs | `topk_w [1, 8]` FP32; `routing [66, 1]` int32 (`k + 1` for a selected expert, 0 otherwise); `mask [67]` int32 (8 active expert ids, then the count 8); optionally `logits [64]` FP32 kept for boundary B8 |
| Work | 64 dot products of length 2048 with BF16 inputs and FP32 accumulation (the reference upcasts both operands to FP32 first, and BF16 x BF16 products are exact in FP32, so the only difference is summation order); FP32 softmax; top-6 by repeated argmax with the lower index winning ties (our choice; `torch.topk` with `sorted=False` leaves tie order unspecified, so an exact tie is a divergence to report, not to match); `topk_w[k] = p[idx_k] * routed_scaling_factor`; slots 6, 7 = experts 64, 65 at 1.0 |
| Resources | 256 KiB weight read by one workgroup; four waves each own 16 experts |
| Replaces | `linear` + `moe_topk_softmax_mi300`, whose `renormalize = true` is wrong for this model (`docs/fleet/99-open-questions.md` Q13) |

### Variants (one-line changes to existing tasks)

| Task | Change | Why |
|---|---|---|
| `embedding`, `input_source = 0` | read `tokens[step]` with an agent-scope (`sc1`) load instead of a plain load | the embed is the first op of the iteration and has no dependency event, so it executes no acquire; its input was written by `argmax_reduce` on another XCD in the previous iteration (`03-synchronization.md`) |
| `argmax_reduce` | emit `runtime_config.tokens + runtime_config.step[0] + 1` as the output pointer (one line in `task_register.cc`) plus one Python kwarg passed through as a param; the Python asserts still want an output tensor | online mode does not copy `output_tokens` into `tokens` (`docs/fleet/04-repo-map.md`) |

## The cross-iteration edge

`argmax_reduce` triggers `EVENT_END_OF_TASK_GRAPH` (1 trigger). The scheduler
that receives it runs `prepare_next_batch`, which advances `step` and either
terminates or pushes `TASK_BEGIN_TASK_GRAPH` for iteration `i + 1` to one of
its workers. That task fires event 1, and every scheduler dispatches the
whole graph again with `iteration_num + 1` encoded in the task ids. Nothing
else changes between iterations: same tensors, same tiles, same placement.
The only per-iteration data that crosses the boundary are `tokens[step + 1]`
(written by the reduce, read by the next embed) and the cache rows appended
by each layer's `mla_prep`, which the next iteration's `mla_attend` reads
after a full chain of release-acquire pairs.

## Graph construction constraints, checked by `sources/graph_counts.py`

| Constraint | Source | Our values |
|---|---|---|
| gang linear: `N % 8 == 0`, `(N / 8) % tile_n == 0` | `persistent_kernel.py` asserts | 3648 / 8 = 456 = 19 x 24; 2048 / 8 = 256 = 8 x 32; 102400 / 8 = 12800 = 200 x 64 |
| gang linear SiLU: `(N_gu / 8 / tile_n)` even | **no assert**: `n_weight_tiles // 2` truncates silently (`persistent_kernel.py:1881`); enforced only by `graph_counts.py` | 22528 / 8 / 64 = 44 |
| CK pipeline: `K % 256 == 0` (linear, W13); `K % 128 == 0` (W2) | `gang_moe_*` assert it (`persistent_kernel.py:1337`, `:1387`); `linear_kernel_ck` **truncates silently** (`NumLoopK = K / 256`, `linear_ck_mi300.cuh:319`), enforced only by `graph_counts.py` | 2048, 11264, 512 (`mla_merge_uv`), 1408 |
| `argmax_partial`: `V % slices == 0` | **no assert**: `input.dim(1) // num_tasks` (`persistent_kernel.py:2221`); enforced only by `graph_counts.py` | 102400 / 50 = 2048 |
| MoE routing: `NUM_EXPERTS` power of two in the stock kernel | `moe_topk_softmax_mi300.cuh:85-86` | not applicable to the new router; `routing` and `mask` are sized for 66 |
| every op shares a tensor with its predecessor | `runtime.cc:531` | by construction of the list above |

## What changed from the discovery draft

`docs/fleet/06-our-task-graph.md` had 17 rows and 80 tasks per MoE layer with a
shared-expert branch beside the router. The code read removed the branch
(chain model), fused `q_proj` with `kv_a_proj` (linking rule), fused the
shared experts into the expert set (D6), fused router and top-k (D7), fused
the merge with `W_UV` (D10), and fixed the head's partition (D13). Result: 12
ops and 68 tasks per MoE layer, 326 ops and 1,880 tasks per token.
