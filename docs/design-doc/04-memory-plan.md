# 04 - Memory plan and KV-cache layout

Everything resident on the device during a generation, how it is laid out,
who reads and writes it, and the per-task LDS and register budget. Sizes are
`checkpoint`-verified shapes times 2 bytes (`docs/deepseek-v2-lite/05-weights.md`);
tensor names are the ones the builder in `02-task-graph.md` uses.

## Device memory at a glance

| Region | Size | Lifetime | Written by |
|---|---|---|---|
| Weights, packed (below) | 31.42 GB | whole run | loader, once |
| Latent KV cache, 27 layers | 31.3 MiB | whole run | prefill capture (positions 0..1022), then `mla_prep` (one row per layer per iteration) |
| RoPE tables `cos`, `sin` `[1056, 64]` | 264 KiB | whole run | captured from the reference model |
| Split-KV partial buffer `[33][16][513]` FP32 | 1,058 KiB | reused by every layer | `mla_attend` |
| Activations (below) | under 1 MiB | reused by every layer | tasks |
| Runtime state: task descriptors, events, counters, queues | a few MiB | whole run | `init_persistent_kernel`, `prepare_kernel` |
| Meta tensors: `step`, `tokens [1, 1056]`, ... | KiB | whole run | host before launch; `prepare_next_batch` and `argmax_reduce` during |

Total under 32 GB of the 192 GB. The HF reference model (31 GB) can stay
resident on the same device during capture and be freed before the timed run;
both fit regardless.

## Weights: packed layout

Loaded once from the four safetensors shards into contiguous device tensors,
attached to the graph with `mpk.attach_input(tensor, name)`. Every tensor is
BF16 and row-major `[out, in]` as PyTorch stores `nn.Linear` weights, which
is the orientation every reused kernel expects (`weight [N, K]`, dot along
`K`).

| Name | Shape | Bytes | Built from | Notes |
|---|---|---|---|---|
| `W_embed` | `[102400, 2048]` | 400 MiB | `model.embed_tokens.weight` | read one row per iteration |
| `W_lm` | `[102400, 2048]` | 400 MiB | `lm_head.weight` | untied; gang linear rows split 8 x 12800 |
| `w_final_norm` | `[2048]` | 4 KiB | `model.norm.weight` | |
| per layer `l` in 0..26 | | | | |
| `w_norm1[l]`, `w_norm2[l]` | `[2048]` each | 8 KiB | `input_layernorm`, `post_attention_layernorm` | |
| `W_qkva[l]` | `[3648, 2048]` | 14.25 MiB | `cat(q_proj.weight, kv_a_proj_with_mqa.weight, dim=0)` | rows 0..3071 = `q` (16 heads x 192, head-major, `nope` then `pe` within a head), 3072..3583 = `c`, 3584..3647 = `k_pe_raw`; the gang linear splits rows 8 ways, 456 per XCD |
| `w_kv_norm[l]` | `[512]` | 1 KiB | `kv_a_layernorm.weight` | |
| `W_uk[l]` | `[16, 128, 512]` | 2 MiB | `kv_b_proj.weight.view(16, 256, 512)[:, :128, :]` | a **view** of `kv_b_proj`; per head `h`, `k_nope[h] = W_uk[h] @ c` |
| `W_uv[l]` | `[16, 128, 512]` | 2 MiB | `kv_b_proj.weight.view(16, 256, 512)[:, 128:, :]` | the other half; `mla_merge_uv` computes `o[h] @ W_uv[h]^T` |
| `W_o[l]` | `[2048, 2048]` | 8 MiB | `o_proj.weight` | |
| layer 0 only | | | | |
| `W_gu_shuffled` | `[22528, 2048]` | 88 MiB | `shuffle_tensors([gate_pad, up_pad], groups)` where `gate_pad`, `up_pad` are `gate_proj`, `up_proj` zero-padded from 10944 to 11264 rows | the interleaving `gang_linear_silu` expects (`docs/fleet/04-repo-map.md`) |
| `W_down_pad` | `[2048, 11264]` | 44 MiB | `down_proj.weight` zero-padded from 10944 to 11264 columns | `00-decisions.md` D8 |
| layers 1..26 only | | | | |
| `W_gate[l]` | `[64, 2048]` | 256 KiB | `mlp.gate.weight` | read by the FP32 router |
| `W13[l]` | `[66, 2816, 2048]` | 726 MiB | experts 0..63: `cat(experts.e.gate_proj, experts.e.up_proj, dim=0)`; expert 64: `cat(shared.gate_proj[:1408], shared.up_proj[:1408])`; expert 65: `cat(shared.gate_proj[1408:], shared.up_proj[1408:])` | `gang_moe_w13` reads `[N = 2816, K = 2048]` per expert: rows 0..1407 are `gate`, 1408..2815 are `up`; `moe_silu_mul` splits at 1408 |
| `W2[l]` | `[66, 2048, 1408]` | 363 MiB | experts 0..63: `experts.e.down_proj`; expert 64: `shared.down_proj[:, :1408]`; expert 65: `shared.down_proj[:, 1408:]` | column halves of the shared `down_proj`, exact split of the reduction |

The shared-expert split is exact: `shared(x) = down(silu(gate x) * up x)` with
`gate`, `up` `[2816, 2048]` and `down` `[2048, 2816]`; the element-wise product
splits by rows of `gate`/`up`, and `down`'s sum over 2816 splits into two sums
over 1408 (`00-decisions.md` D6). The MoE combine then adds the two halves
with weight 1.0 in FP32 (`moe_mul_sum_add_mi300.cuh:41-48`).

Per-token traffic through these tensors is in `01-execution-flow.md`:
4,679.5 MiB of weights plus 30.4 MiB of cache. The packed sizes above sum to
the checkpoint's 31,412,968,448 bytes plus 3.75 MiB of layer-0 padding; the
shared-expert re-packing moves bytes but adds none.

## The KV cache

```
c_kv[l] : bf16 [1056][512]     1,081,344 B per layer    post kv_a_layernorm
k_pe[l] : bf16 [1056][ 64]       135,168 B per layer    post RoPE
                                 1,216,512 B per layer;  32,845,824 B = 31.3 MiB for 27 layers
```

- Two separate contiguous arrays per layer, not interleaved and not paged
  (`00-decisions.md` D5). Row `s` is position `s`. Rows 0..1022 are written
  by the prefill capture (`05-prefill-interface.md`); row `step` is written by
  `mla_prep` in iteration `step - 1023`; rows above `step` are never read.
- `mla_attend` reads rows `[0, step]` as 33 tiles of 32 rows; the last live
  tile is clipped to `step + 1`.
- Both arrays are read with ordinary cached loads. Whether they stay in any
  cache across operators is the question `03-synchronization.md` settles
  against L2 (every task's acquire invalidates it) and leaves to the
  Infinity Cache; the loads are not marked non-temporal so that a
  memory-side cache can keep them if it exists.
- The reference stores decompressed `[16, S, 192]` keys and `[16, S, 128]`
  values, 10 KiB per position per layer; ours is 1,152 B, 8.9x smaller
  (`docs/deepseek-v2-lite/02-mla.md`).

## Activations and workspace (`mpk.new_tensor`)

| Name | Shape, dtype | Bytes | Producer -> consumer | Partition on write |
|---|---|---|---|---|
| `x_res` | `[1, 2048]` BF16 | 4 KiB | residual stream; written in place by `o_proj`, `down`, `moe_mul_sum_add` | columns by XCD (gang), or `[H/256]` slices |
| `h` | `[1, 2048]` BF16 | 4 KiB | `rmsnorm` -> next op | whole |
| `qkva` | `[1, 3648]` BF16 | 7 KiB | `qkv_a_proj` -> `mla_prep` | columns by XCD |
| `ql_nope` | `[16, 512]` BF16 | 16 KiB | `mla_prep` -> `mla_attend` | whole |
| `q_pe` | `[16, 64]` BF16 | 2 KiB | `mla_prep` -> `mla_attend` | whole |
| `partials` | `[33, 16, 513]` FP32 | 1,058 KiB | `mla_attend` -> `mla_merge_uv` | splits by XCD |
| `attn` | `[1, 2048]` BF16 | 4 KiB | `mla_merge_uv` -> `o_proj` | columns by XCD (2 heads = 256 columns) |
| `act` | `[1, 11264]` BF16 | 22 KiB | layer 0 `gate_up` -> `down` | columns by XCD |
| `topk_w` | `[1, 8]` FP32 | 32 B | router -> `moe_mul_sum_add` | whole |
| `routing` | `[66, 1]` int32 | 264 B | router -> `W13`, `W2` | whole |
| `mask` | `[67]` int32 | 268 B | router -> `W13`, `W2` | whole |
| `logits_router` | `[1, 64]` FP32 | 256 B | router, kept for boundary B8 | whole |
| `mid` | `[1, 8, 2816]` BF16 | 44 KiB | `W13` -> `moe_silu_mul` | expert slot by XCD |
| `act8` | `[1, 8, 1408]` BF16 | 22 KiB | `moe_silu_mul` -> `W2` | slot |
| `out8` | `[1, 8, 2048]` BF16 | 32 KiB | `W2` -> `moe_mul_sum_add` | slot by XCD |
| `logits` | `[1, 102400]` BF16 | 200 KiB | `lm_head` -> `argmax_partial` | columns by XCD |
| `amax_v`, `amax_i` | `[1, 50]` BF16, int64 | 500 B | `argmax_partial` -> `argmax_reduce` | slice |
| `cos`, `sin` | `[1056, 64]` BF16 | 132 KiB each | captured once; read by `mla_prep` at row `step` | - |

Activations are written with non-temporal stores by the CK linears
(`linear_ck_mi300.cuh:39-45`) so they do not displace anything; at these
sizes it does not matter. `partials` is the one workspace that is not tiny:
33 x 32 KiB written and read once per layer, 2 x 1 MiB per layer of traffic,
counted in `docs/mla-decode/04-our-kernel-spec.md` as 2% of the layer.

One `partials` buffer is enough for all 27 layers because the chain
serializes layers: layer `l`'s `mla_merge_uv` has consumed it before layer
`l + 1`'s `mla_attend` can start (transitively through `o_proj`, the norm,
the MLP, and the next norm).

## Meta tensors (the runtime's contract)

`PersistentKernel` takes ten tensors by name (`persistent_kernel.py:2591-2600`);
all are allocated by our script, and in online mode most are inert.

| Name | Shape, dtype | Value we set | Read by |
|---|---|---|---|
| `step` | `[1]` int32 | 1022 before launch (the seeded `prepare_next_batch` makes it 1023) | `embed`, `mla_prep`, `mla_attend`, `mla_merge_uv`, `argmax_reduce` variant, `prepare_next_batch` |
| `tokens` | `[1, 1056]` int64 | prompt in `[0, 1024)`, zeros after | `embed` (row `step`); `argmax_reduce` writes row `step + 1`; the host reads `[1024, 1056)` after the run |
| `input_tokens`, `output_tokens` | `[1, 1]` int64 | unused (offline-mode plumbing) | - |
| `num_new_tokens` (`new_token_nums` on the C side) | `[1]` int32 | 1 | `prepare_next_batch` (online) |
| `prompt_lengths` | `[1]` int32 | 1024 | offline mode only |
| `qo_indptr_buffer` | `[2]` int32 | `[0, 1]` | gang linears read `[1]` as `num_active_tokens`; **must not be left at the zeros `init_kernel` writes** |
| `paged_kv_indptr_buffer`, `paged_kv_indices_buffer`, `paged_kv_last_page_len_buffer` | `[2]`, `[1]`, `[1]` int32 | zeros | paged attention only; none of our tasks |

The host writes these after `mpk.compile()` (which runs `init_kernel`) and
before `mpk()`; `prepare_kernel` does not touch them.

## Per-task LDS and register budget

Every worker has 57 KiB of dynamic LDS (60 KiB minus 3 KiB the runtime
reserves for its queues, `runtime_header.h:35-42`) and runs 256 threads, four
waves, at one wave per SIMD. The register budget of the megakernel is the
union over all task types, so every kernel we add is measured with
`-Rpass-analysis=kernel-resource-usage` before and after (`docs/fleet/99-open-questions.md` Q3).

| Task | LDS | Registers (per lane, estimate) | Binding resource |
|---|---|---|---|
| CK linears (`gang_linear*`, `gang_moe_*`) at `16 x 64 x 256` | A tile 16 x 256 + B tile 64 x 256 BF16 with padding, 41,984 B = 41.0 KiB (`GemmPipelineSmallTilePolicy`, `linear_ck_mi300.cuh:134-289`) | CK's pipeline; the union's likely maximum today | LDS |
| `mla_attend` (spec kernel) | `acc [16, 512]` FP32 32 KiB + one `c_kv` tile `[32, 512]` BF16 32 KiB would be 64 KiB, over budget; so `acc` in LDS (32 KiB) and the `c_kv` tile streamed through registers in 16-row halves (16 KiB), plus `k_pe` tile 4 KiB and scores 2 KiB: about 54 KiB | prefetch 4-8 `dwordx4` = 16-32 VGPRs, MFMA accumulators 16 | LDS, at the limit; this is `OPEN-PROBLEMS.md` MIN-24 |
| `mla_attend` (CK FMHA, if available) | `DecodePipeline::GetSmemSize()` with a `static_assert` against 57 KiB in the wrapper (`paged_attention_ck_fmha_split_kv_mi300.cuh:118-121`) | CK's | LDS; the assert answers it at compile time |
| `mla_prep` | `q [16, 192]` + `c [512]` staged, 8 KiB | 16 x `[1,128] x [128,512]` accumulations spread over 256 lanes: 32 FP32 per lane | none |
| `mla_merge_uv` | `acc [512]` FP32 + `m`, `l` per split: 4 KiB | `o [512]` then `[512] x [128]`: small | none |
| `moe_router` | `logits [64]` FP32, reductions: under 1 KiB | 2048/256 = 8 BF16 per lane per expert | none |
| `rmsnorm`, `argmax_*`, `moe_silu_mul`, `moe_mul_sum_add` | as shipped, small | as shipped | none |

If the union grows to the point of spills after adding `mla_attend`, the
fallback is to keep `acc` at `[16, 256]` and run the 512-wide `V` dimension
in two passes over the same tile, halving LDS at the cost of reading each
`c_kv` tile twice for the `p . c_kv` product (the scores need only one
pass); that doubles attention's cache traffic from 1.1 to 2.2 MiB per layer,
still under 2% of the layer.

## Cache policy per buffer class

| Class | Load policy | Store policy | Reason |
|---|---|---|---|
| Weights | default; `sc1 nt` (cache-stream, MALL no-allocate) under `USE_NT_WEIGHTS=1` | - | read exactly once per token; the switch is the residency experiment (`06-optimization-strategy.md`) |
| KV cache | default (cached) | plain stores by `mla_prep`, written back by its release flush | the only data with cross-iteration reuse |
| `partials` | default | plain stores | written and read once per layer; no policy helps |
| Activations | default | non-temporal (already in the CK linears) | transient |
| Task-graph state (counters, queues) | the runtime's atomics and volatile loads | the runtime's | untouched |

## Alignment and stride facts the kernels rely on

- All tensors row-major; `attach_input` asserts it (`persistent_kernel.py:473-475`).
- `W_qkva`, `W_lm`, `W_gu_shuffled` rows are split 8 ways on the row index;
  the per-XCD slice is 456, 12800 and 2816 rows respectively, and the tile
  loops inside `linear_kernel_ck` clamp the last N tile.
- `K` of every CK linear is a multiple of 256 (2048, 11264, 512) and of the
  W2 kernel a multiple of 128 (1408); `sources/graph_counts.py` asserts these.
- `c_kv` rows are 1 KiB and `k_pe` rows 128 B, so a 32-row tile is 32 KiB
  and 4 KiB, each a whole number of 16-byte `dwordx4` loads per lane.
- `partials[j]` is `513 x 16` FP32 = 32,832 B; the `lse` for head `h` is the
  513th float of row `h`.
