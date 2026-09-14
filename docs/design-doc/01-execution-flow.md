# 01 - Model execution flow

One generation, then one decode iteration op by op, in the order the Fleet
runtime will execute it. Shapes are `checkpoint`-verified
(`docs/deepseek-v2-lite/01-config.md`); the op order follows the chain
dependency model of the runtime (`docs/fleet/03-runtime.md`); counts and bytes
are produced by `sources/graph_counts.py`, not typed.

## The run

```
host                                          GPU
----                                          ---
1. reference prefill (HF, BF16, 1,024 tokens)  ->  logits, per-layer module outputs
2. capture latent cache for positions 0..1022  ->  c_KV[l][0..1023), k_pe[l][0..1023)   (05-prefill-interface.md)
3. set step=1022, num_new_tokens=1,
   qo_indptr=[0,1], tokens[0..1024) = prompt
4. mpk()                                       ->  prepare_kernel
                                                   worker_kernel (296 x 256 threads)   \  run until
                                                   scheduler_kernel (8 x 128 threads)  /  step 1055
5. read tokens[1024..1056) = 32 outputs
```

Steps 1-3 are outside the timed window. Step 4 is one `mpk()` call and three
kernel dispatches; inside it, the worker kernel runs **32 iterations** of the
task graph. Step 5 is the correctness check (`07-correctness.md`).

## Iteration state

| Name | Where | Iteration `i` (0..31) |
|---|---|---|
| `step` | meta tensor, advanced by `prepare_next_batch` | `1023 + i` |
| input token | `tokens[step]` | prompt token 1023 for `i = 0`, output `i - 1` after |
| RoPE position | `= step` | `1023 + i` |
| cache write index | `= step` | the new position |
| positions attended | `0 .. step` | `S_eff = 1024 + i` |
| output | `tokens[step + 1]` | output token `i` |

The seeded end-of-graph event runs `prepare_next_batch` once before iteration
0, which moves the host's `step = 1022` to 1023 (`docs/fleet/03-runtime.md`,
`00-decisions.md` D14). After iteration 31, `step = 1054` satisfies
`step + 2 >= max_seq_length = 1056` and the schedulers terminate the workers.

## Constants every kernel must agree on

| Constant | Value | Source |
|---|---|---|
| `softmax_scale` | `0.1147213867929261` = `192^-0.5 x mscale^2`, `mscale = 0.1 x 0.707 x ln(40) + 1` | `modeling_deepseek.py:745-751`, `01-config.md` |
| RoPE `cos`/`sin` | the model's own `rotary_emb` tables (YaRN, factor 40, `rope_theta` 10000), **unscaled** because `mscale == mscale_all_dim` makes the table multiplier 1.0; rows 0..1055 attached as inputs | `modeling_deepseek.py:316-326` |
| RMSNorm `eps` | `1e-6` for `input_layernorm`, `post_attention_layernorm`, `kv_a_layernorm`, `model.norm`; Fleet's `rmsnorm` emits exactly this | `01-config.md`; `task_register.cc:122` |
| Router | logits and softmax in FP32 from BF16 inputs; top-6 of 64; weights **not** renormalized; `routed_scaling_factor = 1.0` | `03-moe.md` |
| Forced experts | slots 6 and 7 hold experts 64 and 65 (the two halves of the shared MLP) at weight 1.0 | `00-decisions.md` D6 |
| Precision | BF16 storage everywhere; FP32 accumulation in every matmul; FP32 softmax; FP32 partials and merge | reference |

## One iteration

Every op reads a tensor the previous op wrote; that is the runtime's linking
rule. `x` is the residual stream, `[1, 2048]` BF16.

### Prologue

| # | Op | Kernel | Tasks | Tiles/task | Weight MiB | Cache MiB |
|---|---|---|---|---|---|---|
| 1 | `x = embed_tokens[tokens[step]]` | `embedding (input_source=0)` | 1 | - | 0.004 | 0.000 |

### Layer 0 (dense)

| # | Op | Kernel | Tasks | Tiles/task | Weight MiB | Cache MiB |
|---|---|---|---|---|---|---|
| 1 | `h = input_layernorm(x)` | `rmsnorm` | 1 | - | 0.004 | 0.000 |
| 2 | `qkva = h @ [W_q ; W_kva]^T` -> `[1, 3648]` = `q [16 x 192]` then `c [512]`, `k_pe_raw [64]` | `gang_linear_mi300` | 8 | 19 | 14.250 | 0.000 |
| 3 | `mla_prep`: `c_KV[step] = kv_a_layernorm(c)`; `k_pe[step] = RoPE(k_pe_raw, step)`; `q_pe = RoPE(q[:,128:], step)`; `ql_nope[h] = q[h,:128] @ W_UK[h]` -> `[16, 512]` | `mla_prep_mi300 (NEW)` | 1 | - | 2.001 | 0.000 |
| 4 | `mla_attend`: per split `j` of 32 positions, `s = (ql_nope . c_KV^T + q_pe . k_pe^T) x scale`, online softmax, `acc += p . c_KV`; emit `o_acc[j] [16, 512]`, `m[j]`, `l[j]` FP32 | `mla_attend_mi300 (NEW)` | 8 | 5 | 0.000 | 1.125 |
| 5 | `mla_merge_uv`: per head, rescale-merge the 33 partials, `o[h] = (acc[h] / l[h]) @ W_UV[h]^T` -> `attn [1, 2048]` | `mla_merge_uv_mi300 (NEW)` | 8 | 2 | 2.000 | 0.000 |
| 6 | `x = x + attn @ W_o^T` | `gang_linear_res_mi300` | 8 | 8 | 8.000 | 0.000 |
| 7 | `h = post_attention_layernorm(x)` | `rmsnorm` | 1 | - | 0.004 | 0.000 |
| 8 | `a = silu(h @ W_gate^T) * (h @ W_up^T)` over the padded width 11264 | `gang_linear_silu_mi300` | 8 | 22 | 88.000 | 0.000 |
| 9 | `x = x + a @ W_down^T` (K padded to 11264) | `gang_linear_res_mi300` | 8 | 8 | 44.000 | 0.000 |
| | **total** | | **51** | | **158.259** | **1.125** |

### MoE layer (each of layers 1-26)

Ops 1-6 are identical to layer 0. Then:

| # | Op | Kernel | Tasks | Tiles/task | Weight MiB | Cache MiB |
|---|---|---|---|---|---|---|
| 7 | `h = post_attention_layernorm(x)` | `rmsnorm` | 1 | - | 0.004 | 0.000 |
| 8 | `router`: `logits = fp32(h) @ fp32(W_gate)^T` `[64]`; `p = softmax(logits)`; top-6 -> `(idx[6], w[6])`; append `(64, 1.0), (65, 1.0)`; write `topk_weight [8]`, `routing_indices [66]`, `mask [67]` | `moe_router_mi300 (NEW)` | 1 | - | 0.250 | 0.000 |
| 9 | `mid[k] = h @ W13[idx[k]]^T` -> `[8, 2816]` for the 8 active experts, one per XCD | `gang_moe_w13_linear_mi300` | 8 | 44 | 88.000 | 0.000 |
| 10 | `act[k] = silu(mid[k][:1408]) * mid[k][1408:]` -> `[8, 1408]` | `moe_silu_mul` | 8 | - | 0.000 | 0.000 |
| 11 | `out[k] = act[k] @ W2[idx[k]]^T` -> `[8, 2048]` | `gang_moe_w2_linear_mi300` | 8 | 32 | 44.000 | 0.000 |
| 12 | `x = x + sum_k w[k] * out[k]` (FP32 accumulate) | `moe_mul_sum_add_mi300` | 8 | - | 0.000 | 0.000 |
| | **total (ops 1-12)** | | **68** | | **158.509** | **1.125** |

The expert tensors are packed at load time as `W13 [66, 2816, 2048]`
(`gate_proj` rows then `up_proj` rows per expert; experts 64 and 65 are the
two 1408-wide halves of the shared MLP) and `W2 [66, 2048, 1408]`
(`04-memory-plan.md`).

### Head

| # | Op | Kernel | Tasks | Tiles/task | Weight MiB | Cache MiB |
|---|---|---|---|---|---|---|
| 1 | `h = model.norm(x)` | `rmsnorm` | 1 | - | 0.004 | 0.000 |
| 2 | `logits = h @ W_lm^T` -> `[1, 102400]` BF16 | `gang_linear_mi300` | 8 | 200 | 400.000 | 0.000 |
| 3 | per slice of 2048: `(max, argmax)` | `argmax_partial` | 50 | - | 0.000 | 0.000 |
| 4 | reduce 50 pairs; write `tokens[step + 1]` | `argmax_reduce (tokens variant)` | 1 | - | 0.000 | 0.000 |
| | **total** | | **60** | | **400.004** | **0.000** |

The end-of-graph event then runs `prepare_next_batch`: `step += 1`, check
for the stop condition, and either re-issue the graph as iteration `i + 1` or
terminate.

## Per iteration

| Quantity | Value |
|---|---|
| Operators (= events in the chain) | 326 |
| Tasks in the graph | 1,880 (1,882 entries with `TASK_BEGIN_TASK_GRAPH` and `TASK_TERMINATE`) |
| Tasks per worker per iteration | 6.35 |
| New kernels | `mla_prep`, `mla_attend`, `mla_merge_uv`, `moe_router`; plus a one-line `argmax_reduce` variant and the two forced-expert entries in routing |
| Weight bytes | 4,679.5 MiB |
| Latent cache bytes (S = 1024) | 30.38 MiB |
| **Total** | **4,709.9 MiB** |
| Roofline at 5.3 / 4.3 / 3.66 TB/s | **931.8 / 1,148.5 / 1,349.4 us** |
| Layer 1 alone | 159.63 MiB; 31.6 / 38.9 / 45.7 us |

The totals differ from `docs/deepseek-v2-lite/07-roofline.md` by the 3.75 MiB
dense padding (D8); everything else is the same traffic re-partitioned.

## What is not in the graph

- Prefill, the cache capture, weight packing and upload: one-time host work
  before `mpk()`.
- Any per-token host work: none. The host sleeps in `mpk()` for the whole
  generation.
- EOS handling: disabled (`eos = -1`), so the reference and the Fleet path
  both produce exactly 32 tokens.
- Sampling: none; greedy is the argmax reduce.

## Reading this against the reference

`modeling_deepseek.py` computes, per layer, `q_proj`, `kv_a_proj_with_mqa`,
`kv_a_layernorm`, `kv_b_proj`, RoPE on the two 64-wide slices, a 16-head
attention over decompressed `[S, 192]` keys and `[S, 128]` values, `o_proj`,
then the router in FP32 and the experts. Our ops 2-6 compute the same
function with `kv_b_proj` applied on the query side (`W_UK`) and the output
side (`W_UV`) instead of the key/value side, which is the reassociation of
`docs/deepseek-v2-lite/02-mla.md`; ops 7-12 compute the same MoE with the
shared MLP expressed as two experts. The boundaries at which the two are
compared, and the tolerances, are in `07-correctness.md`.
