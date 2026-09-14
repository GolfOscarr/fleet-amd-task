# 04 — Tensor Flow, Batch-1 Decode

One token, `B=1`, `T=1`, context `S=1024`. All activations BF16 unless noted.
Each row is a candidate node in the Fleet task graph.

Weight bytes are BF16 = 2 B/element. "Traffic" is weights + cache read; the
activations are negligible (a 2048-wide vector is 4 KB).

## Layer 0 — dense

| # | Op | Input | Weight | Output | Weight bytes | Depends on |
|---|---|---|---|---|---|---|
| 0.1 | `input_layernorm` (RMSNorm) | `[1,2048]` | `[2048]` | `[1,2048]` | 4 KB | residual |
| 0.2 | `q_proj` | `[1,2048]` | `[3072,2048]` | `[1,3072]` | 12 MB | 0.1 |
| 0.3 | `kv_a_proj_with_mqa` | `[1,2048]` | `[576,2048]` | `[1,576]` | 2.25 MB | 0.1 |
| 0.4 | `kv_a_layernorm` | `[1,512]` | `[512]` | `[1,512]` | 1 KB | 0.3 (latent half only) |
| 0.5 | RoPE on `q_pe`,`k_pe` | `[1,16,64]`,`[1,1,64]` | cos/sin | same | — | 0.2, 0.3 |
| 0.6 | KV-cache append | `[1,576]` | — | cache | — | 0.4, 0.5 |
| 0.7 | scores (absorbed) | `q'[1,16,512]`, `q_pe[1,16,64]` | cache `[1024,512]`,`[1024,64]` | `[1,16,1024]` | 1.125 MB cache | 0.6 |
| 0.8 | softmax (**FP32**) | `[1,16,1024]` | — | `[1,16,1024]` | — | 0.7 |
| 0.9 | weighted sum over `c_KV` | `[1,16,1024]` | cache `[1024,512]` | `[1,16,512]` | (same read as 0.7) | 0.8 |
| 0.10 | `o_proj` (absorbed `W_UV`) | `[1,2048]` | `[2048,2048]` | `[1,2048]` | 8 MB | 0.9 |
| 0.11 | residual add | | | `[1,2048]` | — | 0.10 |
| 0.12 | `post_attention_layernorm` | `[1,2048]` | `[2048]` | `[1,2048]` | 4 KB | 0.11 |
| 0.13 | `mlp.gate_proj` | `[1,2048]` | `[10944,2048]` | `[1,10944]` | 42.75 MB | 0.12 |
| 0.14 | `mlp.up_proj` | `[1,2048]` | `[10944,2048]` | `[1,10944]` | 42.75 MB | 0.12 |
| 0.15 | `silu(gate) * up` | | | `[1,10944]` | — | 0.13, 0.14 |
| 0.16 | `mlp.down_proj` | `[1,10944]` | `[2048,10944]` | `[1,2048]` | 42.75 MB | 0.15 |
| 0.17 | residual add | | | `[1,2048]` | — | 0.16 |

**Layer 0 weight traffic: 154.5 MB** (26.25 attention + 128.25 dense MLP).

Ops 0.13 and 0.14 are independent — a parallel pair. Ops 0.2 and 0.3 are
independent of each other. Both are free concurrency for the task graph.

## Layers 1–26 — MoE

Attention is identical (rows 0.1–0.11, 26.25 MB). The MLP block differs:

| # | Op | Input | Weight | Output | Weight bytes | Depends on |
|---|---|---|---|---|---|---|
| N.12 | `post_attention_layernorm` | `[1,2048]` | `[2048]` | `[1,2048]` | 4 KB | residual |
| N.13 | `mlp.gate` router (**FP32**) | `[1,2048]` | `[64,2048]` | `[1,64]` | 256 KB | N.12 |
| N.14 | softmax + top-6 (**FP32**) | `[1,64]` | — | idx`[6]`, w`[6]` | — | N.13 |
| N.15 | shared `gate_proj` | `[1,2048]` | `[2816,2048]` | `[1,2816]` | 11 MB | **N.12 only** |
| N.16 | shared `up_proj` | `[1,2048]` | `[2816,2048]` | `[1,2816]` | 11 MB | **N.12 only** |
| N.17 | shared `down_proj` | `[1,2816]` | `[2048,2816]` | `[1,2048]` | 11 MB | N.15, N.16 |
| N.18 | expert `e` `gate_proj` ×6 | `[1,2048]` | `[1408,2048]` | `[1,1408]` | 5.5 MB each | **N.14** |
| N.19 | expert `e` `up_proj` ×6 | `[1,2048]` | `[1408,2048]` | `[1,1408]` | 5.5 MB each | **N.14** |
| N.20 | expert `e` `down_proj` ×6 | `[1,1408]` | `[2048,1408]` | `[1,2048]` | 5.5 MB each | N.18, N.19 |
| N.21 | weighted sum of 6 | 6 × `[1,2048]` | — | `[1,2048]` | — | N.20 ×6 |
| N.22 | `+ shared` then residual | | | `[1,2048]` | — | N.17, N.21 |

**MoE layer weight traffic: 158.5 MB** = 26.25 attention + 0.25 router + 33
shared + 99 routed (6 × 16.5).

The critical structural fact: **N.15–N.17 (shared, 33 MB) depend only on N.12,
while N.18–N.20 (routed, 99 MB) depend on N.14.** The shared path can be issued
immediately and overlap the whole router latency. The routed path cannot start
until the router finishes — it is the only truly data-dependent address
computation in the model.

## Head

| Op | Input | Weight | Output | Bytes |
|---|---|---|---|---|
| `model.norm` | `[1,2048]` | `[2048]` | `[1,2048]` | 4 KB |
| `lm_head` | `[1,2048]` | `[102400,2048]` | `[1,102400]` | **400 MB** |
| argmax (greedy) | `[1,102400]` | — | token id | — |

`lm_head` is 400 MB — **8.5% of all per-token traffic in a single op**, and it
runs once per token rather than 27 times. It is untied (`tie_word_embeddings:
false`), so it cannot share the embedding table. Worth treating as its own
chiplet-parallel task: split the 102,400 rows across 8 XCDs, argmax locally,
then reduce 8 partial (value, index) pairs. That is a clean, high-value
Chiplet-task and a good early target.

`embed_tokens` is a single-row gather — 4 KB — and is negligible.

## Totals per token

| Component | Traffic |
|---|---|
| Layer 0 (dense) | 154.5 MB |
| Layers 1–26 (MoE) | 26 × 158.5 = 4,121 MB |
| `lm_head` | 400 MB |
| **Weights subtotal** | **4,675.5 MB** |
| MLA latent KV cache (1024 ctx, 27 layers) | 30.4 MB |
| **Total** | **4,705.9 MB** |

## Operation count — the "GPU launches" baseline

Per layer, counting the way a stock eager implementation launches kernels:

- Attention: ~10–14 kernels (norms, 3 projections, RoPE, cache update, 2
  matmuls, softmax, transposes)
- Dense MLP: 4 (gate, up, mul+silu, down)
- MoE: 2 (router, softmax/topk) + 3 shared + 6 × 3 routed + combine ≈ 24+

So roughly **30–40 kernel launches per MoE layer**, times 26, plus layer 0 and
the head: **on the order of 800–1,000 launches per token** for an eager
reference. A Fleet megakernel targets **1**.

This is a static estimate from the reference's structure, not a measurement —
the task requires no baseline profiling, but the number frames what the "GPU
launches" metric is measuring against. Confirm with a `rocprofv3 --kernel-trace`
of the reference when convenient; it costs one run.
