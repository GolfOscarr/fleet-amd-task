# 05 — Weights and Loading

## Checkpoint

| | |
|---|---|
| Repo | `deepseek-ai/DeepSeek-Coder-V2-Lite-Base` |
| Format | safetensors, 4 shards |
| Total size | 31,412,968,448 B = **31.41 GB** |
| Tensors | 5,291 |
| Dtype | BF16 throughout |
| Fits in 192 GB HBM | yes, with ~160 GB spare |

| Shard | Tensors |
|---|---|
| `model-00001-of-000004.safetensors` | 1,329 |
| `model-00002-of-000004.safetensors` | 1,491 |
| `model-00003-of-000004.safetensors` | 1,491 |
| `model-00004-of-000004.safetensors` | 980 |

The model was not downloaded in full here — shapes came from a range request on
shard 1's safetensors header (first 164,586 bytes). The full download belongs on
the MI300X box.

## Parameter names

Top level:

```
model.embed_tokens.weight      [102400, 2048]
model.norm.weight              [2048]
lm_head.weight                 [102400, 2048]      # untied
```

Every layer `N` in 0..26:

```
model.layers.N.input_layernorm.weight              [2048]
model.layers.N.post_attention_layernorm.weight     [2048]
model.layers.N.self_attn.q_proj.weight             [3072, 2048]
model.layers.N.self_attn.kv_a_proj_with_mqa.weight [576, 2048]
model.layers.N.self_attn.kv_a_layernorm.weight     [512]
model.layers.N.self_attn.kv_b_proj.weight          [4096, 512]
model.layers.N.self_attn.o_proj.weight             [2048, 2048]
```

Layer 0 only (dense):

```
model.layers.0.mlp.gate_proj.weight   [10944, 2048]
model.layers.0.mlp.up_proj.weight     [10944, 2048]
model.layers.0.mlp.down_proj.weight   [2048, 10944]
```

Layers 1–26 (MoE) — 203 tensors each:

```
model.layers.N.mlp.gate.weight                       [64, 2048]
model.layers.N.mlp.shared_experts.gate_proj.weight   [2816, 2048]
model.layers.N.mlp.shared_experts.up_proj.weight     [2816, 2048]
model.layers.N.mlp.shared_experts.down_proj.weight   [2048, 2816]
model.layers.N.mlp.experts.{0..63}.gate_proj.weight  [1408, 2048]
model.layers.N.mlp.experts.{0..63}.up_proj.weight    [1408, 2048]
model.layers.N.mlp.experts.{0..63}.down_proj.weight  [2048, 1408]
```

203 = 1 router + 3 shared + 192 expert + 2 norms + 5 attention.

## Loading plan

**Do not load 5,291 tensors as 5,291 separate allocations.** The MoE weights
should be packed into contiguous per-layer arenas so an expert's address is
arithmetic, not a pointer lookup:

```
experts[layer] : bf16[64][3][...]     # or three arrays, one per projection
    gate_proj : [64, 1408, 2048]      # expert e at stride e*1408*2048
    up_proj   : [64, 1408, 2048]
    down_proj : [64, 2048, 1408]
```

Reasons this matters for Fleet specifically:

- The router produces expert **indices**. With a packed arena, a worker computes
  the address as `base + idx * stride` with no indirection — which matters when
  the address is produced on-device and consumed on-device with no host round
  trip. That is the whole point of the design.
- A packed layout lets us choose expert→XCD placement by address arithmetic.
- It removes 4,992 descriptor lookups per token from the critical path.

### Transposition

`down_proj` is stored `[2048, 1408]` (out × in) while `gate/up_proj` are
`[1408, 2048]`. PyTorch `nn.Linear` stores `[out, in]` and computes `x @ W.T`.
Our GEMV kernels must agree on orientation; decide once, convert at load time,
and document it. A silent transpose is the classic way to get plausible-looking
garbage.

### Absorbed-MLA: reassociate at runtime, do NOT materialize fused weights

Per `02-mla.md`, the absorbed form reorders the matmuls so attention runs
against the 512-wide latent. There are two ways to realize it, and the
arithmetic strongly favours one.

**Variant A — materialize fused weights at load time.** Precompute
`W_UQ'[h] = W_UQ[h] @ W_UK[h]` giving `[16, 2048, 512]`, and
`W_O' = W_O @ blockdiag(W_UV)` giving `[2048, 8192]`.

**Variant B — keep the original weights and reassociate at runtime.** Compute
`q_nope = q_proj(x)` as usual, then `q'[h] = q_nope[h] @ W_UK[h]` on the fly
(16 tiny `[1,128] × [128,512]` products); symmetrically apply `W_UV` to the
attention output before `o_proj`. `kv_b_proj` is still read, just used
differently.

| Per layer, per token | Naive | A: materialized | B: runtime |
|---|---|---|---|
| `q_proj` / `W_UQ'` | 12.00 MB | 36.00 MB | 12.00 MB |
| `kv_a_proj_with_mqa` | 2.25 MB | 2.25 MB | 2.25 MB |
| `kv_b_proj` | 4.00 MB | — (folded) | 4.00 MB |
| `o_proj` / `W_O'` | 8.00 MB | 32.00 MB | 8.00 MB |
| **weights subtotal** | **26.25 MB** | **70.25 MB** | **26.25 MB** |
| KV-cache read (S=1024) | 10.00 MB | 1.12 MB | 1.12 MB |
| **total attention traffic** | **36.25 MB** | **71.38 MB** | **27.38 MB** |
| × 27 layers | 978.75 MB | 1,927.12 MB | **739.12 MB** |

**Variant A is nearly 2× worse than doing nothing.** Materializing the fusion
inflates the Q projection 3× and the output projection 4×, because both grow
from the 128-wide per-head `v`/`nope` dimension to the 512-wide latent. That
extra 44 MB per layer of weight traffic dwarfs the 8.9 MB per layer saved on the
cache at this context length.

**Variant B is strictly better than both** — identical weight traffic to naive,
with the 8.9× cache reduction on top. The added compute is 32 tiny GEMVs per
layer, irrelevant in a memory-bound regime.

**Decision: use Variant B.** No load-time weight fusion, no precomputed `W_UQ'`
or `W_O'`, no FP32 fusion products to validate. Keep `kv_b_proj` in memory and
split it once into its `W_UK` and `W_UV` halves (a view, not a copy).

The crossover where Variant A starts to win is where the cache term dominates —
well beyond our 1,024-token context. Worth stating in the design spec, since
"use the absorbed form" is the conventional advice and the conventional advice
is wrong at this context length if implemented as a weight fusion.

The roofline in `07-roofline.md` assumes Variant B.

## Environment

Download on the MI300X box with `huggingface-cli download`; 31 GB is not worth
transferring twice. The repo has been openly downloadable (no gating observed —
the config and index fetched without a token), but confirm for the weight
shards, which are sometimes gated separately.
