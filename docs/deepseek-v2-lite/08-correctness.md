# 08 — Correctness Methodology

The task requires "correctness evidence at every completed boundary" and reports
"correctness error" as a metric. This defines both, in advance, so the threshold
is a decision rather than an argument made after seeing results.

## Oracle

HF `modeling_deepseek.py` on CPU or GPU, `torch_dtype=torch.bfloat16`, loaded
with `trust_remote_code=True`, **`do_sample=False`** (the shipped
`generation_config.json` defaults to sampling at temperature 0.3 — see
`01-config.md`).

Fixed inputs, committed to the repo:
- One 1,024-token prompt, stored as **token IDs**, not text — so tokenizer
  version drift cannot change the experiment.
- The 32 reference output token IDs.
- Per-boundary reference tensors for layer 0 and layer 1, saved as `.safetensors`.

## Boundaries

Comparison points, in the order we will reach them:

| # | Boundary | Tensor | Shape |
|---|---|---|---|
| B1 | `input_layernorm` out | hidden | `[1,2048]` |
| B2 | `q_proj` out | q | `[1,3072]` |
| B3 | `kv_a_proj` out, post-`kv_a_layernorm` | `c_KV`, `k_pe` | `[1,512]`, `[1,64]` |
| B4 | post-RoPE `q_pe`, `k_pe` | | `[1,16,64]`, `[1,1,64]` |
| B5 | attention scores, pre-softmax | | `[1,16,1024]` |
| B6 | attention output, pre-`o_proj` | | `[1,16,128]` |
| B7 | `o_proj` out + residual | hidden | `[1,2048]` |
| B8 | router logits (**FP32**) | | `[1,64]` |
| B9 | **top-6 expert indices** | | `[6]` |
| B10 | top-6 weights | | `[6]` |
| B11 | each expert output | | `[1,2048]` |
| B12 | shared-expert output | | `[1,2048]` |
| B13 | MoE block out + residual | hidden | `[1,2048]` |
| B14 | final `model.norm` out | | `[1,2048]` |
| B15 | logits | | `[1,102400]` |
| B16 | argmax token id | scalar | |

B9 is special: **expert indices must match exactly.** It is discrete, so there
is no tolerance — a mismatch means the router diverged, and every downstream
number is meaningless. Check it first and check it loudly.

## Metrics and thresholds

For each boundary, report all three:

```
max_abs_err  = max|a - b|
rel_err      = ||a - b||_2 / ||b||_2
cos_sim      = <a,b> / (||a|| ||b||)
```

Proposed thresholds, BF16 (~3 decimal digits, eps ≈ 7.8e-3):

| Boundary class | `rel_err` threshold | Rationale |
|---|---|---|
| Single GEMV (B2, B3, B7) | < 2e-2 | One matmul of BF16 accumulation-order difference |
| Post-RoPE (B4) | < 1e-2 | Elementwise; should be tight |
| Scores / softmax (B5) | < 3e-2 | FP32 softmax over 1024, reassociated matmul |
| Expert / MoE (B11–B13) | < 3e-2 | SwiGLU + weighted sum |
| Accumulated per layer (B13) | < 5e-2 | Compounding across the block |
| Logits (B15) | < 5e-2 | 27 layers of accumulation |
| **Expert indices (B9)** | **exact** | Discrete |
| **Output token IDs (B16)** | **exact, all 32** | The real end-to-end test |

These are starting points, not measurements. **Calibrate them first**: run the
HF reference twice in different orders (or CPU vs GPU) and measure the error
BF16 alone produces. Our threshold should be a small multiple of that noise
floor, not a number picked from intuition. Record the calibration in the repo —
it is what makes the thresholds defensible rather than arbitrary.

## Known legitimate sources of divergence

Document these; do not try to eliminate them.

1. **Runtime reassociation of MLA** (`02-mla.md`) reorders matmuls. Algebraically
   equivalent, not bitwise. Accumulate the small per-head products in FP32.
2. **Expert accumulation order** — `topk` uses `sorted=False`, so the reference's
   own order is unspecified. BF16 summation is not associative.
3. **Softmax in FP32 then cast to BF16** — match the reference exactly here; this
   one is a choice, not noise, and diverging is a bug.
4. **The router in FP32** — likewise. A BF16 router can flip expert selection
   near ties and break B9.
5. **Chiplet-parallel reductions** — splitting a GEMV across 8 XCDs and reducing
   changes summation order versus a single-workgroup reduction. This will grow
   as we add chiplet parallelism; expect it, and re-baseline when the
   parallelization changes.

## Protocol per milestone

| Milestone | Required evidence |
|---|---|
| Single operator | That operator's boundary within threshold, 100 random inputs |
| **Layer 1 (MoE)** | **B1–B13 all within threshold, B9 exact**, given the reference's layer-1 input |
| N consecutive layers | B13 at each layer boundary; error growth reported as a curve |
| End-to-end | **All 32 token IDs identical**, plus B15 within threshold at each step |

The end-to-end test is binary and unambiguous, which is why greedy decoding is
in the task spec: 32 matching integers is evidence that needs no interpretation.

## Practical notes

- Compare in **FP32**: upcast both sides before computing errors, so the metric
  is not itself quantized.
- Save reference tensors once and commit them (they are small — a layer's worth
  of boundaries is a few MB). Regenerating the oracle on every run invites
  version drift.
- Test layer 1 with the **reference's actual layer-1 input**, not a random
  vector. Random inputs produce uniform routing; real hidden states do not, and
  routing behaviour is the thing we most need to be right.
- Log the chosen expert indices for all 32 steps. It is required for B9 and it
  doubles as the routing-distribution data for Q4.
