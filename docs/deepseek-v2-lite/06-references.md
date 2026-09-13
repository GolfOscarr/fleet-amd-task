# 06 — Reference Implementations

## HF `modeling_deepseek.py` — the oracle

Shipped in the repo and loaded via `trust_remote_code=True`. This is the
**canonical definition** of the model and our numerical reference. Copy in
`sources/modeling_deepseek.py` (78 KB), pinned at `transformers_version
4.39.3`.

Key classes:

| Class | Line | Notes |
|---|---|---|
| `DeepseekV2RMSNorm` | 94 | |
| `DeepseekV2YarnRotaryEmbedding` | 262 | YaRN; `yarn_get_mscale` at 247 |
| `DeepseekV2MLP` | 374 | SwiGLU; used for dense layer and experts |
| `MoEGate` | 393 | FP32 router — see `03-moe.md` |
| `DeepseekV2MoE` | 521 | `moe_infer` at 590 is the inference path |
| `DeepseekV2Attention` | 683 | **Naive MLA** — caches decompressed K/V |
| `DeepseekV2FlashAttention2` | 916 | FA2 variant, also naive MLA |
| `DeepseekV2DecoderLayer` | 1198 | |
| `DeepseekV2ForCausalLM` | 1588 | |

What to take from it:
- It is **slow and naive** — `moe_infer` sorts tokens by expert and loops. At
  batch 1 that is 6 tiny GEMVs in a Python loop. Fine as an oracle, useless as
  a performance reference.
- It implements **neither absorption nor a latent cache**. Both variants cache
  `key_states`/`value_states`. So the prefill→decode conversion in `02-mla.md`
  is required no matter which HF path we use.
- `output_hidden_states=True` gives per-layer hidden states, which is how we
  build the latent cache from a stock prefill, and how we get per-boundary
  reference tensors.

## vLLM on ROCm

Production MLA decode for DeepSeek-V2 with AITER and Triton backends. Useful
for:
- A realistic stock prefill, if we want one faster than HF.
- Prior art on absorbed MLA and paged latent KV layout — worth reading before
  we design ours, since they have already made these trade-offs.
- A performance point of reference, though the task requires no baseline.

Caveat: its KV cache is **paged**, so the conversion to our contiguous layout is
a second transformation on top of the latent conversion. If we use vLLM for
prefill, budget for that and keep it outside the measured window.

## SGLang

Also supports DeepSeek MLA on ROCm, with its own MLA decode kernels. Same role
and same caveats as vLLM. Worth one look to compare its latent-cache layout
against vLLM's before we commit to ours.

## AITER

AMD's tuned operator library — see `../mi300x/05-software-stack.md`. Relevant
here as a source of MLA-decode and fused-MoE kernels for gfx942, usable as
**documented fallbacks** while the Fleet path is incomplete. The task expects
fallbacks to exist and be counted, so using them is legitimate.

Caveat repeated from the hardware notes: AITER's newest MLA work reportedly
targets gfx950 rather than our gfx942, and gfx942 coverage is uneven. Verify
before depending on it (Q10 in `../mi300x/99-open-questions.md`).

## Recommended split

| Role | Choice |
|---|---|
| Numerical oracle | **HF `modeling_deepseek.py`**, FP32-upcast comparisons, greedy |
| Stock prefill | HF first (simplest, exposes hidden states); vLLM only if prefill time becomes painful |
| Latent-cache layout prior art | Read vLLM and SGLang; design our own contiguous layout |
| Fallback kernels | AITER / hipBLASLt, each one counted and reported |

Reading the reference is cheap and settles questions that guessing does not.
Every non-obvious constant in `01-config.md` — the `mscale²` factor, the FP32
router, the unnormalized top-k weights, the norm that applies to the latent but
not to `k_pe` — came from reading it rather than from the paper.
