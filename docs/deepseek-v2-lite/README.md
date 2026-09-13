# DeepSeek-Coder-V2-Lite-Base — Model Reference

Working notes on the target model for the Fleet-style batch-1 decode path.
Companion to `../mi300x/`.

Checkpoint: `deepseek-ai/DeepSeek-Coder-V2-Lite-Base` (HF), `model_type:
deepseek_v2`, architecture `DeepseekV2ForCausalLM`.

## Files

| File | Contents |
|---|---|
| `01-config.md` | Every `config.json` field, what it controls, verified tensor shapes |
| `02-mla.md` | MLA math, naive vs absorbed, decoupled RoPE, KV-cache layout decision |
| `03-moe.md` | Routing, experts, shared experts, batch-1 behaviour |
| `04-tensor-flow.md` | Per-op shape/dtype/bytes table for a dense and an MoE layer |
| `05-weights.md` | Shards, parameter names, sizes, loading plan |
| `06-references.md` | HF reference, vLLM, SGLang — what each gives us |
| `07-roofline.md` | Working-set arithmetic, TPOT lower bound, L2 fit analysis |
| `08-correctness.md` | Oracle, boundaries, tolerances, greedy-match protocol |
| `99-open-questions.md` | Unverified claims → the check that settles each |
| `sources/` | `config.json`, `modeling_deepseek.py`, safetensors index, etc. |

## Verification convention

Same as `../mi300x/`, with one extra level:

| `verified` | Meaning |
|---|---|
| `checkpoint` | Read directly out of the downloaded checkpoint files in `sources/` |
| `primary` | Stated in the reference implementation or the DeepSeek-V2 paper |
| `derived` | Computed here; the arithmetic is shown and reproducible |
| `secondary` | Blog/search only — a hypothesis |
| `machine` | Confirmed by running on hardware |

Everything in `01`–`05` and `07` is `checkpoint` or `derived`. Nothing is
`machine` yet.

## The model in one table

| | |
|---|---|
| Layers | 27 — **layer 0 dense, layers 1–26 MoE** (`first_k_dense_replace: 1`) |
| Hidden size | 2048 |
| Attention | MLA, 16 heads, `kv_lora_rank` 512, **no Q compression** (`q_lora_rank: null`) |
| Head dims | `qk_nope` 128 + `qk_rope` 64 = 192 for Q/K; `v_head_dim` 128 |
| MoE | 64 routed experts, top-6, 2 shared experts, `moe_intermediate_size` 1408 |
| Dense MLP (layer 0) | `intermediate_size` 10944 |
| Vocab | 102,400 |
| Dtype | BF16 |
| Total params | 15,706,484,224 (15.71 B) → 31.41 GB on disk |
| Active params/token | 2.45 B |

## The five facts that most shape the design

1. **Layer 1 is the first MoE layer.** `first_k_dense_replace: 1` means layer 0
   is a dense MLP and layers 1–26 are MoE. The task's required milestone — a
   validated MoE layer at index ≥ 1 — is therefore **layer 1**, and it is
   representative of 26 of the 27 layers.
2. **The MLA latent KV cache for 1,024 tokens is 30.4 MB — it very nearly fits
   in the MI300X's 32 MB of aggregate L2.** The naive (HF) cache is 270 MB and
   does not. This is the strongest argument in the whole project for the
   absorbed-MLA form, and it is exactly the chiplet-locality story Fleet is
   about. See `07-roofline.md`.
3. **The HF reference caches decompressed K and V, not the latent.** So the
   prefill→decode interface *requires* a conversion, precisely as the task
   anticipates. See `02-mla.md`.
4. **Routed experts dominate memory traffic**: 99 MB of the 158.5 MB read per
   MoE layer per token. Any win at batch 1 is an expert-loading win.
5. **Top-k weights are not normalized.** `norm_topk_prob: false` and
   `routed_scaling_factor: 1.0`, so the gate outputs raw softmax probabilities
   that do **not** sum to 1. Easy to "fix" into a wrong answer.

## Roofline headline

Reading 4,705.9 MiB per token against 5.3 TB/s gives a **lower bound of 931 µs
per token (1,074 tok/s)**. Derivation and caveats in `07-roofline.md`.

## Sources

| Document | Location | Retrieved |
|---|---|---|
| `config.json` | `sources/config.json` | 2026-09-13 |
| `modeling_deepseek.py` (reference impl.) | `sources/modeling_deepseek.py` | 2026-09-13 |
| `configuration_deepseek.py` | `sources/configuration_deepseek.py` | 2026-09-13 |
| `model.safetensors.index.json` | `sources/model.safetensors.index.json` | 2026-09-13 |
| `generation_config.json`, `tokenizer_config.json` | `sources/` | 2026-09-13 |

Tensor shapes were read from the safetensors header of shard 1 via an HTTP
range request — no full download was needed.
