# 01 — Configuration

Source: `sources/config.json`, read from the checkpoint. Everything here is
`checkpoint`-verified unless marked otherwise.

## Every field

| Field | Value | What it controls |
|---|---|---|
| `architectures` | `["DeepseekV2ForCausalLM"]` | Model class |
| `model_type` | `deepseek_v2` | Dispatch; V2 family, **not** V3 |
| `num_hidden_layers` | **27** | Decoder layers |
| `hidden_size` | **2048** | Residual stream width |
| `vocab_size` | **102400** | Embedding / lm_head rows |
| `torch_dtype` | `bfloat16` | Matches the task's BF16 requirement |
| `tie_word_embeddings` | `false` | `lm_head` is a **separate** 400 MB tensor |
| `rms_norm_eps` | 1e-06 | RMSNorm epsilon |
| `hidden_act` | `silu` | SwiGLU activation in MLP/experts |
| `attention_bias` | `false` | No bias on attention projections |
| `attention_dropout` | 0.0 | No-op at inference |
| **MLA** | | |
| `num_attention_heads` | **16** | Query heads |
| `num_key_value_heads` | 16 | Present but not meaningful for MLA |
| `q_lora_rank` | **`null`** | **No Q compression** — single `q_proj` |
| `kv_lora_rank` | **512** | Latent KV width |
| `qk_nope_head_dim` | **128** | Non-positional part of Q/K per head |
| `qk_rope_head_dim` | **64** | RoPE part (decoupled, shared across heads for K) |
| `v_head_dim` | **128** | Value width per head |
| **MoE** | | |
| `first_k_dense_replace` | **1** | Layers `[0, 1)` dense → **layer 0 dense, 1–26 MoE** |
| `moe_layer_freq` | 1 | Every layer after the dense prefix is MoE |
| `n_routed_experts` | **64** | Routed expert count |
| `num_experts_per_tok` | **6** | Top-k |
| `n_shared_experts` | **2** | Always-on experts |
| `moe_intermediate_size` | **1408** | Per-expert FFN width |
| `intermediate_size` | **10944** | Dense MLP width (layer 0 only) |
| `scoring_func` | `softmax` | Gate scoring |
| `topk_method` | `greedy` | Plain top-k — **no group-limited routing** |
| `n_group` / `topk_group` | 1 / 1 | Grouping inert |
| `norm_topk_prob` | **`false`** | Top-k weights are **not** renormalized |
| `routed_scaling_factor` | **1.0** | Weights pass through unchanged |
| `aux_loss_alpha`, `seq_aux` | 0.001, `true` | Training only; inert at inference |
| **RoPE** | | |
| `rope_theta` | 10000 | Base |
| `max_position_embeddings` | 163840 | Post-YaRN limit |
| `rope_scaling.type` | `yarn` | YaRN scaling active |
| `rope_scaling.factor` | 40 | 4096 × 40 = 163840 |
| `rope_scaling.original_max_position_embeddings` | 4096 | Pre-scaling context |
| `rope_scaling.beta_fast` / `beta_slow` | 32 / 1 | YaRN ramp |
| `rope_scaling.mscale` / `mscale_all_dim` | 0.707 / 0.707 | **Attention-scale correction — see below** |
| **Tokens** | | |
| `bos_token_id` / `eos_token_id` | 100000 / 100001 | |

## Derived structure

| Quantity | Value | Derivation |
|---|---|---|
| `q_head_dim` | **192** | `qk_nope_head_dim + qk_rope_head_dim` = 128 + 64 |
| Q projection output | **3072** | 16 heads × 192 |
| KV-a projection output | **576** | `kv_lora_rank` 512 + `qk_rope_head_dim` 64 |
| KV-b projection output | **4096** | 16 heads × (128 nope + 128 v) |
| Shared-expert width | **2816** | `moe_intermediate_size` 1408 × `n_shared_experts` 2 (fused into one MLP) |
| Dense layers | layer 0 only | `first_k_dense_replace = 1` |
| MoE layers | layers 1–26 (26 layers) | |

## Verified tensor shapes

Read from the safetensors header of `model-00001-of-000004.safetensors` via an
HTTP range request. All BF16.

| Tensor | Shape | Confirms |
|---|---|---|
| `model.embed_tokens.weight` | `[102400, 2048]` | vocab × hidden |
| `model.layers.N.self_attn.q_proj.weight` | `[3072, 2048]` | `q_lora_rank: null` → one full projection, 16 × 192 |
| `model.layers.N.self_attn.kv_a_proj_with_mqa.weight` | `[576, 2048]` | 512 latent + 64 RoPE, fused |
| `model.layers.N.self_attn.kv_a_layernorm.weight` | `[512]` | Norm on the **latent only**, not on `k_pe` |
| `model.layers.N.self_attn.kv_b_proj.weight` | `[4096, 512]` | Decompress latent → 16 × (128 + 128) |
| `model.layers.N.self_attn.o_proj.weight` | `[2048, 2048]` | 16 × 128 → hidden |
| `model.layers.0.mlp.gate_proj.weight` | `[10944, 2048]` | Dense MLP |
| `model.layers.0.mlp.down_proj.weight` | `[2048, 10944]` | |
| `model.layers.1.mlp.gate.weight` | `[64, 2048]` | Router: 64 experts |
| `model.layers.1.mlp.shared_experts.gate_proj.weight` | `[2816, 2048]` | 2 shared experts fused |
| `model.layers.1.mlp.experts.K.gate_proj.weight` | `[1408, 2048]` | One routed expert |
| `model.layers.1.mlp.experts.K.down_proj.weight` | `[2048, 1408]` | |

Note `kv_a_layernorm` is `[512]`: the RMSNorm applies to the 512-wide latent
**only**. The 64-wide `k_pe` slice bypasses it. Getting this wrong is a silent
accuracy bug.

## The attention softmax scale — do not use `1/sqrt(d)`

From `modeling_deepseek.py`:

```python
self.softmax_scale = self.q_head_dim ** (-0.5)
if self.config.rope_scaling is not None:
    mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
    scaling_factor = self.config.rope_scaling["factor"]
    if mscale_all_dim:
        mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
        self.softmax_scale = self.softmax_scale * mscale * mscale
```

with `yarn_get_mscale(scale, mscale) = 0.1 * mscale * log(scale) + 1.0`.

Since `mscale_all_dim = 0.707` is truthy and `factor = 40`:

```
mscale          = 0.1 × 0.707 × ln(40) + 1.0 = 1.2608037774058554
softmax_scale   = 192^(-0.5) × mscale²
                = 0.07216878364870322 × 1.5896261651208736
                = 0.1147213867929261
```

**The effective scale is 0.11472, which is 1.59× the naive `192^-0.5`.** Hard-
coding `1/sqrt(192)` produces plausible-looking but wrong logits. This constant
goes in our kernel as a literal, and it is the first thing to check if
attention outputs are off by a smooth factor.

## Generation config — must be overridden

`sources/generation_config.json` specifies `do_sample: true`, `temperature:
0.3`, `top_p: 0.95`. The task requires **32 greedily decoded tokens**, so the
reference run must explicitly pass `do_sample=False` (or `num_beams=1` with
greedy) rather than inheriting these defaults. A reference that silently
samples makes token-level comparison meaningless.
