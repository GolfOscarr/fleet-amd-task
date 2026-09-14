"""Reproduces the precision scenarios in ../01-precision.md from config.json.

The router (mlp.gate) and all norms are BF16 in the RedHatAI checkpoint and are
treated as BF16 in every scenario -- quantising a [64, 2048] router would save
3.25 MiB/token and risk expert-selection flips.
"""
import json, pathlib

cfg = pathlib.Path(__file__).parents[2] / "deepseek-v2-lite/sources/config.json"
c = json.load(open(cfg))
H, L, nh = c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"]
dc, dn, dr, dv = (c["kv_lora_rank"], c["qk_nope_head_dim"],
                  c["qk_rope_head_dim"], c["v_head_dim"])
E, K, SH = c["n_routed_experts"], c["num_experts_per_tok"], c["n_shared_experts"]
mi, inter, V = c["moe_intermediate_size"], c["intermediate_size"], c["vocab_size"]
nd = c["first_k_dense_replace"]; nmoe = L - nd
S = 1024; MB = 1024 ** 2

BW = {"theo": 5.3e12, "meas": 4.3e12, "consv": 3.66e12}   # see ../../mi300x/07

cat = {
    "attn q_proj":    L * (dn + dr) * nh * H,
    "attn kv_a":      L * (dc + dr) * H,
    "attn kv_b":      L * nh * (dn + dv) * dc,
    "attn o_proj":    L * H * nh * dv,
    "dense MLP":      nd * 3 * inter * H,
    "router":         nmoe * E * H,
    "shared experts": nmoe * 3 * (mi * SH) * H,
    "routed experts": nmoe * K * 3 * mi * H,
    "lm_head":        V * H,
}
ROUTER_BF16 = {"router"}          # true in the shipped checkpoint


def run(name, bf16_keys, cache_bytes=2):
    bf16 = set(bf16_keys) | ROUTER_BF16
    w = sum(v * (2 if k in bf16 else 1) for k, v in cat.items()) / MB
    t = w + (dc + dr) * cache_bytes * S * L / MB
    us = {k: t * MB / bw * 1e6 for k, bw in BW.items()}
    print(f"{name:46s} {t:8.1f} MiB " +
          " ".join(f"{us[k]:7.1f}" for k in BW) + " us  " +
          f"{1e6/us['theo']:5.0f}/{1e6/us['meas']:5.0f}/{1e6/us['consv']:5.0f} tok/s")


print(f"{'scenario':46s} {'traffic':>12s} " +
      " ".join(f"{k:>7s}" for k in BW) + "        theo/ meas/consv")
run("BF16 everywhere", set(cat))
run("RedHatAI FP8 as shipped (lm_head BF16)", {"lm_head"})
run("  + kv_b_proj dequantized (vLLM behaviour)", {"lm_head", "attn kv_b"})
run("  + lm_head also FP8 (hypothetical)", set())
run("  + FP8 latent KV cache", set(), cache_bytes=1)
run("FP8 experts only (all else BF16)", set(cat) - {"shared experts", "routed experts"})
