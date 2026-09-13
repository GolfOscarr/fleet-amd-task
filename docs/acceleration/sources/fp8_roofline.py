"""Reproduces the precision scenarios in ../01-precision.md from config.json."""
import json, pathlib

cfg = pathlib.Path(__file__).parents[2] / "deepseek-v2-lite/sources/config.json"
c = json.load(open(cfg))
H = c["hidden_size"]; L = c["num_hidden_layers"]; nh = c["num_attention_heads"]
dc = c["kv_lora_rank"]; dn = c["qk_nope_head_dim"]; dr = c["qk_rope_head_dim"]
dv = c["v_head_dim"]; E = c["n_routed_experts"]; K = c["num_experts_per_tok"]
SH = c["n_shared_experts"]; mi = c["moe_intermediate_size"]
inter = c["intermediate_size"]; V = c["vocab_size"]
nd = c["first_k_dense_replace"]; nmoe = L - nd
S = 1024; MB = 1024 ** 2; BW = 5.3e12

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


def run(name, bf16_keys, cache_bytes=2):
    w = sum(v * (2 if k in bf16_keys else 1) for k, v in cat.items()) / MB
    t = w + (dc + dr) * cache_bytes * S * L / MB
    us = t * MB / BW * 1e6
    print(f"{name:50s} {t:8.1f} MiB {us:7.1f} us {1e6 / us:6.0f} tok/s")


print(f"{'scenario':50s} {'traffic':>12s} {'TPOT':>9s} {'rate':>11s}")
run("BF16 everywhere", set(cat))
run("RedHatAI FP8 as shipped (lm_head BF16)", {"lm_head"})
run("  + kv_b_proj dequantized (vLLM behaviour)", {"lm_head", "attn kv_b"})
run("  + lm_head also FP8", {"attn kv_b"})
run("FP8 weights + FP8 KV cache", {"attn kv_b"}, cache_bytes=1)
run("FP8 experts only (all else BF16)", set(cat) - {"shared experts", "routed experts"})
