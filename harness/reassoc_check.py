#!/usr/bin/env python3
"""CPU check of the runtime MLA reassociation at the real shapes (L3, MIN-1).

Builds the reference model at the real attention dimensions (hidden 2048,
16 heads, latent 512, rope 64) with one dense layer and random weights,
prefills 1,023 random tokens, runs position 1,023 as a one-token step, and
compares the design's path (numpy_ref: q_nope @ W_uk on the query side,
scores against the latent, W_uv after the merge) with the reference's
decompressed attention at the same position. Reports rel_err on B5
(scores) and B6 (attention output) for several seeds and for "peaky"
queries (q_proj scaled so the softmax concentrates on few positions), and
writes harness/results/reassoc_check.json.

This bounds the error the design accepts as legitimate (07-correctness.md,
"Known legitimate sources of divergence", item 1). It is not the BF16
noise floor of the GPU (that is calibrate.py, on the machine).

    python harness/reassoc_check.py [--seeds 3] [--out harness/results/reassoc_check.json]
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import numpy_ref as R  # noqa: E402
import run_reference as RR  # noqa: E402

N_PROMPT = 1024
P = N_PROMPT - 1
S_MAX = N_PROMPT + 1


def real_dims_one_layer(config):
    """Real attention dims; one dense layer; a small vocabulary (irrelevant here)."""
    config.update(dict(num_hidden_layers=1, vocab_size=4096, max_position_embeddings=4096))
    return config


def build_model(seed, q_scale):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(common.MODEL, trust_remote_code=True)
    config = real_dims_one_layer(config)
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                             attn_implementation="eager").eval()
    at = model.model.layers[0].self_attn
    with torch.no_grad():
        # nn.Linear's default init gives tiny scores; scale to a realistic regime
        # where the softmax is neither flat nor one-hot, and make q_proj sharper for the peaky case
        at.q_proj.weight.mul_(q_scale)
        at.kv_b_proj.weight.mul_(4.0)
    return model


def rel(a, b):
    a, b = np.asarray(a, np.float64).reshape(-1), np.asarray(b, np.float64).reshape(-1)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def one_case(seed, q_scale, split):
    from transformers import DynamicCache

    model = build_model(seed, q_scale)
    cfg = model.config
    at = model.model.layers[0].self_attn
    d = dict(nh=cfg.num_attention_heads, d_n=cfg.qk_nope_head_dim, d_r=cfg.qk_rope_head_dim,
             d_c=cfg.kv_lora_rank)
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, cfg.vocab_size, (1, N_PROMPT), generator=g)

    cap = RR.Capture()
    RR.register_kva_hooks(model, cap)
    RR.register_boundary_hooks(model, cap, [0])
    t0 = time.time()
    with torch.inference_mode():
        out = model(ids[:, :P], attention_mask=torch.ones(1, P, dtype=torch.long),
                    past_key_values=DynamicCache(), use_cache=True)
        pkv = out.past_key_values
        kva_prefill = [cap.store["kva.L0"]]
        cos, sin = RR.rope_tables(model, S_MAX)
        out = model(ids[:, P:P + 1], attention_mask=torch.ones(1, P + 1, dtype=torch.long),
                    past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        (c_prefill, k_prefill), = RR.latent_rows(model, kva_prefill, torch.arange(P)[None], cos, sin)
        derived = RR.step0_boundaries(model, cap, pkv, 0, P, cos, sin)
    t_ref = time.time() - t0

    # the design's path in NumPy on the model's own intermediates
    q = R.from_torch_bf16(cap.store["L0.B2.q"]).reshape(-1)
    kva = R.from_torch_bf16(cap.store["kva.L0"]).reshape(-1)
    qkva = np.concatenate([q, kva])
    w_kvb = R.from_torch_bf16(at.kv_b_proj.weight).reshape(d["nh"], d["d_n"] + at.v_head_dim, d["d_c"])
    W_uk, W_uv = w_kvb[:, :d["d_n"]], w_kvb[:, d["d_n"]:]
    w_kv_norm = R.from_torch_bf16(at.kv_a_layernorm.weight)
    cos_row, sin_row = R.from_torch_bf16(cos[P]), R.from_torch_bf16(sin[P])
    t1 = time.time()
    c_row, k_pe_row, ql_nope, q_pe = R.mla_prep(qkva, w_kv_norm, W_uk, cos_row, sin_row, **d)
    c_kv = np.zeros((S_MAX, d["d_c"]), np.float32)
    k_pe = np.zeros((S_MAX, d["d_r"]), np.float32)
    c_kv[:P], k_pe[:P] = R.from_torch_bf16(c_prefill), R.from_torch_bf16(k_prefill)
    c_kv[P], k_pe[P] = c_row, k_pe_row
    scale = float(at.softmax_scale)
    partials, scores = R.mla_attend(ql_nope, q_pe, c_kv, k_pe, P, scale, split=split, debug_scores=True)
    attn = R.mla_merge_uv(partials, W_uv, P, split=split)
    t_ours = time.time() - t1

    ref_scores = derived["B5.scores"].numpy()
    ref_attn = R.from_torch_bf16(cap.store["L0.B6.attn"]).reshape(-1)
    probs = R.softmax_f32(ref_scores)
    # baseline: the reference's own decompressed arithmetic recomputed in NumPy
    # (same math, another summation order); if the reassociated path is no
    # further from the model than this, reassociation is not the dominant term
    q_nope = q.reshape(d["nh"], -1)[:, :d["d_n"]]
    dec, _ = R.attention_reference_decompressed(q_nope, q_pe, c_kv, k_pe, W_uk, W_uv, P, scale)
    return {
        "seed": seed, "q_scale": q_scale, "split": split, "positions": P + 1,
        "b3_c_kv_rel_err": rel(c_row, R.from_torch_bf16(derived["B3.c_kv"])),
        "b3_k_pe_exact": bool(np.array_equal(k_pe_row, R.from_torch_bf16(derived["B3.k_pe"]))),
        "b4_q_pe_exact": bool(np.array_equal(q_pe, R.from_torch_bf16(derived["B4.q_pe"]))),
        "b5_scores_rel_err": rel(scores, ref_scores),
        "b5_scores_max_abs_err": float(np.abs(scores - ref_scores).max()),
        "b6_attn_rel_err": rel(attn, ref_attn),
        "b6_attn_max_abs_err": float(np.abs(attn - ref_attn).max()),
        "b6_attn_cos_sim": float(np.dot(attn.astype(np.float64), ref_attn) /
                                 (np.linalg.norm(attn) * np.linalg.norm(ref_attn))),
        "baseline_same_math_other_order_b6_rel_err": rel(dec, ref_attn),
        "scores_abs_max": float(np.abs(ref_scores).max()),
        "softmax_max_prob_mean_over_heads": float(probs.max(axis=1).mean()),
        "scores_std": float(ref_scores.std()),
        "time_reference_s": round(t_ref, 2), "time_numpy_s": round(t_ours, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=str(common.ROOT / "harness/results/reassoc_check.json"))
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    cases = []
    for seed in range(args.seeds):
        for q_scale in (4.0, 16.0):                      # moderate and peaky softmax
            r = one_case(seed, q_scale, split=32)
            cases.append(r)
            print(f"seed {seed} q_scale {q_scale:>4}: B5 rel {r['b5_scores_rel_err']:.3e}  "
                  f"B6 rel {r['b6_attn_rel_err']:.3e} (same-math baseline "
                  f"{r['baseline_same_math_other_order_b6_rel_err']:.3e})  cos {r['b6_attn_cos_sim']:.6f}  "
                  f"|s| max {r['scores_abs_max']:.0f}  max prob {r['softmax_max_prob_mean_over_heads']:.3f}")
    summary = {
        "what": "runtime MLA reassociation (numpy_ref path) versus the reference's decompressed "
                "attention at position 1023 over 1024 positions, real attention dims, random weights",
        "b5_scores_rel_err_max": max(c["b5_scores_rel_err"] for c in cases),
        "b6_attn_rel_err_max": max(c["b6_attn_rel_err"] for c in cases),
        "b6_attn_cos_sim_min": min(c["b6_attn_cos_sim"] for c in cases),
        "baseline_same_math_other_order_b6_rel_err_max": max(
            c["baseline_same_math_other_order_b6_rel_err"] for c in cases),
        "reading": "B5 differs from the reference by the BF16 rounding of ql_nope (about 2^-9 relative). "
                   "B6 grows with the score magnitude because the reference rounds its scores to BF16 "
                   "before the softmax (matmul in BF16): at |s| ~ 40 the same math in another summation "
                   "order already differs from the model by as much as the reassociated path does, so the "
                   "attention threshold must be the calibrated floor on the real model (calibrate.py), "
                   "not the starting 3e-2; reassociation itself is within that floor.",
        "starting_threshold_scores": common.CLASS_THRESHOLD["scores"],
        "starting_threshold_attention": common.CLASS_THRESHOLD["attention"],
        "platform": platform.platform(), "torch": torch.__version__,
        "cases": cases,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"max B5 rel_err {summary['b5_scores_rel_err_max']:.3e} (threshold {summary['starting_threshold_scores']}), "
          f"max B6 rel_err {summary['b6_attn_rel_err_max']:.3e} (threshold {summary['starting_threshold_attention']}) "
          f"-> {out}")


if __name__ == "__main__":
    main()
