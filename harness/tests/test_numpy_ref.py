"""numpy_ref against the reference model's own modules (tiny random model).

The smoke model of run_reference.py is loaded, one decode step is run with
hooks, and each NumPy kernel is fed the model's actual intermediates. Where
the design says "match the reference" the comparison is exact or within a
few BF16 ulps; where the design reassociates (attention through the latent)
the comparison is a tolerance and the error is printed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HARNESS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARNESS))
import numpy_ref as R  # noqa: E402
import run_reference as RR  # noqa: E402


class Args:
    model = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
    smoke = True
    device = "cpu"


@pytest.fixture(scope="module")
def step():
    """Tiny model, prefill 15 tokens, one-token step at position 15, hooks on layer 1."""
    torch.manual_seed(0)
    model = RR.load_model(Args)
    cfg = model.config
    from transformers import DynamicCache

    ids = torch.randint(0, cfg.vocab_size, (1, 16))
    P = 15
    cap = RR.Capture()
    RR.register_kva_hooks(model, cap)
    RR.register_route_hooks(model, cap)
    RR.register_boundary_hooks(model, cap, [0, 1])
    with torch.inference_mode():
        out = model(ids[:, :P], attention_mask=torch.ones(1, P, dtype=torch.long),
                    past_key_values=DynamicCache(), use_cache=True)
        pkv = out.past_key_values
        kva_prefill = [cap.store[f"kva.L{l}"] for l in range(cfg.num_hidden_layers)]
        s_max = 24
        cos, sin = RR.rope_tables(model, s_max)
        out = model(ids[:, P:P + 1], attention_mask=torch.ones(1, P + 1, dtype=torch.long),
                    past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        rows = RR.latent_rows(model, kva_prefill, torch.arange(P)[None], cos, sin)
        derived = {l: RR.step0_boundaries(model, cap, pkv, l, P, cos, sin) for l in (0, 1)}
    return dict(model=model, cfg=cfg, cap=cap, pkv=pkv, P=P, s_max=s_max, cos=cos, sin=sin,
                rows=rows, derived=derived)


def dims(cfg):
    return dict(nh=cfg.num_attention_heads, d_n=cfg.qk_nope_head_dim, d_r=cfg.qk_rope_head_dim,
                d_c=cfg.kv_lora_rank)


def layer_inputs(st, l):
    """qkva (fused q | kv_a output), the norm weight, W_uk/W_uv, cos/sin row for the step."""
    cap, model, P = st["cap"], st["model"], st["P"]
    at = model.model.layers[l].self_attn
    q = R.from_torch_bf16(cap.store[f"L{l}.B2.q"]).reshape(-1)
    kva = R.from_torch_bf16(cap.store[f"kva.L{l}"]).reshape(-1)
    qkva = np.concatenate([q, kva])
    d = dims(st["cfg"])
    w_kvb = R.from_torch_bf16(at.kv_b_proj.weight).reshape(d["nh"], d["d_n"] + at.v_head_dim, d["d_c"])
    W_uk, W_uv = w_kvb[:, :d["d_n"]], w_kvb[:, d["d_n"]:]
    w_kv_norm = R.from_torch_bf16(at.kv_a_layernorm.weight)
    cos_row = R.from_torch_bf16(st["cos"][P])
    sin_row = R.from_torch_bf16(st["sin"][P])
    return qkva, w_kv_norm, W_uk, W_uv, cos_row, sin_row


def rel(a, b):
    a, b = np.asarray(a, np.float64).reshape(-1), np.asarray(b, np.float64).reshape(-1)
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_bf16_rounding():
    x = np.array([1.0, 1.00390625, 1.001, 3.14159, -2.5e-3, 65504.0], np.float32)
    got = R.bf16(x)
    exp = torch.tensor(x).bfloat16().float().numpy()
    assert np.array_equal(got, exp)
    xs = np.random.default_rng(0).standard_normal(10000).astype(np.float32) * 100
    assert np.array_equal(R.bf16(xs), torch.tensor(xs).bfloat16().float().numpy())


def test_rmsnorm_matches_module(step):
    model, cap = step["model"], step["cap"]
    for l in (0, 1):
        layer = model.model.layers[l]
        x = R.from_torch_bf16(cap.store[f"L{l}.layer_in"]).reshape(-1)
        w = R.from_torch_bf16(layer.input_layernorm.weight)
        got = R.rmsnorm(x, w, layer.input_layernorm.variance_epsilon)
        exp = R.from_torch_bf16(cap.store[f"L{l}.B1.norm1"]).reshape(-1)
        assert rel(got, exp) < 2e-3, rel(got, exp)


def test_mla_prep_matches_reference(step):
    """c_kv row and k_pe row (B3), q_pe (B4) against the model's own functions;
    ql_nope against a direct FP32 product."""
    d = dims(step["cfg"])
    for l in (0, 1):
        qkva, w_kv_norm, W_uk, W_uv, cos_row, sin_row = layer_inputs(step, l)
        c_row, k_pe_row, ql_nope, q_pe = R.mla_prep(qkva, w_kv_norm, W_uk, cos_row, sin_row, **d)
        der = step["derived"][l]
        assert np.array_equal(k_pe_row, R.from_torch_bf16(der["B3.k_pe"]))        # RoPE: exact
        assert np.array_equal(q_pe, R.from_torch_bf16(der["B4.q_pe"]))            # RoPE: exact
        assert rel(c_row, R.from_torch_bf16(der["B3.c_kv"])) < 2e-3               # norm: reduction order
        q_nope = qkva[:d["nh"] * (d["d_n"] + d["d_r"])].reshape(d["nh"], -1)[:, :d["d_n"]]
        direct = R.bf16(np.einsum("hn,hnc->hc", q_nope, W_uk))
        assert np.array_equal(ql_nope, direct)


def test_attend_merge_against_reference_attention(step):
    """The reassociated path (latent scores, merge, W_uv after) versus the
    model's attention output B6 and its scores B5; tolerance, error printed."""
    d = dims(step["cfg"])
    P, s_max = step["P"], step["s_max"]
    at0 = step["model"].model.layers[0].self_attn
    for l in (0, 1):
        qkva, w_kv_norm, W_uk, W_uv, cos_row, sin_row = layer_inputs(step, l)
        c_row, k_pe_row, ql_nope, q_pe = R.mla_prep(qkva, w_kv_norm, W_uk, cos_row, sin_row, **d)
        c_prefill, k_prefill = step["rows"][l]
        c_kv = np.zeros((s_max, d["d_c"]), np.float32)
        k_pe = np.zeros((s_max, d["d_r"]), np.float32)
        c_kv[:P] = R.from_torch_bf16(c_prefill)
        k_pe[:P] = R.from_torch_bf16(k_prefill)
        c_kv[P], k_pe[P] = c_row, k_pe_row
        scale = float(at0.softmax_scale)
        partials, scores = R.mla_attend(ql_nope, q_pe, c_kv, k_pe, P, scale, split=4, debug_scores=True)
        assert partials.shape == (s_max // 4, d["nh"], d["d_c"] + 1)
        assert np.isneginf(partials[-1, :, d["d_c"]]).all()            # rows beyond step: empty
        attn = R.mla_merge_uv(partials, W_uv, P, split=4)
        ref_attn = R.from_torch_bf16(step["cap"].store[f"L{l}.B6.attn"]).reshape(-1)
        ref_scores = step["derived"][l]["B5.scores"].numpy()
        e_s, e_a = rel(scores, ref_scores), rel(attn, ref_attn)
        print(f"layer {l}: reassociation rel_err scores {e_s:.3e}, attention out {e_a:.3e}")
        assert e_s < 3e-2 and e_a < 3e-2
        # one split versus many splits: the merge is exact up to FP32 rounding
        one = R.mla_attend(ql_nope, q_pe, c_kv, k_pe, P, scale, split=s_max)
        attn_one = R.mla_merge_uv(one, W_uv, P, split=s_max)
        assert rel(attn_one, attn) < 1e-2
        # the decompressed reference arithmetic in NumPy lands on the model's output
        q = qkva[:d["nh"] * (d["d_n"] + d["d_r"])].reshape(d["nh"], -1)
        dec, dec_scores = R.attention_reference_decompressed(q[:, :d["d_n"]], q_pe, c_kv, k_pe, W_uk, W_uv, P, scale)
        assert rel(dec, ref_attn) < 1e-2 and rel(dec_scores, ref_scores) < 1e-2


def test_router_matches_gate(step):
    model, cap, cfg = step["model"], step["cap"], step["cfg"]
    l = 1
    gate = model.model.layers[l].mlp.gate
    h = R.from_torch_bf16(cap.store[f"L{l}.gate_in"]).reshape(-1)
    W = R.from_torch_bf16(gate.weight)
    E, K = cfg.n_routed_experts, cfg.num_experts_per_tok
    forced = (E, E + 1)
    logits, topk_w, routing, mask = R.moe_router(h, W, topk=K, n_experts=E, forced=forced,
                                                 scaling=cfg.routed_scaling_factor)
    idx_ref, w_ref, _ = cap.store[f"route.L{l}"]
    idx_ref, w_ref = idx_ref[0].tolist(), w_ref[0].float().numpy()
    ref_logits = torch.nn.functional.linear(cap.store[f"L{l}.gate_in"].reshape(-1).float(),
                                            gate.weight.float()).detach().numpy()
    assert rel(logits, ref_logits) < 1e-5
    assert sorted(mask[:K].tolist()) == sorted(idx_ref)
    for k, e in enumerate(mask[:K].tolist()):
        assert abs(topk_w[k] - w_ref[idx_ref.index(e)]) < 1e-6
    assert mask[K:K + 2].tolist() == list(forced) and mask[E + 2] == K + 2
    assert topk_w[K:].tolist() == [1.0, 1.0]
    assert routing.shape == (E + 2,) and (routing > 0).sum() == K + 2
    for k, e in enumerate(mask[:K + 2].tolist()):
        assert routing[e] == k + 1


def test_router_tie_break_and_combine():
    h = np.ones(8, np.float32)
    W = np.zeros((4, 8), np.float32)
    W[1] = W[3] = 1.0                                   # experts 1 and 3 tie at the top
    logits, topk_w, routing, mask = R.moe_router(h, W, topk=2, n_experts=4, forced=(4, 5))
    assert mask[:2].tolist() == [1, 3] and routing.tolist() == [0, 1, 0, 2, 3, 4]
    out8 = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
    got = R.moe_combine(out8, np.array([0.5, 0.25, 1.0, 1.0], np.float32), np.zeros(6, np.float32))
    exp = R.bf16(0.5 * out8[0] + 0.25 * out8[1] + out8[2] + out8[3])
    assert np.array_equal(got, exp)
