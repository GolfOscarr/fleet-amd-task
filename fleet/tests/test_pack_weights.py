"""Unit tests for fleet/pack_weights.py on random BF16 tensors of small shapes,
plus a shapes-only test at the real dims."""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fleet import pack_weights as pw  # noqa: E402

torch.manual_seed(0)

SMALL = pw.Dims(H=256, NH=2, D_N=16, D_R=8, D_V=16, D_C=32, V=1024, L=2,
                I_DENSE=2000, I_MOE=128, E=8, TOPK=2, N_SHARED=2)


def rnd(*shape):
    return (torch.randn(*shape) * 0.1).to(torch.bfloat16)


def test_small_dims():
    d = SMALL
    assert d.I_DENSE_PAD == 2048 and d.E_TOTAL == 10 and d.I_SHARED == 256
    assert d.Q_OUT == 48 and d.KVA_OUT == 40 and d.SILU_GROUPS == 16


def test_shuffle_rows_matches_index_construction_and_round_trips():
    G = 4
    a = torch.arange(16 * 3).reshape(16, 3)
    b = torch.arange(1000, 1000 + 8 * 3).reshape(8, 3)
    got = pw.shuffle_rows([a, b], G)
    rows = []
    for g in range(G):
        rows += [a[r] for r in range(g * 4, (g + 1) * 4)]
        rows += [b[r] for r in range(g * 2, (g + 1) * 2)]
    assert torch.equal(got, torch.stack(rows))
    a2, b2 = pw.unshuffle_rows(got, [16, 8], G)
    assert torch.equal(a2, a) and torch.equal(b2, b)


def test_silu_shuffle_layout_as_kernel_reads_it():
    d = SMALL
    p = pw.pack_dense_mlp(d, rnd(d.I_DENSE, d.H), rnd(d.I_DENSE, d.H), rnd(d.H, d.I_DENSE))
    W = p["W_gu_shuffled"]
    assert W.shape == (2 * d.I_DENSE_PAD, d.H)
    assert p["num_groups"] == d.I_DENSE_PAD // 128
    tile_n = 64
    chunk = 2 * d.I_DENSE_PAD // 8           # weight rows per XCD
    out_per_xcd = d.I_DENSE_PAD // 8         # output columns per XCD
    n_tiles = out_per_xcd // tile_n
    for x in range(8):
        c = W[x * chunk:(x + 1) * chunk]
        for t in range(n_tiles):
            grp, sub = t // 2, t % 2
            g_tile = (grp * 4 + sub) * tile_n
            u_tile = (grp * 4 + 2 + sub) * tile_n
            col0 = x * out_per_xcd + t * tile_n
            assert torch.equal(c[g_tile:g_tile + tile_n], p["gate_pad"][col0:col0 + tile_n]), (x, t)
            assert torch.equal(c[u_tile:u_tile + tile_n], p["up_pad"][col0:col0 + tile_n]), (x, t)


def test_padded_dense_mlp_is_exact():
    d = SMALL
    g, u, dn = rnd(d.I_DENSE, d.H), rnd(d.I_DENSE, d.H), rnd(d.H, d.I_DENSE)
    p = pw.pack_dense_mlp(d, g, u, dn)
    h = torch.randn(d.H)
    ref = dn.float() @ (F.silu(g.float() @ h) * (u.float() @ h))
    got = p["W_down_pad"].float() @ (F.silu(p["gate_pad"].float() @ h) * (p["up_pad"].float() @ h))
    assert torch.equal(p["W_down_pad"][:, d.I_DENSE:], torch.zeros(d.H, d.I_DENSE_PAD - d.I_DENSE, dtype=torch.bfloat16))
    assert torch.allclose(got, ref, rtol=1e-5, atol=1e-5)


def test_shared_expert_split_is_exact():
    d = SMALL
    experts = [(rnd(d.I_MOE, d.H), rnd(d.I_MOE, d.H), rnd(d.H, d.I_MOE)) for _ in range(d.E)]
    sg, su, sd = rnd(d.I_SHARED, d.H), rnd(d.I_SHARED, d.H), rnd(d.H, d.I_SHARED)
    p = pw.pack_moe(d, rnd(d.E, d.H), experts, (sg, su, sd))
    assert p["W13"].shape == (d.E_TOTAL, 2 * d.I_MOE, d.H) and p["W2"].shape == (d.E_TOTAL, d.H, d.I_MOE)
    x = torch.randn(d.H)
    ref = sd.float() @ (F.silu(sg.float() @ x) * (su.float() @ x))
    got = torch.zeros(d.H)
    for s in range(d.N_SHARED):
        w13, w2 = p["W13"][d.E + s].float(), p["W2"][d.E + s].float()
        mid = w13 @ x
        got += w2 @ (F.silu(mid[:d.I_MOE]) * mid[d.I_MOE:])
    assert (got - ref).norm() / ref.norm() < 1e-5


def test_routed_experts_layout():
    d = SMALL
    experts = [(rnd(d.I_MOE, d.H), rnd(d.I_MOE, d.H), rnd(d.H, d.I_MOE)) for _ in range(d.E)]
    shared = (rnd(d.I_SHARED, d.H), rnd(d.I_SHARED, d.H), rnd(d.H, d.I_SHARED))
    router = rnd(d.E, d.H)
    p = pw.pack_moe(d, router, experts, shared)
    assert torch.equal(p["W_gate"], router)
    for e, (g, u, dn) in enumerate(experts):
        assert torch.equal(p["W13"][e, :d.I_MOE], g)
        assert torch.equal(p["W13"][e, d.I_MOE:], u)
        assert torch.equal(p["W2"][e], dn)
    for s in range(d.N_SHARED):
        rows = slice(s * d.I_MOE, (s + 1) * d.I_MOE)
        assert torch.equal(p["W13"][d.E + s, :d.I_MOE], shared[0][rows])
        assert torch.equal(p["W13"][d.E + s, d.I_MOE:], shared[1][rows])
        assert torch.equal(p["W2"][d.E + s], shared[2][:, rows])


def test_uk_uv_views_reproduce_kv_b_proj():
    d = SMALL
    w_kvb = rnd(d.KVB_OUT, d.D_C)
    w_uk, w_uv = pw.split_uk_uv(d, w_kvb)
    assert w_uk.shape == (d.NH, d.D_N, d.D_C) and w_uv.shape == (d.NH, d.D_V, d.D_C)
    assert w_uk.data_ptr() == w_kvb.data_ptr()          # a view
    c = torch.randn(d.D_C)
    kv = (w_kvb.float() @ c).view(d.NH, d.D_N + d.D_V)
    for h in range(d.NH):
        assert torch.allclose(w_uk[h].float() @ c, kv[h, :d.D_N])
        assert torch.allclose(w_uv[h].float() @ c, kv[h, d.D_N:])
    uk_c, uv_c = pw.split_uk_uv(d, w_kvb, contiguous=True)
    assert uk_c.is_contiguous() and torch.equal(uk_c, w_uk) and torch.equal(uv_c, w_uv)


def test_pack_qkva_row_ranges():
    d = SMALL
    w_q, w_kva = rnd(d.Q_OUT, d.H), rnd(d.KVA_OUT, d.H)
    W = pw.pack_qkva(w_q, w_kva)
    assert W.shape == (d.Q_OUT + d.KVA_OUT, d.H)
    assert torch.equal(W[:d.Q_OUT], w_q)
    assert torch.equal(W[d.Q_OUT:d.Q_OUT + d.D_C], w_kva[:d.D_C])
    assert torch.equal(W[d.Q_OUT + d.D_C:], w_kva[d.D_C:])


def test_pack_attention_and_pack_layer():
    d = SMALL
    lw = {
        "input_layernorm.weight": rnd(d.H), "post_attention_layernorm.weight": rnd(d.H),
        "self_attn.q_proj.weight": rnd(d.Q_OUT, d.H),
        "self_attn.kv_a_proj_with_mqa.weight": rnd(d.KVA_OUT, d.H),
        "self_attn.kv_a_layernorm.weight": rnd(d.D_C),
        "self_attn.kv_b_proj.weight": rnd(d.KVB_OUT, d.D_C),
        "self_attn.o_proj.weight": rnd(d.H, d.H),
        "mlp.gate_proj.weight": rnd(d.I_DENSE, d.H), "mlp.up_proj.weight": rnd(d.I_DENSE, d.H),
        "mlp.down_proj.weight": rnd(d.H, d.I_DENSE),
    }
    out = pw.pack_layer(d, 0, lw)
    assert set(out) == {"w_norm1", "w_norm2", "W_qkva", "w_kv_norm", "W_uk", "W_uv", "W_o",
                        "W_gu_shuffled", "W_down_pad"}
    assert set(n[len("model.layers.0."):] for n in pw.layer_names(d, 0)) == set(lw)
    assert len(pw.layer_names(d, 1)) == 7 + 1 + 3 + 3 * d.E


def test_real_dims_shapes_and_bytes():
    d = pw.REAL_DIMS
    assert d.I_DENSE_PAD == 11264 and d.E_TOTAL == 66 and d.I_SHARED == 2816
    assert d.Q_OUT == 3072 and d.KVA_OUT == 576 and d.KVB_OUT == 4096
    assert d.SILU_GROUPS == 88
    pw.check_tiling(d)
    assert pw.packed_bytes(d) == pw.EXPECTED_PACKED_BYTES == 31_412_968_448 + 3_932_160
    # per-layer sizes of 04-memory-plan.md
    MiB = 2 ** 20
    assert (d.Q_OUT + d.KVA_OUT) * d.H * 2 == 14.25 * MiB
    assert d.E_TOTAL * 2 * d.I_MOE * d.H * 2 == 726 * MiB
    assert d.E_TOTAL * d.H * d.I_MOE * 2 == 363 * MiB
    assert 2 * d.I_DENSE_PAD * d.H * 2 == 88 * MiB and d.H * d.I_DENSE_PAD * 2 == 44 * MiB
