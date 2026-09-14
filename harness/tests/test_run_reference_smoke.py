"""Consistency checks on the artifacts of run_reference.py --smoke.

The smoke model is random and tiny, so these tests check the capture
plumbing, not the model: every artifact exists, boundary tensors relate to
each other the way the reference forward relates them, the cache rows are
what the attention consumed, and the routing log matches the boundaries.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

HARNESS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARNESS))
import common  # noqa: E402

STEPS = 4


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("ref_smoke")
    subprocess.run(
        [sys.executable, str(HARNESS / "run_reference.py"), "--smoke", "--steps", str(STEPS),
         "--out", str(out)],
        check=True, capture_output=True, text=True,
    )
    return {
        "dir": out,
        "meta": json.loads((out / "ref_run_meta.json").read_text()),
        "ids": json.loads((out / "ref_output_ids.json").read_text()),
        "ids_gen": json.loads((out / "ref_output_ids_generate.json").read_text()),
        "route": json.loads((out / "ref_route_log.json").read_text()),
        "b": load_file(str(out / "ref_boundaries_step0.safetensors")),
        "hidden": load_file(str(out / "ref_hidden_per_layer_step0.safetensors"))["hidden"],
        "row": load_file(str(out / "ref_cache_row1023.safetensors")),
        "cache": load_file(str(out / "ref_cache.safetensors")),
    }


def test_artifacts_and_ids(smoke):
    m = smoke["meta"]
    assert m["smoke"] and m["n_steps"] == STEPS
    assert len(smoke["ids"]) == STEPS
    assert m["generate_agrees_with_loop"] is True
    assert smoke["ids"] == smoke["ids_gen"]
    assert m["prefill_len"] == m["n_prompt"] - 1 == m["handover_position"]


def test_boundary_keys_present(smoke):
    b = smoke["b"]
    for l in (0, 1):
        for suffix in ("B1.norm1", "B2.q", "B3.c_kv", "B3.k_pe", "B4.q_pe", "B5.scores",
                       "B6.attn", "B7.x_res_attn", "B13.layer_out"):
            assert f"L{l}.{suffix}" in b, suffix
    # layer 0 is dense, layer 1 is MoE
    assert "L0.B8.router_logits" not in b
    for suffix in ("B8.router_logits", "B9.topk_idx", "B10.topk_w", "B12.shared"):
        assert f"L1.{suffix}" in b
    for k in ("head.B14.norm", "head.B15.logits", "head.B16.token"):
        assert k in b
    assert all(common.boundary_class(k) is not None for k in b if ".B" in k), \
        [k for k in b if ".B" in k and common.boundary_class(k) is None]


def test_token_is_argmax_of_logits(smoke):
    b = smoke["b"]
    assert int(b["head.B16.token"]) == smoke["ids"][0]
    assert int(torch.argmax(b["head.B15.logits"].float())) == smoke["ids"][0]


def test_scores_reproduce_attention_output(smoke):
    """softmax(B5) in FP32, cast to BF16, times the value cache == B6 (o_proj input).

    run_reference.py performs the check while it still holds the value cache
    and records the max abs error per layer; BF16 output, so a few ulps.
    """
    b, meta = smoke["b"], smoke["meta"]
    for l in (0, 1):
        s = b[f"L{l}.B5.scores"]
        assert s.dtype == torch.float32
        assert s.shape[1] == meta["handover_position"] + 1
        assert meta["b5_selfcheck_max_abs_err"][f"L{l}"] < 1e-2


def test_residual_sums(smoke):
    """B7 = layer input + o_proj output (BF16 add, as the layer does)."""
    b = smoke["b"]
    for l in (0, 1):
        x_in = b[f"L{l}.layer_in"]
        o = b[f"L{l}.o_proj_out"]
        assert torch.equal(x_in + o, b[f"L{l}.B7.x_res_attn"])


def test_moe_combine(smoke):
    """B13 - B7 == sum_k w_k * expert_k + shared, up to BF16 rounding."""
    b = smoke["b"]
    l = 1
    idx = b[f"L{l}.B9.topk_idx"].tolist()
    w = b[f"L{l}.B10.topk_w"]
    routed = sum(w[k].float() * b[f"L{l}.B11.expert_{e}"].float() for k, e in enumerate(idx))
    y = routed.to(torch.bfloat16) + b[f"L{l}.B12.shared"]
    got = b[f"L{l}.B13.layer_out"].float() - b[f"L{l}.B7.x_res_attn"].float()
    assert torch.allclose(got, y.float(), atol=2e-2 * y.float().abs().max().item() + 1e-3)
    # only the selected experts are present
    present = sorted(int(k.split("_")[-1]) for k in b if k.startswith(f"L{l}.B11.expert_"))
    assert present == sorted(idx)


def test_router_logits_match_topk(smoke):
    b = smoke["b"]
    logits = b["L1.B8.router_logits"]
    idx = b["L1.B9.topk_idx"].tolist()
    p = torch.softmax(logits, dim=-1)
    top = torch.topk(p, k=len(idx)).indices.tolist()
    assert sorted(top) == sorted(idx)
    # weights are the softmax probabilities (norm_topk_prob = false)
    for k, e in enumerate(idx):
        assert abs(float(p[e]) - float(b["L1.B10.topk_w"][k])) < 1e-6


def test_topk_ordered_by_weight(smoke):
    b = smoke["b"]
    w = b["L1.B10.topk_w"].tolist()
    assert w == sorted(w, reverse=True)
    r0 = smoke["route"][0][0]
    assert r0["w"] == sorted(r0["w"], reverse=True) and r0["idx"] == b["L1.B9.topk_idx"].tolist()


def test_route_log_matches_boundaries(smoke):
    r0 = smoke["route"][0]           # step 0, all MoE layers
    b = smoke["b"]
    assert len(smoke["route"]) == STEPS
    moe0 = r0[0]                     # layer 1 is the first MoE layer
    assert sorted(moe0["idx"]) == sorted(b["L1.B9.topk_idx"].tolist())


def test_cache_rows(smoke):
    cache, row, b, meta = smoke["cache"], smoke["row"], smoke["b"], smoke["meta"]
    P, s_max = meta["prefill_len"], meta["s_max"]
    L = cache["c_kv"].shape[0]
    assert cache["c_kv"].shape == (L, s_max, b["L0.B3.c_kv"].shape[0])
    assert cache["k_pe"].shape == (L, s_max, b["L0.B3.k_pe"].shape[0])
    assert cache["cos"].shape == (s_max, b["L0.B3.k_pe"].shape[0])
    # rows >= P are left for the Fleet path
    assert torch.count_nonzero(cache["c_kv"][:, P:]) == 0
    assert torch.count_nonzero(cache["k_pe"][:, P:]) == 0
    assert torch.count_nonzero(cache["c_kv"][:, :P]) > 0
    # the hand-over row equals the boundary B3 of the captured layers
    for l in (0, 1):
        assert torch.equal(row["c_kv"][l], b[f"L{l}.B3.c_kv"])
        assert torch.equal(row["k_pe"][l], b[f"L{l}.B3.k_pe"])


def test_hidden_per_layer(smoke):
    h, b = smoke["hidden"], smoke["b"]
    assert h.shape == (smoke["cache"]["c_kv"].shape[0], b["L0.B13.layer_out"].shape[0])
    for l in (0, 1):
        assert torch.equal(h[l], b[f"L{l}.B13.layer_out"])
