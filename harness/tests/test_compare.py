"""compare.py on synthetic reference/Fleet pairs."""
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

HARNESS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARNESS))
import compare  # noqa: E402
import common  # noqa: E402

H, NH, S, E, K, V = 64, 2, 17, 8, 2, 256


def make_ref(g):
    b = {}
    for l in (0, 1):
        b[f"L{l}.B1.norm1"] = torch.randn(H, generator=g).bfloat16()
        b[f"L{l}.B2.q"] = torch.randn(NH * 24, generator=g).bfloat16()
        b[f"L{l}.B3.c_kv"] = torch.randn(32, generator=g).bfloat16()
        b[f"L{l}.B3.k_pe"] = torch.randn(8, generator=g).bfloat16()
        b[f"L{l}.B4.q_pe"] = torch.randn(NH, 8, generator=g).bfloat16()
        b[f"L{l}.B5.scores"] = torch.randn(NH, S, generator=g)
        b[f"L{l}.B6.attn"] = torch.randn(NH * 16, generator=g).bfloat16()
        b[f"L{l}.B7.x_res_attn"] = torch.randn(H, generator=g).bfloat16()
        b[f"L{l}.B13.layer_out"] = torch.randn(H, generator=g).bfloat16()
        b[f"L{l}.layer_in"] = torch.randn(H, generator=g).bfloat16()      # auxiliary, not a boundary
    b["L1.B8.router_logits"] = torch.randn(E, generator=g)
    b["L1.B9.topk_idx"] = torch.tensor([5, 2])
    b["L1.B10.topk_w"] = torch.tensor([0.4, 0.3])
    for e in (5, 2):
        b[f"L1.B11.expert_{e}"] = torch.randn(H, generator=g).bfloat16()
    b["L1.B12.shared"] = torch.randn(H, generator=g).bfloat16()
    b["head.B14.norm"] = torch.randn(H, generator=g).bfloat16()
    b["head.B15.logits"] = torch.randn(V, generator=g)
    b["head.B16.token"] = torch.tensor([42])
    return b


@pytest.fixture
def dirs(tmp_path):
    g = torch.Generator().manual_seed(0)
    ref = make_ref(g)
    ref_dir, fleet_dir = tmp_path / "ref", tmp_path / "fleet"
    ref_dir.mkdir()
    fleet_dir.mkdir()
    save_file(ref, str(ref_dir / "ref_boundaries_step0.safetensors"))
    ids = list(range(100, 132))
    (ref_dir / "ref_output_ids.json").write_text(json.dumps(ids))
    route = [[{"idx": [5, 2], "w": [0.4, 0.3]}, {"idx": [1, 7], "w": [0.5, 0.2]}] for _ in range(32)]
    (ref_dir / "ref_route_log.json").write_text(json.dumps(route))
    hidden = torch.randn(3, H, generator=g).bfloat16()
    save_file({"hidden": hidden}, str(ref_dir / "ref_hidden_per_layer_step0.safetensors"))
    return ref_dir, fleet_dir, ref, ids, route, hidden


def write_fleet(fleet_dir, b, ids=None, route=None, hidden=None):
    save_file(b, str(fleet_dir / "fleet_boundaries.safetensors"))
    if ids is not None:
        (fleet_dir / "fleet_output_ids.json").write_text(json.dumps(ids))
    if route is not None:
        (fleet_dir / "fleet_route_log.json").write_text(json.dumps(route))
    if hidden is not None:
        save_file({"hidden": hidden}, str(fleet_dir / "fleet_hidden_per_layer.safetensors"))


def go(ref_dir, fleet_dir, cal=None):
    return compare.run(ref_dir, fleet_dir, cal, fleet_dir / "correctness_report.md")


def by_key(result):
    return {r["key"]: r for r in result["boundaries"]}


def perturb(t, rel):
    """Scale a tensor's error to a given rel_err in FP32, then keep its dtype."""
    d = torch.randn(t.shape, generator=torch.Generator().manual_seed(1))
    d = d / torch.linalg.vector_norm(d) * torch.linalg.vector_norm(t.float()) * rel
    return (t.float() + d).to(t.dtype)


def test_identical_passes_everything(dirs):
    ref_dir, fleet_dir, ref, ids, route, hidden = dirs
    write_fleet(fleet_dir, dict(ref), ids, route, hidden)
    r = go(ref_dir, fleet_dir)
    assert r["overall"] == "PASS"
    assert r["n_fail"] == 0 and r["n_missing"] == 0
    rows = by_key(r)
    assert "L0.layer_in" not in rows                      # auxiliary keys are not rows
    assert rows["L1.B9.topk_idx"]["result"] == "PASS"
    assert rows["head.B16.token"]["result"] == "PASS"
    assert rows["L0.B7.x_res_attn"]["rel_err"] == 0.0
    assert rows["L0.B7.x_res_attn"]["cos_sim"] == pytest.approx(1.0, abs=1e-12)
    assert r["output_ids"]["all_match"] and r["route_log"]["result"] == "PASS"
    assert r["growth_curve"]["result"] == "PASS" and r["growth_curve"]["max_rel_err"] == 0.0
    md = (fleet_dir / "correctness_report.md").read_text()
    assert "| `L1.B13.layer_out` | B13 | layer |" in md and "Overall: **PASS**" in md
    assert (fleet_dir / "correctness_report.json").exists()


def test_thresholds_and_calibration(dirs):
    ref_dir, fleet_dir, ref, ids, route, hidden = dirs
    b = dict(ref)
    b["L1.B7.x_res_attn"] = perturb(ref["L1.B7.x_res_attn"], 0.03)   # gemv threshold 2e-2
    b["L1.B1.norm1"] = perturb(ref["L1.B1.norm1"], 0.005)            # norm threshold 1e-2
    write_fleet(fleet_dir, b)
    r = go(ref_dir, fleet_dir)
    rows = by_key(r)
    assert rows["L1.B7.x_res_attn"]["result"] == "FAIL"
    assert 0.02 < rows["L1.B7.x_res_attn"]["rel_err"] < 0.04
    assert rows["L1.B1.norm1"]["result"] == "PASS"
    assert r["overall"] == "FAIL"
    assert "not calibrated" in r["threshold_source"]
    # a calibrated floor of 1e-2 for the gemv class moves the threshold to 4e-2
    cal = fleet_dir / "calibration.json"
    cal.write_text(json.dumps({"floor": {"gemv": 1e-2}}))
    r = go(ref_dir, fleet_dir, cal)
    rows = by_key(r)
    assert rows["L1.B7.x_res_attn"]["threshold"] == pytest.approx(4e-2)
    assert rows["L1.B7.x_res_attn"]["floor"] == pytest.approx(1e-2)
    assert rows["L1.B7.x_res_attn"]["result"] == "PASS"
    assert rows["L1.B1.norm1"]["threshold"] == common.CLASS_THRESHOLD["norm"]   # untouched class
    assert r["overall"] == "PASS"


def test_exact_checks(dirs):
    ref_dir, fleet_dir, ref, ids, route, hidden = dirs
    b = dict(ref)
    b["L1.B9.topk_idx"] = torch.tensor([2, 5])            # same set, other order
    write_fleet(fleet_dir, b)
    assert by_key(go(ref_dir, fleet_dir))["L1.B9.topk_idx"]["result"] == "PASS"
    b["L1.B9.topk_idx"] = torch.tensor([2, 6])
    b["head.B16.token"] = torch.tensor([41])
    write_fleet(fleet_dir, b)
    rows = by_key(go(ref_dir, fleet_dir))
    assert rows["L1.B9.topk_idx"]["result"] == "FAIL" and "fleet [2, 6] ref [2, 5]" in rows["L1.B9.topk_idx"]["detail"]
    assert rows["head.B16.token"]["result"] == "FAIL"


def test_b10_aligned_by_expert_id_and_nonlayer_keys(dirs):
    ref_dir, fleet_dir, ref, ids, route, hidden = dirs
    b = dict(ref)
    b["L1.B9.topk_idx"] = torch.tensor([2, 5])             # the reference has [5, 2] / [0.4, 0.3]
    b["L1.B10.topk_w"] = torch.tensor([0.3, 0.4])
    b["prologue.embed"] = torch.randn(H).bfloat16()        # a run stopped after the embed
    write_fleet(fleet_dir, b)
    r = go(ref_dir, fleet_dir)
    rows = by_key(r)
    assert rows["L1.B9.topk_idx"]["result"] == "PASS"
    assert rows["L1.B10.topk_w"]["result"] == "PASS" and rows["L1.B10.topk_w"]["rel_err"] == 0.0
    b["L1.B9.topk_idx"] = torch.tensor([2, 6])
    write_fleet(fleet_dir, b)
    rows = by_key(go(ref_dir, fleet_dir))
    assert rows["L1.B10.topk_w"]["result"] == "FAIL" and "expert sets differ" in rows["L1.B10.topk_w"]["detail"]


def test_output_ids_and_route_log(dirs):
    ref_dir, fleet_dir, ref, ids, route, hidden = dirs
    bad_ids = list(ids)
    bad_ids[5] = 999
    fleet_route = [[{"idx": [2, 5, 64, 65], "w": [0.3, 0.4, 1.0, 1.0]},
                    {"idx": [7, 1, 64, 65], "w": [0.2, 0.5, 1.0, 1.0]}] for _ in range(4)]
    write_fleet(fleet_dir, dict(ref), bad_ids, fleet_route)
    r = go(ref_dir, fleet_dir)
    assert r["output_ids"]["result"] == "FAIL" and r["output_ids"]["first_divergence"] == 5
    assert r["output_ids"]["matched_prefix"] == 5
    assert r["route_log"]["result"] == "PASS" and r["route_log"]["steps_compared"] == 4   # forced 64, 65 ignored
    fleet_route[3][1]["idx"] = [7, 3, 64, 65]
    write_fleet(fleet_dir, dict(ref), ids[:8], fleet_route)
    r = go(ref_dir, fleet_dir)
    assert r["output_ids"]["result"] == "PARTIAL" and r["output_ids"]["first_divergence"] is None  # 8 of 32
    assert r["output_ids"]["matched_prefix"] == 8 and not r["output_ids"]["all_match"]
    assert r["route_log"]["result"] == "FAIL" and r["route_log"]["mismatches"][0] == {
        "step": 3, "moe_layer_index": 1, "ref": [1, 7], "fleet": [3, 7]}


def test_missing_and_shape_mismatch(dirs):
    ref_dir, fleet_dir, ref, ids, route, hidden = dirs
    b = dict(ref)
    b["L1.B11.expert_3"] = torch.randn(H).bfloat16()       # Fleet chose an expert the ref did not
    b["L0.B2.q"] = torch.randn(NH * 24 + 1).bfloat16()
    write_fleet(fleet_dir, b)
    r = go(ref_dir, fleet_dir)
    rows = by_key(r)
    assert rows["L1.B11.expert_3"]["result"] == "MISSING_REF"
    assert rows["L0.B2.q"]["result"] == "FAIL" and "shape mismatch" in rows["L0.B2.q"]["detail"]
    assert r["overall"] == "FAIL" and r["n_missing"] == 1


def test_growth_curve(dirs):
    ref_dir, fleet_dir, ref, ids, route, hidden = dirs
    fh = hidden.clone()
    fh[2] = perturb(hidden[2], 0.2)                        # layer threshold 5e-2
    write_fleet(fleet_dir, dict(ref), hidden=fh)
    r = go(ref_dir, fleet_dir)
    assert r["growth_curve"]["result"] == "FAIL"
    assert r["growth_curve"]["rel_err_per_layer"][0] == 0.0 and r["growth_curve"]["rel_err_per_layer"][2] > 0.1
    assert r["overall"] == "FAIL"


def test_metrics_edge_cases():
    z = torch.zeros(4)
    assert compare.metrics(z, z)["rel_err"] == 0.0 and compare.metrics(z, z)["cos_sim"] == 1.0
    m = compare.metrics(torch.ones(4), torch.zeros(4))
    assert m["rel_err"] == float("inf") and m["cos_sim"] == 0.0
    m = compare.metrics(torch.tensor([1.0, 2.0]).bfloat16(), torch.tensor([1.0, 2.0]))
    assert m["rel_err"] == 0.0 and m["n"] == 2
