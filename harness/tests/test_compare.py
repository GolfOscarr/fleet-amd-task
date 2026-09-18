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


# ---- F1 of docs/gpu-experiments/05-final: the route log's tie rule --------------------------------

def _w_all(top):
    """Eight experts' weights with the given (id: weight) pairs, the rest small and distinct."""
    w = [0.01 + 0.001 * i for i in range(8)]
    for i, v in top.items():
        w[i] = v
    return w


def _route(steps, top_ids, top_w, w_all):
    return [[{"idx": top_ids, "w": top_w, "w_all": w_all}, {"idx": [1, 7], "w": [0.5, 0.2],
              "w_all": _w_all({1: 0.5, 7: 0.2})}] for _ in range(steps)]


def test_route_log_tie_rule_classifies_a_near_tie_a_cascade_and_a_disagreement(dirs):
    ref_dir, fleet_dir, ref, ids, route, hidden = dirs
    # the reference's sixth-slot expert 2 at 0.030 against expert 3 at 0.0297: within 4 x floor
    ref_route = _route(6, [5, 2], [0.4, 0.030], _w_all({5: 0.4, 2: 0.030, 3: 0.0297, 4: 0.020}))
    (ref_dir / "ref_route_log.json").write_text(json.dumps(ref_route))
    fleet_route = [[{"idx": [5, 2, 64, 65], "w": []}, {"idx": [1, 7, 64, 65], "w": []}] for _ in range(6)]
    fleet_route[1][0]["idx"] = [5, 3, 64, 65]            # step 1: expert 3 took expert 2's slot
    fleet_route[3][0]["idx"] = [3, 4, 64, 65]            # step 3: two experts differ, after the tie
    fleet_route[4][1]["idx"] = [1, 6, 64, 65]            # step 4: a single swap far from a tie, after it
    write_fleet(fleet_dir, dict(ref), ids, fleet_route)
    cal = ref_dir / "calibration.json"
    cal.write_text(json.dumps({"floor": {"router": 0.0036}}))
    r = go(ref_dir, fleet_dir, cal=cal)["route_log"]
    assert r["rule"] == "tie" and abs(r["tol_rel"] - 4 * 0.0036) < 1e-12
    assert r["result"] == "PASS" and len(r["mismatches"]) == 3
    assert [m["class"] for m in r["mismatches"]] == ["tie", "cascade", "cascade"]
    t = r["ties"][0]
    assert (t["left"], t["came"]) == (2, 3) and abs(t["gap"] - 0.0003) < 1e-9 and t["gap"] <= t["tol"]
    # the same swap before any tie and outside the tolerance: a disagreement, FAIL
    ref_route = _route(6, [5, 2], [0.4, 0.030], _w_all({5: 0.4, 2: 0.030, 3: 0.020}))
    (ref_dir / "ref_route_log.json").write_text(json.dumps(ref_route))
    fleet_route = [[{"idx": [5, 2, 64, 65], "w": []}, {"idx": [1, 7, 64, 65], "w": []}] for _ in range(6)]
    fleet_route[1][0]["idx"] = [5, 3, 64, 65]
    fleet_route[3][0]["idx"] = [3, 4, 64, 65]
    write_fleet(fleet_dir, dict(ref), ids, fleet_route)
    r = go(ref_dir, fleet_dir, cal=cal)["route_log"]
    # a mismatch after a disagreement is a disagreement too: only ties seed cascades
    assert r["result"] == "FAIL" and [m["class"] for m in r["mismatches"]] == ["disagreement", "disagreement"]
    assert r["disagreements"][0]["gap"] > r["disagreements"][0]["tol"]
    # a multi-expert difference with no earlier tie is a disagreement too; no calibration: the fallback tolerance
    fleet_route[1][0]["idx"] = [5, 2, 64, 65]
    write_fleet(fleet_dir, dict(ref), ids, fleet_route)
    r = go(ref_dir, fleet_dir)["route_log"]
    assert r["result"] == "FAIL" and r["tol_rel"] == compare.ROUTE_TOL_FALLBACK
    assert [m["class"] for m in r["mismatches"]] == ["disagreement"]
    report = (fleet_dir / "correctness_report.md").read_text()
    assert "1 disagreements (the tie rule" in report


def test_route_log_without_weights_keeps_the_exact_rule(dirs):
    ref_dir, fleet_dir, ref, ids, route, hidden = dirs      # the fixture's reference has no w_all
    fleet_route = [[{"idx": [5, 2, 64, 65], "w": []}, {"idx": [1, 7, 64, 65], "w": []}] for _ in range(6)]
    fleet_route[2][1]["idx"] = [1, 3, 64, 65]
    write_fleet(fleet_dir, dict(ref), ids, fleet_route)
    r = go(ref_dir, fleet_dir)["route_log"]
    assert r["rule"] == "exact" and r["tol_rel"] is None and r["result"] == "FAIL"
    assert r["mismatches"] == [{"step": 2, "moe_layer_index": 1, "ref": [1, 7], "fleet": [1, 3]}]
    assert "the exact rule" in (fleet_dir / "correctness_report.md").read_text()


def test_round4_finals_route_mismatches_are_single_swaps_and_cascades():
    """The replay on the record (docs/gpu-experiments/05-final/03-local-preparation.md, F1): every
    mismatch of the fifteen 48-task finals of 2026-09-18 is a single swap, or a multi-expert difference
    after a single swap of the same run; the round-4 reference has no weights, so the tolerance itself
    is first read on the VM (R1)."""
    ROOT = HARNESS.parent
    ref_log = json.loads((ROOT / "harness/ref/ref_route_log.json").read_text())
    runs = sorted((ROOT / "env/hw/20260918/runs").glob("L27_head_it3[012]*lg48*"))
    runs = [r for r in runs if (r / "fleet_route_log.json").exists()]
    assert len(runs) >= 15, [r.name for r in runs]
    total = singles = 0
    for run in runs:
        r = compare.compare_route_log(ref_log, json.loads((run / "fleet_route_log.json").read_text()))
        assert r["rule"] == "exact" and r["result"] == "FAIL"
        first_single = None
        for m in r["mismatches"]:
            total += 1
            left, came = set(m["ref"]) - set(m["fleet"]), set(m["fleet"]) - set(m["ref"])
            if len(left) == 1 and len(came) == 1:
                singles += 1
                first_single = m["step"] if first_single is None else first_single
            else:
                assert first_single is not None and m["step"] > first_single, (run.name, m)
    assert total >= 300 and singles >= 280, (total, singles)


# ---- F2 of docs/gpu-experiments/05-final: iteration-aware boundaries ------------------------------

def test_boundary_iteration_rule():
    assert common.boundary_iteration("head.B15.logits", 1) == 0
    assert common.boundary_iteration("head.B15.logits", 32) == 31
    assert common.boundary_iteration("L1.B2.q", 29) == 28
    for key in ("L0.B3.c_kv", "L1.B3.k_pe", "head.B16.token"):     # written at step 0 and never again
        assert common.boundary_iteration(key, 32) == 0


def test_later_iteration_boundaries_are_not_comparable_and_the_ids_decide(dirs):
    """The it32 rows of the round-4 record: head.B15.logits dumped from iteration 31 failed against
    step 0 by construction; now it is NOT_COMPARABLE, the cache rows and the token still compare,
    and the verdict is the ids' and the route log's."""
    ref_dir, fleet_dir, ref, ids, route, hidden = dirs
    fleet = dict(ref)
    fleet["head.B15.logits"] = ref["head.B15.logits"] * 3.0             # a different token's logits
    write_fleet(fleet_dir, fleet, ids, route)
    (fleet_dir / "fleet_run_meta.json").write_text(json.dumps({"iters": 32, "head": True}))
    r = go(ref_dir, fleet_dir)
    rows = by_key(r)
    assert r["iters"] == 32 and r["later_reference"] is None
    assert rows["head.B15.logits"]["result"] == "NOT_COMPARABLE" and rows["head.B15.logits"]["iteration"] == 31
    assert "iteration 31" in rows["head.B15.logits"]["detail"]
    assert rows["L1.B2.q"]["result"] == "NOT_COMPARABLE"                 # every non-invariant boundary
    assert rows["L0.B3.c_kv"]["result"] == "PASS" and rows["head.B16.token"]["result"] == "PASS"
    assert r["overall"] == "PASS" and r["n_fail"] == 0 and r["n_not_comparable"] > 0
    assert r["output_ids"]["result"] == "PASS" and r["route_log"]["result"] == "PASS"
    report = (fleet_dir / "correctness_report.md").read_text()
    assert "dumped from iteration 31" in report and "not comparable" in report
    # the same run at one iteration: the head compares and fails as before
    (fleet_dir / "fleet_run_meta.json").write_text(json.dumps({"iters": 1, "head": True}))
    r = go(ref_dir, fleet_dir)
    assert by_key(r)["head.B15.logits"]["result"] == "FAIL" and r["overall"] == "FAIL"
    # a failing id with every boundary not comparable is still a FAIL
    bad_ids = list(ids); bad_ids[3] = 999
    write_fleet(fleet_dir, fleet, bad_ids, route)
    (fleet_dir / "fleet_run_meta.json").write_text(json.dumps({"iters": 32, "head": True}))
    r = go(ref_dir, fleet_dir)
    assert r["overall"] == "FAIL" and r["output_ids"]["result"] == "FAIL"


def test_boundaries_of_layers_the_reference_did_not_capture_are_reported_not_failed(dirs):
    """A 27-layer run dumps the cache rows of layers 2 to 25 and layer 26's boundaries (the last
    writers); the reference has layers 0 and 1: NOT_CAPTURED, not MISSING_REF (the round-4 record's
    64 missing rows). A key absent inside a captured layer is still a missing reference."""
    ref_dir, fleet_dir, ref, ids, route, hidden = dirs
    fleet = dict(ref)
    fleet["L5.B3.c_kv"] = ref["L1.B3.c_kv"].clone()
    fleet["L26.B2.q"] = ref["L1.B2.q"].clone()
    write_fleet(fleet_dir, fleet, ids, route)
    r = go(ref_dir, fleet_dir)
    rows = by_key(r)
    assert rows["L5.B3.c_kv"]["result"] == "NOT_CAPTURED" and rows["L26.B2.q"]["result"] == "NOT_CAPTURED"
    assert "layers 0, 1" in rows["L5.B3.c_kv"]["detail"]
    assert r["overall"] == "PASS" and r["n_missing"] == 0 and r["n_not_captured"] == 2
    assert "2 of layers the reference did not capture" in (fleet_dir / "correctness_report.md").read_text()
    fleet["L1.B99.made_up"] = ref["L1.B2.q"].clone()                  # not a boundary key: ignored
    fleet["L1.B4.q_pe"] = ref["L1.B4.q_pe"].clone()
    del fleet["L5.B3.c_kv"]
    write_fleet(fleet_dir, fleet, ids, route)
    ref_short = {k: v for k, v in ref.items() if k != "L1.B4.q_pe"}
    save_file(ref_short, str(ref_dir / "ref_boundaries_step0.safetensors"))
    r = go(ref_dir, fleet_dir)
    assert by_key(r)["L1.B4.q_pe"]["result"] == "MISSING_REF" and r["overall"] == "FAIL" and r["n_missing"] == 1


def test_later_iteration_boundaries_compare_against_the_reference_step_file(dirs):
    """The right form: ref_boundaries_step31.safetensors present, the head compares against it."""
    ref_dir, fleet_dir, ref, ids, route, hidden = dirs
    later = dict(ref)
    later["head.B15.logits"] = ref["head.B15.logits"] * 3.0
    save_file(later, str(ref_dir / "ref_boundaries_step31.safetensors"))
    fleet = dict(ref)
    fleet["head.B15.logits"] = later["head.B15.logits"].clone()
    write_fleet(fleet_dir, fleet, ids, route)
    (fleet_dir / "fleet_run_meta.json").write_text(json.dumps({"iters": 32, "head": True}))
    r = go(ref_dir, fleet_dir)
    rows = by_key(r)
    assert r["later_reference"] == "ref_boundaries_step31.safetensors"
    assert rows["head.B15.logits"]["result"] == "PASS" and rows["head.B15.logits"]["iteration"] == 31
    assert rows["L0.B3.c_kv"]["result"] == "PASS" and r["overall"] == "PASS" and r["n_not_comparable"] == 0
    fleet["head.B15.logits"] = ref["head.B15.logits"].clone()          # step 0's logits at iteration 31: FAIL
    write_fleet(fleet_dir, fleet, ids, route)
    r = go(ref_dir, fleet_dir)
    assert by_key(r)["head.B15.logits"]["result"] == "FAIL" and "step31" in (fleet_dir / "correctness_report.md").read_text()


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
