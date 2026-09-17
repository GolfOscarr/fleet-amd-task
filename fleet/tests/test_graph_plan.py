"""graph_plan / build_graph dry run against the design's counts."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from fleet import graph_plan as G  # noqa: E402
from fleet import build_graph as B  # noqa: E402
from fleet.pack_weights import REAL_DIMS, Dims  # noqa: E402


def graph_counts_output():
    import subprocess
    return subprocess.run([sys.executable, str(ROOT / "docs/design-doc/sources/graph_counts.py")],
                          check=True, capture_output=True, text=True).stdout


def test_full_graph_matches_graph_counts():
    plan, calls = B.dry_run()
    s = G.summary(plan)
    assert s["ops"] == 326 and s["tasks"] == 1880
    assert s["by_status"] == {"reuse": 217, "variant": 2, "new": 107}
    assert s["n_splits"] == 33
    assert len(calls) == 326
    gc_out = graph_counts_output()        # the design's counting script agrees
    assert "ops (events): 326" in gc_out and "tasks in graph: 1880" in gc_out


def test_per_layer_structure():
    plan, calls = B.dry_run(layers=2, head=False, s_max=1025)
    methods = [c["method"] for c in calls]
    assert methods[0] == "embed_layer"
    layer0 = methods[1:10]
    assert layer0 == ["rmsnorm_layer", "gang_linear_layer", "mla_prep_mi300@new", "mla_attend_mi300@new",
                      "mla_merge_uv_mi300@new", "gang_linear_with_residual_layer", "rmsnorm_layer",
                      "gang_linear_silu_layer", "gang_linear_with_residual_layer"]
    layer1 = methods[10:22]
    assert layer1[:7] == layer0[:7]
    assert layer1[7:] == ["moe_router_mi300@new", "gang_moe_w13_linear_layer", "moe_silu_mul_layer",
                          "gang_moe_w2_linear_layer", "moe_mul_sum_add_layer"]
    assert plan.n_ops == 22 and plan.n_tasks == 1 + 51 + 68
    # every op shares a tensor with its predecessor (the runtime's linking rule)
    prev = None
    for c in plan.calls:
        names = set()
        for v in c.args.values():
            if isinstance(v, str) and v in plan.tensors:
                names.add(v)
            elif isinstance(v, tuple):
                names.update(x for x in v if isinstance(x, str) and x in plan.tensors)
        if prev is not None:
            assert names & prev, (c.method, names, prev)
        prev = names


def test_tiles_and_params():
    plan, calls = B.dry_run()
    by_type = {}
    for c in calls:
        by_type.setdefault(c["task_type"], []).append(c)
    qkva = [c for c in by_type["gang_linear_mi300"] if c["output"] == "qkva"][0]
    assert qkva["params"] == [3648, 24, 1, 1, 19, 19, 0]
    lm = [c for c in by_type["gang_linear_mi300"] if c["output"] == "logits"][0]
    assert lm["params"] == [102400, 64, 1, 1, 200, 200, 0]
    o = [c for c in by_type["gang_linear_res_mi300"] if c["input"] == "attn"][0]
    assert o["params"][:2] == [2048, 32] and o["params"][4] == 8
    silu = by_type["gang_linear_silu_mi300"][0]
    assert silu["params"] == [11264, 64, 1, 1, 22, 22, 0]
    w13 = by_type["gang_moe_w13_linear_mi300"][0]
    assert w13["params"] == [44, 9, 396]                    # n_tiles, max experts per XCD, bound
    w2 = by_type["gang_moe_w2_linear_mi300"][0]
    assert w2["params"] == [32, 9, 288]
    att = by_type["mla_attend_mi300"][0]
    assert att["params"][1:] == [32, 33, 5, 16, 512, 64]
    assert att["params"][0] == G.float_bits(0.1147213867929261)
    assert att["imaps"][-1] == [0, -1, -1]                   # partials partitioned on the split dim
    mrg = by_type["mla_merge_uv_mi300"][0]
    assert mrg["params"] == [32, 33, 2, 16, 128, 512]
    rt = by_type["moe_router_mi300"][0]
    assert rt["params"][:3] == [6, 64, 2] and rt["params"][4] == 0 and rt["params"][5] == 2048
    assert by_type["moe_router_mi300"][-1]["params"][4] == 25   # layer_index of layer 26
    assert by_type["argmax_partial"][0]["params"] == [50]
    assert by_type["argmax_reduce"][0]["params"] == [2048, 1]
    assert by_type["embedding"][0]["params"] == [0]
    assert len(by_type["mla_prep_mi300"]) == 27 and len(by_type["moe_router_mi300"]) == 26


def test_float_bits_round_trips():
    import struct
    for x in (0.1147213867929261, 1.0, -2.5):
        b = G.float_bits(x)
        assert struct.unpack("<f", struct.pack("<i", b))[0] == pytest.approx(x, rel=1e-7)


def test_debug_and_head_options():
    plan, calls = B.dry_run(layers=3, head=True, debug=True)
    assert [c for c in calls if c["task_type"] == "copy_mi300"] and plan.n_ops == 1 + 9 + 12 + 12 + 3 + 4
    assert "dbg_x_res_2" in plan.tensors and "W_lm" in plan.tensors
    plan, calls = B.dry_run(layers=1, head=False)
    assert "W_lm" not in plan.tensors and "topk_w" not in plan.tensors
    assert plan.n_ops == 10


def test_debug_scores_option():
    plan, calls = B.dry_run(layers=1, head=False, debug_scores=True)
    att = [c for c in calls if c["task_type"] == "mla_attend_mi300"][0]
    assert att["inputs"] == ["ql_nope", "q_pe", "c_kv_0", "k_pe_0", "partials", "scores"]
    assert plan.tensors["scores"].shape == (16, 1056) and plan.tensors["scores"].dtype == "f32"
    plan, calls = B.dry_run(layers=1, head=False)
    assert "scores" not in plan.tensors and len([c for c in calls if c["task_type"] == "mla_attend_mi300"][0]["inputs"]) == 5


def test_input_bytes_and_shapes():
    plan, _ = B.dry_run()
    inputs = {n: t for n, t in plan.tensors.items() if t.kind == "input"}
    weights = sum(G._bytes(t) for n, t in inputs.items()
                  if not t.source.startswith(("capture:", "meta:")))
    from fleet.pack_weights import EXPECTED_PACKED_BYTES
    assert weights == EXPECTED_PACKED_BYTES
    assert plan.tensors["partials"].shape == (33, 16, 516)   # padded row P_ROW (P2)
    assert plan.tensors["c_kv_0"].shape == (1056, 512) and plan.tensors["k_pe_26"].shape == (1056, 64)
    assert plan.tensors["route_log"].shape == (32, 26, 8)


def test_constraints_are_enforced():
    bad = Dims(**{**REAL_DIMS.__dict__, "V": 102400 + 8})     # 102408 / 8 = 12801, not a multiple of 64
    with pytest.raises(AssertionError):
        B.dry_run(dims=bad)
    bad = Dims(**{**REAL_DIMS.__dict__, "I_MOE": 1400})
    with pytest.raises(AssertionError):
        B.dry_run(dims=bad)


# ---- the runtime's chain rule (docs/gpu-experiments/02-validation/01-preparation.md, P3) ----------------

@pytest.mark.parametrize("layers,head,debug,stop_after", [
    (27, True, False, None), (27, True, True, None), (27, False, True, None), (2, False, True, None),
    (8, True, False, "L7.combine"), (2, True, True, "L1.mla_prep"), (1, False, True, None),
    (3, True, False, "head.argmax_reduce"),
])
def test_every_operator_reads_what_its_predecessor_wrote(layers, head, debug, stop_after):
    plan, _ = B.dry_run(layers=layers, head=head, debug=debug, stop_after=stop_after)
    assert plan.chain_violations() == []


def test_debug_snapshot_feeds_the_next_norm():
    plan, calls = B.dry_run(layers=3, head=True, debug=True)
    by_label = {c.label: c for c in plan.calls}
    assert by_label["L0.norm1"].args["input"] == "x_res"
    assert by_label["L1.norm1"].args["input"] == "dbg_x_res_0"
    assert by_label["L2.norm1"].args["input"] == "dbg_x_res_1"
    assert by_label["head.norm"].args["input"] == "dbg_x_res_2"
    assert by_label["L1.o_proj"].args["residual"] == "x_res"        # the residual path is untouched
    plan, _ = B.dry_run(layers=3, head=True, debug=False)
    assert all(c.args["input"] == "x_res" for c in plan.calls if c.label.endswith("norm1") or c.label == "head.norm")


def test_chain_rule_catches_the_broken_snapshot_wiring():
    plan = G.build_plan(layers=3, head=True, debug=True)
    for c in plan.calls:
        if c.label in ("L1.norm1", "head.norm"):
            c.args["input"] = "x_res"                                  # the wiring that failed on 2026-09-15
    assert plan.chain_violations() == [("L0.snapshot", "L1.norm1"), ("L2.snapshot", "head.norm")]


# ---- per-tile linears (docs/gpu-experiments/02-validation/01-preparation.md, P5) ------------------------

def test_tile_linears_flips_the_four_dense_linears():
    from fleet.graph_plan import grid_for_linear, REAL_DIMS as D
    plan, _ = B.dry_run(layers=2, head=True, tile_linears=True)
    by = {c.label: c for c in plan.calls}
    # qkva, o_proj (both layers), down (layer 0), lm_head become the stock non-gang linears
    assert by["L0.qkva"].method == "linear_layer" and by["L1.qkva"].method == "linear_layer"
    assert by["L0.o_proj"].method == "linear_with_residual_layer"
    assert by["L1.o_proj"].method == "linear_with_residual_layer"
    assert by["L0.down"].method == "linear_with_residual_layer"
    assert by["head.lm_head"].method == "linear_layer"
    # each spreads over many tasks instead of the gang's 8, with the demo's grid heuristic
    assert by["L0.qkva"].tasks == grid_for_linear(D.Q_OUT + D.KVA_OUT) == 96
    assert by["L0.o_proj"].tasks == grid_for_linear(D.H) == 64
    assert by["head.lm_head"].tasks == grid_for_linear(D.V) == 400
    assert by["L0.qkva"].args["grid_dim"] == (96, 1, 1)
    # the silu-fused gate_up and the MoE linears stay gang
    assert by["L0.gate_up"].method == "gang_linear_silu_layer"
    assert by["L1.w13"].method == "gang_moe_w13_linear_layer"


def test_tile_linears_default_off_leaves_the_gang_plan():
    plan_gang, _ = B.dry_run(layers=2, head=True)
    plan_tile, _ = B.dry_run(layers=2, head=True, tile_linears=True)
    gang = {c.label: c.method for c in plan_gang.calls}
    assert gang["L0.qkva"] == "gang_linear_layer" and gang["head.lm_head"] == "gang_linear_layer"
    assert gang["L0.o_proj"] == "gang_linear_with_residual_layer"
    # the two plans differ only in those linear methods and their task counts
    assert plan_gang.n_ops == plan_tile.n_ops and plan_tile.n_tasks > plan_gang.n_tasks


def test_tile_linears_output_sizes_divide_the_grid():
    from fleet.graph_plan import grid_for_linear, REAL_DIMS as D
    for size in (D.Q_OUT + D.KVA_OUT, D.H, D.V):
        assert size % grid_for_linear(size) == 0


@pytest.mark.parametrize("head,debug", [(True, False), (True, True), (False, False)])
def test_tile_linears_keeps_the_chain_rule(head, debug):
    plan, _ = B.dry_run(layers=27, head=head, debug=debug, tile_linears=True)
    assert plan.chain_violations() == []


def test_fuse_norm2_folds_the_post_attention_norm_into_the_router():
    """O1 (docs/gpu-experiments/03-acceleration): no L{l}.norm2 in the MoE layers, the router
    reads x_res and the norm weight and writes h; layer 0 keeps its norm; the chain holds."""
    plan, calls = B.dry_run(layers=27, head=True, fuse_norm2=True)
    labels = [c.label for c in plan.calls]
    assert "L0.norm2" in labels and not any(l.endswith(".norm2") for l in labels if l != "L0.norm2")
    assert plan.n_ops == 300 and plan.n_tasks == 1854 and not plan.chain_violations()
    by = {c.label: c for c in plan.calls}
    r = by["L5.router"]
    assert r.args["input"] == "x_res" and r.args["w_norm"] == "w_norm2_5" and r.args["h"] == "h"
    assert r.args["eps"] == G.RMS_EPS
    # the recorded call: the fused registration with three inputs, six outputs and seven params
    rec = [c for c in calls if c["method"] == "moe_router_norm_mi300@new"]
    assert len(rec) == 26
    assert rec[0]["inputs"] == ["x_res", "w_norm2_1", "W_gate_1", "h", "topk_w", "routing", "mask",
                                "logits_router", "route_log"]
    assert len(rec[0]["params"]) == 7 and rec[0]["params"][6] == G.float_bits(G.RMS_EPS)
    # the gate-up still reads h, now written by the router (the chain's shared tensor)
    assert by["L5.w13"].args["input"] == "h"


def test_fuse_norm2_default_off_leaves_the_plan_unchanged():
    plan_off, calls_off = B.dry_run(layers=27, head=True)
    assert plan_off.n_ops == 326 and plan_off.n_tasks == 1880
    assert not any(c["method"] == "moe_router_norm_mi300@new" for c in calls_off)
    assert sum(c["method"] == "moe_router_mi300@new" for c in calls_off) == 26
    r = {c.label: c for c in plan_off.calls}["L5.router"]
    assert r.args["input"] == "h" and "w_norm" not in r.args and "h" not in r.args


def test_fuse_silu_folds_the_silu_into_the_expert_down_projection():
    """O2 (docs/gpu-experiments/03-acceleration): no L{l}.silu, w2 reads mid and a per-tile scratch;
    with O1 the MoE layer has 10 operators and the graph 274."""
    from fleet.graph_plan import REAL_DIMS as D, XCDS
    plan, calls = B.dry_run(layers=27, head=True, fuse_norm2=True, fuse_silu=True)
    labels = [c.label for c in plan.calls]
    assert not any(l.endswith(".silu") for l in labels)
    assert plan.n_ops == 274 and plan.n_tasks == 1854 - 26 * 8 and not plan.chain_violations()
    by = {c.label: c for c in plan.calls}
    w2 = by["L5.w2"]
    assert w2.method == "gang_moe_w2_silu_linear_layer" and w2.tasks == XCDS
    assert w2.args["input"] == "mid" and w2.args["output"] == "out8" and w2.args["scratch"] == "w2_scratch"
    assert plan.tensors["w2_scratch"].shape == (XCDS * ((D.E_TOTAL + 7) // 8) * (D.H // 64), D.I_MOE)
    assert "act8" not in plan.tensors
    rec = [c for c in calls if c["method"] == "gang_moe_w2_silu_linear_mi300@new"]
    assert len(rec) == 26
    assert rec[0]["inputs"] == ["mid", "W2_1", "routing", "mask", "out8", "w2_scratch"]
    assert rec[0]["params"] == [D.H // 64, (D.E_TOTAL + 7) // 8, ((D.E_TOTAL + 7) // 8) * (D.H // 64)]
    # the stock w2's imaps, plus the whole scratch
    assert rec[0]["imaps"] == [[-1, -1, -1], [-1, 1, -1], [-1, -1, -1], [-1, -1, -1], [-1, 2, -1], [-1, -1, -1]]
    # the combine still reads out8, written by the fused w2 (the chain's shared tensor)
    assert by["L5.combine"].args["input"] == "out8"


def test_fuse_silu_default_off_leaves_the_plan_unchanged():
    plan_off, calls_off = B.dry_run(layers=27, head=True)
    assert plan_off.n_ops == 326 and "act8" in plan_off.tensors and "w2_scratch" not in plan_off.tensors
    assert not any(c["method"] == "gang_moe_w2_silu_linear_mi300@new" for c in calls_off)
    plan_fs, _ = B.dry_run(layers=27, head=True, fuse_silu=True)
    assert plan_fs.n_ops == 300 and plan_fs.n_tasks == 1880 - 26 * 8
