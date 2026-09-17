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
    assert s["ops"] == 326 and s["tasks"] == 1880 + 27 * 15
    assert s["by_status"] == {"reuse": 217, "variant": 2, "new": 107}
    assert s["n_splits"] == 33
    assert len(calls) == 326
    gc_out = graph_counts_output()        # the design's counting script agrees
    assert "ops (events): 326" in gc_out and "tasks in graph: 2285" in gc_out


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
    assert plan.n_ops == 22 and plan.n_tasks == 1 + 51 + 68 + 2 * 15
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
    assert plan.n_ops == 300 and plan.n_tasks == 1854 + 27 * 15 and not plan.chain_violations()
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
    assert plan_off.n_ops == 326 and plan_off.n_tasks == 1880 + 27 * 15
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
    assert plan.n_ops == 274 and plan.n_tasks == 1854 + 27 * 15 - 26 * 8 and not plan.chain_violations()
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
    assert plan_fs.n_ops == 300 and plan_fs.n_tasks == 1880 + 27 * 15 - 26 * 8


def test_fuse_norm1_folds_the_input_norm_into_the_per_tile_linear():
    """O3 (docs/gpu-experiments/03-acceleration): no L{l}.norm1 and no head.norm; qkva and lm_head are
    the fused per-tile linear reading x_res, the norm weight and a per-task scratch; with O1 and O2
    the graph has 246 operators; the chain holds."""
    from fleet.graph_plan import grid_for_linear, REAL_DIMS as D
    plan, calls = B.dry_run(layers=27, head=True, fuse_norm2=True, fuse_silu=True, fuse_norm1=True, tile_linears=True)
    labels = [c.label for c in plan.calls]
    assert not any(l.endswith(".norm1") for l in labels) and "head.norm" not in labels
    assert plan.n_ops == 246 and plan.n_tasks == 5954 + 27 * 15 and not plan.chain_violations()
    by = {c.label: c for c in plan.calls}
    q = by["L5.qkva"]
    assert q.method == "linear_norm_layer" and q.tasks == grid_for_linear(D.Q_OUT + D.KVA_OUT) == 96
    assert q.args["input"] == "x_res" and q.args["w_norm"] == "w_norm1_5" and q.args["scratch"] == "qkva_scratch"
    assert q.args["eps"] == G.RMS_EPS and by["L5.mla_prep"].args["qkva"] == "qkva"
    lm = by["head.lm_head"]
    assert lm.method == "linear_norm_layer" and lm.tasks == grid_for_linear(D.V) == 400
    assert lm.args["input"] == "x_res" and lm.args["w_norm"] == "w_final_norm" and lm.args["scratch"] == "lm_scratch"
    assert plan.tensors["qkva_scratch"].shape == (96, D.H) and plan.tensors["lm_scratch"].shape == (400, D.H)
    # the recorded call: three inputs, two outputs, the stock linear's imaps plus the scratch on dim 0
    rec = [c for c in calls if c["method"] == "linear_norm_mi300@new"]
    assert len(rec) == 28
    assert rec[0]["inputs"] == ["x_res", "w_norm1_0", "W_qkva_0", "qkva", "qkva_scratch"]
    assert rec[0]["imaps"] == [[-1, -1, -1], [-1, -1, -1], [0, -1, -1], [1, -1, -1], [0, -1, -1]]
    assert rec[0]["params"] == [G.float_bits(G.RMS_EPS)]
    assert rec[-1]["inputs"] == ["x_res", "w_final_norm", "W_lm", "logits", "lm_scratch"]
    # the fused linear is per-tile whether or not --tile-linears is set
    plan_g, _ = B.dry_run(layers=27, head=True, fuse_norm2=True, fuse_silu=True, fuse_norm1=True)
    assert plan_g.n_ops == 246 and plan_g.n_tasks == 1646 - 28 + 27 * (96 - 8) + (400 - 8) + 27 * 15
    assert {c.label: c.method for c in plan_g.calls}["L0.qkva"] == "linear_norm_layer"


def test_fuse_norm1_default_off_and_debug_keep_the_stock_norms():
    plan_off, calls_off = B.dry_run(layers=27, head=True)
    assert plan_off.n_ops == 326 and "qkva_scratch" not in plan_off.tensors and "lm_scratch" not in plan_off.tensors
    assert not any(c["method"] == "linear_norm_mi300@new" for c in calls_off)
    # under --debug the stock norms stay (the snapshot wiring reads the copies)
    plan_dbg, calls_dbg = B.dry_run(layers=27, head=True, debug=True, fuse_norm1=True)
    plan_dbg0, _ = B.dry_run(layers=27, head=True, debug=True)
    assert plan_dbg.n_ops == plan_dbg0.n_ops and not any(c["method"] == "linear_norm_mi300@new" for c in calls_dbg)
    assert {c.label: c for c in plan_dbg.calls}["L1.norm1"].args["input"] == "dbg_x_res_0"
    assert not plan_dbg.chain_violations()


# ---- the GEMV linear (docs/gpu-experiments/04-kernels, L2 and L5) --------------------------

def test_gemv_linears_flips_the_three_dense_linears():
    """L2: qkva, o_proj and lm_head become one GEMV task type, with the input norm and the residual
    add as its flags; layer 0's down (K 11,264) stays the stock per-tile linear; no norm operator,
    no scratch tensor, and the chain holds."""
    from fleet.graph_plan import grid_for_linear, REAL_DIMS as D
    plan, calls = B.dry_run(layers=27, head=True, gemv_linears=True)
    labels = [c.label for c in plan.calls]
    assert not any(l.endswith(".norm1") for l in labels) and "head.norm" not in labels
    assert "qkva_scratch" not in plan.tensors and "lm_scratch" not in plan.tensors
    assert not plan.chain_violations()
    by = {c.label: c for c in plan.calls}
    for label in ("L5.qkva", "L5.o_proj", "head.lm_head"):
        assert by[label].method == "linear_gemv_layer" and by[label].status == "new"
    q = by["L5.qkva"]
    assert q.tasks == grid_for_linear(D.Q_OUT + D.KVA_OUT) == 96 and q.args["input"] == "x_res"
    assert q.args["w_norm"] == "w_norm1_5" and q.args["norm"] and not q.args["residual_add"]
    assert q.args["residual"] is None and q.args["eps"] == G.RMS_EPS
    o = by["L5.o_proj"]
    assert o.tasks == grid_for_linear(D.H) == 64 and o.args["input"] == "attn"
    assert o.args["residual"] == "x_res" and o.args["output"] == "x_res" and o.args["residual_add"]
    assert o.args["w_norm"] is None and not o.args["norm"]
    dn = by["L0.down"]                      # the stock per-tile residual linear, K 11,264
    assert dn.method == "linear_with_residual_layer" and dn.args["weight"] == "W_down_pad"
    lm = by["head.lm_head"]
    assert lm.tasks == grid_for_linear(D.V) == 400 and lm.args["w_norm"] == "w_final_norm" and lm.args["norm"]
    # the gate-up and the MoE linears stay gang, as under --tile-linears
    assert by["L0.gate_up"].method == "gang_linear_silu_layer"
    assert by["L5.w13"].method == "gang_moe_w13_linear_layer"
    # the recorded calls: the registration's input order, the imaps of the fused per-tile linear
    # plus the residual partitioned like the output, and the three params
    rec = [c for c in calls if c["method"] == "linear_gemv_mi300@new"]
    assert len(rec) == 27 + 27 + 1
    assert rec[0]["inputs"] == ["x_res", "w_norm1_0", "W_qkva_0", "qkva"]
    assert rec[0]["imaps"] == [[-1, -1, -1], [-1, -1, -1], [0, -1, -1], [1, -1, -1]]
    assert rec[0]["params"] == [1, 0, G.float_bits(G.RMS_EPS)]
    assert rec[1]["inputs"] == ["attn", "W_o_0", "x_res", "x_res"]
    assert rec[1]["imaps"] == [[-1, -1, -1], [0, -1, -1], [1, -1, -1], [1, -1, -1]]
    assert rec[1]["params"] == [0, 1, G.float_bits(0.0)]
    assert rec[-1]["inputs"] == ["x_res", "w_final_norm", "W_lm", "logits"]
    assert rec[-1]["params"] == [1, 0, G.float_bits(G.RMS_EPS)]


def test_gemv_linears_keeps_the_counts_of_the_fused_per_tile_plan():
    """L2: the operator count is unchanged and so is the task count at the default grids: the flag
    swaps the kernel of the four linears, it does not add or remove an operator or a task."""
    plan, _ = B.dry_run(layers=27, head=True, gemv_linears=True)
    ref, _ = B.dry_run(layers=27, head=True, fuse_norm1=True, tile_linears=True)
    assert (plan.n_ops, plan.n_tasks) == (ref.n_ops, ref.n_tasks) == (298, 6593)
    plan2, _ = B.dry_run(layers=27, head=True, fuse_norm2=True, fuse_silu=True, gemv_linears=True)
    ref2, _ = B.dry_run(layers=27, head=True, fuse_norm2=True, fuse_silu=True, fuse_norm1=True, tile_linears=True)
    assert (plan2.n_ops, plan2.n_tasks) == (ref2.n_ops, ref2.n_tasks) == (246, 6359)
    assert not plan2.chain_violations()


def test_gemv_linears_grid_overrides():
    """L2's --linear-grid and L5's --head-grid: the task count is the only thing that moves, and an
    override that does not divide an operator's rows (48 against o_proj's 2,048) leaves it alone."""
    from fleet.graph_plan import REAL_DIMS as D
    base, _ = B.dry_run(layers=27, head=True, gemv_linears=True)
    lg48, _ = B.dry_run(layers=27, head=True, gemv_linears=True, linear_grid=48)
    by = {c.label: c for c in lg48.calls}
    assert by["L5.qkva"].tasks == 48 and by["L5.qkva"].args["grid_dim"] == (48, 1, 1)
    assert by["L5.o_proj"].tasks == 64 and by["L0.down"].tasks == 64      # 48 does not divide 2,048
    assert lg48.n_ops == base.n_ops and lg48.n_tasks == base.n_tasks - 27 * (96 - 48) == 5297
    lg32, _ = B.dry_run(layers=27, head=True, gemv_linears=True, linear_grid=32)
    by32 = {c.label: c for c in lg32.calls}
    assert by32["L5.qkva"].tasks == 32 and by32["L5.o_proj"].tasks == 32
    assert by32["L0.down"].tasks == 64                                   # the page's override names qkva and o_proj
    assert lg32.n_tasks == base.n_tasks - 27 * (96 - 32) - 27 * (64 - 32) == 4001
    hg320, _ = B.dry_run(layers=27, head=True, gemv_linears=True, head_grid=320)
    assert {c.label: c for c in hg320.calls}["head.lm_head"].tasks == 320
    assert hg320.n_ops == base.n_ops and hg320.n_tasks == base.n_tasks - 80 == 6513
    for plan in (lg48, lg32, hg320):
        assert not plan.chain_violations()
    # the head's override is a single operator: it must divide the vocabulary
    with pytest.raises(AssertionError):
        B.dry_run(layers=1, head=True, gemv_linears=True, head_grid=300)
    with pytest.raises(AssertionError):
        B.dry_run(layers=1, head=True, gemv_linears=True, linear_grid=100)   # divides neither 3,648 nor 2,048
    with pytest.raises(AssertionError):
        B.dry_run(layers=1, head=True, linear_grid=32)                       # only under the flag
    assert D.V % 320 == 0


def test_gemv_linears_default_off_and_debug_keep_the_stock_plan():
    plan_off, calls_off = B.dry_run(layers=27, head=True)
    assert plan_off.n_ops == 326 and not any(c["method"] == "linear_gemv_mi300@new" for c in calls_off)
    assert {c.label: c.method for c in plan_off.calls}["L0.qkva"] == "gang_linear_layer"
    # under --debug the stock norms and the snapshot wiring stay, as with --fuse-norm1
    plan_dbg, calls_dbg = B.dry_run(layers=27, head=True, debug=True, gemv_linears=True)
    plan_dbg0, _ = B.dry_run(layers=27, head=True, debug=True)
    assert plan_dbg.n_ops == plan_dbg0.n_ops and plan_dbg.n_tasks == plan_dbg0.n_tasks
    assert not any(c["method"] == "linear_gemv_mi300@new" for c in calls_dbg)
    assert not plan_dbg.chain_violations()


@pytest.mark.parametrize("head,layers", [(True, 27), (False, 2), (True, 1)])
def test_gemv_linears_keeps_the_chain_rule(head, layers):
    plan, _ = B.dry_run(layers=layers, head=head, gemv_linears=True, fuse_norm2=True, fuse_silu=True)
    assert plan.chain_violations() == []


# ---- the w13 GEMV gang task (docs/gpu-experiments/04-kernels, L4) --------------------------

def w13_ranges(n=2816, tiles=37):
    """The tile ranges the kernel computes by arithmetic, re-derived here from the page's own
    words (4 tiles of 77 rows from 77 t, then 76 from 308 + 76 (t - 4)) rather than from
    graph_plan.w13_tile_rows, so the two formulas are compared and not just repeated."""
    out = []
    for t in range(tiles):
        out.append((77 * t, 77) if t < 4 else (308 + 76 * (t - 4), 76))
    return out


def test_w13_tile_ranges_partition_the_expert_rows():
    """S1: the 37 tiles cover 2,816 = 4 x 77 + 33 x 76 exactly once, and graph_plan's formula
    (which the suite's reference reuses) is the same one."""
    from fleet.graph_plan import REAL_DIMS as D, W13_GEMV_TILES, w13_tile_rows
    n = 2 * D.I_MOE
    assert (n, W13_GEMV_TILES) == (2816, 37) and G.NUM_WORKERS // 8 == 37
    ranges = [w13_tile_rows(t) for t in range(W13_GEMV_TILES)]
    assert ranges == w13_ranges(n, W13_GEMV_TILES)
    assert [r for _, r in ranges] == [77] * 4 + [76] * 33
    assert sum(r for _, r in ranges) == n == 4 * 77 + 33 * 76
    covered = [0] * n
    for r0, rows in ranges:
        assert 0 <= r0 and r0 + rows <= n
        for i in range(r0, r0 + rows):
            covered[i] += 1
    assert covered == [1] * n                      # a partition: every row in exactly one tile
    assert ranges[0][0] == 0 and ranges[4][0] == 308 and ranges[-1] == (2740, 76)
    with pytest.raises(AssertionError):
        w13_tile_rows(W13_GEMV_TILES)
    # a wave of the tile takes ceil(rows / 4) with the last one short, the GEMV's own map
    assert [-(-r // 4) for _, r in ranges] == [20] * 4 + [19] * 33


def test_w13_store_map_covers_every_row_once():
    """The kernel's epilogue: wave w of a tile owns rows [w * RPW, min((w + 1) * RPW, rows)) with
    RPW = ceil(rows / 4), walks them in batches of eight, and lane l < 8 of the batch stores row
    r0 + l when r0 + l is inside the wave's range. Emulated here over the whole grid: every one of
    the 2,816 rows of the expert is stored exactly once, by one (tile, wave, batch, lane)."""
    from fleet.graph_plan import REAL_DIMS as D, W13_GEMV_TILES, w13_tile_rows
    n, batch, waves = 2 * D.I_MOE, 8, 4
    stored = {}
    for tile in range(W13_GEMV_TILES):
        row0, rows = w13_tile_rows(tile, n)
        for wave in range(waves):
            rpw = -(-rows // waves)
            r_begin, r_end = wave * rpw, min((wave + 1) * rpw, rows)
            r_end = max(r_end, r_begin)
            for r0 in range(r_begin, r_end, batch):
                for lane in range(batch):
                    r = r0 + lane
                    if r < r_end:
                        key = row0 + r
                        assert key not in stored, (key, stored.get(key), (tile, wave, r0, lane))
                        stored[key] = (tile, wave, r0, lane)
    assert sorted(stored) == list(range(n))
    # the clamped rows a short batch loads are inside the wave's range, so no load leaves the tile
    for tile in range(W13_GEMV_TILES):
        row0, rows = w13_tile_rows(tile, n)
        rpw = -(-rows // waves)
        for wave in range(waves):
            r_begin, r_end = wave * rpw, min((wave + 1) * rpw, rows)
            for r0 in range(r_begin, max(r_end, r_begin), batch):
                for u in range(batch):
                    r = r0 + u if r0 + u < r_end else r_end - 1
                    assert 0 <= row0 + r < n


def test_gemv_w13_flips_every_moe_layer_and_records_37_tiles():
    """L4: the flag swaps the method of the 26 w13 operators, keeping the label, the tensors and
    the 8 tasks; the tiles recorded are 37 per expert and 9 x 37 per XCD (the registration's
    third param), against the stock 44 and 9 x 44."""
    from fleet.graph_plan import W13_GEMV_TILES
    plan, calls = B.dry_run(layers=27, head=True, gemv_w13=True)
    base, base_calls = B.dry_run(layers=27, head=True)
    assert (plan.n_ops, plan.n_tasks) == (base.n_ops, base.n_tasks) == (326, 2285)
    assert not plan.chain_violations()
    by, by_base = {c.label: c for c in plan.calls}, {c.label: c for c in base.calls}
    w13 = [c for c in plan.calls if c.label.endswith(".w13")]
    assert len(w13) == 26                          # layer 0 is dense: no expert gate-up
    for c in w13:
        assert c.method == "gang_moe_w13_gemv_layer" and c.status == "new"
        assert c.tasks == 8 and c.tiles == W13_GEMV_TILES == 37
        assert c.args["input"] == "h" and c.args["output"] == "mid"
        assert c.args["moe_routing_indices"] == "routing" and c.args["moe_mask"] == "mask"
        assert c.args["tiles_per_expert"] == 37
    assert by_base["L5.w13"].method == "gang_moe_w13_linear_layer" and by_base["L5.w13"].tiles == 44
    assert by["L5.w2"].method == by_base["L5.w2"].method          # the down projection is untouched
    assert by["L5.router"].method == by_base["L5.router"].method
    assert plan.tensors.keys() == base.tensors.keys()             # no tensor added or dropped
    rec = [c for c in calls if c["method"] == "gang_moe_w13_gemv_mi300@new"]
    assert len(rec) == 26
    assert rec[0]["inputs"] == ["h", "W13_1", "routing", "mask", "mid"]
    assert rec[0]["imaps"] == [[-1, -1, -1], [-1, 1, -1], [-1, -1, -1], [-1, -1, -1], [-1, 2, -1]]
    assert rec[0]["params"] == [37, 9, 9 * 37] == [37, 9, 333]    # tiles, max experts per XCD, total
    assert rec[0]["grid_dim"] == (8, 1, 1)
    # the stock form's params for comparison: 44 tiles, 9 x 44 per XCD
    stock = [c for c in base_calls if c["task_type"] == "gang_moe_w13_linear_mi300"]
    assert len(stock) == 26 and stock[0]["params"] == [44, 9, 9 * 44]


def test_gemv_w13_default_off_and_independent_of_the_other_flags():
    plan_off, calls_off = B.dry_run(layers=27, head=True)
    assert not any(c["method"] == "gang_moe_w13_gemv_mi300@new" for c in calls_off)
    assert {c.label: c.method for c in plan_off.calls}["L5.w13"] == "gang_moe_w13_linear_layer"
    # it is not disabled by --debug (it changes no norm) and composes with the other flags
    plan_dbg, calls_dbg = B.dry_run(layers=27, head=True, debug=True, gemv_w13=True)
    assert sum(c["method"] == "gang_moe_w13_gemv_mi300@new" for c in calls_dbg) == 26
    assert not plan_dbg.chain_violations()
    both, calls_both = B.dry_run(layers=27, head=True, gemv_linears=True, fuse_norm2=True,
                                 fuse_silu=True, gemv_w13=True)
    ref, _ = B.dry_run(layers=27, head=True, gemv_linears=True, fuse_norm2=True, fuse_silu=True)
    assert (both.n_ops, both.n_tasks) == (ref.n_ops, ref.n_tasks) == (246, 6359)
    assert sum(c["method"] == "gang_moe_w13_gemv_mi300@new" for c in calls_both) == 26
    assert not both.chain_violations()


@pytest.mark.parametrize("head,layers", [(True, 27), (False, 2), (True, 1)])
def test_gemv_w13_keeps_the_chain_rule(head, layers):
    plan, calls = B.dry_run(layers=layers, head=head, gemv_w13=True)
    assert plan.chain_violations() == []
    # one layer is the dense MLP alone: the flag then changes nothing
    assert sum(c["method"] == "gang_moe_w13_gemv_mi300@new" for c in calls) == max(layers - 1, 0)


def test_prefetch_adds_side_operators_that_the_chain_rule_skips():
    """O8 (docs/gpu-experiments/03-acceleration): --prefetch adds three side operators per layer
    (the layer's W_o after qkva, the next layer's W_qkva after o_proj, the active experts' W2 after
    w13), each registered right after its host; the chain of non-side operators is unchanged."""
    from fleet.graph_plan import grid_for_linear, PREFETCH_PARTS, TOPK_TOTAL_SLOTS, REAL_DIMS as D
    plan, calls = B.dry_run(layers=27, head=True, fuse_norm2=True, fuse_silu=True, fuse_norm1=True, prefetch=True)
    base, _ = B.dry_run(layers=27, head=True, fuse_norm2=True, fuse_silu=True, fuse_norm1=True)
    side = [c for c in plan.calls if c.side]
    assert len(side) == 27 + 26 + 26 and plan.n_ops == 246 + len(side)
    assert [c.label for c in plan.chain()] == [c.label for c in base.calls] and not plan.chain_violations()
    labels = [c.label for c in plan.calls]
    # each side operator directly follows its host
    assert labels[labels.index("L3.prefetch_W_o") - 1] == "L3.qkva"
    assert labels[labels.index("L3.prefetch_W_qkva_next") - 1] == "L3.o_proj"
    assert labels[labels.index("L3.prefetch_W2") - 1] == "L3.w13"
    assert "L26.prefetch_W_qkva_next" not in labels          # no next layer
    by = {c.label: c for c in plan.calls}
    assert by["L3.prefetch_W_o"].tasks == grid_for_linear(D.H) == 64 and by["L3.prefetch_W_o"].args["weight"] == "W_o_3"
    assert by["L3.prefetch_W_qkva_next"].args["weight"] == "W_qkva_4"
    assert by["L3.prefetch_W2"].tasks == TOPK_TOTAL_SLOTS * PREFETCH_PARTS and by["L3.prefetch_W2"].args["weight"] == "W2_3"
    assert plan.tensors["pf_dummy_w2"].shape == (TOPK_TOTAL_SLOTS * PREFETCH_PARTS, 4)
    # the recorded calls: the dense one partitions the weight and the dummy on dim 0, the expert one reads mask
    rec = [c for c in calls if c["method"] == "prefetch_mi300@new"]
    assert len(rec) == 53 and rec[0]["inputs"] == ["W_o_0", "pf_dummy_o"] and rec[0]["imaps"] == [[0, -1, -1], [0, -1, -1]]
    rec = [c for c in calls if c["method"] == "prefetch_moe_mi300@new"]
    assert len(rec) == 26 and rec[0]["inputs"] == ["W2_1", "mask", "pf_dummy_w2"] and rec[0]["params"] == [PREFETCH_PARTS]
    # a probe before an operator whose predecessor carries side operators lands right before the operator
    plan.insert_probe("L3.mla_prep")
    labels = [c.label for c in plan.calls]
    i = labels.index("L3.probe_mla_prep")
    assert labels[i - 1] == "L3.prefetch_W_o" and labels[i + 1] == "L3.mla_prep" and not plan.chain_violations()
    assert {c.label: c for c in plan.calls}["L3.mla_prep"].args["qkva"] == "qkva_probe"


def test_probe_before_an_operator_with_a_pair_of_inputs():
    """The argmax reduce reads (amax_v, amax_i); the probe rewires the pair's shared tensor."""
    plan, _ = B.dry_run(layers=1, head=True, probe_before="head.argmax_reduce")
    by = {c.label: c for c in plan.calls}
    assert by["head.probe_argmax_reduce"].args["output"] == "amax_v_probe"
    assert by["head.argmax_reduce"].args["input"] == ("amax_v_probe", "amax_i") and not plan.chain_violations()


def test_prefetch_default_off_leaves_the_plan_unchanged():
    plan, calls = B.dry_run(layers=27, head=True)
    assert plan.n_ops == 326 and not any(c.side for c in plan.calls)
    assert not any(c["method"].startswith("prefetch") for c in calls)
    assert "pf_dummy_o" not in plan.tensors


def test_empty_ladder_is_a_chain_of_copy_operators_with_one_event_per_boundary():
    """I3 (docs/gpu-experiments/03-acceleration): M operators of N copy tasks over two [N, 256]
    tensors; each reads its input whole and writes its own row (the output partitioned on dim 0),
    so the boundary is one event with N triggers; with --spin the first operator's tasks print."""
    from fleet.graph_plan import build_empty_plan, EMPTY_WIDTH
    plan = build_empty_plan(ops=5, tasks=40, spin=1000)
    assert plan.n_ops == 5 and plan.n_tasks == 200 and not plan.chain_violations()
    assert plan.tensors["empty_a"].shape == (40, EMPTY_WIDTH) and plan.tensors["empty_a"].kind == "new"
    assert [c.args["input"] for c in plan.calls] == ["empty_a", "empty_b", "empty_a", "empty_b", "empty_a"]
    assert plan.calls[0].args["spin_print"] == 1 and all(c.args["spin_print"] == 0 for c in plan.calls[1:])
    _, calls = B.dry_run(plan=plan)
    rec = [c for c in calls if c["method"] == "copy_mi300@new"]
    assert len(rec) == 5 and rec[0]["imaps"] == [[-1, -1, -1], [0, -1, -1]]     # whole in, a row out
    assert rec[0]["params"] == [EMPTY_WIDTH, 1000, 1] and rec[1]["params"] == [EMPTY_WIDTH, 1000, 0]
    # without spin the registration keeps its one parameter; a single task keeps the whole-tensor imaps
    _, calls = B.dry_run(plan=build_empty_plan(ops=3, tasks=1))
    rec = [c for c in calls if c["method"] == "copy_mi300@new"]
    assert rec[0]["params"] == [EMPTY_WIDTH] and rec[0]["imaps"] == [[-1, -1, -1], [-1, -1, -1]]
    # the model's copy operators are unchanged (a [1, H] snapshot, one parameter)
    _, calls = B.dry_run(layers=2, head=False, debug=True)
    rec = [c for c in calls if c["method"] == "copy_mi300@new"]
    assert rec and rec[0]["params"] == [REAL_DIMS.H] and rec[0]["imaps"] == [[-1, -1, -1], [-1, -1, -1]]


def test_stream_probe_plan_counts_tensors_and_chain():
    """L6 (M7 of docs/gpu-experiments/04-kernels/01-gemv-ideas.md): M operators of N tasks reading
    kb kilobytes each over two weight tensors and two dummies, the empty ladder's alternation;
    every operator reads the dummy the one before it wrote, which is what the runtime's chain rule
    needs from an operator whose only output is that dummy."""
    from fleet.graph_plan import build_stream_plan, stream_rows, STREAM_K
    plan = build_stream_plan(ops=10, tasks=96, kb=152)          # qkva's shape: 38 rows per task
    rows = stream_rows(152)
    assert rows == 38 and plan.n_ops == 10 and plan.n_tasks == 960 and not plan.chain_violations()
    assert plan.tensors["stream_w_a"].shape == (96 * rows, STREAM_K)
    assert plan.tensors["stream_w_a"].kind == "new" and plan.tensors["stream_dummy_a"].dtype == "i32"
    assert plan.tensors["stream_dummy_a"].shape == (96, 4)
    assert [c.args["weight"] for c in plan.calls[:3]] == ["stream_w_a", "stream_w_b", "stream_w_a"]
    assert [c.args["dummy"] for c in plan.calls[:3]] == ["stream_dummy_a", "stream_dummy_b", "stream_dummy_a"]
    assert [c.args["prev"] for c in plan.calls[:3]] == ["stream_dummy_b", "stream_dummy_a", "stream_dummy_b"]
    assert [c.label for c in plan.calls[:2]] == ["S0.stream", "S1.stream"]
    _, calls = B.dry_run(plan=plan)
    rec = [c for c in calls if c["task_type"] == "stream_mi300"]
    assert len(rec) == 10 and rec[0]["params"] == [] and rec[0]["grid_dim"] == (96, 1, 1)
    # the weight by the grid, the chain's dummy whole and ignored, the task's dummy row out
    assert rec[0]["imaps"] == [[0, -1, -1], [-1, -1, -1], [0, -1, -1]]
    assert rec[0]["inputs"] == ["stream_w_a", "stream_dummy_b", "stream_dummy_a"]
    # the 296-task row of G5, one task per CU at 256 KB
    plan = build_stream_plan(ops=10, tasks=296, kb=256)
    assert plan.n_ops == 10 and plan.n_tasks == 2960 and not plan.chain_violations()
    assert plan.tensors["stream_w_a"].shape == (296 * 64, STREAM_K)


def test_stream_probe_gang_plan_is_eight_tasks_of_tiles_per_xcd():
    """The gang row of G5: 8 XCD slots x 37 tiles, each tile reading 304 KB of one whole tensor
    (w13's shape at 8 active experts), the dummy whole with a row per tile."""
    from fleet.graph_plan import build_stream_plan, stream_rows, STREAM_K
    plan = build_stream_plan(ops=10, tasks=37, kb=304, gang=True)
    rows = stream_rows(304)
    assert rows == 76 and plan.n_ops == 10 and plan.n_tasks == 80 and not plan.chain_violations()
    assert all(c.tasks == 8 and c.tiles == 37 for c in plan.calls)
    assert plan.tensors["stream_w_a"].shape == (8 * 37 * rows, STREAM_K)
    assert plan.tensors["stream_dummy_a"].shape == (8 * 37, 4)
    assert 8 * 37 * 304 * 1024 == 8 * 37 * rows * STREAM_K * 2        # 90 MiB, w13's eight experts
    _, calls = B.dry_run(plan=plan)
    rec = [c for c in calls if c["task_type"] == "stream_gang_mi300"]
    assert len(rec) == 10 and rec[0]["params"] == [rows, 37] and rec[0]["grid_dim"] == (8, 1, 1)
    assert rec[0]["imaps"] == [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]


def test_stream_probe_kb_must_be_whole_rows():
    """A task reads whole 4 KB rows, so a --kb that is not a multiple of 4 is refused rather than
    rounded: the byte count is the numerator of the rate the probe reports. w13's 305 KB of the
    pages is run at 304."""
    from fleet.graph_plan import build_stream_plan, stream_rows
    with pytest.raises(AssertionError) as e:
        build_stream_plan(ops=2, tasks=4, kb=305, gang=True)
    assert "304" in str(e.value) and "308" in str(e.value)
    assert stream_rows(4) == 1 and stream_rows(256) == 64


def test_probe_before_inserts_a_one_task_copy_and_rewires_the_consumer():
    """O5 (docs/gpu-experiments/03-acceleration): a copy of the chain's tensor in front of the
    named operator, which then reads the twin; one more operator and task, the chain intact."""
    base, _ = B.dry_run(layers=2, head=True, tile_linears=True)
    plan, calls = B.dry_run(layers=2, head=True, tile_linears=True, probe_before="L0.o_proj")
    assert plan.n_ops == base.n_ops + 1 and plan.n_tasks == base.n_tasks + 1 and not plan.chain_violations()
    i = plan.index_of("L0.probe_o_proj")
    probe, o_proj = plan.calls[i], plan.calls[i + 1]
    assert plan.calls[i - 1].label == "L0.mla_merge_uv" and o_proj.label == "L0.o_proj"
    assert probe.method == "copy_layer" and probe.tasks == 1
    assert probe.args["input"] == "attn" and probe.args["output"] == "attn_probe"
    assert plan.tensors["attn_probe"].shape == plan.tensors["attn"].shape == (1, 2048)
    assert o_proj.args["input"] == "attn_probe" and o_proj.args["residual"] == "x_res" and o_proj.args["output"] == "x_res"
    rec = [c for c in calls if c["method"] == "copy_mi300@new"]
    assert len(rec) == 1 and rec[0]["inputs"] == ["attn", "attn_probe"] and rec[0]["params"] == [2048]
    # the probe composes with --stop-after (applied first) and refuses a tensor the copy cannot take
    cut, _ = B.dry_run(layers=2, head=True, tile_linears=True, probe_before="L0.o_proj", stop_after="L0.o_proj")
    assert [c.label for c in cut.calls][-2:] == ["L0.probe_o_proj", "L0.o_proj"]
    with pytest.raises(AssertionError):
        B.dry_run(layers=2, head=True, probe_before="L1.combine")      # out8 is [1, 8, 2048]
