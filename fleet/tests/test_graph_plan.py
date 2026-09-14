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


def test_input_bytes_and_shapes():
    plan, _ = B.dry_run()
    inputs = {n: t for n, t in plan.tensors.items() if t.kind == "input"}
    weights = sum(G._bytes(t) for n, t in inputs.items()
                  if not t.source.startswith(("capture:", "meta:")))
    from fleet.pack_weights import EXPECTED_PACKED_BYTES
    assert weights == EXPECTED_PACKED_BYTES
    assert plan.tensors["partials"].shape == (33, 16, 513)
    assert plan.tensors["c_kv_0"].shape == (1056, 512) and plan.tensors["k_pe_26"].shape == (1056, 64)
    assert plan.tensors["route_log"].shape == (32, 26, 8)


def test_constraints_are_enforced():
    bad = Dims(**{**REAL_DIMS.__dict__, "V": 102400 + 8})     # 102408 / 8 = 12801, not a multiple of 64
    with pytest.raises(AssertionError):
        B.dry_run(dims=bad)
    bad = Dims(**{**REAL_DIMS.__dict__, "I_MOE": 1400})
    with pytest.raises(AssertionError):
        B.dry_run(dims=bad)
