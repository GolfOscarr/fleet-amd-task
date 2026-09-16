"""run_fleet.py's boundary mapping on a dry-run plan with synthetic buffers, and
measure.py's parsers on synthetic inputs."""
import json
import sys
from pathlib import Path

import pytest
import torch

HARNESS = Path(__file__).resolve().parent.parent
ROOT = HARNESS.parent
sys.path.insert(0, str(HARNESS))
sys.path.insert(0, str(ROOT))
import common  # noqa: E402
import measure  # noqa: E402
import run_fleet  # noqa: E402
from fleet import build_graph as B  # noqa: E402
from fleet.pack_weights import REAL_DIMS  # noqa: E402


def host_buffers(plan_json, fill):
    """One CPU tensor per plan tensor, filled with a recognisable value per name."""
    h = {}
    dt = {"bf16": torch.bfloat16, "f32": torch.float32, "i32": torch.int32, "i64": torch.int64}
    for i, (name, t) in enumerate(plan_json["tensors"].items()):
        h[name] = torch.full(t["shape"], fill(i), dtype=dt[t["dtype"]]) if t["dtype"] != "i64" else \
            torch.full(t["shape"], i, dtype=torch.int64)
    return h


def dump(layers, head, stop_after=None, iters=1, debug=False, debug_scores=False):
    plan, _ = B.dry_run(REAL_DIMS, 1024 + iters, layers, head, debug, stop_after, debug_scores)
    pj = B.plan_json(plan)
    h = host_buffers(pj, lambda i: float(i % 7 + 1))
    h["mask"] = torch.tensor([5, 2, 9, 1, 40, 63, 64, 65, 8] + [-1] * 58, dtype=torch.int32)
    tokens = torch.zeros(1, 1024 + iters, dtype=torch.int64)
    tokens[0, 1024] = 4242
    b, notes = run_fleet.boundary_dump(pj, h, tokens, 1024, {"Q_OUT": REAL_DIMS.Q_OUT, "TOPK": REAL_DIMS.TOPK})
    return b, notes, h


def test_dump_after_o_proj_of_layer1():
    b, _, h = dump(layers=2, head=False, stop_after="L1.o_proj")
    keys = {k for k in b if ".B" in k}
    assert keys == {"L1.B1.norm1", "L1.B2.q", "L1.B4.q_pe", "L1.B6.attn", "L1.B7.x_res_attn",
                    "L0.B3.c_kv", "L0.B3.k_pe", "L1.B3.c_kv", "L1.B3.k_pe"}
    assert b["L1.B2.q"].shape == (3072,) and b["L1.B7.x_res_attn"].shape == (2048,)
    assert torch.equal(b["L1.B3.c_kv"], h["c_kv_1"][1023])
    assert all(common.boundary_class(k) for k in keys)


def test_dump_full_layer1_and_head():
    b, notes, h = dump(layers=2, head=True)
    keys = {k for k in b if ".B" in k}
    assert {"L1.B8.router_logits", "L1.B9.topk_idx", "L1.B10.topk_w", "L1.B12.shared",
            "L1.B13.layer_out", "L1.B2.q", "L1.B6.attn", "head.B14.norm", "head.B15.logits",
            "head.B16.token"} <= keys
    assert "L1.B7.x_res_attn" not in keys and "L1.B1.norm1" not in keys      # overwritten in place
    assert b["L1.B9.topk_idx"].tolist() == [5, 2, 9, 1, 40, 63]
    assert {k for k in keys if ".B11." in k} == {f"L1.B11.expert_{e}" for e in (5, 2, 9, 1, 40, 63)}
    assert torch.equal(b["L1.B12.shared"], h["out8"][0, 6].float() + h["out8"][0, 7].float())
    assert int(b["head.B16.token"]) == 4242
    assert not notes


def test_dump_debug_scores():
    b, _, h = dump(layers=2, head=False, stop_after="L1.mla_attend", debug_scores=True)
    assert b["L1.B5.scores"].shape == (16, 1024) and torch.equal(b["L1.B5.scores"], h["scores"][:, :1024])
    b, _, _ = dump(layers=2, head=False, stop_after="L1.mla_attend")
    assert "L1.B5.scores" not in b


def test_dump_layer0_dense_and_debug():
    b, notes, _ = dump(layers=1, head=False, debug=True)
    assert "L0.B13.layer_out" in b and "_hidden" in b and b["_hidden"].shape == (1, 2048)
    plan, _ = B.dry_run(REAL_DIMS, 1026, 3, False)
    pj = B.plan_json(plan)
    h = host_buffers(pj, lambda i: 1.0)
    _, notes = run_fleet.boundary_dump(pj, h, torch.zeros(1, 1026, dtype=torch.int64), 1024,
                                       {"Q_OUT": REAL_DIMS.Q_OUT, "TOPK": REAL_DIMS.TOPK}, iters=2)
    assert notes and "iteration 1" in notes[0]


def test_fwd_pass_and_percentiles():
    text = "junk\n[FWD_PASS] iter=1 time_ms=1.500 num_active_tokens=1\n[FWD_PASS] iter=2 time_ms=1.700 num_active_tokens=1\n" \
           "[FWD_PASS] iter=3 time_ms=1.600 num_active_tokens=1 compute_after_dispatch_us=3.0\n"
    p = measure.parse_fwd_pass(text)
    assert p == [(1, 1.5, 1), (2, 1.7, 1), (3, 1.6, 1)]
    st = measure.percentiles([1500.0, 1700.0, 1600.0])
    assert st["p50"] == 1600.0 and st["n"] == 3 and st["min"] == 1500.0 and abs(st["p95"] - 1690.0) < 1e-9


def test_event_timing_iterations_and_ops():
    # 3 events per iteration (begin, op A, end), 4 iterations, 100 MHz ticks
    entries, t = [], 0
    for it in range(4):
        for e, gap in ((0, 50), (1, 200), (2, 300)):
            t += gap
            entries.append((e, t))
    iters = measure.event_iterations(entries, end_event_idx=2)
    assert len(iters) == 3 and all(abs(x - 5.5) < 1e-9 for x in iters)     # 550 ticks = 5.5 us
    ops = measure.event_per_op(entries, 3, ["begin", "opA", "end"])
    by = {r["op"]: r for r in ops}
    assert abs(by["opA"]["mean_us"] - 2.0) < 1e-9 and abs(by["end"]["mean_us"] - 3.0) < 1e-9


def test_pmc_and_trace_parsers(tmp_path):
    pmc = tmp_path / "pmc.csv"
    pmc.write_text("Dispatch_Id,Kernel_Name,TCC_BUBBLE_sum,TCC_EA0_RDREQ_sum,"
                   "TCC_EA0_RDREQ_32B_sum,TCC_EA0_WRREQ_sum,TCC_EA0_WRREQ_64B_sum,"
                   "TCC_HIT_sum,TCC_MISS_sum\n"
                   "1,worker,1000,1000,0,10,10,30,70\n2,scheduler,24,24,0,2,2,1,1\n")
    c = measure.parse_pmc(pmc)
    assert c["TCC_EA0_RDREQ_sum"] == 1024 and c["TCC_HIT_sum"] == 31
    tr = measure.traffic_from_counters(c, iters=32)
    # Every read request is a 128 B TCC_BUBBLE one here, so reads are 128 x 1024.
    assert abs(tr["read_MiB_per_iteration"] - 1024 * 128 / 2**20 / 32) < 1e-12
    assert abs(tr["write_MiB_per_iteration"] - 12 * 64 / 2**20 / 32) < 1e-12
    assert abs(tr["l2_hit_rate"] - 31 / 102) < 1e-12
    pmc2 = tmp_path / "pmc2.csv"
    pmc2.write_text("Counter_Name,Counter_Value\nTCC_EA0_RDREQ_sum,5\nTCC_EA0_RDREQ_sum,7\n")
    assert measure.parse_pmc(pmc2)["TCC_EA0_RDREQ_sum"] == 12
    kt = tmp_path / "kt.csv"
    kt.write_text("Kernel_Name,Start\nprepare_kernel,1\nworker_kernel,2\nscheduler_kernel,3\n")
    assert measure.parse_kernel_trace(kt) == {"dispatches": 3, "by_kernel": {"prepare_kernel": 1, "worker_kernel": 1,
                                                                            "scheduler_kernel": 1}}


def test_bytes_from_requests_matches_the_measured_copy():
    """The counters a 1 GiB device-to-device copy produced on the VM
    (env/hw/20260915, group I3) must come back as 1 GiB each way.

    A read on MI300 is a 128-byte request counted by TCC_BUBBLE. The flat 64 B
    per request this file used before the collection reports half the reads."""
    counters = {
        "TCC_BUBBLE_sum": 8388608.0,
        "TCC_EA0_RDREQ_sum": 8388760.0,
        "TCC_EA0_RDREQ_32B_sum": 0.0,
        "TCC_EA0_WRREQ_sum": 16777216.0,
        "TCC_EA0_WRREQ_64B_sum": 16777216.0,
    }
    gib = float(1 << 30)
    assert measure.bytes_written(counters) == gib
    assert abs(measure.bytes_read(counters) - gib) / gib < 0.0001
    assert abs(64 * counters["TCC_EA0_RDREQ_sum"] - gib / 2) / gib < 0.0001

    # A 32 B request contributes 32 and a plain 64 B request 64.
    mixed = {"TCC_BUBBLE_sum": 1.0, "TCC_EA0_RDREQ_sum": 3.0,
             "TCC_EA0_RDREQ_32B_sum": 1.0,
             "TCC_EA0_WRREQ_sum": 3.0, "TCC_EA0_WRREQ_64B_sum": 1.0}
    assert measure.bytes_read(mixed) == 128 + 64 + 32
    assert measure.bytes_written(mixed) == 64 + 32 + 32

    # Without TCC_BUBBLE_sum the read side cannot be computed at all.
    assert measure.bytes_read({k: v for k, v in counters.items()
                               if k != "TCC_BUBBLE_sum"}) is None
    assert measure.traffic_from_counters({"TCC_HIT_sum": 1.0, "TCC_MISS_sum": 1.0},
                                         iters=1) == {"l2_hit_rate": 0.5}


def test_measure_end_to_end(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "fwd_pass.log").write_text("".join(f"[FWD_PASS] iter={i} time_ms={1.2 + 0.01 * i:.3f} num_active_tokens=1\n"
                                              for i in range(1, 33)))
    (run / "wall.json").write_text(json.dumps({"mpk_wall_s": 0.045, "iters": 32}))
    plan, _ = B.dry_run(REAL_DIMS, 1056, 1, False)
    (run / "plan.json").write_text(json.dumps(B.plan_json(plan)))
    entries, t = [], 0
    n_events = len(plan.calls) + 1
    for it in range(3):
        for e in range(n_events):
            t += 100
            entries.append([e, t])
    (run / "event_timing.json").write_text(json.dumps({"entries": entries, "num_events": n_events}))
    m = measure.measure(run)
    assert m["fwd_pass"]["iterations_logged"] == 32
    assert abs(m["wall"]["per_iteration_us_from_wall"] - 0.045 / 32 * 1e6) < 1e-6
    assert m["event_timing"]["per_iteration_us"]["n"] == 2
    assert m["event_timing"]["per_op"][1]["op"] == "embed_layer"
    md = measure.report_table(m)
    assert "| launches per generation | 3 | - |" in md and "embed_layer" in md


# ---- P1 of docs/round-2/01-preparation.md: the address-shift flag ----------------

def test_pad_alloc_argument_and_run_name():
    p = run_fleet.build_parser()
    a = p.parse_args(["--layers", "8", "--head", "--iters", "2", "--model-dir", "x", "--pad-alloc", "2"])
    assert a.pad_alloc == 2.0 and run_fleet.run_name(a) == "L8_head_it2_pad2"
    a = p.parse_args(["--layers", "8", "--model-dir", "x", "--pad-alloc", "0.5", "--stop-after", "L7.o_proj"])
    assert run_fleet.run_name(a) == "L8_it1_L7.o_proj_pad0.5"
    a = p.parse_args(["--layers", "27", "--head", "--iters", "32", "--model-dir", "x"])
    assert a.pad_alloc == 0.0 and run_fleet.run_name(a) == "L27_head_it32"     # unchanged without the flag


def test_tensor_addresses_records_every_host_tensor():
    b, _, h = dump(2, True)
    addr = run_fleet.tensor_addresses(h)
    assert set(addr) == set(h) and all(isinstance(v, int) for v in addr.values())
    assert addr["x_res"] == h["x_res"].data_ptr()


def test_fault_bisection_labels_are_in_plan_order():
    plan, _ = B.dry_run(REAL_DIMS, 1026, 8, True, False, None, False)
    labels = [c.label for c in plan.calls]
    want = [l.strip() for l in (ROOT / "env/session/queue-fault.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    assert all(w in labels for w in want)
    assert [l for l in labels if l in want] == want
    assert want[0] == "L7.norm1" and want[-1] == "head.argmax_reduce"
