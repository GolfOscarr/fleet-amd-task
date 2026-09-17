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


def test_worker_timing_and_spin_parsers():
    """I1 and I2: the runtime's per-worker lines and the spin line, in the exact printf formats."""
    text = ("[WORKER_XCD] worker_id=0 block=0 xcd=0\n[WORKER_XCD] worker_id=1 block=1 xcd=1\n"
            "[WORKER_XCD] worker_id=2 block=2 xcd=2\n"
            "[SPIN] block=0 iters=1000 cycles=2100 ticks=100 x=123\n[SPIN] block=1 iters=1000 cycles=2000 ticks=100 x=5\n"
            "[FWD_PASS] iter=1 time_ms=1.000 num_active_tokens=1\n"
            "[TIMING] worker=0 tasks=10 poll_iters=5 dep_iters=7 poll_cycles=100 dep_cycles=4000 exec_cycles=21000 signal_cycles=300\n"
            "[TASK_TIME] worker=0 linear=0/0 linear_res=0/0 attn=0/0 rms=0/0 silu=0/0 fused=0/0\n"
            "[TASK_TIME2] worker=0 prep=0/0 attend=0/0 merge=0/0 router=0/0 copy=21000/10 w2silu=0/0 lnorm=0/0 prefetch=0/0\n"
            "[TIMING] worker=1 tasks=6 poll_iters=1 dep_iters=2 poll_cycles=10 dep_cycles=2000 exec_cycles=12600 signal_cycles=100\n"
            "[TASK_TIME] worker=1 linear=0/0 linear_res=0/0 attn=0/0 rms=0/0 silu=0/0 fused=0/0\n"
            "[TASK_TIME2] worker=1 prep=0/0 attend=0/0 merge=0/0 router=0/0 copy=12600/6 w2silu=0/0 lnorm=0/0 prefetch=0/0\n"
            "[TIMING] worker=2 tasks=0 poll_iters=900 dep_iters=0 poll_cycles=99999 dep_cycles=0 exec_cycles=0 signal_cycles=0\n")
    w = measure.parse_worker_timing(text)
    assert set(w) == {0, 1, 2} and w[0]["tasks"] == 10 and w[0]["xcd"] == 0 and w[1]["classes"]["copy"] == {"cycles": 12600, "count": 6}
    spins = measure.parse_spin(text)
    assert spins == [(0, 1000, 2100, 100), (1, 1000, 2000, 100)]
    mhz = measure.sclk_mhz(spins)
    assert mhz == 2100.0                                    # 2100 cycles over 100 ticks of 10 ns: the median of the two lines
    s = measure.worker_timing_summary(w, iters=2, mhz=mhz)
    assert s["workers_reporting"] == 3 and s["workers_with_tasks"] == 2 and s["tasks"] == 16
    assert s["tasks_per_xcd"] == {"0": 10, "1": 6}
    assert s["exec_cycles_per_task"] == 2100.0 and s["exec_us_per_task"] == 1.0
    assert s["per_class"]["copy"] == {"count": 16, "cycles_per_task": 2100.0, "us_per_task": 1.0}
    assert "attend" not in s["per_class"]                   # classes without tasks are left out
    assert abs(s["dep_wait_us_per_iteration_per_busy_worker"] - (6000 / 2 / 2) / 2100.0) < 1e-9
    # without a spin line the microseconds are absent, the cycles stay
    s0 = measure.worker_timing_summary(w, iters=2, mhz=None)
    assert s0["exec_us_per_task"] is None and s0["exec_cycles_per_task"] == 2100.0
    # the report renders the rows and the class table
    m = {"predicted": measure.PREDICTED, "worker_timing": dict(s, workers={})}
    rep = measure.report_table(m)
    assert "workers with tasks" in rep and "| copy | 16 | 2100 | 1.00 |" in rep


def test_clock_log_parser_and_summary():
    """I6: the queue's clock.log (blocks of a timestamp and the text of amd-smi metric --clock, the format
    recorded in round 1) -> per-sample GFX and memory clocks and their medians."""
    block = ("GPU: 0\n    CLOCK:\n        GFX_0:\n            CLK: {g0} MHz\n            MIN_CLK: 500 MHz\n"
             "            MAX_CLK: 2100 MHz\n        GFX_1:\n            CLK: {g1} MHz\n            MAX_CLK: 2100 MHz\n"
             "        MEM_0:\n            CLK: {m} MHz\n            MAX_CLK: 1300 MHz\n")
    text = ("### 2026-09-18T10:00:00Z\n" + block.format(g0=131, g1=133, m=900)
            + "### 2026-09-18T10:00:01Z\n" + block.format(g0=2100, g1=2050, m=1300)
            + "### 2026-09-18T10:00:02Z\n" + block.format(g0=1900, g1=2100, m=1300))
    s = measure.parse_clock_log(text)
    assert len(s) == 3 and s[0]["gfx"] == [131.0, 133.0] and s[0]["mem"] == 900.0 and s[1]["t"] == "2026-09-18T10:00:01Z"
    c = measure.clock_summary(s)
    # per sample the median over the XCDs (the upper of two), then the median and max over the samples
    assert c["samples"] == 3 and c["gfx_mhz_median"] == 2100.0 and c["gfx_mhz_max"] == 2100.0 and c["gfx_mhz_min"] == 133.0
    assert c["mem_mhz_median"] == 1300.0
    m = {"predicted": measure.PREDICTED, "clock": c}
    assert "GFX clock from amd-smi" in measure.report_table(m)
    assert measure.clock_summary(measure.parse_clock_log("")) == {"samples": 0}


def test_pmc_and_trace_parsers(tmp_path):
    pmc = tmp_path / "pmc.csv"
    pmc.write_text("Dispatch_Id,Kernel_Name,TCC_BUBBLE_sum,TCC_EA0_RDREQ_sum,"
                   "TCC_EA0_RDREQ_32B_sum,TCC_EA0_WRREQ_sum,TCC_EA0_WRREQ_64B_sum,"
                   "TCC_HIT_sum,TCC_MISS_sum\n"
                   "1,worker_kernel(mirage::runtime::RuntimeConfig),1000,1000,0,10,10,30,70\n"
                   "2,scheduler_kernel(mirage::runtime::RuntimeConfig),24,24,0,2,2,1,1\n"
                   "3,__amd_rocclr_copyBuffer,5000,5000,0,900,900,0,0\n")   # the packing blit: excluded
    c = measure.parse_pmc(pmc)
    assert c["TCC_EA0_RDREQ_sum"] == 1024 and c["TCC_HIT_sum"] == 31
    assert measure.parse_pmc(pmc, kernel_filter="")["TCC_EA0_RDREQ_sum"] == 6024   # no filter: everything
    tr = measure.traffic_from_counters(c, iters=32)
    # Every read request is a 128 B TCC_BUBBLE one here, so reads are 128 x 1024.
    assert abs(tr["read_MiB_per_iteration"] - 1024 * 128 / 2**20 / 32) < 1e-12
    assert abs(tr["write_MiB_per_iteration"] - 12 * 64 / 2**20 / 32) < 1e-12
    assert abs(tr["l2_hit_rate"] - 31 / 102) < 1e-12
    pmc2 = tmp_path / "pmc2.csv"
    pmc2.write_text("Counter_Name,Counter_Value\nTCC_EA0_RDREQ_sum,5\nTCC_EA0_RDREQ_sum,7\n")
    assert measure.parse_pmc(pmc2)["TCC_EA0_RDREQ_sum"] == 12
    kt = tmp_path / "kt.csv"
    kt.write_text("Kernel_Name,Start_Timestamp,End_Timestamp\n__amd_rocclr_copyBuffer,0,500\n"
                  "prepare_kernel(mirage::runtime::RuntimeConfig),1000,2000\n"
                  "worker_kernel(mirage::runtime::RuntimeConfig),2000,32002000\n"
                  "scheduler_kernel(mirage::runtime::RuntimeConfig),2000,32001000\n")
    lt = measure.parse_kernel_trace(kt)
    assert lt["dispatches"] == 4 and lt["megakernel_dispatches"] == 3
    assert lt["by_kernel"]["__amd_rocclr_copyBuffer"] == 1
    assert lt["megakernel_us"]["worker_kernel(mirage::runtime::RuntimeConfig)"] == 32000.0
    assert lt["megakernel_total_us"] == 1.0 + 32000.0 + 31999.0


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


def test_kernel_filter_on_the_recorded_counter_csv(tmp_path):
    """The 2026-09-15 record: the probe copied 1 GiB; filtered to copy_kernel the read formula
    gives exactly 1 GiB, and unfiltered it also counts the fill and warm-up dispatches."""
    raw = ROOT / "env/hw/20260915/raw"
    files = measure.pmc_files(raw)
    assert len(files) == 4 and files[-1].endswith("counter_collection.csv") and "/pmc4/" in files[-1]
    c = measure.parse_pmc(raw, kernel_filter="copy_kernel")           # the directory form
    assert abs(measure.bytes_read(c) - 2**30) / 2**30 < 1e-4 and abs(measure.bytes_written(c) - 2**30) / 2**30 < 1e-4
    assert measure.parse_pmc(",".join(files), kernel_filter="copy_kernel") == c   # the list form
    assert measure.bytes_read(measure.parse_pmc(raw, kernel_filter="")) > 2**30  # fill and warm-up too
    assert measure.parse_pmc(raw) == {}                                # no megakernel dispatch in a probe run
    # a concatenation of the four runs would double count the counter present in two of them
    merged = tmp_path / "pmc_all.csv"
    with merged.open("w") as out:
        for i, f in enumerate(files):
            lines = Path(f).read_text().splitlines(keepends=True)
            out.writelines(lines if i == 0 else lines[1:])
    assert measure.bytes_read(measure.parse_pmc(merged, kernel_filter="copy_kernel")) > 1.4 * 2**30


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
    # num_events is the runtime's buffer capacity (498 for every graph on the VM, 2026-09-17), not the
    # graph's count: the iteration marker must be the highest index that fires
    (run / "event_timing.json").write_text(json.dumps({"entries": entries, "num_events": 498}))
    m = measure.measure(run)
    assert m["fwd_pass"]["iterations_logged"] == 32
    assert abs(m["wall"]["per_iteration_us_from_wall"] - 0.045 / 32 * 1e6) < 1e-6
    assert m["event_timing"]["per_iteration_us"]["n"] == 2
    assert m["event_timing"]["per_op"][1]["op"] == "iteration_start"
    assert m["event_timing"]["per_op"][2]["op"] == "embed_layer"      # event i is the (i - 1)-th operator's completion
    md = measure.report_table(m)
    assert "| launches per generation | 3 | - |" in md and "embed_layer" in md
    kt = tmp_path / "kt.csv"
    kt.write_text("Kernel_Name,Start_Timestamp,End_Timestamp\n__amd_rocclr_copyBuffer,0,500\n"
                  "prepare_kernel(mirage::runtime::RuntimeConfig),1000,2000\n"
                  "worker_kernel(mirage::runtime::RuntimeConfig),2000,32002000\n"
                  "scheduler_kernel(mirage::runtime::RuntimeConfig),2000,32001000\n")
    m = measure.measure(run, kernel_trace=kt)
    assert abs(m["launches"]["per_iteration_us_from_trace"] - 64000.0 / 32) < 1e-9
    md = measure.report_table(m)
    assert "| launches per generation | 3 | 3 megakernel of 4 dispatches in the run |" in md
    assert "| time per iteration from the kernel trace (us) |  | 2000.0 |" in md


# ---- P1 of docs/gpu-experiments/02-validation/01-preparation.md: the address-shift flag ----------------

def test_pad_alloc_argument_and_run_name():
    p = run_fleet.build_parser()
    a = p.parse_args(["--layers", "8", "--head", "--iters", "2", "--model-dir", "x", "--pad-alloc", "2"])
    assert a.pad_alloc == 2.0 and run_fleet.run_name(a) == "L8_head_it2_pad2"
    a = p.parse_args(["--layers", "8", "--model-dir", "x", "--pad-alloc", "0.5", "--stop-after", "L7.o_proj"])
    assert run_fleet.run_name(a) == "L8_it1_L7.o_proj_pad0.5"
    a = p.parse_args(["--layers", "27", "--head", "--iters", "32", "--model-dir", "x"])
    assert a.pad_alloc == 0.0 and run_fleet.run_name(a) == "L27_head_it32"     # unchanged without the flag
    a = p.parse_args(["--layers", "2", "--iters", "32", "--tile-linears", "--model-dir", "x"])
    assert a.tile_linears and run_fleet.run_name(a) == "L2_it32_tile"
    # round 3: the fusions, the probe and the streaming loads each name the run (O1, O2, O5, O6)
    a = p.parse_args(["--layers", "27", "--head", "--iters", "32", "--tile-linears", "--fuse-norm2", "--fuse-silu",
                      "--nt-weights", "--nt-streams", "--model-dir", "x"])
    assert run_fleet.run_name(a) == "L27_head_it32_tile_fn2_fs_nt_nts"
    a = p.parse_args(["--layers", "2", "--iters", "32", "--probe-before", "L0.o_proj", "--model-dir", "x"])
    assert run_fleet.run_name(a) == "L2_it32_probe_L0.o_proj"
    a = p.parse_args(["--layers", "2", "--iters", "32", "--fuse-norm1", "--fuse-norm2", "--model-dir", "x"])
    assert a.fuse_norm1 and run_fleet.run_name(a) == "L2_it32_fn1_fn2"     # O3
    a = p.parse_args(["--layers", "2", "--iters", "32", "--nt-streams", "--mfma-attend", "--model-dir", "x"])
    assert a.mfma_attend and run_fleet.run_name(a) == "L2_it32_nts_mfma"   # O7
    a = p.parse_args(["--layers", "2", "--iters", "32", "--prefetch", "--probe-before", "L1.o_proj", "--model-dir", "x"])
    assert a.prefetch and run_fleet.run_name(a) == "L2_it32_pf_probe_L1.o_proj"   # O8
    a = p.parse_args(["--layers", "2", "--iters", "32", "--worker-timing", "--model-dir", "x"])
    assert a.worker_timing and run_fleet.run_name(a) == "L2_it32_wt"              # I1
    a = p.parse_args(["--graph", "empty", "--ops", "100", "--tasks", "40", "--iters", "32", "--event-timing",
                      "--worker-timing", "--spin", "1000", "--model-dir", "x"])
    assert run_fleet.run_name(a) == "E100x40_spin1000_it32_wt"                    # I3
    # the = form: a value starting with a dash is an option to argparse otherwise
    a = p.parse_args(["--layers", "2", "--iters", "32", "--runtime-flags=-DMPK_NO_COMPLETION_FENCE",
                      "--runtime-flags=-DMPK_POLL_SLEEP=8", "--model-dir", "x"])
    assert a.runtime_flags == ["-DMPK_NO_COMPLETION_FENCE", "-DMPK_POLL_SLEEP=8"]
    assert run_fleet.run_name(a) == "L2_it32_rf_nocompletionfence+pollsleep8"     # I4
    assert run_fleet.runtime_flags_slug(["-DMPK_NO_BCAST_CAS"]) == "nobcastcas" and run_fleet.runtime_flags_slug([]) == ""
    assert run_fleet.runtime_flags_slug(['"-DMPK_NO_BCAST_CAS"']) == "nobcastcas"   # a stray quote from a queue row


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


# ---- the candidate M4 fault fixes as flags (docs/gpu-experiments/02-validation, session A row A4) -----------

def test_aligned_copy_rebases_and_preserves_values():
    for align in (512, 4096, 65536):
        t = torch.arange(1000, dtype=torch.float32).reshape(10, 100) * 0.5
        a = B.aligned_copy(torch, t, align)
        assert a.data_ptr() % align == 0 and a.shape == t.shape and a.dtype == t.dtype
        assert torch.equal(a, t) and a.data_ptr() != t.data_ptr()
    b16 = torch.ones(3, 7, dtype=torch.bfloat16)
    a = B.aligned_copy(torch, b16, 4096)
    assert a.dtype == torch.bfloat16 and a.data_ptr() % 4096 == 0 and torch.equal(a, b16)
    with pytest.raises(AssertionError):
        B.aligned_copy(torch, t, 3000)


def test_single_row_workspaces_are_backed_by_sixteen_rows():
    # the M4 fault (session A, 2026-09-16): gang_linear_silu_kernel reads 16 rows of its
    # [1, D] input; the rows behind the returned row must belong to the same allocation
    plan, _ = B.dry_run(REAL_DIMS, 1026, 1, False)
    rows = [t for t in plan.tensors.values() if t.kind != "input" and len(t.shape) == 2 and t.shape[0] == 1]
    assert rows, "the plan has single-row activations"
    for t in rows:
        for align in (0, 65536):
            w = B.new_workspace(torch, t, align, device="cpu")
            assert tuple(w.shape) == t.shape
            backing = w.untyped_storage().nbytes() - (w.storage_offset() * w.element_size())
            assert backing >= B.ROW_SLACK * t.shape[1] * w.element_size(), t.name
            if align:
                assert w.data_ptr() % align == 0
    ws = B.allocate_workspaces(torch, plan, device="cpu")
    assert all(tuple(ws[t.name].shape) == t.shape for t in rows)


def test_allocate_workspaces_and_make_tensors_reuse_them():
    plan, _ = B.dry_run(REAL_DIMS, 1026, 1, False)
    ws = B.allocate_workspaces(torch, plan, align=4096, device="cpu")
    names = {t.name for t in plan.tensors.values() if t.kind != "input"}
    assert set(ws) == names and all(v.data_ptr() % 4096 == 0 for v in ws.values())
    assert all(tuple(ws[n].shape) == plan.tensors[n].shape for n in names)
    assert all(float(ws[n].float().abs().sum()) == 0.0 for n in names)
    # make_tensors attaches the given buffers instead of allocating
    dt = {"bf16": torch.bfloat16, "f32": torch.float32, "i32": torch.int32, "i64": torch.int64}
    packed, capture, meta = {}, {}, {}
    for t in plan.tensors.values():
        if t.kind != "input":
            continue
        buf = torch.zeros(t.shape, dtype=dt[t.dtype])
        if t.source.startswith("meta:"):
            meta[t.source[5:]] = buf
        elif t.source.startswith("capture:"):
            capture[t.source[8:]] = buf
        else:
            packed[t.source] = buf
    fake = B.FakeMPK()
    _, host = B.make_tensors(fake, plan, packed, capture, meta, torch, workspaces=ws)
    assert all(host[n] is ws[n] for n in names)


def test_fault_fix_flags_and_run_names():
    p = run_fleet.build_parser()
    a = p.parse_args(["--layers", "8", "--head", "--iters", "2", "--model-dir", "x", "--align-alloc", "65536"])
    assert a.align_alloc == 65536 and run_fleet.run_name(a) == "L8_head_it2_al65536"
    a = p.parse_args(["--layers", "8", "--head", "--iters", "2", "--model-dir", "x", "--workspaces-first"])
    assert a.workspaces_first and run_fleet.run_name(a) == "L8_head_it2_wsfirst"
    a = p.parse_args(["--layers", "8", "--head", "--iters", "2", "--model-dir", "x", "--align-alloc", "4096",
                      "--workspaces-first", "--pad-alloc", "1"])
    assert run_fleet.run_name(a) == "L8_head_it2_al4096_wsfirst_pad1"
