#!/usr/bin/env python3
"""Measurement: per-iteration latency, per-operator time, traffic, launches (L11).

Parses what a Fleet run leaves behind (harness/run_fleet.py writes them
into its output directory) and, when rocprofv3 outputs are given, the
counters; writes metrics.json and report_table.md in the shape of
docs/design-doc/09-expected-performance.md ("The report table").

Inputs (all optional; what is present is reported):
  fwd_pass.log          stdout of the run with the [FWD_PASS] lines (every iteration after the patch)
  event_timing.json     {"entries": [[event_idx, timestamp_100MHz], ...], "num_events": N}
                        written by PersistentKernel when MPK_EVENT_TIMING=1
  plan.json             the operator list of the run (fleet/build_graph.py dry-run format)
  rocprof kernel trace  --kernel-trace CSV: dispatches per generation
  rocprof pmc CSV       --pmc CSV with TCC_BUBBLE_sum, TCC_EA0_RDREQ_sum, TCC_EA0_RDREQ_32B_sum,
                        TCC_EA0_WRREQ_sum, TCC_EA0_WRREQ_64B_sum, TCC_HIT_sum, TCC_MISS_sum
                        (two counters per rocprofv3 run; concatenate or pass the run directory)
  wall.json             {"mpk_wall_s": ..., "iters": K} from run_fleet.py

    python harness/measure.py --run harness/fleet_out [--kernel-trace k.csv] [--pmc p.csv]
"""
import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

FWD_RE = re.compile(r"\[FWD_PASS\] iter=(\d+) time_ms=([0-9.]+) num_active_tokens=(\d+)")
TICK_US = 0.01                  # the event-timing clock is 100 MHz (persistent_kernel.py:_read_event_timing)

PREDICTED = {
    "bytes_per_iteration_MiB": {"weights_and_cache": 4709.9, "with_activations_and_partials": 4772.0},
    "time_per_iteration_us": {"bandwidth_floor_at_5.3TBs": 931.8, "band_at_4.3TBs": 1148.5,
                              "band_at_3.66TBs": 1349.4, "boundaries": 326, "t_b_us": "unknown"},
    "layer1_MiB": 159.63, "layer1_us_band": [38.9, 45.7], "layer1_boundaries": 12,
    "launches_per_generation": 3,
    "release_flushes_per_iteration": 1838,
}


# ----------------------------------------------------------------------------
# parsers


def parse_fwd_pass(text: str):
    """[(iter, time_ms, num_active_tokens)] in file order."""
    return [(int(a), float(b), int(c)) for a, b, c in FWD_RE.findall(text)]


def percentiles(values, ps=(50, 95)):
    if not values:
        return {f"p{p}": None for p in ps}
    xs = sorted(values)
    out = {}
    for p in ps:
        k = (len(xs) - 1) * p / 100.0
        lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
        out[f"p{p}"] = xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
    out["mean"] = statistics.fmean(xs)
    out["min"], out["max"], out["n"] = xs[0], xs[-1], len(xs)
    return out


def event_iterations(entries, end_event_idx):
    """Per-iteration times from consecutive firings of the end-of-graph event, in us."""
    ts = sorted(t for e, t in entries if e == end_event_idx)
    return [(ts[i] - ts[i - 1]) * TICK_US for i in range(1, len(ts))]


def event_per_op(entries, num_events, op_names=None, skip_first_iteration=True):
    """Gap between consecutive event firings, attributed to the event that fired
    (the operator whose completion it marks); averaged over iterations."""
    entries = sorted(entries, key=lambda x: x[1])
    per_event = {}
    for i in range(1, len(entries)):
        e, t = entries[i]
        gap = (t - entries[i - 1][1]) * TICK_US
        if 0 <= gap < 1e5:
            per_event.setdefault(e, []).append(gap)
    rows = []
    for e in sorted(per_event):
        d = per_event[e][1:] if skip_first_iteration and len(per_event[e]) > 1 else per_event[e]
        name = op_names[e] if op_names and e < len(op_names) else f"event_{e}"
        rows.append({"event": e, "op": name, "mean_us": statistics.fmean(d), "min_us": min(d),
                     "max_us": max(d), "n": len(d)})
    return rows


# The megakernel's dispatches: three per generation on MI300 (persistent_kernel.cuh,
# launch_persistent_kernel: prepare_kernel, worker_kernel, scheduler_kernel; persistent_kernel on
# the single-launch path). init_kernel runs once at compile time. A profiled run_fleet.py run
# also dispatches the weight packing, the cache copies and rocclr's fill and copy blits, so the
# counters and the launch count are filtered to these names (docs/gpu-experiments/02-validation/01-preparation.md, P7).
DEFAULT_KERNEL_FILTER = "prepare_kernel,worker_kernel,scheduler_kernel,persistent_kernel"


def kernel_matches(name, kernel_filter):
    """True if any comma-separated pattern of kernel_filter is a substring of name; an empty
    filter matches everything."""
    if not kernel_filter:
        return True
    return any(pat and pat in (name or "") for pat in kernel_filter.split(","))


def _name_key(rows):
    return next((k for k in rows[0] if k and k.lower() in ("kernel_name", "name")), None) if rows else None


def parse_kernel_trace(path, kernel_filter=DEFAULT_KERNEL_FILTER):
    """rocprofv3 --kernel-trace CSV: every dispatch by name, the megakernel's dispatches
    (kernel_filter), and their wall time from the trace's own timestamps (ns) when present."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    name_key = _name_key(rows)
    counts = {}
    for r in rows:
        n = r[name_key] if name_key else ""
        counts[n] = counts.get(n, 0) + 1
    mega = [r for r in rows if name_key and kernel_matches(r[name_key], kernel_filter)]
    out = {"dispatches": len(rows), "by_kernel": counts, "kernel_filter": kernel_filter,
           "megakernel_dispatches": len(mega)}
    if mega and "Start_Timestamp" in mega[0] and "End_Timestamp" in mega[0]:
        durations = [(int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3 for r in mega]
        out["megakernel_us"] = {r[name_key]: round(d, 1) for r, d in zip(mega, durations)}
        out["megakernel_total_us"] = round(sum(durations), 1)
    return out


def _parse_pmc_file(path, kernel_filter):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    name_key = _name_key(rows)
    if name_key:
        rows = [r for r in rows if kernel_matches(r[name_key], kernel_filter)]
    # two layouts seen: one column per counter, or (Counter_Name, Counter_Value) pairs
    if rows and "Counter_Name" in rows[0] and "Counter_Value" in rows[0]:
        sums = {}
        for r in rows:
            sums[r["Counter_Name"]] = sums.get(r["Counter_Name"], 0.0) + float(r["Counter_Value"])
        return sums
    sums = {}
    for k in (rows[0] if rows else {}):
        if k and (k.startswith("TCC_") or k.startswith("SQ_")):
            sums[k] = sum(float(r[k] or 0) for r in rows)
    return sums


def pmc_files(spec):
    """The counter CSVs named by spec: a file, a comma-separated list, or a directory searched
    recursively for rocprofv3's *counter_collection.csv, in sorted order."""
    if isinstance(spec, (list, tuple)):
        out = []
        for x in spec:
            out += pmc_files(x)
        return out
    spec = str(spec)
    if "," in spec:
        return pmc_files(spec.split(","))
    p = Path(spec)
    if p.is_dir():
        return sorted(str(f) for f in p.rglob("*counter_collection.csv"))
    return [spec]


def parse_pmc(spec, kernel_filter=DEFAULT_KERNEL_FILTER):
    """Counter sums over the dispatches whose Kernel_Name matches kernel_filter (all rows when
    a file has no Kernel_Name column or the filter is empty), from one or several rocprofv3
    --pmc runs (see pmc_files). Each run holds one counter pair; a counter present in several
    runs takes the value of the later file, never a sum across runs, so the last run should be
    the TCC_BUBBLE_sum + TCC_EA0_RDREQ_sum pair that the read formula subtracts as one
    (env/hw/summarize.py does the same; env/session/queue.sh runs the pairs in that order)."""
    sums = {}
    for f in pmc_files(spec):
        sums.update(_parse_pmc_file(f, kernel_filter))
    return sums


def bytes_read(c):
    """Bytes fetched, by rocprofv3's own FETCH_SIZE expression on MI300.

    A read is a 128-byte request counted by TCC_BUBBLE; what is left of
    TCC_EA0_RDREQ after the 128 B and the 32 B requests are removed is 64 B
    each. The flat 64 B per request this file used until the day-1 collection
    reported half the real traffic: measured on the VM against a copy of known
    size, TCC_BUBBLE 8,388,608 and TCC_EA0_RDREQ 8,388,760 are 1.0000 GiB,
    while 64 x TCC_EA0_RDREQ is 0.5000 GiB (env/hw/20260915/summary.md, I3).
    """
    bubble, rdreq = c.get("TCC_BUBBLE_sum"), c.get("TCC_EA0_RDREQ_sum")
    rd32 = c.get("TCC_EA0_RDREQ_32B_sum")
    if bubble is None or rdreq is None or rd32 is None:
        return None
    return 128 * bubble + 64 * (rdreq - bubble - rd32) + 32 * rd32


def bytes_written(c):
    """Bytes written: 64 B per 64-byte request, 32 B for the rest."""
    wrreq, wr64 = c.get("TCC_EA0_WRREQ_sum"), c.get("TCC_EA0_WRREQ_64B_sum")
    if wrreq is None or wr64 is None:
        return None
    return 64 * wr64 + 32 * (wrreq - wr64)


def traffic_from_counters(c, iters):
    hit, miss = c.get("TCC_HIT_sum"), c.get("TCC_MISS_sum")
    out = {}
    reads, writes = bytes_read(c), bytes_written(c)
    if reads is not None:
        out["read_MiB_per_iteration"] = reads / 2**20 / iters
    if writes is not None:
        out["write_MiB_per_iteration"] = writes / 2**20 / iters
    if hit is not None and miss is not None and hit + miss > 0:
        out["l2_hit_rate"] = hit / (hit + miss)
    return out


# ----------------------------------------------------------------------------


def measure(run_dir: Path, kernel_trace=None, pmc=None, kernel_filter=DEFAULT_KERNEL_FILTER):
    run_dir = Path(run_dir)
    m = {"run_dir": str(run_dir), "predicted": PREDICTED, "kernel_filter": kernel_filter}
    wall = json.loads((run_dir / "wall.json").read_text()) if (run_dir / "wall.json").exists() else {}
    iters = wall.get("iters") or common.N_STEPS
    m["wall"] = wall

    fp = run_dir / "fwd_pass.log"
    if fp.exists():
        passes = parse_fwd_pass(fp.read_text())
        times_us = [t * 1000.0 for _, t, _ in passes]
        m["fwd_pass"] = {"iterations_logged": len(passes), "per_iteration_us": percentiles(times_us),
                         "first_five_us": times_us[:5]}
    et = run_dir / "event_timing.json"
    plan = json.loads((run_dir / "plan.json").read_text()) if (run_dir / "plan.json").exists() else None
    if et.exists():
        data = json.loads(et.read_text())
        entries = [(int(e), int(t)) for e, t in data["entries"]]
        num_events = int(data.get("num_events", 0))
        op_names = None
        if plan:
            # event index i marks the completion of operator i - 1 (event 0 is the begin-graph event);
            # verify on the machine against task_graph.json (docs/design-doc/03-synchronization.md)
            op_names = ["begin"] + [c["method"] for c in plan["calls"] if not c.get("side")]   # side operators (O8) add no event
        end_idx = num_events - 1 if num_events else max(e for e, _ in entries)
        iter_us = event_iterations(entries, end_idx)
        m["event_timing"] = {"num_events": num_events, "firings": len(entries),
                             "per_iteration_us": percentiles(iter_us),
                             "per_op": event_per_op(entries, num_events, op_names)}
    if kernel_trace:
        m["launches"] = parse_kernel_trace(kernel_trace, kernel_filter)
        if m["launches"].get("megakernel_total_us") and iters:
            m["launches"]["per_iteration_us_from_trace"] = m["launches"]["megakernel_total_us"] / iters
    if pmc:
        counters = parse_pmc(pmc, kernel_filter)
        m["counters"] = counters
        m["traffic"] = traffic_from_counters(counters, iters)
        per_it = m.get("fwd_pass", {}).get("per_iteration_us", {}).get("p50")
        if per_it and "read_MiB_per_iteration" in m["traffic"]:
            m["traffic"]["achieved_read_TBs"] = m["traffic"]["read_MiB_per_iteration"] * 2**20 / (per_it * 1e-6) / 1e12
    if wall.get("mpk_wall_s") and iters:
        m["wall"]["per_iteration_us_from_wall"] = wall["mpk_wall_s"] / iters * 1e6
    return m


def report_table(m):
    def g(*keys):
        d = m
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                return None
            d = d[k]
        return d

    def f(x, nd=1):
        return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))

    p = m["predicted"]
    rows = [
        ("bytes per iteration (read, MiB)", f"{p['bytes_per_iteration_MiB']['weights_and_cache']} weights + cache; "
         f"about {p['bytes_per_iteration_MiB']['with_activations_and_partials']} with activations",
         f(g("traffic", "read_MiB_per_iteration"))),
        ("time per iteration, median (us)", f"{p['time_per_iteration_us']['band_at_4.3TBs']}-"
         f"{p['time_per_iteration_us']['band_at_3.66TBs']} + 326 t_b + T_serial",
         f(g("fwd_pass", "per_iteration_us", "p50"))),
        ("time per iteration, P95 (us)", "", f(g("fwd_pass", "per_iteration_us", "p95"))),
        ("time per iteration from event timing, median (us)", "", f(g("event_timing", "per_iteration_us", "p50"))),
        ("time per iteration from host wall clock (us)", "", f(g("wall", "per_iteration_us_from_wall"))),
        ("time per iteration from the kernel trace (us)", "", f(g("launches", "per_iteration_us_from_trace"))),
        ("achieved read bandwidth (TB/s)", "3.66-4.3 over T_bw", f(g("traffic", "achieved_read_TBs"), 2)),
        ("launches per generation", str(p["launches_per_generation"]),
         "-" if g("launches") is None else
         f"{g('launches', 'megakernel_dispatches')} megakernel of {g('launches', 'dispatches')} dispatches in the run"),
        ("L2 hit rate", "16-17% (Fleet's batch-1 figure)", f(g("traffic", "l2_hit_rate"), 3)),
        ("tokens per second", "", f(1e6 / g("fwd_pass", "per_iteration_us", "p50")) if g("fwd_pass", "per_iteration_us", "p50") else "-"),
    ]
    lines = ["# Measurement report", "", "| Quantity | Predicted | Measured |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    ops = g("event_timing", "per_op") or []
    if ops:
        lines += ["", "## Per-operator time (event gaps, mean over iterations after the first)", "",
                  "| Event | Operator | mean us | min us | max us | n |", "|---|---|---|---|---|---|"]
        lines += [f"| {r['event']} | {r['op']} | {r['mean_us']:.2f} | {r['min_us']:.2f} | {r['max_us']:.2f} | {r['n']} |"
                  for r in ops]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run_fleet.py output directory")
    ap.add_argument("--kernel-trace", default=None,
                    help="rocprofv3 --kernel-trace CSV")
    ap.add_argument("--kernel-filter", default=DEFAULT_KERNEL_FILTER,
                    help="comma-separated kernel-name substrings the counters and launch count are "
                         "restricted to; '' sums every dispatch of the profiled process")
    ap.add_argument("--pmc", default=None,
                    help="rocprofv3 --pmc CSV, a comma-separated list of them, or a directory of "
                         "runs (searched for *counter_collection.csv). Traffic needs TCC_BUBBLE_sum, "
                         "TCC_EA0_RDREQ_sum and TCC_EA0_RDREQ_32B_sum for reads "
                         "and TCC_EA0_WRREQ_sum with TCC_EA0_WRREQ_64B_sum for "
                         "writes; a read is a 128 B request counted by TCC_BUBBLE")
    args = ap.parse_args()
    m = measure(Path(args.run), args.kernel_trace, args.pmc, args.kernel_filter)
    out = Path(args.run)
    (out / "metrics.json").write_text(json.dumps(m, indent=2) + "\n")
    (out / "report_table.md").write_text(report_table(m))
    print(report_table(m))
    print(f"-> {out / 'metrics.json'}, {out / 'report_table.md'}")


if __name__ == "__main__":
    main()
