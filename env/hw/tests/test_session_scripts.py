"""env/session/{common,vm,queue,laptop}.sh against a fake run_fleet.py in a temporary tree
(docs/gpu-experiments/02-validation/01-preparation.md, P4): the queue's status rows, the record copies, the stop rule,
the bisection over the P1 label file, and the DRY modes of vm.sh and laptop.sh."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SESSION = ROOT / "env/session"
PY = sys.executable

FAKE_RUN_FLEET = r'''
import os, sys, json
from pathlib import Path
sys.path.insert(0, str(Path(os.environ["REAL_ROOT"]) / "harness"))
import run_fleet
args = run_fleet.build_parser().parse_args(sys.argv[1:])
name = run_fleet.run_name(args)
out = Path(args.out) if args.out else Path(os.environ["FLEET_OUT"]) / name
out.mkdir(parents=True, exist_ok=True)
(out / "plan.json").write_text("{}\n")
(out / "big.safetensors").write_bytes(b"0" * 10)
labels = [l.strip() for l in Path(os.environ["FAKE_LABELS"]).read_text().splitlines()
          if l.strip() and not l.startswith("#")]
fault_at = os.environ.get("FAKE_FAULT_AT")
faults = False
if fault_at and args.layers >= 7:
    faults = args.stop_after is None or labels.index(args.stop_after) >= labels.index(fault_at)
if faults:
    print("RuntimeError: HIP error: an illegal memory access was encountered")
    print("torch.AcceleratorError")
    sys.exit(1)
if args.stop_after and os.environ.get("FAKE_ERROR_AT") == args.stop_after:   # a JIT failure: FAIL without a fault line
    print("subprocess.CalledProcessError: Command '['/opt/rocm/bin/hipcc', ...]' returned non-zero exit status 1.")
    sys.exit(1)
for i in range(args.iters):
    print(f"[FWD_PASS] iter={i} time_ms=1.0 num_active_tokens=1")
(out / "fleet_run_meta.json").write_text(json.dumps({"layers": args.layers}))
(out / "wall.json").write_text(json.dumps({"mpk_wall_s": 0.001 * args.iters, "iters": args.iters}))
print(f"ids [25]; 3 boundary tensors; mpk() 12.0 ms for {args.iters} iterations -> {out}")
'''

FAKE_COMPARE = r'''
import sys
from pathlib import Path
d = Path(sys.argv[sys.argv.index("--fleet") + 1])
(d / "correctness_report.md").write_text("Overall: **PASS**\n")
print("Overall: **PASS**")
'''


@pytest.fixture
def tree(tmp_path):
    """A temporary tree with the fake tools; the scripts read env overrides."""
    (tmp_path / "run_fleet_fake.py").write_text(FAKE_RUN_FLEET)
    (tmp_path / "compare_fake.py").write_text(FAKE_COMPARE)
    env = dict(os.environ)
    env.update({
        "ROOT": str(ROOT), "REAL_ROOT": str(ROOT), "PY": PY,
        "RUN_FLEET": f"{PY} {tmp_path / 'run_fleet_fake.py'}",
        "COMPARE": f"{PY} {tmp_path / 'compare_fake.py'}",
        "FLEET_OUT": str(tmp_path / "fleet_out"), "RECORD": str(tmp_path / "record"),
        "LOGDIR": str(tmp_path / "logs"), "SNAP": str(tmp_path / "snap"),
        "FAKE_LABELS": str(SESSION / "queue-fault.txt"),
        "REF_DIR": str(tmp_path / "ref"),
    })
    (tmp_path / "ref").mkdir()
    (tmp_path / "ref/ref_cache.safetensors").write_bytes(b"0")      # the reference stage has run
    env.pop("FAKE_FAULT_AT", None)
    return tmp_path, env


def sh(args, env, cwd=ROOT):
    return subprocess.run(["bash", *args], env=env, cwd=cwd, capture_output=True, text=True)


def test_scripts_parse():
    for f in ("common.sh", "vm.sh", "queue.sh", "laptop.sh"):
        assert subprocess.run(["bash", "-n", str(SESSION / f)]).returncode == 0, f


def test_queue_runs_records_and_continues(tree):
    tmp, env = tree
    env["FAKE_FAULT_AT"] = "L7.w13"
    q = tmp / "queue.txt"
    q.write_text("# a comment\n--layers 2 --iters 1 compare\n\n--layers 8 --head --iters 2 continue\n--layers 2 --iters 4\n")
    r = sh([str(SESSION / "queue.sh"), "run", str(q)], env)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = (tmp / "logs/queue.status").read_text().splitlines()
    assert len(rows) == 4 and rows[-1].endswith(f"DONE {q}")
    assert " L2_it1 PASS rc=0 mpk=1 fault=0 fwd=1 " in rows[0] and rows[0].endswith("compare=PASS")
    assert " L8_head_it2 FAIL rc=1 mpk=0 fault=2 fwd=0 " in rows[1]
    assert " L2_it4 PASS rc=0 mpk=1 fault=0 fwd=4 " in rows[2]
    # the record holds the reports and the log, not the tensors
    rec = tmp / "record/runs/L2_it1"
    assert (rec / "plan.json").exists() and (rec / "run.out").exists() and (rec / "correctness_report.md").exists()
    assert not (rec / "big.safetensors").exists()
    assert (tmp / "record/runs/L8_head_it2/run.out").read_text().count("AcceleratorError") == 1


def test_queue_keeps_the_earlier_record_of_a_rerun(tree):
    """A run name that ran before keeps its earlier record directory beside the new one
    (round 3, 2026-09-17: identical final rows overwrote one directory)."""
    tmp, env = tree
    q = tmp / "queue.txt"
    q.write_text("--layers 2 --iters 1\n")
    assert sh([str(SESSION / "queue.sh"), "run", str(q)], env).returncode == 0
    first = (tmp / "record/runs/L2_it1/run.out").read_text()
    assert sh([str(SESSION / "queue.sh"), "run", str(q)], env).returncode == 0
    runs = sorted(d.name for d in (tmp / "record/runs").iterdir())
    assert runs[0] == "L2_it1" and len(runs) == 2 and runs[1].startswith("L2_it1.prev-")
    assert (tmp / "record/runs" / runs[1] / "run.out").read_text() == first
    assert (tmp / "record/runs/L2_it1/run.out").exists()


def test_queue_stops_on_fail_without_continue(tree):
    tmp, env = tree
    env["FAKE_FAULT_AT"] = "L7.w13"
    q = tmp / "queue.txt"
    q.write_text("--layers 8 --head --iters 2\n--layers 2 --iters 1\n")
    r = sh([str(SESSION / "queue.sh"), "run", str(q)], env)
    assert r.returncode == 1
    rows = (tmp / "logs/queue.status").read_text().splitlines()
    assert len(rows) == 2 and " L8_head_it2 FAIL " in rows[0] and rows[1].endswith("STOP L8_head_it2")
    assert not (tmp / "fleet_out/L2_it1").exists()


def test_queue_bitdiff_dry_run_prints_the_command(tree):
    """DRY=1: no directories need to exist, the command is printed instead of run (L7)."""
    tmp, env = tree
    env["DRY"] = "1"
    r = sh([str(SESSION / "queue.sh"), "bitdiff", "run_a", "run_b"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    expect = (f"+ {PY} harness/bitdiff.py {tmp / 'fleet_out/run_a'} {tmp / 'fleet_out/run_b'} "
             f"--out {tmp / 'record/bitdiff_run_a_run_b.md'}")
    assert expect in r.stdout


def test_queue_bitdiff_missing_args_usage(tree):
    tmp, env = tree
    r = sh([str(SESSION / "queue.sh"), "bitdiff", "run_a"], env)
    assert r.returncode == 2 and "usage: queue.sh bitdiff" in r.stdout


def test_queue_bitdiff_writes_the_record_and_prints_its_path(tree):
    tmp, env = tree
    env["BITDIFF"] = f"{PY} {tmp / 'bitdiff_fake.py'}"
    (tmp / "bitdiff_fake.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "out = Path(sys.argv[sys.argv.index('--out') + 1])\n"
        "out.parent.mkdir(parents=True, exist_ok=True)\n"
        "out.write_text('# Bit-diff report\\n')\n"
    )
    r = sh([str(SESSION / "queue.sh"), "bitdiff", "run_a", "run_b"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    report = tmp / "record/bitdiff_run_a_run_b.md"
    assert r.stdout.strip().splitlines()[-1] == str(report)
    assert report.exists() and report.read_text() == "# Bit-diff report\n"


@pytest.mark.parametrize("fault_at,expect_runs", [("L7.w13", 5), ("L7.norm1", 5), ("head.argmax_reduce", 5)])
def test_bisect_finds_the_first_faulting_label(tree, fault_at, expect_runs):
    tmp, env = tree
    env["FAKE_FAULT_AT"] = fault_at
    r = sh([str(SESSION / "queue.sh"), "bisect", str(SESSION / "queue-fault.txt"), "--",
            "--layers", "8", "--head", "--iters", "2"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    result = (tmp / "logs/bisect.result").read_text().strip()
    assert result.startswith(f"BISECT first-fault={fault_at} runs=")
    assert int(result.split("runs=")[1]) <= expect_runs
    rows = (tmp / "logs/queue.status").read_text().splitlines()
    assert rows[-1].endswith(f"BISECT first-fault={fault_at} runs={result.split('runs=')[1]}")
    assert all(" bisect" in row for row in rows[:-1])


def test_bisect_stops_on_a_fail_without_a_fault_line(tree):
    # session A, 2026-09-16: five runs failed in the JIT (no fault line) and the bisection narrowed on
    # them as if they were the fault; an error row now ends the bisection with an error verdict
    tmp, env = tree
    env["FAKE_FAULT_AT"] = "L7.w13"
    env["FAKE_ERROR_AT"] = "L7.router"       # the first probe of the 16-label list (index 7)
    r = sh([str(SESSION / "queue.sh"), "bisect", str(SESSION / "queue-fault.txt"), "--",
            "--layers", "8", "--head", "--iters", "2"], env)
    assert r.returncode == 0, r.stdout + r.stderr
    result = (tmp / "logs/bisect.result").read_text().strip()
    assert result.startswith("BISECT error=L7.router (FAIL without a fault line"), result
    assert result.endswith("runs=1")
    rows = (tmp / "logs/queue.status").read_text().splitlines()
    assert rows[0].split()[2] == "ERROR" and "fault=0" in rows[0], rows[0]


def test_bisect_reports_no_fault_when_nothing_faults(tree):
    tmp, env = tree
    r = sh([str(SESSION / "queue.sh"), "bisect", str(SESSION / "queue-fault.txt"), "--", "--layers", "8"], env)
    assert r.returncode == 0
    assert (tmp / "logs/bisect.result").read_text().startswith("BISECT no-fault-up-to=head.argmax_reduce")


def test_vm_dry_runs_every_stage_without_touching_the_machine(tree):
    tmp, env = tree
    env["DRY"] = "1"
    for stage in ("download", "image", "setup", "checks", "reference", "kernels"):
        r = sh([str(SESSION / "vm.sh"), stage], env)
        assert r.returncode == 0, stage + r.stdout + r.stderr
        assert "+ " in r.stdout, stage
    status = (tmp / "logs/session.status").read_text()
    assert status.count("PASS") == 6 and "FAIL" not in status
    r = sh([str(SESSION / "vm.sh"), "start", "queue", "q.txt"], env)
    assert "setsid nohup" in r.stdout and "START queue q.txt" in status + r.stdout
    r = sh([str(SESSION / "vm.sh"), "status"], env)
    assert r.returncode == 0 and "session.status" in r.stdout


def test_vm_measure_row_in_dry_mode_names_the_four_pmc_pairs(tree):
    tmp, env = tree
    env["DRY"] = "1"
    q = tmp / "queue.txt"
    q.write_text("--layers 27 --head --iters 32 --event-timing measure\n--layers 2 --iters 32 --event-timing table\n")
    r = sh([str(SESSION / "queue.sh"), "run", str(q)], env)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout + r.stderr
    assert "measure.py --run" in out.replace("harness/measure.py", "measure.py") or "--run" in out
    assert (tmp / "logs/queue.status").read_text().rstrip().splitlines()[1].endswith("table=PASS")
    for pair in ("TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_32B_sum", "TCC_EA0_WRREQ_sum TCC_EA0_WRREQ_64B_sum",
                 "TCC_HIT_sum TCC_MISS_sum", "TCC_BUBBLE_sum TCC_EA0_RDREQ_sum"):
        assert f"--pmc {pair}" in out
    assert "--kernel-trace" in out and out.count("rocprofv3") == 5
    assert "--pmc " in out and "pmc_all" not in out      # the runs are merged per file by measure.py, not concatenated
    assert (tmp / "logs/queue.status").read_text().rstrip().splitlines()[0].endswith("measure=PASS")


def test_laptop_dry_mode(tree, tmp_path):
    tmp, env = tree
    env["DRY"] = "1"
    env["VM_IP_FILE"] = str(tmp_path / "vm.ip")
    env["TUI"] = "echo TUI"
    r = sh([str(SESSION / "laptop.sh"), "ip"], env)
    assert r.returncode == 0 and (tmp_path / "vm.ip").read_text().strip() == "0.0.0.0"
    r = sh([str(SESSION / "laptop.sh"), "push"], env)
    assert "rsync -az" in r.stdout and "hotaisle@0.0.0.0:/home/hotaisle/metalOps/" in r.stdout
    assert "--exclude .venv-fleet" in r.stdout and "--exclude docs/report" in r.stdout
    r = sh([str(SESSION / "laptop.sh"), "start", "queue", "env/session/queue-a.txt"], env)
    assert "vm.sh start queue env/session/queue-a.txt" in r.stdout and "</dev/null" not in r.stdout
    r = sh([str(SESSION / "laptop.sh"), "delete"], env)
    assert r.returncode == 2 and "--yes" in r.stdout
    r = sh([str(SESSION / "laptop.sh"), "delete", "--yes"], env)
    assert r.returncode == 0 and "DOWN DOWN DOWN DOWN DOWN DOWN" in r.stdout


def test_queue_files_parse_as_run_fleet_arguments():
    sys.path.insert(0, str(ROOT / "harness"))
    import run_fleet
    for f in ("queue-a.txt", "queue-a2.txt", "queue-b.txt", "queue-fix.txt"):
        rows = [l.split("#")[0].split() for l in (SESSION / f).read_text().splitlines()]
        rows = [r for r in rows if r]
        assert rows, f
        for r in rows:
            args = [w for w in r if w not in ("compare", "table", "measure", "continue")]
            a = run_fleet.build_parser().parse_args(args + ["--model-dir", "x"])
            assert 1 <= a.iters <= 32, (f, r)


def test_record_dir_is_pinned_for_the_session(tree):
    """common.sh writes env/logs/record.dir on the first call and reuses it afterwards, so a
    session that crosses UTC midnight keeps one record directory (RECORD in the env overrides)."""
    tmp, env = tree
    env.pop("RECORD")
    env["LOGDIR"] = str(tmp / "logs")
    r = sh(["-c", f"source {SESSION / 'common.sh'}; echo $RECORD"], env)
    first = r.stdout.strip()
    assert first.startswith(str(ROOT / "env/hw/20")) and (tmp / "logs/record.dir").read_text().strip() == first
    (tmp / "logs/record.dir").write_text(str(ROOT / "env/hw/29991231") + "\n")
    r = sh(["-c", f"source {SESSION / 'common.sh'}; echo $RECORD"], env)
    assert r.stdout.strip() == str(ROOT / "env/hw/29991231")
    env["RECORD"] = str(tmp / "explicit")
    r = sh(["-c", f"source {SESSION / 'common.sh'}; echo $RECORD"], env)
    assert r.stdout.strip() == str(tmp / "explicit")


FAKE_PROFILER = r'''
import os, sys, subprocess
from pathlib import Path
# rocprofv3 look-alike: [--kernel-trace | --pmc A B] --output-format csv -d DIR -- cmd...
args = sys.argv[1:]
mode = "trace" if "--kernel-trace" in args else "pmc"
counters = []
if mode == "pmc":
    i = args.index("--pmc") + 1
    while not args[i].startswith("-"):
        counters.append(args[i]); i += 1
d = Path(args[args.index("-d") + 1]) / "fakehost"
cmd = args[args.index("--") + 1:]
subprocess.run(cmd, check=True)
d.mkdir(parents=True, exist_ok=True)
names = ["__amd_rocclr_copyBuffer", "prepare_kernel(mirage::runtime::RuntimeConfig)",
         "worker_kernel(mirage::runtime::RuntimeConfig)", "scheduler_kernel(mirage::runtime::RuntimeConfig)"]
if mode == "trace":
    rows = ["Kernel_Name,Start_Timestamp,End_Timestamp"]
    for k, n in enumerate(names):
        rows.append(f'"{n}",{1000 * k},{1000 * k + (4000000 if "worker" in n else 500)}')
    (d / "1_kernel_trace.csv").write_text("\n".join(rows) + "\n")
else:
    rows = ["Dispatch_Id,Kernel_Name,Counter_Name,Counter_Value"]
    for k, n in enumerate(names):
        for c in counters:
            v = 99999 if n.startswith("__amd") else 4096   # the blit must be filtered out; the three megakernel dispatches count
            rows.append(f'{k},"{n}",{c},{v}')
    (d / "1_counter_collection.csv").write_text("\n".join(rows) + "\n")
'''


def test_queue_measure_row_end_to_end_with_a_fake_profiler(tree):
    """The non-DRY measure path: five profiled runs through a rocprofv3 look-alike, the real
    measure.py on the directory of runs, the metrics filtered to the megakernel, and the
    per-run CSVs copied into the record."""
    tmp, env = tree
    (tmp / "rocprofv3_fake.py").write_text(FAKE_PROFILER)
    env["PROFILER"] = f"{PY} {tmp / 'rocprofv3_fake.py'}"
    env["MEASURE"] = f"{PY} {ROOT / 'harness/measure.py'}"
    q = tmp / "queue.txt"
    q.write_text("--layers 2 --iters 4 measure\n")
    r = sh([str(SESSION / "queue.sh"), "run", str(q)], env)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = (tmp / "logs/queue.status").read_text().splitlines()
    assert " L2_it4 PASS " in rows[0] and rows[0].endswith("measure=PASS"), rows
    rec = tmp / "record/runs/L2_it4"
    assert (tmp / "logs/prof/L2_it4/ktrace").is_dir() and (tmp / "logs/prof/L2_it4/pmc4").is_dir()   # per run, not shared
    m = json.loads((rec / "metrics.json").read_text())
    assert m["launches"]["dispatches"] == 4 and m["launches"]["megakernel_dispatches"] == 3
    assert abs(m["launches"]["per_iteration_us_from_trace"] - (4000 + 0.5 + 0.5) / 4) < 1e-9
    # every counter is 4096 on each of the three megakernel dispatches, 99999 on the blit (excluded);
    # with BUBBLE = RDREQ = RDREQ_32B = 12288 the validated decomposition gives 128 - 64 + 32 = 96 B
    # per request, over 4 iterations
    assert m["counters"]["TCC_BUBBLE_sum"] == 3 * 4096 and m["counters"]["TCC_HIT_sum"] == 3 * 4096
    sys.path.insert(0, str(ROOT / "harness")); import measure
    assert measure.bytes_read(m["counters"]) == 3 * 4096 * 96
    assert abs(m["traffic"]["read_MiB_per_iteration"] - 3 * 4096 * 96 / 2**20 / 4) < 1e-12
    assert sorted(p.name for p in (rec / "prof").iterdir()) == [
        "ktrace_1_kernel_trace.csv", "pmc1_1_counter_collection.csv", "pmc2_1_counter_collection.csv",
        "pmc3_1_counter_collection.csv", "pmc4_1_counter_collection.csv"]
    assert (rec / "report_table.md").read_text().count("3 megakernel of 4 dispatches") == 1


def test_queue_guards_fail_a_row_before_it_runs(tree):
    tmp, env = tree
    q = tmp / "queue.txt"
    q.write_text("--layers 2 --iters 64 continue\n--layers 27 --debug --iters 4 continue\n"
                 "--layers 2 --iters 1 compare\n")
    (tmp / "ref/ref_cache.safetensors").unlink()                      # no reference tensors
    r = sh([str(SESSION / "queue.sh"), "run", str(q)], env)
    assert r.returncode == 1
    rows = (tmp / "logs/queue.status").read_text().splitlines()
    assert "FAIL guard: --iters 64 is above 32" in rows[0]
    assert "FAIL guard: --debug is the growth curve" in rows[1]
    assert "FAIL guard: compare needs" in rows[2] and rows[3].endswith("STOP L2_it1")
    assert not (tmp / "fleet_out").exists()                            # nothing ran
    (tmp / "ref/ref_cache.safetensors").write_bytes(b"0")
    env["PROFILER"] = "/nonexistent/rocprofv3"
    q.write_text("--layers 2 --iters 1 measure\n")
    r = sh([str(SESSION / "queue.sh"), "run", str(q)], env)
    assert "FAIL guard: measure needs /nonexistent/rocprofv3" in (tmp / "logs/queue.status").read_text()


# ---- the helpers for the VM (docs/gpu-experiments/02-validation/02-session-plan.md, "Helpers") ----------------

def test_pf_toggle_sets_both_kernels(tmp_path):
    import shutil
    d = tmp_path / "fleet/tasks/mi300"; d.mkdir(parents=True)
    for f in ("mla_attend_mi300.cuh", "mla_merge_uv_mi300.cuh"):
        shutil.copy(ROOT / "fleet/tasks/mi300" / f, d / f)
    env = dict(os.environ, ROOT=str(tmp_path))
    # round 3: the toggle sets the VALU attention's PF only and prints the merge's batch constants
    r = sh([str(SESSION / "pf.sh"), "1"], env)
    assert r.returncode == 0 and r.stdout.count("PF = 1;") == 1 and "ROWS_IN_FLIGHT" in r.stdout, r.stdout + r.stderr
    att = d / "mla_attend_mi300.cuh"
    assert "  constexpr int PF = 1;" in att.read_text() and "PF = 4;" not in att.read_text()
    assert (d / "mla_merge_uv_mi300.cuh").read_text() == (ROOT / "fleet/tasks/mi300/mla_merge_uv_mi300.cuh").read_text()
    r = sh([str(SESSION / "pf.sh")], env)
    assert r.returncode == 0 and r.stdout.count("PF = 1;") == 1
    r = sh([str(SESSION / "pf.sh"), "4"], env)
    assert r.returncode == 0 and "  constexpr int PF = 4;" in att.read_text()
    assert sh([str(SESSION / "pf.sh"), "3"], env).returncode == 2
    # the tree itself is at 4 and untouched
    assert "  constexpr int PF = 4;" in (ROOT / "fleet/tasks/mi300/mla_attend_mi300.cuh").read_text()


def test_queue_flag_adds_flags_before_keywords_and_comments(tmp_path):
    q = tmp_path / "q.txt"
    q.write_text("# head\n--layers 8 --head --iters 2                # A8.1\n--layers 27 --head --iters 32 compare\n\n"
                 "--layers 2 --iters 32 --event-timing --tile-linears table   # already\n")
    r = subprocess.run([PY, str(SESSION / "queue_flag.py"), str(q), "--align-alloc", "65536", "--tile-linears"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[0] == "# head" and lines[3] == ""
    assert lines[1].startswith("--layers 8 --head --iters 2 --align-alloc 65536 --tile-linears") and lines[1].rstrip().endswith("# A8.1")
    assert lines[2] == "--layers 27 --head --iters 32 --align-alloc 65536 --tile-linears compare"
    assert lines[4].startswith("--layers 2 --iters 32 --event-timing --tile-linears --align-alloc 65536 table") and lines[4].count("--tile-linears") == 1
    r = subprocess.run([PY, str(SESSION / "queue_flag.py"), str(q), "--align-alloc", "65536", "--in-place"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and q.read_text().count("--align-alloc 65536") == 3
    again = subprocess.run([PY, str(SESSION / "queue_flag.py"), str(q), "--align-alloc", "65536"], capture_output=True, text=True)
    assert again.stdout == q.read_text()                                   # idempotent
    # every row still parses for run_fleet.py
    sys.path.insert(0, str(ROOT / "harness")); import run_fleet
    for row in q.read_text().splitlines():
        words = [w for w in row.split("#")[0].split() if w not in ("compare", "table", "measure", "continue")]
        if words:
            run_fleet.build_parser().parse_args(words + ["--model-dir", "x"])


def test_addr_diff_reports_moved_tensors_and_the_4gib_crossing(tmp_path):
    a = {"layers": 8, "head": True, "iters": 2, "pad_alloc_gb": 0, "pad_addr": None, "align_alloc": 0,
         "workspaces_first": False, "output_ids": [0, 0],
         "addresses": {"W_embed": 0x7f00fff00000, "x_res": 0x7f00fffff000, "partials": 0x7f0100000000}}
    b = dict(a, pad_alloc_gb=1, pad_addr=0x7e0000000000, output_ids=[25, 16228],
             addresses={"W_embed": 0x7f0100100000, "x_res": 0x7f00fffff000, "partials": 0x7f0140000000, "extra": 1})
    (tmp_path / "a.json").write_text(json.dumps(a)); (tmp_path / "b.json").write_text(json.dumps(b))
    r = subprocess.run([PY, str(SESSION / "addr_diff.py"), str(tmp_path / "a.json"), str(tmp_path / "b.json")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "A: layers 8 head True iters 2 pad 0 GiB" in out and "B: layers 8 head True iters 2 pad 1 GiB" in out
    assert "W_embed" in out and "partials" in out and "x_res" not in out.split("tensor")[1]   # unchanged: hidden
    assert "extra" in out and "only one run" in out
    assert "2 of 3 tensors moved; 1 crossed a 4 GiB (bit 32) boundary" in out
    r = subprocess.run([PY, str(SESSION / "addr_diff.py"), str(tmp_path / "a.json"), str(tmp_path / "b.json"), "--all"],
                       capture_output=True, text=True)
    assert "x_res" in r.stdout


def test_vm_check_and_wait_follow_the_last_start(tree):
    tmp, env = tree
    status = tmp / "logs/session.status"
    status.parent.mkdir(exist_ok=True)
    V = str(SESSION / "vm.sh")
    assert sh([V, "check", "setup"], env).stdout.startswith("NOT STARTED setup")
    status.write_text("2026-09-16T10:00:00Z START setup\n")
    r = sh([V, "check", "setup"], env)
    assert r.returncode == 1 and r.stdout.startswith("RUNNING setup (since 2026-09-16T10:00:00Z)")
    status.write_text(status.read_text() + "2026-09-16T10:20:00Z FAIL setup 1200s: see x\n")
    r = sh([V, "check", "setup"], env)
    assert r.returncode == 1 and " FAIL setup " in r.stdout
    # restarted: the earlier FAIL row no longer counts
    status.write_text(status.read_text() + "2026-09-16T10:21:00Z START setup\n")
    assert sh([V, "check", "setup"], env).stdout.startswith("RUNNING setup (since 2026-09-16T10:21:00Z)")
    status.write_text(status.read_text() + "2026-09-16T10:40:00Z PASS setup 1140s\n")
    r = sh([V, "check", "setup"], env)
    assert r.returncode == 0 and " PASS setup 1140s" in r.stdout
    r = sh([V, "wait", "setup", "1"], env)                                   # returns at once: the row is there
    assert r.returncode == 0 and " PASS setup 1140s" in r.stdout
    assert sh([V, "wait", "nothing", "1"], env).returncode == 2
    # a queue stage is keyed by the word 'queue' whatever the file
    status.write_text(status.read_text() + "2026-09-16T11:00:00Z START queue env/session/queue-a.txt\n2026-09-16T11:30:00Z PASS queue 1800s\n")
    assert sh([V, "check", "queue"], env).returncode == 0


def test_vm_preflight_kill_gdb_and_laptop_wait_report_in_dry_mode(tree, tmp_path):
    tmp, env = tree
    env["DRY"] = "1"
    V = str(SESSION / "vm.sh")
    r = sh([V, "preflight"], env)
    assert r.returncode == 0 and "+ amd-smi list" in r.stdout and "PASS preflight" in (tmp / "logs/session.status").read_text()
    r = sh([V, "kill", "--all"], env)
    assert r.returncode == 0 and "no graph run is running" in r.stdout
    r = sh([V, "gdb", "L7.w13"], env)
    assert r.returncode == 0 and "+ rocgdb --batch -ex run -ex bt" in r.stdout and "--stop-after L7.w13" in r.stdout
    env["VM_IP_FILE"] = str(tmp_path / "vm.ip"); env["TUI"] = "echo TUI"
    (tmp_path / "vm.ip").write_text("0.0.0.0\n")
    r = sh([str(SESSION / "laptop.sh"), "wait", "setup", "5"], env)
    assert r.returncode == 0 and "vm.sh check setup" in r.stdout and "polled every 30 s" in r.stdout
    r = sh([str(SESSION / "laptop.sh"), "report"], env)
    assert "no provisioning time recorded" in r.stdout
    (tmp_path / "vm.started").write_text(str(int(__import__("time").time()) - 90 * 60))
    r = sh([str(SESSION / "laptop.sh"), "report"], env)
    assert "minute 90 since provisioning" in r.stdout and "about $4.49 billed" in r.stdout   # 1.5 h at 2.99


def test_queue_guards_refuse_compare_on_a_synthetic_graph(tree):
    """Round 4 (L8): --graph empty and --graph stream load no reference and have no boundaries, so a
    compare word on such a row is a FAIL row without a run; without the word the row runs even when the
    reference tensors are absent."""
    tmp, env = tree
    env["DRY"] = "1"
    q = tmp / "queue.txt"
    q.write_text("--graph stream --ops 10 --tasks 37 --kb 304 --gang --iters 32 --nt-streams table compare continue\n"
                 "--graph stream --ops 10 --tasks 37 --kb 304 --gang --iters 32 --nt-streams table continue\n")
    r = sh([str(SESSION / "queue.sh"), "run", str(q)], env)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = (tmp / "logs/queue.status").read_text().rstrip().splitlines()
    assert "FAIL guard: a synthetic graph" in rows[0] and "S10x37_304kb_gang_it32_nts" in rows[0]
    assert rows[1].split()[1] == "S10x37_304kb_gang_it32_nts" and " PASS " in rows[1] and "table=PASS" in rows[1]
    (tmp / "ref/ref_cache.safetensors").unlink()                       # no reference stage: the stream row still runs
    env["DRY"] = "0"
    q.write_text("--graph stream --ops 10 --tasks 96 --kb 152 --iters 32 --nt-streams continue\n"
                 "--layers 2 --iters 32 continue\n")
    r = sh([str(SESSION / "queue.sh"), "run", str(q)], env)
    rows = (tmp / "logs/queue.status").read_text().rstrip().splitlines()
    assert any(l.split()[1] == "S10x96_152kb_it32_nts" and " PASS " in l for l in rows), rows
    assert any("L2_it32 FAIL guard: run_fleet.py loads" in l for l in rows), rows
