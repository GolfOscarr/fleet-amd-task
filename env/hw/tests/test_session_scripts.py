"""env/session/{common,vm,queue,laptop}.sh against a fake run_fleet.py in a temporary tree
(docs/round-2/01-preparation.md, P4): the queue's status rows, the record copies, the stop rule,
the bisection over the P1 label file, and the DRY modes of vm.sh and laptop.sh."""
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
for i in range(args.iters):
    print(f"[FWD_PASS] iter={i} time_ms=1.0 num_active_tokens=1")
(out / "fleet_run_meta.json").write_text(json.dumps({"layers": args.layers}))
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
    })
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
    for f in ("queue-a.txt", "queue-a2.txt", "queue-b.txt"):
        rows = [l.split("#")[0].split() for l in (SESSION / f).read_text().splitlines()]
        rows = [r for r in rows if r]
        assert rows, f
        for r in rows:
            args = [w for w in r if w not in ("compare", "table", "measure", "continue")]
            a = run_fleet.build_parser().parse_args(args + ["--model-dir", "x"])
            assert 1 <= a.iters <= 32, (f, r)
