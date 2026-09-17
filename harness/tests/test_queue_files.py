"""Every row of every round-3 queue file parses as run_fleet.py arguments, names a run, and carries only
the queue's trailing words (docs/gpu-experiments/03-acceleration/05-session-plan.md); a typo fails here."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_fleet  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
QUEUES = sorted((ROOT / "env/session").glob("queue-[cd]*.txt"))
WORDS = {"compare", "table", "measure", "continue"}


def rows(path):
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            yield line.split()


def test_round3_queue_files_exist():
    names = {p.name for p in QUEUES}
    assert {"queue-c2.txt", "queue-c3.txt", "queue-c4.txt", "queue-c5.txt", "queue-c6.txt",
            "queue-d7.txt", "queue-d8.txt", "queue-d9.txt"} <= names


def test_every_row_parses_and_names_a_run():
    p = run_fleet.build_parser()
    seen = {}
    for q in QUEUES:
        for toks in rows(q):
            args = [t for t in toks if t not in WORDS]
            words = [t for t in toks if t in WORDS]
            assert toks[len(args):] == words, f"{q.name}: the words go last: {toks}"
            a = p.parse_args(args + ["--model-dir", "x"])
            name = run_fleet.run_name(a)
            assert name and " " not in name and '"' not in name, (q.name, name)
            assert a.iters <= 32, (q.name, toks)
            seen.setdefault(name, []).append(q.name)
    # no two rows across the files produce the same run directory unless they are the repeats of G9
    dup = {n: fs for n, fs in seen.items() if len(fs) > 1 and not all(f == "queue-d9.txt" for f in fs)}
    assert not dup, dup


def test_knob_rows_use_the_equals_form():
    for toks in rows(ROOT / "env/session/queue-c4.txt"):
        for t in toks:
            assert not t.startswith("--runtime-flags ") and (not t.startswith("--runtime-flags") or "=" in t), toks
