"""Every row of every round-3 and round-4 queue file parses as run_fleet.py arguments, names a run, and
carries only the queue's trailing words (docs/gpu-experiments/03-acceleration/05-session-plan.md,
04-kernels/07-session-plan.md); a typo fails here, and a round-4 row's flags must build a plan."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import run_fleet  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
ROUND4 = sorted((ROOT / "env/session").glob("queue-[fg][0-9].txt"))   # not the fixtures queue-fault*, queue-fix*
ROUND5 = sorted((ROOT / "env/session").glob("queue-h[0-9].txt"))      # docs/gpu-experiments/05-final, F8
QUEUES = sorted((ROOT / "env/session").glob("queue-[cd]*.txt")) + ROUND4 + ROUND5
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


def test_round4_queue_files_exist():
    names = {p.name for p in ROUND4}
    assert {"queue-f2.txt", "queue-f3.txt", "queue-f4.txt", "queue-f5.txt", "queue-f6.txt",
            "queue-g1.txt", "queue-g2.txt", "queue-f7.txt", "queue-f8.txt", "queue-f9.txt", "queue-g3.txt", "queue-g4.txt"} == names


def test_round5_queue_files_exist():
    names = {p.name for p in ROUND5}
    assert {"queue-h1.txt", "queue-h2.txt", "queue-h3.txt", "queue-h4.txt", "queue-h5.txt", "queue-h6.txt", "queue-h7.txt"} == names
    for q in ROUND5:                       # every round-5 row runs the finals' stack (F3)
        for toks in rows(q):
            assert "--final" in toks, (q.name, toks)


def test_every_row_parses_and_names_a_run():
    p = run_fleet.build_parser()
    seen = {}
    for q in QUEUES:
        for toks in rows(q):
            args = [t for t in toks if t not in WORDS]
            words = [t for t in toks if t in WORDS]
            assert toks[len(args):] == words, f"{q.name}: the words go last: {toks}"
            a = run_fleet.parse_args(args + ["--model-dir", "x"])
            name = run_fleet.run_name(a)
            assert name and " " not in name and '"' not in name, (q.name, name)
            assert a.iters <= 32, (q.name, toks)
            seen.setdefault(name, []).append(q.name)
    # no two rows across the files produce the same run directory unless they are the repeats of G9
    dup = {n: fs for n, fs in seen.items() if len(fs) > 1 and not all(f == "queue-d9.txt" for f in fs)}
    assert not dup, dup


def test_knob_rows_use_the_equals_form():
    for q in (ROOT / "env/session/queue-c4.txt", ROOT / "env/session/queue-f6.txt", ROOT / "env/session/queue-f2.txt"):
        for toks in rows(q):
            for t in toks:
                assert not t.startswith("--runtime-flags ") and (not t.startswith("--runtime-flags") or "=" in t), toks


def test_round4_and_round5_rows_build_their_plans_and_obey_the_rules():
    """Every model row of the round-4 and round-5 files builds its plan (the flag asserts fire here, not on the VM),
    never pairs a fence knob with a counter form, and never probes the o_proj label under the fold; the
    stream rows read whole 4 KB rows. (The load policy is not asserted: G1.3 and the stream rows A/B it.)"""
    sys.path.insert(0, str(ROOT))
    from fleet import build_graph as B
    p = run_fleet.build_parser()
    seen = set()
    for q in ROUND4 + ROUND5:
        for toks in rows(q):
            a = run_fleet.parse_args([t for t in toks if t not in WORDS] + ["--model-dir", "x"])
            assert a.iters <= 32 and (a.iters == 1 or not a.debug), (q.name, toks)
            assert not run_fleet.fence_knob_conflict(a), (q.name, toks)
            if a.graph == "stream":
                assert a.kb % 4 == 0 and a.ops >= 1 and a.tasks >= 1, (q.name, toks)
                continue
            key = (a.layers, a.head, a.gemv_linears, a.linear_grid, a.head_grid, a.gemv_w13, a.merge_tasks,
                   a.merge_halves, a.router_tasks, a.merge_oproj, a.fuse_norm1, a.fuse_norm2, a.fuse_silu,
                   a.tile_linears, a.attend_tasks, a.probe_before, a.argmax_slices, a.stop_after)
            if key in seen:
                continue
            seen.add(key)
            B.dry_run(layers=a.layers, head=a.head, debug=a.debug, stop_after=a.stop_after,
                      debug_scores=a.debug_scores, tile_linears=a.tile_linears, attend_tasks=a.attend_tasks,
                      fuse_norm2=a.fuse_norm2, fuse_silu=a.fuse_silu, probe_before=a.probe_before,
                      fuse_norm1=a.fuse_norm1, prefetch=a.prefetch, gemv_linears=a.gemv_linears,
                      linear_grid=a.linear_grid, head_grid=a.head_grid, gemv_w13=a.gemv_w13,
                      merge_tasks=a.merge_tasks, merge_halves=a.merge_halves, router_tasks=a.router_tasks,
                      merge_oproj=a.merge_oproj, argmax_slices=a.argmax_slices or 50)


def test_queue_flag_removes_a_failed_lever(tmp_path):
    """The agent's mid-session edit: a lever that failed its compare taken out of the stacked rows."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("queue_flag", ROOT / "env/session/queue_flag.py")
    qf = importlib.util.module_from_spec(spec); spec.loader.exec_module(qf)
    text = ("# comment\n--layers 27 --head --iters 32 --fuse-norm2 --fuse-silu --fuse-norm1 table compare continue   # G9\n"
            "--layers 2 --iters 32 --runtime-flags=-DMPK_POLL_SLEEP=8 --fuse-silu table continue\n")
    out = qf.remove_flags(text, {"--fuse-silu"})
    assert out.splitlines()[1].startswith("--layers 27 --head --iters 32 --fuse-norm2 --fuse-norm1 table compare continue")
    assert out.splitlines()[1].endswith("# G9") and "--fuse-silu" not in out
    assert "--runtime-flags=-DMPK_POLL_SLEEP=8" in out.splitlines()[2]
    # a flag with a separate value goes with its value; keywords and comments stay
    out = qf.remove_flags("--layers 2 --iters 32 --probe-before L0.o_proj table continue\n", {"--probe-before"})
    assert out == "--layers 2 --iters 32 table continue\n"
    assert qf.add_flags(out, ["--nt-streams"]) == "--layers 2 --iters 32 --nt-streams table continue\n"
