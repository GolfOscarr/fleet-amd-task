"""fleet/tasks/kernel_tests.py without a GPU: the file contract with the binary
(serialization round trips, the tensor tables against kernel_tests_mi300.cu),
the comparison rows, and the --dry-run path end to end."""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(ROOT / "fleet/tasks"))
import kernel_tests as K  # noqa: E402
import numpy_ref as R  # noqa: E402

CU = ROOT / "fleet/tasks/kernel_tests_mi300.cu"


# ----------------------------------------------------------------------------
# serialization


def test_bf16_file_round_trip():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 512), dtype=np.float32) * 3
    raw = K.to_file(x, "bf16")
    assert raw.dtype == np.dtype("<u2") and raw.shape == x.shape
    back = K.from_file(raw.tobytes(), "bf16", x.shape)
    assert back.dtype == np.float32 and np.array_equal(back, R.bf16(x))
    assert np.array_equal(K.from_file(K.to_file(back, "bf16").tobytes(), "bf16", x.shape), back)


def test_f32_i32_round_trip_little_endian():
    f = np.array([1.5, -2.25, 1e-30], np.float32)
    assert np.array_equal(K.from_file(K.to_file(f, "f32").tobytes(), "f32", (3,)), f)
    i = np.array([-7, 0, 65], np.int32)
    assert np.array_equal(K.from_file(K.to_file(i, "i32").tobytes(), "i32", (3,)), i)
    assert K.to_file(np.array([1], np.int32), "i32").tobytes() == b"\x01\x00\x00\x00"
    assert K.to_file(np.array([1.0], np.float32), "bf16").tobytes() == b"\x80\x3f"


def test_params_and_float_bits(tmp_path):
    p = {"step": 1040, "softmax_scale_bits": K.G.float_bits(K.G.SOFTMAX_SCALE), "debug_scores": 1}
    K.write_params(tmp_path / "params.txt", p)
    assert K.read_params(tmp_path / "params.txt") == p
    assert K.bits_to_float(p["softmax_scale_bits"]) == np.float32(K.G.SOFTMAX_SCALE)


def test_trial_files_and_shapes(tmp_path):
    rng = np.random.default_rng(1)
    for name, kernel in K.KERNELS.items():
        tensors, params = kernel.make(rng)
        d = tmp_path / name
        K.write_trial(d, kernel, tensors, params)
        specs = kernel.tensors(params)
        assert (d / "params.txt").exists()
        for s in specs:
            size = (d / f"{s.name}.bin").stat().st_size
            assert size == int(np.prod(s.shape)) * K.FILE_DTYPE[s.dtype].itemsize, (name, s.name)
        K.fake_binary(kernel, [d])
        t2, p2, got = K.read_trial(d, kernel)
        assert p2 == params
        for s in specs:
            assert t2[s.name].shape == s.shape
            assert np.array_equal(t2[s.name], tensors[s.name], equal_nan=True)
            if s.out:
                assert got[s.name].shape == s.shape
                assert got[s.name].dtype == (np.int32 if s.dtype == "i32" else np.float32)
    # the shapes the binary hard-codes
    assert K.QKVA == 3648 and K.N_SPLITS == 33 and K.TILES_PER_XCD == 5 and K.ROUTE_SHAPE == (32, 26, 8)


class Launcher:
    """The Python-facing contract parsed out of kernel_tests_mi300.cu: the constexpr
    dims, the Spec tables (name, byte-count expression, output flag), the dynamic
    byte counts a run_* function sets, and the params.txt names it reads."""

    def __init__(self, src):
        self.consts = {}
        for m in re.finditer(r"^constexpr int ([^;]*);", src, re.M):
            for decl in m.group(1).split(","):
                name, expr = (x.strip() for x in decl.split("="))
                if "::" not in expr:
                    self.consts[name] = self.evaluate(expr)
        self.tables = {}
        for m in re.finditer(r"static const Spec SPEC_(\w+)\[\] = \{(.*?)\};", src, re.S):
            entries = re.findall(r'\{"(\w+)",\s*([^,]+),\s*(true|false)\}', m.group(2))
            self.tables[m.group(1).lower()] = [(n, self.evaluate(e), o == "true") for n, e, o in entries]
        m = re.search(r'SPEC_MLA_ATTEND_SCORES = \{"(\w+)", ([^,]+), (true|false)\};', src)
        self.scores = (m.group(1), self.evaluate(m.group(2)), m.group(3) == "true")
        self.runs = {}
        for m in re.finditer(r"^void run_(\w+)\(std::string const &dir\) \{(.*?)^\}", src, re.M | re.S):
            body = m.group(2)
            self.runs[m.group(1)] = {
                "required": set(re.findall(r'\bparam\(p, "(\w+)"\)', body)),
                "optional": set(re.findall(r'\bparam_or\(p, "(\w+)"', body)),
                "dynamic": {int(i): e for i, e in re.findall(r"b\.specs\[(\d+)\]\.bytes = ([^;]+);", body)},
            }

    def evaluate(self, expr, **names):
        # integer arithmetic on the launcher's own constants; the expression comes
        # from the checked-in .cu and is restricted to identifiers and + - * / ( )
        expr = expr.replace("(size_t)", "")
        assert re.fullmatch(r"[\w\s()+*/-]+", expr), expr
        return int(eval(expr, {"__builtins__": {}}, {**self.consts, **names}))


def python_bytes(spec):
    return int(np.prod(spec.shape)) * K.FILE_DTYPE[spec.dtype].itemsize


def test_launcher_contract():
    L = Launcher(CU.read_text())
    assert L.consts["QKVA"] == K.QKVA and L.consts["S_MAX"] == K.S_MAX and L.consts["N_TOTAL"] == K.N_TOTAL
    assert (L.consts["ROUTE_STEPS"], L.consts["ROUTE_LAYERS"], L.consts["N_SLOTS"]) == K.ROUTE_SHAPE
    assert set(L.tables) == set(K.KERNELS) == set(L.runs)
    for name, kernel in K.KERNELS.items():
        for make in ([kernel.make] + ([K.make_mla_attend_scores] if name == "mla_attend" else [])):
            _, params = make(np.random.default_rng(0))
            specs = kernel.tensors(params)
            table = list(L.tables[name])
            if params.get("debug_scores"):
                table.append(L.scores)
            # names, order and output flags
            assert [(n, o) for n, _, o in table] == [(s.name, s.out) for s in specs], name
            # byte counts: static from the table, dynamic from the run_* body
            run = L.runs[name]
            for i, (spec, (_, size, _)) in enumerate(zip(specs, table)):
                if i in run["dynamic"]:
                    size = L.evaluate(run["dynamic"][i], n_splits=params["n_splits"])
                assert size == python_bytes(spec), (name, spec.name, size, python_bytes(spec))
            # every param the binary requires is written, and everything written is read
            assert run["required"] <= set(params), (name, run["required"] - set(params))
            assert set(params) <= run["required"] | run["optional"], (name, set(params) - run["required"])
    # the 1-split launch of mla_attend_splits reuses the same contract
    p1 = K.attend_params(1030, split=K.S_MAX, n_splits=1)
    assert L.evaluate(L.runs["mla_attend"]["dynamic"][4], n_splits=1) == python_bytes(K.tensors_mla_attend(p1)[4])


def test_dry_run_output_is_gitignored():
    assert K.DEFAULT_OUT_DRY.name == "kernel_tests_dryrun.json" and K.DEFAULT_OUT.name == "kernel_tests.json"
    assert "fleet/tasks/results/*_dryrun.json" in (ROOT / ".gitignore").read_text().splitlines()


# ----------------------------------------------------------------------------
# comparison rows


def test_metrics_match_compare_py():
    torch = pytest.importorskip("torch")
    import compare  # noqa: E402  (harness/compare.py)

    rng = np.random.default_rng(2)
    b = rng.standard_normal(1000).astype(np.float32)
    a = b + rng.standard_normal(1000).astype(np.float32) * 1e-3
    m = K.metrics(a, b)
    ref = compare.metrics(torch.tensor(a), torch.tensor(b))
    for k in ("max_abs_err", "rel_err", "cos_sim"):
        assert abs(m[k] - ref[k]) < 1e-12, k
    assert m["n"] == ref["n"]


def bf1(v):
    return R.bf16(np.array([v], np.float32))[0]


def test_rows():
    exp = R.bf16(np.random.default_rng(3).standard_normal(4096, dtype=np.float32))
    got = exp.copy()
    assert K.row_bf16("x", got, exp)["ok"]
    got[7] = bf1(exp[7] + abs(exp[7]) * 2 ** -8)                  # one BF16 ulp up
    assert got[7] != exp[7] and K.row_bf16("x", got, exp)["ok"]
    j = int(np.argmax(np.abs(exp)))                              # the largest element: no cancellation slack
    got[7] = exp[7]
    got[j] = bf1(exp[j] * (1 + 8 * 2 ** -8))                      # four ulps of the largest element
    assert not K.row_bf16("x", got, exp)["ok"]
    got = exp.copy()
    k = int(np.argmin(np.abs(exp)))                              # a small element: its own ulp, not the array's
    got[k] = bf1(exp[k] + 0.02 * np.abs(exp).max())
    assert K.metrics(got, exp)["rel_err"] < K.BF16_REL and not K.row_bf16("x", got, exp)["ok"]
    got[k] = exp[k] + 0.5 * K.BF16_ABS_FLOOR * np.abs(exp).max()  # within the noise floor
    assert K.row_bf16("x", got, exp)["ok"]
    got = exp.copy()
    got[3] = np.nan                                              # an unwritten sentinel
    assert not K.row_exact("x", got, exp)["ok"] and not K.row_bf16("x", got, exp)["ok"]
    sent = np.full(8, np.nan, np.float32)
    assert K.row_exact("untouched", sent, sent)["ok"]
    assert not K.row_exact("i", np.array([1, -7], np.int32), np.array([1, 2], np.int32))["ok"]
    assert K.row_rel("r", exp * (1 + 5e-5), exp, 1e-4)["ok"] and not K.row_rel("r", exp * 1.001, exp, 1e-4)["ok"]


def test_partials_rows_and_router_selection():
    rng = np.random.default_rng(4)
    p = K.random_partials(rng, 1023, 32, 33)                     # step 1023: split 32 is empty
    assert np.isneginf(p[32, :, 512]).all() and np.isfinite(p[31, :, 512]).all()
    rows = K.rows_partials("p", p, p, K.PARTIALS_O_REL)
    assert [r["ok"] for r in rows] == [True, True, True]
    bad = p.copy()
    bad[32, 0, 512] = 0.0                                        # an empty split given a finite lse
    assert not K.rows_partials("p", bad, p, K.PARTIALS_O_REL)[1]["ok"]
    # the selection from given logits is numpy_ref's own top-k
    t, prm = K.make_moe_router(rng)
    exp = K.ref_moe_router(t, prm)
    w, routing, mask = K.router_selection(exp["logits"], prm)
    assert np.array_equal(mask, exp["mask"]) and np.array_equal(routing, exp["routing"])
    assert np.allclose(w, exp["topk_w"], rtol=1e-6)
    rows = K.check_moe_router(t, prm, exp, dict(exp))
    assert all(r["ok"] for r in rows) and [r["output"] for r in rows][-1] == "topk ids vs reference logits"
    got = dict(exp)
    got["route_log"] = exp["route_log"].copy()
    got["route_log"][0, 0, 0] = 3                                # a write outside the slot
    assert not [r for r in K.check_moe_router(t, prm, exp, got) if r["output"] == "route_log"][0]["ok"]


# ----------------------------------------------------------------------------
# the dry-run path


def test_dry_run_all_tests(tmp_path):
    out = tmp_path / "kernel_tests.json"
    rc = K.main(["--dry-run", "--n", "2", "--work-dir", str(tmp_path / "work"), "--out", str(out)])
    assert rc == 0
    res = json.loads(out.read_text())
    assert res["dry_run"] and res["overall"] == "PASS"
    assert set(res["tests"]) == set(K.TESTS)
    for name, t in res["tests"].items():
        assert t["result"] == "PASS" and t["trials"] == 2, name
        if name != "mla_attend_splits":
            for o, m in t["outputs"].items():
                assert m["max_abs_err"] == 0.0 and m["rel_err"] == 0.0, (name, o)
                assert abs(m["cos_sim"] - 1.0) < 1e-12, (name, o)      # FP64 rounding of the norms
        for f in t["failures"]:
            assert "cos_sim" in f
    # every row carries compare.py's three metrics
    row = K.row_rel("r", np.ones(4), np.ones(4), 1e-4)
    assert {"max_abs_err", "rel_err", "cos_sim"} <= set(row) and row["cos_sim"] == 1.0
    splits = res["tests"]["mla_attend_splits"]["outputs"]
    assert 0 < splits["attn: 33 splits vs 1 split"]["rel_err"] < K.SPLITS_REL
    # the trial directories are what the binary would read
    d = tmp_path / "work/mla_attend_splits/000"
    assert {p.name for p in d.iterdir()} == {"attend33", "attend1", "merge33", "merge1"}
    assert K.read_params(d / "attend1/params.txt")["n_splits"] == 1
    assert (d / "attend33/partials.out.bin").stat().st_size == 33 * 16 * 513 * 4


def test_dry_run_cli_single_kernel(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "fleet/tasks/kernel_tests.py"), "--dry-run", "--n", "1",
                        "--kernel", "copy", "--out", str(tmp_path / "r.json")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "PASS copy (1 trials, 0 failed)" in r.stdout and "overall: PASS" in r.stdout
    assert list(json.loads((tmp_path / "r.json").read_text())["tests"]) == ["copy"]


def test_missing_binary_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        K.main(["--n", "1", "--kernel", "copy", "--bin", str(tmp_path / "nope"), "--out", str(tmp_path / "r.json")])
