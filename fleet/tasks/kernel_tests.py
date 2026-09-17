#!/usr/bin/env python3
"""Kernel tests: each new MI300 kernel in isolation, random inputs, against numpy_ref.py.

    python fleet/tasks/kernel_tests.py [--n 100] [--seed 0] [--kernel NAME ...]
        [--bin fleet/tasks/build/kernel_tests] [--bin-debug fleet/tasks/build/kernel_tests_debug]
        [--dry-run] [--work-dir DIR] [--keep] [--out fleet/tasks/results/kernel_tests.json]
    (--dry-run writes fleet/tasks/results/kernel_tests_dryrun.json, which is gitignored)

The kernel_tests.py row of docs/design-doc/07-correctness.md. The binary is
fleet/tasks/kernel_tests_mi300.cu (build line in its header); it launches one
kernel per trial directory exactly as the runtime's emitted call does. This
driver generates the inputs at the real shapes with a fixed seed, runs the
binary once per test over all trial directories, reads the outputs back and
compares them with harness/numpy_ref.py output by output.

Tests (--kernel selects; default all):
  mla_prep, mla_attend, mla_merge_uv, moe_router, copy, prefetch, prefetch_moe
      n random trials each, one launch per trial (the two prefetch suites, O8: the XOR of the
      streamed slice per wave, so the stripe and the expert (slot, part) indexing are exact)
  mla_attend_scores   the -DMLA_ATTEND_DEBUG_SCORES build's second output (boundary B5)
  mla_attend_splits   one split of 1056 rows versus 33 splits of 32 rows through
                      both the attend and the merge kernel (isolates the merge)

--dry-run stands in for the binary: the reference outputs are written to the
.out.bin files instead, so the plumbing (file layout, dtypes, shapes, the
comparison) runs here without a GPU; every single-kernel test then reports
zero error, and mla_attend_splits reports the genuine difference between the
two NumPy paths.

File contract with the binary (per trial directory): params.txt with
"name value" lines (floats as IEEE-754 bit patterns, as register_task takes
them), one raw little-endian <name>.bin per tensor of the test's table
(BF16 as uint16, FP32, int32); the kernel's outputs are uploaded from their
.bin too (pre-filled here with a sentinel: NaN or -7) and come back as
<name>.out.bin, so entries the kernel must not touch are checked as well.

Activations are unit-scale BF16, weights BF16 at scale 1 / sqrt(fan_in),
step uniform in 1023..1054 (the 32 decode positions of the run), the RoPE
tables cos / sin of the plain RoPE formula (the real tables are captured
from the model; the values have the same range).
"""
import argparse
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "harness"))
import common  # noqa: E402
import numpy_ref as R  # noqa: E402
from fleet import graph_plan as G  # noqa: E402
from fleet.pack_weights import REAL_DIMS, XCDS  # noqa: E402

F32 = np.float32
D = REAL_DIMS
S_MAX = common.S_MAX                                    # 1056
STEP_MIN, STEP_MAX = common.HANDOVER, common.HANDOVER + common.N_STEPS - 1   # 1023 .. 1054
SPLIT = G.SPLIT                                          # 32
N_SPLITS = -(-S_MAX // SPLIT)                            # 33
TILES_PER_XCD = -(-N_SPLITS // XCDS)                     # 5
QKVA = D.Q_OUT + D.KVA_OUT                               # 3648
N_SLOTS = D.TOPK + G.N_FORCED                            # 8
N_TOTAL = D.E + G.N_FORCED                               # 66
ROUTE_SHAPE = (common.N_STEPS, D.L - 1, G.TOPK_TOTAL_SLOTS)   # [32, 26, 8] (graph_plan.py)
FORCED = tuple(range(D.E, D.E + G.N_FORCED))

DEFAULT_BIN = ROOT / "fleet/tasks/build/kernel_tests"
DEFAULT_BIN_DEBUG = ROOT / "fleet/tasks/build/kernel_tests_debug"
DEFAULT_OUT = ROOT / "fleet/tasks/results/kernel_tests.json"
DEFAULT_OUT_DRY = ROOT / "fleet/tasks/results/kernel_tests_dryrun.json"     # gitignored
BUILD_HINT = ("build it from the repository root with the line in its header, starting with "
              "'mkdir -p fleet/tasks/build && hipcc ...' (fleet/tasks/README.md), or use --dry-run")

# Tolerances, on the metrics of harness/compare.py (rel_err = ||a - b|| / ||b||).
#
# exact  Bit-exact wherever numpy_ref's arithmetic is order-independent: the
#        RoPE outputs (each rounding follows one FP32 multiply or add of BF16
#        values, so the bits are the same on any IEEE machine), the router's
#        top-k ids, routing, mask and route log (integer selection; the near-tie
#        case is handled in check_moe_router), the copy, and every cache row
#        or buffer entry the kernel must leave untouched.
# f32    FP32 outputs of FP32 accumulations (router logits, debug scores). The
#        kernel sums in another order than NumPy; the rounding error of a
#        K-term sum with cancellation (random signs, K = 576 or 2048, terms
#        ~ 1 / sqrt(K)) is ~ 2e-6 relative per element in either order, so
#        the difference is a few 1e-6 and 1e-4 leaves a decade of margin
#        while any indexing error is O(1).
# bf16   BF16-stored outputs of FP32 accumulations (c_kv row, ql_nope, attn).
#        The FP32 sums differ by ~1e-6 relative, which flips the final BF16
#        rounding of an element with probability ~1e-6 / 2^-8 = 3e-4; a flip
#        is exactly one BF16 ulp of that element, at most 2^-7 |v|, so the
#        bound is per element: |got - ref| <= 2^-7 |ref| + 1e-5 max|ref|.
#        The absolute floor is for near-zero elements, where the FP32 noise
#        (absolute, ~1e-6 of the output scale) exceeds their own ulp; it stays
#        below the ulp of any element above 1e-2 of the array maximum. A
#        handful of flips among 512..8192 elements keeps rel_err near 1e-4,
#        hence rel_err <= 2e-3.
# partials  Both operands of the score dot are BF16, so every product is
#        exact in FP32 and the only difference to NumPy is the order of the
#        576-term sum: ~1e-6 relative on a score, hence on exp(s - m). The
#        probabilities are rounded to BF16 in the kernel and the reference
#        alike, and a rounding flips only when the FP32 value lies within that
#        1e-6 of a boundary, so flips are rare and a flip moves one split-head
#        by 2^-8 q_i, diluted over the [n_splits, 16, 512] array. An emulation
#        of the kernel's sequential order measured rel_err 1.1e-6 on o and
#        3.8e-6 absolute on lse (review of this harness); 5e-4 on o is two
#        decades above that and three below an indexing error. A FAIL with
#        max_abs_err ~1e-3 on a single split-head and rel_err just above the
#        bound is a probability flip on a dominant position, not an indexing
#        error. lse = m + ln(l) sums the unrounded probabilities: 1e-4 absolute.
# splits The one-split kernel rounds every probability at its running maximum
#        of the pass, the 33-split path and NumPy's single pass at the final
#        one, so every probability may differ by one BF16 ulp: 2^-9 relative
#        on o in quadrature. 1e-2 is the bound test_numpy_ref.py uses for the
#        same comparison between the two NumPy paths.
F32_REL = 1e-4
BF16_REL = 2e-3
BF16_ULP = 2.0 ** -7     # one BF16 ulp of v is at most 2^-7 |v|
BF16_ABS_FLOOR = BF16_ULP   # one ulp of the largest element: the cancellation bound (row_bf16)
PARTIALS_O_REL = 5e-4
P_ROW = ((D.D_C + 1 + 3) // 4) * 4     # padded partials row (P2): 513 -> 516, 16-byte-aligned rows


def pad_partials(logical):
    """Logical [..., D_C+1] -> device [..., P_ROW], the extra columns zeroed as the kernel inits them."""
    pad = P_ROW - logical.shape[-1]
    if pad <= 0:
        return logical
    return np.concatenate([logical, np.zeros(logical.shape[:-1] + (pad,), logical.dtype)], axis=-1)
LSE_ABS = 1e-4
SPLITS_REL = 1e-2
NEAR_TIE = 1e-4          # logit gap below which the reference's own top-k is ambiguous


# ----------------------------------------------------------------------------
# serialization: raw little-endian files, BF16 as its 16 bits


FILE_DTYPE = {"bf16": np.dtype("<u2"), "f32": np.dtype("<f4"), "i32": np.dtype("<i4")}
SENTINEL = {"bf16": np.nan, "f32": np.nan, "i32": -7}


@dataclass(frozen=True)
class T:
    """One tensor of a kernel's table: name, file dtype, shape, written by the kernel."""
    name: str
    dtype: str
    shape: tuple
    out: bool = False


def to_file(x, dtype):
    if dtype == "bf16":
        # numpy_ref.bf16 leaves the low 16 bits zero, so the top half is the BF16 pattern
        return (R.bf16(x).view(np.uint32) >> 16).astype(FILE_DTYPE["bf16"])
    return np.ascontiguousarray(x, FILE_DTYPE[dtype])


def from_file(raw, dtype, shape):
    a = np.frombuffer(raw, FILE_DTYPE[dtype])
    if dtype == "bf16":
        a = (a.astype(np.uint32) << 16).view(F32)
    return a.reshape(shape)


def write_tensor(path, x, dtype):
    Path(path).write_bytes(to_file(x, dtype).tobytes())


def read_tensor(path, dtype, shape):
    return from_file(Path(path).read_bytes(), dtype, shape)


def write_params(path, params):
    Path(path).write_text("".join(f"{k} {int(v)}\n" for k, v in params.items()))


def read_params(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        k, v = line.split()
        out[k] = int(v)
    return out


def bits_to_float(bits):
    return struct.unpack("<f", struct.pack("<i", bits))[0]


# ----------------------------------------------------------------------------
# metrics and comparison rows


def metrics(a, b):
    """compare.py's metrics in NumPy: a = kernel, b = reference, FP64, flattened."""
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    if a.size != b.size:
        return {"shape_mismatch": [a.size, b.size]}
    d = a - b
    nb = float(np.linalg.norm(b))
    na = float(np.linalg.norm(a))
    max_abs = float(np.abs(d).max()) if d.size else 0.0
    rel = float(np.linalg.norm(d)) / nb if nb > 0 else (0.0 if max_abs == 0 else np.inf)
    cos = float(a @ b) / (na * nb) if na > 0 and nb > 0 else (1.0 if max_abs == 0 else 0.0)
    return {"max_abs_err": max_abs, "rel_err": rel, "cos_sim": cos, "n": int(a.size)}


def _row(output, kind, threshold, got, exp, ok, note=""):
    got, exp = np.asarray(got), np.asarray(exp)
    if np.issubdtype(exp.dtype, np.floating) and got.shape == exp.shape:
        # a sentinel the kernel correctly left in place is zero error, not NaN
        both = np.isnan(got) & np.isnan(exp)
        if both.any():
            got, exp = np.where(both, 0.0, got), np.where(both, 0.0, exp)
    m = metrics(got, exp)
    row = {"output": output, "kind": kind, "threshold": threshold, "ok": bool(ok)}
    if "shape_mismatch" in m:
        row.update(ok=False, note=f"shape mismatch {m['shape_mismatch']}")
        return row
    row.update(max_abs_err=m["max_abs_err"], rel_err=m["rel_err"], cos_sim=m["cos_sim"])
    if note:
        row["note"] = note
    return row


def row_exact(output, got, exp, note=""):
    """Bit equality; a NaN sentinel in the expectation (an entry the kernel must not
    touch) requires the sentinel to still be there, and a reference value never is NaN."""
    got, exp = np.asarray(got), np.asarray(exp)
    ok = got.shape == exp.shape and np.array_equal(got, exp, equal_nan=np.issubdtype(exp.dtype, np.floating))
    return _row(output, "exact", 0.0, got, exp, ok, note)


def row_rel(output, got, exp, threshold):
    m = metrics(got, exp)
    ok = "shape_mismatch" not in m and m["rel_err"] <= threshold
    return _row(output, "rel", threshold, got, exp, ok)


def row_abs(output, got, exp, threshold):
    m = metrics(got, exp)
    ok = "shape_mismatch" not in m and m["max_abs_err"] <= threshold
    return _row(output, "abs", threshold, got, exp, ok)


def row_bf16(output, got, exp, threshold=BF16_REL):
    """A BF16-stored FP32 accumulation: rel_err bound, and per element at most one
    BF16 ulp of that element plus one ulp of the tensor's largest element. The
    second term is the cancellation bound: a merge of 33 split-KV partials adds
    terms of magnitude max|ref| that cancel, so an element near zero can differ
    from an FP64 merge of the same partials by a rounding of those terms (seen
    on the VM 2026-09-15: max_abs 3.9e-3 at rel_err 5e-4, 10 of 100 trials
    with the earlier 1e-5 floor)."""
    m = metrics(got, exp)
    g, e = np.asarray(got, np.float64), np.asarray(exp, np.float64)
    within = False
    if "shape_mismatch" not in m:
        floor = BF16_ABS_FLOOR * float(np.abs(e).max()) if e.size else 0.0
        within = bool(np.all(np.abs(g - e) <= BF16_ULP * np.abs(e) + floor))   # NaN fails
    ok = within and m["rel_err"] <= threshold
    return _row(output, "bf16", threshold, got, exp, ok)


def rows_partials(prefix, got, exp, o_rel, d_c=None):
    """o with a rel_err bound, lse absolute on the live splits, the -inf pattern exact.
    d_c defaults to exp.shape[-1] - 1; pass it for a padded row (P2), where the width is P_ROW."""
    if d_c is None:
        d_c = exp.shape[-1] - 1
    rows = [row_rel(f"{prefix}.o", got[..., :d_c], exp[..., :d_c], o_rel)]
    lse_g, lse_e = got[..., d_c], exp[..., d_c]
    live = np.isfinite(lse_e)
    rows.append(row_exact(f"{prefix}.lse[empty splits]", np.isneginf(lse_g), np.isneginf(lse_e)))
    rows.append(row_abs(f"{prefix}.lse", lse_g[live], lse_e[live], LSE_ABS))
    return rows


# ----------------------------------------------------------------------------
# the kernels: tables, random inputs, references, checks


def bf16_normal(rng, shape, scale=1.0):
    return R.bf16(rng.standard_normal(shape, dtype=F32) * F32(scale))


def rope_tables(s_max=S_MAX, d_r=D.D_R, theta=10000.0):
    """cos / sin [s_max, d_r] BF16 of the HF layout cat(freqs, freqs), plain RoPE."""
    inv_freq = 1.0 / (theta ** (np.arange(0, d_r, 2, dtype=np.float64) / d_r))
    freqs = np.arange(s_max, dtype=np.float64)[:, None] * inv_freq[None, :]
    emb = np.concatenate([freqs, freqs], axis=-1)
    return R.bf16(np.cos(emb).astype(F32)), R.bf16(np.sin(emb).astype(F32))


def random_step(rng):
    return int(rng.integers(STEP_MIN, STEP_MAX + 1))


def sentinel(dtype, shape):
    return np.full(shape, SENTINEL[dtype], F32 if dtype != "i32" else np.int32)


@dataclass
class Kernel:
    """One binary test: the tensor table (the binary's order; out = written back),
    make(rng) -> (tensors, params), reference(tensors, params) -> expected full
    output arrays, check(tensors, params, expected, got) -> rows."""
    name: str
    tensors: callable            # params -> [T]
    make: callable
    reference: callable
    check: callable


# --- mla_prep ---------------------------------------------------------------

def tensors_mla_prep(params):
    return [T("qkva", "bf16", (QKVA,)), T("w_kv_norm", "bf16", (D.D_C,)),
            T("w_uk", "bf16", (D.NH, D.D_N, D.D_C)), T("cos", "bf16", (S_MAX, D.D_R)),
            T("sin", "bf16", (S_MAX, D.D_R)),
            T("c_kv", "bf16", (S_MAX, D.D_C), True), T("k_pe", "bf16", (S_MAX, D.D_R), True),
            T("ql_nope", "bf16", (D.NH, D.D_C), True), T("q_pe", "bf16", (D.NH, D.D_R), True)]


def make_mla_prep(rng):
    cos, sin = rope_tables()
    t = {"qkva": bf16_normal(rng, (QKVA,)),
         "w_kv_norm": R.bf16(1.0 + 0.1 * rng.standard_normal(D.D_C, dtype=F32)),
         "w_uk": bf16_normal(rng, (D.NH, D.D_N, D.D_C), D.D_N ** -0.5),
         "cos": cos, "sin": sin,
         # the cache holds other rows; only row step may change
         "c_kv": bf16_normal(rng, (S_MAX, D.D_C)), "k_pe": bf16_normal(rng, (S_MAX, D.D_R)),
         "ql_nope": sentinel("bf16", (D.NH, D.D_C)), "q_pe": sentinel("bf16", (D.NH, D.D_R))}
    return t, {"step": random_step(rng)}


def ref_mla_prep(t, p):
    step = p["step"]
    c_row, k_row, ql_nope, q_pe = R.mla_prep(t["qkva"], t["w_kv_norm"], t["w_uk"], t["cos"][step],
                                             t["sin"][step], nh=D.NH, d_n=D.D_N, d_r=D.D_R, d_c=D.D_C)
    c_kv, k_pe = t["c_kv"].copy(), t["k_pe"].copy()
    c_kv[step], k_pe[step] = c_row, k_row
    return {"c_kv": c_kv, "k_pe": k_pe, "ql_nope": ql_nope, "q_pe": q_pe}


def check_mla_prep(t, p, exp, got):
    step = p["step"]
    others = np.arange(S_MAX) != step
    return [row_bf16("c_kv[step]", got["c_kv"][step], exp["c_kv"][step]),
            row_exact("c_kv[other rows]", got["c_kv"][others], exp["c_kv"][others]),
            row_exact("k_pe", got["k_pe"], exp["k_pe"]),
            row_bf16("ql_nope", got["ql_nope"], exp["ql_nope"]),
            row_exact("q_pe", got["q_pe"], exp["q_pe"])]


# --- mla_attend -------------------------------------------------------------

def tensors_mla_attend(params):
    n_splits = params["n_splits"]
    t = [T("ql_nope", "bf16", (D.NH, D.D_C)), T("q_pe", "bf16", (D.NH, D.D_R)),
         T("c_kv", "bf16", (S_MAX, D.D_C)), T("k_pe", "bf16", (S_MAX, D.D_R)),
         T("partials", "f32", (n_splits, D.NH, P_ROW), True)]
    if params.get("debug_scores"):
        t.append(T("scores", "f32", (D.NH, S_MAX), True))
    return t


def attend_params(step, split=SPLIT, n_splits=N_SPLITS, debug_scores=False):
    """The registration's parameters for one launch (build_graph.py mla_attend_layer)."""
    p = {"step": step, "split": split, "n_splits": n_splits, "tiles_per_xcd": -(-n_splits // XCDS),
         "softmax_scale_bits": G.float_bits(G.SOFTMAX_SCALE)}
    if debug_scores:
        p["debug_scores"] = 1
    return p


def make_mla_attend(rng, **kw):
    p = attend_params(random_step(rng), **kw)
    t = {"ql_nope": bf16_normal(rng, (D.NH, D.D_C)), "q_pe": bf16_normal(rng, (D.NH, D.D_R)),
         "c_kv": bf16_normal(rng, (S_MAX, D.D_C)), "k_pe": bf16_normal(rng, (S_MAX, D.D_R)),
         "partials": sentinel("f32", (p["n_splits"], D.NH, P_ROW))}
    if p.get("debug_scores"):
        t["scores"] = sentinel("f32", (D.NH, S_MAX))
    return t, p


def make_mla_attend_scores(rng):
    return make_mla_attend(rng, debug_scores=True)


def ref_mla_attend(t, p):
    step, scale = p["step"], bits_to_float(p["softmax_scale_bits"])
    debug = bool(p.get("debug_scores"))
    out = R.mla_attend(t["ql_nope"], t["q_pe"], t["c_kv"], t["k_pe"], step, scale,
                       split=p["split"], n_splits=p["n_splits"], debug_scores=debug)
    if not debug:
        return {"partials": pad_partials(out)}
    partials, scores = out
    exp_scores = t["scores"].copy()                    # columns beyond step stay untouched
    exp_scores[:, :step + 1] = scores
    return {"partials": pad_partials(partials), "scores": exp_scores}


def check_mla_attend(t, p, exp, got):
    rows = rows_partials("partials", got["partials"], exp["partials"], PARTIALS_O_REL, d_c=D.D_C)
    if p.get("debug_scores"):
        s = p["step"] + 1
        rows.append(row_rel("scores[:, :step+1]", got["scores"][:, :s], exp["scores"][:, :s], F32_REL))
        rows.append(row_exact("scores[:, step+1:]", got["scores"][:, s:], exp["scores"][:, s:]))
    return rows


# --- mla_merge_uv -----------------------------------------------------------

def tensors_mla_merge_uv(params):
    return [T("partials", "f32", (params["n_splits"], D.NH, P_ROW)),
            T("w_uv", "bf16", (D.NH, D.D_V, D.D_C)), T("attn", "bf16", (D.NH * D.D_V,), True)]


def merge_params(step, split=SPLIT, n_splits=N_SPLITS):
    return {"step": step, "split": split, "n_splits": n_splits}


def random_partials(rng, step, split, n_splits):
    """CK-convention partials: normalized o and lse per live split, o = 0 and lse = -inf beyond."""
    partials = rng.standard_normal((n_splits, D.NH, D.D_C + 1), dtype=F32)
    partials[:, :, D.D_C] *= 2.0
    live = -(-(step + 1) // split)
    partials[live:, :, :D.D_C] = 0.0
    partials[live:, :, D.D_C] = -np.inf
    return partials


def make_mla_merge_uv(rng):
    p = merge_params(random_step(rng))
    t = {"partials": pad_partials(random_partials(rng, p["step"], p["split"], p["n_splits"])),
         "w_uv": bf16_normal(rng, (D.NH, D.D_V, D.D_C), D.D_C ** -0.5),
         "attn": sentinel("bf16", (D.NH * D.D_V,))}
    return t, p


def ref_mla_merge_uv(t, p):
    return {"attn": R.mla_merge_uv(t["partials"], t["w_uv"], p["step"], split=p["split"], d_c=D.D_C)}


def check_mla_merge_uv(t, p, exp, got):
    return [row_bf16("attn", got["attn"], exp["attn"])]


# --- moe_router -------------------------------------------------------------

def tensors_moe_router(params):
    # the fused form (O1, docs/gpu-experiments/03-acceleration): the norm folded in, NORM = true
    return [T("x_res", "bf16", (D.H,)), T("w_norm", "bf16", (D.H,)), T("w_gate", "bf16", (D.E, D.H)),
            T("h", "bf16", (D.H,), True),
            T("topk_w", "f32", (N_SLOTS,), True), T("routing", "i32", (N_TOTAL,), True),
            T("mask", "i32", (N_TOTAL + 1,), True), T("logits", "f32", (D.E,), True),
            T("route_log", "i32", ROUTE_SHAPE, True)]


def make_moe_router(rng):
    p = {"step": random_step(rng), "prompt_length": common.N_PROMPT,
         "layer_index": int(rng.integers(0, ROUTE_SHAPE[1])),
         "scaling_bits": G.float_bits(G.ROUTED_SCALING), "eps_bits": G.float_bits(G.RMS_EPS)}
    t = {"x_res": bf16_normal(rng, (D.H,), 4.0), "w_norm": bf16_normal(rng, (D.H,)),
         "w_gate": bf16_normal(rng, (D.E, D.H), D.H ** -0.5),
         "h": sentinel("bf16", (D.H,)),
         "topk_w": sentinel("f32", (N_SLOTS,)), "routing": sentinel("i32", (N_TOTAL,)),
         "mask": sentinel("i32", (N_TOTAL + 1,)), "logits": sentinel("f32", (D.E,)),
         "route_log": np.full(ROUTE_SHAPE, -1, np.int32)}
    return t, p


def router_selection(logits, p):
    """topk_w, routing, mask from given logits: numpy_ref.moe_router with the identity
    as W_gate reproduces its softmax and top-k on the logits exactly."""
    _, topk_w, routing, mask = R.moe_router(np.asarray(logits, F32), np.eye(D.E, dtype=F32),
                                            topk=D.TOPK, n_experts=D.E, forced=FORCED,
                                            scaling=bits_to_float(p["scaling_bits"]))
    return topk_w, routing, mask


def route_log_expected(initial, p, mask):
    log = initial.copy()
    log[p["step"] - (p["prompt_length"] - 1), p["layer_index"]] = mask[:N_SLOTS]
    return log


def ref_moe_router(t, p):
    h, logits, topk_w, routing, mask = R.moe_router_norm(
        t["x_res"], t["w_norm"], t["w_gate"], eps=bits_to_float(p["eps_bits"]), topk=D.TOPK,
        n_experts=D.E, forced=FORCED, scaling=bits_to_float(p["scaling_bits"]))
    return {"h": h, "topk_w": topk_w, "routing": routing, "mask": mask, "logits": logits,
            "route_log": route_log_expected(t["route_log"], p, mask)}


def check_moe_router(t, p, exp, got):
    """The GEMV with a tolerance; the selection exactly, derived from the kernel's own
    logits so that a near tie at the FP32 noise level cannot fail the exact checks."""
    rows = [row_rel("h", got["h"], exp["h"], BF16_REL),          # the folded norm (O1)
            row_rel("logits", got["logits"], exp["logits"], F32_REL)]
    if not np.all(np.isfinite(got["logits"])):
        rows.append(row_exact("selection", got["mask"], exp["mask"], note="logits not finite"))
        return rows
    topk_w, routing, mask = router_selection(got["logits"], p)
    rows += [row_rel("topk_w", got["topk_w"], topk_w, F32_REL),
             row_exact("routing", got["routing"], routing),
             row_exact("mask", got["mask"], mask),
             row_exact("route_log", got["route_log"], route_log_expected(t["route_log"], p, mask))]
    same = set(got["mask"][:D.TOPK].tolist()) == set(exp["mask"][:D.TOPK].tolist())
    top = np.sort(exp["logits"])[::-1]
    gap = float(top[D.TOPK - 1] - top[D.TOPK])
    note = "" if same else (f"near tie, reference logit gap {gap:.2e}" if gap < NEAR_TIE
                            else f"ids differ with a logit gap of {gap:.2e}")
    r = row_exact("topk ids vs reference logits", np.sort(got["mask"][:D.TOPK]),
                  np.sort(exp["mask"][:D.TOPK]), note)
    r["ok"] = bool(same or gap < NEAR_TIE)
    rows.append(r)
    return rows


# --- prefetch (O8) ----------------------------------------------------------

PF_GRID, PF_ROWS = 4, 32          # the launcher's constants: a W_o-like [128, 2048] in 4 stripes
PF_N, PF_K, PF_PARTS = 32, 256, 2  # an expert weight [N_TOTAL, 32, 256], each active expert in 2 parts


def xor_words_per_wave(slice_bf16):
    """What prefetch_mi300.cuh's stream_chunks writes: the slice as 16-byte chunks, chunk c to thread
    c mod 256, the XOR of every 32-bit word of a wave's chunks in dummy[wave] (int32)."""
    words = np.ascontiguousarray(to_file(slice_bf16, "bf16")).reshape(-1).view(np.uint32)
    chunks = words.reshape(-1, 4)
    out = np.zeros(4, np.uint32)
    for c in range(chunks.shape[0]):
        w = (c % 256) // 64
        out[w] ^= np.bitwise_xor.reduce(chunks[c])
    return out.view(np.int32)


def tensors_prefetch(params):
    return [T("w", "bf16", (PF_GRID * PF_ROWS, D.H)), T("dummy", "i32", (PF_GRID, 4), True)]


def make_prefetch(rng):
    return {"w": bf16_normal(rng, (PF_GRID * PF_ROWS, D.H)), "dummy": sentinel("i32", (PF_GRID, 4))}, {}


def ref_prefetch(t, p):
    dummy = np.stack([xor_words_per_wave(t["w"][b * PF_ROWS:(b + 1) * PF_ROWS]) for b in range(PF_GRID)])
    return {"dummy": dummy}


def check_prefetch(t, p, exp, got):
    return [row_exact("dummy", got["dummy"], exp["dummy"], "the XOR of every word of the stripe, per wave")]


def tensors_prefetch_moe(params):
    return [T("w", "bf16", (N_TOTAL, PF_N, PF_K)), T("mask", "i32", (N_TOTAL + 1,)),
            T("dummy", "i32", (N_SLOTS * PF_PARTS, 4), True)]


def make_prefetch_moe(rng):
    n_active = int(rng.integers(1, N_SLOTS + 1))
    ids = rng.choice(N_TOTAL, size=n_active, replace=False)
    mask = np.full(N_TOTAL + 1, -1, np.int32)
    mask[:n_active] = ids
    mask[N_TOTAL] = n_active
    return {"w": bf16_normal(rng, (N_TOTAL, PF_N, PF_K)), "mask": mask,
            "dummy": sentinel("i32", (N_SLOTS * PF_PARTS, 4))}, {}


def ref_prefetch_moe(t, p):
    mask = t["mask"]
    n_active = int(mask[N_TOTAL])
    dummy = sentinel("i32", (N_SLOTS * PF_PARTS, 4))     # a slot past the active count leaves its row untouched
    rows = PF_N // PF_PARTS
    for b in range(N_SLOTS * PF_PARTS):
        slot, part = b // PF_PARTS, b % PF_PARTS
        if slot < n_active:
            e = int(mask[slot])
            dummy[b] = xor_words_per_wave(t["w"][e, part * rows:(part + 1) * rows])
    return {"dummy": dummy}


def check_prefetch_moe(t, p, exp, got):
    return [row_exact("dummy", got["dummy"], exp["dummy"], "per (slot, part): the active expert's rows; sentinel past the count")]


# --- copy -------------------------------------------------------------------

def tensors_copy(params):
    return [T("x", "bf16", (D.H,)), T("y", "bf16", (D.H,), True)]


def make_copy(rng):
    return {"x": bf16_normal(rng, (D.H,)), "y": sentinel("bf16", (D.H,))}, {}


def ref_copy(t, p):
    return {"y": t["x"].copy()}


def check_copy(t, p, exp, got):
    return [row_exact("y", got["y"], exp["y"])]


KERNELS = {
    "mla_prep": Kernel("mla_prep", tensors_mla_prep, make_mla_prep, ref_mla_prep, check_mla_prep),
    "mla_attend": Kernel("mla_attend", tensors_mla_attend, make_mla_attend, ref_mla_attend, check_mla_attend),
    "mla_merge_uv": Kernel("mla_merge_uv", tensors_mla_merge_uv, make_mla_merge_uv, ref_mla_merge_uv,
                           check_mla_merge_uv),
    "moe_router": Kernel("moe_router", tensors_moe_router, make_moe_router, ref_moe_router, check_moe_router),
    "copy": Kernel("copy", tensors_copy, make_copy, ref_copy, check_copy),
    "prefetch": Kernel("prefetch", tensors_prefetch, make_prefetch, ref_prefetch, check_prefetch),
    "prefetch_moe": Kernel("prefetch_moe", tensors_prefetch_moe, make_prefetch_moe, ref_prefetch_moe, check_prefetch_moe),
}


# ----------------------------------------------------------------------------
# trial directories and the binary


def write_trial(d, kernel, tensors, params):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    write_params(d / "params.txt", params)
    for spec in kernel.tensors(params):
        x = tensors[spec.name]
        assert tuple(x.shape) == spec.shape, (spec.name, x.shape, spec.shape)
        write_tensor(d / f"{spec.name}.bin", x, spec.dtype)


def read_trial(d, kernel):
    """(tensors as written, params, outputs the binary wrote back)."""
    d = Path(d)
    params = read_params(d / "params.txt")
    tensors, got = {}, {}
    for spec in kernel.tensors(params):
        tensors[spec.name] = read_tensor(d / f"{spec.name}.bin", spec.dtype, spec.shape)
        if spec.out:
            got[spec.name] = read_tensor(d / f"{spec.name}.out.bin", spec.dtype, spec.shape)
    return tensors, params, got


def fake_binary(kernel, dirs):
    """--dry-run: the reference outputs stand in for the kernel's."""
    for d in dirs:
        params = read_params(Path(d) / "params.txt")
        specs = kernel.tensors(params)
        tensors = {s.name: read_tensor(Path(d) / f"{s.name}.bin", s.dtype, s.shape) for s in specs}
        exp = kernel.reference(tensors, params)
        for s in specs:
            if s.out:
                write_tensor(Path(d) / f"{s.name}.out.bin", exp[s.name], s.dtype)


def run_binary(binary, kernel, dirs):
    if binary is None:
        fake_binary(kernel, dirs)
        return
    cmd = [str(binary), kernel.name] + [str(d) for d in dirs]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        tail = (r.stdout + r.stderr).strip().splitlines()[-10:]
        raise RuntimeError(f"{binary} {kernel.name} failed with status {r.returncode}:\n" + "\n".join(tail))


# ----------------------------------------------------------------------------
# the tests


@dataclass
class Ctx:
    binary: object                # Path, or None for --dry-run
    binary_debug: object
    work: Path
    n: int
    seed: int
    results: dict = field(default_factory=dict)


def summarize(name, trial_rows):
    """Per-output worst case over the trials, the failing trials, PASS/FAIL."""
    outputs = {}
    failures = []
    for i, rows in enumerate(trial_rows):
        for r in rows:
            o = outputs.setdefault(r["output"], {"kind": r["kind"], "threshold": r["threshold"],
                                                 "max_abs_err": 0.0, "rel_err": 0.0, "cos_sim": 1.0,
                                                 "fails": 0, "notes": []})
            for k in ("max_abs_err", "rel_err"):
                if k in r and not (r[k] <= o[k]):          # NaN counts as worse
                    o[k] = r[k]
            if "cos_sim" in r and not (r["cos_sim"] >= o["cos_sim"]):   # the worst cosine, same rule
                o["cos_sim"] = r["cos_sim"]
            if not r["ok"]:
                o["fails"] += 1
                if len(failures) < 20:
                    failures.append({"trial": i, **r})
            if r.get("note") and len(o["notes"]) < 5:
                o["notes"].append(f"trial {i}: {r['note']}")
    failed_trials = sum(1 for rows in trial_rows if not all(r["ok"] for r in rows))
    return {"result": "PASS" if failed_trials == 0 else "FAIL", "trials": len(trial_rows),
            "failed_trials": failed_trials, "outputs": outputs, "failures": failures}


def trial_rng(ctx, name):
    """Per-test stream of the seed; crc32 rather than hash(), which is salted per process."""
    return np.random.default_rng([ctx.seed, zlib.crc32(name.encode())])


def run_single(ctx, name, kernel, make=None, binary="bin"):
    """n trials of one kernel: write all, run the binary once, check each."""
    rng = trial_rng(ctx, name)
    make = make or kernel.make
    dirs = []
    for i in range(ctx.n):
        d = ctx.work / name / f"{i:03d}"
        tensors, params = make(rng)
        write_trial(d, kernel, tensors, params)
        dirs.append(d)
    run_binary(getattr(ctx, "binary" if binary == "bin" else "binary_debug"), kernel, dirs)
    rows = []
    for d in dirs:
        tensors, params, got = read_trial(d, kernel)
        rows.append(kernel.check(tensors, params, kernel.reference(tensors, params), got))
    return summarize(name, rows)


def run_splits(ctx, name):
    """One split of S_max rows versus 33 splits of 32, both through the attend kernel
    and then the merge kernel, on the same inputs."""
    rng = trial_rng(ctx, name)
    att, mrg = KERNELS["mla_attend"], KERNELS["mla_merge_uv"]
    trials = []
    for i in range(ctx.n):
        base = ctx.work / name / f"{i:03d}"
        tensors, p33 = make_mla_attend(rng)
        p1 = attend_params(p33["step"], split=S_MAX, n_splits=1)
        t1 = dict(tensors, partials=sentinel("f32", (1, D.NH, P_ROW)))
        write_trial(base / "attend33", att, tensors, p33)
        write_trial(base / "attend1", att, t1, p1)
        w_uv = bf16_normal(rng, (D.NH, D.D_V, D.D_C), D.D_C ** -0.5)
        trials.append((base, p33, p1, w_uv))
    run_binary(ctx.binary, att, [b / "attend33" for b, *_ in trials] + [b / "attend1" for b, *_ in trials])
    for base, p33, p1, w_uv in trials:
        for sub, p in (("33", p33), ("1", p1)):
            _, _, got = read_trial(base / f"attend{sub}", att)
            write_trial(base / f"merge{sub}", mrg,
                        {"partials": got["partials"], "w_uv": w_uv, "attn": sentinel("bf16", (D.NH * D.D_V,))},
                        merge_params(p["step"], p["split"], p["n_splits"]))
    run_binary(ctx.binary, mrg, [b / "merge33" for b, *_ in trials] + [b / "merge1" for b, *_ in trials])
    rows = []
    for base, p33, p1, w_uv in trials:
        t33, _, g33 = read_trial(base / "attend33", att)
        _, _, g1 = read_trial(base / "attend1", att)
        _, _, m33 = read_trial(base / "merge33", mrg)
        _, _, m1 = read_trial(base / "merge1", mrg)
        step = p33["step"]
        # the 33 kernel splits merged in NumPy against the kernel's single split
        merged = R.merge_partials(g33["partials"], step, split=SPLIT, d_c=D.D_C)
        lse33 = g33["partials"][:, :, D.D_C]
        live = -(-(step + 1) // SPLIT)
        M = lse33[:live].max(axis=0)
        lse_merged = M + np.log(np.exp(lse33[:live] - M[None, :], dtype=F32).sum(axis=0, dtype=F32))
        one = g1["partials"][0]
        r = [row_rel("o: 33 splits merged vs 1 split", merged, one[:, :D.D_C], SPLITS_REL),
             row_abs("lse: 33 splits merged vs 1 split", lse_merged, one[:, D.D_C], LSE_ABS)]
        # each path against its own NumPy reference
        r += rows_partials("1 split vs numpy_ref", g1["partials"], att.reference(t33, p1)["partials"], SPLITS_REL, d_c=D.D_C)
        r.append(row_bf16("attn(33 splits) vs numpy_ref merge of the kernel's partials", m33["attn"],
                          R.mla_merge_uv(g33["partials"], w_uv, step, split=SPLIT, d_c=D.D_C)))
        r.append(row_bf16("attn(1 split) vs numpy_ref merge of the kernel's partials", m1["attn"],
                          R.mla_merge_uv(g1["partials"], w_uv, step, split=S_MAX, d_c=D.D_C)))
        # the two kernel paths end to end
        r.append(row_rel("attn: 33 splits vs 1 split", m33["attn"], m1["attn"], SPLITS_REL))
        rows.append(r)
    return summarize(name, rows)


TESTS = {
    "mla_prep": lambda ctx: run_single(ctx, "mla_prep", KERNELS["mla_prep"]),
    "mla_attend": lambda ctx: run_single(ctx, "mla_attend", KERNELS["mla_attend"]),
    "mla_merge_uv": lambda ctx: run_single(ctx, "mla_merge_uv", KERNELS["mla_merge_uv"]),
    "moe_router": lambda ctx: run_single(ctx, "moe_router", KERNELS["moe_router"]),
    "copy": lambda ctx: run_single(ctx, "copy", KERNELS["copy"]),
    "prefetch": lambda ctx: run_single(ctx, "prefetch", KERNELS["prefetch"]),
    "prefetch_moe": lambda ctx: run_single(ctx, "prefetch_moe", KERNELS["prefetch_moe"]),
    "mla_attend_scores": lambda ctx: run_single(ctx, "mla_attend_scores", KERNELS["mla_attend"],
                                                make=make_mla_attend_scores, binary="debug"),
    "mla_attend_splits": lambda ctx: run_splits(ctx, "mla_attend_splits"),
}
NEEDS_DEBUG_BINARY = {"mla_attend_scores"}


# ----------------------------------------------------------------------------


def fmt(x):
    return "-" if x is None else (f"{x:.2e}" if isinstance(x, float) else str(x))


def print_summary(name, s):
    for o, m in s["outputs"].items():
        flag = "PASS" if m["fails"] == 0 else "FAIL"
        print(f"  {flag:4s} {o:58s} {m['kind']:5s} max_abs {fmt(m['max_abs_err'])} rel {fmt(m['rel_err'])} "
              f"thr {fmt(m['threshold'])}" + (f"  fails {m['fails']}" if m["fails"] else ""))
        for note in m["notes"]:
            print(f"       {note}")
    print(f"{s['result']} {name} ({s['trials']} trials, {s['failed_trials']} failed)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=100, help="random trials per kernel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kernel", action="append", choices=sorted(TESTS), help="test to run (default: all)")
    ap.add_argument("--bin", default=str(DEFAULT_BIN), help="the compiled kernel_tests_mi300.cu")
    ap.add_argument("--bin-debug", default=str(DEFAULT_BIN_DEBUG), help="the -DMLA_ATTEND_DEBUG_SCORES build")
    ap.add_argument("--dry-run", action="store_true", help="no binary: the references stand in for the kernels")
    ap.add_argument("--work-dir", default=None, help="trial directories (default: a temporary directory)")
    ap.add_argument("--keep", action="store_true", help="keep the trial directories")
    ap.add_argument("--out", default=None,
                    help=f"results JSON (default: {DEFAULT_OUT.relative_to(ROOT)}, "
                         f"{DEFAULT_OUT_DRY.relative_to(ROOT)} for --dry-run)")
    args = ap.parse_args(argv)

    names = args.kernel or list(TESTS)
    binary = binary_debug = None
    if not args.dry_run:
        binary, binary_debug = Path(args.bin), Path(args.bin_debug)
        if not binary.exists():
            sys.exit(f"no binary at {binary} (the compiled fleet/tasks/kernel_tests_mi300.cu): {BUILD_HINT}")
        if not binary_debug.exists():
            binary_debug = None
    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="kernel_tests."))
    work.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(binary, binary_debug, work, args.n, args.seed)

    result = {"n": args.n, "seed": args.seed, "dry_run": args.dry_run,
              "binary": None if args.dry_run else str(binary),
              "binary_debug": None if args.dry_run or binary_debug is None else str(binary_debug),
              "tolerances": {"f32_rel": F32_REL, "bf16_rel": BF16_REL, "bf16_ulp": BF16_ULP,
                             "bf16_abs_floor": BF16_ABS_FLOOR, "partials_o_rel": PARTIALS_O_REL, "lse_abs": LSE_ABS, "splits_rel": SPLITS_REL},
              "tests": {}}
    try:
        for name in names:
            if name in NEEDS_DEBUG_BINARY and not args.dry_run and binary_debug is None:
                result["tests"][name] = {"result": "SKIP", "note": f"no debug binary at {args.bin_debug}"}
                print(f"SKIP {name} (no debug binary at {args.bin_debug})")
                continue
            s = TESTS[name](ctx)
            result["tests"][name] = s
            print_summary(name, s)
    finally:
        if not args.keep and not args.work_dir:
            shutil.rmtree(work, ignore_errors=True)
    results = [t["result"] for t in result["tests"].values()]
    result["overall"] = "PASS" if results and all(r in ("PASS", "SKIP") for r in results) else "FAIL"
    out = Path(args.out) if args.out else (DEFAULT_OUT_DRY if args.dry_run else DEFAULT_OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"overall: {result['overall']} -> {out}" + (" (dry run)" if args.dry_run else ""))
    return 0 if result["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
