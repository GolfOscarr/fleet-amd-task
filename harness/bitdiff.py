#!/usr/bin/env python3
"""The ULP-level story of two runs' boundary tensors, beside compare.py's tolerance check (L7,
docs/gpu-experiments/04-kernels/05-local-preparation.md; I5 of 01-gemv-ideas.md).

    python harness/bitdiff.py <dir-a> <dir-b> [--out file.md]

Loads fleet_boundaries.safetensors from both directories (the file run_fleet.py writes,
boundary_dump in harness/run_fleet.py) and reports, for every key present in both, the
element count, the count of differing elements, the max ULP distance and the max absolute
difference, as a markdown table sorted by key; a key present on only one side gets one line
below the table instead of a row.

BF16 and FP32 tensors compare their ULP distance as the plain distance between the raw bit
patterns, read as 16-bit or 32-bit unsigned integers (no sign-magnitude reordering: a pair
that straddles zero reads as a large distance, the same as it is between the two integer
patterns). Integer tensors have no ULP of their own, so their ULP column carries the max
absolute difference again, as an integer. Everything runs on the CPU. This is a report, not
a check: it always exits 0, whatever it finds.
"""
import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

# float dtype -> (the same-width integer dtype its bits are read as, the bit width)
BIT_DTYPE = {torch.bfloat16: (torch.int16, 16), torch.float32: (torch.int32, 32)}


def load_boundaries(run_dir) -> dict:
    return load_file(str(Path(run_dir) / "fleet_boundaries.safetensors"))


def ulp_distance(a: torch.Tensor, b: torch.Tensor) -> int:
    """Max distance between the raw bit patterns of two BF16 or FP32 tensors of the same
    dtype and shape, read as unsigned 16-bit or 32-bit integers."""
    int_dtype, bits = BIT_DTYPE[a.dtype]
    mask = (1 << bits) - 1
    ai = a.contiguous().view(int_dtype).to(torch.int64) & mask
    bi = b.contiguous().view(int_dtype).to(torch.int64) & mask
    return int((ai - bi).abs().max().item())


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    d = (a.double() - b.double()).abs()
    return float(d.max().item())


def compare_key(a: torch.Tensor, b: torch.Tensor) -> dict:
    """a, b: the same key from the two reports. Returns the table row's fields, or a 'note'
    when the two sides cannot be compared element-wise (a shape mismatch)."""
    a, b = a.detach().cpu(), b.detach().cpu()
    row = {"n": a.numel(), "dtype": str(a.dtype).replace("torch.", "")}
    if a.shape != b.shape:
        row["note"] = f"shape mismatch: {list(a.shape)} vs {list(b.shape)}"
        return row
    if a.numel() == 0:
        row.update(differing=0, ulp=0, max_abs=0.0)
        return row
    row["differing"] = int((a != b).sum().item())
    row["max_abs"] = max_abs_diff(a, b)
    if a.dtype in BIT_DTYPE and a.dtype == b.dtype:
        row["ulp"] = ulp_distance(a, b)
    elif not a.is_floating_point() and not b.is_floating_point():
        row["ulp"] = int((a.to(torch.int64) - b.to(torch.int64)).abs().max().item())
    else:
        # a float dtype outside BF16/FP32 (or the two sides disagree on dtype): no bit-pattern
        # reading defined here, so the ULP column repeats the max absolute difference
        row["ulp"] = int(round(row["max_abs"]))
    return row


def fmt(x):
    """compare.py's float formatting (harness/compare.py:fmt), so the two reports read alike."""
    if isinstance(x, float):
        return f"{x:.3e}" if (x != 0 and abs(x) < 1e-2) else f"{x:.3f}"
    return str(x)


def build_report(a: dict, b: dict) -> str:
    both = sorted(set(a) & set(b))
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    lines = ["# Bit-diff report", "",
             "| Key | Dtype | N | Differing | Max ULP | Max abs diff |",
             "|---|---|---|---|---|---|"]
    for key in both:
        r = compare_key(a[key], b[key])
        if "note" in r:
            lines.append(f"| `{key}` | {r['dtype']} | {r['n']} | - | - | {r['note']} |")
        else:
            lines.append(f"| `{key}` | {r['dtype']} | {r['n']} | {r['differing']} | "
                         f"{r['ulp']} | {fmt(r['max_abs'])} |")
    lines.append("")
    for key in only_a:
        lines.append(f"`{key}`: only in a.")
    for key in only_b:
        lines.append(f"`{key}`: only in b.")
    return "\n".join(lines) + "\n"


def run(dir_a, dir_b, out=None) -> str:
    text = build_report(load_boundaries(dir_a), load_boundaries(dir_b))
    if out:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--out", default=None, help="default: print the report to stdout")
    args = ap.parse_args()
    text = run(args.dir_a, args.dir_b, args.out)
    print(str(Path(args.out)) if args.out else text)
    sys.exit(0)


if __name__ == "__main__":
    main()
