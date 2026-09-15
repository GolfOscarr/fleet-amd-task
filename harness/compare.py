#!/usr/bin/env python3
"""Compare Fleet-side boundary tensors with the reference and write the report.

Pairs tensors by boundary key (harness/common.py), computes the three
metrics of docs/design-doc/07-correctness.md on FP32 upcasts, applies the
class threshold (starting values, or 4 x the calibrated floor when
calibration.json exists), runs the exact checks (B9 as a set, B16, the 32
output ids, the route log), and writes correctness_report.md and .json.

    python harness/compare.py --ref harness/ref --fleet harness/fleet_out
    exit status 1 if any boundary FAILs or an exact check fails.

Fleet-side files (written by run_fleet.py), all optional except the first:
  fleet_boundaries.safetensors     same keys as ref_boundaries_step0.safetensors
  fleet_output_ids.json            the ids the Fleet path produced (any count <= 32)
  fleet_route_log.json             [step][moe layer] {"idx": [...], "w": [...]}
  fleet_hidden_per_layer.safetensors  {"hidden": [L, H]} for the growth curve
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


# ----------------------------------------------------------------------------
# metrics


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """a = Fleet, b = reference; both upcast (FP64 for the reductions) and flattened."""
    # on the CPU: a captured GPU tensor against a reference loaded from disk
    # (calibrate.py on the VM, 2026-09-15, failed on the device mismatch)
    a = a.detach().cpu().double().reshape(-1)
    b = b.detach().cpu().double().reshape(-1)
    if a.numel() != b.numel():
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    d = a - b
    nb = torch.linalg.vector_norm(b).item()
    na = torch.linalg.vector_norm(a).item()
    max_abs = d.abs().max().item() if d.numel() else 0.0
    rel = torch.linalg.vector_norm(d).item() / nb if nb > 0 else (0.0 if max_abs == 0 else math.inf)
    cos = (torch.dot(a, b).item() / (na * nb)) if na > 0 and nb > 0 else (1.0 if max_abs == 0 else 0.0)
    return {"max_abs_err": max_abs, "rel_err": rel, "cos_sim": cos, "n": a.numel()}


def thresholds(calibration: dict | None) -> tuple[dict, str]:
    """Per-class rel_err threshold and where it came from."""
    if not calibration or "floor" not in calibration:
        return dict(common.CLASS_THRESHOLD), "starting values (07-correctness.md); not calibrated"
    th = dict(common.CLASS_THRESHOLD)
    for cls, floor in calibration["floor"].items():
        if cls in th and cls != "exact":
            th[cls] = common.THRESHOLD_MULTIPLIER * float(floor)
    return th, f"{common.THRESHOLD_MULTIPLIER:g} x calibrated floor"


# ----------------------------------------------------------------------------
# checks


def exact_set(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, str]:
    sa, sb = sorted(a.reshape(-1).tolist()), sorted(b.reshape(-1).tolist())
    return sa == sb, f"fleet {sa} ref {sb}"


def exact_scalar(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, str]:
    va, vb = int(a.reshape(-1)[0]), int(b.reshape(-1)[0])
    return va == vb, f"fleet {va} ref {vb}"


def compare_boundaries(ref: dict, fleet: dict, th: dict, floor: dict | None) -> list[dict]:
    rows = []
    for key in sorted(fleet.keys(), key=sort_key):
        cls = common.boundary_class(key)
        if cls is None:
            continue                      # auxiliary tensors are not boundaries
        row = {"key": key, "boundary": common.boundary_id(key), "class": cls}
        if key not in ref:
            row.update(result="MISSING_REF")
            rows.append(row)
            continue
        a, b = fleet[key], ref[key]
        if row["boundary"] == "B10":
            # the weights are matched by expert id through the B9 tensors (the
            # reference's top-k order is unspecified, the kernel's is by weight)
            a, b, detail = align_by_ids(key, fleet, ref)
            if detail:
                row.update(result="FAIL", detail=detail)
                rows.append(row)
                continue
        if cls == "exact":
            ok, detail = exact_set(a, b) if row["boundary"] == "B9" else exact_scalar(a, b)
            row.update(result="PASS" if ok else "FAIL", detail=detail)
            rows.append(row)
            continue
        m = metrics(a, b)
        if "shape_mismatch" in m:
            row.update(result="FAIL", detail=f"shape mismatch {m['shape_mismatch']}")
            rows.append(row)
            continue
        row.update(m)
        row["threshold"] = th[cls]
        row["floor"] = None if not floor else floor.get(cls)
        row["result"] = "PASS" if m["rel_err"] <= th[cls] else "FAIL"
        rows.append(row)
    return rows


def align_by_ids(key, fleet, ref):
    """B10 weights of both sides reordered by expert id via their B9 tensors."""
    k9 = key.replace(".B10.topk_w", ".B9.topk_idx")
    if k9 not in fleet or k9 not in ref:
        return fleet[key], ref[key], None
    fa = dict(zip(fleet[k9].reshape(-1).tolist(), fleet[key].reshape(-1).float().tolist()))
    rb = dict(zip(ref[k9].reshape(-1).tolist(), ref[key].reshape(-1).float().tolist()))
    if set(fa) != set(rb):
        return None, None, f"expert sets differ: fleet {sorted(fa)} ref {sorted(rb)}"
    ids = sorted(rb)
    return torch.tensor([fa[i] for i in ids]), torch.tensor([rb[i] for i in ids]), None


def sort_key(key: str):
    m = re.match(r"^L(\d+)\.", key)
    if not m:
        return (10_000, key)              # head.* and any other non-layer key
    l = int(m.group(1))
    b = common.boundary_id(key)
    return (l, int(b[1:]) if b.startswith("B") and b[1:].isdigit() else 99, key)


def compare_ids(ref_ids: list, fleet_ids: list) -> dict:
    n = min(len(ref_ids), len(fleet_ids))
    first = next((i for i in range(n) if ref_ids[i] != fleet_ids[i]), None)
    matched = n if first is None else first
    if first is not None:
        result = "FAIL"
    elif len(fleet_ids) == len(ref_ids):
        result = "PASS"
    else:
        result = "PARTIAL"                # a truncated run: every id produced matched
    return {
        "n_ref": len(ref_ids), "n_fleet": len(fleet_ids), "compared": n,
        "matched_prefix": matched, "first_divergence": first,
        "all_match": result == "PASS", "result": result,
    }


def compare_route_log(ref_log: list, fleet_log: list) -> dict:
    """Exact set equality of the top-k ids per (step, MoE layer), up to the
    number of steps and layers the Fleet log has."""
    mismatches = []
    steps = min(len(ref_log), len(fleet_log))
    for s in range(steps):
        layers = min(len(ref_log[s]), len(fleet_log[s]))
        for j in range(layers):
            r = sorted(ref_log[s][j]["idx"])
            f = sorted(fleet_log[s][j]["idx"][: len(r)])   # Fleet may append the forced 64, 65
            if r != f:
                mismatches.append({"step": s, "moe_layer_index": j, "ref": r, "fleet": f})
    return {"steps_compared": steps, "mismatches": mismatches,
            "result": "PASS" if not mismatches and steps > 0 else ("FAIL" if mismatches else "SKIP")}


def growth_curve(ref_hidden: torch.Tensor, fleet_hidden: torch.Tensor, th_layer: float) -> dict:
    n = min(ref_hidden.shape[0], fleet_hidden.shape[0])
    curve = [metrics(fleet_hidden[l], ref_hidden[l])["rel_err"] for l in range(n)]
    non_monotone = [l for l in range(1, n) if curve[l] < curve[l - 1] * 0.5]
    return {"layers": n, "rel_err_per_layer": curve,
            "max_rel_err": max(curve) if curve else None,
            "within_threshold": all(c <= th_layer for c in curve),
            "note": ("monotone" if not non_monotone else f"drops by more than half at layers {non_monotone}"),
            "result": "PASS" if curve and all(c <= th_layer for c in curve) else ("FAIL" if curve else "SKIP")}


# ----------------------------------------------------------------------------
# report


def fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}e}" if (x != 0 and abs(x) < 1e-2) else f"{x:.{nd}f}"
    return str(x)


def write_report(path: Path, result: dict):
    lines = ["# Correctness report", ""]
    lines.append(f"Thresholds: {result['threshold_source']}.")
    lines.append("")
    lines.append("| Key | Boundary | Class | max_abs_err | rel_err | cos_sim | floor | threshold | Result |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in result["boundaries"]:
        if r["class"] == "exact" or "rel_err" not in r:
            lines.append(f"| `{r['key']}` | {r['boundary']} | {r['class']} | - | - | - | - | exact | "
                         f"**{r['result']}** {r.get('detail', '')} |")
        else:
            lines.append(f"| `{r['key']}` | {r['boundary']} | {r['class']} | {fmt(r['max_abs_err'])} | "
                         f"{fmt(r['rel_err'])} | {fmt(r['cos_sim'], 5)} | {fmt(r['floor'])} | "
                         f"{fmt(r['threshold'])} | **{r['result']}** |")
    lines.append("")
    ids = result.get("output_ids")
    if ids:
        lines.append(f"Output ids: **{ids['result']}**, {ids['matched_prefix']} of {ids['n_ref']} matched"
                     + ("" if ids["first_divergence"] is None else f", first divergence at index {ids['first_divergence']}")
                     + ".")
    rl = result.get("route_log")
    if rl:
        lines.append(f"Route log: **{rl['result']}**, {rl['steps_compared']} steps compared, "
                     f"{len(rl['mismatches'])} mismatching (step, layer) pairs.")
    gc = result.get("growth_curve")
    if gc:
        lines.append(f"Growth curve (B13 per layer): **{gc['result']}**, {gc['layers']} layers, "
                     f"max rel_err {fmt(gc['max_rel_err'])}, {gc['note']}.")
    lines.append("")
    lines.append(f"Overall: **{result['overall']}** ({result['n_pass']} pass, {result['n_fail']} fail, "
                 f"{result['n_missing']} missing).")
    path.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------


def run(ref_dir: Path, fleet_dir: Path, calibration_path: Path | None, report: Path):
    from safetensors.torch import load_file

    ref = load_file(str(ref_dir / "ref_boundaries_step0.safetensors"))
    fleet = load_file(str(fleet_dir / "fleet_boundaries.safetensors"))
    calibration = None
    if calibration_path and calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())
    th, source = thresholds(calibration)
    floor = calibration.get("floor") if calibration else None

    result = {"threshold_source": source, "thresholds": th}
    result["boundaries"] = compare_boundaries(ref, fleet, th, floor)

    f_ids = fleet_dir / "fleet_output_ids.json"
    if f_ids.exists():
        result["output_ids"] = compare_ids(json.loads((ref_dir / "ref_output_ids.json").read_text()),
                                           json.loads(f_ids.read_text()))
    f_route = fleet_dir / "fleet_route_log.json"
    if f_route.exists():
        result["route_log"] = compare_route_log(json.loads((ref_dir / "ref_route_log.json").read_text()),
                                                json.loads(f_route.read_text()))
    f_hidden = fleet_dir / "fleet_hidden_per_layer.safetensors"
    if f_hidden.exists():
        rh = load_file(str(ref_dir / "ref_hidden_per_layer_step0.safetensors"))["hidden"]
        fh = load_file(str(f_hidden))["hidden"]
        result["growth_curve"] = growth_curve(rh, fh, th["layer"])

    results = [r["result"] for r in result["boundaries"]]
    for k in ("output_ids", "route_log", "growth_curve"):
        if k in result and result[k]["result"] != "SKIP":
            results.append(result[k]["result"])
    result["n_pass"] = results.count("PASS") + results.count("PARTIAL")
    result["n_fail"] = results.count("FAIL")
    result["n_missing"] = results.count("MISSING_REF")
    result["overall"] = "PASS" if result["n_fail"] == 0 and result["n_missing"] == 0 and results else "FAIL"

    report.parent.mkdir(parents=True, exist_ok=True)
    write_report(report, result)
    report.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(common.REF_DIR))
    ap.add_argument("--fleet", required=True)
    ap.add_argument("--calibration", default=None,
                    help="calibration.json (default: <ref>/calibration.json if present)")
    ap.add_argument("--report", default=None, help="default: <fleet>/correctness_report.md")
    args = ap.parse_args()
    ref_dir, fleet_dir = Path(args.ref), Path(args.fleet)
    cal = Path(args.calibration) if args.calibration else ref_dir / "calibration.json"
    report = Path(args.report) if args.report else fleet_dir / "correctness_report.md"
    result = run(ref_dir, fleet_dir, cal, report)
    for r in result["boundaries"]:
        extra = f" rel_err {fmt(r.get('rel_err'))} thr {fmt(r.get('threshold'))}" if "rel_err" in r else f" {r.get('detail', '')}"
        print(f"{r['result']:12s} {r['key']:28s}{extra}")
    for k in ("output_ids", "route_log", "growth_curve"):
        if k in result:
            print(f"{result[k]['result']:12s} {k}")
    print(f"overall: {result['overall']}  -> {report}")
    sys.exit(0 if result["overall"] == "PASS" else 1)


if __name__ == "__main__":
    main()
