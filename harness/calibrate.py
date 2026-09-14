#!/usr/bin/env python3
"""BF16 noise-floor calibration (L8, OPEN-PROBLEMS MIN-2).

Two more runs of the reference that differ from the committed one only in
accumulation order, compared at every boundary (07-correctness.md):

  run B  the same GPU forward with the batch padded to 2 identical rows
         (a different GEMM tiling), row 0 taken
  run C  the same model on CPU (--cpu; slow: a 1,023-token prefill of the
         31 GB model in BF16 on the host)

The floor of a class is the largest rel_err over that class's boundaries
and over the runs; compare.py then uses 4 x floor as the threshold.
Written once to <ref>/calibration.json and never edited after a Fleet
result has been seen.

    python harness/calibrate.py --device cuda [--cpu]     # the real run, after run_reference.py
    python harness/calibrate.py --smoke                   # tiny random model, checks the plumbing
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import compare  # noqa: E402
import run_reference as RR  # noqa: E402


def capture(model, ids, layers, n_rows, s_max):
    """Prefill ids[:, :P] and run position P as a one-token step; step-0 boundaries."""
    from transformers import DynamicCache

    device = next(model.parameters()).device
    P = ids.shape[1] - 1
    cap = RR.Capture()
    RR.register_kva_hooks(model, cap)
    RR.register_route_hooks(model, cap)
    RR.register_boundary_hooks(model, cap, layers)
    with torch.inference_mode():
        x = ids[:, :P].repeat(n_rows, 1)
        out = model(x, attention_mask=torch.ones(n_rows, P, device=device, dtype=torch.long),
                    past_key_values=DynamicCache(), use_cache=True)
        pkv = out.past_key_values
        cos, sin = RR.rope_tables(model, s_max)
    if n_rows > 1:
        cap.store = {k: RR.first_row(v, n_rows) for k, v in cap.store.items()}
    st = RR.capture_step0(model, cap, ids, P, layers, cos, sin, pkv, n_rows=n_rows)
    cap.remove()
    return st["boundaries"]


def floors(ref, runs):
    """Per-boundary rel_err per run and the per-class maxima."""
    per_boundary = {}
    for name, b in runs.items():
        for key in ref:
            cls = common.boundary_class(key)
            if cls is None or cls == "exact" or key not in b:
                continue
            m = compare.metrics(b[key], ref[key])
            if "shape_mismatch" in m:
                continue
            per_boundary.setdefault(key, {})[name] = m["rel_err"]
    floor = {}
    for key, by_run in per_boundary.items():
        cls = common.boundary_class(key)
        floor[cls] = max(floor.get(cls, 0.0), max(by_run.values()))
    for cls in common.CLASS_THRESHOLD:
        if cls != "exact":
            floor.setdefault(cls, 0.0)
    exact = {}
    for name, b in runs.items():
        for key in ref:
            if common.boundary_class(key) == "exact" and key in b:
                ok = (compare.exact_set if common.boundary_id(key) == "B9" else compare.exact_scalar)(b[key], ref[key])[0]
                exact.setdefault(key, {})[name] = ok
    return floor, per_boundary, exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.MODEL)
    ap.add_argument("--prompt", default=str(common.PROMPT_IDS))
    ap.add_argument("--ref", default=str(common.REF_DIR))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cpu", action="store_true", help="also run the model on the CPU (slow)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ref_dir = Path(args.ref)
    if args.smoke and args.ref == str(common.REF_DIR):
        ref_dir = common.ROOT / "harness/ref_smoke"
    out = Path(args.out) if args.out else ref_dir / "calibration.json"
    if out.exists() and not args.smoke:
        sys.exit(f"{out} exists; the floor is recorded once. Delete it deliberately to redo.")

    from safetensors.torch import load_file

    if not (ref_dir / "ref_boundaries_step0.safetensors").exists():
        sys.exit(f"run run_reference.py{' --smoke' if args.smoke else ''} first ({ref_dir})")
    ref = load_file(str(ref_dir / "ref_boundaries_step0.safetensors"))
    meta = json.loads((ref_dir / "ref_run_meta.json").read_text())

    class A:
        model = args.model
        smoke = args.smoke
        device = args.device

    t0 = time.time()
    model = RR.load_model(A)
    cfg = model.config
    layers = [l for l in common.BOUNDARY_LAYERS if l < cfg.num_hidden_layers]
    if args.smoke:
        torch.manual_seed(1)
        ids = torch.randint(0, cfg.vocab_size, (1, meta["n_prompt"]))
    else:
        ids = torch.tensor(json.loads(Path(args.prompt).read_text()))[None]
    ids = ids.to(next(model.parameters()).device)
    s_max = meta["s_max"]
    runs = {}
    timings = {"load": round(time.time() - t0, 2)}

    t1 = time.time()
    runs["B_batch2_row0"] = capture(model, ids, layers, 2, s_max)
    timings["B"] = round(time.time() - t1, 2)
    if args.smoke:
        # the smoke reference ran on the same CPU; a second ordering: fewer threads
        n = torch.get_num_threads()
        torch.set_num_threads(1)
        t1 = time.time()
        runs["C_cpu_1thread"] = capture(model, ids, layers, 1, s_max)
        torch.set_num_threads(n)
        timings["C"] = round(time.time() - t1, 2)
    elif args.cpu:
        model_cpu = model.to("cpu")
        t1 = time.time()
        runs["C_cpu"] = capture(model_cpu, ids.cpu(), layers, 1, s_max)
        timings["C"] = round(time.time() - t1, 2)
        model.to(args.device)

    floor, per_boundary, exact = floors(ref, runs)
    import transformers

    result = {
        "what": "BF16 noise floor: rel_err of the reference against itself under a different "
                "accumulation order, per boundary and per class (max over boundaries and runs)",
        "runs": list(runs.keys()), "smoke": args.smoke,
        "device": args.device, "torch": torch.__version__, "transformers": transformers.__version__,
        "platform": platform.platform(), "timings_s": timings,
        "floor": floor,
        "threshold_at_4x": {cls: 4 * v for cls, v in floor.items()},
        "starting_threshold": {k: v for k, v in common.CLASS_THRESHOLD.items() if k != "exact"},
        "exact_checks_hold": exact,
        "per_boundary": per_boundary,
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    for cls, v in sorted(floor.items()):
        print(f"{cls:10s} floor {v:.3e}  threshold {4 * v:.3e}  (starting {common.CLASS_THRESHOLD[cls]:.0e})")
    bad = [k for k, by in exact.items() if not all(by.values())]
    print("exact checks hold in every run" if not bad else f"EXACT CHECKS DIFFER between orderings: {bad}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
