#!/usr/bin/env python3
"""Expert-routing correlation across decode steps (L14, OPEN-PROBLEMS MIN-6).

Reads ref_route_log.json ([step][moe layer] {"idx": [...], "w": [...]}) and
reports, per layer and overall: the overlap of the selected expert set
between consecutive steps, how many distinct experts each layer touched,
the expert-usage entropy, and the share of steps at which at least half of
the previous step's experts recur. This decides whether expert-to-XCD
affinity across steps is worth a next-step recommendation.

    python harness/route_analysis.py [--log harness/ref/ref_route_log.json] [--out harness/results/route_analysis.json]
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def analyze(log, n_experts=64):
    steps = len(log)
    n_layers = len(log[0]) if steps else 0
    k = len(log[0][0]["idx"]) if steps and n_layers else 0
    per_layer = []
    for j in range(n_layers):
        sets = [set(log[s][j]["idx"]) for s in range(steps)]
        overlaps = [len(sets[s] & sets[s - 1]) / k for s in range(1, steps)]
        counts = Counter(e for st in sets for e in st)
        total = sum(counts.values())
        entropy = -sum(c / total * math.log2(c / total) for c in counts.values()) if total else 0.0
        top_share = sum(c for _, c in counts.most_common(8)) / total if total else 0.0
        weights = [w for s in range(steps) for w in log[s][j]["w"]]
        per_layer.append({
            "moe_layer_index": j,
            "mean_consecutive_overlap": sum(overlaps) / len(overlaps) if overlaps else None,
            "steps_with_half_or_more_recurring": (sum(o >= 0.5 for o in overlaps) / len(overlaps)) if overlaps else None,
            "distinct_experts": len(counts),
            "usage_entropy_bits": entropy,
            "max_entropy_bits": math.log2(n_experts),
            "top8_share_of_selections": top_share,
            "mean_topk_weight": sum(weights) / len(weights) if weights else None,
            "mean_weight_mass_of_topk": sum(sum(log[s][j]["w"]) for s in range(steps)) / steps if steps else None,
        })
    overlaps_all = [x["mean_consecutive_overlap"] for x in per_layer if x["mean_consecutive_overlap"] is not None]
    summary = {
        "steps": steps, "moe_layers": n_layers, "topk": k, "n_experts": n_experts,
        "mean_consecutive_overlap_all_layers": sum(overlaps_all) / len(overlaps_all) if overlaps_all else None,
        "layers_with_overlap_above_half": sum(o > 0.5 for o in overlaps_all),
        "mean_distinct_experts_per_layer": sum(x["distinct_experts"] for x in per_layer) / n_layers if n_layers else None,
        "reading": None,
        "per_layer": per_layer,
    }
    m = summary["mean_consecutive_overlap_all_layers"]
    if m is not None:
        if m >= 0.5:
            summary["reading"] = ("consecutive steps reuse at least half of their experts on average: "
                                  "expert-to-XCD affinity across steps could keep expert weights resident "
                                  "in a per-XCD cache level and is worth a next-step recommendation")
        else:
            summary["reading"] = ("consecutive steps reuse fewer than half of their experts on average: "
                                  "cross-step affinity would rarely pay; placement by slot order stands")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(common.REF_DIR / "ref_route_log.json"))
    ap.add_argument("--out", default=str(common.ROOT / "harness/results/route_analysis.json"))
    ap.add_argument("--n-experts", type=int, default=64)
    args = ap.parse_args()
    log = json.loads(Path(args.log).read_text())
    summary = analyze(log, args.n_experts)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{summary['steps']} steps x {summary['moe_layers']} MoE layers, top-{summary['topk']}")
    print(f"mean consecutive-step overlap: {summary['mean_consecutive_overlap_all_layers']}")
    print(f"layers with overlap above one half: {summary['layers_with_overlap_above_half']} of {summary['moe_layers']}")
    print(f"mean distinct experts per layer over the run: {summary['mean_distinct_experts_per_layer']}")
    print(summary["reading"])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
