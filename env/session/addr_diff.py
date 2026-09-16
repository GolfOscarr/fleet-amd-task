#!/usr/bin/env python3
"""Which tensors moved between two runs, from the address record of fleet_run_meta.json
(docs/gpu/02-validation/02-session-plan.md, row A5.2 and the fault decision tree; the record comes
from run_fleet.py --pad-alloc / --align-alloc / --workspaces-first, P1).

    python3 env/session/addr_diff.py <run A>/fleet_run_meta.json <run B>/fleet_run_meta.json [--all]

Prints the pad and the alignment flags of both runs, every tensor whose address differs
(name, address in A, address in B, delta, and whether bit 32 or above changed, the signature
of a 32-bit offset that overflowed), then a summary. --all prints the unchanged ones too.
"""
import json
import sys
from pathlib import Path


def load(path):
    m = json.loads(Path(path).read_text())
    return m, m.get("addresses", {})


def diff(a, b):
    rows = []
    for name in sorted(set(a) | set(b), key=lambda n: (a.get(n, b.get(n, 0)), n)):
        pa, pb = a.get(name), b.get(name)
        if pa is None or pb is None:
            rows.append((name, pa, pb, None, None))
            continue
        rows.append((name, pa, pb, pb - pa, (pa >> 32) != (pb >> 32)))
    return rows


def report(pa, pb, rows, show_all=False):
    lines = []
    for label, m in (("A", pa), ("B", pb)):
        lines.append(f"{label}: layers {m.get('layers')} head {m.get('head')} iters {m.get('iters')} "
                     f"pad {m.get('pad_alloc_gb', 0)} GiB at {m.get('pad_addr')} align {m.get('align_alloc', 0)} "
                     f"workspaces_first {m.get('workspaces_first', False)} ids {m.get('output_ids')}")
    lines.append(f"{'tensor':32} {'A':>16} {'B':>16} {'delta':>14} hi32")
    moved = high = 0
    for name, x, y, d, hi in rows:
        if d is None:
            lines.append(f"{name:32} {str(x):>16} {str(y):>16} {'only one run':>14}")
            continue
        if d == 0 and not show_all:
            continue
        moved += d != 0
        high += bool(hi)
        lines.append(f"{name:32} {x:>#16x} {y:>#16x} {d:>+14d} {'YES' if hi else ''}")
    lines.append(f"{moved} of {sum(1 for r in rows if r[3] is not None)} tensors moved; "
                 f"{high} crossed a 4 GiB (bit 32) boundary")
    return "\n".join(lines)


def main():
    args = [a for a in sys.argv[1:] if a != "--all"]
    if len(args) != 2:
        sys.exit(__doc__)
    (ma, aa), (mb, ab) = load(args[0]), load(args[1])
    print(report(ma, mb, diff(aa, ab), "--all" in sys.argv))


if __name__ == "__main__":
    main()
