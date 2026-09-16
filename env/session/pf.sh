#!/usr/bin/env bash
# The A4 fallback in one command: set the prefetch depth PF of the two attention kernels
# (docs/round-2/02-session-plan.md, row A4; 01-preparation.md, P6).
#
#   bash env/session/pf.sh 1      # the old kernels: one load in flight
#   bash env/session/pf.sh 4      # the P6 kernels (the default in the tree)
#   bash env/session/pf.sh        # show the current values
#
# Then `laptop.sh push` and `laptop.sh start kernels`. ROOT can point at another tree (the tests).
set -uo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FILES=("$ROOT/fleet/tasks/mi300/mla_attend_mi300.cuh" "$ROOT/fleet/tasks/mi300/mla_merge_uv_mi300.cuh")
want="${1:-}"
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
  if [ -n "$want" ]; then
    case "$want" in 1|2|4|8) ;; *) echo "PF must be 1, 2, 4 or 8 (the column loops batch PF loads of 8)"; exit 2;; esac
    sed -i.bak -E "s/^(  constexpr int PF = )[0-9]+;/\1$want;/" "$f" && rm -f "$f.bak"
  fi
  n="$(grep -cE "^  constexpr int PF = $want;" "$f" 2>/dev/null || true)"
  line="$(grep -nE '^  constexpr int PF = [0-9]+;' "$f" | head -1)"
  [ -n "$line" ] || { echo "no PF line in $f"; exit 1; }
  if [ -n "$want" ] && [ "$n" != "1" ]; then echo "PF not set in $f"; exit 1; fi
  echo "$(basename "$f"): $line"
done
