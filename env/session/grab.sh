#!/usr/bin/env bash
# Poll the Hot Aisle provisioning list and provision the first 1x MI300X that appears
# (authorized by the user on 2026-09-17: "check every 5 seconds; if there's a slot, reserve immediately").
#
#   bash env/session/grab.sh            # loops until a VM address is saved to env/session/vm.ip, or a provision fails
#   GRAB_MAX=720 bash env/session/grab.sh   # give up after N polls (default 720, about an hour)
#
# One TUI session at a time; each poll is one ssh to the TUI (about 10 s) plus a 5 s pause.
# The log is env/logs/grab.log; every poll writes one line. Exit 0 with the address, 1 otherwise.
set -uo pipefail
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
TUI="${TUI:-python3 $ROOT/env/hotaisle/tui.py}"
LOG="$ROOT/env/logs/grab.log"
GRAB_MAX="${GRAB_MAX:-720}"
mkdir -p "$ROOT/env/logs"
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }

[ -f "$ROOT/env/session/vm.ip" ] && { say "vm.ip exists already; not provisioning"; exit 1; }
say "start: polling for 1x MI300X (max $GRAB_MAX polls)"
n=0
while [ "$n" -lt "$GRAB_MAX" ]; do
  n=$((n + 1))
  # shellcheck disable=SC2086
  page="$($TUI 6 n WAIT4 2>/dev/null)"
  line="$(echo "$page" | grep -oE '1x MI300X VM[^│]*\([0-9]+ available\)' | head -1)"
  if [ -z "$line" ]; then
    say "poll $n: none"
    sleep 5
    continue
  fi
  say "poll $n: $line -> provisioning"
  # shellcheck disable=SC2086
  out="$($TUI 6 n WAIT4 ENTER WAIT8 y WAIT25 2>/dev/null)"
  if echo "$out" | grep -q "not found"; then
    say "provision failed: $(echo "$out" | grep -oE 'available VM[^│]*' | head -1); polling again"
    sleep 5
    continue
  fi
  date +%s > "$ROOT/env/session/vm.started"
  say "provision request accepted at $(cat "$ROOT/env/session/vm.started"); reading the address"
  sleep 15
  for _ in 1 2 3 4 5 6; do
    if addr="$(bash "$ROOT/env/session/laptop.sh" ip 2>/dev/null | tail -1)" && [ -n "$addr" ] && [ -f "$ROOT/env/session/vm.ip" ]; then
      say "VM address: $addr"
      exit 0
    fi
    sleep 20
  done
  say "provision accepted but no address after 2 minutes; check the TUI by hand"
  exit 1
done
say "gave up after $n polls"
exit 1
