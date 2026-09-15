#!/usr/bin/env bash
# Everything that can be checked without the MI300X, in one command, with one
# PASS/FAIL line per check. Run before a GPU session and paste the summary
# into the session log. The optional offline gfx942 compile needs Docker
# (env/offline_gfx942/README.md).
#
#   bash env/preflight.sh                  # local checks
#   OFFLINE_COMPILE=1 bash env/preflight.sh  # also the Docker hipcc compile
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLEET="$ROOT/repos/fleet-chiplet-megakernel"
PY="${PY:-$ROOT/.venv/bin/python}"
cd "$ROOT"

RESULTS=()
fail=0
check() {
  # $1 name, rest: command. Output goes to the log; the summary keeps one line.
  local name="$1"; shift
  local out
  if out=$("$@" 2>&1); then
    RESULTS+=("PASS  $name")
  else
    RESULTS+=("FAIL  $name")
    fail=1
    echo "---- $name failed:"; echo "$out" | tail -20
  fi
}

check "python venv present ($PY)" test -x "$PY"
check "test suite (harness/tests fleet/tests env/hw/tests)" "$PY" -m pytest harness/tests fleet/tests env/hw/tests -q
check "prompt ids match the pinned source (make_prompt.py --check)" "$PY" harness/make_prompt.py
check "graph builder dry run (326 ops / 1,880 tasks)" "$PY" fleet/build_graph.py --dry-run --layers 27
check "kernel syntax against the stub headers (check_syntax.sh)" bash fleet/tasks/check_syntax.sh
check "env scripts parse (bash -n)" bash -c 'for f in env/*.sh fleet/tasks/*.sh; do bash -n "$f" || exit 1; done'
check "hardware collection script parses (bash -n env/collect_hw.sh)" bash -n env/collect_hw.sh
check "submodule at the pinned commit 51dce4f" bash -c '[ "$(git -C "$1" rev-parse --short HEAD)" = "51dce4f" ]' _ "$FLEET"

# Both patches, in order, on a clean throwaway worktree of the submodule.
patches_apply() {
  local wt
  wt="$(mktemp -d)"
  git -C "$FLEET" worktree add -q "$wt" HEAD || return 1
  local rc=0
  git -C "$wt" apply --check "$ROOT/fleet/patches/gfx942.patch" || rc=1
  [ $rc = 0 ] && git -C "$wt" apply "$ROOT/fleet/patches/gfx942.patch" || rc=1
  [ $rc = 0 ] && git -C "$wt" apply --check "$ROOT/fleet/patches/new_tasks.patch" || rc=1
  git -C "$FLEET" worktree remove --force "$wt"
  return $rc
}
check "gfx942.patch then new_tasks.patch apply on a clean tree" patches_apply

if [ "${OFFLINE_COMPILE:-0}" = "1" ]; then
  check "offline gfx942 compile of the patched megakernel (Docker, hipcc 7.0)" bash env/offline_gfx942/run.sh
  check "offline gfx942 compile of the six hardware probes (Docker, hipcc 7.0)" bash env/hw/probes/compile_offline.sh
fi

echo
echo "== preflight $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname), $(git rev-parse --short HEAD)"
printf '%s\n' "${RESULTS[@]}"
exit $fail
