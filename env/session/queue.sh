#!/usr/bin/env bash
# The graph-run queue: one run_fleet.py run at a time, each recorded (docs/round-2/01-preparation.md, P4).
#
#   bash env/session/queue.sh run <queue file>
#   bash env/session/queue.sh bisect <label file> -- <base run_fleet.py args>
#
# Queue file: one row per run, the run_fleet.py arguments (without --model-dir, added here),
# with optional trailing words:  compare   run compare.py on the result
#                                table     run measure.py on the result alone (the per-operator table from --event-timing)
#                                measure   profile the same graph under rocprofv3 (kernel trace, four PMC pairs), then measure.py
#                                continue  a FAIL of this row does not stop the queue
# '#' starts a comment. Example:
#   --layers 8 --head --iters 2 continue
#   --layers 27 --head --iters 32 compare
#   --layers 27 --head --iters 32 --event-timing measure
#
# Bisect: the labels of the label file are in graph order; the run stopped after the last
# label is known to fault and the graph stopped after the first operator is known to run.
# A binary search over --stop-after finds the first label whose run faults, in about
# log2(N) runs; the answer goes to env/logs/bisect.result and the queue status.
#
# One row of env/logs/queue.status per run:
#   <utc> <name> PASS|FAIL rc=<n> mpk=<n> fault=<n> fwd=<n> wall=<s>s [compare=PASS|FAIL] [measure=PASS|FAIL]
# and STOP <name> when a FAIL without 'continue' ends the queue.
# Guards, before a row runs: --iters at most 32; --debug only with --iters 1; compare (and any
# untruncated run) needs harness/ref/ref_cache.safetensors; measure needs the profiler on PATH.
# Overrides for the tests: RUN_FLEET, COMPARE, MEASURE, PROFILER, PY, SNAP, FLEET_OUT, RECORD, LOGDIR, REF_DIR.
set -uo pipefail
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$ROOT"

PY="${PY:-python}"
REF_DIR="${REF_DIR:-$ROOT/harness/ref}"
RUN_FLEET="${RUN_FLEET:-$PY harness/run_fleet.py}"
COMPARE="${COMPARE:-$PY harness/compare.py}"
MEASURE="${MEASURE:-$PY harness/measure.py}"
PROFILER="${PROFILER:-rocprofv3}"
PMC_SETS=(
  "TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_32B_sum"
  "TCC_EA0_WRREQ_sum TCC_EA0_WRREQ_64B_sum"
  "TCC_HIT_sum TCC_MISS_sum"
  "TCC_BUBBLE_sum TCC_EA0_RDREQ_sum"
)

# the run directory name run_fleet.py will use for these arguments
name_of() {
  # shellcheck disable=SC2086
  "$PY" -c "import sys; sys.path.insert(0, 'harness'); import run_fleet
print(run_fleet.run_name(run_fleet.build_parser().parse_args(sys.argv[1:] + ['--model-dir', 'x'])))" "$@"
}

# the prerequisites of a row, checked before anything runs: prints the reason and returns 1
# (a FAIL row without a run; docs/round-2/02-session-plan.md, "guards")
row_guard() {
  local flags="$1"; shift
  local args=("$@") iters=1 i
  for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
      --iters) iters="${args[$((i + 1))]}";;
      --iters=*) iters="${args[$i]#--iters=}";;
    esac
  done
  if [ "$iters" -gt 32 ] 2>/dev/null; then echo "guard: --iters $iters is above 32 (the RoPE tables hold 1,056 positions)"; return 1; fi
  if [[ " ${args[*]} " == *" --debug "* ]] && [ "$iters" != "1" ]; then
    echo "guard: --debug is the growth curve at step 0; it needs --iters 1 (got $iters)"; return 1; fi
  if [[ "$flags" == *compare* ]] && [ ! -f "$REF_DIR/ref_cache.safetensors" ]; then
    echo "guard: compare needs $REF_DIR/ref_cache.safetensors (run the reference stage first)"; return 1; fi
  if [[ " ${args[*]} " != *" --stop-after "* ]] && [ ! -f "$REF_DIR/ref_cache.safetensors" ] && [ "$DRY" != "1" ]; then
    echo "guard: run_fleet.py loads $REF_DIR/ref_cache.safetensors (run the reference stage first)"; return 1; fi
  if [[ "$flags" == *measure* ]] && [ "$DRY" != "1" ] && ! command -v "${PROFILER%% *}" >/dev/null 2>&1; then
    echo "guard: measure needs $PROFILER on PATH"; return 1; fi
  return 0
}

# run one graph: sets NAME, RESULT, RC, MPK, FAULT, FWD, WALL
run_graph() {
  local args=("$@")
  NAME="$(name_of "${args[@]}")" || { echo "cannot parse: ${args[*]}"; RESULT=FAIL; RC=2; return 1; }
  local snap; snap="$(snap_dir)"
  local log="$LOGDIR/runs/$NAME.out" t0; t0=$(date +%s)
  echo "== $NAME: $RUN_FLEET ${args[*]} --model-dir $snap"
  if [ "$DRY" = "1" ]; then echo "+ $RUN_FLEET ${args[*]} --model-dir $snap > $log"; RESULT=PASS; RC=0; MPK=1; FAULT=0; FWD=0; WALL=0; return 0; fi
  # shellcheck disable=SC2086
  $RUN_FLEET "${args[@]}" --model-dir "$snap" > "$log" 2>&1; RC=$?
  WALL=$(( $(date +%s) - t0 ))
  MPK="$(grep -c 'mpk()' "$log" || true)"
  FAULT="$(grep -c -E 'AcceleratorError|illegal|Memory access fault|HSA_STATUS_ERROR' "$log" || true)"
  FWD="$(grep -c 'FWD_PASS' "$log" || true)"
  if [ "$RC" = "0" ] && [ "$FAULT" = "0" ] && [ "$MPK" -ge 1 ]; then RESULT=PASS; else RESULT=FAIL; fi
  mkdir -p "$RECORD/runs/$NAME"
  [ -d "$FLEET_OUT/$NAME" ] && record_copy "$FLEET_OUT/$NAME" "$RECORD/runs/$NAME"
  cp "$log" "$RECORD/runs/$NAME/run.out" 2>/dev/null || true
  [ "$RESULT" = "PASS" ]
}

do_compare() {
  local name="$1" log="$LOGDIR/runs/$1.compare.out"
  if [ "$DRY" = "1" ]; then echo "+ $COMPARE --fleet $FLEET_OUT/$name"; echo PASS; return 0; fi
  # shellcheck disable=SC2086
  if $COMPARE --fleet "$FLEET_OUT/$name" > "$log" 2>&1; then echo PASS; else echo FAIL; fi
  cp "$FLEET_OUT/$name"/correctness_report.* "$RECORD/runs/$name/" 2>/dev/null || true
  cp "$log" "$RECORD/runs/$name/compare.out" 2>/dev/null || true
}

do_table() {
  local name="$1" log="$LOGDIR/runs/$1.measure.out"
  if [ "$DRY" = "1" ]; then echo "+ $MEASURE --run $FLEET_OUT/$name" >&2; echo PASS; return 0; fi
  # shellcheck disable=SC2086
  if $MEASURE --run "$FLEET_OUT/$name" > "$log" 2>&1; then
    cp "$FLEET_OUT/$name"/metrics.json "$FLEET_OUT/$name"/report_table.md "$RECORD/runs/$name/" 2>/dev/null || true
    echo PASS
  else echo FAIL; fi
}

# the same graph under the profiler: one kernel trace, one run per PMC pair, then measure.py
do_measure() {
  local name="$1"; shift
  local args=("$@") snap prof="$LOGDIR/prof/$1"; snap="$(snap_dir)"
  mkdir -p "$prof"
  local k=0 set_ ktrace
  if [ "$DRY" = "1" ]; then    # the commands go to stderr: stdout is the verdict the caller captures
    echo "+ $PROFILER --kernel-trace --output-format csv -d $prof/ktrace -- $RUN_FLEET ${args[*]} --model-dir $snap --out $FLEET_OUT/${name}_ktrace" >&2
    for set_ in "${PMC_SETS[@]}"; do k=$((k + 1)); echo "+ $PROFILER --pmc $set_ --output-format csv -d $prof/pmc$k -- $RUN_FLEET ${args[*]} --model-dir $snap --out $FLEET_OUT/${name}_pmc$k" >&2; done
    echo "+ $MEASURE --run $FLEET_OUT/$name --kernel-trace <csv> --pmc $prof" >&2; echo PASS; return 0
  fi
  # shellcheck disable=SC2086
  $PROFILER --kernel-trace --output-format csv -d "$prof/ktrace" -- $RUN_FLEET "${args[@]}" --model-dir "$snap" --out "$FLEET_OUT/${name}_ktrace" > "$prof/ktrace.out" 2>&1 || { echo FAIL; return 1; }
  for set_ in "${PMC_SETS[@]}"; do
    k=$((k + 1))
    # shellcheck disable=SC2086
    $PROFILER --pmc $set_ --output-format csv -d "$prof/pmc$k" -- $RUN_FLEET "${args[@]}" --model-dir "$snap" --out "$FLEET_OUT/${name}_pmc$k" > "$prof/pmc$k.out" 2>&1 || { echo FAIL; return 1; }
  done
  ktrace="$(find "$prof/ktrace" -name '*kernel_trace.csv' | head -1)"
  # the four PMC runs are passed as a directory: measure.py merges them per file (a counter in two
  # runs takes the later run's value; pmc4 is the TCC_BUBBLE + RDREQ pair), never as one concatenation
  # shellcheck disable=SC2086
  if $MEASURE --run "$FLEET_OUT/$name" --kernel-trace "$ktrace" --pmc "$prof" > "$prof/measure.out" 2>&1; then
    cp "$FLEET_OUT/$name"/metrics.json "$FLEET_OUT/$name"/report_table.md "$RECORD/runs/$name/" 2>/dev/null || true
    mkdir -p "$RECORD/runs/$name/prof"
    local f
    for f in $(find "$prof" -name '*counter_collection.csv' -o -name '*kernel_trace.csv'); do
      cp "$f" "$RECORD/runs/$name/prof/$(basename "$(dirname "$(dirname "$f")")")_$(basename "$f")"
    done
    echo PASS
  else echo FAIL; fi
}

queue_run() {
  local file="$1"
  [ -f "$file" ] || { echo "no queue file $file"; return 2; }
  local line words w args flags
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"; [ -z "${line// }" ] && continue
    # shellcheck disable=SC2206
    words=($line); args=(); flags=""
    for w in "${words[@]}"; do
      case "$w" in compare|table|measure|continue) flags="$flags $w";; *) args+=("$w");; esac
    done
    local extra="" why
    if ! why="$(row_guard "$flags" "${args[@]}")"; then
      NAME="$(name_of "${args[@]}" 2>/dev/null || echo "${args[*]}")"
      row "$QSTATUS" "$NAME FAIL $why"
      if [[ "$flags" != *continue* ]]; then row "$QSTATUS" "STOP $NAME"; return 1; fi
      continue
    fi
    if run_graph "${args[@]}"; then :; fi
    [[ "$flags" == *compare* ]] && extra="$extra compare=$(do_compare "$NAME")"
    [[ "$flags" == *table* ]] && [ "$RESULT" = "PASS" ] && extra="$extra table=$(do_table "$NAME")"
    [[ "$flags" == *measure* ]] && [ "$RESULT" = "PASS" ] && extra="$extra measure=$(do_measure "$NAME" "${args[@]}" | tail -1)"
    row "$QSTATUS" "$NAME $RESULT rc=$RC mpk=$MPK fault=$FAULT fwd=$FWD wall=${WALL}s$extra"
    if [ "$RESULT" = "FAIL" ] && [[ "$flags" != *continue* ]]; then row "$QSTATUS" "STOP $NAME"; return 1; fi
  done < "$file"
  row "$QSTATUS" "DONE $file"
}

queue_bisect() {
  local file="$1"; shift
  [ "${1:-}" = "--" ] && shift
  local base=("$@") labels=() l
  while IFS= read -r l || [ -n "$l" ]; do l="${l%%#*}"; l="${l// }"; [ -n "$l" ] && labels+=("$l"); done < "$file"
  local n=${#labels[@]} lo=0 hi=$((${#labels[@]} - 1)) mid runs=0
  [ "$n" -ge 2 ] || { echo "need at least two labels"; return 2; }
  local seen=" "   # " label=RESULT ..." (no associative arrays: the laptop's bash is 3.2)
  probe() {   # sets RESULT for label $1, running at most once per label
    case "$seen" in
      *" $1="*) RESULT="${seen#* $1=}"; RESULT="${RESULT%% *}";;
      *) run_graph "${base[@]}" --stop-after "$1" || true
         runs=$((runs + 1)); seen="$seen$1=$RESULT "
         row "$QSTATUS" "$NAME $RESULT rc=$RC mpk=$MPK fault=$FAULT fwd=$FWD wall=${WALL}s bisect";;
    esac
  }
  while [ "$lo" -lt "$hi" ]; do
    mid=$(( (lo + hi) / 2 ))
    probe "${labels[$mid]}"
    if [ "$RESULT" = "PASS" ]; then lo=$((mid + 1)); else hi=$mid; fi
  done
  probe "${labels[$lo]}"
  local verdict
  if [ "$RESULT" = "FAIL" ]; then verdict="first-fault=${labels[$lo]}"; else verdict="no-fault-up-to=${labels[$lo]}"; fi
  echo "BISECT $verdict runs=$runs" | tee "$LOGDIR/bisect.result"
  row "$QSTATUS" "BISECT $verdict runs=$runs"
}

case "${1:-}" in
  run) shift; queue_run "$@";;
  bisect) shift; queue_bisect "$@";;
  *) sed -n 2,26p "$0"; exit 2;;
esac
