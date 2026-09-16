#!/usr/bin/env bash
# The session stages on the VM (docs/round-2/01-preparation.md, P4; 02-session-plan.md).
#
#   bash env/session/vm.sh <stage> [args]          # run a stage in the foreground
#   bash env/session/vm.sh start <stage> [args]    # run it detached: env/logs/<stage>.out, a row in env/logs/session.status
#   bash env/session/vm.sh status                  # the status files and the last lines of the running stage logs
#   bash env/session/vm.sh snapshot-logs           # copy env/logs/*.out and the status files into the record (before a pull)
#
# Stages, in session order:
#   download    the model into the Hugging Face cache (79 s on 2026-09-15)
#   image       docker build of env/docker/Dockerfile; push if a registry login exists (laptop.sh login); logout
#   setup       env/setup.sh with SKIP_DOWNLOAD=1 (both venvs, the patches, the Fleet build; gate 1)
#   hw          env/collect_hw.sh, skipped when this host and ROCm version are already in an env/hw/*/summary.md
#   checks      env/check_day1.sh (7 PASS lines)
#   reference   run_reference.py, calibrate.py, route_analysis.py under .venv (the reference tensors are not in the repo)
#   kernels     the kernel-test launcher (built if missing) and kernel_tests.py, 100 trials per suite
#   queue F     env/session/queue.sh run F   (one graph run at a time)
#   bisect F -- ARGS   env/session/queue.sh bisect F -- ARGS
#
# Every stage writes "PASS <stage>" or "FAIL <stage>: <reason>" to the status file; a START row
# without a PASS or FAIL row is a stage still running. DRY=1 prints the commands instead.
set -uo pipefail
# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$ROOT" || exit 1

stage_download() {
  local venv=/tmp/hfdl
  if [ ! -x "$venv/bin/pip" ]; then
    run python3 -m venv --without-pip "$venv" || return 1
    run bash -c "curl -sS https://bootstrap.pypa.io/get-pip.py | $venv/bin/python3 - -q" || return 1
  fi
  run "$venv/bin/pip" install -q huggingface_hub || return 1
  run "$venv/bin/python3" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL_ID'))" || return 1
  [ "$DRY" = "1" ] && return 0
  local d; d="$(snap_dir)"
  [ -n "$d" ] && [ -f "$d/config.json" ] && [ "$(ls "$d"/*.safetensors 2>/dev/null | wc -l)" -gt 0 ] \
    || { echo "no snapshot with config.json and safetensors under $HF_CACHE"; return 1; }
  echo "snapshot $d"
}

stage_image() {
  local tag log="$LOGDIR/docker_build.out"
  tag="$IMAGE_REPO:$(date -u +%Y%m%d)"
  run docker pull rocm/dev-ubuntu-24.04:7.2 || return 1
  run bash -c "docker build -f env/docker/Dockerfile -t $tag . > $log 2>&1" || { echo "build failed: $(grep -m1 -iE 'error' "$log" 2>/dev/null)"; return 1; }
  if [ "$DRY" != "1" ] && ! grep -q "import mirage OK" "$log"; then echo "build ended without 'import mirage OK'"; return 1; fi
  echo "built $tag"
  if [ "$DRY" = "1" ] || grep -q '"ghcr.io"' "$HOME/.docker/config.json" 2>/dev/null; then
    run docker push "$tag" || { echo "push failed"; run docker logout ghcr.io; return 1; }
    run docker logout ghcr.io
    echo "pushed $tag"
  else
    echo "not pushed: no ghcr.io login (laptop.sh login); image stays local as $tag"
  fi
}

stage_setup() {
  SKIP_DOWNLOAD=1 run bash env/setup.sh || return 1
  [ "$DRY" = "1" ] && return 0
  fleet_env
  (cd "$FLEET" && python -c "import mirage") || { echo "import mirage failed after setup.sh"; return 1; }
}

stage_hw() {
  local host rocm
  host="$(hostname)"; rocm="$(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"
  local prev
  prev="$(grep -l "$host" env/hw/2*/summary.md 2>/dev/null | xargs -r grep -l "$rocm" 2>/dev/null | head -1)"
  if [ -n "$prev" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "skipped: $host with ROCm $rocm is already recorded in $prev (FORCE=1 to re-collect)"; return 0
  fi
  run bash env/collect_hw.sh || return 1
  [ "$DRY" = "1" ] || [ -f "$RECORD/summary.md" ] || { echo "no $RECORD/summary.md"; return 1; }
}

stage_checks() {
  local out="$LOGDIR/check_day1.out"
  run bash -c "bash env/check_day1.sh > $out 2>&1"
  [ "$DRY" = "1" ] && return 0
  local fails; fails="$(grep -c '^FAIL' "$out" || true)"
  grep -E '^(PASS|FAIL)' "$out"
  [ "$fails" = "0" ] || { echo "$fails check(s) FAIL"; return 1; }
}

stage_reference() {
  local py="$ROOT/.venv/bin/python"
  [ "$DRY" = "1" ] && py=python
  run "$py" harness/run_reference.py --device cuda || return 1
  run "$py" harness/calibrate.py --device cuda || return 1
  run "$py" harness/route_analysis.py || return 1
  [ "$DRY" = "1" ] && return 0
  [ -f harness/ref/ref_cache.safetensors ] && [ -f harness/ref/calibration.json ] \
    || { echo "reference artifacts missing"; return 1; }
}

build_kernel_tests() {
  local defs="-D__HIP_PLATFORM_AMD__=1 -DMIRAGE_AMD_MI300 -DMIRAGE_BACKEND_USE_ROCM -DMPK_TARGET_CC=94 -DMODE_ONLINE"
  local inc="-I fleet -I $FLEET/include -I $FLEET/include/mirage/persistent_kernel"
  mkdir -p fleet/tasks/build
  # shellcheck disable=SC2086
  [ -x fleet/tasks/build/kernel_tests ] || run hipcc --offload-arch=gfx942 -O2 -std=c++17 $defs $inc \
      fleet/tasks/kernel_tests_mi300.cu -o fleet/tasks/build/kernel_tests || return 1
  # shellcheck disable=SC2086
  [ -x fleet/tasks/build/kernel_tests_debug ] || run hipcc --offload-arch=gfx942 -O2 -std=c++17 $defs $inc \
      -DMLA_ATTEND_DEBUG_SCORES fleet/tasks/kernel_tests_mi300.cu -o fleet/tasks/build/kernel_tests_debug || return 1
}

stage_kernels() {
  fleet_env
  build_kernel_tests || { echo "kernel_tests did not build"; return 1; }
  run python fleet/tasks/kernel_tests.py --n 100 || return 1
  [ "$DRY" = "1" ] && return 0
  local fails; fails="$(grep -c '"FAIL"' fleet/tasks/results/kernel_tests.json || true)"
  [ "$fails" = "0" ] || { echo "$fails suite(s) FAIL in fleet/tasks/results/kernel_tests.json"; return 1; }
  record_copy fleet/tasks/results "$RECORD/kernel_tests"
}

stage_queue() { fleet_env; bash env/session/queue.sh run "$@"; }
stage_bisect() { fleet_env; bash env/session/queue.sh bisect "$@"; }

snapshot_logs() {
  mkdir -p "$RECORD/logs"
  local f
  for f in "$LOGDIR"/*.out "$LOGDIR"/*.status "$LOGDIR"/runs/*.out; do
    [ -f "$f" ] || continue
    if [ "$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f")" -gt 409600 ]; then
      echo "dropped (over 400 KB): $f"; continue
    fi
    cp "$f" "$RECORD/logs/"
  done
  echo "logs in $RECORD/logs"
}

main() {
  local cmd="${1:-}"; shift || true
  case "$cmd" in
    start)
      local stage="${1:-}"; shift || true
      [ -n "$stage" ] || { echo "usage: vm.sh start <stage> [args]"; return 2; }
      row "$STATUS" "START $stage $*"
      if [ "$DRY" = "1" ]; then echo "+ setsid nohup bash $0 $stage $* > $LOGDIR/$stage.out"; return 0; fi
      (setsid nohup bash "$0" "$stage" "$@" > "$LOGDIR/$stage.out" 2>&1 < /dev/null &)
      echo "started $stage; log $LOGDIR/$stage.out";;
    status)
      echo "== $STATUS"; [ -f "$STATUS" ] && cat "$STATUS"
      echo "== $QSTATUS"; [ -f "$QSTATUS" ] && cat "$QSTATUS"
      local s
      for s in $(awk '$2 == "START" {print $3}' "$STATUS" 2>/dev/null | sort -u); do
        if ! grep -qE " (PASS|FAIL) $s( |$)" "$STATUS"; then echo "== running: $s"; tail -3 "$LOGDIR/$s.out" 2>/dev/null; fi
      done;;
    snapshot-logs) snapshot_logs;;
    download|image|setup|hw|checks|reference|kernels|queue|bisect)
      local t0; t0=$(date +%s)
      grep -q " START $cmd" "$STATUS" 2>/dev/null || row "$STATUS" "START $cmd $*"
      if "stage_$cmd" "$@"; then row "$STATUS" "PASS $cmd $(( $(date +%s) - t0 ))s"
      else row "$STATUS" "FAIL $cmd $(( $(date +%s) - t0 ))s: see $LOGDIR/$cmd.out"; return 1; fi;;
    *) sed -n 2,24p "$0"; return 2;;
  esac
}

main "$@"
