#!/usr/bin/env bash
# The session stages on the VM (docs/gpu-experiments/02-validation/01-preparation.md, P4; 02-session-plan.md).
#
#   bash env/session/vm.sh <stage> [args]          # run a stage in the foreground
#   bash env/session/vm.sh start <stage> [args]    # run it detached: env/logs/<stage>.out, a row in env/logs/session.status
#   bash env/session/vm.sh status                  # the status files and the last lines of the running stage logs
#   bash env/session/vm.sh snapshot-logs           # copy env/logs/*.out and the status files into the record (before a pull)
#   bash env/session/vm.sh check <stage>           # one line: the PASS/FAIL row of the stage's last start, or RUNNING / NOT STARTED
#   bash env/session/vm.sh wait <stage> [minutes]  # block until that row appears (default 60 minutes), print it and the log tail
#   bash env/session/vm.sh preflight               # ten seconds: GPU visible, disk, docker, hipcc, rocprofv3, the venvs, the model, the reference
#   bash env/session/vm.sh kill <pattern>|--all    # stop running graph runs by anchored pid (never pkill -f)
#   bash env/session/vm.sh gdb <label> [args]      # the A7b recipe: compile with line tables, run the truncated graph under rocgdb, save the backtrace
#
# Stages, in session order:
#   download    the model into the Hugging Face cache (79 s on 2026-09-15)
#   image       docker build of env/docker/Dockerfile; push if a registry login exists (laptop.sh login); logout
#   setup       env/setup.sh with SKIP_DOWNLOAD=1 (both venvs, the patches, the Fleet build; gate 1)
#   hw          env/collect_hw.sh, skipped when this host and ROCm version are already in an env/hw/*/summary.md
#   checks      env/check_day1.sh (7 PASS lines)
#   reference   run_reference.py, calibrate.py, route_analysis.py under .venv (the reference tensors are not in the repo)
#   kernels     the kernel-test launcher (built if missing) and kernel_tests.py, 100 trials per suite
#   ktime [variant] [launches] [copies]   the standalone times (KT_TIME) of the attention, the merges, the
#               GEMV linear, the router and the gang GEMV forms; round 4's variants nt, nt_b4, nt_b16,
#               nt_strided (G0's batch constant and lane map under the graph rows' load policy)
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
  run rm -f harness/ref/calibration.json     # the floor is recorded once per machine; the tracked one is the laptop's copy
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
  # O6 (docs/gpu-experiments/03-acceleration): the streaming-loads build, for KT_TIME against the plain one
  # shellcheck disable=SC2086
  [ -x fleet/tasks/build/kernel_tests_nt ] || run hipcc --offload-arch=gfx942 -O2 -std=c++17 $defs $inc \
      -DMLA_NT_STREAMS fleet/tasks/kernel_tests_mi300.cu -o fleet/tasks/build/kernel_tests_nt || return 1
  # O7: the MFMA attention; run the suite against it with kernel_tests.py --bin fleet/tasks/build/kernel_tests_mfma
  # shellcheck disable=SC2086
  [ -x fleet/tasks/build/kernel_tests_mfma ] || run hipcc --offload-arch=gfx942 -O2 -std=c++17 $defs $inc \
      -DMLA_ATTEND_MFMA fleet/tasks/kernel_tests_mi300.cu -o fleet/tasks/build/kernel_tests_mfma || return 1
  # shellcheck disable=SC2086
  [ -x fleet/tasks/build/kernel_tests_mfma_debug ] || run hipcc --offload-arch=gfx942 -O2 -std=c++17 $defs $inc \
      -DMLA_ATTEND_MFMA -DMLA_ATTEND_DEBUG_SCORES fleet/tasks/kernel_tests_mi300.cu -o fleet/tasks/build/kernel_tests_mfma_debug || return 1
  # round 4 (L3, L4): the gang GEMV forms take the XCD id from blockIdx.y under -DKT_FAKE_XCD; without
  # this build kernel_tests.py SKIPs the gang_w13_gemv and gang_w2_gemv rows
  # shellcheck disable=SC2086
  [ -x fleet/tasks/build/kernel_tests_xcd ] || run hipcc --offload-arch=gfx942 -O2 -std=c++17 $defs $inc \
      -DKT_FAKE_XCD fleet/tasks/kernel_tests_mi300.cu -o fleet/tasks/build/kernel_tests_xcd || return 1
}

# The standalone timing builds of round 4 (G0 and H0 of docs/gpu-experiments/04-kernels: the batch
# constant and the lane map of the GEMV linear and the router, under the graph rows' load policy
# -DMLA_NT_STREAMS): the nt build at 4 and 16 rows per batch, the strided map, and the nt build with
# the fake XCD for the gang forms. Built by the ktime stage when its variant names one of them.
build_ktime_variant() {
  local v="$1" flags
  case "$v" in
    nt_xcd) flags="-DMLA_NT_STREAMS -DKT_FAKE_XCD";;
    nt_b4) flags="-DMLA_NT_STREAMS -DGEMV_BATCH=4 -DROUTER_BATCH=4";;
    nt_b16) flags="-DMLA_NT_STREAMS -DGEMV_BATCH=16 -DROUTER_BATCH=16";;
    nt_strided) flags="-DMLA_NT_STREAMS -DGEMV_STRIDED -DROUTER_STRIDED";;
    *) return 0;;                                    # the round-3 builds come from build_kernel_tests
  esac
  local defs="-D__HIP_PLATFORM_AMD__=1 -DMIRAGE_AMD_MI300 -DMIRAGE_BACKEND_USE_ROCM -DMPK_TARGET_CC=94 -DMODE_ONLINE"
  local inc="-I fleet -I $FLEET/include -I $FLEET/include/mirage/persistent_kernel"
  # shellcheck disable=SC2086
  [ -x "fleet/tasks/build/kernel_tests_$v" ] || run hipcc --offload-arch=gfx942 -O2 -std=c++17 $defs $inc \
      $flags fleet/tasks/kernel_tests_mi300.cu -o "fleet/tasks/build/kernel_tests_$v" || return 1
}

# kernels [variant]: the suites against fleet/tasks/build/kernel_tests (no argument) or against
# kernel_tests_<variant> and kernel_tests_<variant>_debug (round 3: mfma for O7, nt for O6; the nt
# build has no debug twin, so its scores suite runs on the plain debug binary)
stage_kernels() {
  fleet_env
  local variant="${1:-}" bin="fleet/tasks/build/kernel_tests" dbg="fleet/tasks/build/kernel_tests_debug" out="fleet/tasks/results"
  local xcd="fleet/tasks/build/kernel_tests_xcd"
  build_kernel_tests || { echo "kernel_tests did not build"; return 1; }
  # the gang GEMV rows run on the variant's _xcd build when there is one (nt: built here), else the plain one
  [ "$variant" = "nt" ] && { build_ktime_variant nt_xcd || { echo "kernel_tests_nt_xcd did not build"; return 1; }; }
  if [ -n "$variant" ]; then
    bin="fleet/tasks/build/kernel_tests_$variant"
    [ -x "fleet/tasks/build/kernel_tests_${variant}_debug" ] && dbg="fleet/tasks/build/kernel_tests_${variant}_debug"
    [ -x "fleet/tasks/build/kernel_tests_${variant}_xcd" ] && xcd="fleet/tasks/build/kernel_tests_${variant}_xcd"
    out="fleet/tasks/results_$variant"
  fi
  run python fleet/tasks/kernel_tests.py --n 100 --bin "$bin" --bin-debug "$dbg" --bin-xcd "$xcd" --out "$out/kernel_tests.json" || return 1
  [ "$DRY" = "1" ] && return 0
  local fails; fails="$(grep -c '"FAIL"' "$out/kernel_tests.json" || true)"
  [ "$fails" = "0" ] || { echo "$fails suite(s) FAIL in $out/kernel_tests.json"; return 1; }
  record_copy "$out" "$RECORD/kernel_tests${variant:+_$variant}"
}

# ktime [variant] [launches] [cache copies]: the standalone time of the attention and the merge grids
# (KT_TIME, KT_COLD of fleet/tasks/README.md) on one trial directory, and the spin's [SPIN] line
# (KT_SPIN, I2); round 3 compares the VALU, nt and mfma builds this way (G2, G7). Round 4 adds the
# GEMV linear (qkva's norm form and o_proj's residual form, the weight rotated over the cache copies),
# the router (its 256 KB gate weight rotated the same way), the merge with o_proj folded in, and the
# gang w13 and w2 forms through the variant's _xcd build when it exists (kernel_tests_xcd for the plain
# build, kernel_tests_nt_xcd for nt; the depth and map variants time the dense kernels only).
stage_ktime() {
  fleet_env
  local variant="${1:-}" n="${2:-50}" cold="${3:-27}" bin="fleet/tasks/build/kernel_tests${1:+_$1}"
  local dir="$LOGDIR/ktime_trial" out="$RECORD/ktime"; mkdir -p "$out"
  local xcd="fleet/tasks/build/kernel_tests${variant:+_$variant}_xcd"
  build_kernel_tests || { echo "kernel_tests did not build"; return 1; }
  build_ktime_variant "$variant" || { echo "kernel_tests_$variant did not build"; return 1; }
  [ "$variant" = "nt" ] && { build_ktime_variant nt_xcd || { echo "kernel_tests_nt_xcd did not build"; return 1; }; }
  [ -d "$dir/mla_attend" ] || run python fleet/tasks/kernel_tests.py --n 1 --kernel mla_attend --kernel mla_merge_uv --kernel copy \
      --kernel linear_gemv_norm --kernel linear_gemv_res --kernel moe_router --kernel mla_merge_oproj \
      --kernel gang_w13_gemv --kernel gang_w2_gemv --work-dir "$dir" --keep || return 1
  if [ "$DRY" = "1" ]; then
    echo "+ KT_TIME=$n KT_COLD=$cold $bin mla_attend $dir/mla_attend/*"; echo "+ KT_TIME=$n $bin mla_merge_uv $dir/mla_merge_uv/*"
    echo "+ KT_SPIN=1000 $bin copy $dir/copy/*"
    echo "+ KT_TIME=$n KT_COLD=$cold $bin linear_gemv_norm $dir/linear_gemv_norm/*"; echo "+ KT_TIME=$n KT_COLD=$cold $bin linear_gemv_res $dir/linear_gemv_res/*"
    echo "+ KT_TIME=$n KT_COLD=$cold $bin moe_router $dir/moe_router/*"; echo "+ KT_TIME=$n $bin mla_merge_oproj $dir/mla_merge_oproj/*/oproj"
    echo "+ KT_TIME=$n $xcd gang_w13_gemv $dir/gang_w13_gemv/*"; echo "+ KT_TIME=$n $xcd gang_w2_gemv $dir/gang_w2_gemv/*"
    return 0
  fi
  {
    echo "### $(utc) $bin launches=$n cache_copies=$cold"
    KT_TIME="$n" KT_COLD="$cold" "$bin" mla_attend "$dir"/mla_attend/* 2>&1 | grep -E "TIME|ok"
    KT_TIME="$n" "$bin" mla_merge_uv "$dir"/mla_merge_uv/* 2>&1 | grep -E "TIME|ok"
    KT_SPIN=1000 "$bin" copy "$dir"/copy/* 2>&1 | grep -E "SPIN|ok"
    KT_TIME="$n" KT_COLD="$cold" "$bin" linear_gemv_norm "$dir"/linear_gemv_norm/* 2>&1 | grep -E "TIME|ok"
    KT_TIME="$n" KT_COLD="$cold" "$bin" linear_gemv_res "$dir"/linear_gemv_res/* 2>&1 | grep -E "TIME|ok"
    KT_TIME="$n" KT_COLD="$cold" "$bin" moe_router "$dir"/moe_router/* 2>&1 | grep -E "TIME|ok"
    KT_TIME="$n" "$bin" mla_merge_oproj "$dir"/mla_merge_oproj/*/oproj 2>&1 | grep -E "TIME|ok"
    if [ -x "$xcd" ]; then
      KT_TIME="$n" "$xcd" gang_w13_gemv "$dir"/gang_w13_gemv/* 2>&1 | grep -E "TIME|ok"
      KT_TIME="$n" "$xcd" gang_w2_gemv "$dir"/gang_w2_gemv/* 2>&1 | grep -E "TIME|ok"
    else
      echo "no $xcd: the gang rows are timed by the plain and nt variants only"
    fi
  } | tee -a "$out/ktime${variant:+_$variant}.txt"
}

# tgcheck <run name>: the side operators' wiring in the run's task graph (O8, fleet/task_graph_check.py);
# the JSON lives under the run's build/, which the record excludes, so only the verdict is kept
stage_tgcheck() {
  fleet_env
  local name="${1:-}"; [ -n "$name" ] || { echo "usage: vm.sh tgcheck <run name>"; return 2; }
  local tg="$FLEET_OUT/$name/build/task_graph_rank0.json"
  if [ "$DRY" = "1" ]; then echo "+ python fleet/task_graph_check.py $tg > $FLEET_OUT/$name/task_graph_check.txt"; return 0; fi
  [ -f "$tg" ] || { echo "no $tg"; return 1; }
  python fleet/task_graph_check.py "$tg" | tee "$FLEET_OUT/$name/task_graph_check.txt"
  mkdir -p "$RECORD/runs/$name" && cp "$FLEET_OUT/$name/task_graph_check.txt" "$RECORD/runs/$name/"
  grep -q "^PASS" "$FLEET_OUT/$name/task_graph_check.txt"
}

stage_queue() { fleet_env; bash env/session/queue.sh run "$@"; }
stage_bisect() { fleet_env; bash env/session/queue.sh bisect "$@"; }

# the PASS/FAIL row written after the stage's last START row, or RUNNING / NOT STARTED
stage_row() {
  local stage="$1"
  [ -f "$STATUS" ] || { echo "NOT STARTED $stage"; return 1; }
  local started; started="$(awk -v s="$stage" '$2 == "START" && $3 == s {n = NR} END {print n + 0}' "$STATUS")"
  [ "$started" != "0" ] || { echo "NOT STARTED $stage"; return 1; }
  local line; line="$(awk -v s="$stage" -v n="$started" 'NR > n && ($2 == "PASS" || $2 == "FAIL") && $3 == s' "$STATUS" | tail -1)"
  [ -n "$line" ] || { echo "RUNNING $stage (since $(sed -n "${started}p" "$STATUS" | cut -d' ' -f1))"; return 1; }
  echo "$line"
  [[ "$line" == *" PASS "* ]]
}

stage_wait() {
  local stage="$1" mins="${2:-60}" t0; t0=$(date +%s)
  while :; do
    local line; line="$(stage_row "$stage")"; local rc=$?
    case "$line" in
      "NOT STARTED"*) echo "$line"; return 2;;
      RUNNING*) ;;
      *) echo "$line"; tail -5 "$LOGDIR/$stage.out" 2>/dev/null; return $rc;;
    esac
    if [ $(( $(date +%s) - t0 )) -gt $(( mins * 60 )) ]; then
      echo "TIMEOUT $stage after $mins min"; tail -5 "$LOGDIR/$stage.out" 2>/dev/null; return 3
    fi
    sleep 20
  done
}

stage_preflight() {
  local bad=0 free
  p() { echo "  $*"; }
  if [ "$DRY" = "1" ]; then echo "+ amd-smi list; df -BG /; command -v docker hipcc rocprofv3 rocgdb"; return 0; fi
  local gpus; gpus="$(amd-smi list 2>/dev/null | grep -c "GPU:" || true)"
  [ "${gpus:-0}" -ge 1 ] && p "PASS gpu: $gpus device(s) (amd-smi list)" || { p "FAIL gpu: none visible (amd-smi list)"; bad=1; }
  free="$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')"
  [ "${free:-0}" -ge 100 ] && p "PASS disk: ${free} GB free on /" || { p "FAIL disk: ${free:-?} GB free on / (100 needed: model 30, image 30, build 10)"; bad=1; }
  local c
  for c in docker hipcc rocprofv3 rocgdb python3 rsync; do
    command -v "$c" >/dev/null 2>&1 && p "PASS $c: $(command -v "$c")" || p "INFO $c: not on PATH"
  done
  [ -x "$ROOT/.venv-fleet/bin/python" ] && p "PASS venv: .venv-fleet present" || p "INFO venv: .venv-fleet absent (setup stage builds it)"
  [ -x "$ROOT/.venv/bin/python" ] && p "PASS venv: .venv present" || p "INFO venv: .venv absent (setup stage builds it)"
  local d; d="$(snap_dir)"
  [ -n "$d" ] && p "PASS model: $d" || p "INFO model: not downloaded yet (download stage)"
  [ -f "$ROOT/harness/ref/ref_cache.safetensors" ] && p "PASS reference: tensors present" || p "INFO reference: tensors absent (reference stage)"
  [ -n "$(cat /opt/rocm/.info/version 2>/dev/null)" ] && p "PASS rocm: $(cat /opt/rocm/.info/version)" || p "INFO rocm: /opt/rocm/.info/version missing"
  p "record: $RECORD"
  return $bad
}

stage_kill() {
  local pat="${1:-}"
  [ -n "$pat" ] || { echo "usage: vm.sh kill <pattern>|--all"; return 2; }
  local pids
  # anchored on the python command, so this shell's own command line never matches
  pids="$(pgrep -f "^python harness/run_fleet.py" || true)"
  [ -n "$pids" ] || { echo "no graph run is running"; return 0; }
  local pid killed=""
  for pid in $pids; do
    local cmd; cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null)"
    if [ "$pat" = "--all" ] || [[ "$cmd" == *"$pat"* ]]; then
      run kill "$pid" && killed="$killed $pid"; echo "killed $pid: $cmd"
    fi
  done
  [ -n "$killed" ] && row "$QSTATUS" "KILLED$killed ($pat)"
  return 0
}

stage_gdb() {
  local label="${1:-}"; shift || true
  [ -n "$label" ] || { echo "usage: vm.sh gdb <label> [run_fleet args, default: --layers 8 --head --iters 2]"; return 2; }
  local args=("$@"); [ ${#args[@]} -gt 0 ] || args=(--layers 8 --head --iters 2)
  fleet_env
  export MPK_EXTRA_HIPCC_FLAGS="-gline-tables-only"      # the gfx942.patch hook (P1)
  local out="$LOGDIR/gdb_$label.out" snap; snap="$(snap_dir)"
  [ "$DRY" = "1" ] || command -v rocgdb >/dev/null 2>&1 || { echo "rocgdb not on PATH (apt: rocm-gdb)"; return 1; }
  echo "rocgdb on ${args[*]} --stop-after $label -> $out"
  if [ "$DRY" = "1" ]; then
    echo "+ rocgdb --batch -ex run -ex bt -ex 'info threads' --args python harness/run_fleet.py ${args[*]} --stop-after $label --model-dir $snap > $out"; return 0
  fi
  rocgdb --batch -ex run -ex bt -ex "info threads" --args python harness/run_fleet.py "${args[@]}" \
      --stop-after "$label" --model-dir "$snap" > "$out" 2>&1 || true
  grep -n -A12 "^#0 \|Thread .* received signal\|Memory access fault" "$out" | head -60
  mkdir -p "$RECORD/logs" && cp "$out" "$RECORD/logs/"
}

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
    check) stage_row "${1:-}";;
    wait) stage_wait "${1:-}" "${2:-60}";;
    kill) stage_kill "${1:-}";;
    gdb) stage_gdb "$@";;
    preflight|download|image|setup|hw|checks|reference|kernels|ktime|tgcheck|queue|bisect)
      local t0; t0=$(date +%s)
      grep -q " START $cmd" "$STATUS" 2>/dev/null || row "$STATUS" "START $cmd $*"
      if "stage_$cmd" "$@"; then row "$STATUS" "PASS $cmd $(( $(date +%s) - t0 ))s"
      else row "$STATUS" "FAIL $cmd $(( $(date +%s) - t0 ))s: see $LOGDIR/$cmd.out"; return 1; fi;;
    *) sed -n 2,30p "$0"; return 2;;
  esac
}

main "$@"
