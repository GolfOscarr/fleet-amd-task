# shellcheck shell=bash
# Shared by vm.sh and queue.sh (docs/round-2/01-preparation.md, P4).
# Sourced, not run. Every variable can be overridden from the environment,
# which is how the tests point the scripts at a temporary tree.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FLEET="${FLEET:-$ROOT/repos/fleet-chiplet-megakernel}"
LOGDIR="${LOGDIR:-$ROOT/env/logs}"
RECORD="${RECORD:-$ROOT/env/hw/$(date -u +%Y%m%d)}"     # never deleted by anything here
FLEET_OUT="${FLEET_OUT:-$ROOT/harness/fleet_out}"
STATUS="${STATUS:-$LOGDIR/session.status}"
QSTATUS="${QSTATUS:-$LOGDIR/queue.status}"
DRY="${DRY:-0}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_ID="deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
IMAGE_REPO="${IMAGE_REPO:-ghcr.io/golfoscarr/fleet-amd-task}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"     # the only device on a 1x VM

mkdir -p "$LOGDIR" "$LOGDIR/runs"

utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# run: execute, or print with DRY=1
run() {
  if [ "$DRY" = "1" ]; then echo "+ $*"; return 0; fi
  "$@"
}

# one row in a status file: "<utc> <words>"
row() { local f="$1"; shift; echo "$(utc) $*" | tee -a "$f"; }

# the model snapshot directory, or empty
snap_dir() {
  if [ -n "${SNAP:-}" ]; then echo "$SNAP"; return; fi
  ls -d "$HF_CACHE/hub/models--deepseek-ai--DeepSeek-Coder-V2-Lite-Base/snapshots"/* 2>/dev/null | head -1
}

# the Fleet environment for a graph run (no-op under DRY without the venv)
fleet_env() {
  export MIRAGE_HOME="$FLEET" AMDGPU_TARGETS=gfx942 ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
  export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH"
  # shellcheck disable=SC1091
  [ -f "$ROOT/.venv-fleet/bin/activate" ] && source "$ROOT/.venv-fleet/bin/activate"
  return 0
}

# copy a directory's small report files into the record (no tensors, no build tree, no log over 400 KB)
record_copy() {
  local src="$1" dst="$2"
  mkdir -p "$dst"
  run rsync -a --exclude '*.safetensors' --exclude 'build/' --exclude '__pycache__' \
    --max-size=400k "$src/" "$dst/"
}
