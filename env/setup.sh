#!/usr/bin/env bash
# Day-1 setup on the MI300X box (docs/design-doc/08-milestones.md, Day 1, item 1-2, 6).
#
#   bash env/setup.sh            # everything; exit status is gate 1 (the gfx942 build)
#   SKIP_DOWNLOAD=1 bash env/setup.sh
#
# Idempotent: re-running skips the venv, the download and an already-applied
# patch, and rebuilds Fleet. Everything is logged to env/logs/setup.<date>.log.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLEET="$ROOT/repos/fleet-chiplet-megakernel"
PATCH="$ROOT/fleet/patches/gfx942.patch"
LOGDIR="$ROOT/env/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/setup.$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "== setup.sh $(date -Is) on $(hostname); log $LOG"

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export ROCM_PATH
export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH"
MODEL="deepseek-ai/DeepSeek-Coder-V2-Lite-Base"

step() { echo; echo "---- $*"; }
try_cmds() {
  # run the first command of a list that succeeds; print which one it was
  local label="$1"; shift
  local c
  for c in "$@"; do
    if out=$(bash -c "$c" 2>&1); then
      echo "[$label] worked: $c"
      echo "$out"
      return 0
    fi
  done
  echo "[$label] none of the candidate commands worked:"
  printf '   %s\n' "$@"
  return 1
}

# ---------------------------------------------------------------------------
step "1. environment record"
try_cmds "rocminfo" "rocminfo | head -40" || true
try_cmds "hipcc" "hipcc --version" || true
try_cmds "rocm version" "cat $ROCM_PATH/.info/version" "cat $ROCM_PATH/.info/version-dev" || true
try_cmds "amd-smi version" "amd-smi version" "rocm-smi --version" || true
try_cmds "partition" \
  "amd-smi static --partition" \
  "amd-smi partition" \
  "amd-smi static -g 0 --partition" \
  "rocm-smi --showcomputepartition --showmemorypartition" || true
python3 --version
python3 -m pip list 2>/dev/null | grep -i "torch\|transformers" || echo "(no torch/transformers in the system python)"

ROCM_VER=$(cat "$ROCM_PATH/.info/version" 2>/dev/null | cut -d- -f1 || true)
ROCM_MM=$(echo "${ROCM_VER:-}" | cut -d. -f1,2)
echo "ROCm version: ${ROCM_VER:-unknown} (major.minor ${ROCM_MM:-unknown})"

# ---------------------------------------------------------------------------
step "2. venv and Python packages"
cd "$ROOT"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
# PyTorch for ROCm: the wheel index is per ROCm major.minor; fall back to the
# newest index we know if the exact one does not exist.
TORCH_INDEX="https://download.pytorch.org/whl/rocm${ROCM_MM:-6.2}"
if ! curl -sSf -o /dev/null "$TORCH_INDEX/torch/" 2>/dev/null; then
  echo "WARNING: no PyTorch wheel index at $TORCH_INDEX for ROCm $ROCM_MM; falling back to rocm6.2"
  echo "         (check https://pytorch.org/get-started/locally/ for the index matching ROCm $ROCM_MM)"
  TORCH_INDEX="https://download.pytorch.org/whl/rocm6.2"
fi
echo "torch index: $TORCH_INDEX"
python -m pip install --index-url "$TORCH_INDEX" torch
python -m pip install -r env/requirements.txt "huggingface_hub[cli]" cmake ninja cython
python -c "import torch; print('torch', torch.__version__, 'hip', torch.version.hip, 'cuda avail', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
python -c "import transformers; print('transformers', transformers.__version__)"

# ---------------------------------------------------------------------------
step "3. model download (31 GB, background)"
if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
  echo "skipped (SKIP_DOWNLOAD=1)"
elif python -c "
from huggingface_hub import try_to_load_from_cache
import sys
p = try_to_load_from_cache('$MODEL', 'model-00004-of-000004.safetensors')
sys.exit(0 if isinstance(p, str) else 1)"; then
  echo "already in the HF cache"
else
  DL_LOG="$LOGDIR/download.$(date +%Y%m%d-%H%M%S).log"
  nohup huggingface-cli download "$MODEL" > "$DL_LOG" 2>&1 &
  echo "downloading in the background, pid $!, log $DL_LOG"
  echo "wait with:   tail -f $DL_LOG      or   wait $!   (same shell)"
fi

# ---------------------------------------------------------------------------
step "4. Fleet submodules (CK, cutlass, json, z3 are nested submodules of the Fleet repo)"
cd "$FLEET"
git submodule update --init --recursive
ls deps

# ---------------------------------------------------------------------------
step "5. gfx942 patch (fleet/patches/README.md)"
if git apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "already applied"
else
  git apply --check "$PATCH"
  git apply "$PATCH"
  echo "applied"
fi
git status --short

# ---------------------------------------------------------------------------
step "6. build Fleet for gfx942 (gate 1)"
# Entry point per the repo README: pip install -e . -v with config.cmake (USE_ROCM ON).
# CMakeLists.txt reads AMDGPU_TARGETS into CMAKE_HIP_ARCHITECTURES (default gfx950).
# setup.py also builds two Rust crates with cargo (installs rustup if missing).
export AMDGPU_TARGETS=gfx942
export MIRAGE_HOME="$FLEET"
BUILD_LOG="$LOGDIR/build.$(date +%Y%m%d-%H%M%S).log"
if AMDGPU_TARGETS=gfx942 python -m pip install -e . -v 2>&1 | tee "$BUILD_LOG"; then
  BUILD_OK=1
else
  BUILD_OK=0
fi
cd "$ROOT"

# ---------------------------------------------------------------------------
step "summary"
echo "log:            $LOG"
echo "build log:      $BUILD_LOG"
echo "AMDGPU_TARGETS: gfx942"
echo "MIRAGE_HOME:    $FLEET   (export it before running any graph)"
if [ "$BUILD_OK" = "1" ] && python -c "import mirage" 2>/dev/null; then
  echo "GATE 1: PASS - Fleet built for gfx942 and imports"
  echo "next: bash env/check_day1.sh"
else
  echo "GATE 1: FAIL - Fleet did not build or does not import; see $BUILD_LOG"
  echo "  first errors:"
  grep -n -m 10 "error:" "$BUILD_LOG" || true
  echo "  fallback runtime: docs/design-doc/08-milestones.md, 'Fallback runtime'"
  exit 1
fi
