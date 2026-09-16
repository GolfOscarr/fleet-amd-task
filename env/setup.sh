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
step "2. two venvs: .venv (reference, transformers 4.46.3) and .venv-fleet (Fleet, its own pins)"
# Why two: the checkpoint's remote modeling code needs transformers 4.x up to
# 4.46 (env/requirements.txt), while Fleet's install_requires pins 4.57.1 and
# the day-1 Qwen3 smoke graph needs >= 4.51 (env/requirements-fleet.txt).
# A single interpreter cannot hold both; nothing needs both at once.
cd "$ROOT"
# Fleet's README requires ROCm 7.0+ (repos/fleet-chiplet-megakernel/README.md).
ROCM_MAJOR=$(echo "${ROCM_MM:-0}" | cut -d. -f1)
if [ "${ROCM_MAJOR:-0}" -lt 7 ] 2>/dev/null; then
  echo "WARNING: ROCm ${ROCM_VER:-unknown} is below the 7.0 Fleet requires; the build may still work, record the outcome"
fi
# PyTorch for ROCm: the wheel index is per ROCm major.minor (rocm6.3, 6.4,
# 7.0, 7.1, 7.2 all exist as of 2026-09); fall back to rocm7.0, the oldest
# Fleet supports, if the exact one does not exist.
TORCH_INDEX="https://download.pytorch.org/whl/rocm${ROCM_MM:-7.0}"
if ! curl -sSf -o /dev/null "$TORCH_INDEX/torch/" 2>/dev/null; then
  echo "WARNING: no PyTorch wheel index at $TORCH_INDEX for ROCm $ROCM_MM; falling back to rocm7.0"
  echo "         (check https://pytorch.org/get-started/locally/ for the index matching ROCm $ROCM_MM)"
  TORCH_INDEX="https://download.pytorch.org/whl/rocm7.0"
fi
echo "torch index: $TORCH_INDEX"

# The Hot Aisle image ships python3 without ensurepip (2026-09-15): a venv is
# created but has no pip and the first install aborts. python3-venv from apt
# fixes it; apt-get update first, the stale lists 404 on the package.
if ! python3 -c "import ensurepip" 2>/dev/null; then
  echo "ensurepip missing: installing python3-venv (needs sudo)"
  PYMM=$(python3 -c "import sys; print('%d.%d' % sys.version_info[:2])")
  sudo apt-get update -qq && sudo apt-get install -y -qq "python${PYMM}-venv"
fi

make_venv() {
  # $1 venv dir, $2 requirements file
  if [ ! -x "$1/bin/pip" ]; then
    rm -rf "$1"
    python3 -m venv "$1"
  fi
  "$1/bin/python" -m pip install -q --upgrade pip
  "$1/bin/python" -m pip install --index-url "$TORCH_INDEX" torch
  "$1/bin/python" -m pip install -r "$2"
  "$1/bin/python" -c "import torch; print('$1: torch', torch.__version__, 'hip', torch.version.hip, 'cuda avail', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
}
make_venv "$ROOT/.venv" "$ROOT/env/requirements.txt"
"$ROOT/.venv/bin/python" -c "import transformers; print('.venv: transformers', transformers.__version__)"
make_venv "$ROOT/.venv-fleet" "$ROOT/env/requirements-fleet.txt"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

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
step "4. Fleet dependencies under deps/ (not submodules at 51dce4f)"
# .gitmodules lists composable_kernel, cutlass, json and z3, but the tree at
# 51dce4f carries a gitlink for none of them (git ls-tree HEAD deps/ shows
# only the vendored deps/rocblas), so `git submodule update` fetches nothing.
# What the ROCm build actually reads: deps/composable_kernel/include (the
# JIT compile line, persistent_kernel.py:293), deps/json (add_subdirectory
# in CMakeLists.txt:172) and deps/cutlass/include (setup.py copy_include(),
# which copytree's it unconditionally, ROCm included, right before the
# Python install; the CMake include dirs are CUDA-only, the copy is not).
# z3 comes from the z3-solver wheel in Fleet's install_requires.
# The commits are the ones Fleet's other branches pin (git ls-tree 58a33d8
# deps/). The patched headers and our kernels were compiled offline for
# gfx942 against this CK (env/offline_gfx942/README.md).
CK_COMMIT=d8ee107a47d8485dbcffc79eb08e4f7c39ea6335
JSON_COMMIT=8c391e04fe4195d8be862c97f38cfe10e2a3472e
CUTLASS_COMMIT=f3fde58372d33e9a5650ba7b80fc48b3b49d40c8
fetch_dep() {
  # $1 deps/<name>, $2 url, $3 commit, $4 a file that proves the checkout
  if [ ! -f "$1/$4" ]; then
    rm -rf "$1"
    git init -q "$1"
    git -C "$1" remote add origin "$2"
    git -C "$1" fetch -q --depth 1 origin "$3"
    git -C "$1" checkout -q FETCH_HEAD
  fi
  echo "$1: $(git -C "$1" log -1 --format='%H %cd')"
}
cd "$FLEET"
fetch_dep deps/composable_kernel https://github.com/ROCm/composable_kernel.git "$CK_COMMIT" include/ck_tile/core.hpp
fetch_dep deps/json https://github.com/nlohmann/json.git "$JSON_COMMIT" include/nlohmann/json.hpp
fetch_dep deps/cutlass https://github.com/NVIDIA/cutlass.git "$CUTLASS_COMMIT" include/cutlass/cutlass.h
ls deps

# ---------------------------------------------------------------------------
step "5. gfx942 patch (fleet/patches/README.md)"
# A tree that carries an older version of a patch (an image or a VM from a
# previous session after a code rsync) passes neither check: the tracked
# files of the submodule are reset to the pinned commit and all three patches
# apply again below (the untracked deps/ and the new task kernels are kept).
if git apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "already applied"
else
  if ! git apply --check "$PATCH" 2>/dev/null; then
    echo "an older version of the patches is in the tree: resetting the tracked files to $(git rev-parse --short HEAD)"
    git checkout -q -- .
  fi
  git apply --check "$PATCH"
  git apply "$PATCH"
  echo "applied"
fi
git status --short

# ---------------------------------------------------------------------------
step "5b. new task kernels and their registration glue (fleet/tasks/README.md)"
cp "$ROOT"/fleet/tasks/mi300/*.cuh "$FLEET/include/mirage/persistent_kernel/tasks/mi300/"
ls "$FLEET"/include/mirage/persistent_kernel/tasks/mi300/ | grep -E "mla_|moe_router|copy_mi300"
PATCH2="$ROOT/fleet/patches/new_tasks.patch"
if git apply --reverse --check "$PATCH2" 2>/dev/null; then
  echo "already applied"
else
  git apply --check "$PATCH2"
  git apply "$PATCH2"
  echo "applied"
fi
git status --short

# ---------------------------------------------------------------------------
step "5c. scheduler queue indexed by the discovered XCD (fleet/patches/README.md, MIN-25)"
# The dispatcher placed block k on XCD (k + 4) mod 8 on the first VM
# (env/hw/20260915, F2-F4); the stock scheduler reads queue k by block id.
PATCH3="$ROOT/fleet/patches/sched_xcd.patch"
if git apply --reverse --check "$PATCH3" 2>/dev/null; then
  echo "already applied"
else
  git apply --check "$PATCH3"
  git apply "$PATCH3"
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
# Into the Fleet venv: its install_requires (transformers 4.57.1, z3-solver,
# accelerate, tg4perfetto from git) must not touch the reference venv.
deactivate 2>/dev/null || true
# shellcheck disable=SC1091
source "$ROOT/.venv-fleet/bin/activate"
# pip's isolated build environment installs its own z3-solver (unpinned in
# Fleet's pyproject build-requires): on 2026-09-15 it linked core.so against
# libz3.so.5.1 while the venv held the pinned 4.15, and import failed; a
# PIP_CONSTRAINT on the build environment did not change that in the image
# build. So the build runs without isolation, against the venv's own cmake,
# cython, setuptools, graphviz and z3 (env/requirements-fleet.txt), and the
# venv's z3/lib goes on the loader path through activate (no wheel puts it
# there by itself).
Z3LIB="$ROOT/.venv-fleet/lib/$("$ROOT/.venv-fleet/bin/python" -c 'import sys; print("python%d.%d" % sys.version_info[:2])')/site-packages/z3/lib"
if ! grep -q "LD_LIBRARY_PATH.*z3/lib" "$ROOT/.venv-fleet/bin/activate"; then
  echo "export LD_LIBRARY_PATH=$Z3LIB:\${LD_LIBRARY_PATH:-}" >> "$ROOT/.venv-fleet/bin/activate"
fi
export LD_LIBRARY_PATH="$Z3LIB:${LD_LIBRARY_PATH:-}"
if AMDGPU_TARGETS=gfx942 python -m pip install -e . -v --no-build-isolation 2>&1 | tee "$BUILD_LOG"; then
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
echo "MIRAGE_HOME:    $FLEET"
echo "venvs:          .venv (run_reference.py, calibrate.py, reassoc_check.py, make_prompt.py)"
echo "                .venv-fleet (run_fleet.py, kernel_tests.py, compare.py, measure.py, the Qwen3 smoke graph)"
echo "export MIRAGE_HOME=$FLEET AMDGPU_TARGETS=gfx942 before running any graph: the compile"
echo "step in persistent_kernel.py defaults --offload-arch to gfx950 (run_fleet.py sets it itself)"
if [ "$BUILD_OK" = "1" ] && (cd "$FLEET" && "$ROOT/.venv-fleet/bin/python" -c "import mirage") 2>/dev/null; then
  echo "GATE 1: PASS - Fleet built for gfx942 and imports"
  echo "next: bash env/check_day1.sh"
else
  echo "GATE 1: FAIL - Fleet did not build or does not import; see $BUILD_LOG"
  echo "  first errors:"
  grep -n -m 10 "error:" "$BUILD_LOG" || true
  echo "  fallback runtime: docs/design-doc/08-milestones.md, 'Fallback runtime'"
  exit 1
fi
