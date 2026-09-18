#!/usr/bin/env bash
# F6 (docs/gpu-experiments/05-final): round 3's standalone launcher tree (fleet/tasks at the round-3
# merge commit 8946804) disassembled with run.sh's launcher line, so round 3's k_mla_merge_uv can be
# read beside round 4's (work/out/dev_kt.s, written by run.sh). Scratch: the tree is extracted into
# work/kt_r3/ (gitignored) and the assembly lands in work/out/dev_kt_r3.s. Needs run.sh's work
# directory (the patched fork copy) and the Docker image.
#
#   bash env/offline_gfx942/kt_r3.sh [commit]      # default 8946804
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/env/offline_gfx942"
WORK="$HERE/work"
IMAGE="${IMAGE:-rocm/dev-ubuntu-22.04:7.0}"
COMMIT="${1:-8946804}"
[ -d "$WORK/fleet/include" ] || { echo "run env/offline_gfx942/run.sh first (no $WORK/fleet)"; exit 2; }
rm -rf "$WORK/kt_r3"; mkdir -p "$WORK/kt_r3"
git -C "$ROOT" archive --format=tar "$COMMIT" fleet/tasks | tar -x -C "$WORK/kt_r3"
echo "== round-3 launcher tree at $(git -C "$ROOT" rev-parse --short "$COMMIT"): $(find "$WORK/kt_r3/fleet/tasks/mi300" -name '*.cuh' | wc -l | tr -d ' ') kernel headers"
docker run --rm --platform linux/amd64 -v "$WORK/kt_r3:/r3" -v "$WORK/fleet:/fleet" -v "$WORK/out:/out" "$IMAGE" bash -c "
  cd /r3 && hipcc -S -x hip --offload-device-only --offload-arch=gfx942 -O2 -std=c++17 \
    -D__HIP_PLATFORM_AMD__=1 -DMIRAGE_AMD_MI300 -DMIRAGE_BACKEND_USE_ROCM -DMPK_TARGET_CC=94 -DMODE_ONLINE \
    -I fleet -I /fleet/include -I /fleet/include/mirage/persistent_kernel \
    fleet/tasks/kernel_tests_mi300.cu -o /out/dev_kt_r3.s > /out/dev_kt_r3.log 2>&1; echo \$? > /out/dev_kt_r3.rc"
rc=$(cat "$WORK/out/dev_kt_r3.rc")
echo "round-3 launcher disassembly (dev_kt_r3.s): hipcc exit $rc, errors: $(grep -c 'error:' "$WORK/out/dev_kt_r3.log" || true)"
exit "$rc"
