#!/usr/bin/env bash
# Offline gfx942 compile of the patched Fleet megakernel headers and our five
# kernels with the real ROCm 7.0 hipcc, in Docker, on any machine (no GPU is
# needed to generate gfx942 code objects). Answers OPEN-PROBLEMS.md MAJ-1
# (does it compile for gfx942), MIN-27 (the gfx950-only items) and the
# register-union part of MAJ-4 before day 1. See README.md here.
#
#   bash env/offline_gfx942/run.sh            # both variants; exit 0 = both compile and link
#
# Work dir env/offline_gfx942/work/ (gitignored): a patched copy of the
# submodule, composable_kernel at Fleet's pinned commit, nlohmann/json.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/env/offline_gfx942"
FLEET="$ROOT/repos/fleet-chiplet-megakernel"
WORK="$HERE/work"
IMAGE="${IMAGE:-rocm/dev-ubuntu-22.04:7.0}"
CK_COMMIT=d8ee107a47d8485dbcffc79eb08e4f7c39ea6335   # env/setup.sh step 4
JSON_TAG=v3.11.3
mkdir -p "$WORK/out"

echo "== patched copy of the submodule"
rm -rf "$WORK/fleet"
git -C "$FLEET" archive --format=tar HEAD | tar -x -C "$WORK" --one-top-level=fleet 2>/dev/null \
  || { mkdir -p "$WORK/fleet" && git -C "$FLEET" archive --format=tar HEAD | tar -x -C "$WORK/fleet"; }
patch -p1 -s -d "$WORK/fleet" < "$ROOT/fleet/patches/gfx942.patch"
patch -p1 -s -d "$WORK/fleet" < "$ROOT/fleet/patches/new_tasks.patch"
cp "$ROOT"/fleet/tasks/mi300/*.cuh "$WORK/fleet/include/mirage/persistent_kernel/tasks/mi300/"

echo "== composable_kernel at $CK_COMMIT and json $JSON_TAG"
if [ ! -f "$WORK/ck/include/ck_tile/core.hpp" ]; then
  rm -rf "$WORK/ck"; git init -q "$WORK/ck"
  git -C "$WORK/ck" remote add origin https://github.com/ROCm/composable_kernel.git
  git -C "$WORK/ck" fetch -q --depth 1 origin "$CK_COMMIT"
  git -C "$WORK/ck" checkout -q FETCH_HEAD
fi
if [ ! -f "$WORK/json/include/nlohmann/json.hpp" ]; then
  git clone -q --depth 1 --branch "$JSON_TAG" https://github.com/nlohmann/json.git "$WORK/json"
fi
mkdir -p "$WORK/fleet/deps" && rm -rf "$WORK/fleet/deps/json" && ln -s "$WORK/json" "$WORK/fleet/deps/json"

echo "== $IMAGE (amd64; runs under emulation on Apple silicon)"
docker pull -q --platform linux/amd64 "$IMAGE" >/dev/null

# Flags: persistent_kernel.py's ROCm compile line for mode=online, batch 1,
# max_seq_length 1056, USE_GANG=1. MPK_USE_CK_FMHA is what the generator
# defines when CK FMHA tasks are registered (runtime.cc:701-706): our graph
# registers none, the day-1 Qwen3 smoke graph does, so both are compiled.
compile() {
  local variant="$1"; shift
  docker run --rm --platform linux/amd64 \
    -v "$WORK/fleet:/fleet" -v "$WORK/ck:/ck" -v "$WORK/json:/json" -v "$HERE:/here" -v "$WORK/out:/out" \
    "$IMAGE" bash -c "
      cd /out && hipcc -c -x hip /here/mk_tu.cu -o /out/mk_$variant.o \
        -O3 -std=c++17 -fPIC --offload-arch=gfx942 \
        -I/fleet/include -I/fleet/include/mirage/persistent_kernel \
        -I/fleet/deps/rocblas/include -I/ck/include -I/opt/rocm/include -I/json/include \
        -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -DMAX_WORKER_PER_SCHEDULER=38 -DMIRAGE_USE_CUTLASS_KERNEL=0 \
        -D__HIP_PLATFORM_AMD__=1 -DMIRAGE_AMD_MI300 -DMIRAGE_BACKEND_USE_ROCM -DMPK_TARGET_CC=94 \
        -DMPK_MAX_NUM_BATCHED_REQUESTS=1 -DMPK_MAX_NUM_BATCHED_TOKENS=1 -DMPK_MAX_NUM_PAGES=1 \
        -DMPK_PAGE_SIZE=64 -DMPK_MAX_SEQ_LENGTH=1056 -DMODE_ONLINE -DMPK_PROFILING_NUM_ITERS=0 \
        -DMPK_ENABLE_GANG_TASKS $* \
        -Rpass-analysis=kernel-resource-usage > /out/mk_$variant.log 2>&1; echo \$? > /out/mk_$variant.rc
      hipcc --version | head -2 > /out/hipcc.txt; cat /opt/rocm/.info/version >> /out/hipcc.txt"
  local rc; rc=$(cat "$WORK/out/mk_$variant.rc")
  echo "variant $variant: hipcc exit $rc, errors: $(grep -c 'error:' "$WORK/out/mk_$variant.log" || true)"
  return "$rc"
}

ok=0
compile ours || ok=1
compile ckfmha -DMPK_USE_CK_FMHA=1 || ok=1
compile debugscores -DMLA_ATTEND_DEBUG_SCORES=1 || ok=1

# The gfx942 assembly of the "ours" variant, for the fence question (MAJ-3):
# does __builtin_amdgcn_fence(..., "agent") lower to buffer_wbl2 sc1 on
# release and buffer_inv sc1 on acquire? Counted per kernel into fences.txt.
docker run --rm --platform linux/amd64 \
  -v "$WORK/fleet:/fleet" -v "$WORK/ck:/ck" -v "$WORK/json:/json" -v "$HERE:/here" -v "$WORK/out:/out" \
  "$IMAGE" bash -c "
    hipcc -S -x hip /here/mk_tu.cu --offload-device-only -o /out/dev_ours.s \
      -O3 -std=c++17 --offload-arch=gfx942 \
      -I/fleet/include -I/fleet/include/mirage/persistent_kernel \
      -I/fleet/deps/rocblas/include -I/ck/include -I/opt/rocm/include -I/json/include \
      -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -DMAX_WORKER_PER_SCHEDULER=38 -DMIRAGE_USE_CUTLASS_KERNEL=0 \
      -D__HIP_PLATFORM_AMD__=1 -DMIRAGE_AMD_MI300 -DMIRAGE_BACKEND_USE_ROCM -DMPK_TARGET_CC=94 \
      -DMPK_MAX_NUM_BATCHED_REQUESTS=1 -DMPK_MAX_NUM_BATCHED_TOKENS=1 -DMPK_MAX_NUM_PAGES=1 \
      -DMPK_PAGE_SIZE=64 -DMPK_MAX_SEQ_LENGTH=1056 -DMODE_ONLINE -DMPK_PROFILING_NUM_ITERS=0 \
      -DMPK_ENABLE_GANG_TASKS > /out/dev_ours.log 2>&1"
python3 - "$WORK/out/dev_ours.s" > "$HERE/fences.txt" <<'PYEOF'
import re, sys, collections
lines = open(sys.argv[1]).read().splitlines()
fn, counts = "(none)", collections.defaultdict(collections.Counter)
kinds = ("buffer_wbl2 sc0 sc1", "buffer_wbl2 sc1", "buffer_wbl2 sc0", "buffer_inv sc0 sc1",
         "buffer_inv sc1", "buffer_inv sc0", "s_getreg_b32", "s_sleep")
for l in lines:
    m = re.match(r"^([_A-Za-z][\w.$@]*):", l)
    if m and not m.group(1).startswith(".L"):
        fn = m.group(1)
        continue
    t = l.strip()
    for k in kinds:
        if t.startswith(k) and not (k.endswith("sc0") and t.startswith(k + " sc1")):
            counts[fn][k] += 1
            break
print("# static instruction counts per function in the gfx942 assembly of mk_tu.cu (variant ours)")
print("# sc1 = agent scope (the cross-XCD fence the design relies on); sc0 sc1 = system scope")
print("# (printf/assert hostcall paths and __threadfence()); s_getreg_b32 = the HW_REG_XCC_ID read")
print(f"# {'function':<58} " + " ".join(f"{k:>20}" for k in kinds))
for f, c in counts.items():
    print(f"{f[:58]:<58} " + " ".join(f"{c.get(k, 0):>20}" for k in kinds))
PYEOF
cat "$HERE/fences.txt"

# One table per variant: every kernel's registers, spills, LDS, occupancy.
{
  echo "# Offline gfx942 compile, $(date -u +%Y-%m-%dT%H:%M:%SZ), $(cat "$WORK/out/hipcc.txt" | tr '\n' ' ')"
  echo "# fleet 51dce4f + gfx942.patch + new_tasks.patch; composable_kernel $CK_COMMIT; json $JSON_TAG"
  for v in ours ckfmha debugscores; do
    echo; echo "## variant $v (hipcc exit $(cat "$WORK/out/mk_$v.rc"))"
    grep -E "Function Name|    VGPRs:|AGPRs|SGPRs Spill|VGPRs Spill|LDS Size|ScratchSize|Occupancy" "$WORK/out/mk_$v.log" \
      | sed 's/.*remark: *//; s/ \[-Rpass.*//; s/Function Name: //' | paste - - - - - - - - \
      | grep -v flush_cache | awk -F'\t' '{printf "%-58s %-12s %-10s %-27s %-25s %-17s %-17s %s\n",$1,$2,$3,$4,$5,$6,$7,$8}'
    grep -E "error:" "$WORK/out/mk_$v.log" | head -10 || true
  done
} > "$HERE/resources.txt"
cat "$HERE/resources.txt"
exit $ok
