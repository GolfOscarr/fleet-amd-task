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
CK_COMMIT=d8ee107a47d8485dbcffc79eb08e4f7c39ea6335     # env/setup.sh step 4
JSON_COMMIT=8c391e04fe4195d8be862c97f38cfe10e2a3472e   # same source
mkdir -p "$WORK/out"

echo "== patched copy of the submodule"
rm -rf "$WORK/fleet"
git -C "$FLEET" archive --format=tar HEAD | tar -x -C "$WORK" --one-top-level=fleet 2>/dev/null \
  || { mkdir -p "$WORK/fleet" && git -C "$FLEET" archive --format=tar HEAD | tar -x -C "$WORK/fleet"; }
patch -p1 -s -d "$WORK/fleet" < "$ROOT/fleet/patches/gfx942.patch"
patch -p1 -s -d "$WORK/fleet" < "$ROOT/fleet/patches/new_tasks.patch"
patch -p1 -s -d "$WORK/fleet" < "$ROOT/fleet/patches/sched_xcd.patch"
cp "$ROOT"/fleet/tasks/mi300/*.cuh "$WORK/fleet/include/mirage/persistent_kernel/tasks/mi300/"

echo "== composable_kernel at $CK_COMMIT and json at $JSON_COMMIT"
fetch_dep() {
  # $1 dir, $2 url, $3 commit, $4 a file that proves the checkout
  if [ ! -f "$1/$4" ] || [ "$(git -C "$1" rev-parse HEAD 2>/dev/null)" != "$3" ]; then
    rm -rf "$1"; git init -q "$1"
    git -C "$1" remote add origin "$2"
    git -C "$1" fetch -q --depth 1 origin "$3"
    git -C "$1" checkout -q FETCH_HEAD
  fi
}
fetch_dep "$WORK/ck" https://github.com/ROCm/composable_kernel.git "$CK_COMMIT" include/ck_tile/core.hpp
fetch_dep "$WORK/json" https://github.com/nlohmann/json.git "$JSON_COMMIT" include/nlohmann/json.hpp

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
# O2 (docs/gpu-experiments/03-acceleration): the fused w2 gang task instantiates CK's small-tile
# GEMM pipeline for gfx942 offline; the day-1 build had done that on the VM only.
compile ckgang -DMK_CK_GANG=1 || ok=1
# O6: our kernels' cache streams with the stock linears' non-temporal policy (raw buffer loads, sc1 nt)
compile ntstreams -DMLA_NT_STREAMS=1 || ok=1
# O3: the per-tile linear with the norm prologue instantiates the CK small-tile linear offline
compile cklinear -DMK_CK_LINEAR=1 || ok=1
# O7: the MFMA attention (v_mfma_f32_16x16x16_bf16) in place of the VALU kernel
compile mfma -DMLA_ATTEND_MFMA=1 || ok=1
# round 4 (docs/gpu-experiments/04-kernels): the GEMV linear's three forms in the worker (L1c),
# and the fused w2's CK path kept under MPK_W2_CK_TILE (L3; ckgang now compiles the GEMV form)
compile gemv -DMK_GEMV=1 || ok=1
compile w2ck -DMK_CK_GANG=1 -DMPK_W2_CK_TILE=1 || ok=1
# I1: the worker timing build (MPK_TIMING=1 of persistent_kernel.py): the per-worker prints of our hunk
compile timing -DMPK_ENABLE_TIMING=1 || ok=1
# I4: the fence knobs (run_fleet.py --runtime-flags), each alone; the two fence knobs are also disassembled below
compile nocfence -DMPK_NO_COMPLETION_FENCE=1 || ok=1
compile noafence -DMPK_NO_ACQUIRE_FENCE=1 || ok=1
compile nobcastcas -DMPK_NO_BCAST_CAS=1 || ok=1
compile nolocalcas -DMPK_NO_LOCAL_CAS=1 || ok=1
compile sleep8 -DMPK_POLL_SLEEP=8 || ok=1

# The patched host sources of the runtime (O8, docs/gpu-experiments/03-acceleration: the
# side-operator branch of register_mugraph; also every registration our patch adds), parsed
# and type-checked with the ROCm clang as the fork's CMake compiles them (host C++, no
# device code), so a patch edit that does not compile is caught before the VM's build. The
# image carries no rocblas-dev or hipblas-dev; stub/ declares what the fork's helper headers
# need from them (rocblas_helper.h parses against the stub, nothing runs).
docker run --rm --platform linux/amd64 -v "$WORK/fleet:/fleet" -v "$WORK/json:/json" -v "$HERE:/here" -v "$WORK/out:/out" "$IMAGE" bash -c "
  cd /fleet && rm -f /out/hostcc.log; rc=0; for f in src/kernel/graph.cc src/kernel/runtime.cc src/kernel/task_register.cc; do
    hipcc -fsyntax-only -x c++ -std=c++17 -D__HIP_PLATFORM_AMD__=1 -DMIRAGE_BACKEND_USE_ROCM -DMIRAGE_FINGERPRINT_USE_ROCM \
      -I/fleet/include -I/json/include -I/opt/rocm/include -I/fleet/deps/rocblas/include -I/here/stub \
      \$f >> /out/hostcc.log 2>&1 || rc=1; done; echo \$rc > /out/hostcc.rc"
echo "host sources (graph.cc, runtime.cc, task_register.cc): syntax check exit $(cat "$WORK/out/hostcc.rc"), errors: $(grep -c 'error:' "$WORK/out/hostcc.log" || true)"
[ "$(cat "$WORK/out/hostcc.rc")" = 0 ] || ok=1

# The standalone kernel-test launcher (fleet/tasks/kernel_tests_mi300.cu),
# with the build line of fleet/tasks/README.md, both variants; it links to an
# executable, so this is the day-2 binary minus the run.
for spec in ":" "_debug:-DMLA_ATTEND_DEBUG_SCORES" "_nt:-DMLA_NT_STREAMS" "_mfma:-DMLA_ATTEND_MFMA"; do
  sfx="${spec%%:*}"; v="${spec#*:}"
  docker run --rm --platform linux/amd64 -v "$ROOT:/w" -v "$WORK/fleet:/fleet" -v "$WORK/out:/out" "$IMAGE" bash -c "
    cd /w && hipcc --offload-arch=gfx942 -O2 -std=c++17 \
      -D__HIP_PLATFORM_AMD__=1 -DMIRAGE_AMD_MI300 -DMIRAGE_BACKEND_USE_ROCM -DMPK_TARGET_CC=94 -DMODE_ONLINE $v \
      -I fleet -I /fleet/include -I /fleet/include/mirage/persistent_kernel \
      -Rpass-analysis=kernel-resource-usage fleet/tasks/kernel_tests_mi300.cu -o /out/kernel_tests$sfx \
      > /out/kernel_tests$sfx.log 2>&1; echo \$? > /out/kernel_tests$sfx.rc"
  rc=$(cat "$WORK/out/kernel_tests$sfx.rc")
  echo "kernel_tests launcher${sfx:+ ($v)}: hipcc exit $rc, errors: $(grep -c 'error:' "$WORK/out/kernel_tests$sfx.log" || true)"
  [ "$rc" = 0 ] || ok=1
done

# The gfx942 assembly of the "ours" variant, for the fence question (MAJ-3):
# does __builtin_amdgcn_fence(..., "agent") lower to buffer_wbl2 sc1 on
# release and buffer_inv sc1 on acquire? Counted per kernel into fences.txt;
# the same table for the two fence knobs of I4 (nocfence, noafence), so the
# sites each knob removes are read as the drop in the sc1 counts.
disasm() {
  local variant="$1"; shift
  docker run --rm --platform linux/amd64 \
    -v "$WORK/fleet:/fleet" -v "$WORK/ck:/ck" -v "$WORK/json:/json" -v "$HERE:/here" -v "$WORK/out:/out" \
    "$IMAGE" bash -c "
      hipcc -S -x hip /here/mk_tu.cu --offload-device-only -o /out/dev_$variant.s \
        -O3 -std=c++17 --offload-arch=gfx942 \
        -I/fleet/include -I/fleet/include/mirage/persistent_kernel \
        -I/fleet/deps/rocblas/include -I/ck/include -I/opt/rocm/include -I/json/include \
        -DCK_TILE_FMHA_FWD_FAST_EXP2=1 -DMAX_WORKER_PER_SCHEDULER=38 -DMIRAGE_USE_CUTLASS_KERNEL=0 \
        -D__HIP_PLATFORM_AMD__=1 -DMIRAGE_AMD_MI300 -DMIRAGE_BACKEND_USE_ROCM -DMPK_TARGET_CC=94 \
        -DMPK_MAX_NUM_BATCHED_REQUESTS=1 -DMPK_MAX_NUM_BATCHED_TOKENS=1 -DMPK_MAX_NUM_PAGES=1 \
        -DMPK_PAGE_SIZE=64 -DMPK_MAX_SEQ_LENGTH=1056 -DMODE_ONLINE -DMPK_PROFILING_NUM_ITERS=0 \
        -DMPK_ENABLE_GANG_TASKS $* > /out/dev_$variant.log 2>&1"
}
disasm ours
disasm gemv -DMK_GEMV=1
disasm ckgang -DMK_CK_GANG=1
# the standalone launcher's device code (round 4): the batch loops of k_linear_gemv, k_moe_router
# and k_mla_merge_uv are named kernels there, so their s_waitcnt vmcnt sequences can be read
docker run --rm --platform linux/amd64 -v "$ROOT:/w" -v "$WORK/fleet:/fleet" -v "$WORK/out:/out" "$IMAGE" bash -c "
  cd /w && hipcc -S -x hip --offload-device-only --offload-arch=gfx942 -O2 -std=c++17 \
    -D__HIP_PLATFORM_AMD__=1 -DMIRAGE_AMD_MI300 -DMIRAGE_BACKEND_USE_ROCM -DMPK_TARGET_CC=94 -DMODE_ONLINE \
    -I fleet -I /fleet/include -I /fleet/include/mirage/persistent_kernel \
    fleet/tasks/kernel_tests_mi300.cu -o /out/dev_kt.s > /out/dev_kt.log 2>&1; echo \$? > /out/dev_kt.rc"
echo "launcher disassembly (dev_kt.s): hipcc exit $(cat "$WORK/out/dev_kt.rc")"
disasm nocfence -DMPK_NO_COMPLETION_FENCE=1
disasm noafence -DMPK_NO_ACQUIRE_FENCE=1
python3 - "$WORK/out" > "$HERE/fences.txt" <<'PYEOF'
import re, sys, collections, os
out = sys.argv[1]
kinds = ("buffer_wbl2 sc0 sc1", "buffer_wbl2 sc1", "buffer_wbl2 sc0", "buffer_inv sc0 sc1",
         "buffer_inv sc1", "buffer_inv sc0", "s_getreg_b32", "s_sleep")
print("# static instruction counts per function in the gfx942 assembly of mk_tu.cu")
print("# sc1 = agent scope (the cross-XCD fence the design relies on); sc0 sc1 = system scope")
print("# (printf/assert hostcall paths; a plain __threadfence() lowers to agent scope on this hipcc, env/hw/probes/fence_probe.cu, 2026-09-15); s_getreg_b32 = the HW_REG_XCC_ID read")
print("# variants: ours (the default build); nocfence and noafence (I4: the completion and the acquire fence knobs)")
for variant in ("ours", "nocfence", "noafence"):
    path = os.path.join(out, f"dev_{variant}.s")
    if not os.path.exists(path):
        print(f"\n## {variant}: no assembly"); continue
    fn, counts = "(none)", collections.defaultdict(collections.Counter)
    for l in open(path).read().splitlines():
        m = re.match(r"^([_A-Za-z][\w.$@]*):", l)
        if m and not m.group(1).startswith(".L"):
            fn = m.group(1)
            continue
        t = l.strip()
        for k in kinds:
            if t.startswith(k) and not (k.endswith("sc0") and t.startswith(k + " sc1")):
                counts[fn][k] += 1
                break
    print(f"\n## {variant}")
    print(f"# {'function':<58} " + " ".join(f"{k:>20}" for k in kinds))
    for f, c in counts.items():
        if "kernel" in f or "printf" in f or "assert" in f:
            print(f"{f[:58]:<58} " + " ".join(f"{c.get(k, 0):>20}" for k in kinds))
PYEOF
cat "$HERE/fences.txt"

# One table per variant: every kernel's registers, spills, LDS, occupancy.
{
  echo "# Offline gfx942 compile, $(date -u +%Y-%m-%dT%H:%M:%SZ), $(cat "$WORK/out/hipcc.txt" | tr '\n' ' ')"
  echo "# fleet 51dce4f + gfx942.patch + new_tasks.patch + sched_xcd.patch; composable_kernel $CK_COMMIT; json $JSON_COMMIT"
  for v in mk_ours mk_ckfmha mk_debugscores mk_ckgang mk_ntstreams mk_cklinear mk_mfma mk_gemv mk_w2ck mk_timing mk_nocfence mk_noafence mk_nobcastcas mk_nolocalcas mk_sleep8 kernel_tests kernel_tests_debug kernel_tests_nt kernel_tests_mfma; do
    echo; echo "## $v (hipcc exit $(cat "$WORK/out/$v.rc"))"
    grep -E "Function Name|    VGPRs:|AGPRs|SGPRs Spill|VGPRs Spill|LDS Size|ScratchSize|Occupancy" "$WORK/out/$v.log" \
      | sed 's/.*remark: *//; s/ \[-Rpass.*//; s/Function Name: //' | paste - - - - - - - - \
      | grep -v flush_cache | awk -F'\t' '{printf "%-58s %-12s %-10s %-27s %-25s %-17s %-17s %s\n",$1,$2,$3,$4,$5,$6,$7,$8}'
    grep -E "error:" "$WORK/out/$v.log" | head -10 || true
  done
} > "$HERE/resources.txt"
cat "$HERE/resources.txt"
exit $ok
