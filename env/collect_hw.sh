#!/usr/bin/env bash
# Hardware collection on the MI300X box, before env/setup.sh runs
# (docs/gpu-bringup/01-plan.md, docs/gpu-bringup/02-checklist.md).
#
#   bash env/collect_hw.sh                       # everything
#   bash env/collect_hw.sh --skip-probes         # no probe compile and no probe run
#   bash env/collect_hw.sh --skip-babelstream    # no BabelStream clone and build
#   bash env/collect_hw.sh --out /tmp/hwdry      # a dry run, outside env/hw
#
# Output, under --out, by default env/hw/<UTC YYYYMMDD>/ :
#   raw/<slug>.txt   one file per command, stdout and stderr together
#   collect.log      this script's whole output
#   summary.md       the filled-in checklist, written by env/hw/summarize.py
#
# No command aborts the run. `set -e` is deliberately not set: a command that
# fails appends an "UNAVAILABLE: exit <n>" line to its own raw file, the row it
# feeds becomes UNAVAILABLE in the summary, and the collection continues.
# System python3 only; summarize.py is stdlib only; the venvs are not needed.
#
# Raw file slugs. env/hw/summarize.py reads exactly these names.
#
#   group A   rocminfo                  rocminfo
#             amd-smi-version           amd-smi version
#             amd-smi-list              amd-smi list
#             amd-smi-static            amd-smi static
#             amd-smi-static-bus        amd-smi static --bus
#             amd-smi-static-driver     amd-smi static --driver
#             amd-smi-static-asic       amd-smi static --asic
#             amd-smi-static-vbios      amd-smi static --vbios
#             hipcc-version             hipcc --version
#             rocm-version              /opt/rocm/.info/version
#             dpkg-rocm                 dpkg -l | grep -i rocm
#             rocm-include              ls /opt/rocm/include, ck_tile and ck presence
#             uname                     uname -a
#             os-release                /etc/os-release
#             amdgpu-version            /sys/module/amdgpu/version
#             docker-version            docker --version
#             python-version            python3 --version
#             python-torch              python3 -c "import torch; print(torch.__version__)"
#             hip-visible-devices       HIP_VISIBLE_DEVICES before and after this script set it
#   group B   kfd-topology              every /sys/class/kfd/kfd/topology/nodes/*/properties
#             occupancy                 occupancy --vgprs 182, then the same with
#                                       --dynamic-lds 58368; both lines, in that order
#             wallclock                 occupancy --clock
#   group C   partition                 first partition query that worked
#             partition-cmd             which query that was
#             amd-smi-set-help          amd-smi set --help
#   group D   amd-smi-metric-clock      amd-smi metric --clock
#             amd-smi-metric-power      amd-smi metric --power
#             amd-smi-metric-mem-usage  amd-smi metric --mem-usage
#             amd-smi-metric-temp       amd-smi metric --temperature
#             amd-smi-metric-throttle   amd-smi metric --throttle
#             amd-smi-static-limit      amd-smi static --limit
#             amd-smi-static-vram       amd-smi static --vram
#             amd-smi-process           amd-smi process
#             rocm-smi-a                rocm-smi -a
#             rocminfo-max-clock        the "Max Clock Freq. (MHz)" lines of rocminfo
#   group E   e1                        stream_read, full occupancy, 1 GiB, unroll 8, 3 runs
#             e2                        stream_read, one wave/SIMD, unroll 1..32
#             e3                        stream_read, grid 608, unroll 1..32
#             e4                        stream_read, full occupancy, 4 MiB .. 1024 MiB
#             babelstream               HIP BabelStream clone, build and run
#                                       (cloned into env/hw/build/, not into the record)
#             rocm-bandwidth-test       rocm-bandwidth-test -a
#   group F   xcc-map                   every xcc_map run
#             xcc-map-296.csv           the per-block dump of the 296-block grid
#             xcc-map-8.csv             the same for the 8-block scheduler grid
#   group G   fence-asm                 the hipcc -S compile of fence_probe.cu
#             fence_probe.s             the generated gfx942 assembly
#             fence                     fence_grep.py on that assembly
#             fence-run                 the fence_probe binary, showing the kernels launch
#   group H   chase                     every chase run
#   group I   rocprof-list              rocprofv3 counter listing
#             pmc1-run .. pmc4-run      the four rocprofv3 --pmc runs
#             pmc1/ .. pmc4/            their CSV output directories
#             ktrace-run, ktrace/       the rocprofv3 --kernel-trace run
#             copy-bytes                copy_bytes without the profiler
#   group J   lscpu, nproc, free, df-root, df-home, ulimit
#             download-log              the last 30 lines of env/logs/download.log
#             hf-cache-size             du -sh ~/.cache/huggingface
#             docker-pull               docker pull of the ROCm image
#             docker-gpu                rocminfo inside that image with the GPU passed through
#   probes    build-<name>              the hipcc compile of each probe
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export ROCM_PATH
export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH"

SKIP_PROBES=0
SKIP_BABELSTREAM=0
OUT_DIR=""
usage() {
  echo "usage: bash env/collect_hw.sh [--out DIR] [--skip-probes] [--skip-babelstream]"
  echo "  --out DIR   where the collection goes; default env/hw/<UTC date>."
  echo "              A dry run must pass a scratch directory outside env/hw:"
  echo "              everything under env/hw/2*/ is a committed record of a"
  echo "              rented machine that cannot be measured again."
}
while [ "$#" -gt 0 ]; do
  case "$1" in
    --skip-probes) SKIP_PROBES=1 ;;
    --skip-babelstream) SKIP_BABELSTREAM=1 ;;
    --out)
      shift
      [ "$#" -gt 0 ] || { echo "--out needs a directory"; usage; exit 2; }
      OUT_DIR="$1"
      ;;
    --out=*) OUT_DIR="${1#--out=}" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1"; usage; exit 2 ;;
  esac
  shift
done

OUT="${OUT_DIR:-$ROOT/env/hw/$(date -u +%Y%m%d)}"
RAW="$OUT/raw"
BUILD="$ROOT/env/hw/build"
PROBE_SRC="$ROOT/env/hw/probes"
mkdir -p "$RAW" "$BUILD"
LOG="$OUT/collect.log"
exec > >(tee -a "$LOG") 2>&1

echo "== collect_hw.sh $(date -u +%Y-%m-%dT%H:%M:%SZ) on $(hostname 2>/dev/null || echo unknown)"
echo "out:  $OUT"
echo "log:  $LOG"
echo "rocm: $ROCM_PATH"
echo "skip-probes=$SKIP_PROBES skip-babelstream=$SKIP_BABELSTREAM"

HIP_VISIBLE_DEVICES_BEFORE="${HIP_VISIBLE_DEVICES:-<unset>}"
export HIP_VISIBLE_DEVICES=0
{
  echo "before: $HIP_VISIBLE_DEVICES_BEFORE"
  echo "after: $HIP_VISIBLE_DEVICES"
} > "$RAW/hip-visible-devices.txt"
echo "HIP_VISIBLE_DEVICES: was $HIP_VISIBLE_DEVICES_BEFORE, now $HIP_VISIBLE_DEVICES"

# ---------------------------------------------------------------------------
# helpers

step() { echo; echo "---- $*"; }

_run_to() {
  # $1 raw file, $2 w|a, rest: the command (run as a command, not a shell string)
  local f="$1" mode="$2"; shift 2
  if [ "$mode" = "w" ]; then
    : > "$f"
  fi
  echo "[run] $(basename "$f") <- $*"
  "$@" >> "$f" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    printf 'UNAVAILABLE: exit %s: %s\n' "$rc" "$*" >> "$f"
    echo "[run] $(basename "$f"): UNAVAILABLE (exit $rc)"
  fi
  return 0
}

cap()    { local s="$1"; shift; _run_to "$RAW/$s.txt" w "$@"; }
capa()   { local s="$1"; shift; _run_to "$RAW/$s.txt" a "$@"; }
capsh()  { local s="$1"; shift; _run_to "$RAW/$s.txt" w bash -c "$*"; }
capsha() { local s="$1"; shift; _run_to "$RAW/$s.txt" a bash -c "$*"; }

unavailable() {
  # $1 slug, rest: the reason
  local s="$1"; shift
  printf 'UNAVAILABLE: %s\n' "$*" >> "$RAW/$s.txt"
  echo "[skip] $s: $*"
}

PROBES_BUILT=""
probe_built() {
  case " $PROBES_BUILT " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

probe() {
  # probe <slug> <name> [args...]   appends to raw/<slug>.txt
  local slug="$1" name="$2"; shift 2
  local f="$RAW/$slug.txt"
  touch "$f"
  if [ "$SKIP_PROBES" = "1" ]; then
    printf 'UNAVAILABLE: %s %s not run (--skip-probes)\n' "$name" "$*" >> "$f"
    echo "[skip] $name $*: --skip-probes"
    return 0
  fi
  if ! probe_built "$name"; then
    printf 'UNAVAILABLE: %s %s not run (%s did not build)\n' "$name" "$*" "$name" >> "$f"
    echo "[skip] $name $*: not built"
    return 0
  fi
  _run_to "$f" a "$BUILD/$name" "$@"
}

# ---------------------------------------------------------------------------
step "0. probe compile (env/hw/build/)"
PROBE_NAMES="occupancy stream_read xcc_map fence_probe chase copy_bytes"
if [ "$SKIP_PROBES" = "1" ]; then
  echo "skipped (--skip-probes); every probe row will be UNAVAILABLE"
else
  for p in $PROBE_NAMES; do
    if [ ! -f "$PROBE_SRC/$p.cu" ]; then
      printf 'UNAVAILABLE: %s is missing\n' "$PROBE_SRC/$p.cu" > "$RAW/build-$p.txt"
      echo "[build] $p: source missing at $PROBE_SRC/$p.cu"
      continue
    fi
    # Remove first: a binary left by an earlier run would otherwise pass the
    # -x test below and be measured as though this compile had produced it.
    rm -f "$BUILD/$p"
    cap "build-$p" hipcc --offload-arch=gfx942 -O2 -std=c++17 "$PROBE_SRC/$p.cu" -o "$BUILD/$p"
    if [ -x "$BUILD/$p" ] && ! grep -q '^UNAVAILABLE:' "$RAW/build-$p.txt"; then
      PROBES_BUILT="$PROBES_BUILT $p"
      echo "[build] $p: ok"
    else
      echo "[build] $p: FAILED, its rows will be UNAVAILABLE (see raw/build-$p.txt)"
    fi
  done
fi
echo "probes built:${PROBES_BUILT:- none}"

# ---------------------------------------------------------------------------
step "A. identity and software"
cap rocminfo rocminfo
cap amd-smi-version amd-smi version
cap amd-smi-list amd-smi list
cap amd-smi-static amd-smi static
cap amd-smi-static-bus amd-smi static --bus
cap amd-smi-static-driver amd-smi static --driver
cap amd-smi-static-asic amd-smi static --asic
cap amd-smi-static-vbios amd-smi static --vbios
cap hipcc-version hipcc --version
cap rocm-version cat "$ROCM_PATH/.info/version"
capsh dpkg-rocm "dpkg -l | grep -i rocm"

rocm_include_report() {
  local d rc=0
  ls "$ROCM_PATH/include" || rc=1
  for d in ck_tile ck; do
    if [ -d "$ROCM_PATH/include/$d" ]; then
      echo "$d: present"
    else
      echo "$d: absent"
    fi
  done
  return "$rc"
}
cap rocm-include rocm_include_report

cap uname uname -a
cap os-release cat /etc/os-release
cap amdgpu-version cat /sys/module/amdgpu/version
cap docker-version docker --version
cap python-version python3 --version
cap python-torch python3 -c "import torch; print(torch.__version__)"

# ---------------------------------------------------------------------------
step "B. topology and residency"
kfd_topology_report() {
  local d found=0
  for d in /sys/class/kfd/kfd/topology/nodes/*/; do
    [ -r "$d/properties" ] || continue
    found=1
    echo "== $d"
    cat "$d/properties"
  done
  [ "$found" = "1" ] || return 1
  return 0
}
cap kfd-topology kfd_topology_report
# amd-smi static --asic is captured in group A and read by both A and B.
# B10 wants both residencies in one file: register-only first, then with the
# 58,368 B dynamic LDS request the runtime launches with (runtime_header.h),
# which is what limits the worker kernel to one block per CU. The two lines are
# told apart by their dynamic_lds= key.
: > "$RAW/occupancy.txt"
probe occupancy occupancy --vgprs 182
probe occupancy occupancy --vgprs 182 --dynamic-lds 58368
: > "$RAW/wallclock.txt"
probe wallclock occupancy --clock

# ---------------------------------------------------------------------------
step "C. partition mode (read only; no set is executed)"
PART_CANDIDATES=(
  "amd-smi static --partition"
  "amd-smi partition"
  "amd-smi static -g 0 --partition"
  "rocm-smi --showcomputepartition --showmemorypartition"
)
: > "$RAW/partition.txt"
: > "$RAW/partition-cmd.txt"
PART_DONE=0
for c in "${PART_CANDIDATES[@]}"; do
  if part_out=$(bash -c "$c" 2>&1); then
    echo "$part_out" > "$RAW/partition.txt"
    echo "$c" > "$RAW/partition-cmd.txt"
    echo "[partition] worked: $c"
    PART_DONE=1
    break
  fi
done
if [ "$PART_DONE" = "0" ]; then
  unavailable partition "no partition query worked; candidates: ${PART_CANDIDATES[*]}"
  unavailable partition-cmd "no partition query worked"
fi
cap amd-smi-set-help amd-smi set --help

# ---------------------------------------------------------------------------
step "D. clocks, power and memory"
cap amd-smi-metric-clock amd-smi metric --clock
cap amd-smi-metric-power amd-smi metric --power
cap amd-smi-metric-mem-usage amd-smi metric --mem-usage
cap amd-smi-metric-temp amd-smi metric --temperature
cap amd-smi-metric-throttle amd-smi metric --throttle
cap amd-smi-static-limit amd-smi static --limit
cap amd-smi-static-vram amd-smi static --vram
cap amd-smi-process amd-smi process
cap rocm-smi-a rocm-smi -a
cap rocminfo-max-clock grep "Max Clock Freq" "$RAW/rocminfo.txt"

# ---------------------------------------------------------------------------
step "E. achievable bandwidth"
: > "$RAW/e1.txt"
probe e1 stream_read --occupancy full --size 1G --unroll 8 --repeat 3
: > "$RAW/e2.txt"
for n in 1 2 4 8 16 32; do
  probe e2 stream_read --occupancy one --size 1G --unroll "$n"
done
: > "$RAW/e3.txt"
for n in 1 2 4 8 16 32; do
  probe e3 stream_read --grid 608 --size 1G --unroll "$n"
done
: > "$RAW/e4.txt"
# --passes keeps the loads per thread constant across the sweep: at 4 MiB the
# buffer is read 256 times, at 1024 MiB once, so every point does the same work
# and the small sizes are not dominated by the launch.
for w in 4 16 32 64 128 256 512 1024; do
  probe e4 stream_read --occupancy full --unroll 8 --size "${w}M" --passes "$((1024 / w))"
done

: > "$RAW/babelstream.txt"
# The VM image ships no cmake. Rather than skip the only vendor cross-check of
# group E, fetch one into a throwaway venv: the cmake wheel carries its own
# binary and nothing outside /tmp is touched. --without-pip plus get-pip.py
# because the image's python3-venv has no bundled pip either.
ensure_cmake() {
  if command -v cmake >/dev/null 2>&1; then
    echo "cmake: $(command -v cmake)"
    return 0
  fi
  local venv="/tmp/hwcmake"
  echo "cmake is not on PATH; installing one into $venv"
  if [ ! -x "$venv/bin/cmake" ]; then
    rm -rf "$venv"
    python3 -m venv --without-pip "$venv" || return 1
    curl -sSL -o "$venv/get-pip.py" https://bootstrap.pypa.io/get-pip.py || return 1
    "$venv/bin/python" "$venv/get-pip.py" || return 1
    "$venv/bin/python" -m pip install -q cmake || return 1
  fi
  [ -x "$venv/bin/cmake" ] || return 1
  PATH="$venv/bin:$PATH"
  export PATH
  echo "cmake: $(command -v cmake)"
  return 0
}

babelstream_run() {
  # Into the gitignored build tree, never under $OUT: $OUT is the committed
  # record and a vendor benchmark's source tree does not belong in it.
  local src="$BUILD/babelstream-src"
  rm -rf "$src"
  ensure_cmake || return 1
  # GIT_TERMINAL_PROMPT=0 so a credential prompt fails instead of hanging.
  GIT_TERMINAL_PROMPT=0 timeout 300 git clone --depth 1 \
    https://github.com/UoB-HPC/BabelStream "$src" || return 1
  timeout 300 cmake -S "$src" -B "$src/build" -DMODEL=hip -DCMAKE_CXX_COMPILER=hipcc \
    -DCXX_EXTRA_FLAGS="--offload-arch=gfx942" || return 1
  timeout 600 cmake --build "$src/build" -j "$(nproc 2>/dev/null || echo 4)" || return 1
  local bin
  bin="$src/build/hip-stream"
  [ -x "$bin" ] || bin="$(find "$src/build" -maxdepth 2 -name 'hip-stream' -type f 2>/dev/null | head -1)"
  [ -n "$bin" ] && [ -x "$bin" ] || return 1
  timeout 300 "$bin" -n 50 -s 268435456 || return 1
  return 0
}
if [ "$SKIP_BABELSTREAM" = "1" ]; then
  unavailable babelstream "not run (--skip-babelstream)"
elif [ "$SKIP_PROBES" = "1" ]; then
  unavailable babelstream "not run (--skip-probes: no GPU work)"
else
  capa babelstream babelstream_run
fi

if command -v rocm-bandwidth-test >/dev/null 2>&1; then
  cap rocm-bandwidth-test timeout 300 rocm-bandwidth-test -a
else
  : > "$RAW/rocm-bandwidth-test.txt"
  unavailable rocm-bandwidth-test "rocm-bandwidth-test is not on PATH"
fi

# ---------------------------------------------------------------------------
step "F. workgroup-to-XCD placement"
: > "$RAW/xcc-map.txt"
# The dumps carry the placement itself; the summary lines carry only counts.
# 296 gives the round-robin offset, 8 gives the scheduler blocks F3 asks about.
probe xcc-map xcc_map --grid 296 --runs 3 --dump "$RAW/xcc-map-296.csv"
probe xcc-map xcc_map --grid 8 --runs 3 --dump "$RAW/xcc-map-8.csv"
probe xcc-map xcc_map --grid 8 --runs 3 --concurrent
for g in 304 608 1000 37; do
  probe xcc-map xcc_map --grid "$g" --runs 3
done

# ---------------------------------------------------------------------------
step "G. fence lowering on this machine's hipcc"
: > "$RAW/fence.txt"
if [ "$SKIP_PROBES" = "1" ]; then
  : > "$RAW/fence-asm.txt"
  unavailable fence-asm "not compiled (--skip-probes)"
  unavailable fence "fence_probe.s was not produced"
elif [ ! -f "$PROBE_SRC/fence_probe.cu" ]; then
  : > "$RAW/fence-asm.txt"
  unavailable fence-asm "$PROBE_SRC/fence_probe.cu is missing"
  unavailable fence "fence_probe.s was not produced"
else
  cap fence-asm hipcc --offload-arch=gfx942 -S --offload-device-only -O2 -std=c++17 \
    "$PROBE_SRC/fence_probe.cu" -o "$RAW/fence_probe.s"
  if [ -s "$RAW/fence_probe.s" ]; then
    _run_to "$RAW/fence.txt" w python3 "$PROBE_SRC/fence_grep.py" "$RAW/fence_probe.s"
  else
    unavailable fence "fence_probe.s is empty or was not produced"
  fi
fi
# The same source also links, so running it shows the four kernels launch and
# the assembly counted above belongs to code that executes.
: > "$RAW/fence-run.txt"
probe fence-run fence_probe

# ---------------------------------------------------------------------------
step "H. latency, fence cost and the cross-XCD round trip"
: > "$RAW/chase.txt"
for s in 1M 64M 1G; do
  probe chase chase --size "$s"
done
probe chase chase --fence release
probe chase chase --fence acquire
probe chase chase --fence release --xcds 8
probe chase chase --fence acquire --xcds 8
probe chase chase --pingpong

# ---------------------------------------------------------------------------
step "I. profiler"
ROCPROF_CANDIDATES=(
  "timeout 300 rocprofv3 --list-avail"
  "timeout 300 rocprofv3 -L"
  "timeout 300 rocprofv3 --list-metrics"
)
: > "$RAW/rocprof-list.txt"
ROCPROF_DONE=0
for c in "${ROCPROF_CANDIDATES[@]}"; do
  if avail_out=$(bash -c "$c" 2>&1); then
    {
      echo "# listing command: $c"
      echo "$avail_out"
    } > "$RAW/rocprof-list.txt"
    echo "[rocprofv3] worked: $c"
    ROCPROF_DONE=1
    break
  fi
done
if [ "$ROCPROF_DONE" = "0" ]; then
  unavailable rocprof-list "no rocprofv3 listing command worked; candidates: ${ROCPROF_CANDIDATES[*]}"
fi

# Two counters per run: more than two on one run multiplexes on this hardware.
# The fourth pair is the read side of the byte arithmetic: a read on MI300 is a
# 128-byte request counted by TCC_BUBBLE, which is what rocprofv3's own
# FETCH_SIZE expression uses.
PMC_SETS=(
  "TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_32B_sum"
  "TCC_EA0_WRREQ_sum TCC_EA0_WRREQ_64B_sum"
  "TCC_HIT_sum TCC_MISS_sum"
  "TCC_BUBBLE_sum TCC_EA0_RDREQ_sum"
)
k=0
for set_ in "${PMC_SETS[@]}"; do
  k=$((k + 1))
  : > "$RAW/pmc$k-run.txt"
  if [ "$SKIP_PROBES" = "1" ] || ! probe_built copy_bytes; then
    unavailable "pmc$k-run" "copy_bytes is not available"
    continue
  fi
  # shellcheck disable=SC2086
  # $set_ is two counter names and must split into two arguments.
  capa "pmc$k-run" timeout 300 rocprofv3 --pmc $set_ -d "$RAW/pmc$k" --output-format csv -- "$BUILD/copy_bytes"
done

: > "$RAW/ktrace-run.txt"
if [ "$SKIP_PROBES" = "1" ] || ! probe_built copy_bytes; then
  unavailable ktrace-run "copy_bytes is not available"
else
  capa ktrace-run timeout 300 rocprofv3 --kernel-trace -d "$RAW/ktrace" --output-format csv -- "$BUILD/copy_bytes"
fi

: > "$RAW/copy-bytes.txt"
probe copy-bytes copy_bytes

# ---------------------------------------------------------------------------
step "J. host, disk and network"
cap lscpu lscpu
cap nproc nproc
cap free free -g
cap df-root df -h /
cap df-home df -h "$HOME"
capsh ulimit "ulimit -a"

# The download of 01-plan.md step 2 writes here and may not have started yet.
mkdir -p "$ROOT/env/logs"
if [ -f "$ROOT/env/logs/download.log" ]; then
  cap download-log tail -n 30 "$ROOT/env/logs/download.log"
else
  : > "$RAW/download-log.txt"
  unavailable download-log "env/logs/download.log does not exist"
fi

if [ -d "$HOME/.cache/huggingface" ]; then
  cap hf-cache-size du -sh "$HOME/.cache/huggingface"
else
  : > "$RAW/hf-cache-size.txt"
  unavailable hf-cache-size "$HOME/.cache/huggingface does not exist"
fi

# J5: the image tag follows A3 (ROCm major.minor) and A7 (Ubuntu version),
# e.g. rocm/dev-ubuntu-22.04:7.0.
ROCM_MM="$(sed -n '1p' "$RAW/rocm-version.txt" 2>/dev/null | cut -d- -f1 | cut -d. -f1,2)"
case "${ROCM_MM:-}" in
  [0-9]*.[0-9]*) : ;;
  *) ROCM_MM="7.0" ;;
esac
UBUNTU_VER="$(sed -n 's/^VERSION_ID="\{0,1\}\([0-9.]*\)"\{0,1\}$/\1/p' "$RAW/os-release.txt" 2>/dev/null | head -1)"
case "${UBUNTU_VER:-}" in
  [0-9]*.[0-9]*) : ;;
  *) UBUNTU_VER="22.04" ;;
esac
DOCKER_IMAGE="rocm/dev-ubuntu-${UBUNTU_VER}:${ROCM_MM}"
echo "docker image for J5: $DOCKER_IMAGE"
cap docker-pull timeout 600 docker pull "$DOCKER_IMAGE"
cap docker-gpu timeout 600 docker run --rm --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render --security-opt seccomp=unconfined \
  "$DOCKER_IMAGE" rocminfo

# ---------------------------------------------------------------------------
step "summary"
if command -v python3 >/dev/null 2>&1; then
  python3 "$ROOT/env/hw/summarize.py" "$OUT" || echo "summarize.py failed; the raw files are still in $RAW"
else
  echo "no python3 on PATH; run env/hw/summarize.py $OUT by hand"
fi
echo
echo "raw files:  $RAW"
echo "summary:    $OUT/summary.md"
echo "log:        $LOG"
echo "next: read the summary against docs/gpu-bringup/02-checklist.md, file every"
echo "      MISMATCH in OPEN-PROBLEMS.md, then commit env/hw/$(basename "$OUT")/"
