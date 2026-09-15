#!/usr/bin/env bash
# Day-1 checks after env/setup.sh (docs/design-doc/08-milestones.md, Day 1, items 1, 3, 4, 5, 7).
# Each check prints PASS / FAIL / UNKNOWN with its evidence. Results go to
# env/check_day1.log (the file the design references) and env/logs/.
#
#   bash env/check_day1.sh                 # all checks
#   SKIP_GRAPH=1 bash env/check_day1.sh    # skip the graph run (no model yet)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLEET="$ROOT/repos/fleet-chiplet-megakernel"
LOGDIR="$ROOT/env/logs"
mkdir -p "$LOGDIR"
LOG="$LOGDIR/check_day1.$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG" | tee "$ROOT/env/check_day1.log") 2>&1
echo "== check_day1.sh $(date -Is) on $(hostname); log $LOG"

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export ROCM_PATH
export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH"
export MIRAGE_HOME="$FLEET"
export AMDGPU_TARGETS=gfx942   # persistent_kernel.py compiles for gfx950 by default
cd "$ROOT"
# shellcheck disable=SC1091
# The Fleet venv: mirage and the Qwen3 smoke graph live there (env/setup.sh step 2).
[ -x .venv-fleet/bin/python ] && source .venv-fleet/bin/activate

RESULTS=()
result() { RESULTS+=("$1  $2"); echo ">>> $1  $2"; }
step() { echo; echo "---- $*"; }

# ---------------------------------------------------------------------------
step "1. partition mode: expect SPX + NPS1 (docs/mi300x/02-chiplet-dispatch.md)"
PART_OUT=""
for c in "amd-smi static --partition" "amd-smi partition" "amd-smi static -g 0 --partition" \
         "rocm-smi --showcomputepartition --showmemorypartition"; do
  if PART_OUT=$(bash -c "$c" 2>&1); then echo "[partition] worked: $c"; echo "$PART_OUT"; break; fi
  PART_OUT=""
done
if [ -z "$PART_OUT" ]; then
  result UNKNOWN "1 partition: no amd-smi/rocm-smi partition query worked; find the flag by hand (amd-smi --help)"
elif echo "$PART_OUT" | grep -qi "SPX" && echo "$PART_OUT" | grep -qi "NPS1"; then
  result PASS "1 partition: SPX + NPS1"
else
  result FAIL "1 partition: output does not show SPX and NPS1 (see above); set with amd-smi set --compute-partition SPX --memory-partition NPS1"
fi

# ---------------------------------------------------------------------------
step "2. import mirage"
if python -c "import mirage, os; print('mirage from', os.path.dirname(mirage.__file__))"; then
  result PASS "2 import mirage"
else
  result FAIL "2 import mirage (gate 1 not passed; run env/setup.sh)"
fi

# ---------------------------------------------------------------------------
step "3. smallest graph through the persistent kernel: [WORKER_XCD] / [SCHED_XCD] (03-synchronization.md, MIN-25)"
# Why this command: demo/qwen3/demo.py --use-mirage is the only shipped entry
# point that builds a task graph and launches worker/scheduler kernels on
# MI300 (tests/runtime_python builds a CUDA torch extension instead). The
# XCD lines are printed by the first 8 workers and all 8 schedulers at launch
# (persistent_kernel.cuh:754 and :1436) regardless of the model, so the
# smallest Qwen3 checkpoint with --num-layers 1 and 4 new tokens is enough.
# Qwen3-0.6B (1.5 GB) is tried first; Qwen3-8B (16 GB, the README's model) if
# the small one fails on shape constraints. permanent_output_dir/ caches the
# compiled kernel and must be removed when flags change.
GRAPH_LOG="$LOGDIR/graph_run.$(date +%Y%m%d-%H%M%S).log"
if [ "${SKIP_GRAPH:-0}" = "1" ]; then
  result UNKNOWN "3 graph run skipped (SKIP_GRAPH=1)"
else
  GRAPH_OK=0
  for m in Qwen/Qwen3-0.6B Qwen/Qwen3-8B; do
    echo "trying $m"
    rm -rf "$FLEET/permanent_output_dir"
    if (cd "$FLEET" && HIP_VISIBLE_DEVICES=0 USE_GANG=1 USE_CK_FMHA=1 USE_FUSED_SILU=1 GANG_M_TILES_GATEUP=1 \
        timeout 1800 python3 demo/qwen3/demo.py --use-mirage --model "$m" --num-layers 1 \
        --max-num-batched-tokens 1 --max-num-batched-requests 1 --max-seq-length 128 \
        --max-new-tokens 4 --ignore-eos --prompt "def add(a, b):" 2>&1 | tee "$GRAPH_LOG"); then
      GRAPH_OK=1; echo "graph run succeeded with $m"; break
    fi
    echo "run with $m failed (see $GRAPH_LOG)"
  done
  if [ "$GRAPH_OK" = "1" ] && grep -q "\[SCHED_XCD\]" "$GRAPH_LOG"; then
    grep "\[WORKER_XCD\]\|\[SCHED_XCD\]" "$GRAPH_LOG" | sort -u
    NSCHED=$(grep -c "\[SCHED_XCD\]" "$GRAPH_LOG" || true)
    # [SCHED_XCD] sched_id=k xcd=k workers_on_xcd=n block=b
    SCHED_MISMATCH=$(grep "\[SCHED_XCD\]" "$GRAPH_LOG" | awk '{split($2,a,"="); split($3,b,"="); if (a[2]!=b[2]) c++} END{print c+0}')
    WORKERS_TOTAL=$(grep "\[SCHED_XCD\]" "$GRAPH_LOG" | awk '{split($4,a,"="); s+=a[2]} END{print s+0}')
    # [WORKER_XCD] worker_id=w block=b xcd=x : expect x == (w + c) mod 8 for one
    # constant c over the 8 printed workers (c = 4 on the first VM, env/hw/20260915
    # F2; the design needs consecutive workers on distinct XCDs, not c = 0)
    WORKER_OFFSET=$(grep "\[WORKER_XCD\]" "$GRAPH_LOG" | sort -u | awk '{split($2,a,"="); split($4,b,"="); print ((b[2]-a[2])%8+8)%8; exit}')
    WORKER_MISMATCH=$(grep "\[WORKER_XCD\]" "$GRAPH_LOG" | sort -u | awk -v c="${WORKER_OFFSET:-0}" '{split($2,a,"="); split($4,b,"="); if ((a[2]+c)%8!=b[2]) k++} END{print k+0}')
    echo "schedulers printed: $NSCHED, sched_id!=xcd: $SCHED_MISMATCH, workers summed over schedulers: $WORKERS_TOTAL, worker offset c=$WORKER_OFFSET, worker (w+c) mod 8 != xcd: $WORKER_MISMATCH"
    if [ "$NSCHED" = "8" ] && [ "$SCHED_MISMATCH" = "0" ] && [ "$WORKERS_TOTAL" = "296" ] && [ "$WORKER_MISMATCH" = "0" ]; then
      result PASS "3 XCD placement: 8 schedulers on their XCDs, 296 workers, xcd == (worker + $WORKER_OFFSET) mod 8"
    else
      result FAIL "3 XCD placement: schedulers=$NSCHED mismatches=$SCHED_MISMATCH workers=$WORKERS_TOTAL worker-mismatches=$WORKER_MISMATCH (MIN-25)"
    fi
    grep "\[FWD_PASS\]" "$GRAPH_LOG" | head -8 || true
  elif [ "$GRAPH_OK" = "1" ]; then
    result FAIL "3 graph ran but printed no [SCHED_XCD] line (is worker_xcd_map allocated? persistent_kernel.cuh:749)"
  else
    result FAIL "3 graph run failed for both models; see $GRAPH_LOG"
  fi
fi

# ---------------------------------------------------------------------------
step "4. disassembly: buffer_wbl2 sc1 / buffer_inv sc1 / sc1 on the counter poll (MAJ-3, docs/mi300x/03-memory-model.md)"
# The generated megakernel is compiled by hipcc into
# <cwd>/permanent_output_dir/test.cpython-38-x86_64-linux-gnu.so (persistent_kernel.py:2424),
# a host .so with the gfx942 code object bundled inside.
SO=$(ls -t "$FLEET"/permanent_output_dir/*.so 2>/dev/null | head -1 || true)
if [ -z "$SO" ]; then
  result UNKNOWN "4 disassembly: no permanent_output_dir/*.so yet (run a graph first)"
else
  DIS="$LOGDIR/megakernel.gfx942.s"
  rm -f "$DIS"
  # candidates, first that produces output wins
  if roc-obj-ls "$SO" >/dev/null 2>&1; then
    echo "[disasm] roc-obj-ls/roc-obj-extract + llvm-objdump"
    (cd "$LOGDIR" && roc-obj-extract "$SO" >/dev/null 2>&1 || true)
    CO=$(ls -t "$LOGDIR"/*gfx942* 2>/dev/null | head -1 || true)
    [ -n "$CO" ] && llvm-objdump -d --arch-name=amdgcn --mcpu=gfx942 "$CO" > "$DIS" 2>/dev/null || true
  fi
  if [ ! -s "$DIS" ] && command -v roc-obj >/dev/null 2>&1; then
    echo "[disasm] roc-obj -d"
    (cd "$LOGDIR" && roc-obj -d -t gfx942 "$SO" >/dev/null 2>&1 || true)
    CO=$(ls -t "$LOGDIR"/*.s 2>/dev/null | grep -v megakernel | head -1 || true)
    [ -n "$CO" ] && cp "$CO" "$DIS"
  fi
  if [ ! -s "$DIS" ]; then
    echo "[disasm] clang-offload-bundler + llvm-objdump"
    clang-offload-bundler --type=o --targets=hip-amdgcn-amd-amdhsa--gfx942 \
      --input="$SO" --output="$LOGDIR/megakernel.gfx942.co" --unbundle 2>/dev/null || true
    [ -s "$LOGDIR/megakernel.gfx942.co" ] && llvm-objdump -d --arch-name=amdgcn --mcpu=gfx942 "$LOGDIR/megakernel.gfx942.co" > "$DIS" 2>/dev/null || true
  fi
  if [ -s "$DIS" ]; then
    WBL2=$(grep -c "buffer_wbl2 sc1" "$DIS" || true)
    INV=$(grep -c "buffer_inv sc1" "$DIS" || true)
    LOADSC1=$(grep -c "load.*sc1" "$DIS" || true)
    ATOMSC1=$(grep -c "atomic.*sc1" "$DIS" || true)
    echo "buffer_wbl2 sc1: $WBL2   buffer_inv sc1: $INV   loads with sc1: $LOADSC1   atomics with sc1: $ATOMSC1   ($DIS)"
    if [ "$WBL2" -gt 0 ] && [ "$INV" -gt 0 ]; then
      result PASS "4 fences: buffer_wbl2 sc1 x$WBL2, buffer_inv sc1 x$INV present (loads sc1 x$LOADSC1: the counter poll must be one of them)"
    else
      result FAIL "4 fences: buffer_wbl2 sc1 x$WBL2, buffer_inv sc1 x$INV; agent-scope fence does not lower as expected (MAJ-3)"
    fi
  else
    result UNKNOWN "4 disassembly: could not extract the gfx942 code object from $SO; try: roc-obj-ls $SO"
  fi
fi

# ---------------------------------------------------------------------------
step "5. CK split-KV FMHA at (kM0=16, kQKHeaddim=576, kN1=512) (DQ3, docs/fleet Q11)"
# Fleet's own compile line puts deps/composable_kernel/include before /opt/rocm/include
# (persistent_kernel.py:293); the probe tries both.
for CKINC in "$FLEET/deps/composable_kernel/include" "$ROCM_PATH/include"; do
  echo "[ck] $CKINC:"
  ls "$CKINC/ck_tile/ops/fmha" >/dev/null 2>&1 || { echo "   no ck_tile/ops/fmha here"; continue; }
  grep -rln "576" "$CKINC/ck_tile/ops/fmha" 2>/dev/null | head -5 || true
  grep -rn "kQKHeaddim *= *576\|HDim *= *576\|hdim_q *= *576\|512.*576\|576.*512" "$CKINC/ck_tile/ops/fmha" 2>/dev/null | head -5 || echo "   no 576/512 mention in ck_tile/ops/fmha"
done
PROBE_LOG="$LOGDIR/probe_ck.$(date +%Y%m%d-%H%M%S).log"
CK_OK=0
for CKINC in "$FLEET/deps/composable_kernel/include" "$ROCM_PATH/include"; do
  [ -d "$CKINC/ck_tile" ] || continue
  echo "[ck probe] hipcc --offload-arch=gfx942 -fsyntax-only with -I$CKINC"
  if hipcc --offload-arch=gfx942 -fsyntax-only -std=c++17 -DCK_TILE_FMHA_FWD_FAST_EXP2=1 \
       -I"$CKINC" -I"$FLEET/include" -I"$FLEET/include/mirage/persistent_kernel" \
       "$ROOT/env/probe_ck_fmha_576_512.cpp" > "$PROBE_LOG" 2>&1; then
    CK_OK=1; echo "compiled with $CKINC"; break
  fi
  echo "failed with $CKINC; first errors:"; grep -m 8 "error" "$PROBE_LOG" || head -20 "$PROBE_LOG"
done
if [ "$CK_OK" = "1" ]; then
  result PASS "5 CK FMHA 576/512 instantiates on gfx942: mla_attend can wrap DecodePipeline (D12); read the LDS size printed by the probe"
else
  result FAIL "5 CK FMHA 576/512 does not instantiate; mla_attend is the spec kernel (docs/mla-decode/04-our-kernel-spec.md); log $PROBE_LOG"
fi

# ---------------------------------------------------------------------------
step "6. rocprofv3 counters (docs/mi300x/06-profiling.md)"
AVAIL=""
for c in "rocprofv3 --list-avail" "rocprofv3 -L" "rocprofv3 --list-metrics"; do
  if AVAIL=$(bash -c "$c" 2>&1); then echo "[rocprofv3] worked: $c"; break; fi
  AVAIL=""
done
if [ -z "$AVAIL" ]; then
  result UNKNOWN "6 rocprofv3: no listing command worked"
else
  MISSING=""
  for ctr in TCC_EA0_RDREQ TCC_EA0_WRREQ TCC_HIT TCC_MISS SQ_LEVEL_WAVES SQ_ACCUM_PREV_HIRES; do
    if echo "$AVAIL" | grep -q "$ctr"; then echo "   $ctr: yes"; else echo "   $ctr: NO"; MISSING="$MISSING $ctr"; fi
  done
  if [ -z "$MISSING" ]; then
    result PASS "6 rocprofv3 counters: all design names present"
  else
    result FAIL "6 rocprofv3 counters missing:$MISSING (find the MI300 names in the listing and fix measure.py)"
  fi
fi

# ---------------------------------------------------------------------------
step "7. CU count seen by the runtime (296 workers + 8 schedulers expected)"
set +e
CU_OUT=$(python - <<'PYEOF'
import torch
p = torch.cuda.get_device_properties(0)
print("device", p.name, "CUs", p.multi_processor_count)
try:
    from mirage.utils import get_configurations_from_gpu   # python/mirage/utils.py
    print("workers, schedulers from the runtime:", get_configurations_from_gpu(0))
except Exception as e:
    print("mirage.utils query failed:", e)
assert p.multi_processor_count == 304, p.multi_processor_count
PYEOF
)
CU_RC=$?
set -e
echo "$CU_OUT"
if [ "$CU_RC" -eq 0 ]; then
  result PASS "7 CU count: 304 CUs (8 x 38); $(echo "$CU_OUT" | grep -o 'workers, schedulers.*' || true)"
else
  result UNKNOWN "7 CU count: torch or mirage query failed (see output above)"
fi

# ---------------------------------------------------------------------------
step "summary"
printf '%s\n' "${RESULTS[@]}"
echo "written: $ROOT/env/check_day1.log and $LOG"
