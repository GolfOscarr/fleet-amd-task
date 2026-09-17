#!/usr/bin/env bash
# Syntax check only: parses each kernel with the host clang++ against the
# stub headers in fleet/tasks/stub (no HIP, no device code generation, no
# semantics), then the launcher kernel_tests_mi300.cu against the same stubs
# plus stub/hip/hip_runtime.h. Correctness is established on the GPU
# (fleet/tasks/kernel_tests.py).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STUB="$ROOT/fleet/tasks/stub"
KDIR="$ROOT/fleet/tasks/mi300"
CXX="${CXX:-clang++}"
# "tasks/mi300/<name>.cuh" resolves under fleet/ (fleet/tasks/mi300), "tasks/common/..." under the stub
fail=0
for f in mla_prep_mi300 mla_attend_mi300 mla_attend_mfma_mi300 mla_merge_uv_mi300 mla_merge_oproj_mi300 moe_router_mi300 copy_mi300 prefetch_mi300 linear_gemv_mi300 stream_mi300 gang_moe_w2_silu_mi300 gang_moe_w13_gemv_mi300; do
  src="$KDIR/$f.cuh"
  # instantiate each template at the real dims so the bodies are parsed and type-checked
  case "$f" in
    mla_prep_mi300) inst='kernel::mla_prep_mi300_task_impl<bfloat16,16,128,64,512>(0,0,0,0,0,0,0,0,0,0,1e-6f,0);' ;;
    mla_attend_mi300) inst='kernel::mla_attend_mi300_task_impl<bfloat16,16,512,64,1056>(0,0,0,0,0,0,0.1f,32,33,5,4,0,0);' ;;
    mla_attend_mfma_mi300) inst='kernel::mla_attend_mi300_task_impl<bfloat16,16,512,64,1056>(0,0,0,0,0,0,0.1f,32,33,5,4,0,0);' ;;   # O7: the MFMA kernel (the same signature; the wrapper's host path)
    # the gang form, then N4's regular one at a whole head and at a half (the two HALVES)
    mla_merge_uv_mi300) inst='kernel::mla_merge_uv_mi300_task_impl<bfloat16,16,128,512>(0,0,0,0,32,33,2,2,0,1,1,0); kernel::mla_merge_uv_tile_mi300_task_impl<bfloat16,16,128,512,1>(0,0,0,0,32,33,0); kernel::mla_merge_uv_tile_mi300_task_impl<bfloat16,16,128,512,2>(0,0,0,0,32,33,0);' ;;
    # N5: the merge with o_proj folded in, one task per (head, half)
    mla_merge_oproj_mi300) inst='kernel::mla_merge_oproj_mi300_task_impl<bfloat16,16,128,512,2048,2>(0,0,0,0,0,0,0,0,32,33,0);' ;;
    # the plain and the fused one-task forms, then N2's four-task split of the fused one
    moe_router_mi300) inst='kernel::moe_router_mi300_task_impl<bfloat16,2048,64,2,6,32,26,false>(0,0,0,0,0,0,0,0,0,0,0,0,1.0f,0.0f); kernel::moe_router_mi300_task_impl<bfloat16,2048,64,2,6,32,26,true>(0,0,0,0,0,0,0,0,0,0,0,0,1.0f,1e-6f); kernel::moe_router_mi300_task_impl<bfloat16,2048,64,2,6,32,26,true,4>(0,0,0,0,0,0,0,0,0,0,0,0,1.0f,1e-6f,0,0);' ;;
    copy_mi300) inst='kernel::copy_mi300_task_impl<bfloat16,2048>(0,0); kernel::copy_mi300_task_impl<bfloat16,256>(0,0,1000,1);' ;;
    prefetch_mi300) inst='kernel::prefetch_mi300_task_impl<bfloat16,32,2048>(0,0); kernel::prefetch_moe_mi300_task_impl<bfloat16,66,2048,1408,32>(0,0,0,0);' ;;   # O8
    # L1: the three forms of the GEMV linear (the norm, the residual, and the plain one)
    linear_gemv_mi300) inst='kernel::linear_gemv_mi300_task_impl<bfloat16,2048,true,false>(0,0,0,0,0,38,3648,1e-6f); kernel::linear_gemv_mi300_task_impl<bfloat16,2048,false,true>(0,0,0,0,0,32,2048,0.0f); kernel::linear_gemv_mi300_task_impl<bfloat16,2048,false,false>(0,0,0,0,0,38,3648,0.0f);' ;;
    # L6: the stream probe, the regular and the gang entry points
    stream_mi300) inst='kernel::stream_mi300_task_impl<bfloat16,2048>(0,0,38); kernel::stream_gang_mi300_task_impl<bfloat16,2048>(0,0,76,37,0);' ;;
    gang_moe_w2_silu_mi300) inst='kernel::gang_moe_w2_silu_linear_kernel<bfloat16,1,2048,2048,1408,2816,66,8,32,32,32>(0,0,0,0,0,0,0);' ;;   # L3: the GEMV form, the default path (the MPK_W2_CK_TILE one needs CK)
    gang_moe_w13_gemv_mi300) inst='kernel::gang_moe_w13_gemv_kernel<bfloat16,2816,2048,66,8,37>(0,0,0,0,0,0);' ;;   # L4: the expert gate-up in 37 tiles per XCD
  esac
  tu="$(mktemp -t "$f.XXXXXX").cpp"
  printf '#define MLA_ATTEND_DEBUG_SCORES 1\n#include "tasks/mi300/%s.cuh"\nvoid instantiate() { %s }\n' "$f" "$inst" > "$tu"
  if out=$("$CXX" -std=c++17 -fsyntax-only -Wall -Wno-unused-parameter -Wno-unused-variable -I "$STUB" -I "$ROOT/fleet" -x c++ "$tu" 2>&1); then
    echo "PASS $f"
  else
    echo "FAIL $f"; echo "$out" | head -30; fail=1
  fi
  rm -f "$tu"
done
# the launcher, in its three build variants (the debug one adds the scores output, the fake-XCD
# one turns on the two MoE gang rows of L3 and L4)
for variant in "" "-DMLA_ATTEND_DEBUG_SCORES" "-DKT_FAKE_XCD"; do
  label="kernel_tests_mi300${variant:+ $variant}"
  if out=$("$CXX" -std=c++17 -fsyntax-only -Wall -Wno-unused-parameter -Wno-unused-variable $variant -I "$STUB" -I "$ROOT/fleet" -x c++ "$ROOT/fleet/tasks/kernel_tests_mi300.cu" 2>&1); then
    echo "PASS $label"
  else
    echo "FAIL $label"; echo "$out" | head -30; fail=1
  fi
done
exit $fail
