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
for f in mla_prep_mi300 mla_attend_mi300 mla_attend_mfma_mi300 mla_merge_uv_mi300 moe_router_mi300 copy_mi300 prefetch_mi300; do
  src="$KDIR/$f.cuh"
  # instantiate each template at the real dims so the bodies are parsed and type-checked
  case "$f" in
    mla_prep_mi300) inst='kernel::mla_prep_mi300_task_impl<bfloat16,16,128,64,512>(0,0,0,0,0,0,0,0,0,0,1e-6f,0);' ;;
    mla_attend_mi300) inst='kernel::mla_attend_mi300_task_impl<bfloat16,16,512,64,1056>(0,0,0,0,0,0,0.1f,32,33,5,4,0,0);' ;;
    mla_attend_mfma_mi300) inst='kernel::mla_attend_mi300_task_impl<bfloat16,16,512,64,1056>(0,0,0,0,0,0,0.1f,32,33,5,4,0,0);' ;;   # O7: the MFMA kernel (the same signature; the wrapper's host path)
    mla_merge_uv_mi300) inst='kernel::mla_merge_uv_mi300_task_impl<bfloat16,16,128,512>(0,0,0,0,32,33,2,2,0,1,1,0);' ;;
    moe_router_mi300) inst='kernel::moe_router_mi300_task_impl<bfloat16,2048,64,2,6,32,26,false>(0,0,0,0,0,0,0,0,0,0,0,0,1.0f,0.0f); kernel::moe_router_mi300_task_impl<bfloat16,2048,64,2,6,32,26,true>(0,0,0,0,0,0,0,0,0,0,0,0,1.0f,1e-6f);' ;;
    copy_mi300) inst='kernel::copy_mi300_task_impl<bfloat16,2048>(0,0); kernel::copy_mi300_task_impl<bfloat16,256>(0,0,1000,1);' ;;
    prefetch_mi300) inst='kernel::prefetch_mi300_task_impl<bfloat16,32,2048>(0,0); kernel::prefetch_moe_mi300_task_impl<bfloat16,66,2048,1408,32>(0,0,0,0);' ;;   # O8
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
# the launcher, in both build variants (the debug one adds the scores output)
for variant in "" "-DMLA_ATTEND_DEBUG_SCORES"; do
  label="kernel_tests_mi300${variant:+ $variant}"
  if out=$("$CXX" -std=c++17 -fsyntax-only -Wall -Wno-unused-parameter -Wno-unused-variable $variant -I "$STUB" -I "$ROOT/fleet" -x c++ "$ROOT/fleet/tasks/kernel_tests_mi300.cu" 2>&1); then
    echo "PASS $label"
  else
    echo "FAIL $label"; echo "$out" | head -30; fail=1
  fi
done
exit $fail
