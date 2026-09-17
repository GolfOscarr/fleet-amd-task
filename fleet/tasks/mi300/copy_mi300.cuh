/* copy_mi300: identity copy of a [1, N] BF16 row (the per-task pointers
 * select the row when the tensors are partitioned on dim 0). Debug builds
 * (fleet/build_graph.py --debug): a snapshot of the residual after each
 * layer for the error-growth curve (docs/design-doc/07-correctness.md);
 * --probe-before (O5): the one-task copy in front of an operator; the empty
 * ladder (I3): M operators of N such tasks.
 *
 * Spin mode (I2 of docs/gpu-experiments/03-acceleration): with spin > 0,
 * thread 0 runs `spin` iterations of a dependent integer chain after the
 * copy and, with print, reports the deltas of clock64 (the shader clock,
 * whose frequency is the SCLK) and of s_memrealtime (100 MHz on the
 * MI300X) as a [SPIN] line, so the cycle counts of the worker timing (I1)
 * convert to microseconds: MHz = cycles / (ticks x 10 ns). The chain's
 * result is printed so the loop cannot be dropped.
 */
#pragma once
#include "tasks/common/common_header.cuh"

namespace kernel {

template <typename T, int N>
__device__ __forceinline__ void copy_mi300_task_impl(void const *in_ptr, void *out_ptr, int spin = 0, int print = 0) {
  T const *in = static_cast<T const *>(in_ptr);
  T *out = static_cast<T *>(out_ptr);
  for (int i = threadIdx.x; i < N; i += NUM_THREADS) {
    out[i] = in[i];
  }
  if (spin > 0 && threadIdx.x == 0) {
#if defined(__HIP_DEVICE_COMPILE__)
    unsigned long long c0 = clock64();
    unsigned long long r0 = __builtin_amdgcn_s_memrealtime();
#else
    unsigned long long c0 = 0, r0 = 0;
#endif
    unsigned x = 12345u;
    for (int i = 0; i < spin; i++) {
      x = x * 1664525u + 1013904223u;   // a dependent chain: one multiply-add per iteration
    }
#if defined(__HIP_DEVICE_COMPILE__)
    unsigned long long c1 = clock64();
    unsigned long long r1 = __builtin_amdgcn_s_memrealtime();
#else
    unsigned long long c1 = 0, r1 = 0;
#endif
    if (print) {
      printf("[SPIN] block=%d iters=%d cycles=%llu ticks=%llu x=%u\n",
             (int)blockIdx.x, spin, c1 - c0, r1 - r0, x);
    }
  }
}

} // namespace kernel
