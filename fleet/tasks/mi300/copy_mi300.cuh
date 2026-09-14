/* copy_mi300: identity copy of a [1, N] BF16 tensor. Debug builds only
 * (fleet/build_graph.py --debug): a snapshot of the residual after each
 * layer for the error-growth curve (docs/design-doc/07-correctness.md).
 */
#pragma once
#include "tasks/common/common_header.cuh"

namespace kernel {

template <typename T, int N>
__device__ __forceinline__ void copy_mi300_task_impl(void const *in_ptr, void *out_ptr) {
  T const *in = static_cast<T const *>(in_ptr);
  T *out = static_cast<T *>(out_ptr);
  for (int i = threadIdx.x; i < N; i += NUM_THREADS) {
    out[i] = in[i];
  }
}

} // namespace kernel
