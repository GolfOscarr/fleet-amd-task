/* prefetch_mi300: the weight prefetch tasks of the side operators (O8,
 * docs/gpu-experiments/03-acceleration/03-local-preparation.md; idea D1 of
 * 01-ideas.md). A side operator's tasks run on the workers that hold no task
 * of the operator they accompany (the runtime patch of new_tasks.patch,
 * register_mugraph: a side operator's tasks take the dependent event of the
 * operator registered before it, sit behind its tasks in the worker queues,
 * and trigger the end-of-graph event). They stream a weight slice with
 * ordinary loads, which allocate the lines in the L2 and the 256 MB
 * memory-side cache, so the operator that reads the weight next finds them
 * there. Every loaded word is folded into an XOR that one lane per wave
 * writes to a dummy output, so the loads cannot be elided.
 *
 * prefetch_mi300_task_impl<T, ROWS, K>(w, dummy): a dense stripe [ROWS, K]
 *   (the weight partitioned on dim 0 by the grid: the runtime hands the task
 *   its stripe; dummy [grid, 4] int32 partitioned the same way, one row per
 *   task, one word per wave).
 * prefetch_moe_mi300_task_impl<T, E, N, K, PARTS>(w, mask, dummy, bid): part
 *   `bid % PARTS` of active expert slot `bid / PARTS` of a [E, N, K] expert
 *   weight (mask[E] = the number of active experts, mask[0..] their ids, as
 *   the stock MoE gang kernels read it); the task index comes from the
 *   runtime's expert_offset metadata (bid.x), as the per-split attention
 *   task's does. A slot past the active count returns without loading.
 */
#pragma once
#include "tasks/common/common_header.cuh"

namespace kernel {

namespace prefetch_detail {
constexpr int WAVE = 64;

__device__ __forceinline__ unsigned wave_xor(unsigned x) {
#pragma unroll
  for (int off = WAVE / 2; off > 0; off >>= 1) {
    x ^= __shfl_xor(x, off, WAVE);
  }
  return x;
}

// Streams `chunks` sixteen-byte chunks from `src` (eight loads in flight per
// thread) and writes the wave's XOR to dummy[wave].
__device__ __forceinline__ void stream_chunks(uint4 const *src, size_t chunks, int *dummy) {
  unsigned acc = 0;
  size_t c = threadIdx.x;
  constexpr int PF = 8;
  for (; c + (PF - 1) * NUM_THREADS < chunks; c += PF * NUM_THREADS) {
    uint4 v[PF];
#pragma unroll
    for (int u = 0; u < PF; u++) {
      v[u] = src[c + u * NUM_THREADS];
    }
#pragma unroll
    for (int u = 0; u < PF; u++) {
      acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
  }
  for (; c < chunks; c += NUM_THREADS) {
    uint4 v = src[c];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  acc = wave_xor(acc);
  if (threadIdx.x % WAVE == 0) {
    dummy[threadIdx.x / WAVE] = static_cast<int>(acc);
  }
}
} // namespace prefetch_detail

template <typename T, int ROWS, int K>
__device__ __forceinline__ void prefetch_mi300_task_impl(void const *w_ptr, void *dummy_ptr) {
  static_assert((static_cast<long long>(ROWS) * K * sizeof(T)) % 16 == 0, "16-byte chunks");
  constexpr size_t CHUNKS = static_cast<size_t>(ROWS) * K * sizeof(T) / 16;
  prefetch_detail::stream_chunks(static_cast<uint4 const *>(w_ptr), CHUNKS, static_cast<int *>(dummy_ptr));
}

template <typename T, int E, int N, int K, int PARTS>
__device__ __forceinline__ void prefetch_moe_mi300_task_impl(void const *w_ptr,
                                                             void const *mask_ptr,
                                                             void *dummy_ptr,
                                                             int bid) {
  static_assert(N % PARTS == 0, "the expert's rows split evenly over the parts");
  constexpr int ROWS = N / PARTS;
  static_assert((static_cast<long long>(ROWS) * K * sizeof(T)) % 16 == 0, "16-byte chunks");
  constexpr size_t CHUNKS = static_cast<size_t>(ROWS) * K * sizeof(T) / 16;
  int const *mask = static_cast<int const *>(mask_ptr);
  int slot = bid / PARTS, part = bid % PARTS;
  if (slot >= mask[E]) {
    return;                                        // fewer active experts than slots
  }
  int e = mask[slot];
  T const *w = static_cast<T const *>(w_ptr) + (static_cast<size_t>(e) * N + static_cast<size_t>(part) * ROWS) * K;
  prefetch_detail::stream_chunks(reinterpret_cast<uint4 const *>(w), CHUNKS, static_cast<int *>(dummy_ptr));
}

} // namespace kernel
