/* stream_mi300: the stream probe (M7 of docs/gpu-experiments/04-kernels/01-gemv-ideas.md,
 * L6 of 05-local-preparation.md). A task reads a slice of a [*, K] BF16 tensor with the
 * GEMV linear's load loop and does no multiply, so the event gap of an operator of such
 * tasks divided into the bytes it read is the rate that load loop achieves: per XCD for
 * the gang form (8 slots x tiles_per_xcd tiles) and per device for the regular form (one
 * task per block of the grid). That rate is the ceiling every gang linear is measured
 * against and the number S1 (w13 in one round) is judged by; the record cannot separate a
 * loaded round trip of about 3.5 us from an XCD that cannot draw its eighth of the
 * machine, and this probe can.
 *
 * stream_mi300_task_impl<T, K>(w, dummy, rows): `rows` rows of the [*, K] tensor whose
 *   pointer the runtime has already offset to this task's rows (the weight partitioned on
 *   dim 0 by the grid, as the prefetch task's stripe is); dummy is this task's [4] int32
 *   row of a [grid, 4] output, one XOR word per wave.
 * stream_gang_mi300_task_impl<T, K>(w, dummy, rows_per_tile, tiles_per_xcd, tile_idx):
 *   the whole tensor and the whole [8 * tiles_per_xcd, 4] dummy; the tile decodes its XCD
 *   slot as the merge does (xcd = tile_idx / tiles_per_xcd, t = tile_idx % tiles_per_xcd,
 *   the runtime setting n_tile_start = bid.x * tiles_per_xcd) and reads the rows
 *   [(xcd * tiles_per_xcd + t) * rows_per_tile, + rows_per_tile) into that tile's dummy row.
 *
 * The load loop is linear_gemv_mi300.cuh's, so the probe measures what the GEMV issues:
 * batches of STREAM_BATCH (8) rows under "#pragma unroll 1" (without the pragma the row
 * loop is unrolled and the loads hoisted, and the batch constant means nothing: the first
 * convention of the round), a row four 16-byte loads per lane at the element chunks
 * 8 * lane + 512 * i so that a wave-load is one contiguous kilobyte (the second), the
 * loads issued through StreamSrc (the sc1 nt policy of the linears under
 * -DMLA_NT_STREAMS), and the first batch issued inside the loop rather than before it
 * (the third). A batch is 32 loads per lane, 128 KB per CU in flight.
 *
 * One difference from the GEMV, for the byte count's sake: the GEMV gives wave w the
 * contiguous rows [w * ceil(rows / 4), ...), and a wave whose count is not a multiple of
 * the batch loads its last row again to fill the batch (the clamped rows of
 * linear_gemv_mi300.cuh, which keep a batch in one basic block). Four such partial
 * batches per task would be 24 row-loads of waste on a 38-row task, which is 63% of the
 * traffic the probe is there to measure. So the rows are cut into ceil(rows / BATCH)
 * whole batches and batch b is wave b % 4's: every wave still walks whole batches of
 * eight contiguous rows with the same map and the same pragma, all four waves have work
 * whenever there are four batches, and only the last batch of the task is partial (at
 * most 7 row-loads of waste, and they are the rows that batch has just read). The runs
 * this probe is built for read whole batches anyway: 152 KB is 38 rows (5 batches),
 * 256 KB is 64 (8) and 304 KB is 76 (10).
 *
 * The device against elision is the prefetch kernel's (prefetch_mi300.cuh): every 32-bit
 * word loaded is folded into one register per lane with XOR, the 64 lanes' words are
 * reduced by a wave-wide XOR butterfly, and lane 0 of each wave stores its word into
 * dummy[wave]. The dummy's value is deterministic (XOR is associative and commutative,
 * and a row read twice cancels), so the suite can check it exactly; nothing else is
 * written and no LDS is used.
 */
#pragma once
#include "tasks/common/common_header.cuh"
#include "tasks/mi300/mla_common_mi300.cuh"

// The batch depth, the GEMV's constant and swept the same way (-DSTREAM_BATCH=4|8|16):
// 4, 8 and 16 rows are 64, 128 and 256 KB in flight per CU.
#ifndef STREAM_BATCH
#define STREAM_BATCH 8
#endif

namespace kernel {

namespace stream_detail {

// The element offset of the lane's chunk i of a row: the coalesced map of K9, one
// contiguous kilobyte per wave-load (linear_gemv_mi300.cuh's chunk_elem without the
// strided A/B, which only changes the summation order the probe does not have).
template <int K>
__device__ __forceinline__ int chunk_elem(int lane, int i) {
  return 8 * lane + 8 * dsv2::WAVE * i;
}

// The batch's loads, all issued before any of them is waited on; a row past `rows` loads
// the last valid row again (a clamped index, no branch), as the GEMV's load_batch does.
template <typename T, int K, int BATCH, int CHUNKS>
__device__ __forceinline__ void load_batch(dsv2::StreamSrc<T> const &w_src, int r0, int rows,
                                           int lane, uint4 (&raw)[BATCH][CHUNKS]) {
#pragma unroll
  for (int u = 0; u < BATCH; u++) {
    int r = (r0 + u < rows) ? r0 + u : rows - 1;
#pragma unroll
    for (int i = 0; i < CHUNKS; i++) {
      raw[u][i] = dsv2::load16_from(w_src, (size_t)r * K + chunk_elem<K>(lane, i));
    }
  }
}

// The 64 lanes' words XORed into one (the prefetch kernel's wave_xor).
__device__ __forceinline__ unsigned wave_xor(unsigned x) {
#pragma unroll
  for (int off = dsv2::WAVE / 2; off > 0; off >>= 1) {
    x ^= __shfl_xor(x, off, dsv2::WAVE);
  }
  return x;
}

} // namespace stream_detail

template <typename T, int K = 2048>
__device__ __forceinline__ void stream_mi300_task_impl(void const *w_ptr, void *dummy_ptr, int rows) {
  using namespace dsv2;
  constexpr int BATCH = STREAM_BATCH;
  static_assert(sizeof(T) == 2, "the 16-byte loads below are BF16's");
  static_assert(K % (8 * WAVE) == 0, "16-byte loads, K / 64 elements per lane");
  static_assert(BATCH >= 1, "at least one row per batch");
  constexpr int CHUNKS = K / (8 * WAVE);      // 4 sixteen-byte loads per row per lane at K = 2048

  int wave = threadIdx.x / WAVE, lane = threadIdx.x % WAVE;
  StreamSrc<T> w_src(static_cast<T const *>(w_ptr));
  int batches = (rows + BATCH - 1) / BATCH;
  unsigned acc = 0;

#pragma unroll 1
  for (int b = wave; b < batches; b += WAVES) {
    uint4 raw[BATCH][CHUNKS];
    stream_detail::load_batch<T, K, BATCH, CHUNKS>(w_src, b * BATCH, rows, lane, raw);
#pragma unroll
    for (int u = 0; u < BATCH; u++) {
#pragma unroll
      for (int i = 0; i < CHUNKS; i++) {
        acc ^= raw[u][i].x ^ raw[u][i].y ^ raw[u][i].z ^ raw[u][i].w;
      }
    }
  }

  acc = stream_detail::wave_xor(acc);
  if (lane == 0) {
    static_cast<int *>(dummy_ptr)[wave] = static_cast<int>(acc);
  }
}

template <typename T, int K = 2048>
__device__ __forceinline__ void stream_gang_mi300_task_impl(void const *w_ptr,
                                                            void *dummy_ptr,
                                                            int rows_per_tile,
                                                            int tiles_per_xcd,
                                                            int tile_idx) {
  int xcd = tile_idx / tiles_per_xcd;              // the merge's decode; n_tile_start = bid.x * tiles_per_xcd
  int t = tile_idx % tiles_per_xcd;
  int tile = xcd * tiles_per_xcd + t;
  T const *w = static_cast<T const *>(w_ptr) + (size_t)tile * rows_per_tile * K;
  stream_mi300_task_impl<T, K>(w, static_cast<int *>(dummy_ptr) + tile * 4, rows_per_tile);
}

} // namespace kernel
