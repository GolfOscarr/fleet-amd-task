/* mla_merge_uv_mi300: gang task, 8 XCD slots x heads_per_xcd tiles
 * (docs/design-doc/02-task-graph.md A5). One tile = one head.
 *
 * Tile decode: xcd = tile_idx / tiles_per_xcd, t = tile_idx % tiles_per_xcd,
 * head h = xcd * heads_per_xcd + t (n_tile_start = bid.x * tiles_per_xcd as
 * for mla_attend).
 *
 * Math and rounding: harness/numpy_ref.py merge_partials + mla_merge_uv.
 * live = ceil((step + 1) / split) splits; M = max_j lse_j; w_j = exp(lse_j - M);
 * o = sum_j w_j o_j / sum_j w_j (FP32), rounded to BF16 (an MFMA operand);
 * attn[h * D_V + v] = sum_c o[c] * W_uv[h, v, c] with FP32 accumulation, BF16 store.
 *
 * Inputs : partials [n_splits, NH, D_C + 1] FP32, W_uv [NH, D_V, D_C] BF16
 * Outputs: attn [1, NH * D_V] BF16
 * Pointer conventions (computed by the registration from the imaps):
 *   partials_xcd_offset_rows: rows already added for this XCD (0 if unpartitioned)
 *   w_uv_local: 1 if the W_uv pointer is this XCD's [heads_per_xcd, D_V, D_C] slice
 *   out_local : 1 if the attn pointer is this XCD's heads_per_xcd * D_V columns
 * LDS    : split weights [64] + total, o [D_C] FP32: about 2.3 KiB.
 */
#pragma once
#include "tasks/common/common_header.cuh"
#include "tasks/mi300/mla_common_mi300.cuh"

namespace kernel {

template <typename T, int NH, int D_V, int D_C>
__device__ __forceinline__ void
    mla_merge_uv_mi300_task_impl(void const *partials_ptr,
                                 void const *w_uv_ptr,
                                 void *attn_ptr,
                                 int step,
                                 int split,
                                 int n_splits,
                                 int tiles_per_xcd,
                                 int heads_per_xcd,
                                 int partials_xcd_offset_rows,
                                 int w_uv_local,
                                 int out_local,
                                 int tile_idx) {
  using namespace dsv2;
  static_assert(D_C % 16 == 0, "two lanes share a row of W_uv, 8 columns per load");
  static_assert(2 * D_V <= NUM_THREADS, "two threads per output element");

  int xcd = tile_idx / tiles_per_xcd;
  int t = tile_idx % tiles_per_xcd;
  if (t >= heads_per_xcd) {
    return;
  }
  int h = xcd * heads_per_xcd + t;
  int live = (step + split) / split;                // ceil((step + 1) / split)
  if (live > n_splits) {
    live = n_splits;
  }
  float const *partials = static_cast<float const *>(partials_ptr)
      - (size_t)xcd * partials_xcd_offset_rows * NH * (D_C + 1);
  T const *w_uv = static_cast<T const *>(w_uv_ptr)
      + (size_t)(w_uv_local ? t : h) * D_V * D_C;
  T *attn = static_cast<T *>(attn_ptr) + (size_t)(out_local ? t : h) * D_V;

  extern __shared__ char smem[];
  float *w_s = reinterpret_cast<float *>(smem);      // [64]
  float *o_s = w_s + WAVE;                           // [D_C]
  float *tot_s = o_s + D_C;                          // [1]
  int tid = threadIdx.x;
  int lane = tid % WAVE;

  // merge weights over the live splits (at most 64), wave 0
  if (tid < WAVE) {
    float lse = (lane < live) ? partials[((size_t)lane * NH + h) * (D_C + 1) + D_C] : -INFINITY;
    float M = wave_max(lse);
    float w = (lane < live) ? expf(lse - M) : 0.0f;
    w_s[lane] = w;
    float tot = wave_sum(w);
    if (lane == 0) {
      tot_s[0] = tot;
    }
  }
  __syncthreads();

  // o[c] = sum_j w_j o_j[c] / sum_j w_j, rounded to BF16
  float inv_tot = 1.0f / tot_s[0];
  for (int c = tid; c < D_C; c += NUM_THREADS) {
    float o = 0.0f;
    for (int j = 0; j < live; j++) {
      o += w_s[j] * partials[((size_t)j * NH + h) * (D_C + 1) + c];
    }
    o_s[c] = bf16r(o * inv_tot);
  }
  __syncthreads();

  // attn[v] = o . W_uv[h, v, :], two lanes per v (each half of D_C)
  int half = tid & 1;
  for (int v = tid >> 1; v < D_V; v += NUM_THREADS / 2) {
    T const *row = w_uv + (size_t)v * D_C;
    float acc = 0.0f;
    for (int c = half * (D_C / 2); c < (half + 1) * (D_C / 2); c += 8) {
      float w[8];
      load8(row + c, w);
#pragma unroll
      for (int k = 0; k < 8; k++) {
        acc += o_s[c + k] * w[k];
      }
    }
    acc += __shfl_xor(acc, 1, WAVE);
    if (half == 0) {
      st(attn + v, bf16r(acc));
    }
  }
}

} // namespace kernel
