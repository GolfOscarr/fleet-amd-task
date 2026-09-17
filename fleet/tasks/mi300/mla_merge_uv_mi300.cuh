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
 * Inputs : partials [n_splits, NH, P_ROW] FP32 (P_ROW = D_C+1 padded to /4), W_uv [NH, D_V, D_C] BF16
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
  constexpr int P_ROW = ((D_C + 1 + 3) / 4) * 4;   // padded partials row (P2); o in [0,D_C), lse at D_C
  static_assert(2 * D_V <= NUM_THREADS, "two threads per output element");
  // loads in flight per thread: PF_P split rows of the partials (round 3: at 4 the merge of 33
  // splits was 17 load round trips, 19 us per head task), PF_W rows of 8 of W_uv (64 FP32 registers)
  constexpr int PF_P = 16;
  constexpr int PF_W = 8;
  static_assert((D_C / 2) % (8 * PF_W) == 0, "the W_uv half-row loop batches PF_W loads of 8");

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
      - (size_t)xcd * partials_xcd_offset_rows * NH * P_ROW;
  StreamSrc<float> partials_stream(partials);         // O6: the partials are read once per iteration
  T const *w_uv = static_cast<T const *>(w_uv_ptr)
      + (size_t)(w_uv_local ? t : h) * D_V * D_C;
  T *attn = static_cast<T *>(attn_ptr) + (size_t)(out_local ? t : h) * D_V;

  extern __shared__ char smem[];
  float *w_s = reinterpret_cast<float *>(smem);      // [64]
  float *o_s = w_s + WAVE;                           // [D_C]
  float *tot_s = o_s + D_C;                          // [1]
  int tid = threadIdx.x;
  int lane = tid % WAVE;

  // merge weights over the live splits, one split per lane of wave 0: n_splits
  // must be at most 64 (asserted at registration and in build_graph.py)
  if (live > WAVE) {
    live = WAVE;   // unreachable when the asserts hold; never read past the wave
  }
  if (tid < WAVE) {
    float lse = (lane < live) ? ldf_from(partials_stream, ((size_t)lane * NH + h) * P_ROW + D_C) : -INFINITY;
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
    int j = 0;
    for (; j + PF_P <= live; j += PF_P) {     // PF_P split rows in flight, then the FMAs in order
      float v[PF_P];
#pragma unroll
      for (int u = 0; u < PF_P; u++) {
        v[u] = ldf_from(partials_stream, ((size_t)(j + u) * NH + h) * P_ROW + c);
      }
#pragma unroll
      for (int u = 0; u < PF_P; u++) {
        o += w_s[j + u] * v[u];
      }
    }
    for (; j < live; j++) {
      o += w_s[j] * ldf_from(partials_stream, ((size_t)j * NH + h) * P_ROW + c);
    }
    o_s[c] = bf16r(o * inv_tot);
  }
  __syncthreads();

  // attn[v] = o . W_uv[h, v, :], two lanes per v (each half of D_C)
  int half = tid & 1;
  for (int v = tid >> 1; v < D_V; v += NUM_THREADS / 2) {
    T const *row = w_uv + (size_t)v * D_C;
    float acc = 0.0f;
    for (int c = half * (D_C / 2); c < (half + 1) * (D_C / 2); c += 8 * PF_W) {
      float w[PF_W][8];
#pragma unroll
      for (int u = 0; u < PF_W; u++) {
        load8(row + c + 8 * u, w[u]);
      }
#pragma unroll
      for (int u = 0; u < PF_W; u++) {
#pragma unroll
        for (int k = 0; k < 8; k++) {
          acc += o_s[c + 8 * u + k] * w[u][k];
        }
      }
    }
    acc += __shfl_xor(acc, 1, WAVE);
    if (half == 0) {
      st(attn + v, bf16r(acc));
    }
  }
}

} // namespace kernel
