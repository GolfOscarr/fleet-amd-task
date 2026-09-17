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
 * LDS    : split weights [64], o [D_C], total [4], the split groups' partial sums [4][D_C] FP32: about 10.5 KiB.
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
  // W_uv rows of 8 in flight per thread (raw BF16 words, 4 registers each): the whole half row
  constexpr int PF_W = 16;                       // two batches per half row (32 measured 1% slower on the model: registers)
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
  float *tot_s = o_s + D_C;                          // [4] (one used; keeps red_s 16-byte aligned)
  float *red_s = tot_s + 4;                          // [GROUPS][D_C]: the split groups' partial sums
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

  // o[c] = sum_j w_j o_j[c] / sum_j w_j, rounded to BF16. Thread t owns eight consecutive
  // columns (chunk q = t % CHUNKS, two 16-byte loads per split row) and the split rows
  // j = s, s + GROUPS, ... (s = t / CHUNKS): every row of a batch is loaded before the first
  // multiply, the FMAs run in ascending j, and the GROUPS partial sums are added in order
  // through LDS (round 3: the per-column scalar form was six HBM round trips per thread)
  constexpr int CHUNK = 8;
  constexpr int CHUNKS = D_C / CHUNK;
  constexpr int GROUPS = NUM_THREADS / CHUNKS;
  constexpr int ROWS_IN_FLIGHT = 12;                  // 24 16-byte loads (96 registers); 33 live splits over 4 groups fit one batch
  static_assert(D_C % CHUNK == 0 && NUM_THREADS % CHUNKS == 0, "the thread map covers the row");
  static_assert((P_ROW * 4) % 16 == 0, "the partials rows are 16-byte aligned");
  float inv_tot = 1.0f / tot_s[0];
  {
    int q = tid % CHUNKS, s = tid / CHUNKS;
    float acc[CHUNK];
#pragma unroll
    for (int k = 0; k < CHUNK; k++) {
      acc[k] = 0.0f;
    }
    for (int j0 = s; j0 < live; j0 += GROUPS * ROWS_IN_FLIGHT) {
      uint4 v[ROWS_IN_FLIGHT][2];                   // FP32 words, converted bit-exactly on use
#pragma unroll
      for (int u = 0; u < ROWS_IN_FLIGHT; u++) {
        int j = j0 + u * GROUPS;
        if (j < live) {
          float const *row = partials + ((size_t)j * NH + h) * P_ROW + q * CHUNK;
          v[u][0] = *reinterpret_cast<uint4 const *>(row);
          v[u][1] = *reinterpret_cast<uint4 const *>(row + 4);
        }
      }
#pragma unroll
      for (int u = 0; u < ROWS_IN_FLIGHT; u++) {
        int j = j0 + u * GROUPS;
        if (j < live) {
          float wj = w_s[j];
          acc[0] += wj * __uint_as_float(v[u][0].x); acc[1] += wj * __uint_as_float(v[u][0].y);
          acc[2] += wj * __uint_as_float(v[u][0].z); acc[3] += wj * __uint_as_float(v[u][0].w);
          acc[4] += wj * __uint_as_float(v[u][1].x); acc[5] += wj * __uint_as_float(v[u][1].y);
          acc[6] += wj * __uint_as_float(v[u][1].z); acc[7] += wj * __uint_as_float(v[u][1].w);
        }
      }
    }
#pragma unroll
    for (int k = 0; k < CHUNK; k++) {
      red_s[s * D_C + q * CHUNK + k] = acc[k];
    }
  }
  __syncthreads();
  for (int c = tid; c < D_C; c += NUM_THREADS) {
    float o = 0.0f;
#pragma unroll
    for (int g = 0; g < GROUPS; g++) {
      o += red_s[g * D_C + c];
    }
    o_s[c] = bf16r(o * inv_tot);
  }
  __syncthreads();

  // attn[v] = o . W_uv[h, v, :], two lanes per v (each half of D_C); PF_W rows of 8 kept as
  // raw BF16 words (4 registers each) so sixteen loads are in flight: two round trips
  static_assert(sizeof(T) == 2, "the raw-word conversion below is BF16's");
  int half = tid & 1;
  for (int v = tid >> 1; v < D_V; v += NUM_THREADS / 2) {
    T const *row = w_uv + (size_t)v * D_C;
    float acc = 0.0f;
    for (int c = half * (D_C / 2); c < (half + 1) * (D_C / 2); c += 8 * PF_W) {
      uint4 raw[PF_W];
#pragma unroll
      for (int u = 0; u < PF_W; u++) {
        raw[u] = *reinterpret_cast<uint4 const *>(row + c + 8 * u);
      }
#pragma unroll
      for (int u = 0; u < PF_W; u++) {
        unsigned const *words = reinterpret_cast<unsigned const *>(&raw[u]);
#pragma unroll
        for (int k = 0; k < 8; k++) {
          unsigned bits = (k & 1) ? (words[k / 2] & 0xffff0000u) : (words[k / 2] << 16);
          acc += o_s[c + 8 * u + k] * __uint_as_float(bits);
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
