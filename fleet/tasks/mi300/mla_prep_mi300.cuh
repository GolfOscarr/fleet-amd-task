/* mla_prep_mi300: one CU-task per layer (docs/design-doc/02-task-graph.md A3).
 *
 *   c_kv[step]  = kv_a_layernorm(qkva[Q_OUT : Q_OUT + D_C])          (BF16)
 *   k_pe[step]  = RoPE(qkva[Q_OUT + D_C : Q_OUT + D_C + D_R], step)   (BF16)
 *   q_pe[h]     = RoPE(q[h, D_N : D_N + D_R], step)                   (BF16)
 *   ql_nope[h]  = q[h, 0 : D_N] @ W_uk[h]      [1, D_N] x [D_N, D_C]  (FP32 acc, BF16 store)
 *
 * Math and rounding: harness/numpy_ref.py mla_prep. Pointer order and
 * params: fleet/build_graph.py mla_prep_layer.
 *
 * Inputs : qkva [1, NH*(D_N+D_R) + D_C + D_R], w_kv_norm [D_C], W_uk [NH, D_N, D_C],
 *          cos [S_max, D_R], sin [S_max, D_R]
 * Outputs: c_kv [S_max, D_C] (row step), k_pe [S_max, D_R] (row step),
 *          ql_nope [NH, D_C], q_pe [NH, D_R]
 * LDS    : q_nope staged as FP32 (NH*D_N*4 = 8 KiB), 4 partial rows for the
 *          W_uk product (4*D_C*4 = 8 KiB), 4 floats of reduction scratch.
 *
 * The 2 MiB W_uk read is spread over all 256 lanes with 16-byte loads: each
 * wave owns a quarter of the D_N rows of a head, each lane 8 consecutive
 * columns, so a wave reads one full 1 KiB row of W_uk[h] per step of n.
 */
#pragma once
#include "tasks/common/common_header.cuh"
#include "tasks/mi300/mla_common_mi300.cuh"

namespace kernel {

template <typename T, int NH, int D_N, int D_R, int D_C>
__device__ __forceinline__ void
    mla_prep_mi300_task_impl(void const *qkva_ptr,
                             void const *w_kv_norm_ptr,
                             void const *w_uk_ptr,
                             void const *cos_ptr,
                             void const *sin_ptr,
                             void *c_kv_ptr,
                             void *k_pe_ptr,
                             void *ql_nope_ptr,
                             void *q_pe_ptr,
                             int step,
                             float eps) {
  using namespace dsv2;
  static_assert(D_R % 2 == 0, "RoPE slice must be even");
  static_assert(D_C % 8 == 0, "W_uk rows are read 8 columns at a time");
  static_assert(D_N % WAVES == 0, "each wave owns D_N / 4 rows of W_uk[h]");
  constexpr int Q_HEAD = D_N + D_R;
  constexpr int Q_OUT = NH * Q_HEAD;
  constexpr int NCG = D_C / 8;               // column groups of 8 (64 for D_C = 512)
  constexpr int N_PER_WAVE = D_N / WAVES;    // 32 for D_N = 128
  static_assert(NCG <= WAVE, "one wave must cover a full W_uk row");

  T const *qkva = static_cast<T const *>(qkva_ptr);
  T const *w_kv_norm = static_cast<T const *>(w_kv_norm_ptr);
  T const *w_uk = static_cast<T const *>(w_uk_ptr);
  T const *cos_row = static_cast<T const *>(cos_ptr) + (size_t)step * D_R;
  T const *sin_row = static_cast<T const *>(sin_ptr) + (size_t)step * D_R;
  T *c_kv_row = static_cast<T *>(c_kv_ptr) + (size_t)step * D_C;
  T *k_pe_row = static_cast<T *>(k_pe_ptr) + (size_t)step * D_R;
  T *ql_nope = static_cast<T *>(ql_nope_ptr);
  T *q_pe = static_cast<T *>(q_pe_ptr);

  extern __shared__ char smem[];
  float *q_nope_s = reinterpret_cast<float *>(smem);          // [NH][D_N]
  float *red_rows = q_nope_s + NH * D_N;                       // [WAVES][D_C]
  float *red = red_rows + WAVES * D_C;                         // [4]

  int tid = threadIdx.x;

  // 1. latent row: kv_a_layernorm of the D_C slice after the q rows
  rmsnorm_row<T, D_C>(qkva + Q_OUT, w_kv_norm, c_kv_row, eps, red);

  // 2. RoPE of the new k_pe row (D_R elements)
  if (tid < D_R) {
    float c = ld(cos_row + tid), s = ld(sin_row + tid);
    st(k_pe_row + tid, rope_elem<T, D_R>(qkva + Q_OUT + D_C, tid, c, s));
  }

  // 3. RoPE of q_pe for every head, and staging of q_nope as FP32
  for (int e = tid; e < NH * D_R; e += NUM_THREADS) {
    int h = e / D_R, i = e % D_R;
    float c = ld(cos_row + i), s = ld(sin_row + i);
    st(q_pe + h * D_R + i, rope_elem<T, D_R>(qkva + h * Q_HEAD + D_N, i, c, s));
  }
  for (int e = tid; e < NH * D_N; e += NUM_THREADS) {
    int h = e / D_N, n = e % D_N;
    q_nope_s[e] = ld(qkva + h * Q_HEAD + n);
  }
  __syncthreads();

  // 4. ql_nope[h] = q_nope[h] @ W_uk[h]: lane -> 8 columns, wave -> D_N / 4 rows
  int wave = tid / WAVE, lane = tid % WAVE;
  for (int h = 0; h < NH; h++) {
    float acc[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      acc[k] = 0.0f;
    }
    if (lane < NCG) {
      int c0 = lane * 8;
      T const *w_head = w_uk + (size_t)h * D_N * D_C;
      float const *q_head = q_nope_s + h * D_N;
      for (int n = wave * N_PER_WAVE; n < (wave + 1) * N_PER_WAVE; n++) {
        float w[8];
        load8(w_head + (size_t)n * D_C + c0, w);
        float q = q_head[n];
#pragma unroll
        for (int k = 0; k < 8; k++) {
          acc[k] += q * w[k];
        }
      }
#pragma unroll
      for (int k = 0; k < 8; k++) {
        red_rows[wave * D_C + c0 + k] = acc[k];
      }
    }
    __syncthreads();
    for (int c = tid; c < D_C; c += NUM_THREADS) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < WAVES; w++) {
        sum += red_rows[w * D_C + c];
      }
      st(ql_nope + h * D_C + c, bf16r(sum));
    }
    __syncthreads();
  }
}

} // namespace kernel
