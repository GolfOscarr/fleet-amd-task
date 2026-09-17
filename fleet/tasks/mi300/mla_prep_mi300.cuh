/* mla_prep_mi300: one task per head (NH tasks per layer; round 3, session S9:
 * the single task cost 143 us per layer, one CU reading 2 MiB of W_uk, and
 * was misnamed as the attention's cost by round 2's event table).
 *
 *   c_kv[step]  = kv_a_layernorm(qkva[Q_OUT : Q_OUT + D_C])          (BF16)   task 0 only
 *   k_pe[step]  = RoPE(qkva[Q_OUT + D_C : Q_OUT + D_C + D_R], step)   (BF16)   task 0 only
 *   q_pe[h]     = RoPE(q[h, D_N : D_N + D_R], step)                   (BF16)   task h
 *   ql_nope[h]  = q[h, 0 : D_N] @ W_uk[h]      [1, D_N] x [D_N, D_C]  (FP32 acc, BF16 store)   task h
 *
 * Math and rounding: harness/numpy_ref.py mla_prep (the accumulation order
 * over n and over the four waves is the round-2 kernel's). Pointer order and
 * params: fleet/build_graph.py mla_prep_layer. The task's head index comes
 * from the runtime's expert_offset metadata (bid.x), as the per-split
 * attention task's and the expert prefetch's do; every tensor is handed to
 * the task whole (no imap), the kernel indexes row h itself.
 *
 * Inputs : qkva [1, NH*(D_N+D_R) + D_C + D_R], w_kv_norm [D_C], W_uk [NH, D_N, D_C],
 *          cos [S_max, D_R], sin [S_max, D_R]
 * Outputs: c_kv [S_max, D_C] (row step), k_pe [S_max, D_R] (row step),
 *          ql_nope [NH, D_C] (row h), q_pe [NH, D_R] (row h)
 * LDS    : q_nope[h] staged as FP32 (D_N*4 = 512 B), 4 partial rows for the
 *          W_uk product (4*D_C*4 = 8 KiB), 4 floats of reduction scratch.
 *
 * The 128 KiB W_uk[h] read is spread over all 256 lanes with 16-byte loads:
 * each wave owns a quarter of the D_N rows, each lane 8 consecutive columns,
 * eight rows' loads in flight per lane.
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
                             float eps,
                             int head) {
  using namespace dsv2;
  static_assert(D_R % 2 == 0, "RoPE slice must be even");
  static_assert(D_C % 8 == 0, "W_uk rows are read 8 columns at a time");
  static_assert(D_N % WAVES == 0, "each wave owns D_N / 4 rows of W_uk[h]");
  constexpr int Q_HEAD = D_N + D_R;
  constexpr int Q_OUT = NH * Q_HEAD;
  constexpr int NCG = D_C / 8;               // column groups of 8 (64 for D_C = 512)
  constexpr int N_PER_WAVE = D_N / WAVES;    // 32 for D_N = 128
  constexpr int ROWS_IN_FLIGHT = 8;          // W_uk rows loaded before the first multiply
  static_assert(NCG <= WAVE, "one wave must cover a full W_uk row");
  static_assert(N_PER_WAVE % ROWS_IN_FLIGHT == 0, "the wave's rows split into whole groups");

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
  float *q_nope_s = reinterpret_cast<float *>(smem);          // [D_N]
  float *red_rows = q_nope_s + D_N;                            // [WAVES][D_C]
  float *red = red_rows + WAVES * D_C;                         // [4]

  int tid = threadIdx.x;
  int h = head;

  // 1, 2. the latent row and the new k_pe row: written once, by the first task
  //       (the block-uniform branch keeps the norm's barriers uniform)
  if (h == 0) {
    rmsnorm_row<T, D_C>(qkva + Q_OUT, w_kv_norm, c_kv_row, eps, red);
    if (tid < D_R) {
      float c = ld(cos_row + tid), s = ld(sin_row + tid);
      st(k_pe_row + tid, rope_elem<T, D_R>(qkva + Q_OUT + D_C, tid, c, s));
    }
  }

  // 3. RoPE of q_pe[h], and staging of q_nope[h] as FP32
  if (tid < D_R) {
    float c = ld(cos_row + tid), s = ld(sin_row + tid);
    st(q_pe + h * D_R + tid, rope_elem<T, D_R>(qkva + h * Q_HEAD + D_N, tid, c, s));
  }
  for (int n = tid; n < D_N; n += NUM_THREADS) {
    q_nope_s[n] = ld(qkva + h * Q_HEAD + n);
  }
  __syncthreads();

  // 4. ql_nope[h] = q_nope[h] @ W_uk[h]: lane -> 8 columns, wave -> D_N / 4 rows,
  //    eight rows' loads issued before their multiplies (the accumulation order
  //    over n and over the waves is unchanged)
  int wave = tid / WAVE, lane = tid % WAVE;
  float acc[8];
#pragma unroll
  for (int k = 0; k < 8; k++) {
    acc[k] = 0.0f;
  }
  if (lane < NCG) {
    int c0 = lane * 8;
    T const *w_head = w_uk + (size_t)h * D_N * D_C;
    for (int n0 = wave * N_PER_WAVE; n0 < (wave + 1) * N_PER_WAVE; n0 += ROWS_IN_FLIGHT) {
      float w[ROWS_IN_FLIGHT][8];
#pragma unroll
      for (int u = 0; u < ROWS_IN_FLIGHT; u++) {
        load8(w_head + (size_t)(n0 + u) * D_C + c0, w[u]);
      }
#pragma unroll
      for (int u = 0; u < ROWS_IN_FLIGHT; u++) {
        float q = q_nope_s[n0 + u];
#pragma unroll
        for (int k = 0; k < 8; k++) {
          acc[k] += q * w[u][k];
        }
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
}

} // namespace kernel
