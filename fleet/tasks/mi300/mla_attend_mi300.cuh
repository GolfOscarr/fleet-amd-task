/* mla_attend_mi300: gang task, 8 XCD slots x tiles_per_xcd tiles
 * (docs/design-doc/02-task-graph.md A4; docs/mla-decode/04-our-kernel-spec.md
 * phase B). One tile = one split of `split` cache rows for all NH heads.
 *
 * Tile decode. The runtime sets task_metadata.n_tile_start = bid.x *
 * tiles_per_xcd for this task type (the same rule as TASK_GANG_ATTN_*), so
 *   xcd = tile_idx / tiles_per_xcd, t = tile_idx % tiles_per_xcd,
 *   split_id = t * 8 + xcd                                (02-task-graph.md)
 * split_id >= n_splits: return without writing. A split whose first position
 * is beyond `step` writes o = 0, lse = -inf.
 *
 * Math and rounding: harness/numpy_ref.py mla_attend. Scores in FP32,
 * online softmax in FP32, probabilities rounded to BF16 before the
 * p . c_kv accumulation, o stored normalized (acc / l) with
 * lse = m + ln(l) at column D_C: the CK split-KV convention the runtime's
 * merge uses (tasks/ampere/merge_splitkv.cuh). A split is processed in
 * passes of TILE = 32 rows with the running-max rescale; for split == 32 that
 * is numpy_ref's single pass exactly, for larger splits it differs from it
 * by FP32 reassociation only.
 *
 * Inputs : ql_nope [NH, D_C], q_pe [NH, D_R], c_kv [S_max, D_C], k_pe [S_max, D_R]
 * Outputs: partials [n_splits, NH, P_ROW] FP32 (P_ROW = D_C+1 padded to a multiple of 4; o in [0,D_C), lse at D_C)
 *          (debug builds, -DMLA_ATTEND_DEBUG_SCORES: scores [NH, S_max] FP32,
 *          the scaled pre-softmax scores, boundary B5)
 * partials_xcd_offset_rows: rows the runtime already added to the partials
 * pointer for this XCD (n_splits / 8 when the output imap partitions dim 0,
 * 0 when it does not); the kernel indexes splits absolutely.
 * LDS    : ql_nope BF16 (NH*D_C*2 = 16 KiB), q_pe BF16 (2 KiB), scores and
 *          probabilities FP32 [NH][32] (4 KiB), m, l, alpha [NH] (192 B):
 *          about 22.3 KiB. The accumulator [NH, D_C] lives in registers:
 *          each thread owns NH/4 heads x 8 columns (32 floats).
 *
 * This is the VALU version (correctness first); the MFMA 16x16x16 version of
 * the kernel spec is the later optimization.
 */
#pragma once
#include "tasks/common/common_header.cuh"
#include "tasks/mi300/mla_common_mi300.cuh"

namespace kernel {

template <typename T, int NH, int D_C, int D_R, int S_MAX>
__device__ __forceinline__ void
    mla_attend_mi300_task_impl(void const *ql_nope_ptr,
                               void const *q_pe_ptr,
                               void const *c_kv_ptr,
                               void const *k_pe_ptr,
                               void *partials_ptr,
                               int step,
                               float softmax_scale,
                               int split,
                               int n_splits,
                               int tiles_per_xcd,
                               int partials_xcd_offset_rows,
                               int tile_idx,
                               void *debug_scores_ptr) {
  using namespace dsv2;
  constexpr int TILE = 32;
  constexpr int XCDS = 8;
  constexpr int NCG = D_C / 8;                 // column groups of 8 (64)
  constexpr int HPT = NH / WAVES;              // heads per thread (4)
  static_assert(NCG == WAVE, "a wave must cover the D_C columns 8 per lane");
  static_assert(NH % WAVES == 0, "heads split over the 4 waves");
  static_assert(NH * TILE <= 2 * NUM_THREADS, "two scores per thread per pass");
  static_assert(D_C % 8 == 0 && D_R % 8 == 0, "16-byte loads");
  // partials row padded up to a multiple of 4 floats: each [split, head] row is then
  // 16-byte aligned when the buffer base is (docs/gpu-experiments/02-validation P2). o in [0, D_C), lse at D_C.
  constexpr int P_ROW = ((D_C + 1 + 3) / 4) * 4;
  // loads in flight per thread (docs/gpu-experiments/02-validation P6): the streaming knee measured on the MI300X
  // is at 4 (env/hw/20260915, E2); one load at a time made a 36 KB tile cost about 144
  // serialised HBM latencies, the 211 us of runs/L27_it32.
  constexpr int PF = 4;
  static_assert(D_C % (8 * PF) == 0 && D_R % (8 * PF) == 0, "the column loops batch PF loads of 8");

  int xcd = tile_idx / tiles_per_xcd;
  int t = tile_idx % tiles_per_xcd;
  int split_id = t * XCDS + xcd;
  if (split_id >= n_splits) {
    return;
  }
  float *partials = static_cast<float *>(partials_ptr)
      - (size_t)xcd * partials_xcd_offset_rows * NH * P_ROW
      + (size_t)split_id * NH * P_ROW;
  int tid = threadIdx.x;
  int lo = split_id * split;
  if (lo > step) {
    for (int e = tid; e < NH * P_ROW; e += NUM_THREADS) {
      partials[e] = (e % P_ROW == D_C) ? -INFINITY : 0.0f;
    }
    return;
  }
  int hi = lo + split;
  if (hi > step + 1) {
    hi = step + 1;
  }

  T const *ql_nope = static_cast<T const *>(ql_nope_ptr);
  T const *q_pe = static_cast<T const *>(q_pe_ptr);
  T const *c_kv = static_cast<T const *>(c_kv_ptr);
  T const *k_pe = static_cast<T const *>(k_pe_ptr);

  extern __shared__ char smem[];
  T *ql_s = reinterpret_cast<T *>(smem);                                   // [NH][D_C]
  T *qpe_s = ql_s + NH * D_C;                                              // [NH][D_R]
  float *s_s = reinterpret_cast<float *>(qpe_s + NH * D_R);                // [NH][TILE]
  float *p_s = s_s + NH * TILE;                                            // [NH][TILE]
  float *m_s = p_s + NH * TILE;                                            // [NH]
  float *l_s = m_s + NH;                                                   // [NH]
  float *alpha_s = l_s + NH;                                               // [NH]

  for (int e = tid; e < NH * D_C; e += NUM_THREADS) {
    ql_s[e] = ql_nope[e];
  }
  for (int e = tid; e < NH * D_R; e += NUM_THREADS) {
    qpe_s[e] = q_pe[e];
  }
  if (tid < NH) {
    m_s[tid] = -INFINITY;
    l_s[tid] = 0.0f;
  }
  __syncthreads();

  int wave = tid / WAVE, lane = tid % WAVE;
  int h0 = wave * HPT;                      // this thread's heads: h0 .. h0 + HPT
  int c0 = lane * 8;                        // this thread's columns: c0 .. c0 + 8
  float acc[HPT][8];
#pragma unroll
  for (int j = 0; j < HPT; j++) {
#pragma unroll
    for (int k = 0; k < 8; k++) {
      acc[j][k] = 0.0f;
    }
  }

  for (int r0 = lo; r0 < hi; r0 += TILE) {
    int rows = (hi - r0 < TILE) ? (hi - r0) : TILE;

    // scores [NH][rows]: two per thread, FP32 dot products over D_C and D_R.
    // Lane -> head, so the 16 lanes of one row share its 16-byte loads (one row per 16
    // lanes per instruction) instead of every lane streaming its own row; PF loads are
    // issued before their FMAs. Each (head, row) is computed exactly once as before.
    for (int e = tid; e < NH * TILE; e += NUM_THREADS) {
      int h = e % NH, p = e / NH;
      float s = -INFINITY;
      if (p < rows) {
        T const *ck = c_kv + (size_t)(r0 + p) * D_C;
        T const *kp = k_pe + (size_t)(r0 + p) * D_R;
        float dot = 0.0f;
        for (int c = 0; c < D_C; c += 8 * PF) {
          float v[PF][8];
#pragma unroll
          for (int u = 0; u < PF; u++) {
            load8(ck + c + 8 * u, v[u]);
          }
#pragma unroll
          for (int u = 0; u < PF; u++) {
#pragma unroll
            for (int k = 0; k < 8; k++) {
              dot += ld(ql_s + h * D_C + c + 8 * u + k) * v[u][k];
            }
          }
        }
        for (int r = 0; r < D_R; r += 8 * PF) {
          float v[PF][8];
#pragma unroll
          for (int u = 0; u < PF; u++) {
            load8(kp + r + 8 * u, v[u]);
          }
#pragma unroll
          for (int u = 0; u < PF; u++) {
#pragma unroll
            for (int k = 0; k < 8; k++) {
              dot += ld(qpe_s + h * D_R + r + 8 * u + k) * v[u][k];
            }
          }
        }
        s = dot * softmax_scale;
#ifdef MLA_ATTEND_DEBUG_SCORES
        if (debug_scores_ptr != nullptr) {
          static_cast<float *>(debug_scores_ptr)[h * S_MAX + r0 + p] = s;
        }
#endif
      }
      s_s[h * TILE + p] = s;
    }
    __syncthreads();

    // online-softmax update per head, one thread per head
    if (tid < NH) {
      int h = tid;
      float mx = -INFINITY;
      for (int p = 0; p < rows; p++) {
        mx = fmaxf(mx, s_s[h * TILE + p]);
      }
      float m_new = fmaxf(m_s[h], mx);
      float alpha = expf(m_s[h] - m_new);       // exp(-inf) = 0 on the first pass
      float psum = 0.0f;
      for (int p = 0; p < TILE; p++) {
        float pv = (p < rows) ? expf(s_s[h * TILE + p] - m_new) : 0.0f;
        psum += pv;                                // l sums the unrounded probabilities
        p_s[h * TILE + p] = bf16r(pv);             // the MFMA operand is BF16
      }
      l_s[h] = l_s[h] * alpha + psum;
      m_s[h] = m_new;
      alpha_s[h] = alpha;
    }
    __syncthreads();

    // acc = acc * alpha + p . c_kv[tile]
#pragma unroll
    for (int j = 0; j < HPT; j++) {
      float a = alpha_s[h0 + j];
#pragma unroll
      for (int k = 0; k < 8; k++) {
        acc[j][k] *= a;
      }
    }
    // rows in batches of PF: PF coalesced row loads in flight per thread; a short tail
    // (rows not a multiple of PF) is finished one row at a time. Same FP32 order per row.
    int p = 0;
    for (; p + PF <= rows; p += PF) {
      float v[PF][8];
#pragma unroll
      for (int u = 0; u < PF; u++) {
        load8(c_kv + (size_t)(r0 + p + u) * D_C + c0, v[u]);
      }
#pragma unroll
      for (int u = 0; u < PF; u++) {
#pragma unroll
        for (int j = 0; j < HPT; j++) {
          float pv = p_s[(h0 + j) * TILE + p + u];
#pragma unroll
          for (int k = 0; k < 8; k++) {
            acc[j][k] += pv * v[u][k];
          }
        }
      }
    }
    for (; p < rows; p++) {
      float v[8];
      load8(c_kv + (size_t)(r0 + p) * D_C + c0, v);
#pragma unroll
      for (int j = 0; j < HPT; j++) {
        float pv = p_s[(h0 + j) * TILE + p];
#pragma unroll
        for (int k = 0; k < 8; k++) {
          acc[j][k] += pv * v[k];
        }
      }
    }
    __syncthreads();
  }

  // o = acc / l, lse = m + ln(l)
#pragma unroll
  for (int j = 0; j < HPT; j++) {
    float inv_l = 1.0f / l_s[h0 + j];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      partials[(h0 + j) * P_ROW + c0 + k] = acc[j][k] * inv_l;
    }
  }
  if (tid < NH) {
    partials[tid * P_ROW + D_C] = m_s[tid] + logf(l_s[tid]);
  }
}

} // namespace kernel
