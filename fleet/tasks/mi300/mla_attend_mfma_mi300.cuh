/* mla_attend_mfma_mi300: the split-KV attention on the matrix cores (O7,
 * docs/gpu-experiments/03-acceleration/03-local-preparation.md; phase B of
 * docs/mla-decode/04-our-kernel-spec.md). Selected by -DMLA_ATTEND_MFMA
 * from mla_attend_mi300.cuh; the same template, signature, tile decode,
 * empty-split fill, partials layout and debug-scores output as the VALU
 * kernel, so the registration, the plan, the launcher and the merge are
 * unchanged. The VALU kernel costs 34 us per tile grid standalone against
 * a 9 us bandwidth floor (round 2); this one reads each cache row once,
 * stages it in LDS and does the two products on v_mfma_f32_16x16x16_bf16.
 *
 * Per pass of TILE = 16 rows (a split of 32 rows is two passes):
 *   - the tile [16][576] BF16 ([c_kv | k_pe] per row, 18 KB) is loaded by
 *     all 256 threads as 16-byte chunks (four or five per thread, all issued
 *     before use) into registers, the next pass's chunks issued before this
 *     pass computes, then written to LDS rows of stride RS = 584 elements
 *     (16-byte aligned; the pad breaks the 16-way bank conflict of a
 *     1,152-byte stride). Rows past the split are zero-filled.
 *   - scores S[h][p] = sum_k Q[h][k] T[p][k] over K = 576 as 36 MFMA steps
 *     of 16, nine per wave: A[h][k] and B[k][p] = T[p][k] are both four
 *     consecutive elements of a row (one ds_read_b64 each; the query is
 *     staged once per task as [16][584] the same way). The four waves'
 *     partial accumulators are summed through LDS by thread (h, p), scaled,
 *     masked (-inf past the split's rows) and written to the debug scores.
 *   - the online softmax by the 16 lanes of a head (xor shuffles of width
 *     16): m_new, alpha = exp(m_old - m_new), p = exp(s - m_new), l summed
 *     from the unrounded p, p rounded to BF16 into LDS as the next A operand;
 *     the reference's rounding points, as the VALU kernel.
 *   - p x V: O[h][c] += sum_p P[h][p] V[p][c] for the 512 columns as 32
 *     blocks of 16, eight per wave (accumulator 32 FP32 per lane); the A
 *     fragment is the head's four probabilities, the B fragment four rows of
 *     one column read from the staged tile (ds_read_u16 x 4); the
 *     accumulator is rescaled by alpha[h] before the step.
 *   - o = acc / l and lse = m + ln(l) at the end, the layout of the VALU kernel.
 * Operand layout: lane l holds A[l % 16][4 (l / 16) + i], B[4 (l / 16) + i][l % 16]
 * and D[4 (l / 16) + i][l % 16] (mla_common_mi300.cuh, mfma_16x16x16_bf16;
 * the index arithmetic is fleet/tests/test_mfma_layout.py's).
 *
 * LDS: q_s [16][584] BF16 (18,688 B), t_s [16][584] BF16 (18,688 B), the
 * score partials [4][16][16] FP32 (4,096 B), p_s [16][16] BF16 (512 B), m, l,
 * alpha [16] FP32 (192 B): 42,176 B of the 57 KiB.
 * Under -DMLA_NT_STREAMS (O6) the tile loads stream (each row is read once).
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
  constexpr int TILE = 16;                          // rows per pass: the MFMA N of the scores, K of p x V
  constexpr int XCDS = 8;
  constexpr int KD = D_C + D_R;                     // the staged row [c_kv | k_pe], 576
  constexpr int RS = KD + 8;                        // its LDS stride, 584: 16-byte aligned, banks spread
  constexpr int KSTEPS = KD / 16;                   // 36 MFMA steps of the scores
  constexpr int KSPW = KSTEPS / WAVES;              // 9 per wave
  constexpr int NCB = D_C / 16;                     // 32 column blocks of p x V
  constexpr int CBPW = NCB / WAVES;                 // 8 per wave
  constexpr int CPR = KD / 8;                       // 72 sixteen-byte chunks per row
  constexpr int CHUNKS = TILE * CPR;                // 1,152 per tile
  constexpr int CPT = (CHUNKS + NUM_THREADS - 1) / NUM_THREADS;   // 5 per thread (the last for half of them)
  constexpr int P_ROW = ((D_C + 1 + 3) / 4) * 4;    // the partials row (P2), o in [0, D_C), lse at D_C
  static_assert(NH == 16 && TILE == 16, "one MFMA block is 16 heads x 16 rows");
  static_assert(KD % 16 == 0 && KSTEPS % WAVES == 0, "the K steps split evenly over the four waves");
  static_assert(D_C % 16 == 0 && NCB % WAVES == 0, "the column blocks split evenly over the four waves");
  static_assert(D_C % 8 == 0 && D_R % 8 == 0 && RS % 8 == 0, "16-byte chunks and 16-byte aligned LDS rows");
  static_assert(NH * TILE == NUM_THREADS, "one thread per (head, row) in the softmax");
  constexpr int LDS_BYTES = (NH + TILE) * RS * (int)sizeof(T) + WAVES * NH * TILE * 4 + NH * TILE * (int)sizeof(T) + 3 * NH * 4;
  static_assert(LDS_BYTES <= 57 * 1024, "the worker's dynamic LDS");

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
  StreamSrc<T> c_kv_stream(c_kv);
  StreamSrc<T> k_pe_stream(k_pe);

  extern __shared__ char smem[];
  T *q_s = reinterpret_cast<T *>(smem);                                     // [NH][RS]   [ql_nope | q_pe]
  T *t_s = q_s + NH * RS;                                                   // [TILE][RS] [c_kv | k_pe]
  float *red_s = reinterpret_cast<float *>(t_s + TILE * RS);                // [WAVES][NH][TILE]
  T *p_s = reinterpret_cast<T *>(red_s + WAVES * NH * TILE);                // [NH][TILE] BF16 probabilities
  float *m_s = reinterpret_cast<float *>(p_s + NH * TILE);                  // [NH]
  float *l_s = m_s + NH;                                                    // [NH]
  float *alpha_s = l_s + NH;                                                // [NH]
  int wave = tid / WAVE, lane = tid % WAVE;

  // the query rows, once per task: chunk e of NH x CPR -> row e / CPR, chunk e % CPR
  for (int e = tid; e < NH * CPR; e += NUM_THREADS) {
    int h = e / CPR, c8 = e % CPR;
    T const *src = (c8 < D_C / 8) ? ql_nope + (size_t)h * D_C + c8 * 8 : q_pe + (size_t)h * D_R + (c8 - D_C / 8) * 8;
    *reinterpret_cast<uint4 *>(q_s + h * RS + c8 * 8) = *reinterpret_cast<uint4 const *>(src);
  }
  if (tid < NH) {
    m_s[tid] = -INFINITY;
    l_s[tid] = 0.0f;
  }

  // the tile's chunks: thread tid takes e = tid + 256 i (row e / CPR, chunk e % CPR); rows past
  // `rows` are zeros so a masked row never brings a stale value into the products
  auto issue = [&](int r0, int rows, uint4 *dst) {
#pragma unroll
    for (int i = 0; i < CPT; i++) {
      int e = tid + i * NUM_THREADS;
      if (e < CHUNKS) {
        int row = e / CPR, c8 = e % CPR;
        if (row < rows) {
          dst[i] = (c8 < D_C / 8)
              ? load16_from(c_kv_stream, (size_t)(r0 + row) * D_C + c8 * 8)
              : load16_from(k_pe_stream, (size_t)(r0 + row) * D_R + (c8 - D_C / 8) * 8);
        } else {
          dst[i].x = 0u; dst[i].y = 0u; dst[i].z = 0u; dst[i].w = 0u;
        }
      }
    }
  };
  auto stage = [&](uint4 const *src) {
#pragma unroll
    for (int i = 0; i < CPT; i++) {
      int e = tid + i * NUM_THREADS;
      if (e < CHUNKS) {
        *reinterpret_cast<uint4 *>(t_s + (e / CPR) * RS + (e % CPR) * 8) = src[i];
      }
    }
  };

  // the accumulator: O[h = 4 (lane / 16) + i][c = 16 cb + lane % 16] for this wave's blocks
  f32x4_t o[CBPW];
#pragma unroll
  for (int j = 0; j < CBPW; j++) {
    o[j] = f32x4_t{0.0f, 0.0f, 0.0f, 0.0f};
  }
  // the lane's fragment addresses (elements): A rows of q_s and B rows of t_s for the scores;
  // the four rows at one column for the values
  int frag_row = lane % 16, frag_k = 4 * (lane / 16);
  T const *qa = q_s + frag_row * RS + frag_k;
  T const *tb = t_s + frag_row * RS + frag_k;
  unsigned short const *vb = reinterpret_cast<unsigned short const *>(t_s) + frag_k * RS + frag_row;
  int sh = tid / TILE, sp = tid % TILE;             // the softmax thread's (head, row)

  uint4 buf[CPT];
  int rows0 = (hi - lo < TILE) ? (hi - lo) : TILE;
  issue(lo, rows0, buf);
  for (int r0 = lo; r0 < hi; r0 += TILE) {
    int rows = (hi - r0 < TILE) ? (hi - r0) : TILE;
    int r1 = r0 + TILE;
    uint4 nbuf[CPT];
    if (r1 < hi) {                                   // the next pass's loads in flight during this one
      issue(r1, (hi - r1 < TILE) ? (hi - r1) : TILE, nbuf);
    }
    __syncthreads();                                 // the previous pass's p x V is done with t_s
    stage(buf);
    __syncthreads();

    // scores: this wave's nine K steps into a 16 x 16 partial accumulator
    f32x4_t acc = f32x4_t{0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int s = 0; s < KSPW; s++) {
      int k0 = 16 * (wave * KSPW + s);
      bf16x4_t a = *reinterpret_cast<bf16x4_t const *>(qa + k0);
      bf16x4_t b = *reinterpret_cast<bf16x4_t const *>(tb + k0);
      acc = mfma_16x16x16_bf16(a, b, acc);
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      red_s[(wave * NH + frag_k + i) * TILE + frag_row] = acc[i];    // D[h = frag_k + i][p = frag_row]
    }
    __syncthreads();

    // thread (h, p): the score, then the online softmax across the head's 16 lanes
    float s = -INFINITY;
    if (sp < rows) {
      s = (red_s[(0 * NH + sh) * TILE + sp] + red_s[(1 * NH + sh) * TILE + sp] +
           red_s[(2 * NH + sh) * TILE + sp] + red_s[(3 * NH + sh) * TILE + sp]) * softmax_scale;
#ifdef MLA_ATTEND_DEBUG_SCORES
      if (debug_scores_ptr != nullptr) {
        static_cast<float *>(debug_scores_ptr)[sh * S_MAX + r0 + sp] = s;
      }
#endif
    }
    float mx = s;
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
      mx = fmaxf(mx, __shfl_xor(mx, off, 16));
    }
    float m_old = m_s[sh];
    float m_new = fmaxf(m_old, mx);
    float pv = (sp < rows) ? expf(s - m_new) : 0.0f;
    float psum = pv;
#pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
      psum += __shfl_xor(psum, off, 16);            // l sums the unrounded probabilities
    }
    st(p_s + sh * TILE + sp, bf16r(pv));            // the MFMA operand is BF16
    if (sp == 0) {
      float alpha = expf(m_old - m_new);            // exp(-inf) = 0 on the first pass
      l_s[sh] = l_s[sh] * alpha + psum;
      m_s[sh] = m_new;
      alpha_s[sh] = alpha;
    }
    __syncthreads();

    // p x V: acc = acc * alpha + P . V[tile] for this wave's eight column blocks
    bf16x4_t pa = *reinterpret_cast<bf16x4_t const *>(p_s + frag_row * TILE + frag_k);
    float al[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      al[i] = alpha_s[frag_k + i];
    }
#pragma unroll
    for (int j = 0; j < CBPW; j++) {
      int c0 = 16 * (wave * CBPW + j);
      bf16x4_t b;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        o[j][i] *= al[i];
        b[i] = (short)vb[i * RS + c0];              // V[frag_k + i][c0 + frag_row]
      }
      o[j] = mfma_16x16x16_bf16(pa, b, o[j]);
    }
    if (r1 < hi) {
#pragma unroll
      for (int i = 0; i < CPT; i++) {
        buf[i] = nbuf[i];
      }
    }
  }

  // o = acc / l, lse = m + ln(l)
  float inv_l[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    inv_l[i] = 1.0f / l_s[frag_k + i];
  }
#pragma unroll
  for (int j = 0; j < CBPW; j++) {
    int c = 16 * (wave * CBPW + j) + frag_row;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      partials[(frag_k + i) * P_ROW + c] = o[j][i] * inv_l[i];
    }
  }
  if (tid < NH) {
    partials[tid * P_ROW + D_C] = m_s[tid] + logf(l_s[tid]);
  }
}

} // namespace kernel
