/* linear_gemv_mi300: one kernel for every dense linear at batch 1
 * (docs/gpu-experiments/04-kernels/01-gemv-ideas.md, K1 and K5 to K9;
 * 05-local-preparation.md, L1). A task owns `rows` rows of a [N, K] BF16
 * weight and multiplies each of them by the one [K] input row with FP32
 * accumulation and a BF16 store, so it replaces the CK small tile whose K
 * loop keeps a single step of loads in flight (linear_norm_mi300.cuh, the
 * kernel this one is written to displace). The input norm and the residual
 * add are template flags, as the CK path's prologue and epilogue are
 * separate kernels and tensors today.
 *
 * Inputs : x [1, K] BF16 (the whole row), w_norm [K] BF16 (NORM only),
 *          W [N, K] BF16 (this task's `rows` rows: the runtime hands the
 *          task a pointer already offset by them), residual [1, N] BF16
 *          (RESIDUAL only; the task's columns).
 * Outputs: out [1, N] BF16 (the task's columns; `o_stride` is the full N).
 * LDS    : the norm's 4-float reduction buffer, then, for NORM, the
 *          normalised row (K BF16, 4 KiB at K = 2048) at the start of the
 *          dynamic LDS. Nothing else; the row of the CK path's [grid, K]
 *          scratch tensor, its wait and its release fence are all gone.
 *
 * The thread map. Four waves of 64; wave w owns rows [w * RPW, min((w + 1) *
 * RPW, rows)) with RPW = ceil(rows / 4), so a 38-row task is 10, 10, 10 and 8
 * rows and the head's 256-row task is 64 rows per wave. A wave walks its rows
 * in batches of GEMV_BATCH under "#pragma unroll 1": without the pragma the
 * compiler unrolls the row loop and hoists the loads across the batches, and
 * the constant then means nothing (the probe of K6, env/offline_gfx942/
 * gemv_probe/results.txt: 8 rows unrolled is 30 loads in flight and 175
 * VGPRs, 8 rows with the pragma is 32 and 158). A row is four 16-byte loads
 * per lane at the element chunks 8 * lane + 512 * i, so one wave-load is one
 * contiguous kilobyte of the row: 8 full 128-byte lines requested, nothing
 * re-fetched by the next three loads (K9). -DGEMV_STRIDED puts the lane's 32
 * elements back where the round-3 router puts them (32 * lane + 8 * i, 32
 * lines touched per wave-load) for the A/B of M5; the products are the same
 * either way, only the summation order inside a lane changes.
 *
 * The x slice is the same 32 elements per lane, in 32 FP32 registers, read
 * once: from the LDS row for NORM (rmsnorm_row with out = nullptr writes only
 * the LDS copy) and from x with 16-byte loads otherwise. The weight goes
 * through StreamSrc, the sc1 nt policy of the linears under -DMLA_NT_STREAMS
 * (round 2, E2: 20% on the CK weight loads); x, the residual and the output
 * are plain.
 *
 * What is in flight across the prologue: the first batch's raw words
 * (GEMV_BATCH x 4 x 4 VGPRs) and, for RESIDUAL, one 2-byte residual value per
 * lane, both issued before the norm runs, so the norm's own loads, its block
 * reduction and its LDS round trip happen under the batch's latency (K8). The
 * live registers of a batch are those raw words, the 32 x-values and
 * GEMV_BATCH row sums; the norm's own need (8 x-values and 8 weight values)
 * is small beside them.
 *
 * The numerics. Every product is a BF16 times a BF16, exact in FP32; the
 * accumulation order is a lane's chain of 32 products (ascending k inside
 * each of its four runs of eight, the runs in ascending order) and then the
 * halving butterfly's tree over the 64 lanes (butterfly_sum<GEMV_BATCH>,
 * mla_common_mi300.cuh: the batch's rows are reduced together, 10 shuffles
 * for eight rows against 48 for eight wave sums, and lane l < GEMV_BATCH ends
 * holding row r0 + l). That order differs from the CK tile's MFMA order, as
 * the two differ from NumPy's; the difference is the FP32 reassociation the
 * boundary compare already tolerates on this path. The residual is added in
 * FP32 before the single BF16 rounding, as the stock epilogue adds it.
 *
 * Rows past the wave's range are loaded as the range's last row (a clamped
 * index, so the batch stays one basic block and its loads in flight together)
 * and their sums are dropped; only the stores are masked,
 * and they are 2-byte stores: a 38-row task's columns start 76 bytes into the
 * output row, so neither out nor residual is 16-byte aligned and no vector
 * access may be made on either (the stock epilogue's packed 8-byte store has
 * the same problem and falls back to scalar stores for it).
 */
#pragma once
#include "tasks/common/common_header.cuh"
#include "tasks/mi300/mla_common_mi300.cuh"

// The batch depth, a macro so the suite and the offline compile can sweep it
// (-DGEMV_BATCH=4|8|16, M5): 4, 8 and 16 rows are 64, 128 and 256 KB in flight
// per CU and 110, 158 and 256 VGPRs in the probe.
#ifndef GEMV_BATCH
#define GEMV_BATCH 8
#endif
// The first batch issued before the prologue (K8 of 01-gemv-ideas: the norm's round trip
// under the batch's latency). Off by default: with this compiler a batch that is live
// across the prologue and reloaded inside the loop is not coalesced with the loop's own,
// and the kernel costs about 60 to 75 more registers (the probe of 2026-09-18: 230
// against 172 at eight rows), which the worker union cannot take. -DGEMV_PRELOAD=1 for
// the VM's A/B.
#ifndef GEMV_PRELOAD
#define GEMV_PRELOAD 0
#endif

namespace kernel {

namespace gemv_detail {

// The element offset of the lane's chunk i of a row: the coalesced map of K9
// (one contiguous KB per wave-load) or, under -DGEMV_STRIDED, the router's
// (32 consecutive elements per lane).
template <int K>
__device__ __forceinline__ int chunk_elem(int lane, int i) {
#ifdef GEMV_STRIDED
  return (K / dsv2::WAVE) * lane + 8 * i;
#else
  return 8 * lane + 8 * dsv2::WAVE * i;
#endif
}

// The batch's weight loads, all issued before any of them is waited on. A row
// past the wave's range loads the range's last row again (a clamped index, no
// branch): a branch per row would put each row's loads in its own basic block,
// and the scheduler keeps only that block's loads in flight (the offline
// disassembly of the guarded form waited vmcnt(3) (2) (1) (0) per row, four
// loads in flight of the batch's 32). The clamped rows' sums are dropped by
// the store's mask.
template <typename T, int K, int BATCH, int CHUNKS>
__device__ __forceinline__ void load_batch(dsv2::StreamSrc<T> const &w_src, int r0, int r_end,
                                           int lane, uint4 (&raw)[BATCH][CHUNKS]) {
#pragma unroll
  for (int u = 0; u < BATCH; u++) {
    int r = (r0 + u < r_end) ? r0 + u : r_end - 1;
#pragma unroll
    for (int i = 0; i < CHUNKS; i++) {
      raw[u][i] = dsv2::load16_from(w_src, (size_t)r * K + chunk_elem<K>(lane, i));
    }
  }
}

} // namespace gemv_detail

// The body as a call rather than inlined into the worker's switch, the stock gang kernels'
// form: the worker union with the three inlined forms measured 256 VGPRs and 222 AGPRs
// offline (2026-09-18; 137 with one form), against 100 without them and 102 with the call,
// whose price is the callee-saved stores per task (67 dwords per lane, 332 bytes of scratch).
// -DGEMV_NOINLINE=0 inlines the body for the VM's A/B.
#ifndef GEMV_NOINLINE
#define GEMV_NOINLINE 1
#endif
#if GEMV_NOINLINE
#define GEMV_INLINE __noinline__
#else
#define GEMV_INLINE __forceinline__
#endif
template <typename T, int K, bool NORM, bool RESIDUAL>
__device__ GEMV_INLINE void
    linear_gemv_mi300_task_impl(void const *x_ptr,
                                void const *w_norm_ptr,
                                void const *weight_ptr,
                                void const *residual_ptr,
                                void *output_ptr,
                                int rows,
                                int o_stride,
                                float eps) {
  using namespace dsv2;
  constexpr int BATCH = GEMV_BATCH;
  static_assert(K % (8 * WAVE) == 0, "16-byte loads, K / 64 elements per lane");
  static_assert(K <= 4096, "a lane's K slice lives in registers (K / 64 values); layer 0's dense down (K 11,264) is not this kernel's");
  static_assert(BATCH >= 1 && BATCH <= WAVE && (BATCH & (BATCH - 1)) == 0,
                "the butterfly reduces a power-of-two batch of rows");
  constexpr int CHUNKS = K / (8 * WAVE);      // 4 sixteen-byte loads per row per lane at K = 2048

  T const *x = static_cast<T const *>(x_ptr);
  T const *residual = static_cast<T const *>(residual_ptr);
  T *out = static_cast<T *>(output_ptr);
  (void)o_stride;   // one token: the output row's stride never enters an address

  extern __shared__ char smem[];
  // LDS: [red 4 FP32] [the normalised row, K BF16 (NORM only)]
  float *red = reinterpret_cast<float *>(smem);
  T *x_s = reinterpret_cast<T *>(smem + 4 * sizeof(float));

  int wave = threadIdx.x / WAVE, lane = threadIdx.x % WAVE;
  int rpw = (rows + WAVES - 1) / WAVES;
  int r_begin = wave * rpw;
  int r_end = (wave + 1) * rpw < rows ? (wave + 1) * rpw : rows;
  if (r_end < r_begin) {
    r_end = r_begin;                          // a wave with no rows (rows < 4) still runs the prologue
  }

  StreamSrc<T> w_src(static_cast<T const *>(weight_ptr));

  // the first batch, and its residual values, before the prologue (K8): the norm
  // then runs under their latency
  uint4 raw[BATCH][CHUNKS];
  float res = 0.0f;
#if GEMV_PRELOAD
  if (r_begin < r_end) {                      // a wave with no rows (rows < 4) issues nothing
    gemv_detail::load_batch<T, K>(w_src, r_begin, r_end, lane, raw);
  }
  if constexpr (RESIDUAL) {
    if (lane < BATCH && r_begin + lane < r_end) {
      res = ld(residual + r_begin + lane);    // 2 bytes per lane: the output row is not 16-byte aligned
    }
  }
#endif

  if constexpr (NORM) {
    // the reference's order (numpy_ref.rmsnorm), the LDS copy only: no scratch row,
    // no global round trip, so no wait and no release fence before the multiply
    rmsnorm_row<T, K>(x, static_cast<T const *>(w_norm_ptr), nullptr, eps, red, x_s);
    __syncthreads();
    x = x_s;
  } else {
    (void)w_norm_ptr;
    (void)eps;
    (void)red;
    (void)x_s;
  }

  float xv[CHUNKS][8];
#pragma unroll
  for (int i = 0; i < CHUNKS; i++) {
    load8(x + gemv_detail::chunk_elem<K>(lane, i), xv[i]);
  }

#pragma unroll 1
  for (int r0 = r_begin; r0 < r_end; r0 += BATCH) {
#if GEMV_PRELOAD
    if (r0 != r_begin)                        // the first batch is already in flight
#endif
    {
      gemv_detail::load_batch<T, K>(w_src, r0, r_end, lane, raw);
      if constexpr (RESIDUAL) {
        res = (lane < BATCH && r0 + lane < r_end) ? ld(residual + r0 + lane) : 0.0f;
      }
    }
    // every row of the batch is multiplied, the clamped ones included (one basic
    // block, so the loads stay in flight together); their sums never reach a store
    float sums[BATCH];
#pragma unroll
    for (int u = 0; u < BATCH; u++) {
      float acc = 0.0f;
#pragma unroll
      for (int i = 0; i < CHUNKS; i++) {
        unsigned const *words = reinterpret_cast<unsigned const *>(&raw[u][i]);
#pragma unroll
        for (int k = 0; k < 8; k++) {
          unsigned bits = (k & 1) ? (words[k / 2] & 0xffff0000u) : (words[k / 2] << 16);
          acc += xv[i][k] * __uint_as_float(bits);
        }
      }
      sums[u] = acc;
    }
    butterfly_sum<BATCH>(sums);               // lane l < BATCH now holds row r0 + l's total
    int r = r0 + lane;
    if (lane < BATCH && r < r_end) {
      float v = sums[0];
      if constexpr (RESIDUAL) {
        v += res;                             // the residual in FP32, before the one rounding
      }
      st(out + r, bf16r(v));
    }
  }
}

} // namespace kernel
