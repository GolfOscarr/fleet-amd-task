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
 * Phases (round 4, M1 to M3 of docs/gpu-experiments/04-kernels/03-router-merge-ideas.md):
 * round 3's four dependent round trips become three (the partials batch, then
 * the W_uv batches), and every wave-load a contiguous KB.
 *   (a) wave w owns the W_uv rows D_V / WAVES * w .. of the head; one row of
 *       D_C BF16 is exactly one wave-load (lane l takes the elements 8 l .. 8 l
 *       + 7), loaded in (f) in batches of MERGE_W_BATCH rows held as raw words.
 *       M2, a batch issued here before the partials, was measured and dropped
 *       (the note at MERGE_W_BATCH below).
 *   (b) the partials batch: thread (q = tid % CHUNKS, s = tid / CHUNKS) reads,
 *       for its rows j = s, s + GROUPS, ... < live, the floats 4 q .. 4 q + 3 and
 *       D_C / 2 + 4 q .. of the head's row in split j (two 16-byte loads, each of
 *       them one contiguous KB across the wave) and, for q = 0, that row's lse at
 *       column D_C: round 3's separate lse round trip folded into this batch.
 *   (c) the q = 0 threads write lse_s, one barrier, and every thread computes M,
 *       the total over the live splits and its own rows' weights from lse_s.
 *   (d) the FMAs in ascending j into eight accumulators (the two runs of four
 *       columns), the GROUPS partial sums added in group order through red_s,
 *       o_s[c] = bf16r(sum * inv_tot), as round 3.
 *   (f) the lane's eight o values are read from o_s[8 lane ..] into registers once;
 *       then one W_uv batch at a time: its rows loaded, per row eight FMAs against
 *       the o values, the batch's row sums reduced by one
 *       halving butterfly (butterfly_sum<MERGE_W_BATCH>, which leaves row r's
 *       total in lane r), and the lanes below the batch store attn as BF16.
 *
 *
 * Two entry points share that body (N4, M6 and M4 of 03-router-merge-ideas.md):
 *   mla_merge_uv_mi300_task_impl  the gang task above, its tile decode unchanged.
 *   mla_merge_uv_tile_mi300_task_impl<..., HALVES>  one regular task per (head, half),
 *       every tensor whole and the index from expert_offset (h = idx / HALVES,
 *       half = idx % HALVES) as the attention's per-tile form takes its split. With
 *       HALVES = 2 the task merges the whole head and multiplies only the W_uv rows
 *       D_V / 2 * half .. + D_V / 2 - 1 (16 per wave, one batch), storing the matching
 *       64 columns of attn; the partials traffic doubles and the W_uv phase halves.
 *       The lane reduction is butterfly_sum<MERGE_W_BATCH> either way, so a row's
 *       sum is the same float in both forms: the two are bit-exact against each other.
 *
 * Inputs : partials [n_splits, NH, P_ROW] FP32 (P_ROW = D_C+1 padded to /4), W_uv [NH, D_V, D_C] BF16
 * Outputs: attn [1, NH * D_V] BF16
 * Pointer conventions (computed by the registration from the imaps; the gang form only,
 * the regular one takes every tensor whole):
 *   partials_xcd_offset_rows: rows already added for this XCD (0 if unpartitioned)
 *   w_uv_local: 1 if the W_uv pointer is this XCD's [heads_per_xcd, D_V, D_C] slice
 *   out_local : 1 if the attn pointer is this XCD's heads_per_xcd * D_V columns
 * LDS    : lse_s [64], o_s [D_C], the split groups' partial sums red_s [4][D_C] FP32:
 *          10.25 KiB. Round 3's weight row and its total in LDS are gone: every thread
 *          holds the total, so (c) needs no second barrier.
 * Registers: through (d) about 72 partials words + 9 lse + the eight accumulators,
 *          and in (f) 64 W_uv words + the eight o values + the batch's row sums:
 *          under 180 with the addressing (the offline build's k_mla_merge_uv line
 *          is the check: 166 VGPRs, no scratch).
 */
#pragma once
#include "tasks/common/common_header.cuh"
#include "tasks/mi300/mla_common_mi300.cuh"

#ifndef MERGE_W_BATCH
#define MERGE_W_BATCH 16     // W_uv rows per batch: two batches of 64 raw registers per wave
#endif
// M2 (the W_uv batches issued before and under the partials) was measured and dropped: a
// batch live across the partials phase costs about 60 to 75 registers with this compiler
// (the probe of 2026-09-18: 244 against 166 at a batch of 16, 205 at 8), which the worker
// union cannot take; the rows are loaded in (f), one batch at a time.

namespace kernel {

// The floats of dynamic LDS the body below uses (lse_s, o_s, red_s), so a caller that adds
// its own scratch after them reads the layout from one place (N5, mla_merge_oproj_mi300.cuh).
template <int D_C>
__device__ __host__ constexpr int mla_merge_uv_lds_floats() {
  return dsv2::WAVE + D_C + (NUM_THREADS / ((D_C / 2) / 4)) * D_C;   // lse_s, o_s, GROUPS x red_s
}

// The shared body: head h's merge, and the W_uv rows of the half `half` (HALVES = 1: the
// whole head). The three pointers are already offset to the head by the caller, so the two
// entry points below differ only in how they find h and the half.
//
// attn_s (N5): when it is not null, the half's attn values are also written to it as floats,
// at the index the half's own columns have (wave * W_ROWS_PER_WAVE + ...), so the caller can
// multiply them by its W_o slice without reading `attn` back. The same BF16-rounded float the
// store writes, so the two forms are the same number.
template <typename T, int NH, int D_V, int D_C, int HALVES>
__device__ __forceinline__ void
    mla_merge_uv_head(float const *partials,
                      T const *w_uv,
                      T *attn,
                      int step,
                      int split,
                      int n_splits,
                      int h,
                      int half,
                      float *attn_s = nullptr) {
  using namespace dsv2;
  static_assert(sizeof(T) == 2, "the raw-word conversion below is BF16's");
  static_assert(D_C == 8 * WAVE, "one row of W_uv is exactly one wave-load");
  constexpr int P_ROW = ((D_C + 1 + 3) / 4) * 4;   // padded partials row (P2); o in [0,D_C), lse at D_C

  // the W_uv map (M3): the wave's rows in batches of MERGE_W_BATCH, one halving
  // butterfly per batch, so the batch is a power of two at most a wave wide
  constexpr int W_BATCH = MERGE_W_BATCH;
  constexpr int W_ROWS_PER_HALF = D_V / HALVES;       // M4: this task's share of the head's rows
  constexpr int W_ROWS_PER_WAVE = W_ROWS_PER_HALF / WAVES;
  constexpr int W_BATCHES = W_ROWS_PER_WAVE / W_BATCH;
  static_assert(HALVES >= 1 && D_V % (WAVES * HALVES) == 0 && W_ROWS_PER_WAVE % W_BATCH == 0,
                "the wave's W_uv rows split into whole batches");
  static_assert(W_BATCH >= 1 && W_BATCH <= WAVE && (W_BATCH & (W_BATCH - 1)) == 0,
                "one halving butterfly reduces a batch");

  int live = (step + split) / split;                // ceil((step + 1) / split)
  if (live > n_splits) {
    live = n_splits;
  }
  StreamSrc<float> partials_stream(partials);         // O6: the partials are read once per iteration
  StreamSrc<T> w_uv_stream(w_uv);

  extern __shared__ char smem[];
  float *lse_s = reinterpret_cast<float *>(smem);    // [64]: the live splits' lse
  float *o_s = lse_s + WAVE;                         // [D_C]
  float *red_s = o_s + D_C;                          // [GROUPS][D_C]: the split groups' partial sums
  int tid = threadIdx.x;
  int lane = tid % WAVE;
  int wave = tid / WAVE;

  // the merge reads one lse per lane of a wavefront, so n_splits is at most 64
  // (asserted at registration and in build_graph.py)
  if (live > WAVE) {
    live = WAVE;   // unreachable when the asserts hold; never read past lse_s
  }

  // (a) the W_uv rows this wave multiplies in (f): w_row0 .. w_row0 + W_ROWS_PER_WAVE - 1
  int w_row0 = half * W_ROWS_PER_HALF + wave * W_ROWS_PER_WAVE;

  // o[c] = sum_j w_j o_j[c] / sum_j w_j, rounded to BF16. Thread t owns two runs of
  // four columns (q = t % CHUNKS: 4 q .. and D_C / 2 + 4 q .., one 16-byte load each,
  // so the 64 addresses of a load are one contiguous KB: K9) and the split rows
  // j = s, s + GROUPS, ... (s = t / CHUNKS): every row of the batch is loaded before
  // the first multiply and the FMAs run in ascending j.
  constexpr int RUN = 4;                              // FP32 values per 16-byte load
  constexpr int CHUNK = 2 * RUN;                      // columns per thread
  constexpr int CHUNKS = (D_C / 2) / RUN;             // threads per split row
  constexpr int GROUPS = NUM_THREADS / CHUNKS;        // the split groups, reduced through red_s
  // the splits the attention writes at the run's S_max: ceil(1056 / 32). A larger
  // n_splits (the registration allows 64) is still summed, in more than one batch.
  constexpr int N_SPLITS_MAX = 33;
  // N_SPLITS_MAX over GROUPS: 18 16-byte loads, 72 registers. A literal, not the
  // expression, so env/session/pf.sh keeps printing it (the asserts hold it to the map)
  constexpr int ROWS_IN_FLIGHT = 9;
  static_assert(ROWS_IN_FLIGHT * GROUPS >= N_SPLITS_MAX, "one batch covers the splits of a run");
  static_assert(ROWS_IN_FLIGHT * GROUPS <= WAVE, "a batch's rows are lse_s entries");
  static_assert(D_C % (2 * RUN) == 0 && NUM_THREADS % CHUNKS == 0, "the thread map covers the row");
  static_assert((P_ROW * 4) % 16 == 0, "the partials rows are 16-byte aligned");
  int q = tid % CHUNKS, s = tid / CHUNKS;

  // (b) the partials batch and, for q = 0, the lse word of each of its rows (M1)
  uint4 v[ROWS_IN_FLIGHT][2];
  float lse[ROWS_IN_FLIGHT];
#pragma unroll
  for (int u = 0; u < ROWS_IN_FLIGHT; u++) {
    int j = s + u * GROUPS;
    size_t base = ((size_t)j * NH + h) * P_ROW;
    if (j < live) {
      float const *p = partials + base;
      v[u][0] = *reinterpret_cast<uint4 const *>(p + q * RUN);
      v[u][1] = *reinterpret_cast<uint4 const *>(p + D_C / 2 + q * RUN);
    }
    lse[u] = (q == 0 && j < live) ? ldf_from(partials_stream, base + D_C) : -INFINITY;
  }

  // (c) the lse row in LDS, one barrier, then M, the total and the weights per thread
  if (q == 0) {
#pragma unroll
    for (int u = 0; u < ROWS_IN_FLIGHT; u++) {
      lse_s[s + u * GROUPS] = lse[u];
    }
    // the splits past one batch (none at N_SPLITS_MAX): read straight into LDS
    for (int j = s + ROWS_IN_FLIGHT * GROUPS; j < live; j += GROUPS) {
      lse_s[j] = ldf_from(partials_stream, ((size_t)j * NH + h) * P_ROW + D_C);
    }
  }
  __syncthreads();
  float M = -INFINITY;
  for (int j = 0; j < live; j++) {
    M = fmaxf(M, lse_s[j]);
  }
  // the same weights round 3 computed, one per lane of a wavefront, summed here in
  // ascending j instead of by the wave's tree: the divisor can differ in its last bit
  float tot = 0.0f;
  for (int j = 0; j < live; j++) {
    tot += expf(lse_s[j] - M);
  }
  float inv_tot = 1.0f / tot;

  // (d) the FMAs, in ascending j within the group; the batch of (b) is the first
  float acc[CHUNK];
#pragma unroll
  for (int k = 0; k < CHUNK; k++) {
    acc[k] = 0.0f;
  }
  for (int j0 = s; j0 < live; j0 += GROUPS * ROWS_IN_FLIGHT) {
    if (j0 != s) {                                    // the further batches, if any
#pragma unroll
      for (int u = 0; u < ROWS_IN_FLIGHT; u++) {
        int j = j0 + u * GROUPS;
        if (j < live) {
          float const *p = partials + ((size_t)j * NH + h) * P_ROW;
          v[u][0] = *reinterpret_cast<uint4 const *>(p + q * RUN);
          v[u][1] = *reinterpret_cast<uint4 const *>(p + D_C / 2 + q * RUN);
        }
      }
    }
#pragma unroll
    for (int u = 0; u < ROWS_IN_FLIGHT; u++) {
      int j = j0 + u * GROUPS;
      if (j < live) {
        float wj = expf(lse_s[j] - M);                // the float round 3 kept in w_s[j]
        acc[0] += wj * __uint_as_float(v[u][0].x); acc[1] += wj * __uint_as_float(v[u][0].y);
        acc[2] += wj * __uint_as_float(v[u][0].z); acc[3] += wj * __uint_as_float(v[u][0].w);
        acc[4] += wj * __uint_as_float(v[u][1].x); acc[5] += wj * __uint_as_float(v[u][1].y);
        acc[6] += wj * __uint_as_float(v[u][1].z); acc[7] += wj * __uint_as_float(v[u][1].w);
      }
    }
  }

  // the groups' partial sums added in group order, then o rounded to BF16
#pragma unroll
  for (int k = 0; k < RUN; k++) {
    red_s[s * D_C + q * RUN + k] = acc[k];
    red_s[s * D_C + D_C / 2 + q * RUN + k] = acc[RUN + k];
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

  // (f) attn[v] = o . W_uv[h, v, :]: the lane's eight o values are the same for every
  // row it touches, so they are read once; a batch's 64 lane sums per row are reduced
  // by one halving butterfly (M3), which leaves row r of the batch in lane r
  float ov[8];
#pragma unroll
  for (int k = 0; k < 8; k++) {
    ov[k] = o_s[8 * lane + k];
  }
  // one batch of rows loaded, multiplied, reduced and stored at a time
#pragma unroll 1
  for (int b = 0; b < W_BATCHES; b++) {
    uint4 w_raw[W_BATCH];
#pragma unroll
    for (int r = 0; r < W_BATCH; r++) {
      w_raw[r] = load16_from(w_uv_stream, (size_t)(w_row0 + b * W_BATCH + r) * D_C + 8 * lane);
    }
    float rs[W_BATCH];
#pragma unroll
    for (int r = 0; r < W_BATCH; r++) {
      unsigned const *words = reinterpret_cast<unsigned const *>(&w_raw[r]);
      float a = 0.0f;
#pragma unroll
      for (int k = 0; k < 8; k++) {
        unsigned bits = (k & 1) ? (words[k / 2] & 0xffff0000u) : (words[k / 2] << 16);
        a += ov[k] * __uint_as_float(bits);
      }
      rs[r] = a;
    }
    float sum = butterfly_sum<W_BATCH>(rs);
    if (lane < W_BATCH) {
      float v = bf16r(sum);
      st(attn + w_row0 + b * W_BATCH + lane, v);
      if (attn_s != nullptr) {                        // N5: the half's values, at the half's own index
        attn_s[wave * W_ROWS_PER_WAVE + b * W_BATCH + lane] = v;
      }
    }
  }
}

// The gang entry point (task type TASK_MLA_MERGE_UV_MI300): the tile decode of the header,
// the three pointers resolved from the imap flags, the whole head in one task.
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
  constexpr int P_ROW = ((D_C + 1 + 3) / 4) * 4;
  int xcd = tile_idx / tiles_per_xcd;
  int t = tile_idx % tiles_per_xcd;
  if (t >= heads_per_xcd) {
    return;
  }
  int h = xcd * heads_per_xcd + t;
  float const *partials = static_cast<float const *>(partials_ptr)
      - (size_t)xcd * partials_xcd_offset_rows * NH * P_ROW;
  T const *w_uv = static_cast<T const *>(w_uv_ptr)
      + (size_t)(w_uv_local ? t : h) * D_V * D_C;
  T *attn = static_cast<T *>(attn_ptr) + (size_t)(out_local ? t : h) * D_V;
  mla_merge_uv_head<T, NH, D_V, D_C, 1>(partials, w_uv, attn, step, split, n_splits, h, 0);
}

// The regular entry point (N4, task type TASK_MLA_MERGE_UV_TILE_MI300): NH * HALVES tasks
// with every tensor whole, the task index from expert_offset as the attention's per-tile
// form takes its split and prep its head; h = idx / HALVES, half = idx % HALVES.
template <typename T, int NH, int D_V, int D_C, int HALVES>
__device__ __forceinline__ void
    mla_merge_uv_tile_mi300_task_impl(void const *partials_ptr,
                                      void const *w_uv_ptr,
                                      void *attn_ptr,
                                      int step,
                                      int split,
                                      int n_splits,
                                      int idx) {
  static_assert(HALVES == 1 || HALVES == 2, "a merge task is a whole head or a half of one");
  int h = idx / HALVES;
  int half = idx % HALVES;
  mla_merge_uv_head<T, NH, D_V, D_C, HALVES>(
      static_cast<float const *>(partials_ptr),
      static_cast<T const *>(w_uv_ptr) + (size_t)h * D_V * D_C,
      static_cast<T *>(attn_ptr) + (size_t)h * D_V,
      step, split, n_splits, h, half);
}

} // namespace kernel
