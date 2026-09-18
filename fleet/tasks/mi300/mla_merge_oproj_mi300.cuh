/* mla_merge_oproj_mi300: the merge with o_proj folded in (N5 of
 * docs/gpu-experiments/04-kernels/05-local-preparation.md, M5 of
 * 03-router-merge-ideas.md). One regular task per (head, half), NH * HALVES of
 * them; the o_proj operator is gone, and the task that arrives last sums the
 * partial vectors into x_res.
 *
 * The contract. Inputs partials [n_splits, NH, P_ROW] FP32, W_uv [NH, D_V, D_C]
 * BF16, W_o [HIDDEN, HIDDEN] BF16 ([N, K] row-major, as the dense linears),
 * x_res [1, HIDDEN] BF16 and counter [1] int32, all whole; outputs x_res (in
 * place, as the stock residual linear names its residual and its output),
 * attn [1, NH * D_V] BF16 and workspace [NH * HALVES, HIDDEN] FP32, all whole.
 * The task index reaches the kernel through expert_offset (bid.x), as it does
 * for mla_merge_uv_tile and mla_prep: h = idx / HALVES, half = idx % HALVES.
 * Math and rounding: harness/numpy_ref.py mla_merge_oproj (the merge of
 * mla_merge_uv, then bf16(x_res + attn @ W_o^T)).
 *
 * The phases. 1 to 3 are the merge exactly as the regular tile form runs it
 * (mla_merge_uv_head, HALVES = 2): the task's 64 attn values are BF16-rounded,
 * stored to `attn` (the boundary keeps its row) and kept in LDS as floats.
 * Then:
 *
 *   4. the W_o slice. The task's columns of W_o are c0 = h * D_V + half * COLS
 *      .. c0 + COLS - 1 (COLS = D_V / HALVES = 64: 128 contiguous bytes of every
 *      row, 128-byte aligned, since c0 * 2 is a multiple of 128). Eight lanes
 *      cover a row (lane l the 16-byte chunk l % 8 of the segment) and eight
 *      rows a wave-load, so a wave-load is eight full 128-byte lines: lane l
 *      reads row 8 g + l / 8 of the wave-load's row group g. Wave w owns rows
 *      w * (HIDDEN / WAVES) .. + HIDDEN / WAVES - 1 (512 of the 2,048) and walks
 *      them in batches of OPROJ_BATCH wave-loads under "#pragma unroll 1" (16 =
 *      32 rows and 4 loads per lane; 8 = 64 rows and 8 loads), with no pre-load:
 *      the first batch is issued inside the loop, after the merge (the third
 *      convention of the page; a batch live across a prologue costs about 70
 *      registers with this compiler). A lane's product is its eight attn values,
 *      read once from LDS at the same chunk l % 8, against its chunk, in FMAs on
 *      raw words in ascending k; the eight lanes of a row are reduced by three
 *      xor steps (1, 2, 4), and the lane at the row's head (l % 8 == 0) writes
 *      workspace[idx][n] as FP32, one 4-byte store per lane with eight rows per
 *      wave-load, so eight stores per instruction.
 *   5. the arrival counter and the last task (the pattern of moe_router's
 *      SPLIT = 4, R5, itself the fork's splitk_linear_res_atomic_kernel with an
 *      acquire fence added): a barrier, an agent-scope release fence in every
 *      thread, thread 0's acq-rel add on the counter and the broadcast of
 *      old == NH * HALVES - 1 through LDS, a second barrier; the tasks that were
 *      not last return. The last one runs an agent-scope acquire fence in every
 *      thread, and thread t owns columns 8 t .. 8 t + 7 of the output: it reads
 *      workspace[k][8 t ..] for k = 0 .. NH * HALVES - 1 (two 16-byte loads per
 *      k, 64 loads in two batches of OPROJ_WS_BATCH under "#pragma unroll 1"),
 *      sums them in ascending k into eight FP32 accumulators, adds the eight
 *      x_res values in FP32 (one 16-byte load; the residual before the one
 *      rounding, as the stock epilogue adds it), rounds with bf16r and stores
 *      the eight BF16 values back to x_res in place (one 16-byte store). Thread
 *      0 then puts the counter back to zero for the next layer's tasks: the
 *      chain serialises the layers, so one counter tensor serves them all.
 *
 * LDS: the merge body's (lse_s, o_s, red_s: 10.25 KiB at D_C = 512) plus 64
 * floats of attn values and the one-word broadcast, 10.5 KiB in all.
 *
 * Registers: the merge is about 155 through its own phases (the header of
 * mla_merge_uv_mi300.cuh); phase 4 holds OPROJ_BATCH x 4 raw words (64 at the
 * batch of 16; 4, 8 and 32 measured offline at the same 248 registers, 32 with
 * 144 more bytes of scratch), the eight attn values and OPROJ_BATCH row sums; the last
 * task holds OPROJ_WS_BATCH x 2 x 4 = 128 raw words in flight and eight
 * accumulators. The three phases are in sequence, so the peak is the merge's;
 * the offline build's k_mla_merge_oproj line is the check (at most 200).
 *
 * Numerics: each output is 32 partial sums of 64 terms each (a lane's chain of
 * eight products and the row's three-step tree over its eight lanes), summed in
 * ascending task order, the residual added in FP32 and the result rounded once:
 * deterministic (no atomics on the data, only on the counter), a different FP32
 * order from the CK MFMA's o_proj and the same class as the GEMV linear's.
 */
#pragma once
#include "tasks/common/common_header.cuh"
#include "tasks/mi300/mla_common_mi300.cuh"
#include "tasks/mi300/mla_merge_uv_mi300.cuh"

// The wave-loads of W_o a wave keeps in flight (4 = 32 rows, 4 loads per lane; 8 = 64 rows,
// 8 loads), swept by the ktime A/B as -DOPROJ_BATCH=8. The default 16 (16 loads per lane, 64 KB
// per CU in flight, four round trips for the 256 KB slice; the review of 2026-09-18 found the
// first draft's 4 would have made sixteen). The constant only holds under
// "#pragma unroll 1", the first convention of the page.
#ifndef OPROJ_BATCH
#define OPROJ_BATCH 16
#endif
// The workspace rows the last task keeps in flight (16 rows = 32 16-byte loads = 128 raw words).
#ifndef OPROJ_WS_BATCH
#define OPROJ_WS_BATCH 16
#endif
// The body as a call rather than inlined into the worker's switch (the third convention, the
// stock gang kernels' form): -DMERGE_OPROJ_NOINLINE=0 inlines it for the VM's A/B.
#ifndef MERGE_OPROJ_NOINLINE
#define MERGE_OPROJ_NOINLINE 1
#endif
#if MERGE_OPROJ_NOINLINE
#define MERGE_OPROJ_INLINE __noinline__
#else
#define MERGE_OPROJ_INLINE __forceinline__
#endif

namespace kernel {

template <typename T, int NH = 16, int D_V = 128, int D_C = 512, int HIDDEN = 2048,
          int HALVES = 2>
__device__ MERGE_OPROJ_INLINE void
    mla_merge_oproj_mi300_task_impl(void const *partials_ptr,
                                    void const *w_uv_ptr,
                                    void const *w_o_ptr,
                                    void *x_res_ptr,
                                    void *counter_ptr,
                                    void *attn_ptr,
                                    void *workspace_ptr,
                                    int step,
                                    int split,
                                    int n_splits,
                                    int idx) {
  using namespace dsv2;
  // the task's arguments, identical across the wave, made scalar (uniform_ptr and
  // uniform_int of mla_common_mi300.cuh: a __noinline__ kernel's arguments arrive in
  // VGPRs, and a buffer resource built from a VGPR pointer costs every W_o load a
  // v_readfirstlane waterfall loop)
  partials_ptr = uniform_ptr(partials_ptr);
  w_uv_ptr = uniform_ptr(w_uv_ptr);
  w_o_ptr = uniform_ptr(w_o_ptr);
  x_res_ptr = uniform_ptr(x_res_ptr);
  counter_ptr = uniform_ptr(counter_ptr);
  attn_ptr = uniform_ptr(attn_ptr);
  workspace_ptr = uniform_ptr(workspace_ptr);
  step = uniform_int(step);
  split = uniform_int(split);
  n_splits = uniform_int(n_splits);
  idx = uniform_int(idx);
  static_assert(sizeof(T) == 2, "the raw-word conversion below is BF16's");
  static_assert(HALVES == 1 || HALVES == 2, "a task is a whole head or a half of one");
  static_assert(HIDDEN % (WAVES * 8) == 0, "the four waves take whole groups of eight rows");
  constexpr int TASKS = NH * HALVES;                   // the operator's tasks, and the workspace rows
  constexpr int COLS = D_V / HALVES;                   // this task's columns of W_o (64)
  constexpr int LANES_PER_ROW = COLS / 8;              // eight 16-byte chunks per row segment
  constexpr int ROWS_PER_LOAD = WAVE / LANES_PER_ROW;  // eight rows per wave-load
  constexpr int ROWS_PER_WAVE = HIDDEN / WAVES;        // 512
  constexpr int LOADS_PER_WAVE = ROWS_PER_WAVE / ROWS_PER_LOAD;
  static_assert(LANES_PER_ROW >= 1 && LANES_PER_ROW <= WAVE &&
                    (LANES_PER_ROW & (LANES_PER_ROW - 1)) == 0,
                "the row's lanes reduce by an xor tree");
  static_assert(COLS % 8 == 0 && ROWS_PER_WAVE % ROWS_PER_LOAD == 0,
                "the segment is whole 16-byte chunks and the wave's rows whole wave-loads");
  static_assert(LOADS_PER_WAVE % OPROJ_BATCH == 0, "the wave's wave-loads split into whole batches");
  static_assert(HIDDEN % (8 * NUM_THREADS) == 0, "the last task's thread owns eight columns");
  static_assert(TASKS % OPROJ_WS_BATCH == 0, "the workspace rows split into whole batches");

  int h = idx / HALVES;
  int half = idx % HALVES;
  int tid = threadIdx.x;
  int wave = tid / WAVE, lane = tid % WAVE;

  extern __shared__ char smem[];
  // LDS: the merge body's lse_s, o_s and red_s, then this task's attn values and the broadcast
  float *attn_s = reinterpret_cast<float *>(smem) + mla_merge_uv_lds_floats<D_C>();   // [COLS]
  int *last_s = reinterpret_cast<int *>(attn_s + COLS);                               // [1]

  // (1 to 3) the merge, exactly as the regular tile form runs it; the values go both to attn
  // (the boundary keeps its row) and to attn_s
  mla_merge_uv_head<T, NH, D_V, D_C, HALVES>(
      static_cast<float const *>(partials_ptr),
      static_cast<T const *>(w_uv_ptr) + (size_t)h * D_V * D_C,
      static_cast<T *>(attn_ptr) + (size_t)h * D_V,
      step, split, n_splits, h, half, attn_s);
  __syncthreads();                                     // each wave's share of them is read by all four

  // (4) the W_o slice: one partial vector [HIDDEN] FP32 into the task's workspace row
  int chunk = lane % LANES_PER_ROW;                    // the lane's 16-byte chunk of the segment
  int row_in_load = lane / LANES_PER_ROW;              // its row within the wave-load's eight
  float av[8];
#pragma unroll
  for (int k = 0; k < 8; k++) {
    av[k] = attn_s[8 * chunk + k];
  }
  size_t col0 = (size_t)h * D_V + (size_t)half * COLS + 8 * chunk;
  int row_base = wave * ROWS_PER_WAVE;
  StreamSrc<T> w_o_src(static_cast<T const *>(w_o_ptr));   // sc1 nt under MLA_NT_STREAMS, as the linears
  float *ws = static_cast<float *>(workspace_ptr) + (size_t)idx * HIDDEN;

#pragma unroll 1
  for (int g0 = 0; g0 < LOADS_PER_WAVE; g0 += OPROJ_BATCH) {
    uint4 raw[OPROJ_BATCH];
#pragma unroll
    for (int u = 0; u < OPROJ_BATCH; u++) {
      int n = row_base + (g0 + u) * ROWS_PER_LOAD + row_in_load;
      raw[u] = load16_from(w_o_src, (size_t)n * HIDDEN + col0);
    }
    float rs[OPROJ_BATCH];
#pragma unroll
    for (int u = 0; u < OPROJ_BATCH; u++) {
      unsigned const *words = reinterpret_cast<unsigned const *>(&raw[u]);
      float a = 0.0f;
#pragma unroll
      for (int k = 0; k < 8; k++) {
        unsigned bits = (k & 1) ? (words[k / 2] & 0xffff0000u) : (words[k / 2] << 16);
        a += av[k] * __uint_as_float(bits);
      }
      rs[u] = a;
    }
#pragma unroll
    for (int u = 0; u < OPROJ_BATCH; u++) {
      float a = rs[u];
#pragma unroll
      for (int s = 1; s < LANES_PER_ROW; s <<= 1) {    // three xor steps: 1, 2, 4
        a += __shfl_xor(a, s, WAVE);
      }
      if (chunk == 0) {                                // eight rows of the wave-load, one store
        ws[row_base + (g0 + u) * ROWS_PER_LOAD + row_in_load] = a;
      }
    }
  }

  // (5) the release, the counter and the broadcast of "this task was the last to arrive"
  __syncthreads();
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  __syncthreads();                                     // the fence in every wave before one thread's atomic (the page's order)
  if (tid == 0) {
    int old = __hip_atomic_fetch_add(static_cast<int *>(counter_ptr), 1, __ATOMIC_ACQ_REL,
                                     __HIP_MEMORY_SCOPE_AGENT);
    last_s[0] = (old == TASKS - 1) ? 1 : 0;
  }
  __syncthreads();
  if (last_s[0] == 0) {
    return;
  }
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");   // the other tasks' workspace rows

  // the last task: thread t sums the TASKS partial vectors over its eight columns in ascending
  // task order, adds the residual in FP32 and rounds once
  float const *ws_all = static_cast<float const *>(workspace_ptr) + 8 * tid;
  T *x_res = static_cast<T *>(x_res_ptr) + 8 * tid;
  float acc[8];
#pragma unroll
  for (int k = 0; k < 8; k++) {
    acc[k] = 0.0f;
  }
#pragma unroll 1
  for (int k0 = 0; k0 < TASKS; k0 += OPROJ_WS_BATCH) {
    uint4 v[OPROJ_WS_BATCH][2];
#pragma unroll
    for (int u = 0; u < OPROJ_WS_BATCH; u++) {
      float const *p = ws_all + (size_t)(k0 + u) * HIDDEN;
      v[u][0] = *reinterpret_cast<uint4 const *>(p);
      v[u][1] = *reinterpret_cast<uint4 const *>(p + 4);
    }
#pragma unroll
    for (int u = 0; u < OPROJ_WS_BATCH; u++) {
      acc[0] += __uint_as_float(v[u][0].x); acc[1] += __uint_as_float(v[u][0].y);
      acc[2] += __uint_as_float(v[u][0].z); acc[3] += __uint_as_float(v[u][0].w);
      acc[4] += __uint_as_float(v[u][1].x); acc[5] += __uint_as_float(v[u][1].y);
      acc[6] += __uint_as_float(v[u][1].z); acc[7] += __uint_as_float(v[u][1].w);
    }
  }
  float res[8];
  load8(x_res, res);                                   // one 16-byte load: the eight BF16 values
  uint4 out;
  unsigned *out_words = reinterpret_cast<unsigned *>(&out);
#pragma unroll
  for (int k = 0; k < 4; k++) {
    unsigned lo = __float_as_uint(bf16r(acc[2 * k] + res[2 * k])) >> 16;
    unsigned hi = __float_as_uint(bf16r(acc[2 * k + 1] + res[2 * k + 1])) & 0xffff0000u;
    out_words[k] = lo | hi;
  }
  *reinterpret_cast<uint4 *>(x_res) = out;             // one 16-byte store, in place

  if (tid == 0) {
    *static_cast<int *>(counter_ptr) = 0;              // ready for the next layer's tasks
  }
}

} // namespace kernel
