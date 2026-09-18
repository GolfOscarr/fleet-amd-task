/* moe_router_mi300: one CU-task per MoE layer (docs/design-doc/02-task-graph.md
 * M1, 00-decisions.md D6/D7). Replaces linear + moe_topk_softmax_mi300, whose
 * renormalize = true is wrong for this model.
 *
 *   logits = fp32(h) . fp32(W_gate[e])   (BF16 x BF16 products are exact in FP32)
 *   p      = softmax(logits)             (FP32)
 *   top-k by repeated argmax, the lower expert index winning ties
 *   topk_w[k]     = p[idx_k] * scaling for k < TOPK; 1.0 for the forced slots
 *   routing[e]    = slot + 1 for a selected expert (slot in 0..TOPK+N_FORCED), else 0
 *   mask[s]       = expert id of slot s; mask[N_EXPERTS + N_FORCED] = TOPK + N_FORCED;
 *                   unused entries -1 (as the stock kernel; the consumers read only
 *                   mask[count] and mask[0..count), gang_moe_linear_mi300.cuh)
 *   route_log[step - (prompt_len - 1)][layer_index][s] = mask[s]  (always on, 07-correctness.md)
 *
 * Math: harness/numpy_ref.py moe_router. Pointer order and params:
 * fleet/build_graph.py moe_router_layer.
 *
 * Inputs : h [1, HIDDEN] BF16, W_gate [N_EXPERTS, HIDDEN] BF16
 * Outputs: topk_w [1, TOPK + N_FORCED] FP32, routing [N_EXPERTS + N_FORCED, 1] int32,
 *          mask [N_EXPERTS + N_FORCED + 1] int32, logits [1, N_EXPERTS] FP32,
 *          route_log [ROUTE_STEPS, ROUTE_LAYERS, TOPK + N_FORCED] int32
 * LDS    : logits [N_EXPERTS] FP32.
 * Work   : four waves own N_EXPERTS / 4 experts each and walk them in batches of
 *          ROUTER_BATCH rows (8; -DROUTER_BATCH=4 or 16) under #pragma unroll 1, so the
 *          constant is the depth of the loads in flight and not a hint the compiler may
 *          hoist away (docs/gpu-experiments/04-kernels, 01-gemv-ideas K6,
 *          03-router-merge-ideas R2). A row is PER_LANE / 8 16-byte loads per lane at the
 *          element chunks 8 * lane + 512 * i, one contiguous KB per wave-load (K9, R3;
 *          -DROUTER_STRIDED restores round 3's lane-contiguous 32 * lane + 8 * i for the
 *          ktime A/B), read through a StreamSrc so the gate weight carries the linears'
 *          sc1 nt policy under MLA_NT_STREAMS (R6). The first batch's loads are issued
 *          before the norm, whose round trip and reductions then run under their latency
 *          (R1); the x slice is read with the same map; the batch's rows are reduced by
 *          one halving butterfly (butterfly_sum<ROUTER_BATCH>) and lane l < ROUTER_BATCH
 *          writes logit_s[e0 + l]. The softmax and the top-k run on wave 0 with one
 *          expert per lane (N_EXPERTS <= 64), and the writes are spread over the lanes
 *          (R4): the initialisations one entry per thread before the GEMV's barrier, the
 *          slots and the log eight lanes wide, the count by lane 0.
 * Numerics: the products are the same as round 3's, the per-lane accumulation is four
 *          runs of eight instead of one run of 32 and the cross-lane sum a butterfly
 *          instead of a wave sum per row, so logits and topk_w move at the FP32 rounding
 *          level; the ids, routing, mask and the route log are unchanged.
 *
 * SPLIT = 4 (N2 of docs/gpu-experiments/04-kernels, R5 of 03-router-merge-ideas.md;
 * registration moe_router_norm4_mi300): the GEMV over four regular tasks of 16
 * experts each and the last-arriving task routes. Every task runs the norm (only
 * part 0 stores h; all of them fill the LDS row), multiplies experts
 * 16 part .. 16 part + 15 (four rows per wave, one batch of 16 loads per lane) and
 * writes its 16 logits to logit_s and to the logits tensor; then a barrier, an
 * agent-scope release fence by every thread, thread 0's acq-rel add on the counter
 * tensor and the broadcast of old == SPLIT - 1 through LDS, and a second barrier.
 * The task that saw the last increment runs an agent-scope acquire fence in every
 * thread (its own L1 and its XCD's L2 lines invalidated), reads the 64 logits back
 * from the tensor into wave 0's lanes and runs the softmax, the top-k and the
 * phase-5 writes exactly as SPLIT = 1 does, then thread 0 resets the counter. The
 * initialisations of routing and mask run in every task before the GEMV, as they do
 * today: the same values, and the release of each task orders them before the last
 * task's slot writes. The counter is a [1] int32 tensor of the plan, zeroed at
 * allocation. A logit is the same float in both forms: a lane's chain of 32 products
 * is unchanged and the cross-lane reduction is butterfly_sum<ROUTER_BATCH> either
 * way (the wave's four experts are one batch of ROUTER_BATCH rows whose unused
 * slots hold zero, and the butterfly never mixes rows), so the batch constant, not
 * the split, fixes the summation order; every output is bit-identical.
 *
 * NORM = true (docs/gpu-experiments/03-acceleration, O1: the post-attention
 * norm folded into the router, registration moe_router_norm_mi300): the
 * input is the residual x_res [1, HIDDEN] and w_norm [HIDDEN] is the norm
 * weight; the task computes h = rmsnorm(x_res) in the reference's order
 * (numpy_ref.rmsnorm: FP32 statistics, the normalized value rounded to BF16,
 * then the BF16 weight multiply), writes it to the extra output h for the
 * expert gate-up, and routes from the LDS copy of the same row. Math:
 * numpy_ref.moe_router_norm. LDS then also holds h [HIDDEN] BF16 (4 KiB) and
 * the reduction scratch. NORM = false is the original task: x is h, w_norm
 * and h_out are unused (nullptr).
 */
#pragma once
#include "tasks/common/common_header.cuh"
#include "tasks/mi300/mla_common_mi300.cuh"

// The rows of the gate weight a wave keeps in flight (R2); a power of two dividing
// N_EXPERTS / WAVES, swept by the ktime A/B as -DROUTER_BATCH=4 or 16.
#ifndef ROUTER_BATCH
#define ROUTER_BATCH 8
#endif

namespace kernel {

// The element offset of lane `lane`'s chunk i of a row of WAVE * PER_LANE elements: the
// coalesced map (8 * lane + 512 * i at PER_LANE = 32), whose wave-load is one contiguous
// KB, or round 3's lane-contiguous slice under -DROUTER_STRIDED.
template <int PER_LANE>
__device__ __forceinline__ int router_chunk(int lane, int i) {
#ifdef ROUTER_STRIDED
  return lane * PER_LANE + 8 * i;
#else
  return 8 * lane + 8 * dsv2::WAVE * i;
#endif
}

// One batch of E_BATCH weight rows, PER_LANE / 8 raw 16-byte words per row per lane, kept
// unconverted (4 VGPRs per 8 elements) and converted on use, exact as load8's conversion.
template <typename T, int HIDDEN, int PER_LANE, int E_BATCH, int LOADS>
__device__ __forceinline__ void router_load_batch(dsv2::StreamSrc<T> const &w_gate, int e0, int lane,
                                                  uint4 (&raw)[E_BATCH][LOADS]) {
#pragma unroll
  for (int u = 0; u < E_BATCH; u++) {
    size_t row = (size_t)(e0 + u) * HIDDEN;
#pragma unroll
    for (int i = 0; i < LOADS; i++) {
      raw[u][i] = dsv2::load16_from(w_gate, row + router_chunk<PER_LANE>(lane, i));
    }
  }
}

template <typename T, int HIDDEN, int N_EXPERTS, int N_FORCED, int TOPK,
          int ROUTE_STEPS, int ROUTE_LAYERS, bool NORM = false, int SPLIT = 1>
__device__ __forceinline__ void
    moe_router_mi300_task_impl(void const *x_ptr,
                               void const *w_norm_ptr,
                               void const *w_gate_ptr,
                               void *h_out_ptr,
                               void *topk_w_ptr,
                               void *routing_ptr,
                               void *mask_ptr,
                               void *logits_ptr,
                               void *route_log_ptr,
                               int step,
                               int prompt_len,
                               int layer_index,
                               float scaling,
                               float eps,
                               int part = 0,
                               void *counter_ptr = nullptr) {
  using namespace dsv2;
  static_assert(N_EXPERTS <= WAVE, "softmax and top-k hold one expert per lane");
  static_assert(SPLIT >= 1 && N_EXPERTS % (WAVES * SPLIT) == 0,
                "the task's experts split over the 4 waves");
  static_assert(HIDDEN % (WAVE * 8) == 0, "16-byte loads, HIDDEN / 64 per lane");
  static_assert(!NORM || (HIDDEN * sizeof(T)) % 16 == 0, "the LDS copy of h keeps the logits 16-byte aligned");
  static_assert(TOPK + N_FORCED <= WAVE, "the slots are written one per lane of wave 0");
  constexpr int N_SLOTS = TOPK + N_FORCED;
  constexpr int N_TOTAL = N_EXPERTS + N_FORCED;
  constexpr int E_PER_TASK = N_EXPERTS / SPLIT;      // R5: this task's experts
  constexpr int E_PER_WAVE = E_PER_TASK / WAVES;
  constexpr int PER_LANE = HIDDEN / WAVE;

  T const *w_gate = static_cast<T const *>(w_gate_ptr);
  float *topk_w = static_cast<float *>(topk_w_ptr);
  int *routing = static_cast<int *>(routing_ptr);
  int *mask = static_cast<int *>(mask_ptr);
  float *logits = static_cast<float *>(logits_ptr);
  int *route_log = static_cast<int *>(route_log_ptr);

  extern __shared__ char smem[];
  // LDS: [h (NORM only, HIDDEN BF16)] [logits N_EXPERTS FP32] [red 4 FP32]
  T *h_s = reinterpret_cast<T *>(smem);                                                 // [HIDDEN]
  float *logit_s = reinterpret_cast<float *>(smem + (NORM ? HIDDEN * sizeof(T) : 0));  // [N_EXPERTS]
  float *red = logit_s + N_EXPERTS;                                                     // [4]
  int *last_s = reinterpret_cast<int *>(red + WAVES);                                   // [1] (SPLIT > 1)
  int tid = threadIdx.x;
  int wave = tid / WAVE, lane = tid % WAVE;

  // E_BATCH expert rows' loads in flight per lane before the first multiply (round 3: one row
  // at a time cost 27 us per layer, a load round trip per expert; R2: the constant only holds
  // under #pragma unroll 1, which the round-3 loop did not carry)
  constexpr int E_BATCH = ROUTER_BATCH;
  // the rows a batch actually loads: the whole batch, or the wave's four experts under
  // SPLIT = 4. The butterfly stays E_BATCH wide (its unused rows hold zero and it never
  // mixes rows), so a logit's cross-lane summation order is the batch constant's in both
  // forms and the four-task split is bit-exact against the one-task kernel.
  constexpr int E_ROWS = E_PER_WAVE < E_BATCH ? E_PER_WAVE : E_BATCH;
  static_assert(E_BATCH >= 1 && E_BATCH <= WAVE && (E_BATCH & (E_BATCH - 1)) == 0,
                "the butterfly wants a power of two");
  static_assert(E_PER_WAVE % E_ROWS == 0, "the wave's experts split into whole batches");
  constexpr int LOADS = PER_LANE / 8;
  // the part offset is written under if constexpr so that SPLIT = 1 keeps today's code exactly
  int e_first = wave * E_PER_WAVE;
  if constexpr (SPLIT > 1) {
    e_first += part * E_PER_TASK;
  }
  StreamSrc<T> w_src(w_gate);                   // sc1 nt under MLA_NT_STREAMS (R6)

  // R1 (the first batch before the norm) was measured and dropped: a batch live across the
  // prologue costs about 60 to 75 registers with this compiler (the probe of 2026-09-18:
  // 238 against 170), which the worker union cannot take; the batches are loaded in the loop

  // R4: the initialisations do not depend on the logits, so they run here, one entry per
  // thread, and the barrier that closes the GEMV separates them from the slot writes below
  for (int e = tid; e <= N_TOTAL; e += NUM_THREADS) {
    if (e < N_TOTAL) {
      routing[e] = 0;                            // routing is [N_TOTAL, 1]
    }
    mask[e] = -1;                                // mask[N_TOTAL] takes the count after the top-k
  }

  // the row the GEMV reads: h in global memory, or the normalized row in LDS
  T const *h = static_cast<T const *>(x_ptr);
  if constexpr (NORM) {
    // R5: every task normalises (the LDS row is what its GEMV reads); only part 0 stores h,
    // and the expert gate-up reads it after the operator's event, so nothing waits on it
    T *h_out = static_cast<T *>(h_out_ptr);
    if constexpr (SPLIT > 1) {
      h_out = (part == 0) ? h_out : nullptr;
    }
    rmsnorm_row<T, HIDDEN>(static_cast<T const *>(x_ptr), static_cast<T const *>(w_norm_ptr),
                           h_out, eps, red, h_s);
    __syncthreads();
    h = h_s;
  } else {
    (void)w_norm_ptr; (void)h_out_ptr; (void)eps; (void)red;
  }

  // FP32 dot products: wave -> E_PER_WAVE experts, lane -> PER_LANE elements, read with the
  // weight's map so that hv[8 i + k] is the element raw[u][i]'s word k multiplies
  float hv[PER_LANE];
#pragma unroll
  for (int i = 0; i < LOADS; i++) {
    load8(h + router_chunk<PER_LANE>(lane, i), hv + 8 * i);
  }
#pragma unroll 1
  for (int e0 = e_first; e0 < e_first + E_PER_WAVE; e0 += E_ROWS) {
    uint4 raw[E_ROWS][LOADS];
    router_load_batch<T, HIDDEN, PER_LANE, E_ROWS, LOADS>(w_src, e0, lane, raw);
    float sums[E_BATCH];
#pragma unroll
    for (int u = E_ROWS; u < E_BATCH; u++) {     // the butterfly's unused rows (SPLIT > 1)
      sums[u] = 0.0f;
    }
#pragma unroll
    for (int u = 0; u < E_ROWS; u++) {
      float acc = 0.0f;
#pragma unroll
      for (int i = 0; i < LOADS; i++) {
        unsigned const *words = reinterpret_cast<unsigned const *>(&raw[u][i]);
#pragma unroll
        for (int k = 0; k < 8; k++) {
          unsigned bits = (k & 1) ? (words[k / 2] & 0xffff0000u) : (words[k / 2] << 16);
          acc += hv[8 * i + k] * __uint_as_float(bits);
        }
      }
      sums[u] = acc;
    }
    // one halving butterfly for the whole batch instead of a wave sum per row: lane l < E_ROWS
    // comes back holding row e0 + l's total
    float total = butterfly_sum<E_BATCH>(sums);
    if (lane < E_ROWS) {
      logit_s[e0 + lane] = total;
      if constexpr (SPLIT > 1) {
        logits[e0 + lane] = total;               // R5: the last task reads the 64 back from here
      }
    }
  }
  __syncthreads();

  // R5: the release, the counter and the broadcast of "this task was the last to arrive";
  // the tasks that were not return here, and the last one acquires what they wrote
  if constexpr (SPLIT > 1) {
    int *counter = static_cast<int *>(counter_ptr);
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    __syncthreads();                             // every wave's fence before one thread's atomic (the page's order)
    if (tid == 0) {
      int old = __hip_atomic_fetch_add(counter, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
      last_s[0] = (old == SPLIT - 1) ? 1 : 0;
    }
    __syncthreads();
    if (last_s[0] == 0) {
      return;
    }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }

  if (wave == 0) {
    // softmax over N_EXPERTS, one expert per lane. With SPLIT > 1 the 64 logits come back
    // from the tensor (one plain load per lane, after the acquire fence): the same floats
    // the four GEMV tasks stored, so the softmax and the top-k see what SPLIT = 1 sees.
    float x;
    if constexpr (SPLIT == 1) {
      x = (lane < N_EXPERTS) ? logit_s[lane] : -INFINITY;
    } else {
      x = (lane < N_EXPERTS) ? logits[lane] : -INFINITY;
    }
    float mx = wave_max(x);
    float ex = (lane < N_EXPERTS) ? expf(x - mx) : 0.0f;
    float sum = wave_sum(ex);
    float p = ex / sum;
    if constexpr (SPLIT == 1) {
      if (lane < N_EXPERTS) {
        logits[lane] = x;
      }
    }
    // top-k by repeated argmax; ties go to the lower index
    float remaining = (lane < N_EXPERTS) ? p : -INFINITY;
    int ids[N_SLOTS];
    float ws[N_SLOTS];
#pragma unroll
    for (int k = 0; k < TOPK; k++) {
      float v = remaining;
      int idx = lane;
#pragma unroll
      for (int off = WAVE / 2; off > 0; off >>= 1) {
        float ov = __shfl_xor(v, off, WAVE);
        int oi = __shfl_xor(idx, off, WAVE);
        if (ov > v || (ov == v && oi < idx)) {
          v = ov;
          idx = oi;
        }
      }
      ids[k] = idx;
      ws[k] = v * scaling;
      if (lane == idx) {
        remaining = -INFINITY;
      }
    }
#pragma unroll
    for (int f = 0; f < N_FORCED; f++) {
      ids[TOPK + f] = N_EXPERTS + f;
      ws[TOPK + f] = 1.0f;
    }
    // R4: the slots go out eight lanes wide. ids and ws are the same in every lane of the wave
    // (the argmax butterfly leaves its winner everywhere), so lane s takes slot s; the pick is
    // an unrolled select rather than ids[lane], which would put the arrays in scratch
    int my_id = 0;
    float my_w = 0.0f;
#pragma unroll
    for (int s = 0; s < N_SLOTS; s++) {
      if (lane == s) {
        my_id = ids[s];
        my_w = ws[s];
      }
    }
    if (lane < N_SLOTS) {
      topk_w[lane] = my_w;
      routing[my_id] = lane + 1;
      mask[lane] = my_id;
    }
    if (lane == 0) {
      mask[N_TOTAL] = N_SLOTS;                       // the count, after the -1 of the prologue
    }
    int row = step - (prompt_len - 1);               // iteration index: 0 at the hand-over position
    if (lane < N_SLOTS && row >= 0 && row < ROUTE_STEPS &&
        layer_index >= 0 && layer_index < ROUTE_LAYERS) {
      int *log = route_log + ((size_t)row * ROUTE_LAYERS + layer_index) * N_SLOTS;
      log[lane] = my_id;
    }
  }

  // R5: the counter goes back to zero for the next layer's four tasks (the chain serialises
  // the routers, so one counter tensor serves them all)
  if constexpr (SPLIT > 1) {
    if (tid == 0) {
      *static_cast<int *>(counter_ptr) = 0;
    }
  }
}

} // namespace kernel
