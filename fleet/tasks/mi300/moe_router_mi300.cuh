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
 * Work   : four waves own N_EXPERTS / 4 experts each; a lane reads HIDDEN / 64
 *          elements of a weight row with 16-byte loads; the softmax and the
 *          top-k run on wave 0 with one expert per lane (N_EXPERTS <= 64).
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

namespace kernel {

template <typename T, int HIDDEN, int N_EXPERTS, int N_FORCED, int TOPK,
          int ROUTE_STEPS, int ROUTE_LAYERS, bool NORM = false>
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
                               float eps) {
  using namespace dsv2;
  static_assert(N_EXPERTS <= WAVE, "softmax and top-k hold one expert per lane");
  static_assert(N_EXPERTS % WAVES == 0, "experts split over the 4 waves");
  static_assert(HIDDEN % (WAVE * 8) == 0, "16-byte loads, HIDDEN / 64 per lane");
  static_assert(!NORM || (HIDDEN * sizeof(T)) % 16 == 0, "the LDS copy of h keeps the logits 16-byte aligned");
  constexpr int N_SLOTS = TOPK + N_FORCED;
  constexpr int N_TOTAL = N_EXPERTS + N_FORCED;
  constexpr int E_PER_WAVE = N_EXPERTS / WAVES;
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
  int tid = threadIdx.x;
  int wave = tid / WAVE, lane = tid % WAVE;

  // the row the GEMV reads: h in global memory, or the normalized row in LDS
  T const *h = static_cast<T const *>(x_ptr);
  if constexpr (NORM) {
    rmsnorm_row<T, HIDDEN>(static_cast<T const *>(x_ptr), static_cast<T const *>(w_norm_ptr),
                           static_cast<T *>(h_out_ptr), eps, red, h_s);
    __syncthreads();
    h = h_s;
  } else {
    (void)w_norm_ptr; (void)h_out_ptr; (void)eps; (void)red;
  }

  // FP32 dot products: wave -> E_PER_WAVE experts, lane -> PER_LANE elements
  float hv[PER_LANE];
#pragma unroll
  for (int i = 0; i < PER_LANE; i += 8) {
    load8(h + lane * PER_LANE + i, hv + i);
  }
  // E_BATCH expert rows' loads in flight per lane before the first multiply (round 3: one row
  // at a time cost 27 us per layer, a load round trip per expert); the rows are kept as raw
  // BF16 words (4 VGPRs per 8 elements) and converted on use, exact as load8's conversion;
  // the FMA order per expert is unchanged
  constexpr int E_BATCH = 4;                    // 16 raw 16-byte words per lane (8 measured 1% slower on the model: registers)
  static_assert(E_PER_WAVE % E_BATCH == 0, "the wave's experts split into whole batches");
  constexpr int LOADS = PER_LANE / 8;
  for (int e0 = wave * E_PER_WAVE; e0 < (wave + 1) * E_PER_WAVE; e0 += E_BATCH) {
    uint4 raw[E_BATCH][LOADS];
#pragma unroll
    for (int u = 0; u < E_BATCH; u++) {
      T const *row = w_gate + (size_t)(e0 + u) * HIDDEN + lane * PER_LANE;
#pragma unroll
      for (int i = 0; i < LOADS; i++) {
        raw[u][i] = *reinterpret_cast<uint4 const *>(row + 8 * i);
      }
    }
#pragma unroll
    for (int u = 0; u < E_BATCH; u++) {
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
      acc = wave_sum(acc);
      if (lane == 0) {
        logit_s[e0 + u] = acc;
      }
    }
  }
  __syncthreads();

  if (wave == 0) {
    // softmax over N_EXPERTS, one expert per lane
    float x = (lane < N_EXPERTS) ? logit_s[lane] : -INFINITY;
    float mx = wave_max(x);
    float ex = (lane < N_EXPERTS) ? expf(x - mx) : 0.0f;
    float sum = wave_sum(ex);
    float p = ex / sum;
    if (lane < N_EXPERTS) {
      logits[lane] = x;
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
    if (lane == 0) {
      for (int e = 0; e < N_TOTAL; e++) {
        routing[e] = 0;                              // routing is [N_TOTAL, 1]
        mask[e] = -1;
      }
      for (int s = 0; s < N_SLOTS; s++) {
        topk_w[s] = ws[s];
        routing[ids[s]] = s + 1;
        mask[s] = ids[s];
      }
      mask[N_TOTAL] = N_SLOTS;
      int row = step - (prompt_len - 1);             // iteration index: 0 at the hand-over position
      if (row >= 0 && row < ROUTE_STEPS && layer_index >= 0 && layer_index < ROUTE_LAYERS) {
        int *log = route_log + ((size_t)row * ROUTE_LAYERS + layer_index) * N_SLOTS;
        for (int s = 0; s < N_SLOTS; s++) {
          log[s] = ids[s];
        }
      }
    }
  }
}

} // namespace kernel
