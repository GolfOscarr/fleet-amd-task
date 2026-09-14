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
 */
#pragma once
#include "tasks/common/common_header.cuh"
#include "tasks/mi300/mla_common_mi300.cuh"

namespace kernel {

template <typename T, int HIDDEN, int N_EXPERTS, int N_FORCED, int TOPK,
          int ROUTE_STEPS, int ROUTE_LAYERS>
__device__ __forceinline__ void
    moe_router_mi300_task_impl(void const *h_ptr,
                               void const *w_gate_ptr,
                               void *topk_w_ptr,
                               void *routing_ptr,
                               void *mask_ptr,
                               void *logits_ptr,
                               void *route_log_ptr,
                               int step,
                               int prompt_len,
                               int layer_index,
                               float scaling) {
  using namespace dsv2;
  static_assert(N_EXPERTS <= WAVE, "softmax and top-k hold one expert per lane");
  static_assert(N_EXPERTS % WAVES == 0, "experts split over the 4 waves");
  static_assert(HIDDEN % (WAVE * 8) == 0, "16-byte loads, HIDDEN / 64 per lane");
  constexpr int N_SLOTS = TOPK + N_FORCED;
  constexpr int N_TOTAL = N_EXPERTS + N_FORCED;
  constexpr int E_PER_WAVE = N_EXPERTS / WAVES;
  constexpr int PER_LANE = HIDDEN / WAVE;

  T const *h = static_cast<T const *>(h_ptr);
  T const *w_gate = static_cast<T const *>(w_gate_ptr);
  float *topk_w = static_cast<float *>(topk_w_ptr);
  int *routing = static_cast<int *>(routing_ptr);
  int *mask = static_cast<int *>(mask_ptr);
  float *logits = static_cast<float *>(logits_ptr);
  int *route_log = static_cast<int *>(route_log_ptr);

  extern __shared__ char smem[];
  float *logit_s = reinterpret_cast<float *>(smem);   // [N_EXPERTS]
  int tid = threadIdx.x;
  int wave = tid / WAVE, lane = tid % WAVE;

  // FP32 dot products: wave -> E_PER_WAVE experts, lane -> PER_LANE elements
  float hv[PER_LANE];
#pragma unroll
  for (int i = 0; i < PER_LANE; i += 8) {
    load8(h + lane * PER_LANE + i, hv + i);
  }
  for (int e = wave * E_PER_WAVE; e < (wave + 1) * E_PER_WAVE; e++) {
    T const *row = w_gate + (size_t)e * HIDDEN + lane * PER_LANE;
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < PER_LANE; i += 8) {
      float w[8];
      load8(row + i, w);
#pragma unroll
      for (int k = 0; k < 8; k++) {
        acc += hv[i + k] * w[k];
      }
    }
    acc = wave_sum(acc);
    if (lane == 0) {
      logit_s[e] = acc;
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
