/* gang_moe_w2_silu_mi300: the expert down projection with the silu-mul folded
 * into its prologue (docs/gpu-experiments/03-acceleration/03-local-preparation.md,
 * O2). A copy of the runtime's gang_moe_w2_linear_kernel
 * (tasks/mi300/gang_moe_linear_mi300.cuh) whose A operand is computed by the
 * task itself: act[slot] = silu(gate) * up from mid[slot] (gate in columns
 * [0, K), up in [K, 2K), the layout the stock silu_mul_task_impl reads),
 * written to a per-tile scratch row and handed to the unchanged CK pipeline.
 * Removes the L{l}.silu operator (8 tasks, about 41 us of event gap per MoE
 * layer in round 2) at the cost of 32 tiles per expert recomputing a 1,408-wide
 * row (microseconds, in L2). The same pattern as the runtime's own dense
 * silu_mul_linear_mi300.cuh ("output = silu(gate) * up @ weight^T").
 *
 * Rounding: fast_silu in FP32, the product rounded to BF16 with
 * __float2bfloat16, element for element what silu_mul_task_impl does, so act
 * and therefore out8 are bit-identical to the un-fused graph's; the boundary
 * compare of out8 (B11, B12) and the layer output (B13) validate it.
 *
 * Scratch: [8 x TOTAL_TILES_PER_XCD, K] BF16, one row per (XCD, tile); the
 * row index is xcd * TOTAL_TILES_PER_XCD + tile_idx because the runtime's
 * tile_idx for a MoE gang task is XCD-local (n_tile_start = 0) and the XCD
 * comes from the hardware register, as in the stock kernel. Reused by every
 * layer (the chain serialises them).
 *
 * Visibility of the scratch row to the CK loads of the same workgroup: the
 * prologue's stores are followed by s_waitcnt 0, a workgroup-scope release fence
 * and __syncthreads() (the stores have completed; the LLVM AMDGPU memory model
 * needs no L1 invalidate at workgroup scope outside threadgroup-split mode),
 * and the A tensor view carries the sc0 bit (CK amd_buffer_coherence_enum
 * GROUP = 1 on gfx942, bit 0 = sc0: the L1 is bypassed for these loads), so a line an earlier task left in this CU's L1
 * cannot be read. The offline disassembly shows the sc0 bit on the A loads.
 *
 * Inputs : mid [1, NUM_TOPK, 2K] BF16 (gate | up per slot), W2 [E_TOTAL, N, K] BF16,
 *          routing [E_TOTAL, 1] int32, mask [E_TOTAL + 1] int32
 * Outputs: out8 [1, NUM_TOPK, N] BF16, scratch [8 x TOTAL_TILES_PER_XCD, K] BF16
 * Params (register_task): [tiles_per_expert, max_experts_per_xcd, total_tiles_per_xcd]
 * (the stock w2's three); the registration takes K from the weight, not from
 * the input whose last dimension is 2K.
 */
#pragma once
#include "tasks/mi300/gang_moe_linear_mi300.cuh"   // ck_tile, _gang_moe_get_xcd_id, the pipeline types
#include "tasks/mi300/silu_mul_mi300.cuh"          // fast_silu

namespace kernel {

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int OUTPUT_STRIDE,
          int REDUCTION_SIZE,
          int IN_STRIDE,            // mid's slot stride: 2 x REDUCTION_SIZE
          int NUM_EXPERTS,
          int NUM_TOPK,
          int TILES_PER_EXPERT,
          int N_TILES,
          int TOTAL_TILES_PER_XCD>
__device__ __noinline__ void
    gang_moe_w2_silu_linear_kernel(void const *mid_ptr,
                                    void const *weight_ptr,
                                    void const *routing_ptr,
                                    void const *mask_ptr,
                                    void *output_ptr,
                                    void *scratch_ptr,
                                    int tile_idx) {
  using namespace ck_tile;
  static_assert(BATCH_SIZE == 1, "the prologue computes one token's row per tile");
  static_assert(IN_STRIDE >= 2 * REDUCTION_SIZE, "mid holds gate then up per slot");
  static_assert(REDUCTION_SIZE % 8 == 0, "16-byte vectors of 8 BF16");

  // --- the stock kernel's tile shape and pipeline, unchanged ---
  constexpr index_t MPerBlock = 16;
  constexpr index_t NPerBlock = 64;
  constexpr index_t KPerBlock = (REDUCTION_SIZE % 256 == 0) ? 256 : 128;
  constexpr index_t NumLoopK = REDUCTION_SIZE / KPerBlock;
  static_assert(REDUCTION_SIZE % KPerBlock == 0, "W2 REDUCTION_SIZE must be divisible by KPerBlock");
  constexpr index_t MWarp = 1;
  constexpr index_t NWarp = 4;
  using BlockTile = sequence<MPerBlock, NPerBlock, KPerBlock>;
  using BlockWarps = sequence<MWarp, NWarp>;
  using WarpTile = sequence<MPerBlock, NPerBlock / NWarp, KPerBlock>;
  using GemmShape = TileGemmShape<BlockTile, BlockWarps, WarpTile>;
  using GemmTraits = TileGemmUniversalTraits<true, false, true, false,
                                             tensor_layout::gemm::RowMajor,
                                             tensor_layout::gemm::ColumnMajor,
                                             tensor_layout::gemm::RowMajor>;
  using Problem = GemmPipelineProblem<bf16, bf16, float, GemmShape, GemmTraits>;
  using PipelinePolicy = GemmPipelineSmallTilePolicy<MPerBlock, NPerBlock, KPerBlock>;
  using Pipeline = GemmPipelineAGmemBGmemCRegV2<Problem, PipelinePolicy>;

  int xcd_id = _gang_moe_get_xcd_id();
  int const *__restrict__ d_mask = static_cast<int const *>(mask_ptr);
  int const num_activated_experts = d_mask[NUM_EXPERTS];
  int expert_local_idx = tile_idx / TILES_PER_EXPERT;
  int tile_within_expert = tile_idx % TILES_PER_EXPERT;
  int ae_idx = xcd_id + expert_local_idx * 8;
  if (ae_idx >= num_activated_experts) return;
  int expert_id = d_mask[ae_idx];
  int w2_tok = tile_within_expert / N_TILES;
  int n_tile = tile_within_expert % N_TILES;
  if (w2_tok >= BATCH_SIZE) return;

  bf16 const *__restrict__ d_mid = static_cast<bf16 const *>(mid_ptr);
  bf16 const *__restrict__ d_weight = static_cast<bf16 const *>(weight_ptr);
  bf16 *__restrict__ d_output = static_cast<bf16 *>(output_ptr);
  bf16 *__restrict__ d_scratch = static_cast<bf16 *>(scratch_ptr);
  int const *__restrict__ d_routing = static_cast<int const *>(routing_ptr);
  int const *expert_routing = d_routing + expert_id * BATCH_SIZE;
  int const route_val = expert_routing[w2_tok];
  if (route_val == 0) return;
  int const topk_slot = route_val - 1;

  // --- the prologue: act[slot] = silu(gate) * up into this tile's scratch row ---
  // (HIP's bf16 for the arithmetic, as silu_mul_task_impl; ck_tile's bf16 for the GEMM below)
  using hbf16 = __hip_bfloat16;
  hbf16 const *gate = reinterpret_cast<hbf16 const *>(d_mid) +
                      static_cast<size_t>(w2_tok) * (NUM_TOPK * IN_STRIDE) +
                      static_cast<size_t>(topk_slot) * IN_STRIDE;
  hbf16 const *up = gate + REDUCTION_SIZE;
  size_t act_row = static_cast<size_t>(xcd_id) * TOTAL_TILES_PER_XCD + static_cast<size_t>(tile_idx);
  hbf16 *act = reinterpret_cast<hbf16 *>(d_scratch) + act_row * REDUCTION_SIZE;
  for (int v = threadIdx.x; v < REDUCTION_SIZE / 8; v += static_cast<int>(blockDim.x)) {
    int off = v * 8;
    uint64_t g_lo = *reinterpret_cast<uint64_t const *>(gate + off);
    uint64_t g_hi = *reinterpret_cast<uint64_t const *>(gate + off + 4);
    uint64_t u_lo = *reinterpret_cast<uint64_t const *>(up + off);
    uint64_t u_hi = *reinterpret_cast<uint64_t const *>(up + off + 4);
    hbf16 const *g = reinterpret_cast<hbf16 const *>(&g_lo);
    hbf16 const *u = reinterpret_cast<hbf16 const *>(&u_lo);
    hbf16 const *g2 = reinterpret_cast<hbf16 const *>(&g_hi);
    hbf16 const *u2 = reinterpret_cast<hbf16 const *>(&u_hi);
    hbf16 out_arr[8];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      out_arr[k] = __float2bfloat16(fast_silu(__bfloat162float(g[k])) * __bfloat162float(u[k]));
      out_arr[4 + k] = __float2bfloat16(fast_silu(__bfloat162float(g2[k])) * __bfloat162float(u2[k]));
    }
    *reinterpret_cast<uint64_t *>(act + off) = *reinterpret_cast<uint64_t *>(&out_arr[0]);
    *reinterpret_cast<uint64_t *>(act + off + 4) = *reinterpret_cast<uint64_t *>(&out_arr[4]);
  }
  // the stores have reached L2 before any thread's pipeline loads the row: the explicit
  // s_waitcnt 0 (vmcnt, expcnt, lgkmcnt) is added because the workgroup-scope release alone
  // lowers to lgkmcnt(0) and the barrier outside threadgroup-split mode (the LLVM AMDGPU
  // memory model relies on the CU's in-order vector memory pipeline; the wait makes the
  // store's completion explicit at the cost of one wait per tile)
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __syncthreads();

  // --- the stock kernel from here on, with the scratch row as A (sc0: L1 bypassed) ---
  bf16 const *a_base = d_scratch + act_row * REDUCTION_SIZE;
  bf16 const *expert_weight = d_weight + static_cast<int64_t>(expert_id) * OUTPUT_STRIDE * REDUCTION_SIZE;
  index_t n_offset = n_tile * NPerBlock;
  bf16 const *b_base = expert_weight + static_cast<size_t>(n_offset) * REDUCTION_SIZE;
  index_t n_size = (n_offset + NPerBlock <= OUTPUT_SIZE) ? NPerBlock : (OUTPUT_SIZE - n_offset);
  extern __shared__ char smem[];

  auto a_tensor_view = make_naive_tensor_view<address_space_enum::global,
                                              memory_operation_enum::set,
                                              static_cast<amd_buffer_coherence_enum>(1)>(   // GROUP: sc0, L1 bypassed
      a_base,
      make_tuple(index_t(1), index_t(REDUCTION_SIZE)),
      make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
      number<8>{},
      number<1>{});
#ifdef MPK_NT_WEIGHT_LOADS
  auto b_tensor_view = make_naive_tensor_view<address_space_enum::global,
                                              memory_operation_enum::set,
                                              static_cast<amd_buffer_coherence_enum>(18)>(
#else
  auto b_tensor_view = make_naive_tensor_view<address_space_enum::global>(
#endif
      b_base,
      make_tuple(n_size, index_t(REDUCTION_SIZE)),
      make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
      number<8>{},
      number<1>{});
  auto a_tile_window = make_tile_window(a_tensor_view, make_tuple(number<MPerBlock>{}, number<KPerBlock>{}), {0, 0});
  auto b_tile_window = make_tile_window(b_tensor_view, make_tuple(number<NPerBlock>{}, number<KPerBlock>{}), {0, 0});
  Pipeline pipeline;
  auto c_block_tile = pipeline(a_tile_window, b_tile_window, NumLoopK, smem);
  block_sync_lds();

  // ---- Epilogue: write result for this token (the stock code) ----
  auto &c_buf = c_block_tile.get_thread_buffer();
  index_t warp_id = threadIdx.x >> 6;
  index_t lane_id = threadIdx.x & 63;
  index_t tile_row = lane_id & 15;
  index_t tile_col_base = warp_id * 16 + ((lane_id >> 4) << 2);
  index_t global_n_base = n_offset + tile_col_base;
  if (tile_row == 0) {
    if (global_n_base + 3 < OUTPUT_SIZE) {
      bf16 *out_addr = d_output + static_cast<size_t>(w2_tok) * (NUM_TOPK * OUTPUT_STRIDE) +
                       static_cast<size_t>(topk_slot) * OUTPUT_STRIDE + global_n_base;
      uint64_t out_packed;
      bf16 *out = reinterpret_cast<bf16 *>(&out_packed);
      out[0] = type_convert<bf16>(c_buf[0]);
      out[1] = type_convert<bf16>(c_buf[1]);
      out[2] = type_convert<bf16>(c_buf[2]);
      out[3] = type_convert<bf16>(c_buf[3]);
      *reinterpret_cast<uint64_t *>(out_addr) = out_packed;
    } else {
#pragma unroll
      for (index_t i = 0; i < 4; i++) {
        index_t global_n = global_n_base + i;
        if (global_n < OUTPUT_SIZE) {
          d_output[static_cast<size_t>(w2_tok) * (NUM_TOPK * OUTPUT_STRIDE) +
                   static_cast<size_t>(topk_slot) * OUTPUT_STRIDE + global_n] = type_convert<bf16>(c_buf[i]);
        }
      }
    }
  }
}

} // namespace kernel
