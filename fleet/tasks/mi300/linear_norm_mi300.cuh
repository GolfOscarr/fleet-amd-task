/* linear_norm_mi300: the stock per-tile linear with the input norm folded
 * into its prologue (docs/gpu-experiments/03-acceleration/03-local-preparation.md,
 * O3). Each task of the grid normalises the [1, K] input row into its own row
 * of a scratch tensor [grid, K] (the runtime hands the task its row: the
 * scratch is partitioned on dim 0 by the grid, as the weight is) and runs the
 * CK small-tile GEMM of the runtime's linear_kernel_ck on that row. Used for
 * the Q/KV projection of every layer (96 tasks, the norm weight w_norm1) and
 * for lm_head (400 tasks, the final norm), which removes the one-task
 * L{l}.norm1 and head.norm operators (28 boundaries, about 13.6 us each in
 * round 2) at the cost of every task redoing a 2,048-element norm (4 KB of L2
 * reads, microseconds).
 *
 * The norm is rmsnorm_row (mla_common_mi300.cuh, shared with the fused
 * router of O1): FP32 statistics, the normalised value rounded to BF16, then
 * the BF16 weight multiply, the reference's DeepseekV2RMSNorm order and the
 * same rounding as the stock rms_norm task, so the scratch row equals the
 * un-fused graph's h and the product equals the stock linear's; the boundary
 * compare of qkva (B2) and of the logits validates it in the graph (the suite
 * has no CK linear).
 *
 * The GEMM is the batch-1 tier of linear_kernel_ck (linear_ck_mi300.cuh,
 * BATCH_SIZE <= 16: 16 x 64 x 256 tiles, 16 x 16 x 16 MFMA, MWarp 1, NWarp 4,
 * the hand-written 16 x 64 epilogue), copied rather than called because the
 * A tensor view's coherence bits live inside that function: here the A view
 * carries sc0 (CK amd_buffer_coherence_enum GROUP = 1 on gfx942, bit 0 = sc0:
 * the L1 is bypassed for the loads of the row this workgroup just wrote), the
 * pattern of the fused w2 of O2. Visibility of the scratch row to those loads:
 * the prologue's stores are followed by s_waitcnt 0, a workgroup-scope release
 * fence and __syncthreads() (O2's sequence, verified in the offline
 * disassembly: flat_store, s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0), s_barrier,
 * then buffer_load ... sc0). The residual path of the stock kernel is not
 * needed (qkva and the logits have no residual) and is left out.
 *
 * Inputs : x [1, K] BF16 (whole), w_norm [K] BF16, W [N, K] BF16 (this task's
 *          N / grid rows); Outputs: out [1, N] BF16 (this task's columns; the
 *          stride is the full N), scratch [grid, K] BF16 (this task's row)
 * Params (register_task): [eps_bits]. output_size and o_stride reach the
 * kernel as the stock linear's do (the partitioned output's dim 1 and the
 * full tensor's stride).
 */
#pragma once
#include "tasks/mi300/linear_ck_mi300.cuh"    // ck_tile, GemmPipelineSmallTilePolicy, nt_store_*, __uniform_addr
#include "tasks/mi300/mla_common_mi300.cuh"   // dsv2::rmsnorm_row

namespace kernel {

// __forceinline__ as the stock linear_kernel: the K = 2048 small tile (KPerBlock 256) is the
// register-heavy tier the stock per-tile linear already inlines into the worker, so the copy adds
// nothing to the union the real build carries (offline, with only our tasks in the unit, the
// union is 256 VGPRs plus 23 AGPRs inlined and 256 plus 35 as a __noinline__ call; no VGPR
// spills either way, and the persistent kernel runs one wave per SIMD regardless of occupancy)
template <typename T, int BATCH_SIZE, int REDUCTION_SIZE>
__device__ __forceinline__ void
    linear_norm_mi300_task_impl(void const *x_ptr,
                                void const *w_norm_ptr,
                                void const *weight_ptr,
                                void *output_ptr,
                                void *scratch_ptr,
                                int num_active_tokens,
                                int output_size,
                                int o_stride,
                                float eps) {
  using namespace ck_tile;
  static_assert(BATCH_SIZE == 1, "the prologue normalises one token's row per task");
  static_assert(REDUCTION_SIZE % 256 == 0, "K must be a multiple of the small tile's KPerBlock");
  (void)num_active_tokens;   // the batch-1 epilogue bounds on BATCH_SIZE, as the stock one does

  extern __shared__ char smem[];
  // the prologue's 4-float reduction buffer at the start of the LDS the pipeline reuses after the barrier
  float *red = reinterpret_cast<float *>(smem);

  // --- the prologue: this task's normalised row into its scratch row ---
  dsv2::rmsnorm_row<T, REDUCTION_SIZE>(static_cast<T const *>(x_ptr), static_cast<T const *>(w_norm_ptr),
                                       static_cast<T *>(scratch_ptr), eps, red);
  // the stores have reached L2 before any thread's pipeline loads the row (O2's sequence: the
  // workgroup-scope release alone lowers to lgkmcnt(0) and the barrier outside threadgroup-split
  // mode, so the vector-memory wait is made explicit)
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __syncthreads();

  // --- the stock small tile of linear_kernel_ck, with the scratch row as A (sc0: L1 bypassed) ---
  constexpr index_t MPerBlock = 16;
  constexpr index_t NPerBlock = 64;
  constexpr index_t KPerBlock = 256;
  constexpr index_t NumLoopK = REDUCTION_SIZE / KPerBlock;
  constexpr index_t MWarp = 1;
  constexpr index_t NWarp = 4;
  using BlockTile = sequence<MPerBlock, NPerBlock, KPerBlock>;
  using BlockWarps = sequence<MWarp, NWarp>;
  using WarpTile = sequence<16, 16, 32>;   // the stock small tile's MFMA dimensions (K = 32)
  using GemmShape = TileGemmShape<BlockTile, BlockWarps, WarpTile>;
  using GemmTraits = TileGemmUniversalTraits<true,   // kPadM
                                             false,  // kPadN
                                             true,   // kPadK
                                             false,  // DoubleSmemBuffer
                                             tensor_layout::gemm::RowMajor,
                                             tensor_layout::gemm::ColumnMajor,
                                             tensor_layout::gemm::RowMajor,
                                             false>; // TransposeC
  using Problem = GemmPipelineProblem<bf16, bf16, float, GemmShape, GemmTraits>;
  using PipelinePolicy = GemmPipelineSmallTilePolicy<MPerBlock, NPerBlock, KPerBlock>;
  using Pipeline = GemmPipelineAGmemBGmemCRegV2<Problem, PipelinePolicy>;

  // wave-uniform pointers and dims, as the stock kernel (no v_readfirstlane waterfalls)
  const bf16 *d_input = reinterpret_cast<const bf16 *>(__uniform_addr(scratch_ptr));
  const bf16 *d_weight = reinterpret_cast<const bf16 *>(__uniform_addr(weight_ptr));
  bf16 *d_output = reinterpret_cast<bf16 *>(__uniform_addr(output_ptr));
  output_size = __builtin_amdgcn_readfirstlane(output_size);
  o_stride = __builtin_amdgcn_readfirstlane(o_stride);
  index_t LoopN = (output_size + NPerBlock - 1) / NPerBlock;

  for (index_t nn = 0; nn < LoopN; nn++) {
    index_t n_offset = nn * NPerBlock;
    index_t n_size = (nn == LoopN - 1) ? (output_size - n_offset) : NPerBlock;
    if (n_size <= 0) continue;

    auto a_tensor_view = make_naive_tensor_view<address_space_enum::global,
                                                memory_operation_enum::set,
                                                static_cast<amd_buffer_coherence_enum>(1)>(   // GROUP: sc0
        d_input,
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
        d_weight + n_offset * REDUCTION_SIZE,
        make_tuple(n_size, index_t(REDUCTION_SIZE)),
        make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
        number<8>{},
        number<1>{});
    auto a_tile_window = make_tile_window(a_tensor_view, make_tuple(number<MPerBlock>{}, number<KPerBlock>{}), {0, 0});
    auto b_tile_window = make_tile_window(b_tensor_view, make_tuple(number<NPerBlock>{}, number<KPerBlock>{}), {0, 0});
    Pipeline pipeline;
    auto c_block_tile = pipeline(a_tile_window, b_tile_window, NumLoopK, smem);
    block_sync_lds();

    // the stock 16 x 64 epilogue (16 x 16 MFMA TransposedCDistribution), m_offset 0, no residual
    index_t warp_id = threadIdx.x >> 6;
    index_t lane_id = threadIdx.x & 63;
    index_t tile_row = lane_id & 15;
    index_t tile_col_base = warp_id * 16 + ((lane_id >> 4) << 2);
    auto &c_buf = c_block_tile.get_thread_buffer();
    index_t global_m = tile_row;
    index_t global_n_base = n_offset + tile_col_base;
    if (global_m < BATCH_SIZE && global_n_base + 3 < output_size) {
      index_t out_base = global_m * o_stride + global_n_base;
      uint64_t out_packed;
      bf16 *out = reinterpret_cast<bf16 *>(&out_packed);
      out[0] = type_convert<bf16>(c_buf[0]);
      out[1] = type_convert<bf16>(c_buf[1]);
      out[2] = type_convert<bf16>(c_buf[2]);
      out[3] = type_convert<bf16>(c_buf[3]);
      nt_store_u64(&d_output[out_base], out_packed);
    } else if (global_m < BATCH_SIZE) {
#pragma unroll
      for (index_t i = 0; i < 4; i++) {
        index_t global_n = global_n_base + i;
        if (global_n < output_size) {
          nt_store_bf16(&d_output[global_m * o_stride + global_n], type_convert<bf16>(c_buf[i]));
        }
      }
    }
  }
}

} // namespace kernel
