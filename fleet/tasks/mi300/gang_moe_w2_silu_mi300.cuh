/* gang_moe_w2_silu_mi300: the expert down projection with the silu-mul folded
 * into its prologue (docs/gpu-experiments/03-acceleration/03-local-preparation.md,
 * O2) and the multiply as the GEMV loop of round 4
 * (docs/gpu-experiments/04-kernels/05-local-preparation.md, L3). A copy of the
 * runtime's gang_moe_w2_linear_kernel (tasks/mi300/gang_moe_linear_mi300.cuh)
 * whose A operand is computed by the task itself: act[slot] = silu(gate) * up
 * from mid[slot] (gate in columns [0, K), up in [K, 2K), the layout the stock
 * silu_mul_task_impl reads). Removes the L{l}.silu operator (8 tasks, about
 * 41 us of event gap per MoE layer in round 2) at the cost of 32 tiles per
 * expert recomputing a 1,408-wide row (microseconds, in LDS).
 *
 * The contract of the default path (no MPK_W2_CK_TILE): the activation row
 * stays in LDS. The prologue writes act[k] for k < REDUCTION_SIZE into
 * REDUCTION_SIZE BF16 of dynamic LDS (2,816 bytes at K = 1,408), a
 * __syncthreads follows, and the tile's 64 rows are multiplied from there --
 * no scratch write, no s_waitcnt 0, no release fence, no sc0 read back (all of
 * those existed for the global scratch row). The scratch pointer stays in the
 * signature and is not written: the registration and the plan keep the output
 * this round (700 KB, unused; dropping it is a registration change for a later
 * pass). Only this path is compiled by default, and it includes no CK header,
 * so the kernel is parsed by the host syntax check (fleet/tasks/check_syntax.sh).
 *
 * -DMPK_W2_CK_TILE (a define passed by --runtime-flags) is the one-flag
 * fallback to round 3's form: the same decode and the same prologue, but with
 * the tile's scratch row as the destination, the fences around it and CK's
 * small-tile pipeline as the multiply. That path pulls the CK headers in and
 * is not parsed by the host syntax check.
 *
 * The multiply (default path; docs/gpu-experiments/04-kernels/01-gemv-ideas.md,
 * K1, K7 and K9). Wave w owns rows 16 w .. 16 w + 15 of the tile's 64 rows of
 * the expert's W2 [OUTPUT_SIZE, K] slice, in batches of W2_BATCH rows (the
 * macro, default 8) under #pragma unroll 1: the batch constant is the depth
 * only under the pragma, the round's first convention. K = 1,408 is 176 chunks
 * of eight elements and lane l owns the chunks l + 64 j for j < 3 with
 * l + 64 j < 176, so lanes 0 to 47 own three and lanes 48 to 63 two; a chunk
 * is one 16-byte load and the 64 lanes of one load read a contiguous KB (K9),
 * the absent load of the last 16 lanes leaving zeros against the zeros of
 * their third x register. The x slice is 24 FP32 registers read once from the
 * LDS row at the same chunks (K7); a weight row is three 16-byte loads per
 * lane through load16_from on a StreamSrc (the sc1 nt policy of the linears
 * under MLA_NT_STREAMS, plain loads without it), so a batch of eight rows is
 * 24 loads in flight per lane, 22 KB per wave and 88 KB per CU. The FMAs run
 * on the raw words in ascending k within each chunk, as the router's loop;
 * butterfly_sum<W2_BATCH> then leaves the total of row r0 + l on lane l, and
 * lane l < W2_BATCH stores one BF16 element.
 *
 * Rounding: fast_silu in FP32, the product rounded to BF16 (round to nearest
 * even, the bits __float2bfloat16 gives), element for element what
 * silu_mul_task_impl does, so act is bit-identical to the un-fused graph's.
 * The BF16 x BF16 products of the multiply are exact in FP32; only the
 * summation order differs from the CK path's (a serial chain per lane over the
 * lane's chunks, then the butterfly tree), which is the difference the
 * boundary compares of out8 (B11, B12) and of the layer output (B13) measure.
 *
 * Inputs : mid [1, NUM_TOPK, 2K] BF16 (gate | up per slot), W2 [E_TOTAL, N, K] BF16,
 *          routing [E_TOTAL, 1] int32, mask [E_TOTAL + 1] int32
 * Outputs: out8 [1, NUM_TOPK, N] BF16; scratch [8 x TOTAL_TILES_PER_XCD, K] BF16
 *          (written by the MPK_W2_CK_TILE path only)
 * Params (register_task): [tiles_per_expert, max_experts_per_xcd, total_tiles_per_xcd]
 * (the stock w2's three); the registration takes K from the weight, not from
 * the input whose last dimension is 2K.
 * LDS    : the activation row, REDUCTION_SIZE BF16 (2,816 bytes at K = 1,408),
 *          within the pipeline's dynamic allocation the registration asks for.
 */
#pragma once
#include "tasks/common/common_header.cuh"
#include "tasks/mi300/mla_common_mi300.cuh"        // StreamSrc, load16_from, load8, butterfly_sum, st
#ifdef MPK_W2_CK_TILE
#include "tasks/mi300/gang_moe_linear_mi300.cuh"   // ck_tile, the pipeline types
#include "tasks/mi300/silu_mul_mi300.cuh"          // fast_silu
#define MPK_W2_HAVE_FAST_SILU 1
#elif __has_include("tasks/mi300/silu_mul_mi300.cuh")
#include "tasks/mi300/silu_mul_mi300.cuh"          // fast_silu, when the fork sources are on the include path
#define MPK_W2_HAVE_FAST_SILU 1
#endif

// The rows of one batch, in flight per lane at three 16-byte loads each (M5
// sweeps it: 4, 8, 16). A power of two up to 64, the butterfly's requirement.
#ifndef W2_BATCH
#define W2_BATCH 8
#endif

namespace kernel {

#ifndef MPK_W2_HAVE_FAST_SILU
// The stock silu_mul task's fast_silu (tasks/mi300/silu_mul_mi300.cuh), for the
// builds whose include path has no fork sources: the host syntax check, where
// nothing runs. Every build that generates device code has the fork's own.
__device__ __forceinline__ float fast_silu(float x) {
  return x / (1.0f + expf(-x));
}
#endif

// The XCD of this workgroup: _gang_moe_get_xcd_id of the stock gang kernel
// (tasks/mi300/gang_moe_linear_mi300.cuh), copied here so the default path
// needs no CK header. -DKT_FAKE_XCD takes it from bid.y instead, which is what
// gives the kernel suite a deterministic (tile, XCD) launch; on the host syntax
// check, which has no hardware register, it is zero.
__device__ __forceinline__ int _gang_moe_w2_xcd_id() {
#if defined(KT_FAKE_XCD)
  return (int)blockIdx.y;
#elif defined(__HIP_DEVICE_COMPILE__)
  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
#else
  return 0;
#endif
}

// FP32 -> BF16 bits, round to nearest even with HIP's NaN rule: the bits
// __float2bfloat16 produces, so the fused row is the stock silu's row.
__device__ __forceinline__ uint16_t _gang_moe_w2_bf16(float x) {
  unsigned u = __float_as_uint(x);
  if (~u & 0x7f800000u) {
    u += 0x7fffu + ((u >> 16) & 1u);   // finite
  } else if (u & 0xffffu) {
    u |= 0x10000u;                     // NaN, kept quiet in 16 bits
  }
  return (uint16_t)(u >> 16);
}

// act[k] = bf16(silu(gate[k]) * up[k]) for k < K_SIZE, the row the multiply
// reads. Both paths share it; only the destination differs (the LDS row in the
// default path, the tile's scratch row under MPK_W2_CK_TILE). Raw 16-bit words
// throughout, so the arithmetic is the same either way and needs no HIP bf16
// type: a BF16 value is its word in the high half of an FP32.
template <int K_SIZE>
__device__ __forceinline__ void
    _gang_moe_w2_silu_row(uint16_t const *gate, uint16_t const *up, uint16_t *act) {
  for (int v = threadIdx.x; v < K_SIZE / 8; v += static_cast<int>(blockDim.x)) {
    int off = v * 8;
    uint64_t g_lo = *reinterpret_cast<uint64_t const *>(gate + off);
    uint64_t g_hi = *reinterpret_cast<uint64_t const *>(gate + off + 4);
    uint64_t u_lo = *reinterpret_cast<uint64_t const *>(up + off);
    uint64_t u_hi = *reinterpret_cast<uint64_t const *>(up + off + 4);
    uint16_t const *g = reinterpret_cast<uint16_t const *>(&g_lo);
    uint16_t const *u = reinterpret_cast<uint16_t const *>(&u_lo);
    uint16_t const *g2 = reinterpret_cast<uint16_t const *>(&g_hi);
    uint16_t const *u2 = reinterpret_cast<uint16_t const *>(&u_hi);
    uint16_t out_arr[8];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      out_arr[k] = _gang_moe_w2_bf16(fast_silu(__uint_as_float((unsigned)g[k] << 16)) *
                                     __uint_as_float((unsigned)u[k] << 16));
      out_arr[4 + k] = _gang_moe_w2_bf16(fast_silu(__uint_as_float((unsigned)g2[k] << 16)) *
                                         __uint_as_float((unsigned)u2[k] << 16));
    }
    *reinterpret_cast<uint64_t *>(act + off) = *reinterpret_cast<uint64_t *>(&out_arr[0]);
    *reinterpret_cast<uint64_t *>(act + off + 4) = *reinterpret_cast<uint64_t *>(&out_arr[4]);
  }
}

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
  static_assert(BATCH_SIZE == 1, "the prologue computes one token's row per tile");
  static_assert(IN_STRIDE >= 2 * REDUCTION_SIZE, "mid holds gate then up per slot");
  static_assert(REDUCTION_SIZE % 8 == 0, "16-byte vectors of 8 BF16");

  constexpr int NPerBlock = 64;     // the tile's rows, the stock kernel's N tile

  // --- the decode, the stock kernel's (both paths) ---
  int xcd_id = _gang_moe_w2_xcd_id();
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

  T const *__restrict__ d_mid = static_cast<T const *>(mid_ptr);
  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);
  T *__restrict__ d_scratch = static_cast<T *>(scratch_ptr);
  int const *__restrict__ d_routing = static_cast<int const *>(routing_ptr);
  int const *expert_routing = d_routing + expert_id * BATCH_SIZE;
  int const route_val = expert_routing[w2_tok];
  if (route_val == 0) return;
  int const topk_slot = route_val - 1;

  T const *gate = d_mid + static_cast<size_t>(w2_tok) * (NUM_TOPK * IN_STRIDE) +
                  static_cast<size_t>(topk_slot) * IN_STRIDE;
  T const *up = gate + REDUCTION_SIZE;
  int n_offset = n_tile * NPerBlock;
  extern __shared__ char smem[];

#ifndef MPK_W2_CK_TILE
  // --- the prologue into LDS, then the GEMV over the tile's rows ---
  using namespace dsv2;
  (void)d_scratch;                                       // the scratch output is not written
  T *act_s = reinterpret_cast<T *>(smem);                // [REDUCTION_SIZE] BF16
  _gang_moe_w2_silu_row<REDUCTION_SIZE>(reinterpret_cast<uint16_t const *>(gate),
                                        reinterpret_cast<uint16_t const *>(up),
                                        reinterpret_cast<uint16_t *>(act_s));
  __syncthreads();

  constexpr int ROWS_PER_WAVE = NPerBlock / WAVES;          // 16
  constexpr int CHUNKS = REDUCTION_SIZE / 8;                // 176 at K = 1,408
  constexpr int CPL = (CHUNKS + WAVE - 1) / WAVE;           // 3; the last lanes own fewer
  static_assert(W2_BATCH >= 1 && (W2_BATCH & (W2_BATCH - 1)) == 0, "the butterfly wants a power of two");
  static_assert(ROWS_PER_WAVE % W2_BATCH == 0, "the wave's rows split into whole batches");

  int wave = static_cast<int>(threadIdx.x) / WAVE;
  int lane = static_cast<int>(threadIdx.x) % WAVE;

  // the lane's slice of the row, 8 FP32 per chunk, read once; the chunks a lane
  // does not own stay zero, against the zeros of their weight words
  float xv[CPL * 8];
#pragma unroll
  for (int j = 0; j < CPL; j++) {
    int chunk = lane + WAVE * j;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      xv[8 * j + k] = 0.0f;
    }
    if (chunk < CHUNKS) {
      load8(act_s + chunk * 8, xv + 8 * j);
    }
  }

  StreamSrc<T> w_src(d_weight + static_cast<size_t>(expert_id) * OUTPUT_STRIDE * REDUCTION_SIZE);
  int row_base = n_offset + wave * ROWS_PER_WAVE;
  T *out_row = d_output + static_cast<size_t>(w2_tok) * (NUM_TOPK * OUTPUT_STRIDE) +
               static_cast<size_t>(topk_slot) * OUTPUT_STRIDE;
#pragma unroll 1
  for (int r0 = 0; r0 < ROWS_PER_WAVE; r0 += W2_BATCH) {
    uint4 raw[W2_BATCH][CPL];
#pragma unroll
    for (int u = 0; u < W2_BATCH; u++) {
      // a row past the last column of a short tile is read as the last row and
      // its result dropped by the store's guard, so the loads stay in range
      int n = row_base + r0 + u;
      size_t row_elem = static_cast<size_t>(n < OUTPUT_SIZE ? n : OUTPUT_SIZE - 1) * REDUCTION_SIZE;
#pragma unroll
      for (int j = 0; j < CPL; j++) {
        int chunk = lane + WAVE * j;
        raw[u][j].x = 0u;
        raw[u][j].y = 0u;
        raw[u][j].z = 0u;
        raw[u][j].w = 0u;
        if (chunk < CHUNKS) {
          raw[u][j] = load16_from(w_src, row_elem + static_cast<size_t>(chunk) * 8);
        }
      }
    }
    float acc[W2_BATCH];
#pragma unroll
    for (int u = 0; u < W2_BATCH; u++) {
      float a = 0.0f;
#pragma unroll
      for (int j = 0; j < CPL; j++) {
        unsigned const *words = reinterpret_cast<unsigned const *>(&raw[u][j]);
#pragma unroll
        for (int k = 0; k < 8; k++) {
          unsigned bits = (k & 1) ? (words[k / 2] & 0xffff0000u) : (words[k / 2] << 16);
          a += xv[8 * j + k] * __uint_as_float(bits);
        }
      }
      acc[u] = a;
    }
    butterfly_sum<W2_BATCH>(acc);        // lane l < W2_BATCH now holds row r0 + l
    int n = row_base + r0 + lane;
    if (lane < W2_BATCH && n < OUTPUT_SIZE) {
      st(out_row + n, bf16r(acc[0]));
    }
  }
#else
  // --- round 3's form: the prologue into the tile's scratch row, then CK ---
  using namespace ck_tile;
  constexpr index_t MPerBlock = 16;
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

  // the scratch row of this (XCD, tile): the runtime's tile_idx for a MoE gang
  // task is XCD-local (n_tile_start = 0) and the XCD comes from the hardware
  // register, as in the stock kernel
  size_t act_row = static_cast<size_t>(xcd_id) * TOTAL_TILES_PER_XCD + static_cast<size_t>(tile_idx);
  _gang_moe_w2_silu_row<REDUCTION_SIZE>(reinterpret_cast<uint16_t const *>(gate),
                                        reinterpret_cast<uint16_t const *>(up),
                                        reinterpret_cast<uint16_t *>(d_scratch) + act_row * REDUCTION_SIZE);
  // the stores have reached L2 before any thread's pipeline loads the row: the explicit
  // s_waitcnt 0 (vmcnt, expcnt, lgkmcnt) is added because the workgroup-scope release alone
  // lowers to lgkmcnt(0) and the barrier outside threadgroup-split mode (the LLVM AMDGPU
  // memory model relies on the CU's in-order vector memory pipeline; the wait makes the
  // store's completion explicit at the cost of one wait per tile)
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
  __syncthreads();

  // the scratch row as A (sc0: the L1 is bypassed, so a line an earlier task left
  // in this CU's L1 cannot be read; the offline disassembly shows the bit)
  bf16 const *a_base = reinterpret_cast<bf16 const *>(d_scratch) + act_row * REDUCTION_SIZE;
  bf16 const *expert_weight =
      reinterpret_cast<bf16 const *>(d_weight) + static_cast<int64_t>(expert_id) * OUTPUT_STRIDE * REDUCTION_SIZE;
  bf16 const *b_base = expert_weight + static_cast<size_t>(n_offset) * REDUCTION_SIZE;
  index_t n_size = (n_offset + NPerBlock <= OUTPUT_SIZE) ? NPerBlock : (OUTPUT_SIZE - n_offset);

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
  bf16 *out8 = reinterpret_cast<bf16 *>(d_output);
  if (tile_row == 0) {
    if (global_n_base + 3 < OUTPUT_SIZE) {
      bf16 *out_addr = out8 + static_cast<size_t>(w2_tok) * (NUM_TOPK * OUTPUT_STRIDE) +
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
          out8[static_cast<size_t>(w2_tok) * (NUM_TOPK * OUTPUT_STRIDE) +
               static_cast<size_t>(topk_slot) * OUTPUT_STRIDE + global_n] = type_convert<bf16>(c_buf[i]);
        }
      }
    }
  }
#endif
}

} // namespace kernel
