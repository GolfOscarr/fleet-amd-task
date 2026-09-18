/* gang_moe_w13_gemv_mi300: the expert gate-up projection as the GEMV loop of
 * round 4, in one round per XCD (docs/gpu-experiments/04-kernels/01-gemv-ideas.md,
 * K8 MOE, K9 and S1; 05-local-preparation.md, L4). A replacement for the
 * runtime's gang_moe_w13_linear_kernel (tasks/mi300/gang_moe_linear_mi300.cuh)
 * with the same decode and the same scatter, but 37 tiles per expert instead of
 * 44: the gang loop hands tile t to the worker of rank t mod 37
 * (persistent_kernel.cuh), so with 37 tiles every one of the XCD's 37 workers
 * holds exactly one tile of about 305 KB and the operator ends when the XCD's
 * stream ends, where 44 tiles are a second round for seven of them (about 10 us
 * per layer in round 3's record).
 *
 * Inputs : h [1, K] BF16 (the whole row: the router's normalised row),
 *          W13 [E_TOTAL, N, K] BF16 (gate rows [0, N/2), up rows [N/2, N)),
 *          routing [E_TOTAL, 1] int32, mask [E_TOTAL + 1] int32
 * Outputs: mid [1, NUM_TOPK, N] BF16 (the slot's row; gate | up, the layout
 *          the silu-mul and the fused w2 prologue read)
 * Params (register_task): the stock w13's [tiles_per_expert, max_experts_per_xcd,
 *          total_tiles_per_xcd] with tiles_per_expert = TILES (37).
 * LDS    : none. No prologue, no scratch row, no barrier.
 *
 * The decode is the stock kernel's, copied through gang_moe_w2_silu_mi300.cuh:
 * mask[NUM_EXPERTS] is the active expert count, ae_idx = xcd + 8 *
 * expert_local_idx picks the XCD's expert (one per XCD at the eight active
 * experts of this model), and the routing entry of that expert gives the slot.
 * The gang loop hands every worker max_experts_per_xcd tiles (9 at 66 experts),
 * so the kernel returns at once for every local expert index past the XCD's one
 * active expert, exactly as the stock w13 returns for the tiles past its own.
 *
 * The tile's rows by arithmetic, no table (S1): 2,816 = 4 x 77 + 33 x 76, so
 * tile t < 4 starts at 77 t with 77 rows and tile t >= 4 at 308 + 76 (t - 4)
 * with 76. Wave w owns rows [w * RPW, min((w + 1) * RPW, rows)) of the tile with
 * RPW = ceil(rows / 4), the map of linear_gemv_mi300.cuh: 19 rows each in a
 * 76-row tile and 20, 20, 20, 17 in a 77-row one (the page's 19, 19, 19, 20 is
 * the same rows in another split; this one is the GEMV's own formula).
 *
 * The multiply (K1, K9). A wave walks its rows in batches of W13_BATCH (the
 * macro, default 8) under "#pragma unroll 1": the batch constant is the depth
 * only under the pragma, the round's first convention. K = 2,048 is 4 chunks of
 * eight elements per lane at 8 * lane + 512 * i, so one wave-load is one
 * contiguous kilobyte of the row (8 full 128-byte lines requested, nothing
 * re-fetched by the next three loads) and a row is four 16-byte loads per lane
 * through load16_from on a StreamSrc (the sc1 nt policy of the linears under
 * MLA_NT_STREAMS, plain loads without it): a batch of eight rows is 32 loads in
 * flight per lane, 32 KB per wave and 128 KB per CU. The x slice is the same 32
 * elements per lane of h, four load8 from global, read once before the loop; no
 * prologue precedes the first batch (the round's third convention: a batch live
 * across a prologue and reloaded inside the loop is not coalesced with the
 * loop's own by this compiler; the pre-load forms were measured and dropped).
 * The arguments are made wave-uniform on entry (uniform_ptr and uniform_int of
 * mla_common_mi300.cuh): a __noinline__ kernel's arguments arrive in VGPRs, and
 * a buffer resource built from a VGPR pointer costs every weight load a
 * v_readfirstlane waterfall loop. The FMAs run on the raw words in ascending k within each
 * chunk, as the dense GEMV's loop; butterfly_sum<W13_BATCH> then leaves the
 * total of row r0 + l on lane l, and lane l < W13_BATCH stores one BF16 element
 * of mid at the stock scatter's address.
 *
 * Registers: the batch's raw words (W13_BATCH x 4 x 4 = 128 VGPRs at eight
 * rows), the 32 FP32 values of the x slice and W13_BATCH row sums, about 168
 * live; no LDS and no scratch tensor.
 *
 * Rows past the wave's range are loaded as the range's last row (a clamped
 * index, so the batch stays one basic block and its loads in flight together)
 * and their sums are dropped: only the stores are masked, and they are 2-byte
 * stores, as a tile's rows start at an odd multiple of 76 or 77.
 *
 * Numerics: every product is a BF16 times a BF16, exact in FP32; the
 * accumulation order is a lane's chain of 32 products and then the butterfly's
 * tree over the 64 lanes, which differs from the CK tile's MFMA order as both
 * differ from NumPy's. That is the FP32 reassociation the boundary compare of
 * mid (B10) already tolerates on this path.
 */
#pragma once
#include "tasks/common/common_header.cuh"
#include "tasks/mi300/mla_common_mi300.cuh"   // StreamSrc, load16_from, load8, butterfly_sum, bf16r, st

// The rows of one batch, in flight per lane at four 16-byte loads each (M5
// sweeps it: 4, 8, 16). A power of two up to 64, the butterfly's requirement.
#ifndef W13_BATCH
#define W13_BATCH 8
#endif

namespace kernel {

// The XCD of this workgroup: _gang_moe_get_xcd_id of the stock gang kernel
// (tasks/mi300/gang_moe_linear_mi300.cuh), copied here so this file needs no CK
// header. -DKT_FAKE_XCD takes it from bid.y instead, which is what gives the
// kernel suite a deterministic (tile, XCD) launch; on the host syntax check,
// which has no hardware register, it is zero.
__device__ __forceinline__ int _gang_moe_w13_xcd_id() {
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

template <typename T,
          int N = 2816,            // the expert's gate|up rows
          int K = 2048,            // the reduction, h's width
          int NUM_EXPERTS = 66,
          int NUM_TOPK = 8,
          int TILES = 37>          // tiles per expert: the XCD's worker count (S1)
__device__ __noinline__ void
    gang_moe_w13_gemv_kernel(void const *input_ptr,
                             void const *weight_ptr,
                             void const *routing_ptr,
                             void const *mask_ptr,
                             void *output_ptr,
                             int tile_idx) {
  using namespace dsv2;
  // the tile's arguments, identical across the wave, made scalar (see the header)
  input_ptr = uniform_ptr(input_ptr);
  weight_ptr = uniform_ptr(weight_ptr);
  routing_ptr = uniform_ptr(routing_ptr);
  mask_ptr = uniform_ptr(mask_ptr);
  output_ptr = uniform_ptr(output_ptr);
  tile_idx = uniform_int(tile_idx);
  constexpr int BATCH = W13_BATCH;
  constexpr int BIG_TILES = 4, BIG_ROWS = 77, SMALL_ROWS = 76;
  static_assert(TILES == 37 && BIG_TILES * BIG_ROWS + (TILES - BIG_TILES) * SMALL_ROWS == N,
                "the 37 tiles partition N exactly: 4 x 77 + 33 x 76 == 2816 (S1)");
  static_assert(K % (8 * WAVE) == 0, "16-byte loads, K / 64 elements per lane");
  static_assert(K <= 4096, "a lane's K slice lives in registers (K / 64 values)");
  static_assert(BATCH >= 1 && BATCH <= WAVE && (BATCH & (BATCH - 1)) == 0,
                "the butterfly reduces a power-of-two batch of rows");
  constexpr int CHUNKS = K / (8 * WAVE);        // 4 sixteen-byte loads per row per lane at K = 2048
  constexpr int OUTPUT_STRIDE = N;              // mid [1, NUM_TOPK, N]: the slot stride is the row

  // --- the decode, the stock w13 kernel's ---
  int xcd_id = _gang_moe_w13_xcd_id();
  int const *__restrict__ d_mask = static_cast<int const *>(mask_ptr);
  int const num_activated_experts = d_mask[NUM_EXPERTS];
  int expert_local_idx = tile_idx / TILES;
  int n_tile = tile_idx % TILES;                // batch 1: the tiles of an expert are its N tiles
  int ae_idx = xcd_id + expert_local_idx * 8;
  if (ae_idx >= num_activated_experts) {
    return;                                     // the local expert indices past this XCD's one active expert
  }
  int expert_id = d_mask[ae_idx];
  int const *__restrict__ d_routing = static_cast<int const *>(routing_ptr);
  int const route_val = d_routing[expert_id];   // routing [E_TOTAL, 1]: the token's entry, batch 1
  if (route_val == 0) {
    return;                                     // the expert routes no slot of this token
  }
  int const topk_slot = route_val - 1;

  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

  // --- the tile's rows, by arithmetic (S1) ---
  int row0 = n_tile < BIG_TILES ? BIG_ROWS * n_tile
                                : BIG_TILES * BIG_ROWS + SMALL_ROWS * (n_tile - BIG_TILES);
  int tile_rows = n_tile < BIG_TILES ? BIG_ROWS : SMALL_ROWS;
  int wave = static_cast<int>(threadIdx.x) / WAVE;
  int lane = static_cast<int>(threadIdx.x) % WAVE;
  int rpw = (tile_rows + WAVES - 1) / WAVES;
  int r_begin = wave * rpw;
  int r_end = (wave + 1) * rpw < tile_rows ? (wave + 1) * rpw : tile_rows;
  if (r_end < r_begin) {
    r_end = r_begin;                            // a wave past the tile's rows (never at 76 or 77)
  }

  // the lane's slice of h, 32 FP32 values at the coalesced chunks, read once
  float xv[CHUNKS][8];
#pragma unroll
  for (int i = 0; i < CHUNKS; i++) {
    load8(d_input + 8 * lane + 8 * WAVE * i, xv[i]);
  }

  StreamSrc<T> w_src(d_weight + static_cast<size_t>(expert_id) * N * K);
  // the stock scatter's address, with the token index 0: one token per tile at batch 1
  constexpr int w2_tok = 0;
  T *out_row = d_output + static_cast<size_t>(w2_tok) * (NUM_TOPK * OUTPUT_STRIDE) +
               static_cast<size_t>(topk_slot) * OUTPUT_STRIDE;
#pragma unroll 1
  for (int r0 = r_begin; r0 < r_end; r0 += BATCH) {
    // the batch's loads, all issued before any of them is waited on; a row past
    // the wave's range loads the range's last row again (a clamped index, no
    // branch: a branch per row puts each row's loads in its own basic block and
    // the scheduler keeps only that block's in flight), its sum dropped below
    uint4 raw[BATCH][CHUNKS];
#pragma unroll
    for (int u = 0; u < BATCH; u++) {
      int r = (r0 + u < r_end) ? r0 + u : r_end - 1;
      size_t row_elem = static_cast<size_t>(row0 + r) * K;
#pragma unroll
      for (int i = 0; i < CHUNKS; i++) {
        raw[u][i] = load16_from(w_src, row_elem + 8 * lane + 8 * WAVE * i);
      }
    }
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
    butterfly_sum<BATCH>(sums);                 // lane l < BATCH now holds row r0 + l's total
    int r = r0 + lane;
    if (lane < BATCH && r < r_end) {
      st(out_row + row0 + r, bf16r(sums[0]));   // 2 bytes: a tile's rows are not 16-byte aligned
    }
  }
}

} // namespace kernel
