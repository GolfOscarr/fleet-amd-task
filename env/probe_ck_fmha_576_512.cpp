// Day-1 probe (docs/design-doc/99-open-questions.md DQ3, docs/fleet Q11):
// does the machine's ck_tile instantiate the split-KV FMHA decode pipeline at
// the MLA head dimensions, QK head dim 576 (512 latent + 64 rope) and V head
// dim 512 (the latent)? If this file passes `hipcc --offload-arch=gfx942
// -fsyntax-only`, mla_attend can be a wrapper around DecodePipeline
// (00-decisions.md D12); otherwise it is the spec kernel.
//
// Modelled on repos/fleet-chiplet-megakernel/include/mirage/persistent_kernel/
// tasks/mi300/paged_attention_ck_fmha_split_kv_mi300.cuh: the same includes,
// traits, variant and problem, with the tile sequence changed from
// <16, 128, 32, 128, 32, 128> = (kM0, kN0, kK0, kN1, kK1, kQKHeaddim) to
// (16, 32, 32, 512, 32, 576): 16 query rows (the 16 heads), 32 keys per
// tile, K-step 32 for QK, V head dim 512, K-step 32 for PV, QK head dim 576.
//
//   hipcc --offload-arch=gfx942 -fsyntax-only -std=c++17 -DCK_TILE_FMHA_FWD_FAST_EXP2=1 \
//     -I<ck_tile include root> env/probe_ck_fmha_576_512.cpp

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_nwarp_sshuffle_qr_ks_vs.hpp"
#include "ck_tile/ops/fmha/block/block_masking.hpp"
#include "ck_tile/ops/fmha/block/block_position_encoding.hpp"
#include "ck_tile/ops/fmha/block/variants.hpp"

#ifndef CK_TILE_FMHA_FWD_FAST_EXP2
#define CK_TILE_FMHA_FWD_FAST_EXP2 1
#endif

namespace probe {

// (kM0, kN0, kK0, kN1, kK1, kQKHeaddim)
using BlockTileMla = ck_tile::sequence<16, 32, 32, 512, 32, 576>;
using Gemm0BlockWarps = ck_tile::sequence<1, 4, 1>;
using Gemm0WarpTile = ck_tile::sequence<16, 16, 16>;
using Gemm1BlockWarps = ck_tile::sequence<1, 4, 1>;
using Gemm1WarpTile = ck_tile::sequence<16, 16, 16>;

using ShapeMla = ck_tile::TileFmhaShape<BlockTileMla, Gemm0BlockWarps, Gemm0WarpTile,
                                        Gemm1BlockWarps, Gemm1WarpTile, true /* VRowMajor */>;

using Variant = ck_tile::ComposedAttention<0, CK_TILE_FMHA_FWD_FAST_EXP2>;

using DecodeTraits = ck_tile::TileFmhaFwdSplitKVTraits<
    true, true, false, false, false,
    ck_tile::BlockAttentionBiasEnum::NO_BIAS,
    false, true, false, true, true,
    true,   // kMergeNumHeadGroupsSeqLenQ
    -1, false>;
using DecodeMask = ck_tile::SimplifiedGenericAttentionMask<false>;

using ProblemMla = ck_tile::BlockFmhaFwdSplitKVPipelineProblem<
    ck_tile::bf16_t, ck_tile::bf16_t, ck_tile::bf16_t,
    float, float, ck_tile::bf16_t, float, ck_tile::bf16_t,
    float, float,
    ShapeMla, true, Variant, DecodeMask, DecodeTraits>;

using PipelineMla = ck_tile::BlockFmhaFwdSplitKVPipelineNWarpSShuffleQRKSVS<ProblemMla>;

// The worker's dynamic LDS budget: 60 KiB minus the runtime's 3 KiB
// (docs/design-doc/04-memory-plan.md). A failure here is informative too.
#ifndef PROBE_SMEM_BUDGET
#define PROBE_SMEM_BUDGET (57 * 1024)
#endif
constexpr int kMaxDynamicSmem = PROBE_SMEM_BUDGET;
constexpr int kSmem = PipelineMla::GetSmemSize();
static_assert(kSmem > 0, "pipeline reports no LDS");
static_assert(kSmem <= kMaxDynamicSmem,
              "CK FMHA MLA pipeline LDS exceeds the 57 KiB worker budget (see the value in the error)");

// Force instantiation of the pipeline's call operator by referencing it.
template <typename P>
__device__ void touch() {
  (void)P::GetSmemSize();
  (void)P::kM0;
  (void)P::kN0;
  (void)P::kN1;
  (void)P::kQKHeaddim;
}
__global__ void probe_kernel() { touch<PipelineMla>(); }

} // namespace probe

// To read the LDS size the pipeline reports, compile once with the budget
// lowered (-DPROBE_SMEM_BUDGET=1): the static_assert then fails and the
// message names the constant; its value is in `kSmem` in the AST dump.
