// Offline gfx942 compile of the patched megakernel headers plus the two
// dispatchers the runtime's code generator emits (runtime.cc:1562-1640),
// written here by hand for the five new tasks with the call code that
// task_register.cc emits for them. Linking the device object therefore
// exercises the worker kernel's register union with our kernels (MAJ-4).
// Compile flags: persistent_kernel.py for mode=online, batch 1, USE_GANG=1.
#include <vector>
#include <thread>
#include <unistd.h>
#include "persistent_kernel.cuh"
#ifdef MK_GEMV
// L1c (docs/gpu-experiments/04-kernels/05-local-preparation.md): the GEMV linear's three forms
// at the model's dims, on the task type L2's patch adds (TASK_LINEAR_GEMV_MI300 = 195). With
// new_tasks.patch applied the enum carries the type and task_header.cuh includes the kernel, so
// this aliases the real enum value; the include stays (the header is #pragma once, and it lets
// the block still stand as the standalone check on an unpatched tree).
#include "tasks/mi300/linear_gemv_mi300.cuh"
#define MK_TASK_LINEAR_GEMV_MI300 TASK_LINEAR_GEMV_MI300
// L6: the stream probe beside it (TASK_STREAM_MI300 = 197 regular, TASK_STREAM_GANG_MI300 = 206
// gang; the gang form took 206 because 200 to 203 are the fork's scheduler task types)
#include "tasks/mi300/stream_mi300.cuh"
#define MK_TASK_STREAM_MI300 TASK_STREAM_MI300
#define MK_TASK_STREAM_GANG_MI300 TASK_STREAM_GANG_MI300
// L4: the expert gate-up as the GEMV gang kernel (TASK_GANG_MOE_W13_GEMV_MI300 = 196)
#include "tasks/mi300/gang_moe_w13_gemv_mi300.cuh"
#define MK_TASK_GANG_MOE_W13_GEMV_MI300 TASK_GANG_MOE_W13_GEMV_MI300
// N4: the merge as regular tasks (TASK_MLA_MERGE_UV_TILE_MI300 = 205). The kernel's header is
// the gang form's, which task_header.cuh already includes.
#define MK_TASK_MLA_MERGE_UV_TILE_MI300 TASK_MLA_MERGE_UV_TILE_MI300
// N2: the fused router in four tasks (TASK_MOE_ROUTER4_MI300 = 204); its kernel header is the
// one-task router's, already included by task_header.cuh
#define MK_TASK_MOE_ROUTER4_MI300 TASK_MOE_ROUTER4_MI300
#endif

using namespace mirage::runtime;

__device__ __forceinline__
void _execute_task(TaskDesc const *task_desc, RuntimeConfig const &runtime_config) {
  if (task_desc->task_type == TASK_MLA_PREP_MI300 && task_desc->variant_id == 0) {
    kernel::mla_prep_mi300_task_impl<bfloat16, 16, 128, 64, 512>(
        task_desc->input_ptrs[0], task_desc->input_ptrs[1], task_desc->input_ptrs[2],
        task_desc->input_ptrs[3], task_desc->input_ptrs[4],
        task_desc->output_ptrs[0], task_desc->output_ptrs[1], task_desc->output_ptrs[2],
        task_desc->output_ptrs[3], runtime_config.step[0], 1e-6f,
        (int)(task_desc->task_metadata.expert_offset & 0xFFFF));
  } else if (task_desc->task_type == TASK_MOE_ROUTER_MI300 && task_desc->variant_id == 0) {
    kernel::moe_router_mi300_task_impl<bfloat16, 2048, 64, 2, 6, 32, 26, false>(
        task_desc->input_ptrs[0], nullptr, task_desc->input_ptrs[1], nullptr,
        task_desc->output_ptrs[0], task_desc->output_ptrs[1], task_desc->output_ptrs[2],
        task_desc->output_ptrs[3], task_desc->output_ptrs[4],
        runtime_config.step[0], runtime_config.prompt_length[0], 1, 1.0f, 0.0f);
  } else if (task_desc->task_type == TASK_MOE_ROUTER_MI300 && task_desc->variant_id == 1) {
    // moe_router_norm_mi300 (O1): the norm folded in; in a real graph it is variant 0 of its own build
    kernel::moe_router_mi300_task_impl<bfloat16, 2048, 64, 2, 6, 32, 26, true>(
        task_desc->input_ptrs[0], task_desc->input_ptrs[1], task_desc->input_ptrs[2], task_desc->output_ptrs[0],
        task_desc->output_ptrs[1], task_desc->output_ptrs[2], task_desc->output_ptrs[3],
        task_desc->output_ptrs[4], task_desc->output_ptrs[5],
        runtime_config.step[0], runtime_config.prompt_length[0], 1, 1.0f, 1e-6f);
  } else if (task_desc->task_type == TASK_MLA_ATTEND_TILE_MI300 && task_desc->variant_id == 0) {
    // --attend-tasks (session B): one regular task per split, tiles_per_xcd 1, the split from bid.x
    kernel::mla_attend_mi300_task_impl<bfloat16, 16, 512, 64, 1056>(
        task_desc->input_ptrs[0], task_desc->input_ptrs[1], task_desc->input_ptrs[2],
        task_desc->input_ptrs[3], task_desc->output_ptrs[0], runtime_config.step[0],
        0.1147213867929261f, 32, 33, 1, 0, (int)(task_desc->task_metadata.expert_offset & 0xFFFF),
#ifdef MLA_ATTEND_DEBUG_SCORES
        task_desc->output_ptrs[1]);
#else
        nullptr);
#endif
  } else if (task_desc->task_type == TASK_COPY_MI300 && task_desc->variant_id == 0) {
    kernel::copy_mi300_task_impl<bfloat16, 2048>(task_desc->input_ptrs[0], task_desc->output_ptrs[0]);
  } else if (task_desc->task_type == TASK_COPY_MI300 && task_desc->variant_id == 1) {
    // I2: the spin mode of the copy (the empty ladder's rows with --spin)
    kernel::copy_mi300_task_impl<bfloat16, 256>(task_desc->input_ptrs[0], task_desc->output_ptrs[0], 1000, 1);
  } else if (task_desc->task_type == TASK_PREFETCH_MI300 && task_desc->variant_id == 0) {
    // O8: the side operators' prefetch tasks: W_o in 64 stripes of 32 rows; W2's active experts in 32 parts
    kernel::prefetch_mi300_task_impl<bfloat16, 32, 2048>(task_desc->input_ptrs[0], task_desc->output_ptrs[0]);
  } else if (task_desc->task_type == TASK_PREFETCH_MOE_MI300 && task_desc->variant_id == 0) {
    kernel::prefetch_moe_mi300_task_impl<bfloat16, 66, 2048, 1408, 32>(
        task_desc->input_ptrs[0], task_desc->input_ptrs[1], task_desc->output_ptrs[0],
        (int)(task_desc->task_metadata.expert_offset & 0xFFFF));
  }
#ifdef MK_GEMV
  // L1c: qkva with the norm (38 rows of 3,648, 96 tasks), o_proj with the residual (32 rows of
  // 2,048, 64 tasks), the head with the final norm (256 rows of 102,400, 400 tasks)
  else if (task_desc->task_type == MK_TASK_LINEAR_GEMV_MI300 && task_desc->variant_id == 0) {
    kernel::linear_gemv_mi300_task_impl<bfloat16, 2048, true, false>(
        task_desc->input_ptrs[0], task_desc->input_ptrs[1], task_desc->input_ptrs[2], nullptr,
        task_desc->output_ptrs[0], 38, 3648, 1e-6f);
#ifndef MK_GEMV_ONE
  } else if (task_desc->task_type == MK_TASK_LINEAR_GEMV_MI300 && task_desc->variant_id == 1) {
    kernel::linear_gemv_mi300_task_impl<bfloat16, 2048, false, true>(
        task_desc->input_ptrs[0], nullptr, task_desc->input_ptrs[1], task_desc->input_ptrs[2],
        task_desc->output_ptrs[0], 32, 2048, 0.0f);
  } else if (task_desc->task_type == MK_TASK_LINEAR_GEMV_MI300 && task_desc->variant_id == 2) {
    kernel::linear_gemv_mi300_task_impl<bfloat16, 2048, true, false>(
        task_desc->input_ptrs[0], task_desc->input_ptrs[1], task_desc->input_ptrs[2], nullptr,
        task_desc->output_ptrs[0], 256, 102400, 1e-6f);
  } else if (task_desc->task_type == MK_TASK_MLA_MERGE_UV_TILE_MI300 && task_desc->variant_id == 0) {
    // N4, --merge-tasks: 16 regular tasks of a whole head, every tensor whole, the task index
    // from expert_offset (the emitted call of fleet/patches/hunks/N4-merge-tile.md)
    kernel::mla_merge_uv_tile_mi300_task_impl<bfloat16, 16, 128, 512, 1>(
        task_desc->input_ptrs[0], task_desc->input_ptrs[1], task_desc->output_ptrs[0],
        runtime_config.step[0], 32, 33,
        (int)(task_desc->task_metadata.expert_offset & 0xFFFF));
  } else if (task_desc->task_type == MK_TASK_MLA_MERGE_UV_TILE_MI300 && task_desc->variant_id == 1) {
    // --merge-halves 2: 32 tasks of a half head (the W_uv rows 64 half .. 64 half + 63)
    kernel::mla_merge_uv_tile_mi300_task_impl<bfloat16, 16, 128, 512, 2>(
        task_desc->input_ptrs[0], task_desc->input_ptrs[1], task_desc->output_ptrs[0],
        runtime_config.step[0], 32, 33,
        (int)(task_desc->task_metadata.expert_offset & 0xFFFF));
  } else if (task_desc->task_type == MK_TASK_MOE_ROUTER4_MI300 && task_desc->variant_id == 0) {
    // N2, --router-tasks: the fused router's GEMV over four tasks, the part from expert_offset
    // and the counter as input 3 (the emitted call of fleet/patches/hunks/N2-router4.md)
    kernel::moe_router_mi300_task_impl<bfloat16, 2048, 64, 2, 6, 32, 26, true, 4>(
        task_desc->input_ptrs[0], task_desc->input_ptrs[1], task_desc->input_ptrs[2],
        task_desc->output_ptrs[0], task_desc->output_ptrs[1], task_desc->output_ptrs[2],
        task_desc->output_ptrs[3], task_desc->output_ptrs[4], task_desc->output_ptrs[5],
        runtime_config.step[0], runtime_config.prompt_length[0], 1, 1.0f, 1e-6f,
        (int)(task_desc->task_metadata.expert_offset & 0xFFFF), task_desc->input_ptrs[3]);
#endif
  }
  // L6: the stream probe's regular form, the two rows of G5 (152 KB over 96 tasks, qkva's shape,
  // and 256 KB over 296, one task per CU); the second is variant 1 as the registration emits it
  else if (task_desc->task_type == MK_TASK_STREAM_MI300 && task_desc->variant_id == 0) {
    kernel::stream_mi300_task_impl<bfloat16, 2048>(
        task_desc->input_ptrs[0], task_desc->output_ptrs[0], 38);
#ifndef MK_GEMV_ONE
  } else if (task_desc->task_type == MK_TASK_STREAM_MI300 && task_desc->variant_id == 1) {
    kernel::stream_mi300_task_impl<bfloat16, 2048>(
        task_desc->input_ptrs[0], task_desc->output_ptrs[0], 64);
#endif
  }
#endif
#ifdef MK_CK_LINEAR
  // O3: the per-tile linear with the norm prologue at the model's dims (batch 1, K 2048); the
  // qkva registration emits output_size 38 and stride 3648 (96 tasks), lm_head 256 and 102400
  // (400 tasks); the dispatcher keys on type and variant, so both appear as variants 0 and 1.
  else if (task_desc->task_type == TASK_LINEAR_NORM_MI300 && task_desc->variant_id == 0) {
    kernel::linear_norm_mi300_task_impl<bfloat16, 1, 2048>(
        task_desc->input_ptrs[0], task_desc->input_ptrs[1], task_desc->input_ptrs[2],
        task_desc->output_ptrs[0], task_desc->output_ptrs[1],
        runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS], 38, 3648, 1e-6f);
  } else if (task_desc->task_type == TASK_LINEAR_NORM_MI300 && task_desc->variant_id == 1) {
    kernel::linear_norm_mi300_task_impl<bfloat16, 1, 2048>(
        task_desc->input_ptrs[0], task_desc->input_ptrs[1], task_desc->input_ptrs[2],
        task_desc->output_ptrs[0], task_desc->output_ptrs[1],
        runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS], 256, 102400, 1e-6f);
  }
#endif
}

__device__ __forceinline__
void _execute_gang_task(TaskDesc const *task_desc, RuntimeConfig const &runtime_config, int tile_idx) {
#ifdef MK_CK_GANG
  // O2: the fused w2 gang task at the model's dims (batch 1, N 2048, K 1408, mid stride 2816,
  // 66 experts, 8 slots, 32 tiles per expert, 32 N tiles, 32 tiles per XCD). This instantiates
  // CK's small-tile GEMM pipeline for gfx942 offline, which the day-1 build had done on the VM only.
  if (task_desc->task_type == TASK_GANG_MOE_W2_SILU_MI300 && task_desc->variant_id == 0) {
    kernel::gang_moe_w2_silu_linear_kernel<bfloat16, 1, 2048, 2048, 1408, 2816, 66, 8, 32, 32, 32>(
        task_desc->input_ptrs[0], task_desc->input_ptrs[1], task_desc->input_ptrs[2], task_desc->input_ptrs[3],
        task_desc->output_ptrs[0], task_desc->output_ptrs[1], tile_idx);
  } else
#endif
#ifdef MK_GEMV
  // L4: the expert gate-up at the model's dims (N 2,816, K 2,048, 66 experts, 8 slots, 37 tiles
  // per expert), the call the registration of fleet/patches/hunks/L4-w13-gemv.md emits
  if (task_desc->task_type == MK_TASK_GANG_MOE_W13_GEMV_MI300 && task_desc->variant_id == 0) {
    kernel::gang_moe_w13_gemv_kernel<bfloat16, 2816, 2048, 66, 8, 37>(
        task_desc->input_ptrs[0], task_desc->input_ptrs[1], task_desc->input_ptrs[2],
        task_desc->input_ptrs[3], task_desc->output_ptrs[0], tile_idx);
  } else
#endif
  if (task_desc->task_type == TASK_MLA_ATTEND_MI300 && task_desc->variant_id == 0) {
    kernel::mla_attend_mi300_task_impl<bfloat16, 16, 512, 64, 1056>(
        task_desc->input_ptrs[0], task_desc->input_ptrs[1], task_desc->input_ptrs[2],
        task_desc->input_ptrs[3], task_desc->output_ptrs[0], runtime_config.step[0],
        0.1147213867929261f, 32, 33, 5, 4, tile_idx,   // offset rows = 33 / 8 (partials imap (0,-1,-1))
#ifdef MLA_ATTEND_DEBUG_SCORES
        task_desc->output_ptrs[1]);
#else
        nullptr);
#endif
  } else if (task_desc->task_type == TASK_MLA_MERGE_UV_MI300 && task_desc->variant_id == 0) {
    kernel::mla_merge_uv_mi300_task_impl<bfloat16, 16, 128, 512>(
        task_desc->input_ptrs[0], task_desc->input_ptrs[1], task_desc->output_ptrs[0],
        runtime_config.step[0], 32, 33, 2, 2, 0, 1, 1, tile_idx);   // tiles_per_xcd == heads_per_xcd == 2
  }
#ifdef MK_GEMV
  // L6: the stream probe's gang form, G5's w13 row (76 rows of 2,048 per tile, 304 KB, 37 tiles
  // per XCD); the tile decode is the merge's
  else if (task_desc->task_type == MK_TASK_STREAM_GANG_MI300 && task_desc->variant_id == 0) {
    kernel::stream_gang_mi300_task_impl<bfloat16, 2048>(
        task_desc->input_ptrs[0], task_desc->output_ptrs[0], 76, 37, tile_idx);
  }
#endif
}
