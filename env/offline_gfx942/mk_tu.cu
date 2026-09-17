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

using namespace mirage::runtime;

__device__ __forceinline__
void _execute_task(TaskDesc const *task_desc, RuntimeConfig const &runtime_config) {
  if (task_desc->task_type == TASK_MLA_PREP_MI300 && task_desc->variant_id == 0) {
    kernel::mla_prep_mi300_task_impl<bfloat16, 16, 128, 64, 512>(
        task_desc->input_ptrs[0], task_desc->input_ptrs[1], task_desc->input_ptrs[2],
        task_desc->input_ptrs[3], task_desc->input_ptrs[4],
        task_desc->output_ptrs[0], task_desc->output_ptrs[1], task_desc->output_ptrs[2],
        task_desc->output_ptrs[3], runtime_config.step[0], 1e-6f);
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
  } else if (task_desc->task_type == TASK_COPY_MI300 && task_desc->variant_id == 0) {
    kernel::copy_mi300_task_impl<bfloat16, 2048>(task_desc->input_ptrs[0], task_desc->output_ptrs[0]);
  }
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
}
