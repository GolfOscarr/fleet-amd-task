/* kernel_tests_mi300: standalone HIP launcher for the five DeepSeek-V2-Lite
 * task kernels of fleet/tasks/mi300, driven by fleet/tasks/kernel_tests.py
 * (docs/design-doc/07-correctness.md, kernel_tests.py row).
 *
 * Each kernel is wrapped in a __global__ function of 256 threads with the
 * worker's dynamic LDS and called exactly as the runtime's emitted call does
 * (fleet/patches/new_tasks.patch, register_*_mi300_task): the same template
 * dims, the same argument order, `step` and `prompt_length` read from device
 * memory the way runtime_config.step[0] and prompt_length[0] are, float
 * parameters reconstructed from their bit patterns as float_literal() does,
 * gang tiles indexed as n_tile_start + t with n_tile_start = bid.x *
 * tiles_per_xcd (persistent_kernel.cuh, the gang tile loop), and per-XCD
 * pointers pre-offset by dim / 8 * bid.x rows wherever the imap in
 * fleet/build_graph.py partitions a dimension (runtime.cc, per-task pointer
 * computation). Only the five new kernels are exercised; the shipped tasks
 * around them are not part of this program.
 *
 * Build, from the repository root, FLEET = repos/fleet-chiplet-megakernel
 * (the -I order makes fleet/tasks/mi300 win over a copy under FLEET):
 *
 *   mkdir -p fleet/tasks/build && hipcc --offload-arch=gfx942 -O2 -std=c++17 \
 *     -D__HIP_PLATFORM_AMD__=1 -DMIRAGE_AMD_MI300 -DMIRAGE_BACKEND_USE_ROCM \
 *     -DMPK_TARGET_CC=94 -DMODE_ONLINE \
 *     -I fleet -I $FLEET/include -I $FLEET/include/mirage/persistent_kernel \
 *     fleet/tasks/kernel_tests_mi300.cu -o fleet/tasks/build/kernel_tests
 *
 * The debug-scores variant (mla_attend also writes the scaled scores, B5):
 * the same line with -DMLA_ATTEND_DEBUG_SCORES -o fleet/tasks/build/kernel_tests_debug.
 * The gang variant (the two MoE gang rows): the same line with -DKT_FAKE_XCD -o
 * fleet/tasks/build/kernel_tests_xcd, in which the gang kernels take their XCD from
 * blockIdx.y instead of the hardware register, so one (tiles, 8) launch covers every
 * (XCD, tile) pair. The defines are the ones persistent_kernel.py passes on its ROCm path.
 *
 * Usage: kernel_tests <test> <dir> [<dir> ...]
 *   test  mla_prep | mla_attend | mla_merge_uv | mla_merge_uv_tile | mla_merge_oproj
 *         | moe_router | moe_router4
 *         | copy | prefetch | prefetch_moe | stream
 *         | linear_gemv | linear_gemv_norm | linear_gemv_res
 *         | gang_w13_gemv | gang_w2_gemv (the -DKT_FAKE_XCD build only)
 *   dir   params.txt ("name value" per line, integers; floats as IEEE-754
 *         bit patterns) and one raw little-endian file <name>.bin per tensor
 *         of the test (BF16 as uint16, FP32, int32) in the order of the
 *         tables below. Output tensors are uploaded from their .bin too, so
 *         the driver can pre-fill them with a sentinel and see what the
 *         kernel left untouched; they are written back to <name>.out.bin.
 * Exit status: 0 ok, 1 usage, 2 the test needs the debug build, 3 HIP error,
 * 4 file error.
 */
#include <hip/hip_runtime.h>

#include "tasks/mi300/mla_prep_mi300.cuh"
#include "tasks/mi300/mla_attend_mi300.cuh"
#include "tasks/mi300/mla_merge_uv_mi300.cuh"
#include "tasks/mi300/mla_merge_oproj_mi300.cuh"
#include "tasks/mi300/moe_router_mi300.cuh"
#include "tasks/mi300/copy_mi300.cuh"
#include "tasks/mi300/prefetch_mi300.cuh"
#include "tasks/mi300/stream_mi300.cuh"
#include "tasks/mi300/linear_gemv_mi300.cuh"
#include "tasks/mi300/gang_moe_w2_silu_mi300.cuh"
#include "tasks/mi300/gang_moe_w13_gemv_mi300.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace {

// The dims register_task derives from the tensors of fleet/graph_plan.py
// (fleet/pack_weights.py REAL_DIMS, S_max 1056, route log [32, 26, 8]).
constexpr int NH = 16, D_N = 128, D_R = 64, D_V = 128, D_C = 512;
constexpr int S_MAX = 1056, HIDDEN = 2048;
constexpr int N_EXPERTS = 64, N_FORCED = 2, TOPK = 6;
constexpr int ROUTE_STEPS = 32, ROUTE_LAYERS = 26;
constexpr int XCDS = 8;
constexpr int QKVA = NH * (D_N + D_R) + D_C + D_R;
constexpr int N_SLOTS = TOPK + N_FORCED;
constexpr int N_TOTAL = N_EXPERTS + N_FORCED;
constexpr int HEADS_PER_XCD = NH / XCDS;
constexpr int OPROJ_HALVES = 2;                  // N5: the merge with o_proj folded in is one task per half head
constexpr int P_ROW = ((D_C + 1 + 3) / 4) * 4;   // padded partials row (P2), matches the kernels
// the prefetch suites (O8): a dense weight in PF_GRID stripes of PF_ROWS rows (a W_o-like [128, 2048]),
// and an expert weight [N_TOTAL, PF_N, PF_K] whose active experts (mask) are streamed in PF_PARTS parts
constexpr int PF_GRID = 4, PF_ROWS = 32;
constexpr int PF_N = 32, PF_K = 256, PF_PARTS = 2;
// the stream probe (L6): STREAM_GRID tasks of STREAM_ROWS rows of a [*, 2048] BF16 tensor; 38
// rows is five batches of eight over four waves, so the round-robin and the clamped last batch
// are both exercised
constexpr int STREAM_GRID = 4, STREAM_ROWS = 38;
// the GEMV linear (L1): the qkva grid of the model (3,648 rows of 2,048 in tasks of 38, the plain
// and the norm form) and the o_proj grid (2,048 rows in tasks of 32, the residual form)
constexpr int GEMV_ROWS = 38, GEMV_GRID = QKVA / GEMV_ROWS;              // 96 tasks
constexpr int GEMV_RES_ROWS = 32, GEMV_RES_GRID = HIDDEN / GEMV_RES_ROWS; // 64 tasks
// the two MoE gang GEMV rows (L3 and L4): the expert gate-up W13 [E, 2 I_MOE, H] in 37 tiles per
// expert (one per worker of an XCD, S1) and the fused down projection W2 [E, H, I_MOE] in 32.
// A trial file holds only the eight active experts' slabs (GANG_EXPERTS): the model's 66 would be
// 761 MB of W13 per trial, so the mask names the ids 0 to 7 instead of the router's own eight
// (which include the forced 64 and 65); the decode reads the mask, so the ids are all it sees.
constexpr int I_MOE = 1408;
constexpr int GANG_EXPERTS = 8;
constexpr int W13_N = 2 * I_MOE, W13_K = HIDDEN, W13_TILES = 37;
constexpr int W2_N = HIDDEN, W2_K = I_MOE, W2_TILES = W2_N / 64;         // 32 tiles per expert
constexpr int MAX_E_PER_XCD = (N_TOTAL + 7) / 8;                         // 9, the registration's
constexpr int W2_TOTAL_TILES = MAX_E_PER_XCD * W2_TILES;                 // 288
// What the worker kernel is launched with (persistent_kernel.cuh); a task
// may use up to this much dynamic LDS.
constexpr int SMEM_BYTES = mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE;

using bf16 = kernel::bfloat16;

// The runtime_config fields the kernels read; device pointers, as in the
// runtime, so the values take the same path (a device load in the task).
struct Meta {
  int const *step;
  int const *prompt_length;
};

// ---------------------------------------------------------------------------
// wrappers: one block = one task (CU-task) or one tile of a gang task

__global__ __launch_bounds__(256, 1) void k_mla_prep(void const *qkva,
                                                     void const *w_kv_norm,
                                                     void const *w_uk,
                                                     void const *cos,
                                                     void const *sin,
                                                     void *c_kv,
                                                     void *k_pe,
                                                     void *ql_nope,
                                                     void *q_pe,
                                                     Meta meta) {
  kernel::mla_prep_mi300_task_impl<bf16, NH, D_N, D_R, D_C>(
      qkva, w_kv_norm, w_uk, cos, sin, c_kv, k_pe, ql_nope, q_pe, meta.step[0], 1e-6f,
      (int)blockIdx.x);   // the head (grid NH; the runtime passes expert_offset)
}

// grid (8, tiles_per_xcd): bid.x is the XCD slot, bid.y the tile t on it.
__global__ __launch_bounds__(256, 1) void k_mla_attend(void const *ql_nope,
                                                       void const *q_pe,
                                                       void const *c_kv,
                                                       void const *k_pe,
                                                       void *partials,
                                                       Meta meta,
                                                       float softmax_scale,
                                                       int split,
                                                       int n_splits,
                                                       int tiles_per_xcd,
                                                       int partials_xcd_offset_rows,
                                                       void *debug_scores) {
  int xcd = blockIdx.x;
  int tile_idx = xcd * tiles_per_xcd + blockIdx.y;
  // partials imap (0, -1, -1): the runtime hands XCD x a pointer n_splits / 8 * x
  // rows in; the registration passes that row count so the kernel undoes it.
  void *partials_xcd = static_cast<float *>(partials)
      + (size_t)xcd * partials_xcd_offset_rows * NH * P_ROW;
  kernel::mla_attend_mi300_task_impl<bf16, NH, D_C, D_R, S_MAX>(
      ql_nope, q_pe, c_kv, k_pe, partials_xcd, meta.step[0], softmax_scale, split, n_splits,
      tiles_per_xcd, partials_xcd_offset_rows, tile_idx, debug_scores);
}

// grid (8, heads_per_xcd): W_uv imap (0, -1, -1) and attn imap (1, -1, -1)
// give XCD x its own NH / 8 heads of W_uv and NH * D_V / 8 columns of attn
// (w_uv_local = out_local = 1); partials are unpartitioned (offset rows 0).
__global__ __launch_bounds__(256, 1) void k_mla_merge_uv(void const *partials,
                                                         void const *w_uv,
                                                         void *attn,
                                                         Meta meta,
                                                         int split,
                                                         int n_splits) {
  int xcd = blockIdx.x;
  int tile_idx = xcd * HEADS_PER_XCD + blockIdx.y;
  void const *w_uv_xcd = static_cast<bf16 const *>(w_uv) + (size_t)xcd * HEADS_PER_XCD * D_V * D_C;
  void *attn_xcd = static_cast<bf16 *>(attn) + (size_t)xcd * (NH * D_V / XCDS);
  kernel::mla_merge_uv_mi300_task_impl<bf16, NH, D_V, D_C>(
      partials, w_uv_xcd, attn_xcd, meta.step[0], split, n_splits, HEADS_PER_XCD, HEADS_PER_XCD,
      0, 1, 1, tile_idx);
}

// N4: the regular form, grid (NH * halves): every tensor whole and the task index, which the
// runtime passes through expert_offset, is blockIdx.x. `halves` is a kernel argument (the
// launcher's Meta is not extended), so the two instantiations stand behind one launch.
__global__ __launch_bounds__(256, 1) void k_mla_merge_uv_tile(void const *partials,
                                                              void const *w_uv,
                                                              void *attn,
                                                              Meta meta,
                                                              int split,
                                                              int n_splits,
                                                              int halves) {
  if (halves == 2) {
    kernel::mla_merge_uv_tile_mi300_task_impl<bf16, NH, D_V, D_C, 2>(
        partials, w_uv, attn, meta.step[0], split, n_splits, (int)blockIdx.x);
  } else {
    kernel::mla_merge_uv_tile_mi300_task_impl<bf16, NH, D_V, D_C, 1>(
        partials, w_uv, attn, meta.step[0], split, n_splits, (int)blockIdx.x);
  }
}

// N5: the merge with o_proj folded in, grid (NH * OPROJ_HALVES): every tensor whole and the task
// index, which the runtime passes through expert_offset, is blockIdx.x. x_res is an input and an
// output of the operator (the same buffer either way), and the counter is a [1] int32 the driver
// pre-fills with zero and reads back (the last task resets it).
__global__ __launch_bounds__(256, 1) void k_mla_merge_oproj(void const *partials,
                                                            void const *w_uv,
                                                            void const *w_o,
                                                            void *x_res,
                                                            void *counter,
                                                            void *attn,
                                                            void *workspace,
                                                            Meta meta,
                                                            int split,
                                                            int n_splits) {
  kernel::mla_merge_oproj_mi300_task_impl<bf16, NH, D_V, D_C, HIDDEN, OPROJ_HALVES>(
      partials, w_uv, w_o, x_res, counter, attn, workspace, meta.step[0], split, n_splits,
      (int)blockIdx.x);
}

__global__ __launch_bounds__(256, 1) void k_moe_router(void const *x_res,
                                                       void const *w_norm,
                                                       void const *w_gate,
                                                       void *h,
                                                       void *topk_w,
                                                       void *routing,
                                                       void *mask,
                                                       void *logits,
                                                       void *route_log,
                                                       Meta meta,
                                                       int layer_index,
                                                       float scaling,
                                                       float eps) {
  // the fused form (O1): NORM = true, the norm of x_res written to h and routed from LDS
  kernel::moe_router_mi300_task_impl<bf16, HIDDEN, N_EXPERTS, N_FORCED, TOPK, ROUTE_STEPS,
                                     ROUTE_LAYERS, true>(
      x_res, w_norm, w_gate, h, topk_w, routing, mask, logits, route_log, meta.step[0],
      meta.prompt_length[0], layer_index, scaling, eps);
}

// N2: the four-task form, grid (4): the part is blockIdx.x (the runtime passes it through
// expert_offset) and the counter a [1] int32 tensor the driver pre-fills with zero.
__global__ __launch_bounds__(256, 1) void k_moe_router4(void const *x_res,
                                                        void const *w_norm,
                                                        void const *w_gate,
                                                        void *counter,
                                                        void *h,
                                                        void *topk_w,
                                                        void *routing,
                                                        void *mask,
                                                        void *logits,
                                                        void *route_log,
                                                        Meta meta,
                                                        int layer_index,
                                                        float scaling,
                                                        float eps) {
  kernel::moe_router_mi300_task_impl<bf16, HIDDEN, N_EXPERTS, N_FORCED, TOPK, ROUTE_STEPS,
                                     ROUTE_LAYERS, true, 4>(
      x_res, w_norm, w_gate, h, topk_w, routing, mask, logits, route_log, meta.step[0],
      meta.prompt_length[0], layer_index, scaling, eps, (int)blockIdx.x, counter);
}

// grid (PF_GRID): block b streams stripe b of w into row b of dummy [PF_GRID, 4] (one word per wave),
// the pointers offset the way the runtime offsets them for a weight partitioned on dim 0
__global__ __launch_bounds__(256, 1) void k_prefetch(void const *w, void *dummy) {
  int b = blockIdx.x;
  kernel::prefetch_mi300_task_impl<bf16, PF_ROWS, HIDDEN>(
      static_cast<bf16 const *>(w) + (size_t)b * PF_ROWS * HIDDEN, static_cast<int *>(dummy) + b * 4);
}

// grid (STREAM_GRID): block b reads rows [b * STREAM_ROWS, ...) of w into row b of dummy
// [STREAM_GRID, 4] (one XOR word per wave), the pointers offset the way the runtime offsets them
// for a weight partitioned on dim 0
__global__ __launch_bounds__(256, 1) void k_stream(void const *w, void *dummy) {
  int b = blockIdx.x;
  kernel::stream_mi300_task_impl<bf16, HIDDEN>(
      static_cast<bf16 const *>(w) + (size_t)b * STREAM_ROWS * HIDDEN,
      static_cast<int *>(dummy) + b * 4, STREAM_ROWS);
}

// grid (N_SLOTS x PF_PARTS): block b is (slot b / PF_PARTS, part b % PF_PARTS) of the active experts in mask,
// the index the runtime passes through the expert_offset metadata
__global__ __launch_bounds__(256, 1) void k_prefetch_moe(void const *w, void const *mask, void *dummy) {
  int b = blockIdx.x;
  kernel::prefetch_moe_mi300_task_impl<bf16, N_TOTAL, PF_N, PF_K, PF_PARTS>(w, mask, static_cast<int *>(dummy) + b * 4, b);
}

// The GEMV linear (L1), one block per task of the grid: the weight is partitioned on dim 0 and the
// output on dim 1, so block b gets the weight pointer b * rows rows in and the output pointer b *
// rows columns in, as the runtime's per-task pointer computation hands them to a task. `rows` and
// `o_stride` reach the kernel as kernel arguments (the launcher's Meta is not extended).
__global__ __launch_bounds__(256, 1) void k_linear_gemv(void const *x, void const *w, void *out,
                                                        int rows, int o_stride) {
  int b = blockIdx.x;
  kernel::linear_gemv_mi300_task_impl<bf16, HIDDEN, false, false>(
      x, nullptr, static_cast<bf16 const *>(w) + (size_t)b * rows * HIDDEN, nullptr,
      static_cast<bf16 *>(out) + (size_t)b * rows, rows, o_stride, 0.0f);
}

__global__ __launch_bounds__(256, 1) void k_linear_gemv_norm(void const *x, void const *w_norm,
                                                             void const *w, void *out,
                                                             int rows, int o_stride, float eps) {
  int b = blockIdx.x;
  kernel::linear_gemv_mi300_task_impl<bf16, HIDDEN, true, false>(
      x, w_norm, static_cast<bf16 const *>(w) + (size_t)b * rows * HIDDEN, nullptr,
      static_cast<bf16 *>(out) + (size_t)b * rows, rows, o_stride, eps);
}

// the residual is [1, N] and partitioned like the output, so it is offset by the task's columns too
__global__ __launch_bounds__(256, 1) void k_linear_gemv_res(void const *x, void const *w,
                                                            void const *residual, void *out,
                                                            int rows, int o_stride) {
  int b = blockIdx.x;
  kernel::linear_gemv_mi300_task_impl<bf16, HIDDEN, false, true>(
      x, nullptr, static_cast<bf16 const *>(w) + (size_t)b * rows * HIDDEN,
      static_cast<bf16 const *>(residual) + (size_t)b * rows,
      static_cast<bf16 *>(out) + (size_t)b * rows, rows, o_stride, 0.0f);
}

// The two MoE gang kernels, grid (tiles, 8): tile_idx is blockIdx.x and the XCD blockIdx.y, which
// -DKT_FAKE_XCD substitutes for the hardware register, so one launch covers every (XCD, tile) pair
// deterministically. Every tensor is whole: a gang task reads the mask and the routing itself.
__global__ __launch_bounds__(256, 1) void k_gang_w13_gemv(void const *h, void const *w13,
                                                          void const *routing, void const *mask,
                                                          void *mid) {
  kernel::gang_moe_w13_gemv_kernel<bf16, W13_N, W13_K, N_TOTAL, N_SLOTS, W13_TILES>(
      h, w13, routing, mask, mid, (int)blockIdx.x);
}

// the scratch output is the MPK_W2_CK_TILE path's alone; the default path never writes it
__global__ __launch_bounds__(256, 1) void k_gang_w2_gemv(void const *mid, void const *w2,
                                                         void const *routing, void const *mask,
                                                         void *out8) {
  kernel::gang_moe_w2_silu_linear_kernel<bf16, 1, W2_N, W2_N, W2_K, W13_N, N_TOTAL, N_SLOTS,
                                         W2_TILES, W2_TILES, W2_TOTAL_TILES>(
      mid, w2, routing, mask, out8, nullptr, (int)blockIdx.x);
}

__global__ __launch_bounds__(256, 1) void k_copy(void const *x, void *y, int spin) {
  // spin > 0 (KT_SPIN, I2): the shader-clock spin after the copy, printed as a [SPIN] line
  kernel::copy_mi300_task_impl<bf16, HIDDEN>(x, y, spin, spin > 0 ? 1 : 0);
}

// ---------------------------------------------------------------------------
// host side

#define HIP_CHECK(call)                                                                  \
  do {                                                                                   \
    hipError_t e_ = (call);                                                              \
    if (e_ != hipSuccess) {                                                              \
      std::fprintf(stderr, "HIP error: %s (%s:%d)\n", hipGetErrorString(e_), __FILE__,   \
                   __LINE__);                                                            \
      std::exit(3);                                                                      \
    }                                                                                    \
  } while (0)

// One tensor of a test: file name, byte count, whether the kernel writes it.
// The tables are the contract with kernel_tests.py (same names, same order).
struct Spec {
  char const *name;
  size_t bytes;
  bool output;
};

static const Spec SPEC_MLA_PREP[] = {
    {"qkva", (size_t)QKVA * 2, false},
    {"w_kv_norm", (size_t)D_C * 2, false},
    {"w_uk", (size_t)NH * D_N * D_C * 2, false},
    {"cos", (size_t)S_MAX * D_R * 2, false},
    {"sin", (size_t)S_MAX * D_R * 2, false},
    {"c_kv", (size_t)S_MAX * D_C * 2, true},
    {"k_pe", (size_t)S_MAX * D_R * 2, true},
    {"ql_nope", (size_t)NH * D_C * 2, true},
    {"q_pe", (size_t)NH * D_R * 2, true},
};

static const Spec SPEC_MLA_ATTEND[] = {
    {"ql_nope", (size_t)NH * D_C * 2, false},
    {"q_pe", (size_t)NH * D_R * 2, false},
    {"c_kv", (size_t)S_MAX * D_C * 2, false},
    {"k_pe", (size_t)S_MAX * D_R * 2, false},
    {"partials", 0, true},   // n_splits * NH * P_ROW * 4, from params
};
static const Spec SPEC_MLA_ATTEND_SCORES = {"scores", (size_t)NH * S_MAX * 4, true};

static const Spec SPEC_MLA_MERGE_UV[] = {
    {"partials", 0, false},  // n_splits * NH * P_ROW * 4, from params
    {"w_uv", (size_t)NH * D_V * D_C * 2, false},
    {"attn", (size_t)NH * D_V * 2, true},
};

// N4: the regular launch reads the same three tensors; `halves` is a params.txt entry
static const Spec SPEC_MLA_MERGE_UV_TILE[] = {
    {"partials", 0, false},  // n_splits * NH * P_ROW * 4, from params
    {"w_uv", (size_t)NH * D_V * D_C * 2, false},
    {"attn", (size_t)NH * D_V * 2, true},
};

// N5: the merge's tensors plus W_o, x_res (read and written in place), the arrival counter (the
// driver writes it as zero and reads it back, so its reset is checked) and the partial workspace,
// which the kernel writes and only its own last task reads
static const Spec SPEC_MLA_MERGE_OPROJ[] = {
    {"partials", 0, false},  // n_splits * NH * P_ROW * 4, from params
    {"w_uv", (size_t)NH * D_V * D_C * 2, false},
    {"w_o", (size_t)HIDDEN * HIDDEN * 2, false},
    {"x_res", (size_t)HIDDEN * 2, true},
    {"counter", 4, true},
    {"attn", (size_t)NH * D_V * 2, true},
    {"workspace", (size_t)NH * OPROJ_HALVES * HIDDEN * 4, false},
};

static const Spec SPEC_MOE_ROUTER[] = {   // the fused form, NORM = true (O1)
    {"x_res", (size_t)HIDDEN * 2, false},
    {"w_norm", (size_t)HIDDEN * 2, false},
    {"w_gate", (size_t)N_EXPERTS * HIDDEN * 2, false},
    {"h", (size_t)HIDDEN * 2, true},
    {"topk_w", (size_t)N_SLOTS * 4, true},
    {"routing", (size_t)N_TOTAL * 4, true},
    {"mask", (size_t)(N_TOTAL + 1) * 4, true},
    {"logits", (size_t)N_EXPERTS * 4, true},
    {"route_log", (size_t)ROUTE_STEPS * ROUTE_LAYERS * N_SLOTS * 4, true},
};

// N2: the same tensors plus the arrival counter, which the driver writes as zero and reads
// back (the last task resets it), so its round trip is checked too
static const Spec SPEC_MOE_ROUTER4[] = {
    {"x_res", (size_t)HIDDEN * 2, false},
    {"w_norm", (size_t)HIDDEN * 2, false},
    {"w_gate", (size_t)N_EXPERTS * HIDDEN * 2, false},
    {"counter", 4, true},
    {"h", (size_t)HIDDEN * 2, true},
    {"topk_w", (size_t)N_SLOTS * 4, true},
    {"routing", (size_t)N_TOTAL * 4, true},
    {"mask", (size_t)(N_TOTAL + 1) * 4, true},
    {"logits", (size_t)N_EXPERTS * 4, true},
    {"route_log", (size_t)ROUTE_STEPS * ROUTE_LAYERS * N_SLOTS * 4, true},
};

static const Spec SPEC_PREFETCH[] = {
    {"w", (size_t)PF_GRID * PF_ROWS * HIDDEN * 2, false},
    {"dummy", (size_t)PF_GRID * 4 * 4, true},
};
static const Spec SPEC_STREAM[] = {
    {"w", (size_t)STREAM_GRID * STREAM_ROWS * HIDDEN * 2, false},
    {"dummy", (size_t)STREAM_GRID * 4 * 4, true},
};
static const Spec SPEC_PREFETCH_MOE[] = {
    {"w", (size_t)N_TOTAL * PF_N * PF_K * 2, false},
    {"mask", (size_t)(N_TOTAL + 1) * 4, false},
    {"dummy", (size_t)N_SLOTS * PF_PARTS * 4 * 4, true},
};
// the GEMV linear (L1): the three forms, at the model's dims
static const Spec SPEC_LINEAR_GEMV[] = {
    {"x", (size_t)HIDDEN * 2, false},
    {"w", (size_t)QKVA * HIDDEN * 2, false},
    {"out", (size_t)QKVA * 2, true},
};
static const Spec SPEC_LINEAR_GEMV_NORM[] = {
    {"x", (size_t)HIDDEN * 2, false},
    {"w_norm", (size_t)HIDDEN * 2, false},
    {"w", (size_t)QKVA * HIDDEN * 2, false},
    {"out", (size_t)QKVA * 2, true},
};
static const Spec SPEC_LINEAR_GEMV_RES[] = {
    {"x", (size_t)HIDDEN * 2, false},
    {"w", (size_t)HIDDEN * HIDDEN * 2, false},
    {"residual", (size_t)HIDDEN * 2, false},
    {"out", (size_t)HIDDEN * 2, true},
};

// the two MoE gang rows (L3, L4): the eight active experts' slabs, the router's routing and mask
static const Spec SPEC_GANG_W13_GEMV[] = {
    {"h", (size_t)W13_K * 2, false},
    {"w13", (size_t)GANG_EXPERTS * W13_N * W13_K * 2, false},
    {"routing", (size_t)N_TOTAL * 4, false},
    {"mask", (size_t)(N_TOTAL + 1) * 4, false},
    {"mid", (size_t)N_SLOTS * W13_N * 2, true},
};
static const Spec SPEC_GANG_W2_GEMV[] = {
    {"mid", (size_t)N_SLOTS * W13_N * 2, false},
    {"w2", (size_t)GANG_EXPERTS * W2_N * W2_K * 2, false},
    {"routing", (size_t)N_TOTAL * 4, false},
    {"mask", (size_t)(N_TOTAL + 1) * 4, false},
    {"out8", (size_t)N_SLOTS * W2_N * 2, true},
};

static const Spec SPEC_COPY[] = {
    {"x", (size_t)HIDDEN * 2, false},
    {"y", (size_t)HIDDEN * 2, true},
};

using Params = std::map<std::string, long long>;

Params read_params(std::string const &dir) {
  std::ifstream f(dir + "/params.txt");
  if (!f) {
    std::fprintf(stderr, "cannot read %s/params.txt\n", dir.c_str());
    std::exit(4);
  }
  Params p;
  std::string k;
  long long v;
  while (f >> k >> v) {
    p[k] = v;
  }
  return p;
}

long long param(Params const &p, char const *name) {
  auto it = p.find(name);
  if (it == p.end()) {
    std::fprintf(stderr, "params.txt lacks %s\n", name);
    std::exit(4);
  }
  return it->second;
}

long long param_or(Params const &p, char const *name, long long dflt) {
  auto it = p.find(name);
  return it == p.end() ? dflt : it->second;
}

// The registration prints float parameters from their bit pattern with
// enough digits to round-trip; reinterpreting the bits is the same value.
float float_from_bits(long long bits) {
  uint32_t u = (uint32_t)bits;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

// Device copies of every tensor of a test, loaded from <dir>/<name>.bin.
struct Buffers {
  std::string dir;
  std::vector<Spec> specs;
  std::vector<void *> dev;

  void load() {
    for (auto const &s : specs) {
      std::string path = dir + "/" + s.name + ".bin";
      std::ifstream f(path, std::ios::binary | std::ios::ate);
      if (!f) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        std::exit(4);
      }
      size_t size = (size_t)f.tellg();
      if (size != s.bytes) {
        std::fprintf(stderr, "%s: %zu bytes, expected %zu\n", path.c_str(), size, s.bytes);
        std::exit(4);
      }
      std::vector<char> host(size);
      f.seekg(0);
      f.read(host.data(), (std::streamsize)size);
      void *d = nullptr;
      HIP_CHECK(hipMalloc(&d, size));
      HIP_CHECK(hipMemcpy(d, host.data(), size, hipMemcpyHostToDevice));
      dev.push_back(d);
    }
  }

  void *get(char const *name) const {
    for (size_t i = 0; i < specs.size(); i++) {
      if (std::strcmp(specs[i].name, name) == 0) {
        return dev[i];
      }
    }
    std::fprintf(stderr, "no tensor %s\n", name);
    std::exit(4);
  }

  void store_outputs() {
    for (size_t i = 0; i < specs.size(); i++) {
      if (!specs[i].output) {
        continue;
      }
      std::vector<char> host(specs[i].bytes);
      HIP_CHECK(hipMemcpy(host.data(), dev[i], specs[i].bytes, hipMemcpyDeviceToHost));
      std::string path = dir + "/" + specs[i].name + ".out.bin";
      std::ofstream f(path, std::ios::binary);
      if (!f) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        std::exit(4);
      }
      f.write(host.data(), (std::streamsize)host.size());
    }
  }

  ~Buffers() {
    for (void *d : dev) {
      (void)hipFree(d);
    }
  }
};

template <size_t N>
std::vector<Spec> specs_of(Spec const (&table)[N]) {
  return std::vector<Spec>(table, table + N);
}

// step and prompt_length live in device memory as the runtime's meta tensors do.
struct DeviceMeta {
  int *mem = nullptr;
  Meta meta;

  DeviceMeta(int step, int prompt_length) {
    int host[2] = {step, prompt_length};
    HIP_CHECK(hipMalloc((void **)&mem, sizeof(host)));
    HIP_CHECK(hipMemcpy(mem, host, sizeof(host), hipMemcpyHostToDevice));
    meta.step = mem;
    meta.prompt_length = mem + 1;
  }
  ~DeviceMeta() {
    (void)hipFree(mem);
  }
};

template <typename K>
void allow_full_lds(K kernel) {
  // as the runtime does for worker_kernel; a launch that needs it fails loudly below
  (void)hipFuncSetAttribute(reinterpret_cast<void const *>(kernel),
                            hipFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
}

void finish_launch() {
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
}

void run_mla_prep(std::string const &dir) {
  Params p = read_params(dir);
  Buffers b{dir, specs_of(SPEC_MLA_PREP), {}};
  b.load();
  DeviceMeta m((int)param(p, "step"), (int)param_or(p, "prompt_length", 0));
  allow_full_lds(k_mla_prep);
  hipLaunchKernelGGL(k_mla_prep, dim3(NH), dim3(256), SMEM_BYTES, 0,
                     b.get("qkva"), b.get("w_kv_norm"), b.get("w_uk"), b.get("cos"), b.get("sin"),
                     b.get("c_kv"), b.get("k_pe"), b.get("ql_nope"), b.get("q_pe"), m.meta);
  finish_launch();
  b.store_outputs();
}

void run_mla_attend(std::string const &dir) {
  Params p = read_params(dir);
  int split = (int)param(p, "split");
  int n_splits = (int)param(p, "n_splits");
  int tiles_per_xcd = (int)param(p, "tiles_per_xcd");
  bool debug = param_or(p, "debug_scores", 0) != 0;
  if (debug) {
#ifndef MLA_ATTEND_DEBUG_SCORES
    std::fprintf(stderr, "debug_scores needs the -DMLA_ATTEND_DEBUG_SCORES build\n");
    std::exit(2);
#endif
  }
  Buffers b{dir, specs_of(SPEC_MLA_ATTEND), {}};
  b.specs[4].bytes = (size_t)n_splits * NH * P_ROW * 4;
  if (debug) {
    b.specs.push_back(SPEC_MLA_ATTEND_SCORES);
  }
  b.load();
  DeviceMeta m((int)param(p, "step"), (int)param_or(p, "prompt_length", 0));
  // xcd_offset_dim0 of the registration: dim 0 of partials over the grid of 8
  int offset_rows = n_splits / XCDS;
  allow_full_lds(k_mla_attend);
  hipLaunchKernelGGL(k_mla_attend, dim3(XCDS, tiles_per_xcd), dim3(256), SMEM_BYTES, 0,
                     b.get("ql_nope"), b.get("q_pe"), b.get("c_kv"), b.get("k_pe"),
                     b.get("partials"), m.meta, float_from_bits(param(p, "softmax_scale_bits")),
                     split, n_splits, tiles_per_xcd, offset_rows,
                     debug ? b.get("scores") : nullptr);
  finish_launch();
  // KT_TIME=N: N more launches under hipEvents, the standalone time of the grid of tiles
  // (session B, 2026-09-16: 145 to 215 us per tile in the graph, which this separates from the runtime)
  if (char const *kt = std::getenv("KT_TIME")) {
    int n = std::atoi(kt);
    // KT_COLD=K: rotate over K copies of the cache (K x 1.2 MB: past the 4 MB L2 of an XCD for K >= 4,
    // past the 256 MB infinity cache for K >= 220), so a launch reads its cache the way the graph does
    // (every layer's cache once per iteration) instead of re-reading one warm copy
    int cold = std::getenv("KT_COLD") ? std::atoi(std::getenv("KT_COLD")) : 1;
    size_t ckv_bytes = (size_t)S_MAX * D_C * 2, kpe_bytes = (size_t)S_MAX * D_R * 2;
    std::vector<void *> ckv(cold), kpe(cold);
    for (int k = 0; k < cold; k++) {
      HIP_CHECK(hipMalloc(&ckv[k], ckv_bytes)); HIP_CHECK(hipMalloc(&kpe[k], kpe_bytes));
      HIP_CHECK(hipMemcpy(ckv[k], b.get("c_kv"), ckv_bytes, hipMemcpyDeviceToDevice));
      HIP_CHECK(hipMemcpy(kpe[k], b.get("k_pe"), kpe_bytes, hipMemcpyDeviceToDevice));
    }
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    hipEventRecord(t0, 0);
    for (int i = 0; i < n; i++) {
      hipLaunchKernelGGL(k_mla_attend, dim3(XCDS, tiles_per_xcd), dim3(256), SMEM_BYTES, 0,
                         b.get("ql_nope"), b.get("q_pe"), ckv[i % cold], kpe[i % cold],
                         b.get("partials"), m.meta, float_from_bits(param(p, "softmax_scale_bits")),
                         split, n_splits, tiles_per_xcd, offset_rows,
                         debug ? b.get("scores") : nullptr);
    }
    hipEventRecord(t1, 0); hipEventSynchronize(t1);
    float ms = 0; hipEventElapsedTime(&ms, t0, t1);
    std::fprintf(stderr, "TIME mla_attend launches=%d grid=%dx%d cache_copies=%d mean_us=%.2f\n",
                 n, XCDS, tiles_per_xcd, cold, ms * 1000.0f / n);
    for (int k = 0; k < cold; k++) { hipFree(ckv[k]); hipFree(kpe[k]); }
  }
  b.store_outputs();
}

void run_mla_merge_uv(std::string const &dir) {
  Params p = read_params(dir);
  int split = (int)param(p, "split");
  int n_splits = (int)param(p, "n_splits");
  Buffers b{dir, specs_of(SPEC_MLA_MERGE_UV), {}};
  b.specs[0].bytes = (size_t)n_splits * NH * P_ROW * 4;
  b.load();
  DeviceMeta m((int)param(p, "step"), (int)param_or(p, "prompt_length", 0));
  allow_full_lds(k_mla_merge_uv);
  hipLaunchKernelGGL(k_mla_merge_uv, dim3(XCDS, HEADS_PER_XCD), dim3(256), SMEM_BYTES, 0,
                     b.get("partials"), b.get("w_uv"), b.get("attn"), m.meta, split, n_splits);
  finish_launch();
  if (char const *kt = std::getenv("KT_TIME")) {
    int n = std::atoi(kt);
    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    hipEventRecord(t0, 0);
    for (int i = 0; i < n; i++) {
  hipLaunchKernelGGL(k_mla_merge_uv, dim3(XCDS, HEADS_PER_XCD), dim3(256), SMEM_BYTES, 0,
                     b.get("partials"), b.get("w_uv"), b.get("attn"), m.meta, split, n_splits);
    }
    hipEventRecord(t1, 0); hipEventSynchronize(t1);
    float ms = 0; hipEventElapsedTime(&ms, t0, t1);
    std::fprintf(stderr, "TIME mla_merge_uv launches=%d mean_us=%.2f\n", n, ms * 1000.0f / n);
  }
  b.store_outputs();
}

// N4: the same tensors as mla_merge_uv, plus the params entry `halves`; the grid is NH * halves
void run_mla_merge_uv_tile(std::string const &dir) {
  Params p = read_params(dir);
  int split = (int)param(p, "split");
  int n_splits = (int)param(p, "n_splits");
  int halves = (int)param_or(p, "halves", 1);
  Buffers b{dir, specs_of(SPEC_MLA_MERGE_UV_TILE), {}};
  b.specs[0].bytes = (size_t)n_splits * NH * P_ROW * 4;
  b.load();
  DeviceMeta m((int)param(p, "step"), (int)param_or(p, "prompt_length", 0));
  allow_full_lds(k_mla_merge_uv_tile);
  hipLaunchKernelGGL(k_mla_merge_uv_tile, dim3(NH * halves), dim3(256), SMEM_BYTES, 0,
                     b.get("partials"), b.get("w_uv"), b.get("attn"), m.meta, split, n_splits, halves);
  finish_launch();
  if (char const *kt = std::getenv("KT_TIME")) {
    int n = std::atoi(kt);
    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    hipEventRecord(t0, 0);
    for (int i = 0; i < n; i++) {
      hipLaunchKernelGGL(k_mla_merge_uv_tile, dim3(NH * halves), dim3(256), SMEM_BYTES, 0,
                         b.get("partials"), b.get("w_uv"), b.get("attn"), m.meta, split, n_splits, halves);
    }
    hipEventRecord(t1, 0); hipEventSynchronize(t1);
    float ms = 0; hipEventElapsedTime(&ms, t0, t1);
    std::fprintf(stderr, "TIME mla_merge_uv_tile launches=%d halves=%d mean_us=%.2f\n", n, halves,
                 ms * 1000.0f / n);
  }
  b.store_outputs();
}

// N5: the same params as the tile row (halves is the kernel's template constant, 2); the grid is
// NH * OPROJ_HALVES and the counter starts at zero
void run_mla_merge_oproj(std::string const &dir) {
  Params p = read_params(dir);
  int split = (int)param(p, "split");
  int n_splits = (int)param(p, "n_splits");
  Buffers b{dir, specs_of(SPEC_MLA_MERGE_OPROJ), {}};
  b.specs[0].bytes = (size_t)n_splits * NH * P_ROW * 4;
  b.load();
  DeviceMeta m((int)param(p, "step"), (int)param_or(p, "prompt_length", 0));
  allow_full_lds(k_mla_merge_oproj);
  hipLaunchKernelGGL(k_mla_merge_oproj, dim3(NH * OPROJ_HALVES), dim3(256), SMEM_BYTES, 0,
                     b.get("partials"), b.get("w_uv"), b.get("w_o"), b.get("x_res"),
                     b.get("counter"), b.get("attn"), b.get("workspace"), m.meta, split, n_splits);
  finish_launch();
  if (char const *kt = std::getenv("KT_TIME")) {
    int n = std::atoi(kt);
    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    hipEventRecord(t0, 0);
    for (int i = 0; i < n; i++) {
      hipLaunchKernelGGL(k_mla_merge_oproj, dim3(NH * OPROJ_HALVES), dim3(256), SMEM_BYTES, 0,
                         b.get("partials"), b.get("w_uv"), b.get("w_o"), b.get("x_res"),
                         b.get("counter"), b.get("attn"), b.get("workspace"), m.meta, split, n_splits);
    }
    hipEventRecord(t1, 0); hipEventSynchronize(t1);
    float ms = 0; hipEventElapsedTime(&ms, t0, t1);
    std::fprintf(stderr, "TIME mla_merge_oproj launches=%d mean_us=%.2f\n", n, ms * 1000.0f / n);
  }
  b.store_outputs();
}

void run_moe_router(std::string const &dir) {
  Params p = read_params(dir);
  Buffers b{dir, specs_of(SPEC_MOE_ROUTER), {}};
  b.load();
  DeviceMeta m((int)param(p, "step"), (int)param(p, "prompt_length"));
  allow_full_lds(k_moe_router);
  hipLaunchKernelGGL(k_moe_router, dim3(1), dim3(256), SMEM_BYTES, 0,
                     b.get("x_res"), b.get("w_norm"), b.get("w_gate"), b.get("h"), b.get("topk_w"),
                     b.get("routing"), b.get("mask"), b.get("logits"), b.get("route_log"), m.meta,
                     (int)param(p, "layer_index"), float_from_bits(param(p, "scaling_bits")),
                     float_from_bits(param(p, "eps_bits")));
  finish_launch();
  b.store_outputs();
}

void run_moe_router4(std::string const &dir) {
  Params p = read_params(dir);
  Buffers b{dir, specs_of(SPEC_MOE_ROUTER4), {}};
  b.load();
  DeviceMeta m((int)param(p, "step"), (int)param(p, "prompt_length"));
  allow_full_lds(k_moe_router4);
  hipLaunchKernelGGL(k_moe_router4, dim3(4), dim3(256), SMEM_BYTES, 0,
                     b.get("x_res"), b.get("w_norm"), b.get("w_gate"), b.get("counter"), b.get("h"),
                     b.get("topk_w"), b.get("routing"), b.get("mask"), b.get("logits"),
                     b.get("route_log"), m.meta,
                     (int)param(p, "layer_index"), float_from_bits(param(p, "scaling_bits")),
                     float_from_bits(param(p, "eps_bits")));
  finish_launch();
  b.store_outputs();
}

void run_prefetch(std::string const &dir) {
  (void)read_params(dir);
  Buffers b{dir, specs_of(SPEC_PREFETCH), {}};
  b.load();
  allow_full_lds(k_prefetch);
  hipLaunchKernelGGL(k_prefetch, dim3(PF_GRID), dim3(256), SMEM_BYTES, 0, b.get("w"), b.get("dummy"));
  finish_launch();
  b.store_outputs();
}

void run_stream(std::string const &dir) {
  (void)read_params(dir);
  Buffers b{dir, specs_of(SPEC_STREAM), {}};
  b.load();
  allow_full_lds(k_stream);
  hipLaunchKernelGGL(k_stream, dim3(STREAM_GRID), dim3(256), SMEM_BYTES, 0, b.get("w"), b.get("dummy"));
  finish_launch();
  b.store_outputs();
}

void run_prefetch_moe(std::string const &dir) {
  (void)read_params(dir);
  Buffers b{dir, specs_of(SPEC_PREFETCH_MOE), {}};
  b.load();
  allow_full_lds(k_prefetch_moe);
  hipLaunchKernelGGL(k_prefetch_moe, dim3(N_SLOTS * PF_PARTS), dim3(256), SMEM_BYTES, 0,
                     b.get("w"), b.get("mask"), b.get("dummy"));
  finish_launch();
  b.store_outputs();
}

void run_linear_gemv(std::string const &dir) {
  (void)read_params(dir);
  Buffers b{dir, specs_of(SPEC_LINEAR_GEMV), {}};
  b.load();
  allow_full_lds(k_linear_gemv);
  hipLaunchKernelGGL(k_linear_gemv, dim3(GEMV_GRID), dim3(256), SMEM_BYTES, 0,
                     b.get("x"), b.get("w"), b.get("out"), GEMV_ROWS, QKVA);
  finish_launch();
  // KT_TIME=N: N more launches of the whole grid under hipEvents (the 96 tasks of qkva, which
  // the graph runs as one operator), the standalone time the ktime stage compares with the
  // per-operator cost inside the megakernel
  if (char const *kt = std::getenv("KT_TIME")) {
    int n = std::atoi(kt);
    // KT_COLD=K: rotate over K copies of the 15 MB weight, as the attention's rotation does over
    // copies of its 1.2 MB cache; one copy already exceeds an XCD's 4 MB L2, and K >= 18 the
    // 256 MB memory-side cache, so the number is L2-cold (the 2x rule of 09-lessons.md, lesson 6)
    int cold = std::getenv("KT_COLD") ? std::atoi(std::getenv("KT_COLD")) : 1;
    size_t w_bytes = (size_t)QKVA * HIDDEN * 2;
    std::vector<void *> w(cold);
    for (int k = 0; k < cold; k++) {
      HIP_CHECK(hipMalloc(&w[k], w_bytes));
      HIP_CHECK(hipMemcpy(w[k], b.get("w"), w_bytes, hipMemcpyDeviceToDevice));
    }
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    hipEventRecord(t0, 0);
    for (int i = 0; i < n; i++) {
      hipLaunchKernelGGL(k_linear_gemv, dim3(GEMV_GRID), dim3(256), SMEM_BYTES, 0,
                         b.get("x"), w[i % cold], b.get("out"), GEMV_ROWS, QKVA);
    }
    hipEventRecord(t1, 0); hipEventSynchronize(t1);
    float ms = 0; hipEventElapsedTime(&ms, t0, t1);
    std::fprintf(stderr, "TIME linear_gemv launches=%d grid=%d rows=%d weight_copies=%d mean_us=%.2f\n",
                 n, GEMV_GRID, GEMV_ROWS, cold, ms * 1000.0f / n);
    for (int k = 0; k < cold; k++) { hipFree(w[k]); }
  }
  b.store_outputs();
}

void run_linear_gemv_norm(std::string const &dir) {
  Params p = read_params(dir);
  Buffers b{dir, specs_of(SPEC_LINEAR_GEMV_NORM), {}};
  b.load();
  allow_full_lds(k_linear_gemv_norm);
  hipLaunchKernelGGL(k_linear_gemv_norm, dim3(GEMV_GRID), dim3(256), SMEM_BYTES, 0,
                     b.get("x"), b.get("w_norm"), b.get("w"), b.get("out"), GEMV_ROWS, QKVA,
                     float_from_bits(param(p, "eps_bits")));
  finish_launch();
  b.store_outputs();
}

void run_linear_gemv_res(std::string const &dir) {
  (void)read_params(dir);
  Buffers b{dir, specs_of(SPEC_LINEAR_GEMV_RES), {}};
  b.load();
  allow_full_lds(k_linear_gemv_res);
  hipLaunchKernelGGL(k_linear_gemv_res, dim3(GEMV_RES_GRID), dim3(256), SMEM_BYTES, 0,
                     b.get("x"), b.get("w"), b.get("residual"), b.get("out"), GEMV_RES_ROWS, HIDDEN);
  finish_launch();
  b.store_outputs();
}

// Both gang rows need the -DKT_FAKE_XCD build: without it the kernel takes its XCD from the
// hardware register and the (tile, XCD) pairs a launch covers are whatever the scheduler chose.
void run_gang_w13_gemv(std::string const &dir) {
#ifndef KT_FAKE_XCD
  (void)dir;
  std::fprintf(stderr, "gang_w13_gemv needs the -DKT_FAKE_XCD build\n");
  std::exit(2);
#else
  (void)read_params(dir);
  Buffers b{dir, specs_of(SPEC_GANG_W13_GEMV), {}};
  b.load();
  allow_full_lds(k_gang_w13_gemv);
  hipLaunchKernelGGL(k_gang_w13_gemv, dim3(W13_TILES, XCDS), dim3(256), SMEM_BYTES, 0,
                     b.get("h"), b.get("w13"), b.get("routing"), b.get("mask"), b.get("mid"));
  finish_launch();
  // KT_TIME=N: N more launches of the whole (37, 8) grid under hipEvents, the standalone time
  // of one layer's expert gate-up the ktime stage compares with its per-operator cost
  if (char const *kt = std::getenv("KT_TIME")) {
    int n = std::atoi(kt);
    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    hipEventRecord(t0, 0);
    for (int i = 0; i < n; i++) {
      hipLaunchKernelGGL(k_gang_w13_gemv, dim3(W13_TILES, XCDS), dim3(256), SMEM_BYTES, 0,
                         b.get("h"), b.get("w13"), b.get("routing"), b.get("mask"), b.get("mid"));
    }
    hipEventRecord(t1, 0); hipEventSynchronize(t1);
    float ms = 0; hipEventElapsedTime(&ms, t0, t1);
    std::fprintf(stderr, "TIME gang_w13_gemv launches=%d grid=%dx%d mean_us=%.2f\n",
                 n, W13_TILES, XCDS, ms * 1000.0f / n);
  }
  b.store_outputs();
#endif
}

void run_gang_w2_gemv(std::string const &dir) {
#ifndef KT_FAKE_XCD
  (void)dir;
  std::fprintf(stderr, "gang_w2_gemv needs the -DKT_FAKE_XCD build\n");
  std::exit(2);
#else
  (void)read_params(dir);
  Buffers b{dir, specs_of(SPEC_GANG_W2_GEMV), {}};
  b.load();
  allow_full_lds(k_gang_w2_gemv);
  hipLaunchKernelGGL(k_gang_w2_gemv, dim3(W2_TILES, XCDS), dim3(256), SMEM_BYTES, 0,
                     b.get("mid"), b.get("w2"), b.get("routing"), b.get("mask"), b.get("out8"));
  finish_launch();
  if (char const *kt = std::getenv("KT_TIME")) {
    int n = std::atoi(kt);
    hipEvent_t t0, t1;
    hipEventCreate(&t0); hipEventCreate(&t1);
    hipEventRecord(t0, 0);
    for (int i = 0; i < n; i++) {
      hipLaunchKernelGGL(k_gang_w2_gemv, dim3(W2_TILES, XCDS), dim3(256), SMEM_BYTES, 0,
                         b.get("mid"), b.get("w2"), b.get("routing"), b.get("mask"), b.get("out8"));
    }
    hipEventRecord(t1, 0); hipEventSynchronize(t1);
    float ms = 0; hipEventElapsedTime(&ms, t0, t1);
    std::fprintf(stderr, "TIME gang_w2_gemv launches=%d grid=%dx%d mean_us=%.2f\n",
                 n, W2_TILES, XCDS, ms * 1000.0f / n);
  }
  b.store_outputs();
#endif
}

void run_copy(std::string const &dir) {
  (void)read_params(dir);
  Buffers b{dir, specs_of(SPEC_COPY), {}};
  b.load();
  allow_full_lds(k_copy);
  hipLaunchKernelGGL(k_copy, dim3(1), dim3(256), SMEM_BYTES, 0, b.get("x"), b.get("y"), 0);
  finish_launch();
  // KT_SPIN=n (I2): one more launch whose thread 0 spins n iterations and prints the clock deltas
  if (char const *ks = std::getenv("KT_SPIN")) {
    hipLaunchKernelGGL(k_copy, dim3(1), dim3(256), SMEM_BYTES, 0, b.get("x"), b.get("y"), std::atoi(ks));
    finish_launch();
  }
  b.store_outputs();
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: %s <mla_prep|mla_attend|mla_merge_uv|mla_merge_uv_tile|mla_merge_oproj|moe_router|moe_router4|copy|prefetch"
                 "|prefetch_moe|stream|linear_gemv|linear_gemv_norm|linear_gemv_res|gang_w13_gemv|gang_w2_gemv> <dir>...\n",
                 argv[0]);
    return 1;
  }
  std::string test = argv[1];
  void (*run)(std::string const &) = nullptr;
  if (test == "mla_prep") {
    run = run_mla_prep;
  } else if (test == "mla_attend") {
    run = run_mla_attend;
  } else if (test == "mla_merge_uv") {
    run = run_mla_merge_uv;
  } else if (test == "mla_merge_uv_tile") {
    run = run_mla_merge_uv_tile;
  } else if (test == "mla_merge_oproj") {
    run = run_mla_merge_oproj;
  } else if (test == "moe_router") {
    run = run_moe_router;
  } else if (test == "moe_router4") {
    run = run_moe_router4;
  } else if (test == "copy") {
    run = run_copy;
  } else if (test == "prefetch") {
    run = run_prefetch;
  } else if (test == "prefetch_moe") {
    run = run_prefetch_moe;
  } else if (test == "stream") {
    run = run_stream;
  } else if (test == "linear_gemv") {
    run = run_linear_gemv;
  } else if (test == "linear_gemv_norm") {
    run = run_linear_gemv_norm;
  } else if (test == "linear_gemv_res") {
    run = run_linear_gemv_res;
  } else if (test == "gang_w13_gemv") {
    run = run_gang_w13_gemv;
  } else if (test == "gang_w2_gemv") {
    run = run_gang_w2_gemv;
  } else {
    std::fprintf(stderr, "unknown test %s\n", test.c_str());
    return 1;
  }
  for (int i = 2; i < argc; i++) {
    run(argv[i]);
    std::printf("ok %s %s\n", test.c_str(), argv[i]);
  }
  return 0;
}
