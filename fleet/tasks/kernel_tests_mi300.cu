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
 * The defines are the ones persistent_kernel.py passes on its ROCm path.
 *
 * Usage: kernel_tests <test> <dir> [<dir> ...]
 *   test  mla_prep | mla_attend | mla_merge_uv | moe_router | copy
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
#include "tasks/mi300/moe_router_mi300.cuh"
#include "tasks/mi300/copy_mi300.cuh"

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
      qkva, w_kv_norm, w_uk, cos, sin, c_kv, k_pe, ql_nope, q_pe, meta.step[0], 1e-6f);
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
      + (size_t)xcd * partials_xcd_offset_rows * NH * (D_C + 1);
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

__global__ __launch_bounds__(256, 1) void k_moe_router(void const *h,
                                                       void const *w_gate,
                                                       void *topk_w,
                                                       void *routing,
                                                       void *mask,
                                                       void *logits,
                                                       void *route_log,
                                                       Meta meta,
                                                       int layer_index,
                                                       float scaling) {
  kernel::moe_router_mi300_task_impl<bf16, HIDDEN, N_EXPERTS, N_FORCED, TOPK, ROUTE_STEPS,
                                     ROUTE_LAYERS>(
      h, w_gate, topk_w, routing, mask, logits, route_log, meta.step[0], meta.prompt_length[0],
      layer_index, scaling);
}

__global__ __launch_bounds__(256, 1) void k_copy(void const *x, void *y) {
  kernel::copy_mi300_task_impl<bf16, HIDDEN>(x, y);
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
    {"partials", 0, true},   // n_splits * NH * (D_C + 1) * 4, from params
};
static const Spec SPEC_MLA_ATTEND_SCORES = {"scores", (size_t)NH * S_MAX * 4, true};

static const Spec SPEC_MLA_MERGE_UV[] = {
    {"partials", 0, false},  // n_splits * NH * (D_C + 1) * 4, from params
    {"w_uv", (size_t)NH * D_V * D_C * 2, false},
    {"attn", (size_t)NH * D_V * 2, true},
};

static const Spec SPEC_MOE_ROUTER[] = {
    {"h", (size_t)HIDDEN * 2, false},
    {"w_gate", (size_t)N_EXPERTS * HIDDEN * 2, false},
    {"topk_w", (size_t)N_SLOTS * 4, true},
    {"routing", (size_t)N_TOTAL * 4, true},
    {"mask", (size_t)(N_TOTAL + 1) * 4, true},
    {"logits", (size_t)N_EXPERTS * 4, true},
    {"route_log", (size_t)ROUTE_STEPS * ROUTE_LAYERS * N_SLOTS * 4, true},
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
  hipLaunchKernelGGL(k_mla_prep, dim3(1), dim3(256), SMEM_BYTES, 0,
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
  b.specs[4].bytes = (size_t)n_splits * NH * (D_C + 1) * 4;
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
  b.store_outputs();
}

void run_mla_merge_uv(std::string const &dir) {
  Params p = read_params(dir);
  int split = (int)param(p, "split");
  int n_splits = (int)param(p, "n_splits");
  Buffers b{dir, specs_of(SPEC_MLA_MERGE_UV), {}};
  b.specs[0].bytes = (size_t)n_splits * NH * (D_C + 1) * 4;
  b.load();
  DeviceMeta m((int)param(p, "step"), (int)param_or(p, "prompt_length", 0));
  allow_full_lds(k_mla_merge_uv);
  hipLaunchKernelGGL(k_mla_merge_uv, dim3(XCDS, HEADS_PER_XCD), dim3(256), SMEM_BYTES, 0,
                     b.get("partials"), b.get("w_uv"), b.get("attn"), m.meta, split, n_splits);
  finish_launch();
  b.store_outputs();
}

void run_moe_router(std::string const &dir) {
  Params p = read_params(dir);
  Buffers b{dir, specs_of(SPEC_MOE_ROUTER), {}};
  b.load();
  DeviceMeta m((int)param(p, "step"), (int)param(p, "prompt_length"));
  allow_full_lds(k_moe_router);
  hipLaunchKernelGGL(k_moe_router, dim3(1), dim3(256), SMEM_BYTES, 0,
                     b.get("h"), b.get("w_gate"), b.get("topk_w"), b.get("routing"), b.get("mask"),
                     b.get("logits"), b.get("route_log"), m.meta, (int)param(p, "layer_index"),
                     float_from_bits(param(p, "scaling_bits")));
  finish_launch();
  b.store_outputs();
}

void run_copy(std::string const &dir) {
  (void)read_params(dir);
  Buffers b{dir, specs_of(SPEC_COPY), {}};
  b.load();
  allow_full_lds(k_copy);
  hipLaunchKernelGGL(k_copy, dim3(1), dim3(256), SMEM_BYTES, 0, b.get("x"), b.get("y"));
  finish_launch();
  b.store_outputs();
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <mla_prep|mla_attend|mla_merge_uv|moe_router|copy> <dir>...\n",
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
  } else if (test == "moe_router") {
    run = run_moe_router;
  } else if (test == "copy") {
    run = run_copy;
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
