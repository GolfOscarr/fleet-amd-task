/* Stub of Fleet's tasks/common/common_header.cuh for a host-side syntax
 * check of the kernels (fleet/tasks/check_syntax.sh). Nothing here runs;
 * it only lets clang++ parse the device code on a machine without HIP.
 * The real header is repos/fleet-chiplet-megakernel/include/mirage/
 * persistent_kernel/tasks/common/common_header.cuh.
 */
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#define __device__
#define __host__
#define __global__
#define __forceinline__ inline
#define __noinline__
#define __shared__
#define __restrict__
#define __launch_bounds__(...)

struct dim3 {
  unsigned x, y, z;
  dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};
static dim3 threadIdx = {0, 0, 0};
static dim3 blockIdx = {0, 0, 0};
static dim3 blockDim = {256, 1, 1};

struct uint4 {
  unsigned x, y, z, w;
};

inline void __syncthreads() {}
inline float __shfl_xor(float v, int, int = 64) { return v; }
inline int __shfl_xor(int v, int, int = 64) { return v; }
inline unsigned __shfl_xor(unsigned v, int, int = 64) { return v; }
inline float __shfl_down(float v, int, int = 64) { return v; }
inline float __shfl(float v, int, int = 64) { return v; }
inline unsigned __float_as_uint(float x) {
  unsigned u;
  __builtin_memcpy(&u, &x, 4);
  return u;
}
inline float __uint_as_float(unsigned u) {
  float x;
  __builtin_memcpy(&x, &u, 4);
  return x;
}
inline unsigned long long clock64() { return 0; }

// The cross-task synchronisation of the last-task pattern (N2 of
// docs/gpu-experiments/04-kernels): clang's amdgcn fence and HIP atomic builtins,
// which exist only for the device target. Macros here, so that the kernel bodies
// parse on the host and the real builtins are used in the fork's build.
#define __HIP_MEMORY_SCOPE_AGENT 3
#define __builtin_amdgcn_fence(order, scope) ((void)0)
#define __hip_atomic_fetch_add(ptr, val, order, scope) (*(ptr) += (val))

constexpr int NUM_THREADS = 256;
constexpr int NUM_THREADS_PER_WARP = 64;

// BF16 storage type with the two conversions the kernels use.
struct bfloat16 {
  uint16_t storage = 0;
  bfloat16() = default;
  explicit bfloat16(float f) {
    unsigned u = __float_as_uint(f);
    unsigned lsb = (u >> 16) & 1u;
    storage = (uint16_t)((u + 0x7FFFu + lsb) >> 16);
  }
  operator float() const { return __uint_as_float((unsigned)storage << 16); }
};
using hip_bfloat16 = bfloat16;

// The real header's spelling (tasks/common/utils.cuh: kernel::bfloat16 =
// type::bfloat16_t) and the worker's dynamic LDS budget (runtime_header.h:
// 60 KiB minus the 3 KiB static reserve on MI300), both used by the launcher.
namespace kernel {
using bfloat16 = ::bfloat16;
}
namespace mirage {
namespace runtime {
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE = 60 * 1024 - 3 * 1024;
}
} // namespace mirage
