// fence_probe - group G of docs/gpu/01-bringup/01-plan.md.
//
// What the machine's hipcc lowers each fence to on gfx942. The design rests
// on agent-scope release becoming buffer_wbl2 sc1 and agent-scope acquire
// becoming buffer_inv sc1 (docs/mi300x/03-memory-model.md, MAJ-3); the
// __threadfence() kernel settles a contradiction the repo still holds, since
// 03-memory-model.md item 5 calls it agent scope while
// env/offline_gfx942/fences.txt attributes system-scope sites to it.
//
// Each kernel has a volatile global store before the fence and a volatile
// global load after it, so the backend has something to order and cannot drop
// the fence. The output of interest is the assembly, not the run:
//
//   hipcc --offload-arch=gfx942 -S --offload-device-only -O2 -std=c++17 \
//       fence_probe.cu -o fence_probe.s
//   python3 fence_grep.py fence_probe.s
//
// The file also links, so the same source is an executable that proves the
// four kernels launch:
//
//   hipcc --offload-arch=gfx942 -O2 -std=c++17 fence_probe.cu -o fence_probe

#define PROBE_NAME "fence_probe"
#include "probe_common.h"

__global__ void k_release(volatile unsigned *p, unsigned *q) {
  p[0] = 1;
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
  q[0] = p[1];
}

__global__ void k_acquire(volatile unsigned *p, unsigned *q) {
  p[0] = 1;
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  q[0] = p[1];
}

__global__ void k_threadfence(volatile unsigned *p, unsigned *q) {
  p[0] = 1;
  __threadfence();
  q[0] = p[1];
}

__global__ void k_atomic_load(volatile unsigned *p, unsigned *q) {
  p[0] = 1;
  const unsigned v =
      __hip_atomic_load(q, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
  p[2] = v;
}

int main(int argc, char **argv) {
  if (argc != 1) {
    PROBE_USAGE("  fence_probe            launches each of the four kernels once\n"
                "  the measurement is the assembly; see fence_grep.py\n");
  }

  hipDeviceProp_t prop;
  probe_init(&prop);

  unsigned *p = nullptr;
  unsigned *q = nullptr;
  HIP_CHECK(hipMalloc(&p, 4 * sizeof(unsigned)));
  HIP_CHECK(hipMalloc(&q, 4 * sizeof(unsigned)));
  HIP_CHECK(hipMemset(p, 0, 4 * sizeof(unsigned)));
  HIP_CHECK(hipMemset(q, 0, 4 * sizeof(unsigned)));

  hipLaunchKernelGGL(k_release, dim3(1), dim3(1), 0, nullptr, p, q);
  HIP_CHECK(hipGetLastError());
  hipLaunchKernelGGL(k_acquire, dim3(1), dim3(1), 0, nullptr, p, q);
  HIP_CHECK(hipGetLastError());
  hipLaunchKernelGGL(k_threadfence, dim3(1), dim3(1), 0, nullptr, p, q);
  HIP_CHECK(hipGetLastError());
  hipLaunchKernelGGL(k_atomic_load, dim3(1), dim3(1), 0, nullptr, p, q);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::printf("fence_probe ran=4\n");

  HIP_CHECK(hipFree(q));
  HIP_CHECK(hipFree(p));
  return 0;
}
