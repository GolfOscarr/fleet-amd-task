// copy_bytes - group I of docs/gpu-experiments/01-bringup/01-plan.md.
//
// A device-to-device copy of exactly 1 GiB, wrapped by rocprofv3 so the
// counter arithmetic can be checked against traffic that is known in advance:
// 1 GiB read and 1 GiB written. I1 to I5 of the checklist compare the
// decomposition of docs/mi300x/06-profiling.md against the flat 64 bytes per
// request that harness/measure.py currently assumes.
//
//   rocprofv3 --pmc TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_32B_sum TCC_EA0_WRREQ_sum \
//       TCC_EA0_WRREQ_64B_sum TCC_HIT_sum TCC_MISS_sum -- copy_bytes
//   rocprofv3 --kernel-trace -- copy_bytes
//
// There is exactly one copy_kernel dispatch, and it is the 1 GiB copy. The
// warm-up runs a separate small kernel with its own name so it can never be
// mistaken for the measured dispatch in the profiler's output.
//
// Build: hipcc --offload-arch=gfx942 -O2 -std=c++17 copy_bytes.cu -o copy_bytes

#define PROBE_NAME "copy_bytes"
#include "probe_common.h"

#include <vector>

static const int kBlock = 256;
static const size_t kWarmupBytes = 1u << 20;
static const size_t kSamples = 4096;   // uint4 elements verified
static const int kWindows = 8;        // spread over this many transfers

__global__ void fill_kernel(uint4 *buf, size_t n4) {
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n4; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const unsigned k = static_cast<unsigned>(i);
    buf[i] = make_uint4(k, k * 2u + 1u, k * 3u + 2u, k * 5u + 3u);
  }
}

__global__ __launch_bounds__(kBlock) void warmup_kernel(
    const uint4 *__restrict__ src, uint4 *__restrict__ dst, size_t n4) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n4) {
    dst[i] = src[i];
  }
}

// The measured kernel: one uint4 in and one uint4 out per thread, no loop and
// no reuse, so the traffic the profiler should report is exactly the buffer
// once in each direction.
__global__ __launch_bounds__(kBlock) void copy_kernel(
    const uint4 *__restrict__ src, uint4 *__restrict__ dst, size_t n4) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n4) {
    dst[i] = src[i];
  }
}

int main(int argc, char **argv) {
  unsigned long long bytes = 1ull << 30;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--bytes") == 0 && i + 1 < argc) {
      if (!probe_parse_bytes(argv[++i], &bytes) || bytes < 16) {
        PROBE_USAGE("  --bytes takes N, NM (MiB) or NG (GiB), got %s\n", argv[i]);
      }
    } else {
      PROBE_USAGE("  copy_bytes [--bytes N[M|G]]\n"
                  "  default: --bytes 1G\n");
    }
  }

  hipDeviceProp_t prop;
  probe_init(&prop);

  const size_t n4 = static_cast<size_t>(bytes / 16);
  const size_t buf_bytes = n4 * 16;
  const size_t warm_n4 = kWarmupBytes / 16;

  uint4 *d_src = nullptr;
  uint4 *d_dst = nullptr;
  uint4 *d_warm_src = nullptr;
  uint4 *d_warm_dst = nullptr;
  HIP_CHECK(hipMalloc(&d_src, buf_bytes));
  HIP_CHECK(hipMalloc(&d_dst, buf_bytes));
  HIP_CHECK(hipMalloc(&d_warm_src, kWarmupBytes));
  HIP_CHECK(hipMalloc(&d_warm_dst, kWarmupBytes));

  hipLaunchKernelGGL(fill_kernel, dim3(1024), dim3(kBlock), 0, nullptr, d_src,
                     n4);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemset(d_dst, 0, buf_bytes));
  HIP_CHECK(hipMemset(d_warm_src, 0, kWarmupBytes));
  HIP_CHECK(hipDeviceSynchronize());

  // Warm-up on its own buffers: clocks under a hypervisor can sit in a low
  // state until load arrives, and the first dispatch also pays the code
  // object load.
  hipLaunchKernelGGL(warmup_kernel,
                     dim3(static_cast<unsigned>((warm_n4 + kBlock - 1) / kBlock)),
                     dim3(kBlock), 0, nullptr, d_warm_src, d_warm_dst, warm_n4);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start;
  hipEvent_t stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  const unsigned grid = static_cast<unsigned>((n4 + kBlock - 1) / kBlock);
  HIP_CHECK(hipEventRecord(start, nullptr));
  hipLaunchKernelGGL(copy_kernel, dim3(grid), dim3(kBlock), 0, nullptr, d_src,
                     d_dst, n4);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipEventRecord(stop, nullptr));
  HIP_CHECK(hipEventSynchronize(stop));

  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

  // Sample the output rather than all of it: pulling 1 GiB back to the host
  // would cost more than the measurement. Eight contiguous windows, one
  // transfer each, so the profiler's kernel trace stays free of blit work.
  const size_t window = kSamples / kWindows;
  std::vector<uint4> sample(window);
  for (int w = 0; w < kWindows; ++w) {
    size_t base = (n4 / kWindows) * static_cast<size_t>(w);
    if (base + window > n4) {
      base = n4 > window ? n4 - window : 0;
    }
    const size_t count = n4 < window ? n4 : window;
    HIP_CHECK(hipMemcpy(sample.data(), d_dst + base, count * sizeof(uint4),
                        hipMemcpyDeviceToHost));
    for (size_t j = 0; j < count; ++j) {
      const unsigned k = static_cast<unsigned>(base + j);
      if (sample[j].x != k || sample[j].y != k * 2u + 1u ||
          sample[j].z != k * 3u + 2u || sample[j].w != k * 5u + 3u) {
        PROBE_FAIL("copy mismatch at uint4 index %zu", base + j);
      }
    }
  }

  // Read plus write: the copy moves the buffer once in each direction.
  const double gbps = 2.0 * static_cast<double>(buf_bytes) / ms / 1.0e6;
  std::printf("copy_bytes bytes=%llu ms=%.6f gbps=%.3f\n",
              static_cast<unsigned long long>(buf_bytes),
              static_cast<double>(ms), gbps);

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(d_warm_dst));
  HIP_CHECK(hipFree(d_warm_src));
  HIP_CHECK(hipFree(d_dst));
  HIP_CHECK(hipFree(d_src));
  return 0;
}
