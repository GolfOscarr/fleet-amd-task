// stream_read - group E of docs/gpu-bringup/01-plan.md.
//
// Achievable read bandwidth: a read-only reduction over a uint4 buffer, with
// the prefetch depth (how many loads are issued before the first wait) and
// the occupancy both under the caller's control.
//
// E1  stream_read --occupancy full --size 1G --unroll 8 --repeat 3
// E2  stream_read --occupancy one  --size 1G --unroll N     N in 1,2,4,8,16,32
// E3  stream_read --grid 608 --size 1G --unroll N
// E4  stream_read --occupancy full --unroll 8 --size W
//
// Build: hipcc --offload-arch=gfx942 -O2 -std=c++17 stream_read.cu -o stream_read

#define PROBE_NAME "stream_read"
#include "probe_common.h"

#include <chrono>

static const int kBlock = 256;
static const size_t kFullLds = 65536;      // the whole LDS of one CU
static const size_t kFallbackLds = 40960;  // if the runtime refuses the whole

__global__ void fill_kernel(uint4 *buf, size_t n4) {
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n4; i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const unsigned k = static_cast<unsigned>(i);
    buf[i] = make_uint4(k * 2654435761u + 1u, k * 2246822519u + 2u,
                        k * 3266489917u + 3u, k * 668265263u + 5u);
  }
}

// UNROLL independent loads are issued into a register array before any of
// them is consumed, so the wait for the batch is a single s_waitcnt and
// UNROLL loads are in flight per thread. Consuming v[u] inside the load loop
// would serialise them back to one outstanding load.
//
// The accumulator is xor-folded across the wavefront and one lane per wave
// atomicXors it into a global the host reads back as the checksum: without a
// consumer the whole load stream is dead code and the compiler deletes it.
// The fold uses shuffles rather than LDS because the "one" occupancy mode
// claims the entire 64 KiB of LDS at launch to cap residency at one block per
// CU, leaving the kernel none to declare statically.
template <int UNROLL>
__global__ __launch_bounds__(kBlock) void stream_read_kernel(
    const uint4 *__restrict__ src, size_t n4, int passes, unsigned *sink) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  const size_t start =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  uint4 acc = make_uint4(0u, 0u, 0u, 0u);

  // Passes hold the loads per thread constant while the working set changes.
  // The unrolled loop runs only while UNROLL further strides still fit, so at
  // full residency a buffer smaller than about a thread's worth of strides
  // falls entirely into the scalar tail and runs at one load in flight. The
  // working-set sweep would then confound prefetch depth with size; the
  // caller keeps passes times size constant instead.
  for (int p = 0; p < passes; ++p) {
    size_t i = start;
    for (; i + static_cast<size_t>(UNROLL - 1) * stride < n4;
         i += static_cast<size_t>(UNROLL) * stride) {
      uint4 v[UNROLL];
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        v[u] = src[i + static_cast<size_t>(u) * stride];
      }
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        acc.x ^= v[u].x;
        acc.y ^= v[u].y;
        acc.z ^= v[u].z;
        acc.w ^= v[u].w;
      }
    }
    for (; i < n4; i += stride) {
      const uint4 v = src[i];
      acc.x ^= v.x;
      acc.y ^= v.y;
      acc.z ^= v.z;
      acc.w ^= v.w;
    }
    // Rotate between passes. Re-reading the same buffer an even number of
    // times would otherwise xor itself back to zero and leave a checksum that
    // proves nothing about whether the loads happened.
    const uint4 t = acc;
    acc = make_uint4(t.y, t.z, t.w, t.x);
  }

  unsigned r = acc.x ^ acc.y ^ acc.z ^ acc.w;
  for (int off = 32; off > 0; off >>= 1) {
    r ^= __shfl_xor(r, off, 64);
  }
  if ((threadIdx.x & 63u) == 0u) {
    atomicXor(sink, r);
  }
}

typedef void (*LaunchFn)(int grid, size_t lds, const uint4 *src, size_t n4,
                         int passes, unsigned *sink);

template <int UNROLL>
static void launch_variant(int grid, size_t lds, const uint4 *src, size_t n4,
                           int passes, unsigned *sink) {
  hipLaunchKernelGGL((stream_read_kernel<UNROLL>), dim3(grid), dim3(kBlock),
                     lds, nullptr, src, n4, passes, sink);
  HIP_CHECK(hipGetLastError());
}

struct Variant {
  int unroll;
  LaunchFn launch;
  const void *fn;
};

static const Variant kVariants[] = {
    {1, &launch_variant<1>, reinterpret_cast<const void *>(&stream_read_kernel<1>)},
    {2, &launch_variant<2>, reinterpret_cast<const void *>(&stream_read_kernel<2>)},
    {4, &launch_variant<4>, reinterpret_cast<const void *>(&stream_read_kernel<4>)},
    {8, &launch_variant<8>, reinterpret_cast<const void *>(&stream_read_kernel<8>)},
    {16, &launch_variant<16>, reinterpret_cast<const void *>(&stream_read_kernel<16>)},
    {32, &launch_variant<32>, reinterpret_cast<const void *>(&stream_read_kernel<32>)},
};
static const int kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);

// hipFuncSetAttribute refusing a large dynamic LDS request is an answer, not
// a failure, so it is deliberately not wrapped in HIP_CHECK here.
static bool lds_is_usable(const void *fn, size_t lds, int *blocks_per_cu) {
  if (lds > 0) {
    const hipError_t rc = hipFuncSetAttribute(
        fn, hipFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(lds));
    if (rc != hipSuccess) {
      (void)hipGetLastError();
      return false;
    }
  }
  int blocks = 0;
  const hipError_t rc =
      hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, kBlock, lds);
  if (rc != hipSuccess) {
    (void)hipGetLastError();
    return false;
  }
  *blocks_per_cu = blocks;
  return blocks > 0;
}

int main(int argc, char **argv) {
  const char *mode = "full";
  long grid_arg = 0;
  unsigned long long bytes = 1ull << 30;
  long unroll = 8;
  long repeat = 1;
  long passes = 1;

  for (int i = 1; i < argc; ++i) {
    long v = 0;
    if (std::strcmp(argv[i], "--occupancy") == 0 && i + 1 < argc) {
      mode = argv[++i];
      if (std::strcmp(mode, "full") != 0 && std::strcmp(mode, "one") != 0) {
        PROBE_USAGE("  --occupancy takes full or one, got %s\n", mode);
      }
    } else if (std::strcmp(argv[i], "--grid") == 0 && i + 1 < argc) {
      if (!probe_parse_long(argv[++i], &grid_arg) || grid_arg < 1) {
        PROBE_USAGE("  --grid takes a positive block count, got %s\n", argv[i]);
      }
    } else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
      if (!probe_parse_bytes(argv[++i], &bytes) || bytes < 16) {
        PROBE_USAGE("  --size takes N, NM (MiB) or NG (GiB), got %s\n", argv[i]);
      }
    } else if (std::strcmp(argv[i], "--unroll") == 0 && i + 1 < argc) {
      if (!probe_parse_long(argv[++i], &unroll)) {
        PROBE_USAGE("  --unroll takes 1, 2, 4, 8, 16 or 32, got %s\n", argv[i]);
      }
    } else if (std::strcmp(argv[i], "--passes") == 0 && i + 1 < argc) {
      if (!probe_parse_long(argv[++i], &passes) || passes < 1) {
        PROBE_USAGE("  --passes takes a positive count, got %s\n", argv[i]);
      }
    } else if (std::strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
      if (!probe_parse_long(argv[++i], &repeat) || repeat < 1) {
        PROBE_USAGE("  --repeat takes a positive count, got %s\n", argv[i]);
      }
    } else {
      PROBE_USAGE("  stream_read [--occupancy full|one] [--grid G]\n"
                  "              [--size N[M|G]] [--unroll N] [--passes P]\n"
                  "              [--repeat R]\n"
                  "  defaults: --occupancy full --size 1G --unroll 8\n"
                  "            --passes 1 --repeat 1\n");
    }
  }

  int variant = -1;
  for (int i = 0; i < kNumVariants; ++i) {
    if (kVariants[i].unroll == unroll) {
      variant = i;
    }
  }
  if (variant < 0) {
    PROBE_USAGE("  --unroll takes 1, 2, 4, 8, 16 or 32, got %ld\n", unroll);
  }

  hipDeviceProp_t prop;
  probe_init(&prop);
  const int cus = prop.multiProcessorCount;
  const void *fn = kVariants[variant].fn;

  // Grid and dynamic LDS. The LDS request is the lever on residency: a block
  // asking for 64 KiB leaves none for a second block on the same CU, so one
  // block per CU is one wave per SIMD at 256 threads. 65536/K bytes leaves
  // room for exactly K.
  int grid = 0;
  size_t lds = 0;
  const char *occ_label = mode;
  if (grid_arg > 0) {
    occ_label = "grid";
    grid = static_cast<int>(grid_arg);
    const int k = (grid + cus - 1) / cus;
    lds = (k <= 1) ? kFullLds : (kFullLds / static_cast<size_t>(k)) & ~size_t(511);
  } else if (std::strcmp(mode, "one") == 0) {
    grid = cus;
    lds = kFullLds;
  } else {
    int blocks_per_cu = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_cu, fn,
                                                           kBlock, 0));
    if (blocks_per_cu < 1) {
      PROBE_FAIL("hipOccupancyMaxActiveBlocksPerMultiprocessor returned 0");
    }
    grid = blocks_per_cu * cus;
    lds = 0;
  }

  // The register footprint of the chosen unroll is reported next to the
  // bandwidth: UNROLL uint4 in flight needs 4*UNROLL registers for the data
  // alone, so a count well below that means the compiler split the batch and
  // the requested prefetch depth was not reached.
  hipFuncAttributes attr;
  HIP_CHECK(hipFuncGetAttributes(&attr, fn));

  int blocks_per_cu_query = 0;
  if (!lds_is_usable(fn, lds, &blocks_per_cu_query)) {
    if (lds == kFullLds && lds_is_usable(fn, kFallbackLds, &blocks_per_cu_query)) {
      lds = kFallbackLds;  // reported in lds_bytes below
    } else {
      PROBE_FAIL("no block fits with %zu bytes of dynamic LDS", lds);
    }
  }

  const size_t n4 = static_cast<size_t>(bytes / 16);
  const size_t buf_bytes = n4 * 16;
  uint4 *d_src = nullptr;
  unsigned *d_sink = nullptr;
  HIP_CHECK(hipMalloc(&d_src, buf_bytes));
  HIP_CHECK(hipMalloc(&d_sink, sizeof(unsigned)));
  hipLaunchKernelGGL(fill_kernel, dim3(1024), dim3(256), 0, nullptr, d_src, n4);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start;
  hipEvent_t stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  // One untimed launch to size the warm-up, then about a second of launches:
  // clocks under a hypervisor can sit in a low state until load arrives.
  HIP_CHECK(hipMemset(d_sink, 0, sizeof(unsigned)));
  HIP_CHECK(hipEventRecord(start, nullptr));
  kVariants[variant].launch(grid, lds, d_src, n4, static_cast<int>(passes),
                            d_sink);
  HIP_CHECK(hipEventRecord(stop, nullptr));
  HIP_CHECK(hipEventSynchronize(stop));
  float probe_ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&probe_ms, start, stop));

  // About a second of launches, however many that is. A fixed ceiling would
  // silently under-warm a small working set, which is the case where clocks
  // sitting in a low state matter most.
  int warmups = 3;
  if (probe_ms > 0.0f) {
    const int needed = static_cast<int>(1000.0f / probe_ms) + 1;
    if (needed > warmups) {
      warmups = needed;
    }
  }
  for (int i = 0; i < warmups; ++i) {
    kVariants[variant].launch(grid, lds, d_src, n4, static_cast<int>(passes),
                            d_sink);
  }
  HIP_CHECK(hipDeviceSynchronize());

  for (long r = 0; r < repeat; ++r) {
    HIP_CHECK(hipMemset(d_sink, 0, sizeof(unsigned)));
    HIP_CHECK(hipEventRecord(start, nullptr));
    kVariants[variant].launch(grid, lds, d_src, n4, static_cast<int>(passes),
                            d_sink);
    HIP_CHECK(hipEventRecord(stop, nullptr));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    unsigned checksum = 0;
    HIP_CHECK(hipMemcpy(&checksum, d_sink, sizeof(checksum),
                        hipMemcpyDeviceToHost));

    // Decimal GB/s over what was actually read: passes times the buffer.
    // bytes / (ms * 1e-3) / 1e9 == bytes / ms / 1e6.
    const double gbps =
        static_cast<double>(buf_bytes) * static_cast<double>(passes) / ms / 1.0e6;
    std::printf("stream_read occupancy=%s grid=%d block=%d lds_bytes=%zu "
                "blocks_per_cu_query=%d size_mib=%llu unroll=%ld ms=%.6f "
                "gbps=%.3f tbps=%.4f checksum=%08x vgprs=%d passes=%ld\n",
                occ_label, grid, kBlock, lds, blocks_per_cu_query,
                static_cast<unsigned long long>(buf_bytes >> 20), unroll,
                static_cast<double>(ms), gbps, gbps / 1000.0, checksum,
                attr.numRegs, passes);
  }

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(d_sink));
  HIP_CHECK(hipFree(d_src));
  return 0;
}
