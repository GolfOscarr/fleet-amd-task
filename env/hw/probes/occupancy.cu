// occupancy - group B of docs/gpu/01-bringup/01-plan.md.
//
// B10: how many 256-thread blocks of a given register footprint are resident
// per CU. The megakernel launches 296 worker blocks plus 8 scheduler blocks
// and deadlocks if they are not all co-resident, so this is the residency the
// design rests on. The worker kernel is 182 VGPRs by the compiler's report
// (env/offline_gfx942/resources.txt) and asks for 58,368 bytes of dynamic
// LDS at launch, which is why --dynamic-lds exists: the LDS request, not the
// registers, may be what limits the block count.
//
// B11: the rate of the constant-rate counter s_memrealtime reads, checked
// against a host-timed 100 ms spin. Every in-kernel time number in this
// project divides ticks by this rate.
//
//   occupancy --vgprs 182 [--dynamic-lds 58368]
//   occupancy --clock
//   occupancy                      both, with the defaults
//
// Build: hipcc --offload-arch=gfx942 -O2 -std=c++17 occupancy.cu -o occupancy

#define PROBE_NAME "occupancy"
#include "probe_common.h"

#include <chrono>

// A kernel whose live register footprint is set by a template array length.
// Every element is written before a runtime-trip-count loop and read after it,
// and inside that loop an empty asm statement claims each element with a "+v"
// constraint. The asm is what makes the footprint predictable: it gives the
// scheduler nothing to pipeline and nothing to rematerialise, so the allocator
// takes exactly the N values plus four for addressing rather than filling
// whatever budget it is given. Without it the allocator jumps from 166 to 254
// registers between adjacent array lengths and the requested footprint cannot
// be hit at all. __launch_bounds__(256, 1) caps the block at 256 threads and
// asks for no more than one wave per SIMD, which is what allows a count above
// the 168 registers three waves per SIMD would leave.
//
// These kernels exist to be measured, not run: the probe only queries their
// attributes and their occupancy.
template <int N>
__global__ __launch_bounds__(256, 1) void vgpr_kernel(const float *in,
                                                      float *out, int iters) {
  float r[N];
#pragma unroll
  for (int i = 0; i < N; ++i) {
    r[i] = in[threadIdx.x] + static_cast<float>(i);
  }
  for (int t = 0; t < iters; ++t) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      asm volatile("" : "+v"(r[i]));
    }
  }
  float s = 0.0f;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    s += r[i];
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// The ladder the probe chooses from: array lengths 60 to 252 in steps of 8,
// which is the register allocation granularity, so every request in 64..256
// lands within four registers of what was asked for. The true count is read
// back with hipFuncGetAttributes rather than assumed.
#define VGPR_VARIANT(LEN) reinterpret_cast<const void *>(&vgpr_kernel<LEN>)

static const void *const kVariants[] = {
    VGPR_VARIANT(60), VGPR_VARIANT(68), VGPR_VARIANT(76), VGPR_VARIANT(84), VGPR_VARIANT(92),
    VGPR_VARIANT(100), VGPR_VARIANT(108), VGPR_VARIANT(116), VGPR_VARIANT(124), VGPR_VARIANT(132),
    VGPR_VARIANT(140), VGPR_VARIANT(148), VGPR_VARIANT(156), VGPR_VARIANT(164), VGPR_VARIANT(172),
    VGPR_VARIANT(180), VGPR_VARIANT(188), VGPR_VARIANT(196), VGPR_VARIANT(204), VGPR_VARIANT(212),
    VGPR_VARIANT(220), VGPR_VARIANT(228), VGPR_VARIANT(236), VGPR_VARIANT(244), VGPR_VARIANT(252),
};
static const int kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);

// What the Fleet runtime asks for at launch: MAX_DYNAMIC_SHARED_MEMORY_SIZE
// of runtime_header.h, 60*1024 - 3*1024 bytes.
static const int kWorkerDynamicLds = 58368;

// Spins on s_memrealtime for a fixed number of ticks and reports how many it
// actually observed. One thread: the point is the counter, not the machine.
__global__ void wallclock_spin_kernel(unsigned long long target,
                                      unsigned long long *out) {
  const unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
  unsigned long long t1 = t0;
  while (t1 - t0 < target) {
    t1 = __builtin_amdgcn_s_memrealtime();
  }
  out[0] = t1 - t0;
}

static void run_occupancy(const hipDeviceProp_t &prop, int requested_vgprs,
                          int dynamic_lds) {
  int best = 0;
  int best_regs = 0;
  int best_delta = 0;
  for (int i = 0; i < kNumVariants; ++i) {
    hipFuncAttributes attr;
    HIP_CHECK(hipFuncGetAttributes(&attr, kVariants[i]));
    const int delta = attr.numRegs > requested_vgprs
                          ? attr.numRegs - requested_vgprs
                          : requested_vgprs - attr.numRegs;
    if (i == 0 || delta < best_delta) {
      best = i;
      best_regs = attr.numRegs;
      best_delta = delta;
    }
  }

  // A request above the default per-block limit has to be opted into before
  // the occupancy query will honour it. A refusal is an answer, not a
  // failure: the query below then reports what actually fits, so the error is
  // cleared rather than raised.
  if (dynamic_lds > 0) {
    const hipError_t rc =
        hipFuncSetAttribute(kVariants[best],
                            hipFuncAttributeMaxDynamicSharedMemorySize,
                            dynamic_lds);
    if (rc != hipSuccess) {
      (void)hipGetLastError();
    }
  }

  // The query is allowed to fail only when there is an LDS request to blame.
  // A block that does not fit is a measurement, not a missing measurement, so
  // it must reach stdout as blocks_per_cu=0: a grid that is not co-resident
  // deadlocks the megakernel and has to be filed rather than look like a row
  // the VM could not produce. With no LDS request a failure here is a real
  // fault and still exits 2.
  int blocks_per_cu = 0;
  const hipError_t occ_rc = hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_cu, kVariants[best], 256, static_cast<size_t>(dynamic_lds));
  if (occ_rc != hipSuccess) {
    if (dynamic_lds == 0) {
      HIP_CHECK(occ_rc);
    }
    (void)hipGetLastError();
    std::fprintf(stderr,
                 "%s warning: no block fits with %d bytes of dynamic LDS (%s); "
                 "reporting blocks_per_cu=0\n",
                 PROBE_NAME, dynamic_lds, hipGetErrorString(occ_rc));
    blocks_per_cu = 0;
  }

  // 256 threads is 4 wavefronts of 64 spread over the CU's 4 SIMDs, so one
  // resident block is one wave per SIMD and the two numbers coincide.
  const int waves_per_simd = blocks_per_cu;
  std::printf(
      "occupancy requested_vgprs=%d actual_vgprs=%d dynamic_lds=%d "
      "blocks_per_cu=%d waves_per_simd=%d cus=%d coresident_blocks=%d\n",
      requested_vgprs, best_regs, dynamic_lds, blocks_per_cu, waves_per_simd,
      prop.multiProcessorCount, blocks_per_cu * prop.multiProcessorCount);
}

static void run_clock() {
  const int rate_khz = probe_wall_clock_khz();
  // 100 ms expressed in the counter's own ticks.
  const unsigned long long target =
      static_cast<unsigned long long>(rate_khz) * 100ull;

  unsigned long long *d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, sizeof(unsigned long long)));
  HIP_CHECK(hipMemset(d_out, 0, sizeof(unsigned long long)));

  hipEvent_t start;
  hipEvent_t stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  const auto host_t0 = std::chrono::steady_clock::now();
  HIP_CHECK(hipEventRecord(start, nullptr));
  hipLaunchKernelGGL(wallclock_spin_kernel, dim3(1), dim3(1), 0, nullptr,
                     target, d_out);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipEventRecord(stop, nullptr));
  HIP_CHECK(hipEventSynchronize(stop));
  const auto host_t1 = std::chrono::steady_clock::now();

  float event_ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&event_ms, start, stop));

  unsigned long long ticks = 0;
  HIP_CHECK(hipMemcpy(&ticks, d_out, sizeof(ticks), hipMemcpyDeviceToHost));

  const double host_ns =
      static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                              host_t1 - host_t0)
                              .count());
  // Ticks the counter should have produced over the host interval, if the
  // attribute is right; a ratio near 1 confirms it.
  const double expected =
      static_cast<double>(rate_khz) * 1.0e3 * host_ns * 1.0e-9;
  const double ratio = expected > 0.0 ? static_cast<double>(ticks) / expected : 0.0;

  std::printf("wallclock rate_khz=%d ticks=%llu host_ns=%.0f ratio=%.6f "
              "event_ms=%.6f\n",
              rate_khz, ticks, host_ns, ratio, static_cast<double>(event_ms));

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(d_out));
}

int main(int argc, char **argv) {
  int requested_vgprs = 182;
  int dynamic_lds = 0;
  bool want_occupancy = true;
  bool want_clock = true;
  bool run_both = (argc == 1);

  for (int i = 1; i < argc; ++i) {
    long v = 0;
    if (std::strcmp(argv[i], "--vgprs") == 0 && i + 1 < argc) {
      if (!probe_parse_long(argv[++i], &v) || v < 64 || v > 256) {
        PROBE_USAGE("  --vgprs N takes 64..256, got %s\n", argv[i]);
      }
      requested_vgprs = static_cast<int>(v);
      want_clock = false;
    } else if (std::strcmp(argv[i], "--dynamic-lds") == 0 && i + 1 < argc) {
      if (!probe_parse_long(argv[++i], &v) || v < 0 || v > 65536) {
        PROBE_USAGE("  --dynamic-lds N takes 0..65536 bytes, got %s\n", argv[i]);
      }
      dynamic_lds = static_cast<int>(v);
      want_clock = false;
    } else if (std::strcmp(argv[i], "--clock") == 0) {
      want_occupancy = false;
      want_clock = true;
    } else {
      PROBE_USAGE("  occupancy [--vgprs N] [--dynamic-lds BYTES]\n"
                  "  occupancy --clock\n"
                  "  occupancy                  --vgprs 182 at 0 and at 58368\n"
                  "                             bytes of dynamic LDS, then --clock\n");
    }
  }

  hipDeviceProp_t prop;
  probe_init(&prop);

  if (want_occupancy) {
    run_occupancy(prop, requested_vgprs, dynamic_lds);
    if (run_both) {
      // The runtime launches the worker kernel with this many bytes of
      // dynamic LDS (runtime_header.h, MAX_DYNAMIC_SHARED_MEMORY_SIZE =
      // 60*1024 - 3*1024), so its residency is LDS-limited rather than
      // register-limited. With no arguments the probe reports both cases.
      run_occupancy(prop, requested_vgprs, kWorkerDynamicLds);
    }
  }
  if (want_clock) {
    run_clock();
  }
  return 0;
}
