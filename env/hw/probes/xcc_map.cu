// xcc_map - group F of docs/gpu-experiments/01-bringup/01-plan.md.
//
// Which XCD each workgroup of a plain launch lands on. The design puts eight
// consecutive gang tasks on eight distinct XCDs and the Fleet runtime indexes
// its scheduler queues by the same rule, so the round-robin
// "xcd == blockIdx.x mod 8" is the assumption under F1 to F6 and MIN-25.
//
// F1  xcc_map --grid 304
// F2  xcc_map --grid 296
// F3  xcc_map --grid 8   and  xcc_map --grid 8 --concurrent
// F4  xcc_map --grid 608 | 1000 | 37
//
// Build: hipcc --offload-arch=gfx942 -O2 -std=c++17 xcc_map.cu -o xcc_map

#define PROBE_NAME "xcc_map"
#include "probe_common.h"

#include <vector>

static const int kNumXcds = 8;
static const unsigned kUnset = 0xffffffffu;

// Thread 0 of every block records the XCD it is running on next to its own
// block id. The block id is stored rather than assumed so a block that never
// ran shows up as the kUnset sentinel instead of a plausible-looking zero.
__global__ void xcc_map_kernel(uint2 *out) {
  if (threadIdx.x == 0) {
    out[blockIdx.x] =
        make_uint2(static_cast<unsigned>(probe_xcc_id()), blockIdx.x);
  }
}

// Occupies the machine for --concurrent: the question is whether the small
// scheduler grid still lands one block per XCD when the worker grid is
// already running on another stream.
__global__ void busy_kernel(unsigned long long ticks) {
  const unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
  while (__builtin_amdgcn_s_memrealtime() - t0 < ticks) {
  }
}

int main(int argc, char **argv) {
  long grid = 304;
  long runs = 3;
  bool concurrent = false;
  const char *dump = nullptr;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--grid") == 0 && i + 1 < argc) {
      if (!probe_parse_long(argv[++i], &grid) || grid < 1) {
        PROBE_USAGE("  --grid takes a positive block count, got %s\n", argv[i]);
      }
    } else if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
      if (!probe_parse_long(argv[++i], &runs) || runs < 1) {
        PROBE_USAGE("  --runs takes a positive count, got %s\n", argv[i]);
      }
    } else if (std::strcmp(argv[i], "--concurrent") == 0) {
      concurrent = true;
    } else if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
      dump = argv[++i];
    } else {
      PROBE_USAGE("  xcc_map [--grid G] [--runs R] [--concurrent] [--dump FILE]\n"
                  "  defaults: --grid 304 --runs 3\n");
    }
  }

  hipDeviceProp_t prop;
  probe_init(&prop);
  const int rate_khz = probe_wall_clock_khz();
  const unsigned long long busy_ticks =
      static_cast<unsigned long long>(rate_khz) * 50ull;  // about 50 ms

  uint2 *d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_out, static_cast<size_t>(grid) * sizeof(uint2)));

  hipStream_t probe_stream;
  hipStream_t busy_stream;
  HIP_CHECK(hipStreamCreate(&probe_stream));
  HIP_CHECK(hipStreamCreate(&busy_stream));

  std::vector<uint2> host(static_cast<size_t>(grid));
  std::vector<unsigned> first_run(static_cast<size_t>(grid), kUnset);

  for (long r = 0; r < runs; ++r) {
    HIP_CHECK(hipMemset(d_out, 0xff, static_cast<size_t>(grid) * sizeof(uint2)));
    if (concurrent) {
      hipLaunchKernelGGL(busy_kernel, dim3(296), dim3(64), 0, busy_stream,
                         busy_ticks);
      HIP_CHECK(hipGetLastError());
    }
    hipLaunchKernelGGL(xcc_map_kernel, dim3(static_cast<unsigned>(grid)),
                       dim3(64), 0, probe_stream, d_out);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamSynchronize(probe_stream));
    HIP_CHECK(hipStreamSynchronize(busy_stream));
    HIP_CHECK(hipMemcpy(host.data(), d_out,
                        static_cast<size_t>(grid) * sizeof(uint2),
                        hipMemcpyDeviceToHost));

    int per_xcd[kNumXcds] = {0};
    int distinct = 0;
    int in_range = 1;
    int violations = 0;
    int stable = 1;
    for (long b = 0; b < grid; ++b) {
      const unsigned xcd = host[static_cast<size_t>(b)].x;
      if (host[static_cast<size_t>(b)].y != static_cast<unsigned>(b)) {
        PROBE_FAIL("block %ld did not report (grid %ld)", b, grid);
      }
      if (xcd < static_cast<unsigned>(kNumXcds)) {
        per_xcd[xcd]++;
      } else {
        in_range = 0;
      }
      if (xcd != static_cast<unsigned>(b % kNumXcds)) {
        violations++;
      }
      if (r == 0) {
        first_run[static_cast<size_t>(b)] = xcd;
      } else if (first_run[static_cast<size_t>(b)] != xcd) {
        stable = 0;
      }
    }
    for (int x = 0; x < kNumXcds; ++x) {
      if (per_xcd[x] > 0) {
        distinct++;
      }
    }

    std::printf("xcc_map grid=%ld run=%ld concurrent=%d distinct=%d "
                "ids_in_range=%d rule_mod8_violations=%d per_xcd=",
                grid, r, concurrent ? 1 : 0, distinct, in_range, violations);
    for (int x = 0; x < kNumXcds; ++x) {
      std::printf("%s%d", x == 0 ? "" : ",", per_xcd[x]);
    }
    std::printf(" stable_vs_run0=%d\n", stable);
  }

  if (dump != nullptr) {
    std::FILE *f = std::fopen(dump, "w");
    if (f == nullptr) {
      PROBE_FAIL("cannot open %s for writing", dump);
    }
    for (long b = 0; b < grid; ++b) {
      std::fprintf(f, "%ld,%u\n", b, host[static_cast<size_t>(b)].x);
    }
    std::fclose(f);
  }

  HIP_CHECK(hipStreamDestroy(probe_stream));
  HIP_CHECK(hipStreamDestroy(busy_stream));
  HIP_CHECK(hipFree(d_out));
  return 0;
}
