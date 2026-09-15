// chase - group H of docs/gpu-bringup/01-plan.md.
//
// Three measurements that the synchronization argument of the design turns
// on, all reported in nanoseconds:
//
//   --size    dependent-load latency over a working set (H1 L2, H2 whatever
//             sits above it, H3 HBM). The prefetch-depth analysis assumes an
//             HBM latency somewhere between 250 ns and 2 us (MIN-21).
//   --fence   the cost of one agent-scope fence, as the difference against
//             the same loop with a workgroup-scope fence (H4, H5, and with
//             --xcds 8 the eight-XCD case H6, docs/mi300x Q5, MIN-16).
//   --pingpong  the cross-XCD one-way latency: a release store on one XCD and
//             an acquire poll on another, which is the t_b of MAJ-5 (H7).
//
//   chase --size 1G
//   chase --fence release [--xcds 8]
//   chase --pingpong [--fine]
//   chase                    all of the above, at 1M, 64M and 1G
//
// Build: hipcc --offload-arch=gfx942 -O2 -std=c++17 chase.cu -o chase

#define PROBE_NAME "chase"
#include "probe_common.h"

#include <chrono>
#include <vector>

static const int kNumXcds = 8;
static const unsigned kUnclaimed = 0xffffffffu;
static const unsigned long long kChaseLoads = 1ull << 20;
static const unsigned long long kChaseWarmup = 1ull << 16;
static const unsigned kFenceIters = 100000u;
static const unsigned kPingpongRounds = 10000u;
static const int kPingpongGrid = 64;
static const int kFenceGridXcds = 64;

// ---------------------------------------------------------------- --size

// One thread follows the permutation. Each load's address is the previous
// load's result, so the hardware cannot overlap them and the loop time is
// iterations times the memory latency.
__global__ void chase_kernel(const unsigned long long *__restrict__ buf,
                             unsigned long long iters,
                             unsigned long long *out) {
  const unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
  unsigned long long i = 0;
  for (unsigned long long k = 0; k < iters; ++k) {
    i = buf[i];
  }
  const unsigned long long t1 = __builtin_amdgcn_s_memrealtime();
  out[0] = t1 - t0;
  out[1] = i;  // consumed by the host, so the chain cannot be deleted
}

// splitmix64 with a fixed seed: the permutation must be identical on every
// machine and every libc, so the standard library's generator is not used.
static unsigned long long next_random(unsigned long long *state) {
  *state += 0x9e3779b97f4a7c15ull;
  unsigned long long z = *state;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

static void run_size(unsigned long long bytes, int rate_khz) {
  const size_t slots = static_cast<size_t>(bytes / sizeof(unsigned long long));
  if (slots < 2) {
    PROBE_FAIL("--size needs at least two 8-byte slots");
  }

  // Sattolo's algorithm, not Fisher-Yates: it draws j from [0, i) rather than
  // [0, i], which is exactly the restriction that makes the result a single
  // cycle of length n. A chase over a permutation with more than one cycle
  // would only visit the cycle it started in, so the working set would be
  // smaller than requested and the latency would read too low.
  std::vector<unsigned long long> perm(slots);
  for (size_t i = 0; i < slots; ++i) {
    perm[i] = i;
  }
  unsigned long long state = 0x243f6a8885a308d3ull;
  for (size_t i = slots - 1; i > 0; --i) {
    const size_t j = static_cast<size_t>(next_random(&state) % i);
    const unsigned long long tmp = perm[i];
    perm[i] = perm[j];
    perm[j] = tmp;
  }

  unsigned long long *d_buf = nullptr;
  unsigned long long *d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, slots * sizeof(unsigned long long)));
  HIP_CHECK(hipMalloc(&d_out, 2 * sizeof(unsigned long long)));
  HIP_CHECK(hipMemcpy(d_buf, perm.data(), slots * sizeof(unsigned long long),
                      hipMemcpyHostToDevice));

  hipLaunchKernelGGL(chase_kernel, dim3(1), dim3(1), 0, nullptr, d_buf,
                     kChaseWarmup, d_out);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t start;
  hipEvent_t stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, nullptr));
  hipLaunchKernelGGL(chase_kernel, dim3(1), dim3(1), 0, nullptr, d_buf,
                     kChaseLoads, d_out);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipEventRecord(stop, nullptr));
  HIP_CHECK(hipEventSynchronize(stop));

  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  unsigned long long out[2] = {0, 0};
  HIP_CHECK(hipMemcpy(out, d_out, sizeof(out), hipMemcpyDeviceToHost));
  if (out[1] >= slots) {
    PROBE_FAIL("the chase left the buffer at index %llu", out[1]);
  }

  const double loads = static_cast<double>(kChaseLoads);
  std::printf("chase mode=size size_mib=%llu loads=%llu ns_per_load_wall=%.3f "
              "ns_per_load_memrealtime=%.3f rate_khz=%d\n",
              static_cast<unsigned long long>(
                  (slots * sizeof(unsigned long long)) >> 20),
              kChaseLoads, static_cast<double>(ms) * 1.0e6 / loads,
              probe_ticks_to_ns(out[0], rate_khz) / loads, rate_khz);

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(d_out));
  HIP_CHECK(hipFree(d_buf));
}

// --------------------------------------------------------------- --fence

// KIND 0 is release, 1 is acquire; SCOPE 0 is agent, 1 is workgroup. The
// workgroup-scope run is the baseline: it is the same loop with the same
// store and load, so the difference of the two is the cost of widening the
// fence to agent scope, which is what buys the cross-XCD visibility.
//
// One thread per participating workgroup, and one participating workgroup per
// XCD: the first block to take its XCD's slot runs the loop and every later
// block on the same XCD leaves. A one-block grid is the single-workgroup case
// and always wins its slot, so both cases share one code path.
template <int KIND, int SCOPE>
__global__ void fence_loop_kernel(volatile unsigned *store_slot,
                                  volatile unsigned *load_slot, unsigned *claim,
                                  unsigned long long *ticks_out, unsigned *sink,
                                  unsigned iters) {
  if (threadIdx.x != 0) {
    return;
  }
  const int xcd = probe_xcc_id();
  if (xcd < 0 || xcd >= kNumXcds) {
    return;
  }
  if (atomicCAS(&claim[xcd], kUnclaimed, blockIdx.x) != kUnclaimed) {
    return;
  }

  unsigned acc = 0;
  const unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
  for (unsigned k = 0; k < iters; ++k) {
    *store_slot = k;  // volatile: cannot be hoisted out of the loop
    if (SCOPE == 0) {
      if (KIND == 0) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
      } else {
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
      }
    } else {
      if (KIND == 0) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
      } else {
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
      }
    }
    acc += *load_slot;  // volatile: cannot be sunk past the fence
  }
  const unsigned long long t1 = __builtin_amdgcn_s_memrealtime();
  ticks_out[blockIdx.x] = t1 - t0;
  sink[blockIdx.x] = acc;
}

typedef void (*FenceLaunch)(int grid, volatile unsigned *store_slot,
                            volatile unsigned *load_slot, unsigned *claim,
                            unsigned long long *ticks_out, unsigned *sink);

template <int KIND, int SCOPE>
static void launch_fence(int grid, volatile unsigned *store_slot,
                         volatile unsigned *load_slot, unsigned *claim,
                         unsigned long long *ticks_out, unsigned *sink) {
  hipLaunchKernelGGL((fence_loop_kernel<KIND, SCOPE>), dim3(grid), dim3(64), 0,
                     nullptr, store_slot, load_slot, claim, ticks_out, sink,
                     kFenceIters);
  HIP_CHECK(hipGetLastError());
}

// Longest-running participant: with eight XCDs in the loop the slowest one is
// the cost the design would pay, not the average.
static double fence_pass(FenceLaunch launch, int grid, volatile unsigned *store_slot,
                         volatile unsigned *load_slot, unsigned *claim,
                         unsigned long long *ticks_out, unsigned *sink,
                         int rate_khz) {
  HIP_CHECK(hipMemset(claim, 0xff, kNumXcds * sizeof(unsigned)));
  HIP_CHECK(hipMemset(ticks_out, 0,
                      static_cast<size_t>(grid) * sizeof(unsigned long long)));
  launch(grid, store_slot, load_slot, claim, ticks_out, sink);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<unsigned long long> ticks(static_cast<size_t>(grid));
  HIP_CHECK(hipMemcpy(ticks.data(), ticks_out,
                      static_cast<size_t>(grid) * sizeof(unsigned long long),
                      hipMemcpyDeviceToHost));
  unsigned long long worst = 0;
  for (size_t i = 0; i < ticks.size(); ++i) {
    if (ticks[i] > worst) {
      worst = ticks[i];
    }
  }
  if (worst == 0) {
    PROBE_FAIL("no workgroup claimed an XCD slot");
  }
  return probe_ticks_to_ns(worst, rate_khz) / static_cast<double>(kFenceIters);
}

static void run_fence(const char *kind, int xcds, int rate_khz) {
  const bool release = std::strcmp(kind, "release") == 0;
  const int grid = (xcds == 1) ? 1 : kFenceGridXcds;

  unsigned *slots = nullptr;   // [0] stored to, [1] loaded from
  unsigned *claim = nullptr;
  unsigned *sink = nullptr;
  unsigned long long *ticks_out = nullptr;
  HIP_CHECK(hipMalloc(&slots, 2 * sizeof(unsigned)));
  HIP_CHECK(hipMalloc(&claim, kNumXcds * sizeof(unsigned)));
  HIP_CHECK(hipMalloc(&sink, static_cast<size_t>(grid) * sizeof(unsigned)));
  HIP_CHECK(hipMalloc(&ticks_out,
                      static_cast<size_t>(grid) * sizeof(unsigned long long)));
  HIP_CHECK(hipMemset(slots, 0, 2 * sizeof(unsigned)));

  volatile unsigned *store_slot = slots;
  volatile unsigned *load_slot = slots + 1;

  const double agent =
      fence_pass(release ? &launch_fence<0, 0> : &launch_fence<1, 0>, grid,
                 store_slot, load_slot, claim, ticks_out, sink, rate_khz);
  const double workgroup =
      fence_pass(release ? &launch_fence<0, 1> : &launch_fence<1, 1>, grid,
                 store_slot, load_slot, claim, ticks_out, sink, rate_khz);

  std::printf("chase mode=fence kind=%s xcds=%d ns_per_iter_agent=%.3f "
              "ns_per_iter_workgroup=%.3f ns_fence_cost=%.3f\n",
              release ? "release" : "acquire", xcds, agent, workgroup,
              agent - workgroup);

  HIP_CHECK(hipFree(ticks_out));
  HIP_CHECK(hipFree(sink));
  HIP_CHECK(hipFree(claim));
  HIP_CHECK(hipFree(slots));
}

// ------------------------------------------------------------ --pingpong

// Two workgroups on two different XCDs alternate ownership of one 32-bit
// flag. The owner publishes the next value with an agent-scope release store
// and the other spins on agent-scope acquire loads with no sleep, so each
// handover is one release flush plus one cross-XCD read, which is the
// boundary cost of docs/design-doc/09-expected-performance.md.
//
// The two players cannot be the same workgroup: each claims its own XCD slot,
// and only XCD 0 and XCD 1 play. The bounded spin is the guard for the case
// where the launch put no block on one of those two XCDs at all; without it
// the survivor would spin until the watchdog killed the queue.
__global__ void pingpong_kernel(unsigned *flag, unsigned *claim, int *xcd_out,
                                unsigned long long *ticks_out, unsigned *status,
                                unsigned rounds,
                                unsigned long long timeout_ticks) {
  if (threadIdx.x != 0) {
    return;
  }
  const int xcd = probe_xcc_id();
  if (xcd != 0 && xcd != 1) {
    return;
  }
  if (atomicCAS(&claim[xcd], kUnclaimed, blockIdx.x) != kUnclaimed) {
    return;
  }
  const bool is_a = (xcd == 0);
  xcd_out[is_a ? 0 : 1] = xcd;

  bool timed_out = false;
  const unsigned long long t0 = __builtin_amdgcn_s_memrealtime();
  for (unsigned r = 0; r < rounds; ++r) {
    const unsigned mine = 2u * r + (is_a ? 1u : 2u);
    const unsigned theirs = 2u * r + (is_a ? 2u : 1u);
    if (is_a) {
      __hip_atomic_store(flag, mine, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
    unsigned spins = 0;
    const unsigned long long w0 = __builtin_amdgcn_s_memrealtime();
    while (__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) !=
           theirs) {
      // The clock is read once every 1024 polls so the guard does not become
      // part of what is being measured.
      if ((++spins & 1023u) == 0u &&
          __builtin_amdgcn_s_memrealtime() - w0 > timeout_ticks) {
        timed_out = true;
        break;
      }
    }
    if (timed_out) {
      break;
    }
    if (!is_a) {
      __hip_atomic_store(flag, mine, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
  const unsigned long long t1 = __builtin_amdgcn_s_memrealtime();
  if (is_a) {
    ticks_out[0] = t1 - t0;
  }
  if (timed_out) {
    atomicExch(status, 1u);
  }
}

static void run_pingpong(bool fine, int rate_khz) {
  unsigned *flag = nullptr;
  if (fine) {
    HIP_CHECK(hipExtMallocWithFlags(reinterpret_cast<void **>(&flag),
                                    sizeof(unsigned),
                                    hipDeviceMallocFinegrained));
  } else {
    // Plain hipMalloc: coarse-grained device memory, which is what the
    // runtime's own counters live in.
    HIP_CHECK(hipMalloc(&flag, sizeof(unsigned)));
  }

  unsigned *claim = nullptr;
  unsigned *status = nullptr;
  int *xcd_out = nullptr;
  unsigned long long *ticks_out = nullptr;
  HIP_CHECK(hipMalloc(&claim, kNumXcds * sizeof(unsigned)));
  HIP_CHECK(hipMalloc(&status, sizeof(unsigned)));
  HIP_CHECK(hipMalloc(&xcd_out, 2 * sizeof(int)));
  HIP_CHECK(hipMalloc(&ticks_out, sizeof(unsigned long long)));
  HIP_CHECK(hipMemset(flag, 0, sizeof(unsigned)));
  HIP_CHECK(hipMemset(claim, 0xff, kNumXcds * sizeof(unsigned)));
  HIP_CHECK(hipMemset(status, 0, sizeof(unsigned)));
  HIP_CHECK(hipMemset(xcd_out, 0xff, 2 * sizeof(int)));
  HIP_CHECK(hipMemset(ticks_out, 0, sizeof(unsigned long long)));

  const unsigned long long timeout_ticks =
      static_cast<unsigned long long>(rate_khz) * 2000ull;  // about 2 s

  hipLaunchKernelGGL(pingpong_kernel, dim3(kPingpongGrid), dim3(64), 0, nullptr,
                     flag, claim, xcd_out, ticks_out, status, kPingpongRounds,
                     timeout_ticks);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  int xcds[2] = {-1, -1};
  unsigned long long ticks = 0;
  unsigned timed_out = 0;
  unsigned final_flag = 0;
  HIP_CHECK(hipMemcpy(xcds, xcd_out, sizeof(xcds), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&ticks, ticks_out, sizeof(ticks), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&timed_out, status, sizeof(timed_out),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&final_flag, flag, sizeof(final_flag),
                      hipMemcpyDeviceToHost));

  if (timed_out != 0 || final_flag != 2u * kPingpongRounds) {
    std::fprintf(stderr,
                 "%s warning: pingpong did not complete (xcd_a=%d xcd_b=%d "
                 "flag=%u of %u); the line below is not a round-trip time\n",
                 PROBE_NAME, xcds[0], xcds[1], final_flag,
                 2u * kPingpongRounds);
  }

  const double total_ns = probe_ticks_to_ns(ticks, rate_khz);
  std::printf("chase mode=pingpong xcd_a=%d xcd_b=%d rounds=%u "
              "ns_one_way=%.3f fine=%d\n",
              xcds[0], xcds[1], kPingpongRounds,
              total_ns / (2.0 * static_cast<double>(kPingpongRounds)),
              fine ? 1 : 0);

  HIP_CHECK(hipFree(ticks_out));
  HIP_CHECK(hipFree(xcd_out));
  HIP_CHECK(hipFree(status));
  HIP_CHECK(hipFree(claim));
  HIP_CHECK(hipFree(flag));
}

// ------------------------------------------------------------------ main

int main(int argc, char **argv) {
  bool do_size = false;
  bool do_fence = false;
  bool do_pingpong = false;
  bool fine = false;
  unsigned long long bytes = 0;
  const char *kind = "release";
  long xcds = 1;

  for (int i = 1; i < argc; ++i) {
    long v = 0;
    if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
      if (!probe_parse_bytes(argv[++i], &bytes) || bytes < 16) {
        PROBE_USAGE("  --size takes N, NM (MiB) or NG (GiB), got %s\n", argv[i]);
      }
      do_size = true;
    } else if (std::strcmp(argv[i], "--fence") == 0 && i + 1 < argc) {
      kind = argv[++i];
      if (std::strcmp(kind, "release") != 0 && std::strcmp(kind, "acquire") != 0) {
        PROBE_USAGE("  --fence takes release or acquire, got %s\n", kind);
      }
      do_fence = true;
    } else if (std::strcmp(argv[i], "--xcds") == 0 && i + 1 < argc) {
      if (!probe_parse_long(argv[++i], &v) || (v != 1 && v != 8)) {
        PROBE_USAGE("  --xcds takes 1 or 8, got %s\n", argv[i]);
      }
      xcds = v;
    } else if (std::strcmp(argv[i], "--pingpong") == 0) {
      do_pingpong = true;
    } else if (std::strcmp(argv[i], "--fine") == 0) {
      fine = true;
    } else {
      PROBE_USAGE("  chase --size N[M|G]\n"
                  "  chase --fence release|acquire [--xcds 1|8]\n"
                  "  chase --pingpong [--fine]\n"
                  "  chase                      all three, sizes 1M 64M 1G,\n"
                  "                             both fences at 1 and 8 XCDs\n");
    }
  }

  hipDeviceProp_t prop;
  probe_init(&prop);
  const int rate_khz = probe_wall_clock_khz();

  const bool run_all = !do_size && !do_fence && !do_pingpong;
  if (run_all) {
    run_size(1ull << 20, rate_khz);
    run_size(64ull << 20, rate_khz);
    run_size(1ull << 30, rate_khz);
    run_fence("release", 1, rate_khz);
    run_fence("acquire", 1, rate_khz);
    run_fence("release", 8, rate_khz);
    run_fence("acquire", 8, rate_khz);
    run_pingpong(fine, rate_khz);
    return 0;
  }
  if (do_size) {
    run_size(bytes, rate_khz);
  }
  if (do_fence) {
    run_fence(kind, static_cast<int>(xcds), rate_khz);
  }
  if (do_pingpong) {
    run_pingpong(fine, rate_khz);
  }
  return 0;
}
