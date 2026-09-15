// Shared scaffolding for the hardware probes in this directory: the HIP error
// check, the bad-argument exit, the small argument parsers, the wall-clock
// rate lookup and the XCC_ID read. Every probe defines PROBE_NAME (a string
// literal) before including this header.
//
// Conventions every probe keeps, because env/hw/summarize.py parses them:
//   - results go to stdout as one line of space-separated key=value pairs,
//     the first token being the probe name;
//   - a failed HIP call prints "<probe> error: <what> at <file>:<line>" to
//     stderr and exits 2;
//   - a bad command line prints "<probe> usage" to stderr and exits 1;
//   - the device is always index 0 as the runtime sees it, so the caller
//     selects the GPU with HIP_VISIBLE_DEVICES.
#ifndef METALOPS_ENV_HW_PROBES_PROBE_COMMON_H
#define METALOPS_ENV_HW_PROBES_PROBE_COMMON_H

#ifndef PROBE_NAME
#error "define PROBE_NAME (a string literal) before including probe_common.h"
#endif

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t probe_err_ = (expr);                                            \
    if (probe_err_ != hipSuccess) {                                            \
      std::fprintf(stderr, "%s error: %s at %s:%d\n", PROBE_NAME,              \
                   hipGetErrorString(probe_err_), __FILE__, __LINE__);         \
      std::exit(2);                                                            \
    }                                                                          \
  } while (0)

// A hard failure that is not a HIP call (an unopenable dump file, a probe that
// could not place its workgroups). Same exit code as a HIP error: the caller
// only needs to know the line was not produced.
#define PROBE_FAIL(...)                                                        \
  do {                                                                         \
    std::fprintf(stderr, "%s error: ", PROBE_NAME);                            \
    std::fprintf(stderr, __VA_ARGS__);                                         \
    std::fprintf(stderr, " at %s:%d\n", __FILE__, __LINE__);                   \
    std::exit(2);                                                              \
  } while (0)

#define PROBE_USAGE(...)                                                       \
  do {                                                                         \
    std::fprintf(stderr, "%s usage\n", PROBE_NAME);                            \
    std::fprintf(stderr, __VA_ARGS__);                                         \
    std::exit(1);                                                              \
  } while (0)

// Device 0 as HIP sees it after HIP_VISIBLE_DEVICES has been applied. Fails
// through HIP_CHECK with hipErrorNoDevice when the VM exposes no GPU.
static inline void probe_init(hipDeviceProp_t *prop) {
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipGetDeviceProperties(prop, 0));
}

// "1G" is 1 GiB, "64M" is 64 MiB, "4K" is 4 KiB, a bare number is bytes.
static inline bool probe_parse_bytes(const char *s, unsigned long long *out) {
  char *end = nullptr;
  unsigned long long v = std::strtoull(s, &end, 10);
  if (end == s) {
    return false;
  }
  unsigned long long mul = 1;
  if (*end == 'K' || *end == 'k') {
    mul = 1ull << 10;
    ++end;
  } else if (*end == 'M' || *end == 'm') {
    mul = 1ull << 20;
    ++end;
  } else if (*end == 'G' || *end == 'g') {
    mul = 1ull << 30;
    ++end;
  }
  if (*end == 'B' || *end == 'b') {
    ++end;
  }
  if (*end != '\0' || v == 0) {
    return false;
  }
  *out = v * mul;
  return true;
}

static inline bool probe_parse_long(const char *s, long *out) {
  char *end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (end == s || *end != '\0') {
    return false;
  }
  *out = v;
  return true;
}

// kHz of the constant-rate counter s_memrealtime reads. Every in-kernel time
// number in this project is ticks divided by this.
static inline int probe_wall_clock_khz() {
  int khz = 0;
  HIP_CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, 0));
  if (khz <= 0) {
    PROBE_FAIL("hipDeviceAttributeWallClockRate returned %d", khz);
  }
  return khz;
}

static inline double probe_ticks_to_ns(unsigned long long ticks, int khz) {
  // ticks / (khz * 1e3) seconds, in nanoseconds.
  return static_cast<double>(ticks) * 1.0e6 / static_cast<double>(khz);
}

// The XCD (XCC) the workgroup is running on. Same spelling the Fleet runtime
// uses (repos/fleet-chiplet-megakernel/include/mirage/persistent_kernel/
// persistent_kernel.cuh:188).
__device__ __forceinline__ int probe_xcc_id() {
  int xcd_id;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
}

#endif  // METALOPS_ENV_HW_PROBES_PROBE_COMMON_H
