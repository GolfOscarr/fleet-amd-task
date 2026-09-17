/* Stub of <hip/hip_runtime.h> for the host-side syntax check of
 * fleet/tasks/kernel_tests_mi300.cu (fleet/tasks/check_syntax.sh). Only the
 * API the launcher uses, with no-op bodies; nothing runs. The device-side
 * vocabulary (dim3, threadIdx, __global__, ...) is the stub common header's.
 */
#pragma once
#include "tasks/common/common_header.cuh"
#include <cstddef>

typedef int hipError_t;
constexpr hipError_t hipSuccess = 0;
typedef void *hipStream_t;
enum hipMemcpyKind { hipMemcpyHostToDevice, hipMemcpyDeviceToHost, hipMemcpyDeviceToDevice };
enum hipFuncAttribute { hipFuncAttributeMaxDynamicSharedMemorySize };

inline hipError_t hipMalloc(void **p, size_t) {
  *p = nullptr;
  return hipSuccess;
}
inline hipError_t hipFree(void *) { return hipSuccess; }
inline hipError_t hipMemcpy(void *, void const *, size_t, hipMemcpyKind) { return hipSuccess; }
inline hipError_t hipDeviceSynchronize() { return hipSuccess; }
inline hipError_t hipGetLastError() { return hipSuccess; }
// events, used by the suite's timing loops (KT_TIME, KT_COLD; session B of round 2)
typedef void *hipEvent_t;
inline hipError_t hipEventCreate(hipEvent_t *e) { *e = nullptr; return hipSuccess; }
inline hipError_t hipEventRecord(hipEvent_t, int = 0) { return hipSuccess; }
inline hipError_t hipEventSynchronize(hipEvent_t) { return hipSuccess; }
inline hipError_t hipEventElapsedTime(float *ms, hipEvent_t, hipEvent_t) { *ms = 0.0f; return hipSuccess; }
inline hipError_t hipEventDestroy(hipEvent_t) { return hipSuccess; }
inline char const *hipGetErrorString(hipError_t) { return "stub"; }
inline hipError_t hipFuncSetAttribute(void const *, hipFuncAttribute, int) { return hipSuccess; }

// Calling the kernel as a plain function type-checks the argument list
// against its signature, which is the point of parsing the launcher here.
template <typename F, typename... A>
void hipLaunchKernelGGL(F f, dim3, dim3, size_t, hipStream_t, A... a) {
  f(a...);
}
