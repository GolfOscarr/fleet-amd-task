/* Stub of rocBLAS for the host syntax check of the patched runtime sources
 * (env/offline_gfx942/run.sh, the host step): the ROCm image carries no
 * rocblas-dev, and the fork's headers (mirage/utils/rocblas_helper.h,
 * rocm_helper.h) only need these declarations to parse. Nothing here runs.
 */
#pragma once
#include <hip/hip_runtime_api.h>

typedef struct _rocblas_handle *rocblas_handle;
typedef enum rocblas_status_ { rocblas_status_success = 0 } rocblas_status;
typedef enum rocblas_operation_ {
  rocblas_operation_none = 111,
  rocblas_operation_transpose = 112,
  rocblas_operation_conjugate_transpose = 113
} rocblas_operation;
typedef enum rocblas_datatype_ {
  rocblas_datatype_f16_r = 150,
  rocblas_datatype_f32_r = 151,
  rocblas_datatype_f64_r = 152,
  rocblas_datatype_i8_r = 160,
  rocblas_datatype_i32_r = 162,
  rocblas_datatype_bf16_r = 168
} rocblas_datatype;
typedef enum rocblas_gemm_algo_ { rocblas_gemm_algo_standard = 0 } rocblas_gemm_algo;

extern "C" {
rocblas_status rocblas_create_handle(rocblas_handle *handle);
rocblas_status rocblas_destroy_handle(rocblas_handle handle);
rocblas_status rocblas_set_stream(rocblas_handle handle, hipStream_t stream);
rocblas_status rocblas_gemm_ex(rocblas_handle handle, rocblas_operation transA, rocblas_operation transB,
                               int m, int n, int k, const void *alpha,
                               const void *a, rocblas_datatype a_type, int lda,
                               const void *b, rocblas_datatype b_type, int ldb,
                               const void *beta, const void *c, rocblas_datatype c_type, int ldc,
                               void *d, rocblas_datatype d_type, int ldd,
                               rocblas_datatype compute_type, rocblas_gemm_algo algo,
                               int solution_index, unsigned flags);
}
