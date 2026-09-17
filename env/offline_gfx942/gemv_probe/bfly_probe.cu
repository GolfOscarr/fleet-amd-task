// A device probe of butterfly_sum: every lane holds v[r] = (lane + 1) * (r + 1); the total of row r
// over 64 lanes is (r + 1) * 2080. Each lane l < R stores row l's total.
#include <hip/hip_runtime.h>
#define NUM_THREADS 256
namespace kernel { namespace dsv2 {
constexpr int WAVE = 64;
__device__ __forceinline__ float wave_sum(float x) {
#pragma unroll
  for (int off = WAVE / 2; off > 0; off >>= 1) x += __shfl_xor(x, off, WAVE);
  return x;
}
template <int R>
__device__ __forceinline__ float butterfly_sum(float (&v)[R]) {
  static_assert(R >= 1 && R <= WAVE && (R & (R - 1)) == 0, "R rows, a power of two up to the wave");
  int lane = threadIdx.x % WAVE;
  int live = R;
#pragma unroll
  for (int w = R / 2; w >= 1; w >>= 1) {
    int half = live / 2;
    bool upper = (lane & w) != 0;
#pragma unroll
    for (int i = 0; i < R / 2; i++) {
      if (i < half) {
        float keep = upper ? v[half + i] : v[i];
        float send = upper ? v[i] : v[half + i];
        v[i] = keep + __shfl_xor(send, w, WAVE);
      }
    }
    live = half;
  }
#pragma unroll
  for (int w = R; w < WAVE; w <<= 1) {
    v[0] += __shfl_xor(v[0], w, WAVE);
  }
  return v[0];
}

} }
template <int R> __global__ void k(float *out) {
  using namespace kernel::dsv2;
  float v[R];
  int lane = threadIdx.x % 64;
#pragma unroll
  for (int r = 0; r < R; r++) v[r] = (float)(lane + 1) * (float)(r + 1);
  float t = butterfly_sum<R>(v);
  if (threadIdx.x < R) out[threadIdx.x] = t;
}
template __global__ void k<8>(float *); template __global__ void k<16>(float *); template __global__ void k<32>(float *); template __global__ void k<1>(float *); template __global__ void k<64>(float *);
