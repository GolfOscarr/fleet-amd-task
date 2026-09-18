// Two loop orders for "the first batch before the prologue": MODE 1 (the wave-1 kernels):
// pre-load; loop { if (not first) load; multiply }. MODE 2: pre-load; loop { multiply;
// if (next) load next }. Same math; the register count is the question.
#include <hip/hip_runtime.h>
#include <stdint.h>
#ifndef BATCH
#define BATCH 8
#endif
#ifndef ADDR
#define ADDR 0
#endif
#ifndef MODE
#define MODE 1
#endif
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
constexpr int WAVE = 64, NT = 256, K = 2048, ROWS = 64, CHUNKS = 4, RPW = ROWS / 4;
__device__ __forceinline__ float wave_sum(float x) {
#pragma unroll
  for (int off = 32; off > 0; off >>= 1) x += __shfl_xor(x, off, WAVE);
  return x;
}
__device__ __forceinline__ void load_batch(const uint16_t *w, int r0, int lane, u32x4 (&raw)[BATCH][CHUNKS]) {
#pragma unroll
  for (int u = 0; u < BATCH; u++) {
#if ADDR == 1
    const uint16_t *rowp = w + (size_t)(r0 + u) * K + 8 * lane;   // the row pointer once, the chunk a constant
#pragma unroll
    for (int i = 0; i < CHUNKS; i++)
      raw[u][i] = *reinterpret_cast<const u32x4 *>(rowp + 512 * i);
#elif ADDR == 2
    size_t off = (size_t)(r0 + u) * K + (size_t)(8 * lane);         // an element offset, the chunk added as size_t
#pragma unroll
    for (int i = 0; i < CHUNKS; i++)
      raw[u][i] = *reinterpret_cast<const u32x4 *>(w + (off + (size_t)(512 * i)));
#else
#pragma unroll
    for (int i = 0; i < CHUNKS; i++)
      raw[u][i] = *reinterpret_cast<const u32x4 *>(w + (size_t)(r0 + u) * K + 8 * lane + 512 * i);
#endif
  }
}
__global__ __launch_bounds__(NT, 1) void k_order(const uint16_t *x, const uint16_t *w, uint16_t *out, float *red) {
  int tid = threadIdx.x, wave = tid / WAVE, lane = tid % WAVE;
  const uint16_t *wb = w + (size_t)blockIdx.x * ROWS * K;
  int r_begin = wave * RPW, r_end = r_begin + RPW;
  u32x4 raw[BATCH][CHUNKS];
#if MODE != 3
  load_batch(wb, r_begin, lane, raw);
#endif
  // a stand-in prologue: a block reduction through LDS and a barrier
  __shared__ float s[4];
  float v = __uint_as_float((unsigned)x[tid] << 16);
  v = wave_sum(v);
  if (lane == 0) s[wave] = v;
  __syncthreads();
  float scale = s[0] + s[1] + s[2] + s[3];
  float xv[CHUNKS][8];
#pragma unroll
  for (int i = 0; i < CHUNKS; i++)
#pragma unroll
    for (int k = 0; k < 8; k++) xv[i][k] = scale * __uint_as_float((unsigned)x[8 * lane + 512 * i + k] << 16);
#if MODE == 4
  {
    float sums[BATCH];
#pragma unroll
    for (int u = 0; u < BATCH; u++) {
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < CHUNKS; i++) {
        const unsigned *wd = reinterpret_cast<const unsigned *>(&raw[u][i]);
#pragma unroll
        for (int k = 0; k < 8; k++)
          acc += xv[i][k] * __uint_as_float((k & 1) ? (wd[k / 2] & 0xffff0000u) : (wd[k / 2] << 16));
      }
      sums[u] = wave_sum(acc);
    }
    if (lane < BATCH) { unsigned b = __float_as_uint(sums[0] + sums[lane & 1]); out[blockIdx.x * ROWS + r_begin + lane] = (uint16_t)(b >> 16); }
  }
#pragma unroll 1
  for (int r0 = r_begin + BATCH; r0 < r_end; r0 += BATCH) {
    load_batch(wb, r0, lane, raw);
#else
#pragma unroll 1
  for (int r0 = r_begin; r0 < r_end; r0 += BATCH) {
#if MODE == 1
    if (r0 != r_begin) load_batch(wb, r0, lane, raw);
#elif MODE == 3
    load_batch(wb, r0, lane, raw);
#endif
#endif
    float sums[BATCH];
#pragma unroll
    for (int u = 0; u < BATCH; u++) {
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < CHUNKS; i++) {
        const unsigned *wd = reinterpret_cast<const unsigned *>(&raw[u][i]);
#pragma unroll
        for (int k = 0; k < 8; k++)
          acc += xv[i][k] * __uint_as_float((k & 1) ? (wd[k / 2] & 0xffff0000u) : (wd[k / 2] << 16));
      }
      sums[u] = wave_sum(acc);
    }
#if MODE == 2
    if (r0 + BATCH < r_end) load_batch(wb, r0 + BATCH, lane, raw);
#endif
    if (lane < BATCH) { unsigned b = __float_as_uint(sums[0] + sums[lane & 1]); out[blockIdx.x * ROWS + r0 + lane] = (uint16_t)(b >> 16); }
  }
}
