// Offline probe for docs/gpu-experiments/04-kernels/01-gemv-ideas.md (K1, K2, K6):
// does hipcc 7.0 keep a batch of B raw 16-byte loads in flight for gfx942, what does
// the tile cost in registers, and does v_dot2_f32_bf16 exist on gfx942.
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifndef ROWS
#define ROWS 64
#endif
#ifndef K
#define K 2048
#endif
#ifndef BATCH
#define BATCH 16
#endif
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
constexpr int WAVE = 64;
constexpr int NT = 256;
constexpr int PER_LANE = K / WAVE;       // 32 elements per lane
constexpr int LOADS = PER_LANE / 8;      // 4 x 16 bytes per row per lane
constexpr int ROWS_PER_WAVE = ROWS / 4;
static_assert(ROWS_PER_WAVE % BATCH == 0, "the wave's rows split into whole batches");

__device__ __forceinline__ float wave_sum(float x) {
#pragma unroll
  for (int off = WAVE / 2; off > 0; off >>= 1) x += __shfl_xor(x, off, WAVE);
  return x;
}

// K1: FMA on raw words, B rows in flight per lane, router-style batches
__global__ __launch_bounds__(NT, 1) void k_gemv_fma(const uint16_t *__restrict__ x,
                                                    const uint16_t *__restrict__ w,
                                                    uint16_t *__restrict__ out) {
  int tid = threadIdx.x, wave = tid / WAVE, lane = tid % WAVE;
  float xv[PER_LANE];
#pragma unroll
  for (int i = 0; i < LOADS; i++) {
    uint4 r = *reinterpret_cast<const uint4 *>(x + lane * PER_LANE + 8 * i);
    const unsigned *wd = reinterpret_cast<const unsigned *>(&r);
#pragma unroll
    for (int k = 0; k < 8; k++)
      xv[8 * i + k] = __uint_as_float((k & 1) ? (wd[k / 2] & 0xffff0000u) : (wd[k / 2] << 16));
  }
  const uint16_t *wbase = w + (size_t)blockIdx.x * ROWS * K;
#ifdef NOHOIST
#pragma unroll 1
#endif
  for (int r0 = wave * ROWS_PER_WAVE; r0 < (wave + 1) * ROWS_PER_WAVE; r0 += BATCH) {
    u32x4 raw[BATCH][LOADS];
#pragma unroll
    for (int u = 0; u < BATCH; u++) {
      const uint16_t *row = wbase + (size_t)(r0 + u) * K + lane * PER_LANE;
#pragma unroll
      for (int i = 0; i < LOADS; i++) raw[u][i] = __builtin_nontemporal_load(reinterpret_cast<const u32x4 *>(row + 8 * i));
    }
#pragma unroll
    for (int u = 0; u < BATCH; u++) {
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < LOADS; i++) {
        const unsigned *wd = reinterpret_cast<const unsigned *>(&raw[u][i]);
#pragma unroll
        for (int k = 0; k < 8; k++)
          acc += xv[8 * i + k] * __uint_as_float((k & 1) ? (wd[k / 2] & 0xffff0000u) : (wd[k / 2] << 16));
      }
      acc = wave_sum(acc);
      if (lane == 0) {
        unsigned b = __float_as_uint(acc);
        b += 0x7fffu + ((b >> 16) & 1u);
        out[blockIdx.x * ROWS + r0 + u] = (uint16_t)(b >> 16);
      }
    }
  }
}

#ifdef TRY_DOT2
typedef short bf16x2_t __attribute__((ext_vector_type(2)));
__global__ void k_dot2(const bf16x2_t *a, const bf16x2_t *b, float *o) {
  o[threadIdx.x] = __builtin_amdgcn_fdot2_f32_bf16(a[threadIdx.x], b[threadIdx.x], 0.0f, false);
}
#endif
#ifdef TRY_DOT2_ASM
__global__ void k_dot2_asm(const unsigned *a, const unsigned *b, float *o) {
  float r;
  asm volatile("v_dot2_f32_bf16 %0, %1, %2, 0" : "=v"(r) : "v"(a[threadIdx.x]), "v"(b[threadIdx.x]));
  o[threadIdx.x] = r;
}
#endif
