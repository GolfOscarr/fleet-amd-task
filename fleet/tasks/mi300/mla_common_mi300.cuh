/* Helpers shared by the DeepSeek-V2-Lite tasks (mla_prep, mla_attend,
 * mla_merge_uv, moe_router, copy). Worker contract of the shipped MI300
 * kernels: 256 threads = 4 wavefronts of 64, dynamic LDS `smem`, FP32
 * accumulation, BF16 storage (docs/fleet/04-repo-map.md).
 *
 * The rounding points mirror harness/numpy_ref.py: `bf16r` is numpy_ref.bf16
 * (round to nearest even to BF16 precision, kept as float).
 */
#pragma once
#include "tasks/common/common_header.cuh"
#include <math.h>

namespace kernel {
namespace dsv2 {

constexpr int WAVE = 64;
constexpr int WAVES = NUM_THREADS / WAVE;   // 4

// Round a float to BF16 precision and back (round to nearest even), the
// same bits as numpy_ref.bf16 and as torch's float -> bfloat16 conversion.
__device__ __forceinline__ float bf16r(float x) {
  unsigned int u = __float_as_uint(x);
  unsigned int lsb = (u >> 16) & 1u;
  u = (u + 0x7FFFu + lsb) & 0xFFFF0000u;
  return __uint_as_float(u);
}

template <typename T>
__device__ __forceinline__ float ld(T const *p) {
  return static_cast<float>(*p);
}

template <typename T>
__device__ __forceinline__ void st(T *p, float x) {
  *p = static_cast<T>(x);
}

// 16-byte load of 8 BF16 values as floats (dwordx4 per lane).
template <typename T>
__device__ __forceinline__ void load8(T const *src, float out[8]) {
  uint4 raw = *reinterpret_cast<uint4 const *>(src);
  T const *v = reinterpret_cast<T const *>(&raw);
#pragma unroll
  for (int k = 0; k < 8; k++) {
    out[k] = static_cast<float>(v[k]);
  }
}

__device__ __forceinline__ float wave_sum(float x) {
#pragma unroll
  for (int off = WAVE / 2; off > 0; off >>= 1) {
    x += __shfl_xor(x, off, WAVE);
  }
  return x;
}

__device__ __forceinline__ float wave_max(float x) {
#pragma unroll
  for (int off = WAVE / 2; off > 0; off >>= 1) {
    x = fmaxf(x, __shfl_xor(x, off, WAVE));
  }
  return x;
}

// Block-wide sum over 256 threads; `red` is 4 floats of LDS. Every thread
// receives the total.
__device__ __forceinline__ float block_sum(float x, float *red) {
  x = wave_sum(x);
  int w = threadIdx.x / WAVE;
  int l = threadIdx.x % WAVE;
  if (l == 0) {
    red[w] = x;
  }
  __syncthreads();
  float total = 0.0f;
#pragma unroll
  for (int i = 0; i < WAVES; i++) {
    total += red[i];
  }
  __syncthreads();
  return total;
}

// DeepseekV2RMSNorm on BF16 input, as numpy_ref.rmsnorm: FP32 statistics,
// the normalized value rounded to BF16, then the BF16 weight multiply.
// Each thread handles the elements i = tid, tid + 256, ...; `red` is 4
// floats of LDS. `out_s`, when given, receives the same row in LDS so the
// caller can read it back without a round trip through global memory
// (the router's norm prologue, docs/gpu-experiments/03-acceleration O1).
template <typename T, int N>
__device__ __forceinline__ void rmsnorm_row(T const *x, T const *w, T *out, float eps,
                                            float *red, T *out_s = nullptr) {
  float ss = 0.0f;
  for (int i = threadIdx.x; i < N; i += NUM_THREADS) {
    float v = ld(x + i);
    ss += v * v;
  }
  float var = block_sum(ss, red) / (float)N;
  float rinv = 1.0f / sqrtf(var + eps);
  for (int i = threadIdx.x; i < N; i += NUM_THREADS) {
    float xn = bf16r(ld(x + i) * rinv);
    float y = bf16r(ld(w + i) * xn);
    st(out + i, y);
    if (out_s != nullptr) {
      st(out_s + i, y);
    }
  }
}

// One element of apply_rotary_pos_emb on a 64-wide slice, as numpy_ref.rope:
// the (even, odd) pairs are de-interleaved into two halves, then
// x * cos + rotate_half(x) * sin with BF16 rounding after every operation.
// `raw` is the un-rotated slice, `i` the output index in [0, D_R).
template <typename T, int D_R>
__device__ __forceinline__ float rope_elem(T const *raw, int i, float c, float s) {
  constexpr int HALF = D_R / 2;
  float x2, rot;
  if (i < HALF) {
    x2 = ld(raw + 2 * i);                 // even entries form the first half
    rot = -ld(raw + 2 * i + 1);           // -(second half)[i] = -odd entry i
  } else {
    x2 = ld(raw + 2 * (i - HALF) + 1);    // odd entries form the second half
    rot = ld(raw + 2 * (i - HALF));       // (first half)[i - HALF]
  }
  return bf16r(bf16r(x2 * c) + bf16r(rot * s));
}

} // namespace dsv2
} // namespace kernel
