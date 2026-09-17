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

// Streaming loads (docs/gpu-experiments/03-acceleration, O6; -DMLA_NT_STREAMS): the
// policy the stock linears give their weight loads under -DMPK_NT_WEIGHT_LOADS,
// CK's amd_buffer_coherence_enum value 18 on gfx942 (bit 4 = sc1, bit 1 = nt:
// device-scope, non-temporal), issued as raw buffer loads so the compiler keeps
// tracking the loads' completion. A StreamSrc holds the tensor's base and, on the
// device with the builtins, its buffer resource; load8_from / ldf_from read at an
// element offset. Without MLA_NT_STREAMS (or on the host syntax check, which has
// no builtins) they are the plain loads, so the math is the same either way.
#if defined(MLA_NT_STREAMS) && defined(__HIP_DEVICE_COMPILE__) && \
    __has_builtin(__builtin_amdgcn_raw_buffer_load_b128) && __has_builtin(__builtin_amdgcn_make_buffer_rsrc)
#define MLA_STREAM_LOADS 1
constexpr int STREAM_AUX = 18;                       // sc1 nt
constexpr int BUFFER_RSRC_WORD3 = 0x00020000;        // CK_TILE_BUFFER_RESOURCE_3RD_DWORD for gfx9
template <typename T>
struct StreamSrc {
  T const *base;
  __amdgpu_buffer_rsrc_t rsrc;
  __device__ __forceinline__ explicit StreamSrc(T const *p)
      : base(p), rsrc(__builtin_amdgcn_make_buffer_rsrc(const_cast<T *>(p), 0, 0xFFFFFFFFu, BUFFER_RSRC_WORD3)) {}
};
template <typename T>
__device__ __forceinline__ void load8_from(StreamSrc<T> const &s, size_t elem, float out[8]) {
  static_assert(sizeof(T) == 2, "8 x 16-bit values per 16-byte load");
  typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
  u32x4 raw = __builtin_amdgcn_raw_buffer_load_b128(s.rsrc, (unsigned)(elem * sizeof(T)), 0, STREAM_AUX);
  T const *v = reinterpret_cast<T const *>(&raw);
#pragma unroll
  for (int k = 0; k < 8; k++) {
    out[k] = static_cast<float>(v[k]);
  }
}
__device__ __forceinline__ float ldf_from(StreamSrc<float> const &s, size_t elem) {
  unsigned int raw = __builtin_amdgcn_raw_buffer_load_b32(s.rsrc, (unsigned)(elem * sizeof(float)), 0, STREAM_AUX);
  return __uint_as_float(raw);
}
#else
template <typename T>
struct StreamSrc {
  T const *base;
  __device__ __forceinline__ explicit StreamSrc(T const *p) : base(p) {}
};
template <typename T>
__device__ __forceinline__ void load8_from(StreamSrc<T> const &s, size_t elem, float out[8]);
__device__ __forceinline__ float ldf_from(StreamSrc<float> const &s, size_t elem) {
  return s.base[elem];
}
#endif

// Raw 16-byte load (8 BF16 values, unconverted) at an element offset: the MFMA attention's
// cooperative tile load (O7). Streams under MLA_NT_STREAMS as load8_from does.
template <typename T>
__device__ __forceinline__ uint4 load16_from(StreamSrc<T> const &s, size_t elem) {
  static_assert(sizeof(T) == 2, "8 x 16-bit values per 16-byte load");
#ifdef MLA_STREAM_LOADS
  typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
  u32x4 raw = __builtin_amdgcn_raw_buffer_load_b128(s.rsrc, (unsigned)(elem * sizeof(T)), 0, STREAM_AUX);
  uint4 r;
  r.x = raw[0]; r.y = raw[1]; r.z = raw[2]; r.w = raw[3];
  return r;
#else
  return *reinterpret_cast<uint4 const *>(s.base + elem);
#endif
}

// The matrix core instruction of the MFMA attention (O7, docs/gpu-experiments/03-acceleration):
// v_mfma_f32_16x16x16_bf16 multiplies a 16 x 16 A (M x K) by a 16 x 16 B (K x N) into a
// 16 x 16 FP32 accumulator held across the 64 lanes, four values per lane. Operand layout
// (CK's WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16; fleet/tests/test_mfma_layout.py):
//   lane l holds A[l % 16][4 (l / 16) + i], B[4 (l / 16) + i][l % 16], D[4 (l / 16) + i][l % 16].
// On the host syntax check (no builtin) the wrapper returns the accumulator unchanged.
typedef short bf16x4_t __attribute__((ext_vector_type(4)));
typedef float f32x4_t __attribute__((ext_vector_type(4)));
#if defined(__HIP_DEVICE_COMPILE__) && __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k)
#define MLA_HAS_MFMA 1
__device__ __forceinline__ f32x4_t mfma_16x16x16_bf16(bf16x4_t a, bf16x4_t b, f32x4_t c) {
  return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}
#else
__device__ __forceinline__ f32x4_t mfma_16x16x16_bf16(bf16x4_t, bf16x4_t, f32x4_t c) {
  return c;
}
#endif

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

#ifndef MLA_STREAM_LOADS
template <typename T>
__device__ __forceinline__ void load8_from(StreamSrc<T> const &s, size_t elem, float out[8]) {
  load8(s.base + elem, out);
}
#endif

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
