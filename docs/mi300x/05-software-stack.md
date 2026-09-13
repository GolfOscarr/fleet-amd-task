# 05 — Software Stack

The "what is CUDA here" question. Nothing in this file is verified on the
machine yet; versions in particular must be pinned from what the MI300X box
actually has.

## The core toolchain

| Piece | CUDA analogue | Notes |
|---|---|---|
| **ROCm** | CUDA Toolkit | Driver + runtime + libraries. Version determines everything else. |
| **HIP** | CUDA runtime/language | `hipMalloc`, `hipLaunchKernelGGL`, `__global__`, `__shared__`. Close to a 1:1 mapping. |
| **hipcc** | `nvcc` | Clang-based. Target flag: `--offload-arch=gfx942`. |
| **ROCm LLVM** | NVCC/PTX backend | AMDGPU backend; `llvm-objdump` for disassembly, which we will need constantly. |
| **rocm-smi / amd-smi** | `nvidia-smi` | Clocks, power, partition mode. |
| **rocgdb** | `cuda-gdb` | Debugger. |

Build line for our work will be roughly:

```
hipcc --offload-arch=gfx942 -O3 \
      -Rpass-analysis=kernel-resource-usage \
      -save-temps  # keep the .s — we must read the emitted cache ops
```

The `-save-temps` / `llvm-objdump` habit is not optional here: per
`03-memory-model.md`, our correctness depends on specific instructions
(`buffer_inv sc1`, `buffer_wbl2 sc1`) actually being emitted.

## Math and operator libraries

| Library | Role for us |
|---|---|
| **hipBLASLt** | BF16 GEMM. ROCm 7 prefers it over rocBLAS on newer GPUs. Our **reference** for dense projections, and the fallback we measure against. |
| **rocBLAS** | Older BLAS. Superseded by hipBLASLt for this part; mentioned only because much existing code still calls it. |
| **Composable Kernel (CK)** | C++ templates generating fused tiled kernels; supports operator fusion and multiple precisions across Instinct ISAs. Useful as a source of proven MFMA tiling patterns for the task bodies. |
| **AITER** | AMD's tuned LLM operator library — the rough analogue of cuBLAS+cuDNN+FlashAttention+TE combined. Ships MLA decode and MoE kernels. Integrated into vLLM and SGLang. |
| **Triton (ROCm)** | Quick reference kernels; vLLM's `TRITON_MLA` backend. Good for a fast second opinion on numerics. |

### On AITER specifically

AITER matters to us twice over: as a **correctness oracle** for MLA decode and
MoE, and as a **source of tuned kernels** we may legitimately call as fallbacks
while the Fleet path is incomplete (the task explicitly asks us to report
"remaining fallbacks", so using them is expected, as long as we count them).

Caveat from secondary sources: AITER's newest MLA decode work targets **gfx950
(CDNA4 / MI350)** rather than our **gfx942**, and coverage on gfx942 is
reportedly uneven. Treat any claim that a given AITER kernel works on gfx942 as
unverified until we run it. There is also an open AITER issue about a gfx942
persistent MLA decode kernel faulting at `page_size=1` — worth knowing about
before we build on that path.

## Reference implementations for correctness

The task requires "a correctness reference" and permits a stock implementation
to do the 1,024-token prefill.

| Option | Pros | Cons |
|---|---|---|
| **HF Transformers on ROCm (PyTorch)** | Simplest, closest to the model's canonical definition, easy to dump per-layer activations | Slow; its KV layout will not match our MLA decode layout |
| **vLLM on ROCm** | Production MLA path, AITER backends, realistic KV cache | Heavier; its cache layout is paged and must be converted |
| **SGLang** | Also has DeepSeek MLA support on ROCm | Same conversion concern |

Recommended: **HF Transformers as the numerical oracle** (per-tensor comparison
at every boundary — that is what "correctness evidence at every completed
boundary" means), and a **stock prefill of our choosing** whose KV layout we
document and convert once, outside the measured decode window, exactly as the
task permits.

DeepSeek-Coder-V2-Lite-Base uses MLA plus MoE, so the model code path in
whichever reference we choose must genuinely support DeepSeek-V2 architecture —
verify before relying on it.

## What to pin and record

On first contact with the machine, capture and commit:

```
rocminfo | head -50            # agent, gfx target, CU count
amd-smi static                 # or rocm-smi -a
hipcc --version
python -c "import torch; print(torch.__version__, torch.version.hip)"
```

The exact ROCm version determines library availability and the code the
compiler emits for atomics, so it is part of the reproducibility claim the
final submission has to make.

## Open items

- ROCm version present on the machine, and whether it is new enough for the
  AITER/vLLM paths we want → `99-open-questions.md` Q9
- Whether AITER's gfx942 MLA decode kernel is usable as an oracle → Q10
