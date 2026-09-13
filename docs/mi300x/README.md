# MI300X Reference Notes

Working notes on the target hardware for the Fleet-style batch-1 decode path
(DeepSeek-Coder-V2-Lite-Base, one AMD MI300X).

## Files

| File | Contents |
|---|---|
| `01-architecture.md` | XCD/CU/cache/HBM hierarchy, register and LDS budgets, BF16 matrix rates |
| `02-chiplet-dispatch.md` | Workgroup→XCD mapping, `XCC_ID`, SPX/DPX/CPX, NPS1/2/4 |
| `03-memory-model.md` | gfx942 coherence, cache-policy bits, exact acquire/release sequences |
| `04-persistent-kernel.md` | Occupancy math, spin/sleep/wake primitives, on-device timing |
| `05-software-stack.md` | ROCm, HIP, hipBLASLt, CK, AITER, Triton, vLLM/SGLang |
| `06-profiling.md` | rocprofv3 / rocprof-compute, counters for each required metric |
| `99-open-questions.md` | Unverified claims and the microbenchmark that settles each |
| `sources/` | Downloaded primary documents |

## Conventions

Every factual table carries a `source` and a `verified` column.

| `verified` | Meaning |
|---|---|
| `primary` | Stated in an AMD or LLVM primary document, quoted in these notes |
| `secondary` | Only from a blog, article, or search summary — treat as a hypothesis |
| `derived` | Computed here from primary values; arithmetic shown |
| `machine` | Confirmed on the actual MI300X by a command or microbenchmark |

Nothing is `machine` yet — we do not have GPU access at time of writing.
Anything not `primary` or `machine` that our design depends on belongs in
`99-open-questions.md`.

## The three facts that most shape the design

1. **A single agent in SPX mode has eight separate, mutually non-coherent L2
   caches** — one per XCD. Cross-XCD visibility is not automatic; it needs
   explicit writeback and invalidate. See `03-memory-model.md`.
2. **`XCC_ID` is readable from inside a kernel** (hardware register 20), so a
   persistent kernel can discover which chiplet it landed on rather than
   guessing from the workgroup ID. See `02-chiplet-dispatch.md`.
3. **`S_WAKEUP` only wakes waves in the same threadgroup**, so the cheap
   sleep/wake handshake does not extend across workgroups; cross-workgroup
   waiting must poll. See `04-persistent-kernel.md`.

## Primary sources

| Document | Location | Retrieved |
|---|---|---|
| *"AMD Instinct MI300" Instruction Set Architecture Reference Guide* (561 pp.) | `sources/cdna3-isa.pdf` | 2026-09-13 |
| *Fleet: Hierarchical Task-based Abstraction for Megakernels on Multi-Die GPUs* | `../paper/fleet.pdf` | 2026-09-13 |
| LLVM *User Guide for AMDGPU Backend*, Memory Model GFX942 | https://llvm.org/docs/AMDGPUUsage.html | 2026-09-13 |
| ROCm *MI300 series microarchitecture* | https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300.html | 2026-09-13 |
| ROCm *MI300X GPU Partitioning Overview* | https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/gpu-partitioning/mi300x/overview.html | 2026-09-13 |
| ROCm *MI300/MI200 performance counters* | https://rocm.docs.amd.com/en/docs-6.3.3/conceptual/gpu-arch/mi300-mi200-performance-counters.html | 2026-09-13 |
