# 01 — Paper Review

*Fleet: Hierarchical Task-based Abstraction for Megakernels on Multi-Die GPUs*,
Chowdhary et al., AMD Research, arXiv:2604.15379v1, 15 Apr 2026. Submitted to
ASPLOS 2027.

## §1–2 Background and motivation

The CUDA/HIP model assumes a monolithic GPU with one unified L2. Chiplet GPUs
break that: MI300X and MI350 have **8 XCDs each with a private, non-coherent
4 MB L2**, plus a 256 MB MALL (Infinity Cache) acting as a victim cache for L2
evictions between L2 and HBM.

> "This architectural shift introduces NUMA-like programming effects within a
> single device which were previously only present between devices."

Table 1 maps the hierarchy: SIMD→wavefront, CU→workgroup, **Chiplet→(no CUDA/HIP
abstraction)**, Device→grid. The Chiplet level is the gap Fleet fills.

### Cache scope bits (paper §2.1)

> "The specific behavior on CDNA3/CDNA4 is governed by two scope bits (SC1, SC0)
> and a non-temporal bit (NT) on each memory instruction. SC1 and SC0 encode the
> coherence scope: `0_0`=wave, `0_1`=group, `1_0`=device, `1_1`=system. Default
> loads and stores use SC1=0, SC0=0, NT=0 (wave scope), which caches data in both
> L1 (TCP, read-only) and L2 with an LRU policy. With wave scope, the L2 treats
> data as local to the XCD: no cross-XCD coherence probes are issued, and stale
> copies on peer XCDs are not invalidated."

> "Making data produced by XCD `i` visible to XCD `j` requires explicit software
> action: the producer must issue `buffer_wbl2` to write back dirty L2 lines, and
> the consumer must invalidate stale L2 entries."

This corroborates `../mi300x/03-memory-model.md` from an independent source.
Cross-check in `05-sync-crosscheck.md`.

### Why decode is the target (§2.2)

Table 2, chiplet-unaware decode on MI350:

| Metric | Linear | Attention |
|---|---|---|
| % of decode time | **95%** | 5% |
| Weight working set / layer | 368 MB | — |
| Weight per XCD (uniform) | 46 MB | — |
| XCD L2 capacity | 4 MB | 4 MB |
| **L2 hit rate (bs=1, no coop)** | **16.4%** | — |
| Cycles per task | ~104K | ~3.8K |

> "a standard serving system executes each operator as a separate GPU kernel,
> resulting in almost 250 launches per decode token across 36 layers."

For Qwen3-8B. Our DeepSeek-V2-Lite estimate is 800–1,000/token
(`../deepseek-v2-lite/04-tensor-flow.md`) — higher because MoE adds ~24 ops per
layer. Same order, worse starting point, which is favourable for us.

### §2.3 Why not HIP graphs

Graph capture bakes in kernel arguments, grid dims, and memory pointers; a
mismatch falls back to eager, "causing multi-× latency spikes". More
fundamentally, graphs keep kernel-scope boundaries: dependent nodes still
serialize, residual launch overhead remains, and **L2 is flushed at each kernel
boundary**, forcing intermediates through HBM. A persistent kernel pays the
launch cost once.

This settles the HIP-graph question I raised in the plan: it is a weaker
alternative, and the paper argues why. Still worth measuring as a cheap
comparison point if the megakernel stalls.

## §3 The task model

Table 3, reproduced:

| Level | HW scope | Memory | Typical op | Workers |
|---|---|---|---|---|
| Wavefront-task | 1 wavefront | Regs, LDS | SiLU, residual add | 1 |
| CU-task | 1 block | LDS, L2 | Attention, RMSNorm | 1 |
| **Chiplet-task** | **1 XCD** | **L2, HBM** | **GEMM partition** | **31** |
| Device-task | 8 XCDs | Global HBM | Full GEMM/attention | 248 |

- Chiplet-tasks "span all workers on a single XCD (31 CUs out of 32 on MI350),
  with an explicit **L2 cache budget**". The programmer specifies the data
  partition, tiling, and L2 working set per chiplet.
- Device-tasks have barrier semantics: complete only when all 8 Chiplet-tasks
  finish. Each XCD writes its columns at a strided offset, "assembling the result
  in place without reduction".
- **Dependencies are expressed as events**: each task declares which event it
  signals and which it waits on. Because a Chiplet-task groups all workers on a
  chiplet into one unit, **one event per chiplet per edge** is needed rather than
  one per worker — a `W`× reduction in synchronization events.

## §4 Chiplet-aware optimization

**Traversal order.** Within a Chiplet-task, *windowed M-major traversal*
(credited to HipKittens): each worker computes `[m,n]` output tiles walking down
activation rows before advancing the weight column, so consecutive workers share
the same weight tile in a short temporal window. N-major would have consecutive
workers each read a *different* tile, preventing reuse.

Two M-distribution strategies: **M-tile** (all XCDs get the same `m_tiles`,
enabling cooperative weight sharing) vs **M-split** (each XCD gets a disjoint
M-tile, isolating scheduling benefit from L2 benefit — the ablation control).

**Three-tier cache modifier policy** (§4.1) — directly reusable by us:

1. **Weight loads**: cache-streaming (`sc1=1, nt=1`). Weight tiles are read once
   per GEMM and need not persist across layers.
2. **Activation stores**: non-temporal (`NT=1`), bypassing L2 so transient GEMM
   outputs, RMSNorm results and SiLU activations "do not evict weight tiles".
3. **Scheduler communication**: non-temporal loads for cross-XCD event polling
   (bypassing stale L2 copies, reading fresh from HBM); intra-XCD communication
   uses volatile loads through the shared L2.

This confirms the hypothesis I recorded in `../mi300x/07-*`/`../deepseek-v2-lite/07-roofline.md`
about non-temporal weight loads — except note the **inversion**: Fleet marks
*weights* streaming and *activations* non-temporal, whereas I had assumed we'd
mark weights non-temporal to protect the *KV cache*. Their model has no latent
KV cache worth protecting; ours does. See `07-gap-analysis.md`.

**Operator fusion.** Fusing SiLU into the gate+up Chiplet-task raised L2 hit
from 9.4% to 17.4% at bs=1 — "the L2 benefit comes from eliminating intermediate
buffer traffic rather than from XCD-scoped coordination". At bs=1 this fusion is
**the entire source** of the measured 16.9% L2 hit rate (§6.4).

**N-split vs K-split.** N-split (partition output columns across XCDs, no
cross-XCD reduction) "dominates at small batch sizes (bs=1–16) where persistent
kernel scheduling overhead is the bottleneck", while K-split with wave-level
reduction "excels at bs≥32". **We are bs=1: use N-split.**

## §5 System design

Covered in `03-runtime.md`.

## §6 Evaluation

Setup: single MI350X, 256 CUs / 8 XCDs / 32 CUs per XCD, 4 MB L2 per XCD, 256 MB
LLC, 288 GB HBM3. Qwen3-8B (4096 hidden, 36 layers, 12288 FFN, GQA 32 query /
8 KV heads), bf16, 64 input tokens / 1024 output tokens. Timed with in-kernel
`s_memrealtime`, decode-only, excluding prefill.

Baselines: (1) Mirage MPK ported to MI350X — the *internal* baseline isolating
Fleet's chiplet-awareness; (2) vLLM v0.17.2 ROCm `--enforce-eager` — the
*external* baseline.

### The batch-1 result, isolated

| | bs=1 |
|---|---|
| vLLM | 10.51 ms |
| Mirage MPK (chiplet-unaware persistent kernel) | 7.83 ms (paper) / 7.993 (README) → **1.34× vs vLLM** |
| Fleet M-tile | 6.82 ms (paper) / 7.076 (README) → **1.54× vs vLLM** |
| Fleet M-split | 6.73 ms (paper) / 7.012 (README) → **1.56× vs vLLM** |

So Fleet's own contribution over the persistent-kernel baseline at bs=1 is
**1.16×** (7.83 → 6.73), not 1.5×.

> "At small batch sizes (bs=1–16), where `m_tiles = 1` and no cooperative weight
> reuse occurs, both Fleet variants perform similarly because the improvement
> comes without cooperative weight sharing: eight Chiplet-tasks per GEMM versus
> 96–256 CU-tasks in the Mirage baseline."

Task-count reduction per transformer layer at bs=1: **1,407 → 543 tasks (2.6×
fewer)**, with SiLU fused into gate+up (Figure 4).

### Table 4 — the number that matters most to us

| BS | L2 Hit% Mirage | Fleet M-tile | Fleet M-split | HBM Rd (×Mirage) M-tile | M-split |
|---|---|---|---|---|---|
| **1** | **16.4** | **16.9** | **17.0** | **0.98** | **0.99** |
| 8 | 20.0 | 20.8 | 20.8 | 1.00 | 1.01 |
| 32 | 38.9 | **51.0** | 39.5 | 0.82 | 1.10 |
| 64 | 39.0 | **61.4** | 47.4 | 0.63 | 1.20 |

**At bs=1 there is no L2 improvement and no HBM traffic reduction.** All of
Fleet's batch-1 gain is reduced scheduler→worker dispatch overhead.

### §6.4 L2 hit rate model

`L2 Hit_weight = (R − 1)/R = 1 − 1/min(W, m_tiles)` where `R` is the number of
workers sharing a weight tile. At bs=1, `R=1`, predicting **zero** weight reuse;
the measured 16.9% comes entirely from fused SiLU. Without fused SiLU, bs=1 L2
hit drops to ~9%.

Table 5 (Qwen3-8B per-GEMM): all four GEMMs' per-XCD partitions exceed 4 MB, but
the *active* working set — one K-chunk tile per worker, 31 × 32 KB ≈ 1 MB —
stays resident. That is why M-major traversal works despite partitions 6× larger
than L2.

## §8 Ablation — the paragraphs that matter to us

**Generality across chiplet architectures.**

> "The Chiplet-task abstraction is parameterized by chiplet count (X), workers
> per chiplet (W), and L2 capacity per chiplet (C). Fleet queries these at
> runtime, allowing the same task graph and scheduling logic to adapt to MI300X
> (8 XCDs, 38 CUs, 4 MB L2) and MI350 (8 XCDs, 32 CUs, 4 MB L2) without code
> changes."

**Register pressure and occupancy** — the most important limitation for us:

> "The persistent megakernel compiles all task types into a single GPU function,
> and the combined register footprint limits occupancy to a single wave per SIMD,
> eliminating latency hiding from wave switching. Every L2 miss directly stalls
> the MFMA pipeline, making the L2 hit rate even more critical. A per-task
> compilation strategy could reduce register pressure but would sacrifice the
> single-launch property."

**1 wave/SIMD** means no latency hiding at all — for a memory-bound batch-1
workload where we *need* many outstanding loads to saturate 5.3 TB/s, this is a
serious structural concern. It also directly answers our
`../mi300x/99-open-questions.md` Q6: expect 1 wave/SIMD, 4 waves/CU.

**Interaction with TP.** Not applicable to us (single GPU, no TP), but note the
reasoning: TP shrinks the per-XCD partition, potentially below `C`, making
cooperative dispatch unnecessary.

## §7 Related work worth knowing

- **HazyResearch megakernel** — fuses a Llama decoder into one persistent kernel
  with an on-GPU interpreter, **78% of H100 memory bandwidth at bs=1**. That is a
  concrete bandwidth-efficiency target for a batch-1 megakernel; 78% of 5.3 TB/s
  would put us at ~1.19 ms/token against our 931 µs roofline.
- **FlashFormer** — whole-model fusion with pipelined shared buffers.
- **Mirage MPK** — the base system; 1.0–1.7× over SGLang/vLLM on A100/H100.
- **HipKittens** — ThunderKittens ported to CDNA3/CDNA4 with XCD grouping, +19%
  on standalone GEMMs. Source of the windowed M-major idea.

## §8 Stated limitations

> "Our evaluation covers one model (Qwen3-8B dense) on a single GPU architecture
> (MI350). We do not evaluate multi-GPU configurations, prefill performance, or
> models requiring tensor parallelism to fit in memory."

So: **the paper evaluates only a dense model on MI350.** The *code*, however,
supports MoE and has MI300 kernels — see `04-repo-map.md`. Evaluation scope and
implementation scope differ, and conflating them (as my pre-reading plan did)
leads to the wrong gap analysis.
