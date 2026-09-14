# Fleet-style batch-1 decode for DeepSeek-Coder-V2-Lite on MI300X

Implementing a [Fleet](https://arxiv.org/abs/2604.15379)-style megakernel decode
path for **DeepSeek-Coder-V2-Lite-Base** on a single **AMD Instinct MI300X**:
keep GPU workers resident and coordinate dependent operations through an
on-device task graph, instead of launching ~800–1,000 separate kernels per
decoded token.

## The task

| | |
|---|---|
| Model | DeepSeek-Coder-V2-Lite-Base — 27 layers, MLA attention, 64-expert MoE, 15.7 B params |
| Hardware | 1 × MI300X (CDNA 3, `gfx942`, 8 XCDs × 38 CUs, 192 GB HBM3) |
| Execution | batch-1 autoregressive decode |
| Precision | BF16 initially |
| Input / output | 1,024-token prompt → 32 greedily decoded tokens |
| Excluded | tensor parallelism, continuous batching, speculative decoding, prefill optimization |
| Required milestone | one validated MoE decoder layer at index ≥ 1 — **layer 1** |
| Time limit | 5 days |

Full brief: [`docs/task-description.pdf`](docs/task-description.pdf).

## Status

**Discovery complete. Nothing built yet.** The next deliverable is the technical
design document; implementation is blocked on MI300X access.

See [`PROGRESS.md`](PROGRESS.md) for the milestone tracker and
[`OPEN-PROBLEMS.md`](OPEN-PROBLEMS.md) for everything still unresolved
(6 major / 18 minor, plus 9 defects found in AMD's and Fleet's own docs).

## Layout

```
docs/
  mi300x/            hardware: architecture, chiplet dispatch, gfx942 memory
                     model, persistent-kernel mechanics, profiling, bandwidth
  deepseek-v2-lite/  model: config, MLA, MoE, per-op tensor flow, weights,
                     roofline, correctness methodology
  fleet/             the approach: paper review, task model, runtime, repo map,
                     sync cross-check, our task graph, gap analysis
  acceleration/      techniques: precision, decode parallelism, kernel craft,
                     and a ledger ranking everything by value
  mla-decode/        the one kernel with no prior art in Fleet: implementation
                     survey and our kernel spec
  paper/             the Fleet paper
repos/
  fleet-chiplet-megakernel/   ROCm/fleet-chiplet-megakernel (submodule)
```

Each doc set has a `README.md` index, numbered topic files, a
`99-open-questions.md`, and a `sources/` directory holding the primary documents
and scripts its claims derive from.

## Where to start

| If you want | Read |
|---|---|
| The plan and its state | [`PROGRESS.md`](PROGRESS.md) |
| What is still unknown | [`OPEN-PROBLEMS.md`](OPEN-PROBLEMS.md) |
| Why this is hard | [`docs/deepseek-v2-lite/07-roofline.md`](docs/deepseek-v2-lite/07-roofline.md) |
| The correctness hazard | [`docs/mi300x/03-memory-model.md`](docs/mi300x/03-memory-model.md) |
| What we actually build | [`docs/mla-decode/04-our-kernel-spec.md`](docs/mla-decode/04-our-kernel-spec.md) |
| The task graph | [`docs/fleet/06-our-task-graph.md`](docs/fleet/06-our-task-graph.md) |

## What the analysis established

**Batch-1 decode is memory-bound and nothing else matters much.** 4,705.9 MiB
must be read per token, 55% of it routed-expert weights. Against MI300X's
5.3 TB/s theoretical peak that is a **931 µs floor**; against realistically
achievable bandwidth (3.66–4.3 TB/s) the band is **1.15–1.35 ms/token,
742–871 tok/s**.

**Fleet's batch-1 benefit is dispatch overhead, not bandwidth.** Their own
measurements show L2 hit rate moving 16.4% → 16.9% and HBM reads at 0.98× at
batch 1; the cooperative-tiling win only activates at batch ≥ 32. We inherit the
task-count collapse (~2,135 tasks in **one** kernel launch), not a traffic
reduction.

**The eight XCD L2 caches are not coherent.** In the default SPX mode a single
agent spans eight private L2s. Cross-XCD visibility requires `buffer_wbl2 sc1`
on release and `buffer_inv sc1` on acquire — and both are documented no-ops on
single-L2 parts, so code can appear correct elsewhere and fail here. Confirmed
independently from LLVM's memory model, the CDNA 3 ISA, and Fleet's own source.

**MoE already works in Fleet; MLA does not.** The gap is one kernel, specified
in `docs/mla-decode/04-our-kernel-spec.md`.

**The conventional absorbed-MLA advice is wrong at this context length.**
Materializing the fused weights costs +44 MiB/layer to save 8.9 — about 2× worse
than doing nothing at 1,024 tokens. Runtime reassociation gets the cache
reduction for free, which is also what vLLM does.

**FP8 is worth more than everything else combined** — 1.83×, and a quantized
checkpoint exists for this exact model. It is a stretch goal, since the task
requires BF16 first.

## Method

Every quantitative claim is derived from a primary source and re-derived by
script. Facts carry a verification level — `primary`, `checkpoint`, `derived`,
`secondary`, `machine` — and anything not `primary` or `checkpoint` that the
design depends on is listed as an open problem with the check that settles it.

The parameter count computed from `config.json` alone reproduces the
checkpoint's published `total_size` of 31,412,968,448 bytes **exactly**, which is
the strongest available evidence that the architecture is understood correctly.

Analysis scripts live beside the documents they support:

```bash
python3 docs/deepseek-v2-lite/sources/roofline.py     # traffic and roofline
python3 docs/acceleration/sources/fp8_roofline.py     # precision scenarios
python3 docs/mi300x/sources/bandwidth_analysis.py     # bandwidth + prefetch depth
```

## Setup

```bash
git clone --recursive https://github.com/GolfOscarr/fleet-amd-task.git
```

The submodule pins `ROCm/fleet-chiplet-megakernel` at `51dce4f`. It targets
`gfx950` by default; building for MI300X needs `AMDGPU_TARGETS=gfx942`, which is
untested and is the day-1 blocking question.

Model weights (31 GB BF16) are not tracked here — download them on the target
machine.

## Deliverables

| Required | Where |
|---|---|
| Technical design | *next* |
| Source, build and run instructions | *pending GPU access* |
| Correctness evidence per boundary | method in [`docs/deepseek-v2-lite/08-correctness.md`](docs/deepseek-v2-lite/08-correctness.md) |
| Profiling commands and results | plan in [`docs/mi300x/06-profiling.md`](docs/mi300x/06-profiling.md) |
| Milestone reached, fallbacks, known failures | [`PROGRESS.md`](PROGRESS.md), [`OPEN-PROBLEMS.md`](OPEN-PROBLEMS.md) |
| Recommended next steps | [`docs/acceleration/04-technique-ledger.md`](docs/acceleration/04-technique-ledger.md) |
