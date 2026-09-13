# Fleet — Review and Adaptation Plan

How to adapt Fleet to DeepSeek-Coder-V2-Lite-Base on MI300X.

Sources: `../paper/fleet.pdf` (11 pp., read in full) and
`../../repos/fleet-chiplet-megakernel` @ `51dce4f`.

## Files

| File | Contents |
|---|---|
| `01-paper-review.md` | Section-by-section, with the batch-1 numbers isolated |
| `02-task-model.md` | The four task scopes, dependencies, events |
| `03-runtime.md` | Scheduler/worker protocol, hierarchical sync, cache policy |
| `04-repo-map.md` | What's in the repo, what builds for gfx942, what exists for MI300 |
| `05-sync-crosscheck.md` | Fleet's sync vs our `../mi300x/03-memory-model.md` |
| `06-our-task-graph.md` | DeepSeek-V2-Lite layer 1 expressed as Fleet tasks |
| `07-gap-analysis.md` | What we must add; corrected after reading the code |
| `99-open-questions.md` | |

## What Fleet actually is

**Fleet is a fork of Mirage Persistent Kernel (MPK)**, not a standalone system.
The repo is the Mirage tree (`python/mirage/`, `MIRAGE_HOME`) with chiplet-aware
scheduling added. Fleet contributes two things on top of MPK:

1. **Chiplet-tasks** — all ~30 workers on one XCD cooperate on tiles of the same
   GEMM, so the weight slice stays in that XCD's 4 MB L2.
2. **Hierarchical synchronization** — two-level event counting that confines
   most coherence traffic to the XCD-local L2, claimed at 14.5× less
   cross-chiplet fence traffic.

## The three findings that most change our plan

**1. At batch 1, Fleet's benefit is dispatch overhead, not cache.** Their own
Table 4 is unambiguous: at bs=1, L2 hit is 16.4% (Mirage) vs 16.9–17.0%
(Fleet), and HBM reads are **0.98–0.99× — no traffic reduction at all**. The
cooperative-tiling story only activates at bs≥32 where `m_tiles ≥ 2`. Their own
model (Eq. 1) predicts *zero* weight reuse at bs=1.

Since we are batch-1-only, **we should not expect and should not promise an L2
or bandwidth win.** Our roofline of 931 µs stands unchanged; what Fleet buys is
the gap between the roofline and reality. Stating this up front is more
defensible than discovering it in the results.

**2. MoE is already implemented for MI300** — my earlier gap analysis was wrong
about this. The repo has `gang_moe_linear_mi300.cuh`, `moe_linear_mi300.cuh`,
`moe_topk_softmax_mi300.cuh`, `moe_mul_sum_add_mi300.cuh`, Python graph-builder
methods (`moe_topk_softmax_routing_layer`, `moe_w13_linear_layer`), and a
Qwen3-30B-A3B MoE demo. Data-dependent expert dispatch is a solved problem in
this codebase. **The real gap is MLA**, not MoE.

**3. MI300X support is designed in, not accidental.** The ablation states the
Chiplet-task abstraction is parameterized by chiplet count, workers per chiplet,
and L2 capacity, "allowing the same task graph and scheduling logic to adapt to
MI300X (8 XCDs, 38 CUs, 4 MB L2) and MI350 (8 XCDs, 32 CUs, 4 MB L2) without
code changes." There is a whole `tasks/mi300/` directory, `MIRAGE_AMD_MI300`
guards throughout, and `AMDGPU_TARGETS=gfx942` overrides the gfx950 CMake
default.

## Headline numbers, as published

Qwen3-8B, bf16, MI350, decode-only TPOT:

| Batch | vLLM | Mirage MPK | Fleet M-tile | Fleet M-split |
|---|---|---|---|---|
| 1 | 10.51 ms | 7.993 ms | 7.076 ms | 7.012 ms |
| 32 | ~11–12 ms | 15.186 ms | 12.357 ms | 12.923 ms |
| 64 | ~11–12 ms | 23.343 ms | 18.176 ms | 21.923 ms |

Note the decomposition at bs=1: **Mirage alone is already 1.34× faster than
vLLM purely from eliminating launch overhead** (paper §6.2: "7.83 vs 10.51");
Fleet adds only a further ~1.16×. The abstract's "1.3–1.5× vs vLLM" is the
product of the two, and **most of it is not Fleet's contribution** — it is the
persistent kernel's.

Caveat: the paper and the repo README report different bs=1 numbers — paper
§6.2 gives Mirage 7.83 / Fleet 6.82 (M-tile) / 6.73 (M-split) ms; the README
table gives 7.993 / 7.076 / 7.012 ms. About 4% apart, with consistent internal
ratios. Unexplained; if we cite Fleet's numbers we should say which source.

## Go / no-go

| Question | Answer | Confidence |
|---|---|---|
| Does the abstraction target MI300X? | Yes, explicitly | paper §8 |
| Do gfx942 kernels exist? | Yes, 31 files in `tasks/mi300/` | repo |
| Does MoE work? | Yes, tasks + Python + MoE demo | repo |
| Does MLA work? | **No** — no MLA/DeepSeek code found | repo |
| Does it build for gfx942? | `AMDGPU_TARGETS=gfx942`; **untested** | repo |
| Is there a DeepSeek model builder? | **No** — only `qwen3` | repo |

**Recommendation: extend the existing runtime rather than reimplement.** Our
work concentrates on MLA tasks plus a DeepSeek-V2 graph builder, with MoE,
scheduling, and synchronization inherited. That is a far better use of five days
than rebuilding a persistent-kernel runtime.
