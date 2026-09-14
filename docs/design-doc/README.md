# Technical design: Fleet-style batch-1 decode for DeepSeek-Coder-V2-Lite-Base on MI300X

The first deliverable of the task. It describes how one decode step of the
model is executed as a Fleet task graph inside one persistent kernel on one
MI300X, what has to be built, how it will be shown correct, what it is
expected to cost, and what is done before the GPU is available. Every
quantitative claim is derived by `sources/graph_counts.py` or cited to a
primary source, the checkpoint, or a line of the Fleet code at `51dce4f`.

## Checklist against the task description

The eight items the "First deliverable" paragraph of `docs/task-description.pdf`
names, then what this set adds.

| Required by the task description | File |
|---|---|
| model execution flow | [`01-execution-flow.md`](01-execution-flow.md) |
| proposed Fleet task graph | [`02-task-graph.md`](02-task-graph.md) |
| synchronization strategy | [`03-synchronization.md`](03-synchronization.md) |
| memory plan | [`04-memory-plan.md`](04-memory-plan.md) |
| interface between reference prefill and Fleet decode, including the KV-cache layout; conversion excluded from timing | [`05-prefill-interface.md`](05-prefill-interface.md), [`04-memory-plan.md`](04-memory-plan.md) |
| correctness methodology | [`07-correctness.md`](07-correctness.md) |
| implementation milestones; the required MoE-layer milestone | [`08-milestones.md`](08-milestones.md) (M2 = layer 1) |
| expected performance | [`09-expected-performance.md`](09-expected-performance.md) |
| work that can be completed locally before using the GPU | [`10-local-work.md`](10-local-work.md) |

| Added by this set | File |
|---|---|
| the optimization strategy (this stage's brief) | [`06-optimization-strategy.md`](06-optimization-strategy.md) |
| decisions, with evidence and reversal conditions | [`00-decisions.md`](00-decisions.md) |
| what is still unknown at the design level | [`99-open-questions.md`](99-open-questions.md) |

The plan that produced this set, and the checks it was written against, is
[`PLAN.md`](PLAN.md).

## One page

**What runs.** The reference (HF, BF16) prefills the 1,024-token prompt; the
latent KV cache for positions 0..1022 is captured from its own intermediate
outputs, exactly. One `mpk()` call then runs 32 decode iterations inside the
Fleet worker kernel: three kernel dispatches for the whole generation. Each
iteration is a chain of 326 operators and 1,880 tasks: an embed, 27 layers,
and the head, all resolved on the device (`01`, `02`).

**What is reused and what is new.** 217 of the 326 operators, moving 96.9% of
the bytes, use kernels Fleet ships for MI300: the gang linears, the gang
MoE experts, norms, argmax. Four kernels are new: the MLA path (`mla_prep`,
`mla_attend`, `mla_merge_uv`) and a fused FP32 router. Two one-line variants
close gaps in the runtime's online mode (`02`, `04`).

**The three decisions that shaped the graph.** The runtime's dependency
model is a strict chain, so the graph fuses where the natural DAG branched:
`q_proj` with `kv_a_proj`; the router with top-k; and the two shared experts
into the routed set as always-selected experts 64 and 65, which also puts
exactly one active expert on each of the eight XCDs (`00` D6, D7, D9).
Runtime reassociation of the MLA weights keeps the cache at 1,152 bytes per
position without materializing anything (`00` D4).

**Synchronization.** The runtime's release-acquire protocol, unchanged:
per-XCD counters, `buffer_wbl2 sc1` on release and `buffer_inv sc1` on
acquire through one compiler builtin, verified by disassembly on day 1.
Our graph has 1,838 releases and 1,879 acquires per iteration; every task's
acquire invalidates its XCD's L2, so the design does not count on L2
residency of anything (`03`).

**Correctness.** Sixteen boundaries against the HF reference, thresholds
set at 4x a calibrated BF16 floor, expert indices and all 32 output tokens
exact. Layer 1 is validated in a truncated one-iteration graph; the
reference's cache row at position 1023 is a free per-layer check of the
append path (`05`, `07`).

**Performance.** Bytes are fixed: 4,709.9 MiB per token, a 931.8 us floor,
a 1,148-1,349 us bandwidth band. Two terms are not fixed and are exposed by
the chain: the latency of 326 operator boundaries, and the serial ops
(`mla_prep` as one workgroup). The layer-1 measurement on day 3 calibrates
both; three boundary-count reductions are documented if they are large. The
design promises the band plus two measured terms, and says what Fleet's own
data supports at batch 1: dispatch reduction, not bandwidth (`06`, `09`).

**Schedule.** Day 1 builds for gfx942 and verifies the machine (gate 1);
day 2 one operator; day 3 layer 1 (gate 2, the required milestone); day 4
all layers and end-to-end; day 5 measurement and the report. FP8 weight-only
(1.83x) is the recommended next step, not scheduled (`08`, `10`).

## Key numbers

| | |
|---|---|
| Operators / tasks per iteration | 326 / 1,880 |
| Kernel dispatches per 32-token generation | 3 |
| Bytes per token | 4,709.9 MiB (weights 4,679.5, cache 30.4) |
| Floor / band | 931.8 us / 1,148-1,349 us at 4.3-3.66 TB/s |
| Layer 1 | 159.63 MiB, 31.6 us floor, 38.9-45.7 us band, 12 boundaries |
| New kernels | 4 (107 of 326 ops, 3.1% of bytes) |
| Release / acquire fences per iteration | 1,838 / 1,879 |
| Latent cache | 2 x 27 arrays, 31.3 MiB, 1,152 B per position |
| Hand-over | position 1023; `step = 1022` before launch; 32 iterations to `step = 1054` |

## Reading order

`00` for the decisions, `01` for what runs, `02` for how it is declared,
`03` for why it is correct across XCDs, `04`-`05` for memory and the
interface, `07`-`08` for how it is shown correct and when, `06` and `09`
for what to expect and why, `10` for what is already done, `99` for what is
not known.
