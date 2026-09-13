# 02 — The Fleet Task Model

## Four scopes

| Level | HW scope | Memory | Typical op | Workers |
|---|---|---|---|---|
| Wavefront-task | 1 wavefront (64 threads) | Registers, LDS | SiLU, elementwise multiply, residual add, RoPE | 1 |
| CU-task | 1 workgroup | LDS, L2 | individual attention heads, RMSNorm, argmax/sampling | 1 |
| **Chiplet-task** | 1 XCD | L2, HBM | GEMM partition | 31 (MI350) / **37 (MI300X)** |
| Device-task | all 8 XCDs | Global HBM | full GEMM / attention | 248 (MI350) / **296 (MI300X)** |

The worker counts are `CUs_per_XCD − 1`, since one CU per XCD is the scheduler.
On MI300X with 38 active CUs per XCD that is 37 workers per chiplet and 296
total — **derived**, not stated in the paper, and worth confirming at runtime.

The abstraction is parameterized by chiplet count `X`, workers per chiplet `W`,
and L2 capacity `C`, queried at runtime. That is what makes MI300X support a
configuration rather than a port.

## Chiplet-tasks are the new idea

A Chiplet-task binds *work* and *data* to one XCD's private L2:

- The scheduler **broadcasts the same task** to every worker on its chiplet
  (versus round-robin dispatch of distinct tasks for CU- and wavefront-tasks).
- The programmer specifies the data partition, the tiling, and an explicit
  **L2 cache budget** for the chiplet's working set.
- For an `[M,K] × [K,N]` GEMM, each Chiplet-task computes an independent
  `[M,K] × [K,N/8] = [M,N/8]` sub-matrix (N-split). No cross-XCD reduction.

## Device-tasks have barrier semantics

A device-task completes only when all 8 Chiplet-tasks finish. Each XCD writes
its output columns at a strided offset, so the result is assembled in place
with no reduction step. For operators partitioned along the reduction dimension
(K-split), the device-task adds a reduction phase after the chiplet-tasks.

## Dependencies are events

Each task declares:
- which **event it signals** on completion, and
- which **events it waits on** before it may start.

The key economy: because a Chiplet-task groups all workers on a chiplet into a
single unit, **one event per chiplet per dependency edge** suffices rather than
one per worker — a `W`× reduction in synchronization events (`W` ≈ 31–37).

A "linear event" has exactly eight tasks, one per XCD, so it triggers exactly
eight fences total. For wavefront- and CU-tasks that run on a single CU, the
worker signals the global event counter directly; no two-level counting is
needed since there is only one worker per task.

## Task graph size, measured

For one Qwen3-8B transformer layer at bs=1 (paper Figure 4):

| | Standard (Mirage) | Fleet |
|---|---|---|
| RMSNorm | 1 task | 1 CU-task |
| QKV Proj | 96 tasks | 8 Chiplet-tasks |
| Attention | 8+8 tasks | 8+8 CU-tasks |
| O-Proj + Res | 256 tasks | 8 Chiplet-tasks |
| RMSNorm | 1 task | 1 CU-task |
| Gate+Up (+SiLU) | 192 tasks + 96 SiLU | 8 Chiplet-tasks, **SiLU fused** |
| Down + Res | 256 tasks | 8 Chiplet-tasks |
| **Total** | **1,407** | **543** (2.6× fewer) |

Every GEMM collapses from 96–256 CU-tasks to exactly **8 Chiplet-tasks**. That
collapse — not cache behaviour — is the batch-1 win.

## Task registration and code generation

Fleet extends the Mirage compilation pipeline. The Mirage compiler generates GPU
kernel code **at model load time** by compiling an abstract task graph with a
runtime description including library code per node.

Two notes that matter for extending it:

- Fleet "currently provide[s] different input code to the Mirage compiler and do
  not rely on compiler-driven super-optimization" for Chiplet-tasks. So
  Chiplet-tasks are hand-written library kernels wired in, not auto-generated.
  **Adding MLA means writing task kernels by hand**, which is what we expected.
- The emitted call is `_execute_xcd_task(tile_idx)` rather than `_execute_task()`:
  within a Chiplet-task each CU receives the *same* base pointer, and the tile
  index selects its slice. So the library kernel is responsible for per-CU
  behaviour based on `tile_idx`. This is visible in the repo's `gang_*` kernels.

## What this means for a data-dependent op (MoE)

The task graph is static in *structure* — the set of tasks and edges is fixed at
compile time. Data dependence is handled **inside** a task: the MoE linear task
receives `routing_ptr` and `mask_ptr` as inputs and resolves which expert weights
to read at execution time. The graph does not grow or shrink per token.

That is the pattern we inherit for DeepSeek's top-6-of-64 routing, and it is
why MoE did not require a new task-model capability. See `07-gap-analysis.md`.
