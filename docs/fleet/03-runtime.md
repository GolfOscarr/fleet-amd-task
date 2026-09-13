# 03 — Runtime: Scheduler, Workers, Synchronization

## Persistent kernel structure

Fleet launches **a single HIP kernel occupying every CU on the GPU**. Per XCD,
one workgroup is the **scheduler**; the rest are **workers**.

- Each scheduler discovers its XCD identity by reading a hardware register (the
  paper says `HW_ID`; the code reads **`HW_REG_XCC_ID`** — see below) and
  maintains per-worker task queues in global memory.
- CUs occupied by scheduler blocks are unavailable for compute: 8 of 256 CUs
  (3.1%) on MI350. On MI300X that is 8 of 304 (2.6%).
- Schedulers perform only lightweight control operations — counter reads, queue
  writes, pointer arithmetic — so workers "rarely stall waiting for task
  assignment". The paper does flag that with very short tasks, scheduling
  overhead could become significant.

### XCD identity — paper vs code

The paper §5.1 says the scheduler reads "the hardware `HW_ID` register". The
code does something more specific:

```c
// include/mirage/persistent_kernel/persistent_kernel.cuh:186
__device__ __forceinline__ int get_current_xcd_id() {
  int xcd_id;
  asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
  return xcd_id;
}
```

This **resolves our `../mi300x/99-open-questions.md` Q2**: the symbolic name
`HW_REG_XCC_ID` is accepted by the ROCm assembler, and it is a distinct register
from `HW_ID` — consistent with CDNA3 ISA Table 7 (hwreg code 20) and *not* with
the paper's prose. Take the code as authoritative.

Note they read **16 bits** (`hwreg(HW_REG_XCC_ID, 0, 16)`) where the ISA defines
`XCC_ID` as bits 3:0. Reading wider is harmless if the upper bits read zero;
worth a one-line check that values land in 0–7 on MI300X.

Leader election is a plain `atomicCAS` on a per-XCD slot:

```c
int old = atomicCAS(&config.xcd_leader_worker[xcd_id], -1, worker_id);
```

## Hierarchical synchronization — four levels

The core claim: match each communication pattern to the narrowest sufficient
memory scope.

| # | Pattern | Scope | Mechanism |
|---|---|---|---|
| 1 | **Task queue** | none | Populated before launch, immutable during execution. No writer contention, so workers and schedulers read task metadata **without any synchronization**. |
| 2 | **Scheduler → worker dispatch** | L2-local | Scheduler writes per-worker queues with device-scope stores/atomics. Scheduler and its workers are on the same XCD, so these resolve in the local L2 — **no cross-XCD coherence traffic**. |
| 3 | **Worker → worker within a chiplet** | L2-local | Sub-task completion counts accumulate in per-XCD counters via device-scope atomics, resolving in the local L2. **No fence required** — all participating workers share the L2 partition. |
| 4 | **XCD → global signaling** | GPU scope | Only the **last worker to complete on each XCD** issues a single `threadfence` and updates the global event counter with a GPU-scope atomic (`flat_atomic_add` with `sc0 sc1`). When the global counter reaches its threshold, the completing worker enqueues the event to the responsible scheduler's queue. |

Level 4 is the whole trick: it amortizes the expensive cross-XCD coherence cost
across all workers in the XCD. Claimed effect: **14.5× less cross-chiplet fence
traffic**. The code puts a number on it:

```c
// Two-level event counting: XCD-local first, batch flush when
// this XCD's share is done. Reduces buffer_wbl2 from ~24K to ~300.
```

~24,000 → ~300 writebacks. Given that `buffer_wbl2` flushes dirty L2 lines and
"penalizes all subsequent memory accesses until the flush completes", this is
plausibly the single most important implementation detail in the system.

## Cache modifier policy

Three tiers (paper §4.1), all reusable by us:

| Data | Policy | Rationale |
|---|---|---|
| **Weight loads** | cache-streaming, `sc1=1, nt=1` | Read once per GEMM; temporarily allocate in L2 and mark for immediate eviction. Cooperative M-tile sharing still hits because consecutive workers touch the same column in a short window. |
| **Activation stores** | non-temporal, `NT=1` | Bypass L2 so transient GEMM outputs, RMSNorm results and SiLU activations do not evict weight tiles. |
| **Scheduler communication** | non-temporal loads cross-XCD; volatile loads intra-XCD | Cross-XCD polling must bypass stale L2 copies and read fresh from HBM; intra-XCD can use the shared L2. |

The repo's `mpk_atoms.cuh` documents the bit behaviour and adds a caution worth
repeating:

> "Why fence is still needed with NT: NT provides cache bypass but NOT memory
> ordering. ... The pattern is: Producer: `st_nt(data)` → fence → `st_nt(flag)`;
> Consumer: `ld_nt(flag)` → fence → `ld_nt(data)`."

It also provides write-through stores (`sc0=1, sc1=1`) that bypass L2 entirely,
with the note "No `buffer_wbl2` needed after WT stores — data is already in
memory."

## Occupancy consequence

From §8: compiling all task types into one GPU function makes the register
footprint the **union** of every task's, which "limits occupancy to a single
wave per SIMD, eliminating latency hiding from wave switching. Every L2 miss
directly stalls the MFMA pipeline."

For our memory-bound batch-1 workload this is the main structural risk: 1
wave/SIMD means few outstanding memory requests per SIMD, and saturating
5.3 TB/s needs many. Adding a large MLA task to the union can only make it
worse. **Track VGPR count per task from the first commit** — see
`99-open-questions.md` Q3.

## In-kernel timing

The runtime already uses `__builtin_amdgcn_s_memrealtime()` (`profiler.h`), and
the paper reports latency from "on-GPU cycle counters ... using GPU-side
per-iteration timestamps (`s_memrealtime`), excluding prefill". So the
instrumentation we identified as necessary in `../mi300x/06-profiling.md`
already exists — we should reuse `profiler.h` rather than build our own.
