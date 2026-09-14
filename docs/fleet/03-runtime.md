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

---

## The code, read line by line (2026-09-14)

Everything above was written from the paper and a first pass over the source.
This section records what `persistent_kernel.cuh` (2,429 lines),
`mpk_atoms.cuh` (345 lines), `src/kernel/runtime.cc` and
`python/mirage/utils.py` actually do, because the design document's task
graph and synchronization sections must describe the code, not the paper.
Line numbers refer to the submodule at `51dce4f`.

### Launch structure: three dispatches per generation

`init_persistent_kernel` sets `split_worker_scheduler = true` unconditionally
(`persistent_kernel.cuh:2036`). `launch_persistent_kernel` (`:2260`) then
issues three kernels:

| Order | Kernel | Grid | Block | Stream | Role |
|---|---|---|---|---|---|
| 1 | `prepare_kernel` | 296 | 128 | default | zero every queue pointer, global and per-XCD event counter, XCD map and threshold table; seed scheduler 0's queue with the `EVENT_END_OF_TASK_GRAPH` event (`:297-377`) |
| 2 | `worker_kernel` | 296 | 256 | `worker_stream` | `execute_worker` (`:2287`) |
| 3 | `scheduler_kernel` | 8 | 128 | `scheduler_stream` | `execute_scheduler`, thread 0 only (`:2293`) |

The combined `persistent_kernel` with atomicCAS scheduler election per XCD
(`:1858-1906`) exists but is not on the path taken. Worker and scheduler
counts come from `utils.py:38-60`: on a device reporting 300 or more CUs,
`workers = sm_cnt - 8 = 296`, `schedulers = 8`. Nothing hard-codes 32 CUs per
XCD; the MI350 branch (`:61-68`) is a separate `elif`. This resolves
`99-open-questions.md` Q2.

Consequence for the metrics section: one generation of 32 tokens is 3 kernel
dispatches on the Fleet path, and the worker kernel runs all 32 decode
iterations without returning. The paper's "one launch" is the worker kernel.

The `__nanosleep` shim (`:75-85`) maps to `s_sleep 1` per 64 ns requested, so
`__nanosleep(10)` is one `s_sleep 1` (`:83`).

### Worker loop (`execute_worker`, `:693-1389`)

1. **Identity.** Thread 0 reads `HW_REG_XCC_ID`, writes `worker_xcd_map[w]`,
   bumps `worker_xcd_ready_count`, and elects an XCD leader by atomicCAS
   (`:743-771`). The leader flag is stored but the polling path below does
   not branch on it.
2. **Fetch.** Poll `worker_queue_last_ready_task_id[w]` with a volatile
   load (`ld_local_u64`) and `s_sleep 1` between polls (`:835-848`). Load
   up to 16 task ids (`TASK_DESCS_BUFFER_LENGTH`, `:702`), then copy the
   `TaskDesc` structs into LDS with `cp.async` (`:828-911`). Worker queues
   are 1,024 entries deep.
3. **Dependency check** (`:914-975`). If the task has a `dependent_event`,
   read `all_event_counters[e]` with a relaxed atomic load, then
   `__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent")` (`:948`). The needed
   count is `num_triggers * iteration_num`: counters are monotone across
   iterations and never reset within a launch. If not yet satisfied, spin
   with `s_sleep 1` and re-fence (`:966`). The comment says the scheduler's
   release-acquire chain makes this satisfied on arrival; the poll is a
   safety net.
4. **Execute.** `TASK_TERMINATE` returns (`:997`). `TASK_BEGIN_TASK_GRAPH`
   on worker 0 prints `[FWD_PASS] iter time_ms` from `get_wallclock_ns()`
   (`:1041-1053`), which is a per-iteration, therefore per-token, wall-clock
   figure available with no extra instrumentation. Gang tasks (the
   Chiplet-task family, `is_gang_task_type` `:226-236`) loop over tiles
   `t = xcd_local_rank; t < n_tile_count; t += workers_on_xcd` (`:1103`),
   where `xcd_local_rank` is the worker's rank among workers on the same
   XCD, computed once from `worker_xcd_map`. Everything else calls
   `_execute_task` once.
5. **Trigger** (`:1215-1275`). Two-level counting. The worker does
   `atom_add_local_u64` on `xcd_local_event_counters[xcd * num_events + e]`
   (a plain device-scope `atomicAdd`). Only when the XCD-local count reaches
   `xcd_event_num_tasks[slot] * iteration_num` (`:1239`, `:1263`) does it
   run `threadfence_gpu()` followed by `atom_add_release_gpu_u64` on the
   global counter. Gang tasks add the number of tiles they executed locally
   and flush 1 to the global counter; non-gang tasks add 1 locally and flush
   `xcd_threshold`. If the global count reaches `num_triggers *
   iteration_num`, the event has fired.
6. **Enqueue** (`:1315-1360`). `EVENT_LAUNCH_MASSIVE_TASKS` and
   `EVENT_LAUNCH_DEPENDENT_TASKS` go to the broadcast queue (index
   `num_schedulers`) with GPU-scope atomics and a fence. Everything else
   goes to the scheduler queue indexed by the worker's own `xcd_id`
   (`get_rand_sched_id`, `:584-597`) using volatile stores and a compiler
   fence only.

### Scheduler loop (`execute_scheduler`, `:1391-1855`)

Thread 0 of each scheduler block. It waits until all 296 workers have
written `worker_xcd_map` (`:1421-1428`), reads its own `HW_REG_XCC_ID`
(`:1430`), and builds `my_workers` as every worker whose map entry matches:
37 per XCD if placement is even. It polls two queues: its own
(`ld_local_u64`) and the broadcast queue (`ld_acquire_gpu_u64`, which on HIP
is `__atomic_load_n(..., __ATOMIC_SEQ_CST)`, `mpk_atoms.cuh:231`).

Event handling, in code order:

| Event | Source | What the scheduler does |
|---|---|---|
| index 0 (termination) | `terminate_schedulers` | push `TaskId 0` (`TASK_TERMINATE`) to each of its workers; return (`:1509-1527`) |
| `EVENT_END_OF_TASK_GRAPH` | last task of the graph; also seeded by `prepare_kernel` | print `[FWD_PASS]`; call `prepare_next_batch`; on `false` call `terminate_schedulers`, else push `compute_task_id(iteration_num + 1, 1)` to one of its workers (`:1528-1603`) |
| `EVENT_LAUNCH_DEPENDENT_TASKS` | fired once per graph by `TASK_BEGIN_TASK_GRAPH` (`runtime.cc:176-181`) | `iteration_num += 1` (`:1605`); dispatch position `first + i * 296 + j` to worker `j` for each of its workers `j` (`:1705`); if the event holds exactly `num_xcds` gang tasks, take task `first + sched_id` and broadcast it to `min(n_tile_count, 37)` of its workers (`:1616-1660`) |
| `EVENT_LAUNCH_MASSIVE_TASKS` (8 or more consumers, `runtime.cc:102-105`) | broadcast queue | split the task range across the 8 schedulers with `get_first_last_ids` (`:1756-1763`); a gang slice must be exactly one task (`:1778`), broadcast to its workers; otherwise round-robin over its own workers |
| `EVENT_LAUNCH_TASKS` (fewer than 8 consumers) | the firing worker's own-XCD queue | round-robin all consumers over its own workers |

On `iteration_num == 1` each dispatch path also fills
`xcd_event_num_tasks[my_xcd * num_events + trigger_event]`, the per-XCD
threshold used by the worker's two-level counting. The graph is identical
every iteration, so this is computed once.

### Placement rules the graph must be built against

These follow from the table above and decide which XCD a task runs on.

1. **Eight gang tasks per operator, one per XCD.** A gang event with exactly
   8 tasks is split one per scheduler. Task `first + k` runs on the XCD of
   scheduler `k`. This is the N-split Chiplet-task mechanism; the tile loop
   inside the worker then spreads `n_tile_count` tiles over the 37 workers.
2. **Small events stay on one XCD.** An `EVENT_LAUNCH_TASKS` (fewer than 8
   consumers) is handled by the scheduler on the XCD of the worker that
   completed the last producer, and all consumers land on that XCD. A 1-task
   consumer such as the merge or a norm therefore runs wherever the last
   producer finished; there is no way to pin it.
3. **Massive non-gang events are sliced contiguously.** Scheduler `k` gets
   tasks `[first + k * n / 8, first + (k + 1) * n / 8)`. Consecutive tasks
   share an XCD, in blocks of `n / 8`.
4. **The dependent-tasks event interleaves by worker id.** Only the graph's
   first fan-out uses it; task `first + m` goes to worker `m mod 296`.
5. **Scheduler `k` is assumed to sit on XCD `k`.** `get_rand_sched_id`
   returns the worker's `xcd_id` as the scheduler-queue index with the
   comment "scheduler_kernel block k runs on XCD k" (`:591`). The scheduler
   itself does not assume this: it discovers its XCD from the register and
   collects matching workers. But the worker-side enqueue writes queue
   `xcd_id` with volatile stores and no fence (`:1345-1360`), so if
   scheduler block `k` were placed on a different XCD, its own-queue reads
   (`ld_local_u64`) would be cross-XCD without a fence. Correctness rests on
   the hardware dispatching scheduler blocks 0..7 to XCDs 0..7 in order,
   which `../mi300x/02-chiplet-dispatch.md` says is the round-robin default
   but which the runtime never checks. The `[SCHED_XCD] sched_id=k xcd=k`
   line printed at `:1436` is the check; it should read `sched_id == xcd`
   on all eight lines. Logged as Q10.

### The primitives (`mpk_atoms.cuh`)

| Name | On gfx942 | Line |
|---|---|---|
| `threadfence_gpu()` | `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent")` | `:300` |
| acquire in worker | `__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent")` | `persistent_kernel.cuh:948` |
| `atom_add_release_gpu_u64` | inline asm `flat_atomic_add_x2 ... sc0 sc1`, `s_waitcnt vmcnt(0) lgkmcnt(0)`; no compiler-generated fence, relies on the preceding `threadfence_gpu()` | `:176` |
| `atom_cas_release_gpu_u64` | inline asm `flat_atomic_cmpswap_x2 ... sc0 sc1` | `:206` |
| `ld_acquire_gpu_u64` | `__atomic_load_n(..., __ATOMIC_SEQ_CST)` | `:231` |
| `ld_relaxed_gpu_u64`, `st_relaxed_gpu_u64` | relaxed atomics | `:260`, `:276` |
| `ld_local_u64`, `st_local_u64` | volatile load and store | `:321`, `:314` |
| `fence_local()` | compiler barrier `asm volatile("" ::: "memory")` | `:328` |
| `atom_add_local_u64` | plain `atomicAdd` | `:334` |

Two things this settles. First, `../mi300x/99-open-questions.md` Q4 and
`99-open-questions.md` Q6 (does the agent-scope fence emit `buffer_wbl2 sc1`
and `buffer_inv sc1`) reduce to one question about one compiler builtin:
the runtime issues no cache-control instruction by hand. Per
`../mi300x/03-memory-model.md`, LLVM's GFX942 lowering of an agent-scope
release fence is `buffer_wbl2 sc1` then `s_waitcnt`, and of an acquire fence
is `s_waitcnt` then `buffer_inv sc1`. The check is a disassembly of
`threadfence_gpu` and of the worker's acquire, nothing more. Second, the
`MPK_DISABLE_THREADFENCE` flag (`:18`, `:294`) exists for exactly the
fence-cost experiment in `../mi300x/99-open-questions.md` Q5: the same build
with the fence removed gives the upper bound on what the fences cost, at the
price of correctness.

Note the same-XCD path is not merely "device-scope atomics": worker-to-
scheduler queue writes are volatile stores plus a compiler barrier. That is
sound only because producer and consumer share one L2, which is why rule 5
above matters.

### Constants that bound our tasks

| Constant | Value | Where | Effect on us |
|---|---|---|---|
| `WORKER_NUM_THREADS` | 256 (4 waves) | `persistent_kernel.cuh:127-135` | every task is written for a 256-thread workgroup |
| `MAX_DYNAMIC_SHARED_MEMORY_SIZE` | 60 KiB minus 3 KiB reserved = 57 KiB | `runtime_header.h:35-42` | the MLA task's 32 KiB FP32 accumulator (`../mla-decode/04-our-kernel-spec.md`) leaves 25 KiB for staging; `MIN-24` is bounded by 57 KiB, not 64 KiB |
| `MAX_NUM_WORKERS` | 304 | `runtime_header.h:89` | fine for 296 |
| `MAX_WORKER_PER_SCHEDULER` | `296 // 8 + 1 = 38` | `persistent_kernel.py:163` | fine for 37 |
| worker and scheduler queue depth | 1,024 | `persistent_kernel.cuh:2027-2028` | a worker's queue holds at most 1,024 pending task ids; with ~2,135 tasks per iteration spread over 296 workers this is not a constraint |
| `TASK_DESCS_BUFFER_LENGTH` | 16 | `:702` | tasks are fetched 16 at a time |
| scheduler queue seeds | `sched_queues[0][0] = END_OF_TASK_GRAPH` | `:370-375` | iteration 1 starts from scheduler 0 via the normal end-of-graph path |

### Timing already in the runtime

Three sources, none of which we need to write:

- `[FWD_PASS] iter=N time_ms=...` from worker 0 at each
  `TASK_BEGIN_TASK_GRAPH` (`:1048`) and from the scheduler at each
  `EVENT_END_OF_TASK_GRAPH` (`:1541`, `:1547`). Per-iteration wall clock;
  the first 10 iterations and every 50th (worker) or 100th (scheduler) are
  printed.
- `MPK_ENABLE_EVENT_TIMING` (`:1299`, `:2157`): every fired event appends
  `(event_index, s_memrealtime)` to a device buffer, read back with
  `get_event_timing` (`:2412`). This gives per-operator timestamps inside
  an iteration and is the instrument for the dispatch-overhead question
  (Q5).
- `MPK_ENABLE_TIMING` and `MPK_ENABLE_DEVICE_TASK_ACCUM`: per-worker
  poll, dependency-wait, execute and signal cycle totals, and per-task-type
  accumulated nanoseconds (`:1012-1039`, `:1126-1158`).

For the design's latency metric, `[FWD_PASS]` is the per-token number and
needs no code change; median and P95 over 32 iterations come from parsing
it, or from the event-timing buffer for a cleaner series.
