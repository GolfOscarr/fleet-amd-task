# 03 - Synchronization strategy

What orders one task's writes before another task's reads, on a device whose
eight L2 caches are not coherent with each other. The mechanism is the
runtime's, read in `docs/fleet/03-runtime.md`; the instruction-level facts are
from the CDNA3 ISA and LLVM's GFX942 memory model in
`docs/mi300x/03-memory-model.md`. Nothing here is new machinery: the design
uses the runtime's protocol unchanged and states exactly what it costs and
where it is verified.

## The central fact

In SPX mode one HIP agent spans eight XCDs with eight private 4 MB L2 caches.
A store from a CU on XCD 3 sits dirty in L2 number 3; a load from XCD 5 can
hit a stale line in L2 number 5. Visibility across XCDs requires the producer
to write back (`buffer_wbl2 sc1`) and the consumer to invalidate
(`buffer_inv sc1`), both agent-scope operations that the ISA defines as
no-ops on parts with a single L2 (`docs/mi300x/03-memory-model.md`).

## The protocol, as the runtime implements it

Every task boundary in our graph is a producer-consumer pair, and after the
prelaunch rewrite every consumer waits in its own worker on a counter
(`docs/fleet/03-runtime.md`, "How the graph is actually dispatched"). The
sequence for one event with producer tasks on several XCDs:

```
producer task (any worker, XCD x)
  1. task body: plain stores to its output slice
  2. atom_add_local(xcd_local_counter[x][e], tiles_done)          device-scope atomic, resolves in L2 x
  3. if this brought the local count to threshold[x][e]:
       threadfence_gpu()  ==  __builtin_amdgcn_fence(RELEASE, "agent")
                          ->  buffer_wbl2 sc1 ; s_waitcnt vmcnt(0) lgkmcnt(0)   (write back L2 x)
       flat_atomic_add_x2 all_event_counters[e], 1  (sc0 sc1)                     (global counter)

consumer task (worker on XCD y), before its body
  4. actual = __atomic_load_n(all_event_counters[e], RELAXED)      expected to lower to an sc1 load; verified by disassembly (below)
  5. __builtin_amdgcn_fence(ACQUIRE, "agent")
                          ->  s_waitcnt ; buffer_inv sc1                         (invalidate L2 y)
  6. if actual < needed: poll with s_sleep 1, then fence again
  7. task body: plain loads of the producer's output
```

Steps 2-3 are `persistent_kernel.cuh:1215-1275`; steps 4-6 are `:914-975`;
the fence definitions are `mpk_atoms.cuh:300` and the acquire builtin at
`persistent_kernel.cuh:948`. `needed = num_triggers x iteration_num` because
counters are never reset within a launch.

Within one XCD the protocol degenerates correctly: the local counter is a
device-scope atomic in the shared L2 and the tile outputs of one gang task
are visible to the same XCD without any fence. That path is used by the
worker-to-scheduler queues and by the tiles of a gang task; it is never used
between two of our operators, because every operator's consumers are spread
over all XCDs by the placement rule.

## Every edge in our graph, classified

| Edge class | Instances per iteration | Producer side | Consumer side |
|---|---|---|---|
| gang op (8 tasks) -> any op | 27 x 4 + 2 + 26 x 2 + 1 = 163 | 8 release flushes (one per XCD, after its last tile) + 8 global atomics | one acquire per consumer task |
| 1-task op -> any op | 27 x 3 + 1 + 26 + 1 = 109 | 1 release flush + 1 atomic | one acquire per consumer task |
| 8-task non-gang op (`moe_silu_mul`, `moe_mul_sum_add`) -> next | 26 x 2 = 52 | 8 flushes + 8 atomics | one acquire per consumer task |
| `argmax_partial` (50 tasks over 8 XCDs) -> reduce | 1 | 8 flushes + 8 atomics | 1 acquire |
| `argmax_reduce` -> next iteration's `embed` | 1 | 1 flush + end-of-graph event | **no acquire** (first op); see below |
| `tokens[step]` -> `embed` | 1 | plain store by the reduce, written back by its flush | agent-scope load in the embed variant |

Totals per iteration, from `sources/graph_counts.py`: **1,838 release
flushes** and **1,879 acquires**. Against the repo's note that two-level
counting "reduces `buffer_wbl2` from ~24K to ~300" for Qwen3-8B, ours is
higher because our operators have one gang task per XCD: the local threshold
is met by the XCD's single task, so every XCD flushes once per operator and
the batching has nothing to batch. That is a structural property of an
8-task-per-op graph, not a misconfiguration; 03 records it so the
measurement in `09-expected-performance.md` has a prediction to compare
against.

## What the acquire does to the L2

The ISA is explicit: on a part with more than one L2, `buffer_inv sc1`
"invalidates non-coherently cached lines" in the issuing XCD's L2
(`docs/mi300x/03-memory-model.md`, Table 51). Every task except the
iteration's first executes one at its start, so each XCD's L2 is invalidated
about 235 times per iteration, once per task it runs, at every operator
boundary. Two consequences the design states rather than discovers later:

1. **Nothing survives in L2 across an operator boundary** unless it is cached
   under an MTYPE the invalidate does not touch. Which MTYPE plain device
   memory carries in SPX + NPS1 is not documented in the sources we read
   (`docs/mi300x/03-memory-model.md`, "MTYPE governs automatic coherence"),
   so the conservative assumption is that it does not survive. The
   "latent KV cache resident in L2" idea of `docs/deepseek-v2-lite/07-roofline.md`
   is therefore **not expected to pay under this runtime**; if any residency
   helps the cache read, it is the memory-side Infinity Cache (256 MB,
   `secondary`, `docs/mi300x/01-architecture.md`), which needs no
   invalidation because it sits in front of HBM for all XCDs. Non-temporal
   weight loads (`USE_NT_WEIGHTS=1`, MALL no-allocate) are then the
   experiment for that layer, not for L2. `OPEN-PROBLEMS.md` MAJ-6 is
   re-scoped accordingly.
2. The invalidate is cheap when the L2 holds little dirty or reusable data,
   which at batch 1 is the normal state: weights are read once, activations
   are kilobytes. Its cost is a latency term at each boundary, not a
   bandwidth term. It is measured, not modelled (`09-expected-performance.md`).

## The one edge outside the event mechanism

The first operator of an iteration (`embed`) is dispatched by the begin-graph
event and has no `dependent_event`, so it runs no acquire. It reads
`tokens[step]`, written by the previous iteration's `argmax_reduce` on some
XCD. The reduce's release flush wrote the value back to HBM, but the
embed's XCD could hold a stale copy of that cache line from its own read of
the neighbouring token in the previous iteration (the reads of 16 consecutive
tokens share one 128-byte line, and the embed runs on the same XCD every
iteration: position 2 maps to XCD 0, on whichever of its workers the
round-robin counter names). In the shipped runtime this is masked by the hundreds of
acquires other tasks execute on that XCD in between. The design does not
rely on that: the embed variant loads the token id with an agent-scope
(`sc1`) load, which is not served from a non-coherent line. One instruction.

The scheduler's own read of `tokens[step + 1]` for the EOS check is a plain
load on the scheduler's XCD; with `eos = -1` its value is irrelevant.

## Assumptions the protocol rests on, and their checks

| Assumption | Where it lives | Check (GPU day 1) |
|---|---|---|
| `__builtin_amdgcn_fence(RELEASE, "agent")` emits `buffer_wbl2 sc1` + `s_waitcnt`, and the ACQUIRE form emits `s_waitcnt` + `buffer_inv sc1` on our ROCm | `mpk_atoms.cuh:300`, `persistent_kernel.cuh:948` | `llvm-objdump -d` of the compiled megakernel; grep both mnemonics near `threadfence_gpu` and the dependency check. `OPEN-PROBLEMS.md` MAJ-3 |
| `__atomic_load_n` on the global counter is an `sc1` load (agent or system scope), so polling is not served from a stale line | `persistent_kernel.cuh:945-960` | same disassembly: the poll's `global_load` carries `sc1` |
| scheduler block `k` sits on XCD `k`, so worker-to-scheduler queues are same-L2 | `persistent_kernel.cuh:591` | `[SCHED_XCD] sched_id=k xcd=k` on all eight lines; `[WORKER_XCD]` shows `xcd == worker mod 8`. `OPEN-PROBLEMS.md` MIN-25 |
| tile outputs of one gang task are disjoint, so no ordering is needed among a task's tiles | kernel design | by construction: `tile_idx` selects disjoint columns, splits, heads, or experts |
| counters never overflow within a launch | `EventCounter` is 64-bit; `needed = triggers x iteration_num` | 32 iterations x at most 50 triggers |
| the cache rows written by `mla_prep` in layer `l` are visible to `mla_attend` in layer `l` of the same iteration and all later iterations | one release-acquire pair between them in the chain | boundary B3/B4 comparisons at every layer (`07-correctness.md`) |

## What we deliberately do not do

- No hand-written `buffer_wbl2` / `buffer_inv` in our kernels. Our tasks
  produce and consume through the runtime's counters like every other task;
  adding a second mechanism would double the surface to verify.
- No non-temporal stores for flags. The runtime's `mpk_atoms.cuh` notes that
  NT bypasses cache but does not order; we keep flags on the atomic path.
- No workgroup-scope or wave-scope tricks across tasks. Scope is agent,
  always, for task-graph state.

## Measurements that belong to this section

| Quantity | How |
|---|---|
| cost of a release flush and of an acquire, in isolation | `MPK_DISABLE_THREADFENCE` build (incorrect results, timing only) versus the normal build on the layer-1 graph: the difference over 1,838 + 1,879 fences is the per-iteration fence cost |
| per-boundary latency | `MPK_ENABLE_EVENT_TIMING`: `s_memrealtime` at each event firing gives the gap between consecutive events with no work in between (the two norm-to-gang boundaries are near-empty) |
| whether L2 residency exists at all | `TCC_HIT` / `TCC_MISS` counters over the layer-1 graph with and without `USE_NT_WEIGHTS=1`, compared to the 1.125 MiB cache read per layer (`docs/mi300x/06-profiling.md`) |
