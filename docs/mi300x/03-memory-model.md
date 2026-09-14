# 03 — gfx942 Memory Model, Coherence, and Synchronization

**This is the correctness foundation of the on-device task graph.** Fleet
resolves dependencies with device-side flags that producers write and consumers
spin on. If the scope or cache-policy bits are wrong, the failure is a silent
stale read or a hang — and it will present as a numerical bug, not a sync bug.

Primary source: LLVM *User Guide for AMDGPU Backend*, section "Memory Model
GFX942" (`llvm/docs/AMDGPUUsage.rst`), plus CDNA3 ISA Tables 50 and 51.

## The central fact

> "The gfx942 can be configured as a number of smaller agents with each having
> a single L2 shared by all CUs on the same agent, or as fewer (possibly one)
> larger agents with groups of CUs on each agent each sharing separate L2
> caches."

MI300X in **SPX mode is the second case**: one agent, eight XCDs, **eight
separate L2 caches**. So within our single logical device:

- Two workgroups on different XCDs do **not** share an L2.
- A write that has reached L2 is **not** thereby visible to another XCD.
- Cross-XCD visibility requires an explicit **writeback** on the producer side
  and an explicit **invalidate** on the consumer side.

The partitioning doc's "Implicit synchronization across XCDs is handled by the
hardware" describes device presentation, not cache coherence. Do not read it as
permission to skip the sequences below.

LLVM states the mechanism directly:

> "To ensure coherence of local memory writes of CUs with different L1 caches
> in the same agent a `buffer_wbl2` is required. It does nothing if the agent is
> configured to have a single L2, or will writeback dirty L2 cache lines if
> configured to have multiple L2 caches."

> "To ensure coherence of local memory reads of CUs with different L1 caches in
> the same agent a `buffer_inv sc1` is required. It does nothing if the agent is
> configured to have a single L2, or will invalidate non-local L2 cache lines if
> configured to have multiple L2 caches."

The "does nothing if ... a single L2" clause is why code developed on a
single-L2 part can appear correct and then break on MI300X in SPX.

## Cache-policy bits

The `sc0` / `sc1` bits are **not** a flat "scope" field; their meaning depends
on the instruction class.

| Instruction class | `sc0` means | `sc1` means |
|---|---|---|
| Vector load/store | L1 coherence (bypass/invalidate) | L2 / agent-vs-system scope |
| **Atomic RMW** | **returns the original value** (not coherence — atomics implicitly bypass L1) | system vs agent scope coherence |

Verbatim from LLVM:

> "Atomic read-modify-write instructions implicitly bypass the L1 cache.
> Therefore, they do not use the sc0 bit for coherence and instead use it to
> indicate if the instruction returns the original value being updated. They do
> use sc1 to indicate system or agent scope coherence."

This matters for our flag protocol: a returning atomic (`atomicAdd` with a used
result, `atomicCAS`) has `sc0` set for the return value, and that is orthogonal
to scope.

### `BUFFER_WBL2` (CDNA3 ISA Table 50)

| SC1 | SC0 | L2 cache behavior |
|---|---|---|
| 0 | any | NOP |
| 1 | 0 | (1 L2 cache): NOP; (>1 L2 cache) Write-back dirty data |
| 1 | 1 | Write back dirty data |

### `BUFFER_INV` (CDNA3 ISA Table 51)

| SC1 | SC0 | CU cache behavior | L2 cache behavior |
|---|---|---|---|
| 0 | 0 | NOP | NOP |
| 0 | 1 | (if TG Split) Invalidate cache; (if not) NOP | NOP |
| 1 | 0 | Invalidate cache | (1 L2): NOP; (>1 L2) Invalidate non-coherently cached lines |
| 1 | 1 | Invalidate cache | Invalidate non-coherently cached lines |

Note `sc1` alone (`1 0`) already invalidates the CU cache **and** conditionally
the L2 — it is the agent-scope acquire on a multi-L2 configuration.

## Code sequences we must reproduce

These are what LLVM emits; a hand-written flag protocol must match them.

### Agent-scope acquire — `load atomic acquire`, global

```
1. buffer/global_load sc1=1
2. s_waitcnt vmcnt(0)      // load must complete before the invalidate
3. buffer_inv sc1=1        // so following loads will not see stale global data
```

### Agent-scope acquire — `fence acquire`

```
1. s_waitcnt lgkmcnt(0) & vmcnt(0)   // after the fence-paired-atomic
2. buffer_inv sc1=1
```

### Agent-scope release — `store atomic release`, global

```
1. buffer_wbl2 sc1=1   // "Performs L2 writeback to ensure previous
                       //  global/generic store/atomicrmw are visible at agent scope."
2. s_waitcnt lgkmcnt(0) & vmcnt(0)
3. (the store atomic itself)
```

### Agent-scope release — `fence release`

```
1. buffer_wbl2 sc1=1
2. s_waitcnt lgkmcnt(0) & vmcnt(0)
```

### System-scope acquire (for contrast)

```
1. buffer/global/flat_load sc0=1 sc1=1
2. s_waitcnt vmcnt(0)
3. buffer_inv sc0=1 sc1=1
```

System scope additionally invalidates the L1 (`sc0`). We do not need system
scope for intra-GPU task-graph flags; using it only costs extra invalidation.

**Note the ordering asymmetry:** on release the writeback comes *first*, before
the `s_waitcnt`; on acquire the invalidate comes *last*, after it. Getting this
backwards produces a race that is rare and load-dependent.

## Other model facts worth holding

- **L1 is per CU, shared by all SIMDs on it.** No action needed between lanes of
  a wave or between waves of a workgroup (except under tgsplit). A `buffer_inv
  sc0` is required for coherence between wavefronts in *different* workgroups,
  since they may be on different CUs.
- **LDS ops involve no caching** and complete in execution order per wave, but
  are reordered *between* waves of a workgroup — `s_waitcnt lgkmcnt(0)` is
  required to order LDS against vector memory across waves of a workgroup.
- **`s_waitcnt vmcnt(0)` is the cross-CU ordering primitive** for vector memory:
  "A `s_waitcnt vmcnt(0)` is required to ensure synchronization between vector
  memory operations of different CUs."
- **The L2 has independent channels serving disjoint virtual address ranges**,
  and each CU has a per-channel request queue, so operations from different CUs
  to the same L2 can be reordered relative to each other.
- **Scalar and vector L1 caches are not coherent.** Scalar memory is only used
  for data proven not to change during a dispatch. A persistent megakernel
  running for the whole decode step stretches "during the execution of the
  kernel dispatch" considerably — never let mutable task-graph state be read
  through scalar loads. Both caches are invalidated by CP only *between* kernel
  dispatches, which for us means *once*, not per layer.
- **MTYPE** governs automatic coherence: RW for memory local to the L2, NC with
  the PTE C-bit for non-local; UC bypasses L2 for PCIe access to the host.
  Kernarg backing memory on a dGPU is MTYPE UC.

## Practical rules for the Fleet task graph

1. All task-graph flags, counters, and barriers are **agent scope**, not system
   scope and not workgroup scope.
2. Producer completing a task: `buffer_wbl2 sc1` (or an agent-scope release
   fence), then `s_waitcnt`, then publish the flag with a release atomic.
3. Consumer waiting on a task: acquire-load the flag, `s_waitcnt vmcnt(0)`,
   `buffer_inv sc1`, then read the payload.
4. In HIP, this is what `__hip_atomic_load/store(..., __ATOMIC_ACQUIRE/RELEASE,
   __HIP_MEMORY_SCOPE_AGENT)` should generate. **Verify by disassembly**
   (`llvm-objdump`), do not assume — confirming that `buffer_inv sc1` /
   `buffer_wbl2 sc1` actually appear is a cheap, high-value check and belongs in
   the first hour of GPU access.
5. A plain `__threadfence()` maps to an agent-scope fence and should be
   sufficient, but it is worth confirming it emits the writeback, since a
   missing `buffer_wbl2` is invisible until it corrupts a result.
6. Allocate task-graph metadata as **device memory**, not fine-grained host-
   visible memory, unless we have measured the difference — MTYPE affects which
   sequences are needed.

## Open items

- Does HIP's agent-scope atomic actually emit `buffer_inv sc1` / `buffer_wbl2
  sc1` on our compiler version? → `99-open-questions.md` Q4
- Cost of `buffer_inv sc1` / `buffer_wbl2 sc1` per invocation, and therefore how
  coarsely we should batch dependency resolution → `99-open-questions.md` Q5
