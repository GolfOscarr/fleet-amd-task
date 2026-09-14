# 04 — Persistent Kernel Mechanics

Fleet is "a persistent kernel device-side runtime with per-chiplet scheduling.
One workgroup per chiplet is designated as a *scheduler*; the rest are
*workers*." This file collects what the hardware gives us to build that.

## Waiting: sleep, wake, and their limits

### `S_SLEEP` (CDNA3 ISA)

The ISA summary table says "Causes the wavefront to sleep for 64 - 8128 clock
cycles." The instruction page is more precise:

> "Cause a wave to sleep for up to ~8000 clocks. The wave sleeps for
> `(64*(SIMM16[6:0]-1) .. 64*SIMM16[6:0])` clocks. The exact amount of delay is
> approximate. ... When `SIMM16[6:0]` is zero then no sleep occurs."

So the immediate is **7 bits, 0..127**, giving a maximum of 64 × 127 = 8128
clocks, and the delay is a *range*, not a fixed value — `s_sleep 1` waits 1–64
clocks, `s_sleep 2` waits 65–128. Exposed in HIP as
`__builtin_amdgcn_s_sleep(N)` with N an immediate.

Because the delay is approximate and coarse (64-clock granularity), a poll loop
cannot use `s_sleep` for precise timing — only for backoff.

This is the right primitive for a spin loop: it keeps a polling wave from
saturating the instruction issue and the memory pipe while it waits.

### `S_WAKEUP` — and the constraint that shapes our design

> "Allow a wave to 'ping' all the other waves in its threadgroup to force them
> to wake up early from an `S_SLEEP` instruction. ... This allows for efficient
> polling on a memory location. The waves which are polling can sit in a long
> `S_SLEEP` between memory reads, but the wave which writes the value can tell
> them all to wake up early now that the data is available. This method is also
> safe from races since any waves that miss the ping resume when they complete
> their `S_SLEEP`."

The ISA describes exactly our use case. But:

> "If the wave executing `S_WAKEUP` is in a threadgroup (in_tg set), then it
> wakes up all waves associated with the same threadgroup ID. Otherwise,
> `S_WAKEUP` is treated as an `S_NOP`."

**`S_WAKEUP` does not cross workgroup boundaries.** A producer workgroup on one
XCD cannot ping a consumer workgroup on another. Consequences:

- **Intra-workgroup** waiting (worker waves waiting on their own scheduler wave,
  or on each other): use `S_SLEEP` + `S_WAKEUP`. Cheap and prompt.
- **Inter-workgroup / cross-XCD** waiting (the actual task-graph dependency
  edges): must be **polling** — `S_SLEEP(N)` in a loop with an agent-scope
  acquire load of the flag, per `03-memory-model.md`.

So the sleep interval `N` is a real tuning parameter with a real cost: too
short and every poll costs a `buffer_inv sc1` and L2 traffic; too long and we
add latency to every dependency edge in the graph. Because the whole point of
this exercise is per-token latency in the tens of milliseconds or less, this
parameter deserves a sweep, and the poll must be as cheap as possible — poll a
single flag with one lane, then broadcast within the workgroup, rather than
having 64 lanes each hammer the same cache line.

## On-device timing

| Instruction | Purpose |
|---|---|
| `S_MEMTIME` | Returns a clock counter |
| `S_MEMREALTIME` | Returns a constant-rate real-time counter |

`S_MEMREALTIME` is the one to use for in-kernel measurement, since it does not
vary with engine clock. Both are documented in CDNA3 ISA §8.2.4–8.2.5; the
destination SGPR must be even for a two-Dword fetch. Exposed as
`__builtin_amdgcn_s_memrealtime()`.

This matters because a megakernel does not produce per-operation kernel traces —
the profiler sees one long kernel. Instrumenting the task graph with
`S_MEMREALTIME` timestamps written to a device-side ring buffer is how we get a
per-task breakdown at all. Build this in early; it is also our evidence for the
"GPU launches" and per-layer latency metrics the task asks for.

## Occupancy

The limits that bound how many workers stay resident:

| Resource | Budget | Source |
|---|---|---|
| VGPRs per wave | up to 512 total (≤256 arch + ≤256 Acc, flexible split) | CDNA3 ISA §3.6.4 |
| VGPR allocation granularity | groups of 8 Dwords | CDNA3 ISA §3.6.4 |
| LDS per CU | 64 kB | CDNA3 ISA §3.6 |
| LDS allocation granularity | 512 B, 512 B-aligned | CDNA3 ISA |
| Max LDS per workgroup | 64 kB | CDNA3 ISA |
| SIMDs per CU | 4 (derived from `HW_ID.SIMD_ID` being 2 bits) | CDNA3 ISA Table 6 |
| CUs | 304 (38 × 8) | ROCm |

A wave using the full 512 VGPRs occupies the entire per-SIMD register budget,
giving 1 wave/SIMD = 4 waves/CU. Halving to 256 VGPRs gives 2 waves/SIMD, and
so on. The exact per-SIMD physical register file size on CDNA3 is **not**
stated in the pages read so far — the "512 VGPRs per wave" figure is an
architectural maximum per wave, not the file size — so the wave-slot arithmetic
should be confirmed with the occupancy API rather than assumed. See
`99-open-questions.md` Q6.

**This is less alarming than it sounds.** `VMCNT` is 6 bits, so a single wave
may have up to 63 outstanding vector loads — memory-level parallelism inside the
wave substitutes for the thread-level parallelism lost to 1 wave/SIMD. The
required prefetch depth is only 4-8. Full analysis in
`07-achievable-bandwidth.md`.

Query occupancy directly rather than deriving it:

```c
hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, threads, dynLDS);
```

and cross-check against the compiler's own resource report:

```
hipcc --offload-arch=gfx942 -Rpass-analysis=kernel-resource-usage ...
```

which prints VGPR/AGPR/SGPR/LDS/occupancy per kernel at compile time. For a
megakernel this is essential: the register footprint is the **union** of every
task's footprint unless we scope things carefully, so a single register-hungry
expert GEMM can silently halve occupancy for the entire graph. Watch this number
from the first commit.

### Launch bounds

`__launch_bounds__(threadsPerBlock, minBlocksPerCU)` constrains the register
allocator. For a persistent kernel we want to *pin* occupancy deliberately —
enough workers to saturate memory, not so many that registers spill.

## Whole-device residency

Fleet requires all workers co-resident: a task graph whose consumers have not
been scheduled yet deadlocks, because a producer workgroup will wait forever on
a consumer that the hardware never dispatched.

Two ways to get that guarantee:

1. **Cooperative launch** — `hipLaunchCooperativeKernel` guarantees co-residency
   and provides a device-wide grid barrier (`cg::grid_group::sync()`). Note a
   known ROCm issue reporting `hipLaunchCooperativeKernel` slowdown
   (ROCm/ROCm#3410) — measure the launch overhead itself, since for batch-1
   decode a per-token launch cost of even tens of microseconds is significant.
2. **Sized-to-fit ordinary launch** — launch exactly
   `304 × blocks_per_CU` workgroups as computed by the occupancy API. Requires no
   cooperative support, but correctness now depends on our occupancy arithmetic
   being right; if it is off by one block, the last workgroup queues behind the
   others and the graph deadlocks.

Recommendation: **start with cooperative launch** for the guarantee, measure its
overhead, and keep the sized-to-fit path as a fallback if the launch cost proves
material. Either way, assert at startup that `gridDim` matches what we expect to
be resident, and include a watchdog (a bounded poll count that aborts with a
diagnostic) so a dependency bug surfaces as an error rather than a hung GPU.

## Structure implied for our kernel

```
persistent_kernel:
  xcd   = XCC_ID                        // hwreg 20, see 02-chiplet-dispatch.md
  role  = (first workgroup on this xcd) ? SCHEDULER : WORKER
  loop:
    scheduler: pick ready tasks from this chiplet's queue, publish to workers
    worker:    wait for task  (poll + s_sleep; s_wakeup only within workgroup)
               execute task
               agent-scope release: buffer_wbl2 sc1 -> s_waitcnt -> set flag
    until end-of-token signal
```

## Open items

- Cooperative-launch overhead per token → `99-open-questions.md` Q7
- Optimal `s_sleep` interval for dependency polling → `99-open-questions.md` Q8
- Real wave slots per SIMD at our register footprint → `99-open-questions.md` Q6
