# 99 — Open Questions

Every claim our design depends on that is **not** confirmed by a primary source,
with the specific check that settles it. Anything here should be resolved in the
first hours of GPU access, before it can poison a design decision.

Status: `open` | `resolved` | `blocked`. Record the answer inline with the date
and the command that produced it.

---

## Q1 — What is the actual workgroup→XCD mapping? `resolved` (2026-09-15)

**Resolved: round-robin at workgroup granularity, with a constant offset.**
Probe `env/hw/probes/xcc_map.cu` over 21 launches at grid sizes 296, 8, 304,
608, 1000 and 37, each run alone and concurrently with a second process
(`env/hw/20260915/`, VM `enc1-gpuvm005`, ROCm 7.2.4). Every block of every
launch satisfied

```
xcd == (blockIdx.x + 4) mod 8
```

and the per-XCD counts were exactly balanced: 37 blocks each at grid 296, one
each at grid 8, 38 each at grid 304, and 5,4,4,4,5,5,5,5 at grid 37. The
mapping was stable across launches and across processes. The offset of 4 is
constant within a session but is not a constant to build on — it is
presumably fixed at boot or by the virtual function — so code must read
`HW_REG_XCC_ID` rather than derive the XCD from `blockIdx`. That is the same
finding that makes `../fleet/99-open-questions.md` Q10 false on this VM. The
original question follows.

**Why it matters.** Chiplet-task placement is the whole point of Fleet. If we
place tasks by assuming `blockIdx.x % 8`, and the real policy differs, we get
correct results with none of the locality benefit — the worst failure mode,
because it looks like the idea does not work.

**What is documented.** Only "Workgroups are automatically distributed across
all XCDs (round-robin)" for SPX. Granularity, starting XCD, and interaction with
launch order are not documented.

**Check.** Launch a grid of N workgroups, each writing its `XCC_ID` (hwreg 20)
and `blockIdx` to a buffer. Dump and tabulate. Repeat for several grid sizes
including exactly 304 blocks, 2×304, and a non-multiple; repeat across runs to
check stability.

---

## Q2 — Assembler syntax for reading `XCC_ID` `resolved` (2026-09-13)

**Why it matters.** Everything in Q1 and all self-organization depends on it.

**What is documented.** ISA Table 7 and the hwreg table: register code 20,
field `XCC_ID` bits 3:0. The `S_GETREG_B32` encoding is also primary:
`SIMM16 = {size[4:0], offset[4:0], hwRegId[5:0]}`. So only the assembler's
accepted spelling is genuinely open.

**Resolved.** Fleet's own gfx942 source uses the symbolic name, so the ROCm
assembler accepts it:

```c
asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd_id));
```

(`persistent_kernel.cuh:186`.) Note they read 16 bits where the ISA defines
`XCC_ID` as bits 3:0 — harmless if the upper bits read zero.

**Residual check on the machine:** confirm returned values land in 0..7 and that
all eight appear across a large grid. Folded into Q1.

**Residual check done (2026-09-15).** `env/hw/probes/xcc_map.cu` reads the
register exactly this way; every value returned was in 0..7 and all eight
appeared on every launch of 8 blocks or more. The upper 12 bits read zero.

---

## Q3 — Partition query/set commands `resolved` (2026-09-15)

**Resolved: the working query is `amd-smi static --partition`, and the VM is
SPX + NPS1.** Several candidate spellings were tried on VM `enc1-gpuvm005`;
which ones failed and which one printed the modes is recorded in
`env/hw/20260915/raw/partition-cmd.txt`, and that query's output is beside
it in `raw/partition.txt`. Both devices report `ACCELERATOR_PARTITION: SPX`
and `MEMORY_PARTITION: NPS1`, which is what every measurement in that
directory was taken under and what the design assumes. The set syntax
was not exercised: the VM was already in the mode we want, and a virtual
function is not the place to repartition. Recording the mode in the startup
assertion still stands as a work item for the runtime path.

**Why it matters.** SPX vs CPX silently changes dispatch, coherence behaviour,
and available memory. A reboot restores SPX; a prior user may have left it in
CPX. Every measurement must record the mode.

**What is documented.** The partitioning overview gives **no command lines** —
only that `amd-smi` can adjust modes at runtime.

**Check.** Find the working query and set syntax (`amd-smi`, and whether
`rocm-smi --showcomputepartition` / `--setcomputepartition` still work). Add an
assertion to our startup path.

---

## Q4 — Does HIP's agent-scope atomic emit the right cache ops? `resolved` (2026-09-15)

**Resolved: yes, and nothing more.** `env/hw/probes/fence_probe.cu` compiled
for gfx942 with the VM's hipcc 7.2.53211 and disassembled
(`env/hw/20260915/`):

| Construct | gfx942 lowering |
|---|---|
| agent-scope release fence | `buffer_wbl2 sc1` |
| agent-scope acquire fence | `buffer_inv sc1` |
| `__threadfence()` | `buffer_wbl2 sc1` then `buffer_inv sc1`, agent scope |
| agent-scope atomic load | `sc1` load followed by `buffer_inv sc1` |

Both required instructions appear, in the required order. No `sc0 sc1`
appears in any of the four kernels, so none of these constructs silently
takes system scope; the system-scope sites seen in the offline build come
from the printf and assert hostcall paths instead (`OPEN-PROBLEMS.md`
MAJ-3). The same result settles `../fleet/99-open-questions.md` Q6. The
follow-up below, a cross-XCD producer/consumer stress test on the real task
path, is still worth running. The original question follows.

**Why it matters.** This is the single highest-risk correctness item in the
project. Per `03-memory-model.md`, cross-XCD visibility needs `buffer_wbl2 sc1`
on release and `buffer_inv sc1` on acquire. If HIP does not emit them, our task
graph reads stale data intermittently.

**Check.** Write a tiny kernel using
`__hip_atomic_store(..., __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT)` and the
matching acquire load, compile for gfx942, and `llvm-objdump -d` the result.
Confirm `buffer_wbl2 sc1` and `buffer_inv sc1` appear in the expected order
(writeback *before* the waitcnt on release; invalidate *after* it on acquire).
Do the same for `__threadfence()`.

**Follow-up.** Also write a cross-XCD producer/consumer stress test that would
fail on stale reads, and run it long enough to trust it.

---

## Q5 — Cost of `buffer_inv sc1` / `buffer_wbl2 sc1` `resolved` (2026-09-15)

**Resolved: a few hundred nanoseconds per fence, uncontended or not.**
`env/hw/probes/fence_probe.cu` times a store-fence-load loop at agent scope
against the identical loop at workgroup scope; the difference is the
cache-operation cost, per iteration (`env/hw/20260915/`):

| Case | Workgroup scope | Agent scope | Difference |
|---|---|---|---|
| One workgroup, release | 410 ns | 525 ns | 115 ns |
| One workgroup, acquire | 756 ns | 894 ns | 137 ns |
| Eight XCDs, release | 829 ns | 798 ns | none measurable |
| Eight XCDs, acquire | 673 ns | 991 ns | 317 ns |

Contention across all eight XCDs leaves release unchanged and roughly
doubles the acquire penalty, to 317 ns. So a release plus acquire pair costs
at most a few hundred nanoseconds of cache work, and per-operator cross-XCD
dependencies are affordable. Per-tensor dependencies at this granularity are
not obviously affordable, and the boundary count remains the term that
matters (`../design-doc/99-open-questions.md` DQ1). This settles
`OPEN-PROBLEMS.md` MIN-16. The original question follows.

**Why it matters.** Sets the granularity of the task graph. If an agent-scope
acquire costs a few hundred nanoseconds, fine-grained per-tensor dependencies
across XCDs are unaffordable and we must batch dependency resolution.

**Check.** Microbenchmark a loop of agent-scope acquire/release pairs against
the same loop with workgroup-scope ops; difference is the cache-op cost. Measure
both uncontended and with 8 XCDs participating.

---

## Q6 — Real wave slots per SIMD at our register footprint `resolved, for our footprint` (2026-09-15)

**Resolved: one wave per SIMD, and LDS is what binds, not registers.**
`env/hw/probes/occupancy.cu` with a 184-VGPR, 256-thread kernel matching the
worker kernel's footprint (`env/hw/20260915/`):

| Limit applied | Blocks per CU |
|---|---|
| registers only | 2 |
| registers plus the worker kernel's 58,368 B dynamic LDS request | 1 |

A 304-block grid was confirmed co-resident on the 304 CUs, so the
cooperative launch the design needs does not deadlock at this footprint. One
256-thread block per CU is four waves over four SIMDs, that is one wave per
SIMD, which is what `07-achievable-bandwidth.md` already assumes. The
question the sources could not answer, the per-SIMD physical register file
size, stays open but is moot for us: the LDS request binds first, and it
would take a much smaller LDS request before registers became the limit. The
original question follows.

**Why it matters.** Determines worker count and whether the whole grid is
co-resident (a non-co-resident graph deadlocks).

**What is documented.** "A wave may have up to 512 total VGPRs, 256 of each
type." That is a per-wave architectural maximum; the **per-SIMD physical
register file size is not stated** in the sources read, so waves-per-SIMD cannot
be derived from it. Secondary sources claim 128 KiB vector registers per SIMD —
unconfirmed.

**Check.** `hipOccupancyMaxActiveBlocksPerMultiprocessor` on the real kernel,
plus `-Rpass-analysis=kernel-resource-usage` at compile time. Cross-check by
launching a grid larger than the computed residency and confirming a
deliberately-planted deadlock watchdog fires.

---

## Q7 — Cooperative launch overhead `open`

**Why it matters.** For batch-1 decode, per-token launch cost is directly part
of TPOT. There is a known ROCm issue (ROCm/ROCm#3410) reporting
`hipLaunchCooperativeKernel` slowdown.

**Check.** Time an empty cooperative kernel vs an empty ordinary kernel at our
grid size, 1000× each; report median and P95.

---

## Q8 — Optimal `s_sleep` interval for dependency polling `open`

**Why it matters.** `S_WAKEUP` does not cross workgroups, so every cross-XCD
dependency edge is resolved by polling. The interval trades poll traffic against
added latency per edge, and there are many edges per token.

**Check.** Sweep the immediate in a producer/consumer microbenchmark; measure
wake-up latency and the L2/EA traffic generated by polling.

---

## Q9 — ROCm version and library availability on the machine `resolved` (2026-09-15)

**Resolved and committed** — the raw `rocminfo`, `hipcc --version` and
`amd-smi static` output is in `env/hw/20260915/`, summarised in that
directory's `summary.md`.

The Hot Aisle VM `enc1-gpuvm005` runs ROCm 7.2.4 with hipcc 7.2.53211 and
presents two AMD Instinct MI300X VF devices; device 0 carried every
measurement. The device reports 304 CUs over 8 XCDs, 4 SIMDs per CU,
wavefront 64, 32 waves per CU, 64 KiB LDS, 4 MiB L2 per XCD and 192 GB of
HBM. `rocminfo` lists no L3 row, which is the negative half of Q13. The
wall-clock rate is 100,000 kHz, that is 100 MHz, and `s_memrealtime` ticks
agree with the host clock to within 2%, so the probes' timings are sound.

Clocks and limits: memory 1,300 MHz maximum against 901 MHz idle, engine
2,100 MHz maximum, 144 W and 46 C at idle. `amd-smi` 25.x has no
`--throttle` flag, so throttle state has to be inferred from the clock and
power readings rather than queried.

Host and environment: Xeon Platinum 8470, 26 cores, 440 GiB RAM, 12 TB free;
Docker with GPU passthrough works, with `rocm/dev-ubuntu-24.04:7.2` seeing
gfx942 inside the container. The VM was usable about two minutes after
provisioning and pulled the 30 GB checkpoint in 79 s. The
`torch.version.hip` half of the check is captured with the Fleet build
rather than here.

---

## Q10 — Is AITER's gfx942 MLA decode usable as an oracle? `open`

**Why it matters.** Affects both our correctness reference and which fallbacks
we can lean on while the Fleet path is incomplete.

**What is known.** Secondary reports say AITER's newest MLA work targets gfx950
rather than gfx942 and that gfx942 coverage is uneven; there is an open issue
about a gfx942 persistent MLA decode kernel faulting at `page_size=1`.

**Check.** Install AITER, run its MLA decode on DeepSeek-V2-Lite shapes, compare
against HF Transformers.

---

## Q11 — Counter names and TCC instance count `resolved` (2026-09-15)

**Resolved: `rocprofv3 --list-avail` lists every counter, PMC collection
works inside the virtual function, and there are 128 TCC instances** — 16
channels per XCC across the 8 XCCs (`env/hw/20260915/`). Collection inside a
VF was the real risk, since some counters are host-only on partitioned
parts, and it is not a problem here. The names used in `06-profiling.md`
were confirmed against the listing before the copy-kernel validation of Q12
was run. The original question follows.

**Why it matters.** The TCC counter names in `06-profiling.md` came through a
summarizer, not verbatim documentation, and the MI200→MI300 prefix change
(`TCC_EA_*` → `TCC_EA0_*`) means a wrong name collects nothing rather than
erroring loudly.

**Check.** `rocprofv3 --list-avail`, and confirm the number of TCC instances.

---

## Q12 — Validate bytes-from-requests arithmetic `resolved` (2026-09-15)

**Resolved: the decomposition is exact, and it shows the read formula in
`06-profiling.md` is wrong.** A 1 GiB copy (`env/hw/probes/copy_bytes.cu`)
under `rocprofv3`, counting the `copy_kernel` rows only
(`env/hw/20260915/`):

```
reads  = 128 * TCC_BUBBLE
       + 64 * (TCC_RDREQ - TCC_BUBBLE - TCC_RDREQ_32B)
       + 32 * TCC_RDREQ_32B
writes = 64 * TCC_WRREQ_64B
       + 32 * (TCC_WRREQ - TCC_WRREQ_64B)
```

gives exactly 1.0000 GiB of reads and exactly 1.000 GiB of writes against a
known 1 GiB each way. The raw counts were `TCC_BUBBLE` 8,388,608, `TCC_RDREQ`
8,388,760, `TCC_RDREQ_32B` 0, `TCC_WRREQ` 16,777,216, all of them 64 B.

**Defect this exposes.** `06-profiling.md` charges a flat 64 B per read
request and has no 128 B term, so on this traffic, which is almost entirely
128 B requests, it undercounts reads by a factor of two. That file is not
edited here; the correction is logged as a work item in `OPEN-PROBLEMS.md`.
Use the formula above, not the one in `06-profiling.md`, until they agree.
The original question follows.

**Why it matters.** "Memory traffic" and "achieved bandwidth" are required
metrics, and the documentation gives no conversion formula — the decomposition
in `06-profiling.md` is our own.

**Check.** Run a copy kernel moving a known number of bytes; confirm the
derived figure matches within a few percent before reporting any real number.

---

## Q13 — Infinity Cache size, bandwidth, and behaviour `open, narrowed` (2026-09-15)

**Narrowed: a tier between L2 and HBM exists on this part. Its size and
bandwidth are still unmeasured.** Pointer-chase latency
(`env/hw/probes/chase.cu`, `env/hw/20260915/`):

| Working set | Latency |
|---|---|
| 1 MiB, inside L2 | 81 ns |
| 64 MiB | 258 ns |
| 1 GiB, HBM | 342 ns |

The 64 MiB point sits clearly between the two ends, which is the plateau the
check predicted, so the memory-side cache is real; the secondary source's
figure for it was 218 ns. `rocminfo` still lists no L3 row, so the hardware
does not advertise it. What is open is capacity and bandwidth: all this
measurement supports is that a 64 MiB working set hits the tier and a 1 GiB
one does not. Whether the `nt` bit controls allocation into it is DQ5 in
`../design-doc/99-open-questions.md`, and it is the remaining residency
candidate for the latent cache (`OPEN-PROBLEMS.md` MAJ-6). The original
question follows.

**The bandwidth half of the check did not answer it.** The working-set sweep
in the same probe is launch-bound below 256 MiB, so only the large points are
bandwidth measurements at all: 3,249 GB/s at 512 MiB and 3,529 GB/s at
1024 MiB. Both are HBM-rate, and the probe cannot see a plateau at sizes it
cannot measure. A sweep with the launch cost amortised is what would give the
capacity number.

**Why it matters.** 256 MB of last-level cache between L2 and HBM would change
the memory model of the whole design — DeepSeek-V2-Lite's active weights per
token may partly live there. But the ROCm microarchitecture page does not
mention Infinity Cache or MALL at all; 256 MB / 17 TB/s is secondary reporting.

**Check.** Find a primary AMD source (MI300X architecture whitepaper or the Hot
Chips 2024 MI300X presentation), and empirically: a pointer-chase / bandwidth
sweep across working-set sizes should show a plateau between L2 (32 MB) and HBM.

---

## Q14 — Does prefetch depth actually control achieved bandwidth? `resolved` (2026-09-15)

**Resolved: the knee is at N=4 and the plateau is where predicted, so no
undocumented queue limit binds before the wave-level one.**
`env/hw/probes/stream_read.cu`, 1 GiB, full residency
(`env/hw/20260915/`), at one wave per SIMD, that is 304 blocks:

| Unroll N | Achieved read bandwidth |
|---|---|
| 1 | 2,706 GB/s |
| 2 | 2,410 GB/s |
| 4 | 4,325 GB/s |
| 8 | 4,304 GB/s |
| 16 | 3,787 GB/s |
| 32 | 4,095 GB/s |

Bandwidth jumps between N=2 and N=4 and is flat at about 4.3 TB/s after it;
the variation above the knee is run-to-run noise, not a trend. Two waves per
SIMD gives the same knee at N=4 and a 4,033 GB/s plateau, so the second wave
buys nothing, which is the conclusion `07-achievable-bandwidth.md` needs.
The headline read rate at unroll 8 is 3.943 TB/s mean with a 0.32% spread
across repeats.

**Corroboration.** BabelStream in HIP (`-n 50 -s 268435456`, double) on the
same VM: Copy 4,319,424 MB/s, Mul 4,231,808, Add 3,894,077, Triad
4,133,844, Dot 4,038,666. All five are above AMD's acceptance thresholds in
`07-achievable-bandwidth.md`, and the 3.66 to 4.3 TB/s band that file
derives holds, so the design's bandwidth term stands as written.

This settles `OPEN-PROBLEMS.md` MIN-22. The original question follows.

**Why.** `07-achievable-bandwidth.md` concludes that 1 wave/SIMD can saturate
HBM provided each wave keeps 4-8 loads in flight, based on `VMCNT` being 6 bits.
That is a necessary condition only; per-CU miss queues and L2 request queues are
undocumented and could bind first.

**Check.** A streaming-read kernel at 1 wave/SIMD occupancy, sweeping unroll
depth N = 1, 2, 4, 8, 16, 32 with all loads issued before the first `s_waitcnt`.
Plot achieved bandwidth vs N. Expect a knee around N=4 and a plateau at
3.7-4.3 TB/s. If bandwidth plateaus well below that, or the knee is much later
than predicted, a queue limit we have not identified is binding.

This single experiment validates or refutes the whole memory-level-parallelism
analysis, and it takes minutes.
