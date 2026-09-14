# 99 — Open Questions

Every claim our design depends on that is **not** confirmed by a primary source,
with the specific check that settles it. Anything here should be resolved in the
first hours of GPU access, before it can poison a design decision.

Status: `open` | `resolved` | `blocked`. Record the answer inline with the date
and the command that produced it.

---

## Q1 — What is the actual workgroup→XCD mapping? `open`

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

---

## Q3 — Partition query/set commands `open`

**Why it matters.** SPX vs CPX silently changes dispatch, coherence behaviour,
and available memory. A reboot restores SPX; a prior user may have left it in
CPX. Every measurement must record the mode.

**What is documented.** The partitioning overview gives **no command lines** —
only that `amd-smi` can adjust modes at runtime.

**Check.** Find the working query and set syntax (`amd-smi`, and whether
`rocm-smi --showcomputepartition` / `--setcomputepartition` still work). Add an
assertion to our startup path.

---

## Q4 — Does HIP's agent-scope atomic emit the right cache ops? `open`

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

## Q5 — Cost of `buffer_inv sc1` / `buffer_wbl2 sc1` `open`

**Why it matters.** Sets the granularity of the task graph. If an agent-scope
acquire costs a few hundred nanoseconds, fine-grained per-tensor dependencies
across XCDs are unaffordable and we must batch dependency resolution.

**Check.** Microbenchmark a loop of agent-scope acquire/release pairs against
the same loop with workgroup-scope ops; difference is the cache-op cost. Measure
both uncontended and with 8 XCDs participating.

---

## Q6 — Real wave slots per SIMD at our register footprint `open`

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

## Q9 — ROCm version and library availability on the machine `open`

**Check.** `rocminfo`, `hipcc --version`, `amd-smi static`, and
`python -c "import torch; print(torch.__version__, torch.version.hip)"`.
Commit the output — it is part of the reproducibility claim.

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

## Q11 — Counter names and TCC instance count `open`

**Why it matters.** The TCC counter names in `06-profiling.md` came through a
summarizer, not verbatim documentation, and the MI200→MI300 prefix change
(`TCC_EA_*` → `TCC_EA0_*`) means a wrong name collects nothing rather than
erroring loudly.

**Check.** `rocprofv3 --list-avail`, and confirm the number of TCC instances.

---

## Q12 — Validate bytes-from-requests arithmetic `open`

**Why it matters.** "Memory traffic" and "achieved bandwidth" are required
metrics, and the documentation gives no conversion formula — the decomposition
in `06-profiling.md` is our own.

**Check.** Run a copy kernel moving a known number of bytes; confirm the
derived figure matches within a few percent before reporting any real number.

---

## Q13 — Infinity Cache size, bandwidth, and behaviour `open`

**Why it matters.** 256 MB of last-level cache between L2 and HBM would change
the memory model of the whole design — DeepSeek-V2-Lite's active weights per
token may partly live there. But the ROCm microarchitecture page does not
mention Infinity Cache or MALL at all; 256 MB / 17 TB/s is secondary reporting.

**Check.** Find a primary AMD source (MI300X architecture whitepaper or the Hot
Chips 2024 MI300X presentation), and empirically: a pointer-chase / bandwidth
sweep across working-set sizes should show a plateau between L2 (32 MB) and HBM.
