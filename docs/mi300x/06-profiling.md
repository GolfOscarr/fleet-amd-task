# 06 — Profiling and Required Metrics

The task requires: correctness error, Fleet-native operations, remaining
fallbacks, consecutive layers completed, **GPU launches, median and P95 latency,
memory traffic, achieved bandwidth, occupancy and GPU resource usage**, plus
TPOT and tokens/s if end-to-end decode is reached.

This file maps each to a tool and a counter *now*, rather than on day 4.

## Tools

| Tool | Use |
|---|---|
| **rocprofv3** | Open-source CLI profiler shipped with ROCm 6.2+. HIP API traces, kernel traces, hardware counter collection. Our primary tool. |
| **rocprof-compute** (formerly Omniperf) | Counter-based analysis: cache hit rates, bandwidth utilization, wavefront occupancy, stall cycles. |
| **rocprof-sys** (formerly Omnitrace) | Timeline tracing across host and device. |
| **rocm-smi / amd-smi** | Clocks, power, partition mode during the run. |
| **In-kernel `S_MEMREALTIME`** | Per-task timing *inside* the megakernel — see below. |

The reference repo ships an `NCU_Usage_Manual.md` (NVIDIA Nsight Compute);
its ROCm counterpart for our purposes is rocprof-compute.

## The megakernel problem

A profiler attributes counters and time **per kernel dispatch**. A Fleet
megakernel is *one* dispatch for an entire decode step, so:

- Per-operator latency is invisible to rocprofv3. We must self-instrument with
  `S_MEMREALTIME` timestamps into a device-side ring buffer, then post-process.
  Build this with the first task, not at the end.
- Counters are aggregated over the whole megakernel, which is fine for total
  memory traffic and achieved bandwidth but useless for per-task attribution.
- **"GPU launches" becomes our headline metric** and is trivially countable from
  a rocprofv3 kernel trace: the point of the exercise is that this number
  collapses from many-per-layer to ~1 per token.

## Metric → measurement plan

| Required metric | How |
|---|---|
| GPU launches | `rocprofv3 --kernel-trace` — count dispatches per decoded token |
| Median / P95 latency | 32 tokens is too few for a stable P95; run the decode loop many times over the same prefilled cache and report the distribution. State N explicitly. |
| Time per output token, tokens/s | Wall-clock over the 32-token generation, excluding prefill and excluding the one-time KV-cache conversion (the task explicitly permits this exclusion) |
| Memory traffic | TCC EA counters, summed across channels — see below |
| Achieved bandwidth | traffic ÷ kernel duration; compare against 5.3 TB/s peak |
| Occupancy | `SQ_LEVEL_WAVES` with `SQ_ACCUM_PREV_HIRES`; plus compile-time `-Rpass-analysis=kernel-resource-usage` |
| GPU resource usage | VGPR/AGPR/SGPR/LDS per kernel from the compiler resource report |
| Correctness error | Not a profiler metric — per-boundary tensor comparison against the reference; see `05-software-stack.md` |

## Counters

From the ROCm MI300/MI200 performance counter documentation. **Names below are
as extracted from that page; the TCC section came through a summarizer rather
than verbatim, so confirm each name against `rocprofv3 --list-avail` on the
machine before trusting a number.**

### Memory traffic / HBM (block: TCC, per-channel `[n]`)

| Counter | Meaning |
|---|---|
| `TCC_EA0_RDREQ[n]` | 32- or 64-byte read requests to the efficiency arbiter |
| `TCC_EA0_RDREQ_32B[n]` | 32-byte subset of the above |
| `TCC_EA0_WRREQ[n]` | 32- and 64-byte write transactions |
| `TCC_EA0_WRREQ_64B[n]` | 64-byte subset of the above |
| `TCC_EA0_RDREQ_DRAM[n]` | EA read requests that reach HBM |
| `TCC_EA0_WRREQ_DRAM[n]` | EA write requests that reach HBM |
| `TCC_EA0_ATOMIC[n]` | 32- or 64-byte atomic requests — **our task-graph flag traffic lands here** |

**Naming caveat:** MI200 uses the `TCC_EA_*` prefix, MI300 uses **`TCC_EA0_*`**.
Copying a counter list from an MI200 blog post will silently collect nothing.

Bytes are derived, not measured. Reads:
`32·RDREQ_32B + 64·(RDREQ − RDREQ_32B)`. Writes:
`64·WRREQ_64B + 32·(WRREQ − WRREQ_64B)`. The documentation gives **no**
bandwidth formula, so this decomposition is our own and must be sanity-checked
against a known-size copy kernel before we report anything from it.

TCC counters are indexed over channels (reportedly `[0..31]`); aggregate with the
`_sum` suffix forms. Confirm the instance count on the machine.

### L2 hit rate (block: TCC)

`TCC_REQ[n]`, `TCC_HIT[n]`, `TCC_MISS[n]`, `TCC_READ[n]`, `TCC_WRITE[n]`,
`TCC_ATOMIC[n]`, `TCC_WRITEBACK[n]`, `TCC_CYCLE[n]`, `TCC_BUSY[n]`.

Derived metric `L2CacheHit` is described as "Percentage of fetch, write, atomic,
and other instructions that hit the data in L2 cache".

**This is the metric that demonstrates the Fleet thesis.** The paper's claim is
that chiplet-aware scheduling raises L2 hit rate (they report 12% → 54% at batch
32 for Qwen3-8B). Our per-XCD L2 hit rate is the direct evidence of whether
Chiplet-task placement is doing anything.

### Occupancy and utilization (block: SQ, GRBM)

| Counter | Meaning |
|---|---|
| `SQ_LEVEL_WAVES` | Number of inflight waves |
| `SQ_ACCUM_PREV_HIRES` | Accumulator; **required** after any level counter |
| `SQ_WAVES` | Wavefronts dispatched to sequencers |
| `SQ_BUSY_CYCLES` | Cycles the sequencer reports busy |
| `SQ_WAIT_ANY` | Cycles waiting |
| `SQ_VALU_MFMA_BUSY_CYCLES` | MFMA busy |
| `GRBM_GUI_ACTIVE` | GPU active cycles |
| `GRBM_COUNT` | Free-running GPU cycles |

Documented rule, verbatim: "All level counters must be followed by
`SQ_ACCUM_PREV_HIRES` counter to measure average latency." The page gives
latency formulas (e.g. vector memory latency = `SQ_ACCUM_PREV_HIRES` ÷
`SQ_INSTS_VMEM`) but **no occupancy formula**.

For a persistent kernel, the interesting occupancy question is not peak but
*steady-state*: how much of the megakernel's duration are workers executing
tasks versus polling. `SQ_WAIT_ANY` versus `SQ_BUSY_CYCLES` is the crude
version; the `S_MEMREALTIME` instrumentation is the honest one, because a
worker spinning in `s_sleep` is "occupant" but idle — and a naive occupancy
number will flatter us.

### Instruction mix (block: SQ)

`SQ_INSTS_VALU`, `SQ_INSTS_MFMA`, `SQ_INSTS_VMEM`, `SQ_INSTS_LDS`,
`SQ_INSTS_SALU`, `SQ_INSTS_SMEM`, `SQ_INSTS_FLAT`, and the MFMA op counters
`SQ_INSTS_VALU_MFMA_MOPS_BF16` (counted "in the unit of 512").

Note the MI300-specific caveat, verbatim: `SQ_INSTS_LDS` is "Number of LDS
instructions issued (MI200: includes flat; MI300: does not include flat)".

### Page-wide caveat, verbatim

> "Preliminary validation of all MI300 and MI200 series performance counters is
> in progress. Those with an asterisk (\*) require further evaluation."

Sanity-check any counter we report against a microbenchmark with known traffic.

## Commands to have ready

```bash
# what counters actually exist on this machine
rocprofv3 --list-avail

# kernel trace: the "GPU launches" metric
rocprofv3 --kernel-trace --output-format csv -- ./decode

# HIP API + kernel timeline
rocprofv3 --hip-trace --kernel-trace -- ./decode

# counter collection
rocprofv3 --pmc TCC_EA0_RDREQ_sum TCC_EA0_WRREQ_sum TCC_HIT_sum TCC_MISS_sum -- ./decode

# full analysis
rocprof-compute profile -n fleet_decode -- ./decode
rocprof-compute analyze -p workloads/fleet_decode
```

Exact flag syntax varies by ROCm version — confirm against `rocprofv3 --help` on
the machine and record the working invocations, since the task requires
"profiling commands and results" to be reproducible.

## Open items

- Verify counter names and TCC instance count via `--list-avail` → `99-open-questions.md` Q11
- Validate our bytes-from-requests arithmetic against a known-size copy → Q12
