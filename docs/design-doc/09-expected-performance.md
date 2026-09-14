# 09 - Expected performance

What the design predicts, as a model with named terms, each with the
measurement that fills it in; the metrics the task requires and the command
that produces each; and the table the final report will contain. Bytes and
counts are from `sources/graph_counts.py`; bandwidths from
`docs/mi300x/07-achievable-bandwidth.md`; per-task figures from Fleet's
paper via `docs/fleet/01-paper-review.md`.

## The model

Per decode iteration:

```
T_iter = T_bw + N_b * t_b + T_serial

T_bw      = bytes / BW                bandwidth term: 4,709.9 MiB at 3.66-5.3 TB/s
N_b       = 326                       operator boundaries in the chain (03-synchronization.md)
t_b       = ?                         exposed latency per boundary: last producer's release flush,
                                      global atomic, consumer poll wake, acquire invalidate
T_serial  = ?                         time in ops that are not bandwidth-bound and not overlapped:
                                      mla_prep (2 MiB by one workgroup), the router (256 KiB by one
                                      workgroup), norms, argmax; each is on the chain
```

`T_bw` is known. `t_b` and `T_serial` are the two unknowns, and the chain
structure means both are fully exposed: nothing in this graph overlaps
anything else across an operator boundary.

### The bandwidth term

| Scope | Bytes | at 5.3 TB/s | at 4.3 TB/s (measured peak) | at 3.66 TB/s (AMD acceptance) |
|---|---|---|---|---|
| one MoE layer (layer 1) | 159.63 MiB | 31.6 us | 38.9 us | 45.7 us |
| layer 0 | 159.38 MiB | 31.5 us | 38.9 us | 45.7 us |
| head | 400.0 MiB | 79.1 us | 97.5 us | 114.6 us |
| **per iteration, weights + cache** | **4,709.9 MiB** | **931.8 us** | **1,148.5 us** | **1,349.4 us** |
| activations and split-KV partials (written and read once; not in the floor) | about 62 MiB: partials 55.8, logits 0.4, MoE intermediates 5, the rest under 1 | 12 us | 15 us | 18 us |
| 32 iterations | | 29.8 ms | 36.8 ms | 43.2 ms |

931.8 us is a floor: any measurement below it is a measurement error (a
skipped layer, a warm cache, a mis-timed window). 1,148-1,349 us is the
band a bandwidth-bound implementation with everything else hidden would
land in.

### The boundary term

Every one of the 326 boundaries costs, in sequence: the last producer's
`buffer_wbl2 sc1` and `s_waitcnt` (writeback of that XCD's dirty lines,
which for our tasks are kilobytes), a `flat_atomic_add sc0 sc1` on the
global counter (a round trip to wherever agent-scope atomics resolve on a
multi-XCD part), the consumer's next poll (`s_sleep 1`, 64 cycles, about
30 ns granularity), and its `buffer_inv sc1`. The design has no measurement
of this on MI300X. Fleet's paper reports that a 104K-cycle linear task
amortizes it and that "very short executing tasks" would not; our tasks are
shorter than theirs (`docs/fleet/07-gap-analysis.md`).

| `t_b` | `N_b x t_b` per iteration | as a fraction of the 1,148 us band |
|---|---|---|
| 0.5 us | 163 us | 14% |
| 1 us | 326 us | 28% |
| 2 us | 652 us | 57% |
| 5 us | 1,630 us | 142% |

This term is the main risk in the design, and it is why `08-milestones.md`
measures layer 1 alone on day 3 before building further: layer 1 has 12
boundaries against a 38.9-45.7 us bandwidth floor, so its measured time
gives `t_b` directly (`(T_layer1 - T_bw) / 12`, with `T_serial` for one
layer subtracted once `mla_prep` and the router are timed). If `t_b` is
around 1 us, the full iteration lands near 1.5 ms; if it is 5 us, the
graph must be restructured before M3 is worth pursuing.

Reductions available if `t_b` is large, in order of ease:

1. Fuse each `rmsnorm` into the op after it: the runtime ships
   `rmsnorm_linear` and `gang_rmsnorm_linear_mi300.cuh`; the router can
   normalize its own input; `model.norm` into `lm_head`. Removes 54
   boundaries (326 to 272).
2. Fuse `moe_silu_mul` into the W13 gang task's epilogue, as
   `gang_linear_silu` does for the dense layer. Removes 26 more (to 246).
3. Fuse `mla_prep` into `qkv_a_proj`'s last XCD or into `mla_attend`
   (redundantly per XCD, at 8x the `W_uk` read: +14 MiB per layer, 3 us).
   Removes 27 (to 219). Only if the boundary cost exceeds 3 us.

Each removes boundaries at the cost of a fused kernel variant; none is in
the baseline plan.

### The serial term

| Op (per layer unless stated) | Bytes | Why it is serial | Estimate |
|---|---|---|---|
| `mla_prep` | 2 MiB | one workgroup reads `W_uk` | a single CU streams perhaps 50-100 GB/s: 20-40 us **per layer** if unhidden; 0.5-1.1 ms per iteration |
| router | 256 KiB | one workgroup | 3-5 us per layer; 80-130 us per iteration |
| `rmsnorm` x 2 | 4 KiB | one workgroup; latency only | ~2 us each |
| `argmax_partial` + reduce (head) | 200 KiB | 50 small tasks then 1 | ~5 us |

`mla_prep` as one workgroup is the largest serial item and the design flags
it: if the day-3 measurement shows it, the `W_uk` product moves into
`mla_attend` as a phase A computed once per XCD for that XCD's splits
(2 MiB read per XCD, 16 MiB per layer, 3.7 us of bandwidth), or `mla_prep`
becomes a gang op with 16 tiles (one head each, 128 KiB per tile). Either
change is local to one kernel and its Python call, and reduction 3 above
is the same move. The router at 256 KiB is tolerable as one workgroup.

### Putting it together

| Scenario | `t_b` | `mla_prep` | Per iteration | Tokens/s |
|---|---|---|---|---|
| bandwidth only (not reachable) | 0 | hidden | 1,148-1,349 us | 741-871 |
| design as written, good case | 1 us | 20 us per layer, serial | 1,148 + 326 + 540 = ~2.0 ms to ~2.2 ms | 450-500 |
| design with reductions 1-3 | 1 us | gang, hidden | 1,148 + 219 = ~1.37 ms to ~1.57 ms | 640-730 |
| boundary-dominated, with reductions 1-3 (219 boundaries) | 5 us | gang | 1,148 + 1,095 = ~2.2 ms to ~2.4 ms | 410-450 |

The design as written targets **M2 correctness first** and reports the
measured `t_b` and `mla_prep` time; the reductions are the documented next
steps if the numbers say so. The design does not claim the band; it claims
the band plus two measured terms, and says which knobs move them.

### Fleet's own reference point

Qwen3-8B on MI350 at batch 1: 6.73 ms per token with 543 tasks per layer over
36 layers (19,548 tasks), 1.16x over the Mirage baseline, from dispatch
reduction alone (`docs/fleet/01-paper-review.md`, Table 4). Our graph has
1,880 tasks per token in 326 operators; the per-task dispatch cost that
Fleet removed is not the term that matters for us, the per-boundary latency
is, and their paper does not report it.

## Metrics required by the task, and how each is produced

| Metric | How | Where reported |
|---|---|---|
| Correctness error | `compare.py` rows per boundary (`07-correctness.md`) | `correctness_report.md` |
| Fleet-native operations | 217 of 326 ops per iteration reuse shipped kernels, 96.9% of bytes; listed per op | `02-task-graph.md` table, confirmed by the generated `task_graph_0.json` |
| Remaining fallbacks | none on the device path; 107 new ops and 2 variants listed | same |
| Consecutive layers completed | the largest `--layers N` with B13 within threshold at every layer | milestone table |
| GPU launches | `rocprofv3 --kernel-trace`: expect 3 dispatches per generation (`prepare_kernel`, `worker_kernel`, `scheduler_kernel`), 0.09 per token | trace CSV |
| Median and P95 latency per token | the event-timing buffer (`MPK_ENABLE_EVENT_TIMING`, read back with `get_event_timing`): the gap between consecutive `EVENT_END_OF_TASK_GRAPH` firings is one iteration; 32 per generation, 5 generations, 160 samples. `[FWD_PASS]` as shipped prints only iterations 1-9 (`persistent_kernel.cuh:1535-1536`, worker variant `:1047`); our patch removes the throttle so it cross-checks the buffer. Host wall clock around `mpk()` divided by 32 is the outer check | `metrics.json` |
| Memory traffic | `rocprofv3 --pmc TCC_EA0_RDREQ_sum TCC_EA0_WRREQ_sum` over one generation, bytes from request counts (64 B per request, validated against a copy kernel of known size first), divided by 32 | `metrics.json`; compared to 4,709.9 MiB |
| Achieved bandwidth | traffic per iteration divided by per-iteration time | same; compared to 3.66-5.3 TB/s |
| L2 hit rate | `TCC_HIT_sum / (TCC_HIT_sum + TCC_MISS_sum)` over a generation; with and without `USE_NT_WEIGHTS=1` | `metrics.json` |
| Occupancy | one wave per SIMD by construction; `SQ_LEVEL_WAVES` with `SQ_ACCUM_PREV_HIRES`; and the honest number, worker busy versus waiting from `MPK_ENABLE_TIMING` (`poll_cycles`, `dep_cycles`, `exec_cycles` per worker) | `metrics.json` |
| GPU resource usage | `-Rpass-analysis=kernel-resource-usage` on the megakernel: VGPR, AGPR, SGPR, LDS, spills; per new kernel before and after | build log, committed |
| Time per output token, tokens/s | median per-iteration time; 1 / that | `metrics.json` |
| Per-operator time | event-timing buffer: the gap between consecutive event firings, mapped to operator names through `task_graph_0.json` | `metrics.json`, per-op table |

A profiler attributes counters to kernel dispatches; ours is one dispatch
per generation, so per-operator numbers come only from the in-kernel event
timestamps (`docs/mi300x/06-profiling.md`). `rocprofv3 --pmc` serializes
dispatches and deadlocks the split worker/scheduler launch
(`persistent_kernel.cuh:2034`); counters are collected with the combined
`persistent_kernel` path or with the `--profiling` Perfetto trace, which is
worker-only. The working invocation is recorded when found.

## The report table

Filled by `measure.py`; predicted columns from this file, measured columns
from the machine.

| Quantity | Predicted | Measured | Ratio |
|---|---|---|---|
| bytes per iteration | 4,709.9 MiB weights + cache; about 4,772 MiB with activations and partials | | |
| time per iteration, median | 1,148-1,349 us + 326 `t_b` + `T_serial` | | |
| time per iteration, P95 | | | |
| achieved bandwidth | 3.66-4.3 TB/s over `T_bw` | | |
| `t_b` from layer 1 | unknown | | |
| `mla_prep` time | 20-40 us | | |
| launches per generation | 3 | | |
| release flushes per iteration | 1,838 | (from `MPK_ENABLE_TIMING` signal cycles) | |
| L2 hit rate | 16-17% (Fleet's batch-1 figure; ours has less fused reuse) | | |
| VGPR per lane, union | under 256, no spills | | |
| LDS per task, maximum | 54 KiB (`mla_attend` spec kernel) or CK's | | |

## What would falsify the design

- Layer 1 measured above 150 us with `t_b` above 5 us: the chain is too
  fine; apply reductions 1-3 and re-measure before M3.
- `mla_attend` above 10 us per layer at 33 splits: the split count or the
  kernel is wrong; sweep E3, then compare CK versus the spec kernel.
- Achieved bandwidth on the gang linears below 3 TB/s with `t_b` and
  `T_serial` subtracted: the CK pipeline at M = 1 is not streaming; check
  prefetch depth by disassembly (D19), then consider VALU GEMVs (D18).
- Bytes per iteration above 4,950 MiB (the 4,709.9 MiB of weights and cache plus about 62 MiB of activations and partials, plus 4% slack): something is read twice; the
  per-op event timing and per-op byte estimates localize it.
