# 08 - Results of round 3

Filled from the record of 2026-09-17 (`env/hw/20260917/runs/`, the session
in `07-session-log.md`). Every number is from the runtime's event clock
(the 100 MHz timestamp of each event, the median over the iterations after
the first) unless marked `FWD_PASS` (the megakernel's own per-iteration
report, without the event timing). The target is the production vLLM
figure quoted for this model on this machine, 4.5 ms per token.

## The number

| Configuration | Run | us per token | ids |
|---|---|---|---|
| round 2's best (per-tile linears, nt weights) | `../02-validation`, `L27_head_it32_tile_nt` | 9,575 | PASS |
| the same on this VM, worker timing on (S5) | `L27_head_it32_tile_nt_wt` | 10,250.7 | PASS |
| plus the three fusions and the MFMA attention (S9) | `L27_head_it32_tile_fn1_fn2_fs_nt_mfma` | 8,904.0 | PASS |
| plus the per-head prep task | `L27_head_it32_tile_fn1_fn2_fs_nt_mfma` (10:40) | 5,006.2 | PASS |
| plus the batched router, merge and norm loads | `L27_head_it32_tile_fn1_fn2_fs_nt_mfma` (11:18) | 4,618.1 | PASS |
| plus the attention as regular tasks: **the round's number** (11:15 and, in the record, the 11:47 rerun) | `L27_head_it30/31/32_tile_at_fn1_fn2_fs_nt_mfma` | **4,599.5, 4,578.2, 4,589.3**; rerun 4,583.1, 4,571.2, 4,596.8 | PASS (32 of 32) |
| the gang attention, the same kernels (rerun) | `L27_head_it32_tile_fn1_fn2_fs_nt_mfma` | 4,575.6 | PASS |
| the same, no event timing, `FWD_PASS` clock | `L27_head_it29_tile_at_fn1_fn2_fs_nt_mfma` | 4,584.0 (P95 4,634); rerun 4,590.0 | 29 of 29 |
| the router at eight rows per lane, the merge's half row in one batch (reverted) | the 11:40 rerun | 4,620.6 to 4,654.4 | PASS |

2.09x faster than round 2 on the same clock; 1.6 to 2.2% above the 4.5 ms target over the seven final runs (4,571 to 4,600 us). The regular and the gang attention are within the run-to-run spread on the rerun.
Correctness of the number's configuration: the step-0 boundaries of the
2-layer graph PASS (`L2_it1_tile_at_fn1_fn2_fs_nt_mfma`, 0 FAIL rows), the
32 ids equal to the reference's, the kernel suites 100 of 100 (the prep,
router and merge suites cover the changed kernels). The route log differs
in the same way as round 2 (BF16 drift after step 22, no id affected).

## Where the time goes, per token (27 layers with the head)

Both columns from the worker-timing builds, the event gaps summed over the
layers with the corrected event-to-operator mapping (`07`, finding 1); the
right column is the last worker-timing run of the session (4,713 us; the
same graph without the worker timing and with the regular attention is
4,590).

| Operator | S5 baseline, us | last, us | what changed |
|---|---|---|---|
| mla_prep | 4,046 (150 per layer) | 287 (10.6) | one task per head instead of one task |
| attention | 1,631 (60.4) | 338 (12.5) | MFMA kernel |
| w13 (gang, 66 MB) | 1,061 (40.8) | 1,104 (42.5) | untouched: the stock CK gang kernel |
| merge | 634 (23.5) | 489 (18.1) | the loads batched (13.7 us exec; 20 before) |
| router | 600 (23.1) | 516 (19.8) | the norm folded in, the GEMV batched (16 us exec; 24 before) |
| w2 (gang, 33 MB) | 586 (22.6) | 621 (23.9) | the silu folded in |
| o_proj, down (per-tile, residual) | 420 (15.0) | 416 (14.8) | untouched: the stock CK tile |
| qkva, lm_head (per-tile) | 374 (13.4) | 415 (14.8) | the input norm folded in |
| norms (55, then 1) | 227 | 5 | folded into the router and the linears |
| iteration start | 219 | 181 | the prelaunch of the iteration's tasks |
| silu, combine | 267 | 147 | the silu folded into w2 |
| lm_head chunks (49 events of 8 tasks) | 144 | 141 | the head near its bandwidth floor |
| **total** | **10,261** | **4,713** | |

The remaining time by family: the CK linears (w13, w2, o_proj, down, qkva,
lm_head) 2.7 ms; the single-task and per-head kernels (router, merge, prep,
attention) 1.6 ms; boundaries and the iteration start about 0.8 ms (the
ladder: 2.3 to 2.9 us per operator, 246 operators).

## The kernels, standalone (`ktime`, the suite binaries, 50 launches)

| Kernel | plain | `nt` | `mfma` | after the session's edits |
|---|---|---|---|---|
| attention grid, cold cache | 33.8 us | 34.0 | 9.25 | |
| merge grid | 11.5 | 11.4 | 11.5 | 10.6 (the first batch change; the restructured merge not timed standalone) |
| shader clock from the spin | 2,107 MHz | 2,103 | 2,107 | |

Inside a graph the same kernels take about twice their standalone time
(the merge 13.7 us exec per head against 10.6 for the whole grid; the
attention 7.4 per split against 9.25 for the grid): their loads come from
HBM after the producer's write-back, where the standalone loop finds them
in the L2.

## The levers, one by one

| Lever | Row | Verdict |
|---|---|---|
| `--fuse-norm2` (O1), `--fuse-silu` (O2), `--fuse-norm1` (O3) | S6 | correct; neutral (a boundary costs 3 to 5 us, the removed operators were cheap); kept |
| `--mfma-attend` (O7) | S7 | 10,250.7 to 8,952.8; on |
| `--prefetch` (O8) | S8 | wiring PASS, correct, slower (12,486.7): off; its tasks occupy workers the linears need |
| the per-head prep task | 10:36 | 8,904 to 5,208; on (part of the graph now, no flag) |
| the batched router, merge, norm | 11:07 | 5,000 to 4,600; on (the kernels) |
| `--attend-tasks` with the MFMA kernel | 11:12 | 4,618 to 4,589 on the first pair, 4,576 against 4,597 on the rerun: within the spread; on |
| deeper batches (eight router rows per lane, the merge's half row at once) | 11:40 | 1% slower (register pressure); reverted |
| `--nt-streams` (O6) | S4 | within 0.2 us of plain standalone; not run in a graph |
| the fence knobs (I4) | S11 | removing either fence fails the step-0 compare; no completion fence also no faster (626.8 against 618.3 us, 2 layers); off. The sleep and CAS knobs' rows overlapped another queue and are discarded |

## Measurement corrections

- The event-to-operator mapping (`07`, finding 1): every per-operator
  number of round 2's `04-results.md` and `01-ideas.md` names the operator
  after the one it belongs to. Their sums per layer are right.
- The event clock's iteration marker: the highest firing index, not
  `num_events - 1` (`num_events` is the buffer capacity).
- A 2-layer compare needs `--iters 1`.

## What remains for 4.5 ms and below

The ladder's slope is the key: an operator of N regular tasks costs about
0.19 us per task in the runtime (2.3 us at N = 1, 8 at N = 40, 56 at
N = 296, with empty tasks), and the per-tile linears sit on that line:
qkva at 96 tasks 18 us predicted, 14.8 measured; `o_proj` and `down` at 64
tasks 12 predicted, 14.8 measured. They are not memory-bound (15 MB in
15 us is 1 TB/s of a 5.3 TB/s machine): they pay a serial cost per task,
which the completion path explains (every task of an operator increments
one event counter with an agent-scope atomic, 96 of them from eight XCDs;
the gang path counts per XCD first and touches the global counter once
per XCD, and the head's 400-task operator is chunked into events of 8 by
the runtime for the same reason). Removing either fence around the atomic
changed nothing (the knob rows), so the atomic itself is the cost.

- **A per-XCD completion hierarchy for regular tasks** (the runtime, next
  round): each event gets eight local counters filled at prelaunch (the
  scheduler knows every task's worker and XCD), the last task per XCD
  touches the global counter. At stake: 82 per-tile operators at 10 us
  each, about 0.8 ms per token, plus the same effect on every multi-task
  operator. This is the one lever that reaches 4.5 ms and below.
- The CK linears' K loop (2.7 ms of gaps, but see above: part of it is the
  per-task cost): the memory-bound CK pipeline (`GemmPipelineAgBgCrMem`)
  computes two prefetch stages for this tile (32 KB in flight per
  workgroup), the same as the current pipeline; more stages need a policy
  with the shuffled register distributions and a smaller K block for the
  registers.
- w13 (1.1 ms): 44 tiles per XCD over 37 workers, two rounds for seven
  workers and one for the rest; 128-row tiles would make it one round of
  22 workers at 64 KB per step.
- The iteration start (0.18 ms): the prelaunch of about 7,000 descriptors;
  fewer tasks per operator once the completion hierarchy makes wide
  operators cheap is the other side of the same coin.
- The router (0.5 ms): one task; its GEMV split over four tasks with the
  top-k as a second operator.
