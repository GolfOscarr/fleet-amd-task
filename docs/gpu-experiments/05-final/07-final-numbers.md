# 07 - Final numbers

The results page of the round, drafted before the session with round 4's
and round 3's numbers in place and the round-5 rows empty, so the session
fills tables instead of writing a page. Filled from the record
(`env/hw/<date>/runs/`, the session in `08-session-log.md`). Every number
is from the runtime's event clock (the median over the iterations after
the first) unless marked `FWD_PASS` (the megakernel's own per-iteration
report, the it29 rows without the event timing). The baselines are round
4's 4,262 to 4,341 us per token (`../04-kernels/10-results.md`) and round
3's 4,571 to 4,600 (`../03-acceleration/08-results.md`); the target is the
production vLLM figure quoted for this model on this machine, 4.5 ms per
token. Nothing in this round moved a kernel: the stack is round 4's,
reached by `run_fleet.py --final`, and the session's job is the number
with every check green.

Session: date, minutes of VM time, cost, the balance decision (S0 of
`05-session-plan.md`): to fill.

## The number

| Configuration | Run | us per token | ids | compare |
|---|---|---|---|---|
| round 3's number (its record) | `L27_head_it30/31/32_tile_at_fn1_fn2_fs_nt_mfma` | 4,571 to 4,600; `FWD_PASS` 4,584 to 4,590 | PASS | the it32 rows fail the head's logits and the route log by construction (round 4's reading) |
| round 4's number: the 48-task stack (its record, fifteen finals) | `L27_head_it30/31/32_..._gv_lg48_mt_mh2` and the knob sets | 4,262 to 4,341; `FWD_PASS` 4,267 to 4,310 | PASS | the same two failures; the ids the judge |
| **A: the stack (`--final`)** | `L27_head_it30_final_tile_at_fn1_fn2_fs_nt_nts_mfma_rf_w2cktile_gv_lg48_mt_mh2` | | | |
| | `..._it31_final_...` | | | |
| | `..._it32_final_...` | | | |
| A, `FWD_PASS` clock | `..._it29_final_...` | | | |
| **B: the stack with `POLL_SLEEP=8`** | `L27_head_it30_final_tile_at_fn1_fn2_fs_nt_nts_mfma_rf_pollsleep8+w2cktile_gv_lg48_mt_mh2` | | | |
| | `..._it31_final_..._pollsleep8+...` | | | |
| | `..._it32_final_..._pollsleep8+...` | | | |
| B, `FWD_PASS` clock | `..._it29_final_..._pollsleep8+...` | | | |
| the batch-4 set (optional, `queue-h6`) | `..._final_..._rf_gemvbatch4+w2cktile_...` it30/31/32, it29 | | | |
| the 8-event set (optional, `queue-h8`, only if R5 chose it) | `..._final_..._as8` (or `_as10`) it30/31/32, it29 | | | |
| A's set again (optional, `queue-h9`, only if A's medians spread by more than 2%) | the three A names again (the first set moved to `<name>.prev-<utc>`) | | | |

The round's number: A's three (or B's, if A1 chose it) and its `FWD_PASS`,
against round 4's 4,262 to 4,341 and the 4,500 target. To fill.

## The compare made green (R1, F1 and F2)

| Row | Boundaries | Route log | Verdict |
|---|---|---|---|
| round 4's reading (the exact rule) | the it1 model rows: layers 0 and 1 PASS, layers 2 to 26 reported MISSING_REF (64 keys); the it32 rows fail `head.B15.logits` (captured after the reference's step) | FAIL: 309 mismatches over the fifteen finals, 287 single swaps, 251 in the lowest slot, every multi-expert one after a swap | FAIL by construction, the ids the judge |
| `L27_head_it1_final_...` (A, one iteration) | expected: 7 PASS (the layers 0 and 1 cache rows, the head's logits and token), 64 `NOT_CAPTURED` | expected: the tie rule, one tie at step 0 in MoE layer 4, zero disagreements; `tol_rel` = 4 x the router floor | |
| `..._it1_final_..._pollsleep8+...` (B) | | | |
| `L2_it1_final_...` (the 2-layer compare) | expected: 19 PASS | expected: PASS by the tie rule | |

The tie's gap against the tolerance (from the report's route line): to fill.

## The head's events (R5, F7)

| Row | Argmax slices | Head events (the report table) | us per 2-layer iteration | ids |
|---|---|---|---|---|
| round 4 (`--head-grid` control, G6) | 50 | 49 events of 8 tasks, 111 us per token for the head's chunks (2.3 us per event) | 651 | PASS |
| `L2_head_it32_final_...` | 50 | | | |
| `..._as8` | 8 | | | |
| `..._as10` | 10 | | | |
| `L27_head_it1_final_..._as8` (the model at 8) | 8 | | | ids and compare: |

DECIDE R5: `--argmax-slices 8` (or 10) into the default if the 2-layer
median falls by more than 2% and the model row passes. Choice: to fill.

## The fault (R4, F5, MIN-36)

F5's readings found nothing static (`04-checklist.md`); the rows locate.

| Row (2 layers, one iteration) | `fault` | Reading |
|---|---|---|
| `--final --no-gemv-linears` (the configuration) | | faults: the layer count is not the cause; runs: `queue-h7` bisects |
| `--stop-after L0.o_proj` | | faults with the row above: the fault is at or before the first stock o_proj |
| `--stop-after L0.mla_merge_uv` | | runs while the row above faults: the stock o_proj after the tile merge |
| `--merge-halves 1` | | the halves' role |
| `queue-h7`: 3, 5, 9, 14 layers (only if the first row ran) | | the first faulting layer count |

The located operator (or the layer count) into MIN-36: to fill.

## The timing build (R3, F4, MIN-35) and the merge (F6)

| Row | Result | Reading |
|---|---|---|
| `L2_it1_final_..._wt` | completes in under a minute, 304 `[TASK_TIME2]` lines, no `[TIMING_MISSING]` line: MIN-35 closes; hangs at the watchdog: MIN-35 stays open with F4's form as its reading | |
| `L2_it32_final_..._wt` | the exec per class in `report_table.md` | |
| `ktime nt`: `mla_merge_uv` | round 4: 16.3 us (round 3: 11.4; the cause read offline, F6: the weights phase) | |

The exec-per-class table (the kernels' own time beside their gaps): to
fill from `report_table.md` if the rows ran.

## The decisions

| Rule | Quantity | Round 4 | Measured | Choice |
|---|---|---|---|---|
| S0 | the balance | $4.69 after round 4 | | taken 2026-09-18: the whole balance, no credits; the hard stop at minutes 70, 78 and 82 |
| R1 | the route log's classes on the two model rows; every captured boundary | FAIL by the exact rule (309 mismatches) | | PASS with zero disagreements is the claim; else dropped with the gap |
| A1 | B's medians against A's, both clocks | `POLL_SLEEP=8` within the spread on the model (-1.3% at 2 layers) | | into the default if B is not above A on either clock |
| A2 | the batch-4 set against A, both clocks | -7% at 2 layers, equal on the model | | into the default only if it wins both clocks |
| R5 | the 2-layer head median at 8 and 10 slices against 50; the model row's ids | 50 events, 111 us per token | | `--argmax-slices 8` (or 10) if the median falls by more than 2% and the model row passes |
| R4 | the four 2-layer rows' fault counts; the bisect if the first row ran | the fault at 27 layers; no 2-layer row had run | | the located operator into MIN-36; nothing in the stack changes |
| R3 | the first timing row completes | the hang (MIN-35) | | MIN-35 closed or kept with F4's form |
| G | A's three finals and their `FWD_PASS` (or B's) | 4,262 to 4,341; `FWD_PASS` 4,267 to 4,310 | | the round's number against round 4's and the 4,500 target |

## What did not run

The rows the minutes did not allow, with the reason: to fill.
