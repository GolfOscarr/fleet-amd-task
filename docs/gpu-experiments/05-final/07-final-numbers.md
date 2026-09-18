# 07 - Final numbers

The results page of the round, drafted before the session with round 4's
and round 3's numbers in place and filled from the record of the session
of 2026-09-18 (`env/hw/20260918/runs/`, the session in
`08-session-log.md`). Every number is from the runtime's event clock (the
median over the iterations after the first) unless marked `FWD_PASS`
(the megakernel's own per-iteration report, the it29 rows without the
event timing; the median over the 28 `[FWD_PASS]` lines of `run.out`).
The baselines are round 4's 4,262 to 4,341 us per token
(`../04-kernels/10-results.md`) and round 3's 4,571 to 4,600
(`../03-acceleration/08-results.md`); the target is the production vLLM
figure quoted for this model on this machine, 4.5 ms per token. Nothing
in this round moved a kernel: the stack is round 4's, reached by
`run_fleet.py --final`, and the session's job was the number with every
check green.

Session: 2026-09-18, 11:09 to 12:26 UTC on `enc1-gpuvm016` (the round-4
host), 77 minutes, $3.74 billed ($4.59 to $0.85); the balance decision of
S0: the whole balance, no credits, the hard stop at minutes 70, 78 and 82
(the last queue started at minute 68, the deletion at 77).

## The number

| Configuration | Run | us per token | ids | compare |
|---|---|---|---|---|
| round 3's number (its record) | `L27_head_it30/31/32_tile_at_fn1_fn2_fs_nt_mfma` | 4,571 to 4,600; `FWD_PASS` 4,584 to 4,590 | PASS | the it32 rows fail the head's logits and the route log by construction (round 4's reading) |
| round 4's number: the 48-task stack (its record, fifteen finals) | `L27_head_it30/31/32_..._gv_lg48_mt_mh2` and the knob sets | 4,262 to 4,341; `FWD_PASS` 4,267 to 4,310 | PASS | the same two failures; the ids the judge |
| **A: the stack (`--final`)** | `L27_head_it30_final_tile_at_fn1_fn2_fs_nt_nts_mfma_rf_w2cktile_gv_lg48_mt_mh2` | **4,287.0** | 30 of 32 (the 30 tokens of the run) | PASS: 7 boundaries, 1 not comparable, 64 not captured; the route log 31 ties, 31 cascades, 0 disagreements |
| | `..._it31_final_...` | **4,290.9** | 31 of 32 | PASS: the same classes, 32 cascades |
| | `..._it32_final_...` | **4,284.2** | PASS, 32 of 32 | PASS: the same classes, 32 ties, 35 cascades |
| A, `FWD_PASS` clock | `..._it29_final_...` | **4,265.0** (28 lines, 4,097 to 4,352) | 28 tokens | not compared (no event timing) |
| **B: the stack with `POLL_SLEEP=8`** | `L27_head_it30_final_tile_at_fn1_fn2_fs_nt_nts_mfma_rf_pollsleep8+w2cktile_gv_lg48_mt_mh2` | 4,261.4 | 30 of 32 | PASS, the same classes as A |
| | `..._it31_final_..._pollsleep8+...` | 4,260.3 | 31 of 32 | PASS |
| | `..._it32_final_..._pollsleep8+...` | 4,257.8 | PASS, 32 of 32 | PASS |
| B, `FWD_PASS` clock | `..._it29_final_..._pollsleep8+...` | 4,274.0 (28 lines, 4,071 to 4,357) | 28 tokens | |
| the batch-4 set (`queue-h6`) | `..._final_..._rf_gemvbatch4+w2cktile_...` it30 / it31 / it32; it29 | 4,293.3 / 4,275.8 / 4,270.6; `FWD_PASS` 4,294.0 | 30, 31, 32 of 32 | PASS on the three (19 ties, 20 to 22 cascades, 0 disagreements) |
| the last minutes (`queue-h0`, the user's ask at minute 66) | `..._rf_pollsleep8+gemvbatch4+...` it30; it29 | 4,287.8; `FWD_PASS` 4,296.5 | 30 of 32 | PASS |
| | `..._rf_pollsleep16+...` it30; it29 | 4,316.0; `FWD_PASS` 4,300.5 | 30 of 32 | PASS |
| | `..._rf_pollsleep4+...` it30 | 4,284.0 | 30 of 32 | PASS |
| the 8-event set (`queue-h8`) | | did not run: R5 kept 50 slices | | |
| A's set again (`queue-h9`) | | did not run: A's spread is 0.16%, under the 2% gate | | |

**The round's number: A's three, 4,284 to 4,291 us per token on the
event clock, `FWD_PASS` 4,265**, with the ids equal on every final and
every compare row green (the route log at zero disagreements, every
captured boundary PASS). Against round 4's 4,262 to 4,341 it is the same
stack on the same host, inside round 4's band and 4.6 to 4.8% below the
4,500 us target; against round 3's 4,571 to 4,600 it is 6.5% lower.

The ids column reads "N of 32" on the it30 and it31 rows because the run
generates N tokens and the compare matches them against the reference's
32; the it32 rows are the full match. The compare's "1 not comparable"
is the head's logits captured at a later iteration than the reference's
step 0 (F2); the 64 "not captured" rows are the layers 2 to 26 cache
boundaries the reference does not dump.

## The compare made green (R1, F1 and F2)

| Row | Boundaries | Route log | Verdict |
|---|---|---|---|
| round 4's reading (the exact rule) | the it1 model rows: layers 0 and 1 PASS, layers 2 to 26 reported MISSING_REF (64 keys); the it32 rows fail `head.B15.logits` (captured after the reference's step) | FAIL: 309 mismatches over the fifteen finals, 287 single swaps, 251 in the lowest slot, every multi-expert one after a swap | FAIL by construction, the ids the judge |
| `L27_head_it1_final_...` (A, one iteration) | 8 PASS (the layers 0 and 1 cache rows, the head's logits and token), 0 FAIL, 64 `NOT_CAPTURED` | PASS: 1 step compared, 1 tie, 0 cascades, 0 disagreements; `tol_rel` 0.0144 (4 x the router floor 0.003593) | **PASS** |
| `..._it1_final_..._pollsleep8+...` (B) | the same: 8 PASS, 64 `NOT_CAPTURED` | the same: 1 tie, 0 disagreements | **PASS** |
| `L2_it1_final_...` (the 2-layer compare) | 19 PASS, 0 FAIL | PASS: 0 mismatches | **PASS** |
| the finals (it30 to it32, A, B and batch 4) | 7 PASS, 1 `NOT_COMPARABLE` (the head's logits at a later iteration), 64 `NOT_CAPTURED` | PASS on every row: 19 to 32 ties, 20 to 35 cascades, 0 disagreements | **PASS** on all nine |

The tie of step 0 is in MoE layer index 4 (the fifth MoE layer): the
reference's sixth expert is 49 at weight 0.039638 and the megakernel's is
2 at 0.039606, a gap of 3.2e-5, 0.08% of the larger weight, against the
tolerance of 1.44%. Every other difference at step 0 is an order swap
within the same six experts (the compare matches sets, not positions).
Over 30 steps the classes read 31 ties and 31 to 35 cascades and zero
disagreements: the round-4 record's 287 single swaps were ties by this
rule, and the multi-expert differences after them are cascades.

## The head's events (R5, F7)

| Row | Argmax slices | Head events (the report table) | us per 2-layer iteration | ids |
|---|---|---|---|---|
| round 4 (`--head-grid` control, G6) | 50 | 49 events of 8 tasks, 111 us per token for the head's chunks (2.3 us per event) | 651 | PASS |
| `L2_head_it32_final_...` | 50 | 50 (87 rows, 37 before the head); `argmax_partial` gap 2.49 us, `argmax_reduce` 0.27 | **615.2** | 32 of 32 |
| `..._as8` | 8 | 8 (45 rows); `argmax_partial` 7.30, `argmax_reduce` 16.87 | 623.6 (+1.4%) | 32 of 32 |
| `..._as10` | 10 | 10 (47 rows); `argmax_partial` 9.73, `argmax_reduce` 14.42 | 635.4 (+3.3%) | 32 of 32 |
| `L27_head_it1_final_..._as8` (the model at 8) | 8 | | | 1 of 32 at one iteration; compare PASS (8 boundaries, 1 tie) |

DECIDE R5: the event count fell from 50 to 8 as designed, and the
iteration got slower, not faster: with 8 partitions each argmax partial
covers 50 tasks' worth of columns and the reduce waits on the widest,
so the two gaps grow by more than the 42 events saved. The slices stay
50; `queue-h8` did not run.

## The fault (R4, F5, MIN-36)

F5's readings found nothing static (`04-checklist.md`); the rows locate.

| Row (one iteration, `--final --no-gemv-linears`) | `fault` | Reading |
|---|---|---|
| 2 layers, halves 2 (the round-4 configuration) | 0, compare PASS (19 boundaries) | the configuration runs at 2 layers: the layer count matters, `queue-h7` bisects |
| 2 layers, `--stop-after L0.o_proj` | 0 | |
| 2 layers, `--stop-after L0.mla_merge_uv` | 0 | |
| 2 layers, `--merge-halves 1` (the halves control) | **faults**: rc 1, the queue's two fault lines (`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` and the `hipErrorIllegalAddress` it raises) | the one-half merge faults at 2 layers already, with round 4's fault class |
| `queue-h7`: 3, 5, 9, 14 layers, halves 2 | 0, 0, 0, 0 (compare PASS on each) | the halves-2 fault needs more than 14 layers, or the head (round 4's faulting run was `L27_head`; the bisect rows have no head) |

Into MIN-36: the fault is not in the stock o_proj after the merge (both
cuts run) and not static in the 2-layer graph at two halves; it moves
with the halves setting (one half faults at 2 layers, two halves run to
14 layers without the head). The next cut, if the problem is taken up
again, is the halves-1 row under `--stop-after L0.mla_merge_uv` at 2
layers (45 s) and the halves-2 row at 27 layers without the head. Nothing
in the stack changes: the finals run with the GEMV linears, where the
merge runs on every row of three rounds.

## The timing build (R3, F4, MIN-35) and the merge (F6)

| Row | Result | Reading |
|---|---|---|
| `L2_it1_final_..._wt_...` | **hung**: the kernel at 100% GFX activity for five minutes, `fwd_pass.log` empty, no `[WORKER_XCD]`, `[TASK_TIME2]` or `[TIMING_MISSING]` line; the loop and the run killed by pid at minute 53 | the buffer form did not fix it: with no device `printf` left in the timing build the hang is the same, so the hostcall path of F4's reading is not the cause. MIN-35 stays open with that as its reading |
| `L2_it32_final_..._wt` | skipped (the rule of R3) | |
| `ktime nt`: `mla_merge_uv` | 16.28 us (round 4: 16.3; round 3: 11.4) | the standalone regression as read offline (F6, the weights phase); unchanged, no kernel change |

The exec-per-class table did not come: the timing build is the only
form that produces it and it hangs at one iteration on the round-4
header. What is known after this session: the hang is not the printing,
not the class switch (the same fifteen types run without the define),
and it is in the timing build's device code under `MPK_ENABLE_TIMING`,
which differs from the plain build only by the cycle counters and the
per-class accumulators in the worker loop.

## The decisions

| Rule | Quantity | Round 4 | Measured | Choice |
|---|---|---|---|---|
| S0 | the balance | $4.69 after round 4 | $4.59 read at S0 (92 minutes) | taken 2026-09-18: the whole balance, no credits; the hard stop at minutes 70, 78 and 82; the deletion at minute 77 on the user's yes, $0.85 left |
| R1 | the route log's classes on the two model rows; every captured boundary | FAIL by the exact rule (309 mismatches) | 1 tie, 0 cascades, 0 disagreements on both; 8 boundaries PASS, 0 FAIL | **the compare claim stands** |
| A1 | B's medians against A's, both clocks | `POLL_SLEEP=8` within the spread on the model (-1.3% at 2 layers) | event clock B 4,258 to 4,261 against A 4,284 to 4,291 (B lower by 0.6%); `FWD_PASS` B 4,274 against A 4,265 (B higher by 0.2%) | B is above A on one clock: **the stack stays A**, the knob stays out of the default |
| A2 | the batch-4 set against A, both clocks | -7% at 2 layers, equal on the model | event clock 4,271 to 4,293 (two of three rows below A, one above); `FWD_PASS` 4,294 against 4,265 | not a win on both clocks: **the constant stays 8** |
| R5 | the 2-layer head median at 8 and 10 slices against 50; the model row's ids | 50 events, 111 us per token | 615.2 at 50, 623.6 at 8, 635.4 at 10; the model row at 8 passes | the median rose: **the slices stay 50**; `queue-h8` skipped |
| R4 | the four 2-layer rows' fault counts; the bisect if the first row ran | the fault at 27 layers; no 2-layer row had run | halves 2 runs at 2, 3, 5, 9 and 14 layers; halves 1 faults at 2 layers | the halves' role and the layer bound into MIN-36; nothing in the stack changes |
| R3 | the first timing row completes | the hang (MIN-35) | hung again in the buffer form | MIN-35 kept open, the reading revised (not the printf) |
| the last minutes | the sleep knob at 4 and 16, and 8 with batch 4 (`queue-h0`) | | 4,284.0; 4,316.0 (`FWD_PASS` 4,300.5); 4,287.8 (`FWD_PASS` 4,296.5) | none wins both clocks: **nothing enters the default** |
| G | A's three finals and their `FWD_PASS` | 4,262 to 4,341; `FWD_PASS` 4,267 to 4,310 | **4,284 to 4,291; `FWD_PASS` 4,265** | the round's number: inside round 4's band, 4.6 to 4.8% below the 4,500 target, every check green |

## What did not run

| Row | Reason |
|---|---|
| `queue-h3`'s second row (the exec per class at 32 iterations) | the first row hung (R3's rule) |
| `queue-h8` (the finals with 8 head events) | R5 kept 50 slices |
| `queue-h9` (A's set again) | A's three medians spread by 0.16%, under the 2% gate |
| a kernel change in the last minutes | the user allowed one at minute 71; with the deletion due by minute 90 a JIT build, a possible fault and the pull did not fit safely, and the one change with a read cause (F6's weights phase) gains nothing in the graph by its own reading; the knob rows of `queue-h0` ran instead |
