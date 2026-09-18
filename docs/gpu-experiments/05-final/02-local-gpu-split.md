# 02 - The work split: laptop first, then one short session

Written 2026-09-18 from `01-ideas.md`; revised the same day after the
double-check of `03-local-preparation.md` (Part 4 there: the knob
`NO_LOCAL_CAS` is unsafe by reading and leaves the session, the head's
event count is a plan constant and gains an item, the fault's first
suspect is out). The rule of rounds 2 to 4 holds: no minute of VM time
goes to work that can be done on the laptop, and this round's VM time is
short by the balance ($4.69, 94 minutes; the $3 stop rule of round 4's
plan leaves 34 unless credits are added). Every item below has a
deliverable, a check that runs on the laptop, a time box and the VM row
that consumes it; every VM row names the laptop items it needs, the PASS
text and the decision it feeds. The items in detail are in
`03-local-preparation.md`; the progress record is `04-checklist.md`.

## What carries over from round 4

- The session scripts and their rules (`env/session/`): push before
  `L start queue`, one graph run at a time, each row under
  `timeout ${ROW_TIMEOUT:-600}`, kill by anchored pid, the finals at
  distinct iteration counts, `fleet_env` installs the headers before every
  stage, a changed `new_tasks.patch` needs `FULL=1 L push` then
  `L start setup` (about 12 minutes on a fresh host with the download and
  the hardware census), the record committed right after every pull.
- The checks: the tests, `check_syntax.sh`, the dry run, the offline gfx942
  compile, the rehearsal (`rehearse.sh`), the preflight with shellcheck.
- The reading of the finals: the ids are the check (`../04-kernels/11-lessons.md`,
  lesson 1); after F1 and F2 the compare rows read PASS as well.
- The reporting rules: report before the GPU is touched and at every
  DECIDE row; no VM, no deletion and no new milestone without the user's
  word; no names, no emoji, `docs/report` never committed.

## Local part (the laptop, before the VM)

Ordered by what the session needs first (`03`, Part 3): the preset (F3)
is in every row, the compare (F1, F2) makes the finals green, the argmax
slices (F7) are the round's one speed item, the tooling (F8) needs their
flags; the diagnoses (F4 to F6) are round 4's items and are dropped first
if the hours run short; the gate (F9) closes.

| Item | Deliverable | Laptop check | Time box | Feeds |
|---|---|---|---|---|
| **F3. The `--final` preset** (D1) | `run_fleet.py --final`: the thirteen flags and the define of the number's stack unless named, `--no-event-timing`, `--no-nt-streams` and `--no-gemv-linears`, the run name's `_final` | `test_run_fleet_and_measure.py`: `--final` equals the spelled-out stack (246 operators and 6,386 tasks, the record's `plan.json`), the overrides, the run name; the dry run | 2 h | R1 to R5 |
| **F1. The route log's tie rule** (C1) | `run_reference.py` stores `w_all`, the 64 softmax weights per (step, MoE layer), from the gate's input and weight; `compare_route_log` classifies every mismatch as tie, cascade or disagreement, prints the counts, fails on a disagreement only, keeps the exact rule for the old format; `harness/README.md` | `test_compare.py`: the four cases; the replay on round 4's fifteen finals' logs gives zero disagreements; `test_run_reference_smoke.py` | 2 to 3 h | R0 (the reference), R1, R2 |
| **F2. Iteration-aware boundaries** (C2) | `compare.run` reads the run's iteration from the meta and reports a boundary dumped from a later iteration as `NOT_COMPARABLE (iteration N)`; the overall verdict by the ids and the route log; the right form (the step-31 reference dump and the file choice) if the box allows | `test_compare.py`: the it32 fixture reads the head as not comparable and the verdict as the ids; the it1 fixture unchanged; the step-31 form compares against its file | 2 h (4 h the right form) | R1, R2 |
| **F7. The head's event count** (N3) | `graph_plan.py`: `argmax_slices` (default 50), `run_fleet.py --argmax-slices N`; at 8 the head's 400 tasks make 8 events instead of 50 (the gcd rule of `runtime.cc`) | `test_graph_plan.py`: the counts at 50 and 8; the dry run | 1 h | R5 |
| **F8. The session tooling** | `queue-h1.txt` (R1), `h2` (R2, the interleaved finals), `h3` (R3, if F4 fixed the hang), `h4` (R4), `h5` (R5); `test_queue_files.py`'s round-5 set; `rehearse.sh` with the round-5 rows writing `06-rehearsal.md`; `05-session-plan.md` with the rows, the PASS texts, the DECIDE rows, the budget with the $3 rule's decision, the playbook | the tests, the DRY runs, the rehearsal | 3 h | R0 to R5 |
| **F5. The half-merge fault** (L2, MIN-36) | three readings (the stock o_proj's input map against the merge tile's whole-tensor maps; the two graphs' task types; the tile form's store at `halves = 2`) and the fix with its suite row if one finds the defect; otherwise the bisect rows for R4 and the readings attached to MIN-36 | the plan tests, the dry runs, the suite row, the offline compile | 2 h | R4 |
| **F4. The worker-timing hang** (L1, MIN-35) | the `unionT` offline variant (`-DMPK_ENABLE_TIMING`), its resource line against `union`'s, the worker's epilogue read; the fix if found (the `printf` split, or the class totals in a device buffer the host prints), the patch regenerated | `unionT` exits 0; `preflight.sh`; `check_syntax.sh`; the printer's test if the buffer form | 3 h | R3 (only if fixed) |
| **F6. The merge's standalone 5 us** (L3) | round 3's merge as a scratch offline variant beside round 4's, the wait sequences and the `attn_s` stores side by side; an `if constexpr` on the store if it is the cost; the cause or the regression written into `../04-kernels/10-results.md` | the offline compile; the suites' merge rows bit-exact | 1 h (2 with the fix) | R3 |
| **F9. The checklist and the gate** | `04-checklist.md` ticked or deferred with reasons; `preflight.sh` 9 PASS with shellcheck; the offline compile of every variant; the fork at zero dirty lines; `07-final-numbers.md` drafted with the round-4 numbers in place; the memory note | the gate | 1 h (2 with the draft) | R0 |

About 17 hours of laptop work (22 with every optional form); F3, F1, F2,
F7, F8 and F9 (11 hours) are the session's minimum, and F4 to F6 are
taken as time allows before it, in the order F5, F4, F6.

## The VM part

One session of about 45 minutes of VM time: the setup on a fresh host (12
minutes) and the rows below at 42 s (a 2-layer row) to 68 s (a model row)
each, round 4's measured durations. The balance holds it without added
credits only if the $3 rule of round 4 is waived for the last session
(the user's decision, recorded in the plan); with the rule, R0 to R2 fit
and R3 to R5 need credits.

| Row | Needs | Command shape | PASS text | RULE or DECIDE |
|---|---|---|---|---|
| **R0. Setup and the suites** | F1, F3, F8, F9 | `grab.sh` or the hand provision (the 13-core host if listed), `FULL=1 L push`, `L start download`, `setup`, `hw`, `V preflight`, `V checks`, `V reference` (F1's weights; F2's step-31 dump if written), `V kernels` (plain and `nt`) | `PASS` on every stage; 19 suites 100 of 100; the reference's floors of round 2 | RULE: a suite FAIL ends the round's kernel claims and the session runs the finals on the round-4 header anyway (F6's fix is the only kernel change, and it is bit-exact by its suite rows) |
| **R1. The compares** | F1, F2, F3 | `queue-h1`: `--layers 27 --head --iters 1 --final compare`; the same with `--runtime-flags=-DMPK_POLL_SLEEP=8`; one `--layers 2 --iters 1 --final compare` | every captured boundary PASS (the layers 0 and 1 cache rows, the head's logits and token at one iteration; layers 2 to 26's rows `NOT_CAPTURED`); the route log PASS with its counts printed (one tie expected at step 0, MoE layer 4, zero disagreements); the 2-layer row as in round 4 | DECIDE: a disagreement names a kernel; the round's compare claim is dropped and the reason written; the finals run regardless |
| **R2. The finals, interleaved** (N1, A1, A2) | F3, F8 | `queue-h2`: `--layers 27 --head --final` at 30, 31, 32 iterations for A (the stack) and B (the stack with `POLL_SLEEP=8`), interleaved A30 B30 A31 B31 A32 B32, each with `compare table`; then A29 and B29 with `--no-event-timing` (`FWD_PASS`); then the same eight with `-DGEMV_BATCH=4` if the session has the minutes | ids PASS on every row; the compare rows green by R1's rule (the it32 rows' boundaries `NOT_COMPARABLE` or PASS against step 31); six per-token medians and two `FWD_PASS` medians per configuration | DECIDE A1: `POLL_SLEEP=8` into the default if B's medians are not above A's on either clock; DECIDE A2: batch 4 into the default only if it wins both clocks; the round's number is the default's three with its `FWD_PASS` |
| **R3. The exec table and the merge** (L1, L3; only if F4 fixed the hang) | F4, F6 | `queue-h3`: `--layers 2 --iters 1 --final --worker-timing compare`, `--layers 2 --iters 32 --final --worker-timing table`; `L start ktime nt` | the `[TASK_TIME2]` lines present and the exec per class in `report_table.md` (the GEMV linear's exec against the CK tile's 18 us of round 3); the merge's `ktime` line against 16.3 | RULE: a hang ends the row at the watchdog and MIN-35 stays open with the offline reading attached |
| **R4. The fault** (L2) | F5 | `queue-h4`: `--layers 27 --head --iters 1 --final --no-gemv-linears --tile-linears` (the round-3 linears with the half merge) with F5's fix; without a fix, the bisect at 3, 5, 9 and 14 layers with `--merge-halves 2`, and `--layers 3 --merge-halves 1` as the control | the row runs and its compare passes; or the first faulting layer count and the halves' role in the record | RULE: a fix that passes closes MIN-36; a bisect only records |
| **R5. The head's events** (N3) | F7 | `queue-h5`: `--layers 2 --head --iters 32 --final --argmax-slices 8 table`, the same at 10, and `--layers 27 --head --iters 1 --final --argmax-slices 8 compare` | the head's events fall from 50 to 8 (or 10) in `report_table.md`; the per-iteration median against R2's 2-layer row; the ids and the compare PASS on the model row | DECIDE: `--argmax-slices 8` into the default if the 2-layer median falls by more than 2% and the model row's ids pass; then one more A set in R2's form with it, if the minutes allow |
| **end** | | `L pull` and its commit; `git push`; `07-final-numbers.md` filled from the record; the deletion after the user's yes | `Hourly Rate: $0.00/hour` | |

## The budget

| | |
|---|---|
| Balance after round 4 | $4.69 at $2.99 per hour: 94 minutes |
| R0 | about 15 minutes on a fresh host ($0.75) |
| R1 and R2 | 3 rows of 68 s and 8 of 68 s: about 13 minutes ($0.65); the batch-4 eight 9 more |
| R3 to R5 | about 7 minutes if all three run; R5's extra A set 4 more |
| The session | 40 to 50 minutes, $2.00 to $2.50; the balance after about $2.20, below round 4's $3 stop rule, so the rule is waived for this last session or credits are added first (the user's decision, recorded in `05-session-plan.md`) |

## The dependency graph

    F3 (--final) -----\
    F1 (tie rule) -----+--> R0 (reference) --> R1 (compares) --> R2 (finals) --> 07-final-numbers.md
    F2 (boundaries) ---/                                           ^
    F7 (argmax slices) -> R5 -------------------------------------/ (an extra A set if it wins)
    F8 (tooling) needs F1, F2, F3, F7; F9 (gate) needs all -> R0
    F4 (timing hang) -> R3 (only if fixed)
    F5 (fault) -> R4
    F6 (merge 5 us) -> R3's ktime line

F4, F5 and F6 are independent of each other and of F1 to F3; if the hours
are short they are dropped in the order F6, F4, F5, and their rows with
them. The session runs on F1, F2, F3, F7, F8 and F9 alone.

## What this split does not cover

- `NO_LOCAL_CAS` as a default: settled by reading (`03`, Part 4, A1): the
  store form can publish past an unwritten slot and the scheduler reads
  by position; the knob stays a probe. Its round-4 finals stand in
  `../04-kernels/10-results.md` as measured.
- Every route of `01-ideas.md`'s last section: the direct-to-LDS weight
  streams, the fusions, the iteration start, the router's latency, MAJ-8.
  They are the next reader's rounds, ranked in `../04-kernels/11-lessons.md`,
  Part 3.
- `docs/report`: written by the user from `07-final-numbers.md`; never
  committed.
