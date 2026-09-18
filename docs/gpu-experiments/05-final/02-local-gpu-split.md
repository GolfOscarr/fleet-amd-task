# 02 - The work split: laptop first, then one short session

Written 2026-09-18 from `01-ideas.md`. The rule of rounds 2 to 4 holds: no
minute of VM time goes to work that can be done on the laptop, and this
round's VM time is short by the balance ($4.69, 94 minutes; the $3 stop
rule of round 4's plan leaves 34 unless credits are added). Every item
below has a deliverable, a check that runs on the laptop, a time box and
the VM row that consumes it; every VM row names the laptop items it
needs, the PASS text and the decision it feeds.

## What carries over from round 4

- The session scripts and their rules (`env/session/`): push before
  `L start queue`, one graph run at a time, each row under
  `timeout ${ROW_TIMEOUT:-600}`, kill by anchored pid, the finals at
  distinct iteration counts, `fleet_env` installs the headers before every
  stage, a changed `new_tasks.patch` needs `FULL=1 L push` then
  `L start setup` (about 12 minutes on a fresh host with the download and
  the hardware census).
- The checks: the tests, `check_syntax.sh`, the dry run, the offline gfx942
  compile, the rehearsal (`rehearse.sh`), the preflight with shellcheck.
- The reading of the finals: the ids are the check (`11-lessons.md`,
  lesson 1); after C1 and C2 the compare rows read PASS as well.
- The reporting rules: report before the GPU is touched and at every
  DECIDE row; no VM, no deletion and no new milestone without the user's
  word; no names, no emoji, `docs/report` never committed.

## Local part (the laptop, before the VM)

Ordered by what the session needs first: the compare (F1, F2) makes the
finals green, the preset (F3) makes them reproducible, the diagnoses (F4
to F6) are the round-4 items, F7 and F8 are the tooling and the gate.

| Item | Deliverable | Laptop check | Time box | Feeds |
|---|---|---|---|---|
| **F1. The route log's tie rule** (C1) | `harness/run_reference.py` stores the 64 router weights per (step, MoE layer) in `ref_route_log.json` (or the logits, if the tolerance is the router floor); `compare_route_log` classifies every mismatch as tie, cascade or disagreement, prints the counts, fails on disagreements only; `harness/README.md` says so | `harness/tests/test_compare.py`: a swap within the tolerance is a tie, one outside fails, a later step after a tie is a cascade; the rule replayed on round 4's fifteen finals' `correctness_report.json` and `fleet_route_log.json` against a weights file built from the reference's top-6 (the ties it can see) gives zero disagreements | 2 h | R1, R2 |
| **F2. Iteration-aware boundaries** (C2, the cheap form; the right form if F1 finishes early) | `compare.py` reads the run's notes and reports a boundary dumped from a later iteration as `NOT_COMPARABLE (iteration N)`, neither PASS nor FAIL; the summary line names the iteration; the right form adds the step-31 dump to `run_reference.py` and the file choice to `compare.py` | `test_compare.py`: a round-4 it32 record reads the head as not comparable and the verdict as the ids; an it1 record unchanged | 2 h (4 h the right form) | R1, R2 |
| **F3. The `--final` preset** (D1) | `run_fleet.py --final`: the thirteen flags and the define of the number's stack, each overridable by naming it; the run name gains `_final`; `graph_plan.py` unchanged | `test_run_fleet_and_measure.py` (or the plan tests): `--final` gives the round-4 finals' operator and task counts from `plan.json` in the record (298 operators); `--final --no-nt-streams` drops one; the dry run | 2 h | R1 to R4 |
| **F4. The worker-timing hang** (L1) | the `unionT` variant of `env/offline_gfx942/run.sh` (`-DMPK_ENABLE_TIMING`), its resource line and the worker's epilogue read; the fix if the pass finds it (the `printf` split, or the totals to a device buffer the host prints) | the offline compile of `unionT` exits 0 and its scratch is within the plain union's; `check_syntax.sh`; the patch regenerates on the pristine fork | 3 h | R3 (only if fixed) |
| **F5. The half-merge fault** (L2) | the `attn` input maps of `register_mla_merge_uv_tile_mi300_task` and of the stock `linear_with_residual` registration side by side; the fix if the stride differs; otherwise the layer-bisect rows written for R4 | the plan tests; the dry run of the faulting configuration; the offline compile | 2 h | R4 |
| **F6. The merge's standalone 5 us** (L3) | the wait sequences of round 3's and round 4's merge in `dev_kt.s` side by side; the cause written into `10-results.md`'s standalone table, or the regression recorded | the offline compile; `ktime` unchanged | 1 h | R3 |
| **F7. The session tooling** | `env/session/queue-h1.txt` (R1: the it1 model compares with `--final`, with and without the knobs), `queue-h2.txt` (R2: the interleaved finals, N1), `queue-h3.txt` (R3: the timing build's 2-layer rows if F4 fixed it; the merge `ktime`), `queue-h4.txt` (R4: the fault's row or bisect), `queue-h5.txt` (R5: the head chunk row if N3's constant is one line); `test_queue_files.py` on the new files; `03-session-plan.md` with the rows, the PASS texts, the DECIDE rows and the hard stop by the balance; `rehearse.sh` regenerating `04-rehearsal.md` | the tests, the DRY runs, the rehearsal | 3 h | R0 to R5 |
| **F8. The checklist and the gate** | `05-checklist.md`: one box per deliverable above; `env/preflight.sh` 9 PASS with shellcheck; the memory note | the gate | 1 h | R0 |

About 16 hours of laptop work (20 with C2's right form); F1 to F3 and F7
to F8 (10 hours) are enough for the session, and F4 to F6 are taken as
time allows before it, in the order F5, F4, F6.

## The VM part

One session of about 45 minutes of VM time: the setup on a fresh host (12
minutes) and the rows below at 45 s to 70 s each. The balance holds it
without added credits only if the $3 rule of round 4 is waived for the
last session (the user's decision, recorded in the plan); with the rule,
R0 to R2 fit and R3 to R5 need credits.

| Row | Needs | Command shape | PASS text | RULE or DECIDE |
|---|---|---|---|---|
| **R0. Setup and the suites** | F3, F7, F8 | `grab.sh` or the hand provision (the 13-core host if listed), `FULL=1 L push`, `L start download`, `setup`, `hw`, `V preflight`, `V checks`, `V reference` (F1's weights, F2's step-31 dump), `V kernels` (plain and `nt`) | `PASS` on every stage; 19 suites 100 of 100; the reference's floors of round 2 | RULE: a suite FAIL ends the round's kernel claims and the session runs the finals on the round-4 header anyway (nothing here changed a kernel) |
| **R1. The compares** | F1, F2, F3 | `queue-h1`: `--layers 27 --head --iters 1 --final compare`, the same with `--runtime-flags=-DMPK_POLL_SLEEP=8`, the same with `-DMPK_NO_LOCAL_CAS`; one 2-layer `--iters 1 --final compare` | every boundary PASS, the route log PASS with its tie count printed (one tie expected at step 0, MoE layer 4), zero disagreements; the it32 rows of R2 read the head as not comparable or PASS against step 31 | DECIDE: a disagreement names a kernel; the round's compare claim is dropped and the reason written |
| **R2. The finals, interleaved** (N1, A1, A2) | F3, F7 | `queue-h2`: `--layers 27 --head --final` at 30, 31, 32 iterations for A (the stack) and B (the stack with `POLL_SLEEP=8`), interleaved A30 B30 A31 B31 A32 B32, each with `compare table`; then A29 and B29 without `--event-timing` (`FWD_PASS`); then the same six with `-DGEMV_BATCH=4` if the session has the minutes | ids PASS on every row; six per-token medians; the compare rows green by R1's rule | DECIDE A1: `POLL_SLEEP=8` into the default if B's median is not above A's on both clocks; DECIDE A2: batch 4 into the default only if it wins both clocks; the round's number is A's three (or B's, if B is the default) with its `FWD_PASS` |
| **R3. The exec table and the merge** (L1, L3, if F4 fixed the hang) | F4, F6 | `queue-h3`: two 2-layer rows with `--worker-timing --final` (it1 compare, it32 table); `L start ktime nt` | the `[TASK_TIME2]` lines present, the exec per class in `report_table.md` (the GEMV linear's exec against the CK tile's 18 us of round 3); the merge's `ktime` line | RULE: a hang ends the row at the watchdog and MIN-35 stays open with the offline reading attached |
| **R4. The fault** (L2) | F5 | `queue-h4`: `--layers 27 --head --iters 1 --tile-linears ... --merge-tasks --merge-halves 2` (the round-3 linears, no GEMV) with F5's fix; without a fix, the bisect at 3, 5, 9 and 14 layers | the row runs and its compare passes; or the first faulting layer count in the record | RULE: a fix that passes closes MIN-36; a bisect only records |
| **R5. The head's chunks** (N3, only if the constant is one line) | F7 | `queue-h5`: `--layers 2 --head --iters 32 --final` with the runtime's chunk constant at 16 and 32 through `--runtime-flags` | the head's events fall from 49 to 25 or 13; the per-iteration median against R2's 2-layer row | DECIDE: into the default if the median falls by more than 2% and the it1 compare with the head passes |
| **end** | | `L pull`; `git push`; `03-final-numbers.md` written from the record; the deletion after the user's yes | `Hourly Rate: $0.00/hour` | |

## The budget

| | |
|---|---|
| Balance after round 4 | $4.69 at $2.99 per hour: 94 minutes |
| R0 | about 15 minutes on a fresh host ($0.75) |
| R1 and R2 | 4 rows of 70 s and 8 of 70 s: about 16 minutes ($0.80); the batch-4 six 8 more |
| R3 to R5 | about 6 minutes if all three run |
| The session | 40 to 50 minutes, $2.00 to $2.50; the balance after about $2.20, below round 4's $3 stop rule, so the rule is waived for this last session or credits are added first (the user's decision) |

## The dependency graph

    F1 (tie rule) ---\
    F2 (boundaries) --+--> R1 (compares) --> R2 (finals) --> 03-final-numbers.md
    F3 (--final) ----/                        ^
    F7 (tooling) needs F1 to F3; F8 (gate) needs all -> R0 -> R1
    F4 (timing hang) -> R3 (only if fixed)
    F5 (fault) -> R4
    F6 (merge 5 us) -> R3's ktime line
    N3's one-line reading -> R5

F4, F5 and F6 are independent of each other and of F1 to F3; if the hours
are short they are dropped in the order F6, F4, F5, and their rows with
them. The session runs on F1, F2, F3, F7 and F8 alone.

## What this split does not cover

- Every route of `01-ideas.md`'s last section: the direct-to-LDS weight
  streams, the fusions, the iteration start, the router's latency, MAJ-8.
  They are the next reader's rounds, ranked in `../04-kernels/11-lessons.md`,
  Part 3.
- `docs/report`: written by the user from `03-final-numbers.md`; never
  committed.
