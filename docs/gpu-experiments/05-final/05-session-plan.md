# 05 - Session plan: one short session on a 1x MI300X, the final numbers with every check green

Written 2026-09-18 after the laptop items F3, F1, F2, F7 and F8 of
`03-local-preparation.md` (the checklist `04-checklist.md`). One session,
in round 4's shape (`../04-kernels/07-session-plan.md`): every row is one
literal command of `env/session/laptop.sh` (`L`), the text that decides
PASS, whether the row runs on its own (AUTO), applies a written rule
(RULE) or reports a decision (DECIDE), and what to do on FAIL. The exact
commands the scripts expand to are in `06-rehearsal.md`, generated in DRY
mode by `ROUND=5 env/session/rehearse.sh`. The run log and the numbers go
to `07-final-numbers.md` (drafted before the session with round 4's
numbers in place).

What is new against round 4's plan:

- Every queue row runs the finals' stack through `--final` (F3: the
  thirteen flags and `-DMPK_W2_CK_TILE`; a flag named on the row keeps its
  value; `--no-event-timing`, `--no-nt-streams`, `--no-gemv-linears`).
- The compare rows read PASS when they should: the route log by the tie
  rule (F1: the reference stage writes the 64 router weights; a single
  swap within 4 x the router floor is a tie, a later mismatch a cascade,
  only a disagreement fails), the boundaries by their iteration and by
  the layers the reference captured (F2: `NOT_COMPARABLE`,
  `NOT_CAPTURED`, neither a failure). The ids stay the check of a final.
- One speed item, the head's event count (F7: `--argmax-slices 8` makes 8
  events of the 400-task head instead of 50, up to 95 us per token).
- No kernel changed since round 4 (F6's fix, if any, is bit-exact by its
  suite rows), so the suites run on the streaming build only, the build
  the finals use; the plain build ran 100 of 100 on the same kernels in
  round 4.
- Each queue row runs under the queue's watchdog (600 s); a hang is a
  FAIL row, not a lost half hour.

## Budget

| | |
|---|---|
| Balance | $4.69 after round 4 (`../04-kernels/09-session-log.md`); rate $0.00, no VM |
| Shape | 1x MI300X, $2.99 per hour, billed per minute (round 4: 167 minutes, $8.22) |
| Minutes | 94 at the balance |
| The session | R0 to R5 about 41 minutes ($2.05); the optional sets of `queue-h6.txt` 9 more; the pull and the deletion 2; about 45 to 52 minutes, $2.25 to $2.60 |
| After | about $2.10 to $2.45: below round 4's $3 stop rule |

Round 4's rule stopped a session when the balance read less than $3. For
this last session the rule cannot hold with the balance as it is, so one
of two things is decided before `grab.sh`, and the choice is written into
the row S0 below: the rule is waived for the last session (the hard stop
is then minute 60 of VM time, $3.00, leaving $1.69), or credits are added
first and the rule stands. Running out of balance deletes the VM with
everything on it, so the pull at the end is not optional and R2's rows
come before every optional one.

## Before the session, on the laptop

| Check | How | State |
|---|---|---|
| the gate | `SHELLCHECK=1 bash env/preflight.sh`: 9 PASS | F9 |
| the fork pristine | `git -C repos/fleet-chiplet-megakernel status --short` shows untracked files only | F9 |
| the branch pushed | `git status` clean, `git log origin/local/round-5..local/round-5` empty | at F9 |
| no VM, no address | `L balance` shows `No virtual machines`; `env/session/vm.ip` absent; no `grab.sh` process | at S0 |
| the queue files | `.venv/bin/python -m pytest harness/tests/test_queue_files.py -q` (every round-5 row parses, names a unique run, builds its plan); `06-rehearsal.md` regenerated | F8 |
| the decisions below | the balance decision; the go for the VM | open |

## Decisions

| Decision | Choice |
|---|---|
| The balance | open: the $3 rule waived for the last session (hard stop at minute 60, $3.00), or credits added first; written into S0 before `grab.sh` |
| Order | R0 (setup, checks, reference, the `nt` suites), R1 (the compares), R2 (the finals interleaved and their `FWD_PASS` rows), R5 (the head's events), R4 (the fault), R3 (the timing build in F4's form, its first row the check), then `queue-h6.txt` as the minutes allow; the pull before anything optional |
| Autonomy | as round 4: every RULE and DECIDE row by its written threshold, reported as it goes; nothing waits for the user except the deletion and the balance decision |
| A compare row fails in R1 | a disagreement in the route log or a FAIL on a captured boundary names a kernel or the tolerance; the round's compare claim is dropped with the reason, the finals run regardless (the ids are their check) |
| A suite row fails in R0 | nothing in this round changed a kernel except F6's optional fix; a FAIL names it, its file is reverted to round 4's on the laptop after the session, and the finals run on the round-4 header as pushed |
| The knobs and the constants | A1: `POLL_SLEEP=8` into the default if B's medians are not above A's on either clock; A2 (batch 4) only if it wins both clocks; R5: `--argmax-slices 8` if the 2-layer median falls by more than 2% and the model row's ids pass; `NO_LOCAL_CAS` is out (unsafe by reading, `03` Part 4) |
| Provisioning | the 13-core host by hand if listed (round 4's `DOWN ENTER`), else `grab.sh`; the user is told before the polling starts |
| Deletion | asked every time; the record pulled and committed, the branch pushed first |

## Reporting protocol

- Before `grab.sh`: the balance, the balance decision, and a one-line
  "polling for a VM now" or "provisioning the 13-core host now".
- After every stage row: the `PASS`/`FAIL` line of `env/logs/session.status`
  verbatim, with the minute mark and the cost (`L report`).
- At every RULE and DECIDE row: the rule, the measured value and the
  choice, in the report that follows; the next command does not wait.
- At the checkpoint (after R2) and at the end: the commit hash of the
  pulled record and the balance.
- At the end: the last `L status`, the `Hourly Rate: $0.00/hour` line and
  the commit hash of the pulled record.

## The session (about 45 minutes of VM time)

Minute marks from the provision, with round 4's measured durations
(download 86 s, hw 130 s, setup 453 s, checks 81 s, reference 71 s, the
`nt` suites 308 s, a 2-layer row 42 s, a model row 68 s).

| Minute | Row | Command | PASS when | Mode | On FAIL |
|---|---|---|---|---|---|
| -5 | S0 | `L balance`; the balance decision written here | `No virtual machines`, `Hourly Rate: $0.00/hour`; the decision taken | DECIDE (user: the balance) | a VM already listed: stop |
| 0 | R0 | the provision (the 13-core host by hand if listed, else `bash env/session/grab.sh`), then `FULL=1 L push` | an address in `env/session/vm.ip`; the push completes | AUTO | no host within an hour: the user is told, nothing is billed |
| 1 | R0 | `L start download`; `L start setup`; `L start hw`; `L ssh "cd /home/hotaisle/metalOps && bash env/session/vm.sh preflight"` | `PASS download`, `PASS setup`, `PASS hw`, `PASS preflight` | AUTO | setup: read `env/logs/setup.out`; the fix on the laptop, `L push`, `L start setup` again |
| 9 | R0 | `L start checks`; `L wait checks 5`; `L start reference`; `L wait reference 6` | `PASS checks` with 7 PASS lines; `PASS reference`; `harness/ref/ref_route_log.json` on the VM has `w_all` (F1's reference guard ran on every step) | AUTO | reference FAIL: the guard names the layer (the scoring formula against the gate) or the stage's own error; no compare can run, the finals still can |
| 12 | R0 | `L start kernels nt`; `L wait kernels 8` | every suite 100 of 100 on the streaming build (19 suites) | RULE: a FAIL names the kernel (F6's file if it changed; otherwise a machine difference, logged) | the finals run on the header as pushed |
| 18 | R1 | `L start queue env/session/queue-h1.txt`; `L wait queue 5` | the two model rows: `compare=PASS`, ids PASS, every captured boundary PASS (the layers 0 and 1 cache rows, the head's logits and token), the route log PASS with its counts (one tie expected at step 0, MoE layer 4, zero disagreements), layers 2 to 26 `NOT_CAPTURED`; the 2-layer row `compare=PASS` on 19 boundaries and the route log | DECIDE R1: a disagreement names a kernel or the tolerance (its gap and tolerance are in the report); the compare claim is dropped with the reason | the finals run regardless |
| 21 | R2 | `L start queue env/session/queue-h2.txt`; `L wait queue 10` | ids PASS on every row; `compare=PASS` on the six it30 to it32 rows (the head `NOT_COMPARABLE`, the route log by the rule); six per-token medians (three A, three B) and the two `FWD_PASS` medians | DECIDE A1 (below); the round's number is the default's three with its `FWD_PASS` | a spread above 2% within A or B: the set is rerun at the end if the minutes allow |
| 30 | | `L pull` (the checkpoint; the record committed right after, before any other commit) | the commit hash | AUTO | |
| 31 | R5 | `L start queue env/session/queue-h5.txt`; `L wait queue 5` | `table=PASS` on the three 2-layer head rows; the head's events 50, 8 and 10 in their `report_table.md`; the model row ids PASS and `compare=PASS` | DECIDE R5 (below) | a FAIL on the model row: the slices stay 50 |
| 35 | R4 | `L start queue env/session/queue-h4.txt`; `L wait queue 4`; then, only if its first row passed, `L start queue env/session/queue-h7.txt`; `L wait queue 5` | four rows at 2 layers: the configuration (`fault=0` or `fault=1`), the cut after `L0.o_proj`, the cut after `L0.mla_merge_uv`, the halves control; the pattern names the operator (the cut after the merge runs and the cut after o_proj faults: the stock o_proj after the tile merge); `queue-h7` gives the first faulting layer count when 2 layers pass | RULE: the rows only record; the located operator goes into MIN-36 | every row faults: the fault is before the first merge, and `--stop-after L0.qkva` is the next cut if a minute remains |
| 40 | R3 | `L start queue env/session/queue-h3.txt`; `L wait queue 4`; then `L start ktime nt`; `L wait ktime 3` | the first row (one iteration, the timing build in F4's buffer form) completes in under a minute with 304 `[TASK_TIME2]` lines in `fwd_pass.log` and no `[TIMING_MISSING]` line; the exec per class in `report_table.md`; `ktime_nt.txt` with the merge's line | RULE: the first row hung at the watchdog (600 s): the second row is skipped (`vm.sh kill`), MIN-35 stays open with F4's form as its reading; the row completes: MIN-35 closes | |
| 42 | | if the minutes allow (the hard stop at minute 60 under the waived rule): `L start queue env/session/queue-h6.txt`; `L wait queue 10` | ids PASS; the batch-4 set's three medians and `FWD_PASS` against A's; the 8-event set's the same | DECIDE A2 and R5's confirmation (below) | |
| end | end | `L pull` and its commit; `git push`; then, after the user's yes, `L delete --yes`; `L balance` shows `Hourly Rate: $0.00/hour` | | DECIDE (user: the deletion) | if the rate is not $0.00, check the TUI by hand |

Hard stops: minute 60 of VM time under the waived rule (`vm.sh kill
--all` after the `queue.sh run` loop, then the pull and the deletion
whatever the state); with credits added, round 4's rule (below $3).

### The DECIDE rows

| Rule | Row | Quantity | Round 4 | Rule |
|---|---|---|---|---|
| R1 | the compares | the route log's classes on the two model rows; every captured boundary | ids PASS, the route log FAIL by the exact rule (309 mismatches over the finals, 287 single swaps), the head FAIL by construction | PASS with zero disagreements is the claim; a disagreement is reported with its gap and tolerance and the claim is dropped |
| A1 | R2 | B's medians (event clock and `FWD_PASS`) against A's | `POLL_SLEEP=8` -1.3% at 2 layers, within the spread on the model | into the default if B is not above A on either clock; otherwise the stack stays A |
| A2 | `queue-h6` | the batch-4 set against A on both clocks | -7% at 2 layers, equal on the model | into the default only if it wins both clocks |
| R5 | the head's events | the 2-layer head median at 8 and 10 slices against 50; the model row's ids | 50 events, 111 us per token | `--argmax-slices 8` (or 10, if better) into the default if the 2-layer median falls by more than 2% and the model row passes; its finals set in `queue-h6` if the minutes allow |
| R4 | the fault | the four 2-layer rows' `fault` counts; the bisect's first faulting layer count if the 2-layer row passed | `hipErrorIllegalAddress` at 27 layers; no 2-layer row of the configuration has run | the located operator (or the layer count) and the halves' role go into MIN-36; nothing in the stack changes |
| G | the number | A's three finals and their `FWD_PASS` (or B's, if A1 chose it) | 4,262 to 4,341 us; `FWD_PASS` 4,267 to 4,310 | the round's number, against round 4's and the 4,500 target, with the compare rows green |

### How the numbers are read

- `report_table.md`: the per-operator gaps from the event clock (event i
  fires when operator i - 1 completes), the per-token median over the
  iterations after the first; the head's events count as the rows named
  `event_N` after the head's linear.
- The compare (`correctness_report.md`): the ids row, the route log line
  with its counts (ties, cascades, disagreements, the tolerance), the
  captured boundaries; `NOT_COMPARABLE` (a later iteration) and
  `NOT_CAPTURED` (a layer the reference does not have) are listed and not
  counted.
- The `FWD_PASS` rows (it29, `--no-event-timing`): the megakernel's own
  clock; the median over the 28 lines of `fwd_pass.log`.

## The queue files

| File | Rows | Used at |
|---|---|---|
| `env/session/queue-h1.txt` | R1: two model compares at one iteration (the stack, the stack with the knob), one 2-layer compare | minute 18 |
| `env/session/queue-h2.txt` | R2: A30 B30 A31 B31 A32 B32 with compare and table, then A29 and B29 (`FWD_PASS`) | minute 21 |
| `env/session/queue-h5.txt` | R5: the 2-layer head at 50, 8 and 10 slices; the model row at 8 | minute 31 |
| `env/session/queue-h4.txt` | R4: the faulting configuration at 2 layers; the graph cut after `L0.o_proj` and after `L0.mla_merge_uv`; the halves control | minute 35 |
| `env/session/queue-h7.txt` | R4: the layer bisect at 3, 5, 9, 14 layers, only if `queue-h4`'s first row passed | minute 38 |
| `env/session/queue-h3.txt` | R3: the timing build's two 2-layer rows (F4's buffer form; the first row is the check) | minute 40 |
| `env/session/queue-h6.txt` | the optional sets: the finals with batch 4, the finals with 8 head events | minute 42 |

Every row runs `--final`; the rows are split on whitespace, a define is
`--runtime-flags=-DNAME`. `harness/tests/test_queue_files.py` parses every
row, names its run and builds its plan; `06-rehearsal.md` shows every row
expanded.

## Failure playbook (round 4's, plus what its session added)

| If you see | Do |
|---|---|
| a row that runs past 600 s | the watchdog ends it as FAIL (rc 124); read its log, drop its flag, move on |
| a queue to stop | `pgrep -f '^bash env/session/queue.sh run'` for the loop, then `vm.sh kill --all`; never `pkill -f` with a pattern that appears in the ssh command line |
| the pull says "logs only: staged" but runs came in | commit the record yourself right after the pull, before any other commit |
| a compare row FAIL on the model | read the classes: a route-log disagreement or a captured boundary FAIL is real; `NOT_COMPARABLE` and `NOT_CAPTURED` rows are not failures, and the it32 rows never compare the head |
| the reference guard raises | the gate's top-k weights are not the softmax at their ids: the checkpoint's routing config differs from the one read on the laptop (softmax, greedy, no normalisation, scaling 1.0); the reference stage fails and the finals still run |
| a `gv` row whose gaps equal the CK row's | the JIT reused an older build's headers: `L start setup`, then the row again |
| `no queue file` from a `queue` stage | the file was edited after the last push: `L push`, then the stage again |
| a repeated final overwrote its run directory | the queue moves an existing record directory aside before a rerun (`<name>.prev-<utc>`) |
| the delete check does not find the rate | the team page prints `Hourly Rate:       $0.00/hour` with several spaces; read it by hand |

## What is pulled back and committed

As round 4: `L pull` brings `env/hw/<date>/` (every run's `plan.json`,
`fleet_run_meta.json`, `fwd_pass.log`, `correctness_report.md` and `.json`,
`report_table.md`, `metrics.json`, `event_timing.json`; `kernel_tests_nt/`;
`ktime/`), `harness/ref/` JSON files (the route log with `w_all`) and the
logs; the record is committed right after the pull.

## What `07-final-numbers.md` needs from the session

- The final table: A's and B's three runs and their `FWD_PASS` rows with
  names, per-token times, ids and the compare verdicts, beside round 4's
  4,262 to 4,341 and round 3's 4,571 to 4,600.
- The compare's first green model rows: the route log's counts and the
  tie's gap against the tolerance.
- The head's events at 50, 8 and 10 and the 2-layer medians; the finals
  with 8 if they ran.
- R4's located operator (or the first faulting layer count) for MIN-36;
  R3's exec table if it ran; the merge's `ktime` line.
- The decisions: every DECIDE row's measured value and choice.
