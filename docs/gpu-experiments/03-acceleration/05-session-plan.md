# 05 - Session plan: two sessions on a 1x MI300X for $15

Written 2026-09-17 after the laptop preparation (`04-checklist.md`: O0 to
O8, I1 to I4, I6, the double-check). Every row below is one literal
command of `env/session/laptop.sh` (`L`), the text that decides PASS,
whether the row runs on its own (AUTO) or needs a decision (DECIDE, with
the rule and who decides), and what to do on FAIL. The exact commands the
scripts expand to are in `06-rehearsal.md`, generated in DRY mode by
`env/session/rehearse.sh`. The run log and the numbers of the sessions go
to `07-session-log.md` and `08-results.md` when they exist.

The shape is round 2's (`../02-validation/02-session-plan.md`). What is
new: every optimization is a flag of `run_fleet.py` that is off by
default, so a row that fails leaves the next row on the round-2 path; the
worker timing (I1) and the clock sampler (I6) run beside every timing row;
the ladder (I3) and the knobs (I4) are queues of their own; the suites run
three times (the plain, the streaming and the MFMA builds).

## Budget

| | |
|---|---|
| Balance | $20.23 (read after round 2's deletion, 2026-09-16; the gate of row C0) |
| Shape | 1x MI300X, $2.99 per hour, billed per minute (149 minutes cost $7.33 in round 2) |
| Hours | 6.7 |
| Session C | up to 2.5 hours ($7.48): the setup, the suites, the timings, the ladder, the knobs, the fusions, the probes and the streams |
| Session D | up to 2.5 hours ($7.48): the setup again, the MFMA attention, the prefetch, the remedy of Group A, the final number |
| Reserve | 1.7 hours ($5.27): a re-provisioning or a rebuild; a third short session if D ends early |

Running out of balance deletes the VM with everything on it, so the
record is pulled every 30 minutes and the balance is read at the
checkpoints. The image on GHCR (`fleet-amd-task:20260916`) is not used:
no stage runs inside it, and `setup` builds Fleet on the VM in about
eight minutes (440 s in round 2 with the image build overlapping), which
is cheaper than wiring a container path on the clock.

## Decisions to take before the session (the user)

| Decision | Proposed |
|---|---|
| Session lengths | 2.5 hours each, hard stop at minute 150 of VM time |
| Provisioning | `grab.sh` polls the list (round 2: 109 polls); the user is told before it starts, nothing is billed before the provision |
| The MFMA suites fail (C3) | the agent's rule: the round runs on the VALU attention; G7's rows are dropped, the standalone number is still read; the kernel is fixed on the laptop between the sessions |
| The remedy of Group A (G3, G4) | the agent applies the rule in the thresholds below and reports; a knob that passes the compare and cuts the gaps stays on for the rest of the session |
| A fusion fails its compare (G5) | that lever is off for the rest of the round; the later rows are run without it (the queue continues either way) |
| G7 early | if session C ends its queues before minute 130, the first two rows of `queue-d7.txt` run in C |
| Deletion of the VM | asked every time, at the end of each session; never on the agent's own judgment |
| Reporting, commits, pushes | round 2's protocol, unchanged (below) |

## Reporting protocol

- Before `grab.sh`: the balance and a one-line "polling for a VM now" to
  the user. Nothing is billed before the provision.
- After every stage row: the `PASS`/`FAIL` line of `env/logs/session.status`
  pasted verbatim, with the minute mark and the balance estimate.
- At every DECIDE row: the rule, the measured value, and the choice, before
  the next command runs.
- At the end: the last `L status`, the `Hourly Rate: $0.00/hour` line (read
  by hand from the team page: the rate has several spaces before the
  value), and the commit hash of the pulled record.

## Session C (up to 2.5 hours of VM time)

Every command is run from the repository root on the laptop. `L` stands
for `bash env/session/laptop.sh`. Status rows are read with `L status`;
the "PASS when" column is the row that must appear in it.

| Minute | Row | Command | PASS when | Mode | On FAIL |
|---|---|---|---|---|---|
| -5 | C0 | `L balance` | `Available Balance` about $20, `No virtual machines`, `Hourly Rate: $0.00/hour` | DECIDE (user: go) | a VM already listed: stop, nothing is provisioned |
| 0 | C0 | `bash env/session/grab.sh`, then `FULL=1 L push` | an address in `env/session/vm.ip`; the push (with the pristine fork) completes in about two minutes | AUTO | `grab.sh` gives up after an hour of polls: the user decides whether to wait |
| 2 | C1 | `L start download`; `L start setup`; `L start hw`; `L ssh "cd /home/hotaisle/metalOps && bash env/session/vm.sh preflight"` | `PASS download` (about a minute if the host has the model, up to ten if not); `PASS setup` (about 8 minutes); `PASS hw` or `skipped` | AUTO | setup: read `env/logs/setup.out`; the fix on the laptop, `L push`, `L start setup` again (an incremental build takes about a minute) |
| 11 | C2 | `L start checks`; then `L start reference` | `PASS checks` with 7 PASS lines; `PASS reference` (about 3 minutes; the compares need it) | AUTO | a check that passed in round 2: a machine difference, log it; reference FAIL: no compare can run, the queue guard says so |
| 15 | C3 | `L start kernels`; `L wait kernels`; `L start kernels nt`; `L wait kernels`; `L start kernels mfma` | 7 suites, 100 of 100, three times; the `mfma` suites are the gate of G7 | DECIDE (agent: the MFMA rule above) | a FAIL in the plain suites: the kernel that changed since round 2 is O1's router (its `h` row) or the copy's spin; `O1` off means no `--fuse-norm2` row; a FAIL only in `mfma`: G7 dropped |
| 19 | C4 | `L start ktime`; `L start ktime nt`; `L start ktime mfma` | `env/hw/<date>/ktime/ktime*.txt`: `TIME mla_attend ... mean_us=`, the merge, and `[SPIN] ... cycles= ticks=` per build | DECIDE (agent: threshold T1) | a build without a `TIME` line: its binary did not run; the row is repeated once |
| 23 | C5 | `L start queue env/session/queue-c2.txt` (G2: the round-2 baseline with the worker timing, 2 layers then the model) | `table=PASS compare=PASS` on both rows; `report_table.md` shows the per-class exec table and the clock row | DECIDE (agent: T2 and T3) | `compare=FAIL` on the model: the round-2 path itself regressed; stop and read the report before anything else |
| 35 | C6 | `L start queue env/session/queue-c3.txt` (G3: the ladder, 12 rows of about a minute) | `table=PASS` per row; the per-operator gap and the per-task exec in each `report_table.md` | DECIDE (agent: T4, the remedy) | a row that faults: the empty graph exercises the runtime alone; the fault goes to the log and the ladder continues (`continue` rows) |
| 55 | C7 | `L pull` (the 30-minute checkpoint); `L start queue env/session/queue-c4.txt` (G4: the knobs, 8 rows) | `compare=PASS` on every row that stays on; the gaps against row 1 | DECIDE (agent: T5) | a knob's `compare=FAIL`: the knob is off, its row is recorded as the cost it did not save |
| 80 | C8 | `L start queue env/session/queue-c5.txt` (G5: the fusions, one at a time, 2 layers then the model) | `compare=PASS` and `output_ids PASS` per lever; the event clock per token lower than the previous row | DECIDE (agent per lever, the rule above) | a lever's compare FAIL: off for the round; the queue continues with the next lever stacked on the ones that passed only if its row is edited (`queue_flag.py`) |
| 105 | C9 | `L start queue env/session/queue-c6.txt` (G6: the probes, the streaming loads) | `compare=PASS`; the probed operators' gaps with a one-task predecessor; `--nt-streams` on the model | DECIDE (agent: T6) | |
| 120 | C10 | `L pull`; `L balance` | balance above $13 (session D needs $7.48 plus the reserve) | DECIDE (agent: hard stop at $13) | below: stop the queue now |
| 130 | C11 | if the queues are done: the first two rows of `queue-d7.txt` (G7 early) | | AUTO (decided above) | |
| 150 | end | `L pull`; `git push`; then, after the user's yes, `L delete --yes`; `L balance` shows `Hourly Rate: $0.00/hour` | | DECIDE (user: the deletion, asked every time) | if the rate is not $0.00, check the TUI by hand |

Hard stops: at 150 minutes of VM time the queue is stopped, the record
pulled and the VM deleted whatever the state; at any point when the
balance shows less than $13 in session C or less than $4 in session D.

### Thresholds for the DECIDE rows (baseline round 2, `../02-validation/04-results.md`)

| Row | Quantity | Round 2 | Rule |
|---|---|---|---|
| T1, C4 | the attention grid standalone, cold cache (`ktime`): plain, `nt`, `mfma` | 34 us (plain) | `mfma` under 15 us: O7 is the kernel fix if the graph agrees (T2); `nt` within 2 us of plain: O6 is not a standalone lever; the `[SPIN]` MHz of the three builds agree within 2% |
| T2, C5 | the attention's exec cycles per task (`[TASK_TIME2] attend`) in microseconds at the spin's SCLK, against its event gap | gap 147.5 us; standalone 34 us | exec about 34 us: the time is around the task, the runtime (G3, G4 decide); exec about 140 us: the kernel itself in the graph (cold cache, contention), G7 is the fix |
| T3, C5 | the SCLK: the spin inside the graph against `ktime`'s spin and against `amd-smi` | 2,100 MHz max (D2) | a gap above 10% between the graph and standalone: the clock is part of the answer; the VF's `amd-smi` reading (131 MHz at idle in round 1) is recorded, not trusted |
| T4, C6 | the ladder: the per-operator gap at N = 1, 8, 40, 296 and the per-task exec of the copy | (new) | a per-operator gap above 10 us at N = 1, or a per-task cost above 1 us at N = 296: the runtime carries the boundary cost, G4's knobs and the laptop's remedy (a scheduler change) are the path; both below: the kernels carry it, G7 and the fusions are the path |
| T5, C7 | each knob: `compare=PASS` and the event gaps against the first row | (new) | a knob that passes and cuts the per-token time by more than 2% stays on for C8 to C11 and D; the poll sleep is chosen among 1, 8, 32, 127 by the same rule |
| T6, C9 | the probed gaps: `o_proj` after a one-task copy against 22 us; `qkva` against 4.6 us; the router against 3.6 us | 22, 4.6, 3.6 us | a probed gap that halves names the producer's spread as the cost, not the operator (A2 of the ideas); `--nt-streams` on the model below the G2 number by more than 1% stays on |
| G5 | each fusion on the model | 9.58 ms per token | each lever: `output_ids PASS` and a lower per-token time; the three together expected near 7.5 ms |

## Session D (up to 2.5 hours, after the laptop has acted on session C)

The VM is provisioned again (round 2's shape has no pause). Between the
sessions the laptop turns session C's answer into `queue-d7b.txt`: the
remedy of Group A (a knob kept on, a poll constant, a scheduler change),
and edits the flags of `queue-d7.txt` to `queue-d9.txt` to the levers
session C kept (`python3 env/session/queue_flag.py <queue> <flags> --in-place`).

| Minute | Row | Command | PASS when | Mode | On FAIL |
|---|---|---|---|---|---|
| 0 | D0 | `L balance`; `bash env/session/grab.sh`; `FULL=1 L push`; `L start download`; `L start setup`; `L start hw` | as C0 and C1 | DECIDE (user: go), then AUTO | |
| 11 | D1 | `L start checks`; `L start reference`; `L start kernels`; `L start kernels mfma`; `L start ktime mfma` | as C2 to C4 (the `mfma` suites again: the build is new) | AUTO | the MFMA suites fail on the fixed kernel: G7 dropped for good, D2 onward |
| 22 | D2 | `L start queue env/session/queue-d7.txt` (G7: the MFMA attention, 2 layers then the model, then with the fusions) | `compare=PASS`; the attention's gap and exec | DECIDE (agent: T7) | compare FAIL: O7 off, the VALU rows of C stand |
| 45 | D3 | `L start queue env/session/queue-d8.txt` (G8: the prefetch); then `L start tgcheck <the first row's run name>` | `PASS` from the task-graph check; `compare=PASS`; the event clock with and without the non-temporal weight loads | DECIDE (agent: T8) | the check FAIL: the side operators are mis-wired; O8 off, its rows recorded |
| 70 | D4 | `L pull`; `L start queue env/session/queue-d7b.txt` (G7b: the remedy written between the sessions) | the per-token time on the event clock | DECIDE (agent, the rule written with the queue) | |
| 100 | D5 | `L start queue env/session/queue-d9.txt` (G9: the final, three runs, the last with the compare) | `output_ids PASS`; three per-token times within 2% | AUTO | |
| 120 | D6 | `L pull`; `L balance` | balance above $4 | DECIDE (agent: hard stop) | |
| 150 | end | `L pull`; `git push`; after the user's yes, `L delete --yes`; the rate is $0.00 | | DECIDE (user: the deletion) | |

| Row | Quantity | Rule |
|---|---|---|
| T7, D2 | the attention's event gap with `--mfma-attend` against C's number (147.5 us in round 2) | under 60 us: O7 stays on; between: on if the per-token time is lower; above: off (the standalone number of C4 is then the only gain to report) |
| T8, D3 | the per-token time with `--prefetch` against the same flags without it | lower by more than 1% with the compare PASS: on; else off; the second row (with `--nt-weights` on the consuming linears) decides whether the two can coexist |
| G9 | the final per-token time, event clock | the number against 4.5 ms; the three clocks in `report_table.md` beside it |

## The queue files

| File | Rows | Used at |
|---|---|---|
| `env/session/queue-c2.txt` | G2: the round-2 baseline with the worker timing, 2 layers and the model | C5 |
| `env/session/queue-c3.txt` | G3: the ladder, N in 1, 8, 40, 296 by M in 10, 100, 300, the first four with the spin | C6 |
| `env/session/queue-c4.txt` | G4: the knobs, one per row, the sleep at 8, 32, 127, each with the compare | C7 |
| `env/session/queue-c5.txt` | G5: the fusions stacked one at a time, 2 layers then the model | C8 |
| `env/session/queue-c6.txt` | G6: three probes; the streaming loads on 2 layers and the model | C9 |
| `env/session/queue-d7.txt` | G7: the MFMA attention alone and with the fusions | D2 (or C11) |
| `env/session/queue-d8.txt` | G8: the prefetch without and with the non-temporal weight loads | D3 |
| `env/session/queue-d7b.txt` | G7b: written between the sessions | D4 |
| `env/session/queue-d9.txt` | G9: the final, three times | D5 |

One `run_fleet.py` argument line per row with the optional trailing words
`compare`, `table`, `measure`, `continue`; a row is split on whitespace
without shell quoting, so a define travels as `--runtime-flags=-DNAME`
and never in quotes. The guards of `queue.sh` fail a row before it runs
when `--iters` is above 32, `--debug` has more than one iteration, a
compare has no reference tensors, or a measure has no profiler; an empty
graph needs no reference. `harness/tests/test_queue_files.py` parses
every row of every file. `06-rehearsal.md` shows every row expanded.

## Helpers (so nothing is typed by hand on the clock)

| Command | Use |
|---|---|
| `bash env/session/grab.sh` | polls the provisioning list every 18 s and provisions the first 1x MI300X; the log is `env/logs/grab.log` |
| `L wait <stage> [min]` | polls `vm.sh check` every 30 s until the stage's PASS or FAIL row |
| `L report [--balance]` | the status message of the protocol |
| `L start kernels [nt\|mfma]` | the suites against `kernel_tests`, `kernel_tests_nt` or `kernel_tests_mfma` (with its `_debug` twin for the scores suite); results under `fleet/tasks/results[_variant]` and the record |
| `L start ktime [nt\|mfma] [launches] [copies]` | the standalone attention and merge grids (`KT_TIME`, `KT_COLD` over 27 copies of the cache) and the `[SPIN]` line of that build, into `env/hw/<date>/ktime/` |
| `L start tgcheck <run name>` | the side operators' wiring in the run's `task_graph_rank0.json` (`fleet/task_graph_check.py`); the verdict into the run's record |
| `L ssh "cd /home/hotaisle/metalOps && rm -f fleet/tasks/build/kernel_tests*"` | after a kernel edit pushed mid-session: the `kernels` stage does not rebuild an existing binary |
| `L ssh "... vm.sh kill <pattern>\|--all"` | stops a hung graph run by anchored pid and writes a `KILLED` row |
| `python3 env/session/queue_flag.py <queue> <flags> --in-place` | a lever that passed into the rows of a later queue (idempotent, keywords and comments kept) |
| `run_fleet.py --worker-timing`, `--probe-before LABEL`, `--runtime-flags=-DNAME`, `--graph empty --ops M --tasks N [--spin S]`, `--fuse-norm2`, `--fuse-silu`, `--fuse-norm1`, `--nt-streams`, `--mfma-attend`, `--prefetch` | the flags of this round; every one off by default (`03-local-preparation.md`) |
| `KT_SPIN=1000 fleet/tasks/build/kernel_tests copy <dir>` | the spin line by hand (`fleet/tasks/README.md`) |

## Failure playbook (rounds 1 and 2, one line each)

| If you see | Do |
|---|---|
| `L push` after the first push of a session overwrote the VM's fork | it cannot now: `push` excludes `repos/` unless `FULL=1`; if the JIT loses a task type anyway, `L start setup` re-applies the patches in about a minute |
| `L pull` changed a tracked file that is not the record | the pull brings only logs and the record; if a test file moved, `git checkout` it before the commit |
| the `kernels` stage says PASS for a binary built before a kernel edit | it does not rebuild an existing binary: remove `fleet/tasks/build/kernel_tests*` on the VM first |
| a `measure` row fails on the profiler | rocprofv3 cannot attach to the torch wheel's bundled runtime; the counters need the suite binary; no `measure` rows this round |
| `setup.sh` fails on a patch | the three patches apply in order `gfx942`, `sched_xcd`, `new_tasks` on the pristine commit; `env/preflight.sh` checks that on the laptop before every push |
| the delete check does not find the rate | the team page prints `Hourly Rate:       $0.00/hour` with several spaces; read it by hand |
| a queue row FAIL with `fault=1` before any `FWD_PASS` | the round-2 fault came back with a new flag: the row's flag is off for the round; the address record is in `fleet_run_meta.json` |
| two graph runs at once | never: the JIT directory is shared; the queue serialises; do not start a second queue or a `ktime` beside a queue |
| an ssh command hangs or the session drops | the scripts detach with `setsid nohup`; kill by pid from `pgrep -f "^python harness/run_fleet.py"` |
| `--iters 64` asserts on the cos table | 32 is the ceiling; the guard refuses |
| the `[SPIN]` lines swamp `fwd_pass.log` | only the first operator's tasks print; a ladder row with N = 296 prints 296 lines once |

## What is pulled back and committed

`L pull` does it: `env/logs/*.out` and the status files (into
`env/hw/<date>/logs/`); `env/hw/<date>/` (`runs/<name>/` for every queue
row with `plan.json`, `wall.json`, `fleet_run_meta.json`, `fwd_pass.log`
with the timing and spin lines, `clock.log`, `correctness_report.md` and
`.json`, `report_table.md`, `metrics.json`, `event_timing.json`,
`task_graph_check.txt`; `kernel_tests[_variant]/`; `ktime/`);
`harness/ref/` JSON files; `env/check_day1.log`. Then one commit on the
branch. Not pulled: the tensors, the build directories (the task-graph
JSON stays on the VM; its verdict is pulled), files above 400 KB.

## Fallbacks decided now

- The MFMA suites fail: the round runs on the VALU attention; the
  standalone number is still read; O7 is laptop work between the sessions.
- A fusion fails its compare: off for the round, the next levers stack on
  the ones that passed.
- The ladder names the runtime and no knob helps: G7b is a scheduler
  change written between the sessions, or, if none is ready, session D
  spends its time on G7 and G8 and the report names the runtime's
  boundary cost as the remainder.
- The balance is short before session D: session D is cut to D1, D2 and
  D5, about 1.2 hours.
- `grab.sh` finds no VM within an hour: the session is moved; nothing is
  billed.

## Expected outputs (round 2, to recognise a deviation at once)

| Stage or row | Expected |
|---|---|
| `checks` | 7 PASS lines (SPX+NPS1; `import mirage`; the schedulers on their XCDs; the agent-scope fences present; the CK FMHA negative; the counter names; 304 CUs) |
| `reference` | 32 ids, calibration floors as in round 2 (router 3.59e-3, scores 2.61e-3, logits 4.1e-2, norm 4.5e-2, layer 4.3e-3) |
| `kernels` | 7 suites, 100 of 100: `mla_prep`, `mla_attend`, `mla_merge_uv`, `moe_router`, `copy`, `mla_attend_scores`, `mla_attend_splits`; the same for `nt`; `mfma` is the question |
| `ktime` | plain: the attention grid 34 us cold, the merge 11.5 us |
| a 2-layer run with `compare` | all 16 boundaries PASS, the route log PASS |
| 27 layers with the head, 32 iterations, `--tile-linears --nt-weights` | 9.58 ms per token on the event clock; per operator in a MoE layer: attention 147.5 us, merge 46 to 59, o_proj 22, silu 41, combine 20, norms 13.6, prep 13.6, w13 20, qkva 4.6, router 3.6, down 5.6 (`01-ideas.md`) |
| the same with the three fusions | 246 operators instead of 326; expected near 7.5 ms |
| the ladder at N = 1 | the per-operator gap is the runtime's floor for one task; round 2's one-task operators cost 3.6 to 13.6 us |
| a session's cost | about $7.5 for 2.5 hours |
