# 05 - Session plan: one session on a 1x MI300X, up to four hours, gains first

Written 2026-09-17 after the laptop preparation (`04-checklist.md`: O0 to
O8, I1 to I4, I6, the double-check) and revised the same day on the user's
decisions (below): one session instead of two, the gains before the
diagnostics, the agent acting on every written rule and reporting as it
goes, the deletion asked. Every row is one literal command of
`env/session/laptop.sh` (`L`), the text that decides PASS, whether the row
runs on its own (AUTO) or applies a rule (RULE, with the rule), and what to
do on FAIL. The exact commands the scripts expand to are in
`06-rehearsal.md`, generated in DRY mode by `env/session/rehearse.sh`. The
run log and the numbers go to `07-session-log.md` and `08-results.md`.

What is new against round 2's plan (`../02-validation/02-session-plan.md`):
every optimization is a flag of `run_fleet.py` that is off by default, so
a row that fails leaves the next row on the round-2 path; the worker timing
(I1) and the clock sampler (I6) run beside every timing row; the queues
are short (round 2's record: a 2-layer row took about 30 s of wall time, a
27-layer row about 50 s, the JIT included), so the whole set is under an
hour of queue time and the session's length is the setup plus the queues
plus the reruns the rules ask for.

## Budget

| | |
|---|---|
| Balance | $20.13 (read from the team page on 2026-09-17 at the gate, no VM, rate $0.00) |
| Shape | 1x MI300X, $2.99 per hour, billed per minute (149 minutes cost $7.33 in round 2) |
| Hours | 6.7 |
| The session | up to 4.0 hours ($11.96), hard stop at minute 240 of VM time; the plan below needs about 100 minutes, the rest is for reruns and the diagnostics |
| Reserve | 2.7 hours ($8.17): a re-provisioning, a rebuild, or a short second session for a remedy written afterwards |

Running out of balance deletes the VM with everything on it, so the
record is pulled every 30 minutes and the balance is read at the
checkpoints; the session stops when the balance shows less than $8. The
image on GHCR (`fleet-amd-task:20260916`) is not used: no stage runs
inside it, and `setup` builds Fleet on the VM in about eight minutes.

## Decisions taken by the user on 2026-09-17

| Decision | Choice |
|---|---|
| Sessions | one, up to 4 hours of VM time; no laptop work in between; the remedy of Group A, if the diagnostics name one, is next-round work |
| Order | gains first: the baseline with the worker timing, the fusions, the MFMA attention, the prefetch, the final number; then the diagnostics (the ladder, the knobs, the probes) as time allows |
| Autonomy | the agent applies every written rule and reports each decision as it goes; it edits queue rows mid-session with `queue_flag.py` (a lever that failed its compare is removed from the later rows; a knob that helped is added to the final rerun); nothing waits for the user except the deletion |
| Provisioning | `grab.sh` polls the list; the user is told when the polling starts, and nothing is billed before the provision |
| Deletion | asked every time; the agent pulls the record and pushes the branch first |
| The MFMA suites fail | the round runs on the VALU attention; G7's rows are dropped; the standalone number is still read |
| A fusion fails its compare | that lever is off for the round; the stacked rows are edited before they run |
| I5, vLLM | skipped; the 4.5 ms figure stays the recruiter's |

## Reporting protocol

- Before `grab.sh`: the balance and a one-line "polling for a VM now".
- After every stage row: the `PASS`/`FAIL` line of `env/logs/session.status`
  pasted verbatim, with the minute mark and the balance estimate.
- At every RULE row: the rule, the measured value, and the choice taken,
  in the report that follows; the next command does not wait.
- At the end: the last `L status`, the `Hourly Rate: $0.00/hour` line (read
  by hand from the team page: the rate has several spaces before the
  value), and the commit hash of the pulled record.

## The session (up to 4.0 hours of VM time)

Every command is run from the repository root on the laptop. `L` stands
for `bash env/session/laptop.sh`. Status rows are read with `L status`;
the "PASS when" column is the row that must appear in it. Minute marks
are from the provision; the queue rows' durations are round 2's.

| Minute | Row | Command | PASS when | Mode | On FAIL |
|---|---|---|---|---|---|
| -5 | S0 | `L balance` | `Available Balance` about $20, `No virtual machines`, `Hourly Rate: $0.00/hour` | AUTO (the go was given with the plan) | a VM already listed: stop, nothing is provisioned |
| 0 | S0 | `bash env/session/grab.sh`, then `FULL=1 L push` | an address in `env/session/vm.ip`; the push (with the pristine fork) completes in about two minutes | AUTO | `grab.sh` gives up after an hour of polls: the user is told; nothing is billed |
| 2 | S1 | `L start download`; `L start setup`; `L start hw`; `L ssh "cd /home/hotaisle/metalOps && bash env/session/vm.sh preflight"` | `PASS download` (about a minute if the host has the model, up to ten if not); `PASS setup` (about 8 minutes); `PASS hw` or `skipped` | AUTO | setup: read `env/logs/setup.out`; the fix on the laptop, `L push`, `L start setup` again (an incremental build takes about a minute) |
| 11 | S2 | `L start checks`; `L wait checks 5`; `L start reference` | `PASS checks` with 7 PASS lines; `PASS reference` (about 3 minutes; the compares need it) | AUTO | a check that passed in round 2: a machine difference, log it; reference FAIL: no compare can run, the queue guard says so |
| 15 | S3 | `L start kernels`; `L wait kernels 15`; `L start kernels nt`; `L wait kernels 5`; `L start kernels mfma`; `L wait kernels 5` | 7 suites, 100 of 100, three times (the first call builds the five suite binaries, about 6 minutes); the `mfma` suites are the gate of S7 | RULE: a FAIL only in `mfma` drops S7 (the VALU attention stays, its standalone number is still read in S4); a FAIL in the plain suites names the kernel that changed since round 2 (O1's router row `h`, the copy's spin): its flag is off for the round | |
| 24 | S4 | `L start ktime`; `L wait ktime 5`; `L start ktime nt`; `L wait ktime 5`; `L start ktime mfma`; `L wait ktime 5` | `env/hw/<date>/ktime/ktime*.txt`: `TIME mla_attend ... mean_us=`, the merge, and `[SPIN] ... cycles= ticks=` per build | RULE T1 | a build without a `TIME` line: its binary did not run; the row is repeated once |
| 28 | S5 | `L start queue env/session/queue-c2.txt` (G2: the round-2 baseline with the worker timing, 2 layers then the model; about 3 minutes) | `table=PASS compare=PASS` on both rows; `report_table.md` shows the per-class exec table and the clock row; the model's per-token time is the reference of every later row | RULE T2, T3 | `compare=FAIL` on the model: the round-2 path itself regressed on this build; the session continues on the diagnostics only and the user is told |
| 32 | S6 | `L start queue env/session/queue-c5.txt` (G5: the three fusions, each alone on 2 layers and the model, then the three together; 8 rows, about 7 minutes) | `compare=PASS` and `output_ids PASS` per row; the event clock per token against S5 | RULE per lever: on if its compare passes and its model row is faster than S5's; a lever that fails is removed from the later queues with `python3 env/session/queue_flag.py <queue> --remove <flag> --in-place` (`queue-d7.txt`, `queue-d8.txt`, `queue-d9.txt`) before they run | |
| 40 | S7 | only if the `mfma` suites passed: `L start queue env/session/queue-d7.txt` (G7: the MFMA attention alone and with the fusions; 4 rows, about 4 minutes) | `compare=PASS`; the attention's event gap and exec cycles | RULE T7: on if the gap is under 60 us or the per-token time is lower than the same row without it | compare FAIL: O7 off; `--mfma-attend` removed from `queue-d9.txt` |
| 45 | S8 | `L start queue env/session/queue-d8.txt` (G8: the prefetch without and with the non-temporal weight loads; 3 rows, about 4 minutes); then `L start tgcheck L2_it32_tile_fn1_fn2_fs_pf_wt` (the first row's run; its name follows the levers kept) | `PASS` from the task-graph check; `compare=PASS`; the event clock against the same flags without `--prefetch` | RULE T8: on if the check passes, the compare passes and the model row is faster by more than 1%; the second and third rows decide whether it coexists with `--nt-weights`; if on, `--prefetch` is added to `queue-d9.txt` | the check FAIL: the side operators are mis-wired; O8 off, its rows recorded |
| 50 | S9 | `L pull` (the checkpoint); `L start queue env/session/queue-d9.txt` (G9: the final, every lever kept, three runs, the last with the compare; about 4 minutes) | `output_ids PASS`; three per-token times within 2%; the three clocks in `report_table.md` | AUTO; the number against 4.5 ms is reported with the levers it carries | a spread above 2%: three more runs |
| 56 | S10 | `L start queue env/session/queue-c3.txt` (G3: the ladder, 12 rows, about 10 minutes) | `table=PASS` per row; the per-operator gap and the per-task exec in each `report_table.md` | RULE T4: the attribution, written into the report; no lever changes | a row that faults: the empty graph exercises the runtime alone; the fault goes to the log and the ladder continues |
| 68 | S11 | `L start queue env/session/queue-c4.txt` (G4: the knobs, 8 rows, about 7 minutes) | `compare=PASS` on every row that stays on; the gaps against row 1 | RULE T5: a knob that passes and cuts the per-token time by more than 2% is added to a rerun of the final (S13) | a knob's `compare=FAIL`: off; its row is recorded as the cost it did not save |
| 76 | S12 | `L start queue env/session/queue-c6.txt` (G6: the probes, the streaming loads; 5 rows, about 4 minutes) | `compare=PASS`; the probed operators' gaps with a one-task predecessor; `--nt-streams` on the model | RULE T6: `--nt-streams` on if its model row is faster than S5's by more than 1%; added to S13 | |
| 82 | S13 | if S11 or S12 kept anything: `queue_flag.py env/session/queue-d9.txt <the flags> --in-place`, then `L start queue env/session/queue-d9.txt` again (the final with the knobs and streams) | as S9 | AUTO | |
| 90 | S14 | `L pull`; `L report --balance` | the record committed; balance above $8 | RULE: below $8 the session ends now | |
| 90 to 235 | S15 | reruns the rules asked for; nothing new is started after minute 220 | | AUTO | |
| end | end | `L pull`; `git push`; then, after the user's yes, `L delete --yes`; `L balance` shows `Hourly Rate: $0.00/hour` | | DECIDE (user: the deletion, asked every time) | if the rate is not $0.00, check the TUI by hand |

Hard stops: at 240 minutes of VM time the queue is stopped, the record
pulled and the VM deleted whatever the state; at any point when the
balance shows less than $8.

### Thresholds for the RULE rows (baseline round 2, `../02-validation/04-results.md`)

| Rule | Row | Quantity | Round 2 | Rule |
|---|---|---|---|---|
| T1 | S4 | the attention grid standalone, cold cache (`ktime`): plain, `nt`, `mfma` | 34 us (plain) | `mfma` under 15 us: O7 is the kernel fix if the graph agrees (T7); `nt` within 2 us of plain: O6 is not a standalone lever; the `[SPIN]` MHz of the three builds agree within 2% |
| T2 | S5 | the attention's exec cycles per task (`[TASK_TIME2] attend`) in microseconds at the spin's SCLK, against its event gap | gap 147.5 us; standalone 34 us | exec about 34 us: the time is around the task, the runtime (the ladder and the knobs attribute it); exec about 140 us: the kernel itself in the graph, O7 is the fix |
| T3 | S5 | the SCLK: the spin inside the graph against `ktime`'s spin and against `amd-smi` | 2,100 MHz max | a gap above 10% between the graph and standalone: the clock is part of the answer; the VF's `amd-smi` reading (131 MHz at idle in round 1) is recorded, not trusted |
| G5 | S6 | each fusion on the model | 9.58 ms per token | on if `output_ids PASS` and the per-token time is below S5's; the three together expected near 7.5 ms |
| T7 | S7 | the attention's event gap with `--mfma-attend` against S5's | 147.5 us | under 60 us: on; between: on if the per-token time is lower; above: off (the standalone number of S4 is then the only gain to report) |
| T8 | S8 | the per-token time with `--prefetch` against the same flags without it | (new) | lower by more than 1% with the check and the compare PASS: on; the row with `--nt-weights` decides whether the two coexist |
| T4 | S10 | the ladder: the per-operator gap at N = 1, 8, 40, 296 and the per-task exec of the copy | (new) | a per-operator gap above 10 us at N = 1, or a per-task cost above 1 us at N = 296: the runtime carries the boundary cost (the knobs and a scheduler change are the path, the latter next-round work); both below: the kernels carry it |
| T5 | S11 | each knob: `compare=PASS` and the event gaps against the first row | (new) | a knob that passes and cuts the per-token time by more than 2% goes into the final rerun; the poll sleep is chosen among 1, 8, 32, 127 by the same rule |
| T6 | S12 | the probed gaps: `o_proj` after a one-task copy against 22 us; `qkva` against 4.6 us; the router against 3.6 us; `--nt-streams` on the model | 22, 4.6, 3.6 us | a probed gap that halves names the producer's spread as the cost, not the operator; `--nt-streams` on if faster by more than 1% |
| G9 | S9, S13 | the final per-token time, event clock | 9.58 ms | the number against 4.5 ms; the three clocks in `report_table.md` beside it |

## The queue files

| File | Rows | Used at |
|---|---|---|
| `env/session/queue-c2.txt` | G2: the round-2 baseline with the worker timing, 2 layers and the model | S5 |
| `env/session/queue-c5.txt` | G5: each fusion alone (2 layers, the model), then the three together | S6 |
| `env/session/queue-d7.txt` | G7: the MFMA attention alone and with the fusions | S7 |
| `env/session/queue-d8.txt` | G8: the prefetch without and with the non-temporal weight loads | S8 |
| `env/session/queue-d9.txt` | G9: the final, three times; rerun at S13 with the knobs and streams | S9, S13 |
| `env/session/queue-c3.txt` | G3: the ladder, N in 1, 8, 40, 296 by M in 10, 100, 300, the first four with the spin | S10 |
| `env/session/queue-c4.txt` | G4: the knobs, one per row, the sleep at 8, 32, 127, each with the compare | S11 |
| `env/session/queue-c6.txt` | G6: three probes; the streaming loads on 2 layers and the model | S12 |

The names keep the letters of the two-session draft; the order above is
the order they run. One `run_fleet.py` argument line per row with the
optional trailing words `compare`, `table`, `measure`, `continue`; a row
is split on whitespace without shell quoting, so a define travels as
`--runtime-flags=-DNAME` and never in quotes. The guards of `queue.sh`
fail a row before it runs when `--iters` is above 32, `--debug` has more
than one iteration, a compare has no reference tensors, or a measure has
no profiler; an empty graph needs no reference.
`harness/tests/test_queue_files.py` parses every row of every file and
tests the mid-session edit. `06-rehearsal.md` shows every row expanded.

## Helpers (so nothing is typed by hand on the clock)

| Command | Use |
|---|---|
| `bash env/session/grab.sh` | polls the provisioning list every 18 s and provisions the first 1x MI300X; the log is `env/logs/grab.log` |
| `L wait <stage> [min]` | polls `vm.sh check` every 30 s until the stage's PASS or FAIL row |
| `L report [--balance]` | the status message of the protocol |
| `L start kernels [nt\|mfma]` | the suites against `kernel_tests`, `kernel_tests_nt` or `kernel_tests_mfma` (with its `_debug` twin for the scores suite); results under `fleet/tasks/results[_variant]` and the record |
| `L start ktime [nt\|mfma] [launches] [copies]` | the standalone attention and merge grids (`KT_TIME`, `KT_COLD` over 27 copies of the cache) and the `[SPIN]` line of that build, into `env/hw/<date>/ktime/` |
| `L start tgcheck <run name>` | the side operators' wiring in the run's `task_graph_rank0.json` (`fleet/task_graph_check.py`); the verdict into the run's record |
| `python3 env/session/queue_flag.py <queue> [--remove] <flags> --in-place` | a lever out of, or a knob into, the rows of a later queue (idempotent, keywords and comments kept) |
| `L ssh "cd /home/hotaisle/metalOps && rm -f fleet/tasks/build/kernel_tests*"` | after a kernel edit pushed mid-session: the `kernels` stage does not rebuild an existing binary |
| `L ssh "... vm.sh kill <pattern>\|--all"` | stops a hung graph run by anchored pid and writes a `KILLED` row |
| `run_fleet.py --worker-timing`, `--probe-before LABEL`, `--runtime-flags=-DNAME`, `--graph empty --ops M --tasks N [--spin S]`, `--fuse-norm2`, `--fuse-silu`, `--fuse-norm1`, `--nt-streams`, `--mfma-attend`, `--prefetch` | the flags of this round; every one off by default (`03-local-preparation.md`) |
| `KT_SPIN=1000 fleet/tasks/build/kernel_tests copy <dir>` | the spin line by hand (`fleet/tasks/README.md`) |

## Failure playbook (rounds 1 and 2, one line each)

| If you see | Do |
|---|---|
| `L push` after the first push of a session overwrote the VM's fork | it cannot now: `push` excludes `repos/` unless `FULL=1`; if the JIT loses a task type anyway, `L start setup` re-applies the patches in about a minute |
| `L pull` changed a tracked file that is not the record | the pull brings only logs and the record; if a test file moved, `git checkout` it before the commit |
| the `kernels` stage says PASS for a binary built before a kernel edit | it does not rebuild an existing binary: remove `fleet/tasks/build/kernel_tests*` on the VM first |
| a `measure` row fails on the profiler | rocprofv3 cannot attach to the torch wheel's bundled runtime; no `measure` rows this round |
| `setup.sh` fails on a patch | the three patches apply in order `gfx942`, `sched_xcd`, `new_tasks` on the pristine commit; `env/preflight.sh` checks that on the laptop before every push |
| the delete check does not find the rate | the team page prints `Hourly Rate:       $0.00/hour` with several spaces; read it by hand |
| a queue row FAIL with `fault=1` before any `FWD_PASS` | the round-2 fault came back with a new flag: the row's flag is off for the round; the address record is in `fleet_run_meta.json` |
| two graph runs at once | never: the JIT directory is shared; the queue serialises; no `ktime` beside a queue |
| an ssh command hangs or the session drops | the scripts detach with `setsid nohup`; kill by pid from `pgrep -f "^python harness/run_fleet.py"` |
| `--iters 64` asserts on the cos table | 32 is the ceiling; the guard refuses |
| the `[SPIN]` lines swamp `fwd_pass.log` | only the first operator's tasks print; a ladder row with N = 296 prints 296 lines once |
| a queue row with a quoted define | the row is split on whitespace: `--runtime-flags=-DNAME`, no quotes (the dry run of `queue-c4.txt` found it) |

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
  standalone number is still read; O7 is next-round work.
- A fusion fails its compare: off for the round, removed from the later
  queues before they run; the others stand.
- The baseline row itself fails its compare (S5): the round-2 path
  regressed on this build; the levers are not measured on a broken
  baseline; the diagnostics run and the user is told.
- The ladder names the runtime and no knob helps: the report names the
  runtime's boundary cost as the remainder; a scheduler change is
  next-round work.
- The balance is short: the session ends at $8 with what it has; the
  final number is in hand by minute 60 in the plan above.
- `grab.sh` finds no VM within an hour: the user is told; nothing is
  billed.

## Expected outputs (round 2, to recognise a deviation at once)

| Stage or row | Expected |
|---|---|
| `checks` | 7 PASS lines (SPX+NPS1; `import mirage`; the schedulers on their XCDs; the agent-scope fences present; the CK FMHA negative; the counter names; 304 CUs) |
| `reference` | 32 ids, calibration floors as in round 2 (router 3.59e-3, scores 2.61e-3, logits 4.1e-2, norm 4.5e-2, layer 4.3e-3) |
| `kernels` | 7 suites, 100 of 100: `mla_prep`, `mla_attend`, `mla_merge_uv`, `moe_router`, `copy`, `mla_attend_scores`, `mla_attend_splits`; the same for `nt`; `mfma` is the question |
| `ktime` | plain: the attention grid 34 us cold, the merge 11.5 us |
| a 2-layer row | about 30 s of wall time; all 16 boundaries PASS with `compare`, the route log PASS |
| a 27-layer row with the head, 32 iterations, `--tile-linears --nt-weights` | about 50 s of wall time; 9.58 ms per token on the event clock; per operator in a MoE layer: attention 147.5 us, merge 46 to 59, o_proj 22, silu 41, combine 20, norms 13.6, prep 13.6, w13 20, qkva 4.6, router 3.6, down 5.6 (`01-ideas.md`) |
| the same with the three fusions | 246 operators instead of 326; expected near 7.5 ms |
| the ladder at N = 1 | the per-operator gap is the runtime's floor for one task; round 2's one-task operators cost 3.6 to 13.6 us |
| the session's cost | about $5 for the plan's 100 minutes; $12 at the hard stop |
