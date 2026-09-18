# 07 - Session plan: one session on a 1x MI300X, the kernels first, the router and merge after

Written 2026-09-18 after the laptop preparation (`06-checklist.md`: L1 to
L8, N1 to N6; the double-check and the final check). One session, as
round 3's (`../03-acceleration/05-session-plan.md`): every row is one
literal command of `env/session/laptop.sh` (`L`), the text that decides
PASS, whether the row runs on its own (AUTO), applies a written rule
(RULE) or reports a decision (DECIDE), and what to do on FAIL. The exact
commands the scripts expand to are in `08-rehearsal.md`, generated in DRY
mode by `env/session/rehearse.sh`. The run log and the numbers go to
`09-session-log.md` and `10-results.md`.

What is new against round 3's plan: the levers are kernels of our own
behind flags of `run_fleet.py` (`--gemv-linears`, `--gemv-w13`,
`--router-tasks`, `--merge-tasks`, `--merge-oproj`) and two forms that
are the pushed header itself (the w2 GEMV multiply under `--fuse-silu`,
with the CK multiply one define away; the router and the merge one level
deeper), so every graph row of this round carries the round-3 stack and
`--nt-streams` (L1c: the plain build waits `vmcnt(0)` per batch); the
first decision (G0, the batch constant) is taken on standalone timings
before any graph row; the host is fresh, so the setup is about 12 minutes.

## Budget

| | |
|---|---|
| Balance | $13.01 (read from the team page after round 3; no VM, rate $0.00) |
| Shape | 1x MI300X, $2.99 per hour, billed per minute |
| Hours | 4.3 |
| The session | up to 3.5 hours ($10.47), hard stop at minute 210 of VM time; the plan below needs about 100 minutes for the rows that decide (G0 to G7, H2 to H6) and 30 more for G8 and G9 |
| Reserve | $2.50: a re-provisioning if the first host fails its setup |

The record is pulled at the checkpoints (S9, S11) and at the end; the
session stops when the balance shows less than $3.

## Decisions to confirm before the VM

| Decision | Proposed |
|---|---|
| Sessions | one, up to 3.5 hours of VM time; the router and merge rows in the same session after the GEMV rows (`04`) |
| Order | G0 (the suites, the standalone times, the batch constant), G1 and G2 (the linears, the grid), G3 and G4 (w2, w13), H2 and H4 (the router in four tasks, the merge as regular tasks), H5 (the o_proj fold), G5 and G6 (the stream probe, the head), the finals (G7 = H6), then G9 and G8 as time allows |
| Autonomy | the agent applies every RULE and reports every DECIDE row as it goes; a lever that failed its compare is removed from the later queue files with `queue_flag.py` before they run; nothing waits for the user except the deletion |
| Provisioning | `grab.sh` polls the list; the user is told when the polling starts; nothing is billed before the provision |
| Deletion | asked every time; the record pulled and the branch pushed first |
| The batch constant | G0's `ktime` decides 4, 8 or 16 rows per batch for the GEMV linear and the router (8 if within noise); a constant other than 8 means one `L push` of the header with the define and `L start setup` (about a minute) before the graph rows |
| A GEMV suite row fails | the kernel is off for the round: its flag is removed from every later file (`--gemv-linears` from f2 to f6 and g1, g2; `--gemv-w13` likewise); the round-3 build runs the router and merge rows |
| `--merge-oproj` fails its compare | off; the finals run with `--merge-tasks` (H4's better half count) if H4 passed, else with the stock merge |
| The fence knobs | not run: the counter forms rely on the runtime's fences (`run_fleet.py` refuses the pair); round 3 measured them on the stock forms |

## Reporting protocol

- Before `grab.sh`: the balance and a one-line "polling for a VM now".
- After every stage row: the `PASS`/`FAIL` line of `env/logs/session.status`
  pasted verbatim, with the minute mark and the balance estimate.
- At every RULE and DECIDE row: the rule, the measured value, and the
  choice taken, in the report that follows; the next command does not wait.
- At the end: the last `L status`, the `Hourly Rate: $0.00/hour` line (read
  by hand from the team page), and the commit hash of the pulled record.

## The session (up to 3.5 hours of VM time)

Every command is run from the repository root on the laptop. `L` stands
for `bash env/session/laptop.sh`. Status rows are read with `L status`;
the "PASS when" column is the row that must appear in it. Minute marks
are from the provision; a 2-layer graph row is about 2 minutes (the JIT
rebuilds per flag set), a model row about 1.5.

| Minute | Row | Command | PASS when | Mode | On FAIL |
|---|---|---|---|---|---|
| -5 | S0 | `L balance` | `Available Balance` about $13, `No virtual machines`, `Hourly Rate: $0.00/hour` | AUTO (the go is given with this plan) | a VM already listed: stop, nothing is provisioned |
| 0 | S0 | `bash env/session/grab.sh`, then `FULL=1 L push` | an address in `env/session/vm.ip`; the push (with the pristine fork and the round-4 patches) completes in about two minutes | AUTO | `grab.sh` gives up after an hour of polls: the user is told; nothing is billed |
| 2 | S1 | `L start download`; `L start setup`; `L start hw`; `L ssh "cd /home/hotaisle/metalOps && bash env/session/vm.sh preflight"` | `PASS download` (a fresh host: up to ten minutes); `PASS setup` (about 8 minutes); `PASS hw` | AUTO | setup: read `env/logs/setup.out`; the fix on the laptop, `L push`, `L start setup` again |
| 14 | S2 | `L start checks`; `L wait checks 5`; `L start reference` | `PASS checks` with 7 PASS lines; `PASS reference` (about 3 minutes; the compares need it) | AUTO | reference FAIL: no compare can run, the queue guard says so |
| 18 | S3 | `L start kernels`; `L wait kernels 20`; `L start kernels nt`; `L wait kernels 8` | every suite 100 of 100 on both builds: round 3's nine, and `linear_gemv`, `linear_gemv_norm`, `linear_gemv_res`, `moe_router4`, `mla_merge_uv_tile`, `mla_merge_uv_tile2`, `mla_merge_oproj`, `stream`, `gang_w13_gemv`, `gang_w2_gemv` (the last two through `kernel_tests_xcd`, built by the stage); the first call builds six binaries, about 7 minutes | RULE K1: a FAIL names the kernel; its flag is off for the round (the GEMV linear: `--gemv-linears`; w13: `--gemv-w13`; w2: the rows gain `--runtime-flags=-DMPK_W2_CK_TILE`; `moe_router4`: `--router-tasks`; a merge form: its flag); a FAIL in a round-3 suite: the kernel changed since round 3 (the router and merge headers), and the round runs the stock rows only | |
| 30 | S4 | `L start ktime nt`; `L wait ktime 6`; `L start ktime nt_b4`; `L wait ktime 5`; `L start ktime nt_b16`; `L wait ktime 5`; `L start ktime nt_strided`; `L wait ktime 5`; `L start ktime`; `L wait ktime 5` | `env/hw/<date>/ktime/ktime_*.txt`: `TIME linear_gemv ... mean_us=`, `TIME moe_router`, `TIME gang_w13_gemv`, `TIME gang_w2_gemv`, `TIME mla_merge_oproj` per build (each variant builds its binary first, about a minute) | DECIDE G0: the batch constant (T1); RULE: the exec counter of a GEMV class must differ from the record's CK class before any graph clock is read (the JIT trap of round 3) | a build without a `TIME` line: its binary did not run; the row is repeated once |
| 40 | S5 | if the constant is not 8: `L push` with the define in the header, `L start setup`, `L wait setup 3`. Then `L start queue env/session/queue-f2.txt` (G1, G2: 7 rows, about 14 minutes) | `compare=PASS` on the two `it1` rows, ids equal; `table=PASS` on every row; the qkva and o_proj gaps in `report_table.md` of the `gv` rows against the CK rows' | DECIDE G1 (T2), DECIDE G2 (T3) | the `gv` compare FAIL: `--gemv-linears` off, removed from f3 to f6, g1, g2 |
| 54 | S6 | `L start queue env/session/queue-f3.txt` (G3, G4: 4 rows, about 8 minutes) | `compare=PASS`, ids equal; the w2 gap and the w13 gap | DECIDE G3 (T4), DECIDE G4 (T5); H1 and H3 read the router and merge exec from the G4 row (T8) | G3 compare FAIL: `--runtime-flags=-DMPK_W2_CK_TILE` added to f4 to f6, g1, g2; G4 compare FAIL: `--gemv-w13` removed likewise |
| 62 | S7 | `L start queue env/session/queue-g1.txt` (H2, H4: 6 rows, about 12 minutes) | `compare=PASS`, the ids and the route log equal; the router gap with `--router-tasks`; the merge gap with `--merge-tasks` and `--merge-halves 2` | DECIDE H2 (T9), DECIDE H4 (T10) | a compare FAIL: the flag off, removed from g2 and f5 |
| 74 | S8 | `L start queue env/session/queue-g2.txt` (H5: 2 rows, about 4 minutes) | `compare=PASS` with the `x_res` boundary the new operator's, ids equal; the `L{l}.o_proj` gap | DECIDE H5 (T11) | compare FAIL: `--merge-oproj` off, f5 gets `--merge-tasks --merge-halves 2` if H4 passed |
| 78 | S9 | `L pull` (the checkpoint); `L start queue env/session/queue-f4.txt` (G5, G6: 6 rows, about 10 minutes) | `table=PASS`; the GB/s column of the stream rows in `report_table.md`; the head's chunk events in the two head rows | RULE G5 (T6, the ceiling written down); DECIDE G6 (T7) | a stream row FAIL: the probe is recorded as not run; nothing depends on it |
| 88 | S10 | `python3 env/session/queue_flag.py env/session/queue-f5.txt --remove <the flags that failed> --in-place` and, if G6 chose it, `--head-grid 320` added; then `L start queue env/session/queue-f5.txt` (G7 = H6: 4 rows, about 7 minutes) | `output_ids PASS` on the compare rows; three per-token times within 2%; the `FWD_PASS` number of the fourth row | AUTO; the number against 4,500 us is reported with the levers it carries | a spread above 2%: three more runs |
| 96 | S11 | `L pull`; `L report --balance` | the record committed; the balance above $3 | RULE: below $3 the session ends now | |
| 100 | S12 | if time: `L start queue env/session/queue-f6.txt` (G9: 10 rows, about 20 minutes) | `compare=PASS` per knob; the per-token time against S5's `gv` row | RULE T12: a knob that passes and cuts the per-token time by more than 2% goes into a rerun of f5 | a knob's compare FAIL: off, its row recorded |
| 120 | S13 | if time: `L ssh "cd /home/hotaisle/metalOps && bash env/session/queue.sh bitdiff <the G1.1 run> <the G1.2 run>"` (G8; the two run names from S5's status rows) | `record/bitdiff_<a>_<b>.md` with the differing elements and the max ULP per boundary | AUTO; the numerics paragraph of the results page | the run directories are gone (a rerun overwrote them): the two `it1` rows again, then the subcommand |
| 125 | S14 | if S12 kept a knob: `queue_flag.py env/session/queue-f5.txt --runtime-flags=-DNAME --in-place`, then `L start queue env/session/queue-f5.txt` again | as S10 | AUTO | |
| 130 to 205 | S15 | reruns the rules asked for; nothing new is started after minute 190 | | AUTO | |
| end | end | `L pull`; `git push`; then, after the user's yes, `L delete --yes`; `L balance` shows `Hourly Rate: $0.00/hour` | | DECIDE (user: the deletion, asked every time) | if the rate is not $0.00, check the TUI by hand |

Hard stops: at 210 minutes of VM time the queue is stopped, the record
pulled and the VM deleted whatever the state; at any point when the
balance shows less than $3.

### Thresholds for the RULE and DECIDE rows (baseline round 3, `../03-acceleration/08-results.md`)

| Rule | Row | Quantity | Round 3 | Rule |
|---|---|---|---|---|
| T1 | S4 | `TIME linear_gemv` cold at 4, 8 and 16 rows per batch and with the strided map (`nt_b4`, `nt`, `nt_b16`, `nt_strided`); `TIME moe_router` the same | (new; the CK tile's 96 tasks about 14.8 us in the graph) | the batch constant is the fastest of the three, 8 if within 5%; the strided map only if faster than the coalesced by more than 5% (then `-DGEMV_STRIDED` joins the define); the plain build's numbers beside them show what `--nt-streams` is worth standalone |
| T2 | S5 | the qkva and o_proj gaps of the `gv` row against the CK row's | 14.8 us each | a gap that fell by 3 us or more is the kernel's: G3 follows; within 1 us of 14.8: MAJ-8 (the runtime's per-task cost) bounds the linears and G2 confirms; the number goes into the report either way |
| T3 | S5 | the qkva gap at 96, 48 and 32 tasks | 14.8 us at 96 | a gap that tracks the task count is completion-bound (MAJ-8 enters the next round); one that tracks the bytes is the kernel's: the grid that measured best goes into f5 with `--linear-grid` |
| T4 | S6 | the w2 gap with the GEMV multiply | 23.9 us (CK) | on if below 16 us; otherwise `--runtime-flags=-DMPK_W2_CK_TILE` joins the later files |
| T5 | S6 | the w13 gap with `--gemv-w13` | 42.5 us (two rounds) | on if below 33 us (the second round gone) |
| T6 | S9 | the stream rows' GB/s per XCD (gang) and per device (regular), with and without `--nt-streams` | (new; the machine's 5.3 TB/s peak, 660 GB/s per XCD) | the ceiling for every gang linear, written into `10-results.md`; `--nt-streams` stays on for the graph rows unless the gang row without it is faster by more than 5% |
| T7 | S9 | the head's chunk events at the default grid and at `--head-grid 320` | 141 us | the grid with the lower head time goes into f5; the default if within 3% |
| T8 | S6 | the router and merge exec (`[TASK_TIME2]` router, merge) in the G4 row | router 16 us, merge 13.7 us of exec; gaps 19.8 and 18.1 | H1 and H3: on (they are the header) and reported as gains if the exec fell by 4 us or more; otherwise the depth constant of the router is revisited from S4's `moe_router` curve |
| T9 | S7 | the router gap with `--router-tasks` | 19.8 us | on if below the G4 row's router gap by 2 us or more |
| T10 | S7 | the merge gap with `--merge-tasks`, then with `--merge-halves 2` | 18.1 us | the better of the two forms; on if within the spread of the G4 row's merge or better (H5 builds on the half form either way) |
| T11 | S8 | the `L{l}.o_proj` gap with `--merge-oproj` | merge 18.1 + o_proj 14.8 us | on if below 26 us; off otherwise |
| T12 | S12 | each knob: `compare=PASS` and the per-token time against S5's `gv` row | (round 3: no knob survived) | a knob that passes and cuts the per-token time by more than 2% joins the finals' rerun |
| G7 | S10, S14 | the final per-token time, event clock, and `FWD_PASS` | 4,570 to 4,600 us | the number against 4,500; the three clocks in `report_table.md` beside it |

## The queue files

| File | Rows | Used at |
|---|---|---|
| `env/session/queue-f2.txt` | G1, G2: round 3's build as the reference, the CK build under `--nt-streams`, the GEMV linears, the grid at 48 and 32 | S5 |
| `env/session/queue-f3.txt` | G3, G4: the w2 GEMV form, w13 in one round | S6 |
| `env/session/queue-g1.txt` | H2, H4: the router in four tasks, the merge as regular tasks (one and two per head) | S7 |
| `env/session/queue-g2.txt` | H5: the merge with o_proj folded in | S8 |
| `env/session/queue-f4.txt` | G5, G6: the stream probe (four rows), the head at two grids | S9 |
| `env/session/queue-f5.txt` | G7 = H6: the finals at 30, 31, 32 iterations and the `FWD_PASS` row, with the whole stack; edited by `queue_flag.py` before it runs | S10, S14 |
| `env/session/queue-f6.txt` | G9: the sleep and CAS knobs, each with the compare | S12 |

One `run_fleet.py` argument line per row with the optional trailing words
`compare`, `table`, `measure`, `continue`; a row is split on whitespace
without shell quoting, so a define travels as `--runtime-flags=-DNAME`.
The guards of `queue.sh` fail a row before it runs when `--iters` is above
32, `--debug` has more than one iteration, a compare has no reference
tensors or names a synthetic graph (`--graph empty` or `stream`), or a
measure has no profiler. `run_fleet.py` itself refuses a fence knob with
`--router-tasks` or `--merge-oproj`, and `--probe-before L{l}.o_proj`
under `--merge-oproj` (the label is the fold's; probe `L{l}.norm2`).
`harness/tests/test_queue_files.py` parses every row of every file,
builds the plan of every model row and tests the mid-session edit.
`08-rehearsal.md` shows every row expanded.

## Helpers (so nothing is typed by hand on the clock)

| Command | Use |
|---|---|
| `bash env/session/grab.sh` | polls the provisioning list every 18 s and provisions the first 1x MI300X; the log is `env/logs/grab.log` |
| `L wait <stage> [min]` | polls `vm.sh check` every 30 s until the stage's PASS or FAIL row |
| `L report [--balance]` | the status message of the protocol |
| `L start kernels [nt\|mfma]` | the suites against `kernel_tests`, `kernel_tests_nt` or `kernel_tests_mfma`, with `kernel_tests_xcd` for the gang GEMV rows; results under `fleet/tasks/results[_variant]` and the record |
| `L start ktime [nt\|nt_b4\|nt_b16\|nt_strided\|mfma] [launches] [copies]` | the standalone attention, merge, GEMV, router, o_proj fold and (through the `_xcd` build of the plain and `nt` variants) the gang GEMV forms (`KT_TIME`, `KT_COLD` over 27 copies), into `env/hw/<date>/ktime/`; the round-4 variants are built on demand |
| `L start tgcheck <run name>` | the side operators' wiring in the run's `task_graph_rank0.json`; the round-4 forms are checked on the first `gv` run and the first `mo` run |
| `python3 env/session/queue_flag.py <queue> [--remove] <flags> --in-place` | a lever out of, or a knob into, the rows of a later queue (idempotent, keywords and comments kept) |
| `L ssh "cd /home/hotaisle/metalOps && bash env/session/queue.sh bitdiff <a> <b>"` | the bit-diff of two runs' boundary tensors into the record (G8) |
| `L ssh "cd /home/hotaisle/metalOps && rm -f fleet/tasks/build/kernel_tests*"` | after a kernel edit pushed mid-session: the `kernels` and `ktime` stages do not rebuild an existing binary |
| `L ssh "... vm.sh kill <pattern>\|--all"` | stops a hung graph run by anchored pid and writes a `KILLED` row |
| `run_fleet.py --gemv-linears [--linear-grid N] [--head-grid N]`, `--gemv-w13`, `--router-tasks`, `--merge-tasks [--merge-halves 2]`, `--merge-oproj`, `--graph stream --ops M --tasks N --kb K [--gang]`, `--runtime-flags=-DMPK_W2_CK_TILE`, `--nt-streams` | the flags of this round; every one off by default (`05-local-preparation.md`); the w2 GEMV form and the deeper router and merge are the header |

## Failure playbook (rounds 1 to 3 and the laptop's own findings, one line each)

| If you see | Do |
|---|---|
| a `gv` row whose GEMV class shows the CK class's exec counter | the JIT reused the round-3 build: `L ssh "... rm -rf harness/fleet_out/<run>/build"` and the row again; no clock is read until the counter moved (round 3's trap) |
| the `gang_w13_gemv` and `gang_w2_gemv` suites SKIP | `kernel_tests_xcd` is missing: the `kernels` stage builds it; `rm -f fleet/tasks/build/kernel_tests_xcd` and the stage again |
| a `--router-tasks` or `--merge-oproj` run hangs at its first layer | the counter never reached its last task: a fence knob in the row (`run_fleet.py` refuses it; a hand-edited row could carry one), or the gang path lost the type (setup again) |
| `--probe-before L0.o_proj` fails the plan under `--merge-oproj` | the label is the fold's operator; probe `L0.norm2` |
| a stream row with `compare` | the guard refuses it: the probe has no boundaries; `table` only |
| the head row at 2 layers shows `output_ids FAIL` | expected: a 2-layer graph's logits cannot match the reference; the head is judged by G7's ids and compare |
| `L push` after the first push of a session overwrote the VM's fork | it cannot: `push` excludes `repos/` unless `FULL=1`; if the JIT loses a task type anyway, `L start setup` re-applies the patches |
| the `kernels` stage says PASS for a binary built before a kernel edit | it does not rebuild an existing binary: remove `fleet/tasks/build/kernel_tests*` on the VM first |
| `setup.sh` fails on a patch | the three patches apply in order `gfx942`, `new_tasks`, `sched_xcd` on the pristine commit; `env/preflight.sh` checks that on the laptop before every push |
| a queue row FAIL with `fault=1` before any `FWD_PASS` | a new flag faulted: the row's flag is off for the round; the address record is in `fleet_run_meta.json` |
| two graph runs at once | never: the JIT directory is shared; the queue serialises; no `ktime` beside a queue |
| a queue row with a quoted define | the row is split on whitespace: `--runtime-flags=-DNAME`, no quotes |

## What is pulled back and committed

`L pull` does it: `env/logs/*.out` and the status files (into
`env/hw/<date>/logs/`); `env/hw/<date>/` (`runs/<name>/` for every queue
row with `plan.json`, `wall.json`, `fleet_run_meta.json`, `fwd_pass.log`
with the timing and spin lines, `clock.log`, `correctness_report.md` and
`.json`, `report_table.md`, `metrics.json`, `event_timing.json`,
`task_graph_check.txt`, `bitdiff_*.md`; `kernel_tests[_variant]/`;
`ktime/`); `harness/ref/` JSON files; `env/check_day1.log`. Then one
commit on the branch. Not pulled: the tensors, the build directories,
files above 400 KB.

## Fallbacks decided now

- A GEMV suite fails: the kernel is off for the round; the round-3 build
  carries the router and merge rows; the standalone number is still read.
- The `gv` row's compare fails: `--gemv-linears` off, its flag removed
  from every later file; G3 and G4 still run (the w2 and w13 forms do not
  need it).
- The w2 GEMV form fails or is slower: `-DMPK_W2_CK_TILE` restores the
  CK multiply; one define, no rebuild of anything else.
- `--merge-oproj` fails: `--merge-tasks --merge-halves 2` if H4 passed,
  else the stock merge; the finals carry what passed.
- The balance is short: the session ends at $3 with what it has; the
  finals are in hand by minute 96 in the plan above.
- `grab.sh` finds no VM within an hour: the user is told; nothing is
  billed.

## Expected outputs (round 3, to recognise a deviation at once)

| Stage or row | Expected |
|---|---|
| `checks` | 7 PASS lines |
| `reference` | 32 ids, the calibration floors of round 2 |
| `kernels` | 19 suites, 100 of 100 (round 3's nine and the ten of this round); the same for `nt` |
| `ktime nt` | the attention grid about 34 us cold, the merge 11.5; the GEMV grid (96 tasks of 38 rows, cold) is the question; the router, w13 and w2 rows new |
| a 2-layer row | about 2 minutes of wall time with the JIT; 16 boundaries PASS with `compare`, the route log PASS |
| the round-3 build on the model | 4,570 to 4,600 us per token on the event clock; per operator in a MoE layer: attention about 60 us, merge 18.1, o_proj 14.8, w2 23.9, w13 42.5, qkva 14.8, router 19.8, head 141 |
| the finals | the stack of `01` and `03`: 0.57 ms certain and 1.25 possible from the GEMV items, 0.37 and 0.8 from the router and merge, against 4,500 |
| the session's cost | about $5 for the plan's 100 minutes; $10.50 at the hard stop |
