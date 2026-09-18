# 07 - Session plan: one session on a 1x MI300X, the kernels first, the router and merge after

Written 2026-09-18 after the laptop preparation (`06-checklist.md`: L1 to
L9, N1 to N6; the double-check and the final check), revised the same day
against round 3's session log and lessons (`../03-acceleration/07`, `09`).
One session, as round 3's (`../03-acceleration/05-session-plan.md`):
every row is one literal command of `env/session/laptop.sh` (`L`), the
text that decides PASS, whether the row runs on its own (AUTO), applies a
written rule (RULE) or reports a decision (DECIDE), and what to do on
FAIL. The exact commands the scripts expand to are in `08-rehearsal.md`,
generated in DRY mode by `env/session/rehearse.sh`. The run log and the
numbers go to `09-session-log.md` and `10-results.md`.

What is new against round 3's plan:

- The levers are kernels of our own behind flags of `run_fleet.py`
  (`--gemv-linears`, `--gemv-w13`, `--router-tasks`, `--merge-tasks`,
  `--merge-oproj`) and two forms that are the pushed header itself: the
  w2 GEMV multiply under `--fuse-silu` (the CK multiply one define away,
  `--runtime-flags=-DMPK_W2_CK_TILE`) and the router and merge one level
  deeper (no flag; their numbers are read from the first round-4 rows).
- Every graph row carries the round-3 stack (`--tile-linears --nt-weights
  --event-timing --fuse-norm2 --fuse-silu --fuse-norm1 --mfma-attend
  --attend-tasks`) and, for the round-4 kernels, `--nt-streams` (L1c: the
  plain build's flat loads wait `vmcnt(0)` per batch); one row (G1.3) runs
  the GEMV linears with plain loads, the A/B of the one question the
  offline pass could not settle.
- The first decision (G0, the batch constant) is taken on standalone
  timings before any graph row, and enters the graph rows as a define
  through `--runtime-flags=-DGEMV_BATCH=N` (the JIT's `MPK_EXTRA_HIPCC_FLAGS`
  hook, the same the knobs use; the header's constants are `#ifndef`), so
  no push and no rebuild of the fork are needed.
- The host is fresh: the setup is about 12 minutes; the durations below
  are round 3's measured ones (`07-session-log.md`: download 95 s, hw
  158 s, setup 483 s, checks 84 s, reference 74 s; a 2-layer row about
  1.6 minutes with its JIT, a model row about 1.5).

## Corrections after the session (2026-09-18, `09-session-log.md`, `11-lessons.md`)

The plan ran as written except where the session proved a row wrong; the
rows below keep their text as the plan of record, with the correction
here:

- S10 and S14, the finals' PASS text: `output_ids PASS` and three per-token
  times within 2%. The it32 compare rows fail `head.B15.logits` (the
  boundary is captured after the reference's step) and the route log
  (MIN-32's tail-expert flips) in round 3's record as in this one; a
  compare of the model's boundaries is a one-iteration row.
- Every graph row carried `--worker-timing`; it hangs the round-4 header
  (MIN-35), so every row ran without it and the exec-per-class table of
  "How the numbers are read" does not exist for this round. The JIT-trap
  rule was applied on the gaps (a `gv` row's linear gaps differ from the CK
  row's).
- G1's reference row carried `-DMPK_W2_CK_TILE` for the inline CK path of
  the round-4 header; that path returned ids `[0]` and hung. The define
  now includes round 3's file verbatim; the A/B ran as `queue-f7.txt` and
  the define joined every later file (T4).
- The queue runs each row under `timeout ${ROW_TIMEOUT:-600}` (a row hung
  20 minutes before it did); the o_proj fold's `ktime` row reads the
  trial's `oproj` half.
- The rows after the plan: `queue-f8.txt` and `f9` (the finals' compare
  bisect and controls, which found the two failures pre-existing),
  `queue-g3.txt` and `g4` (the GEMV linears at batch 4, the finals with
  `POLL_SLEEP=8` and with `GEMV_BATCH=4`, at the user's request).

## Budget

| | |
|---|---|
| Balance | $13.01 (read from the team page after round 3; no VM, rate $0.00) |
| Shape | 1x MI300X, $2.99 per hour, billed per minute (round 3: 144 minutes, $7.12) |
| Hours | 4.3 |
| The session | up to 3.5 hours ($10.47), hard stop at minute 210 of VM time; the rows that decide (S0 to S11) need about 90 minutes, G9 and G8 about 30 more |
| Reserve | $2.50: a second provisioning if the first host fails its setup (a second setup costs about 12 minutes, $0.60) |

The record is pulled at the checkpoints (S9, S11) and at the end; the
session stops when the balance shows less than $3. Running out of
balance deletes the VM with everything on it.

## Before the session, on the laptop

Every line below is checked and reported before `grab.sh`:

| Check | How | State on 2026-09-18 |
|---|---|---|
| the gate | `SHELLCHECK=1 bash env/preflight.sh`: 9 PASS | PASS (commit 2043a1b and after) |
| the fork pristine | `git -C repos/fleet-chiplet-megakernel status --short` shows untracked files only (the kernel copies) | 0 dirty tracked lines |
| the branch pushed | `git status` clean, `git log origin/local/round-4..local/round-4` empty | pushed |
| no VM, no address | `L balance` shows `No virtual machines`; `env/session/vm.ip` absent; no `grab.sh` process | to check at S0 |
| the queue files | `.venv/bin/python -m pytest harness/tests/test_queue_files.py -q` (every row parses and builds its plan); `08-rehearsal.md` regenerated | 6 passed; 41 rows |
| the decisions below | the four put to the user taken on 2026-09-18; the go for the VM | the go is open |

## Decisions taken by the user on 2026-09-18

Four decisions were put to the user and taken; the rest of the table are
the plan's own defaults, stated here so the session runs on written rules.

| Decision | Choice |
|---|---|
| Budget | up to 3.5 hours of VM time ($10.47 of the $13.01); hard stop at minute 210 or below $3; $2.50 kept for a second provisioning if the first host fails its setup |
| Sessions and order | one session: S3 and S4 (the suites, the standalone times, the batch constant), S5 (the linears and the grid, G1 and G2), S6 (w2, w13: G3, G4), S7 (the router in four tasks, the merge as regular tasks: H2, H4), S8 (the o_proj fold: H5), S9 (the stream probe, the head: G5, G6), S10 (the finals: G7 = H6), then G9 and G8 as time allows |
| Autonomy | the agent applies every RULE and DECIDE row by its written threshold and reports as it goes; a lever that failed is removed from the later queue files with `queue_flag.py` before they run; nothing waits for the user except the deletion |
| A round-4 kernel fails its suite or its step-0 compare | the lever is dropped and the rest runs: its flag is removed from every later file, the round-3 build carries the other rows, the failing trial directory is recorded for the laptop afterwards; no debugging on the VM's clock |
| Provisioning | `grab.sh` polls the list; the user is told when the polling starts; nothing is billed before the provision |
| Deletion | asked every time; the record pulled and the branch pushed first |
| The batch constant | G0's `ktime` decides 4, 8 or 16 rows per batch for the GEMV linear and the router (8 if within 5%); a constant other than 8 enters the later files as `--runtime-flags=-DGEMV_BATCH=N` (and `-DROUTER_BATCH=M`) through `queue_flag.py`, no push |
| The load policy | `--nt-streams` on every round-4 row; G1.3 (plain loads) is read beside G1.2 and the stream rows beside each other; if plain wins both by more than 5%, `--nt-streams` is removed from the later files (T13) |
| A GEMV suite row fails | the kernel is off for the round: its flag is removed from every later file (`--gemv-linears` from f2 to f6, g1, g2; `--gemv-w13` likewise); the round-3 build runs the router and merge rows |
| `--merge-oproj` fails its compare | off; the finals run with `--merge-tasks --merge-halves 2` if H4 passed, else with the stock merge |
| The fence knobs | not run: the counter forms rely on the runtime's fences (`run_fleet.py` refuses the pair); round 3 measured them on the stock forms, both failed the compare |
| The task-graph check | run on the first `mo` run (S8) and read for the type counts (below); a wrong count is a plan defect, the row's flag is off until it is understood |

## Reporting protocol

- Before `grab.sh`: the balance and a one-line "polling for a VM now".
- After every stage row: the `PASS`/`FAIL` line of `env/logs/session.status`
  pasted verbatim, with the minute mark and the balance estimate (`L report`
  prints both).
- At every RULE and DECIDE row: the rule, the measured value, and the
  choice taken, in the report that follows; the next command does not wait.
- At every checkpoint (S9, S11): the commit hash of the pulled record and
  the balance.
- At the end: the last `L status`, the `Hourly Rate: $0.00/hour` line (read
  by hand from the team page: the rate has several spaces before the
  value), and the commit hash of the pulled record.

## The session (up to 3.5 hours of VM time)

Every command is run from the repository root on the laptop. `L` stands
for `bash env/session/laptop.sh`. Status rows are read with `L status`;
the "PASS when" column is the row that must appear in it. Minute marks
are from the provision and follow round 3's measured durations.

| Minute | Row | Command | PASS when | Mode | On FAIL |
|---|---|---|---|---|---|
| -5 | S0 | `L balance` | `Available Balance` about $13, `No virtual machines`, `Hourly Rate: $0.00/hour` | AUTO (the go is given with this plan) | a VM already listed: stop, nothing is provisioned |
| 0 | S0 | `bash env/session/grab.sh`, then `FULL=1 L push` | an address in `env/session/vm.ip`; the push (the tree with the pristine fork, about a minute) completes | AUTO | `grab.sh` gives up after an hour of polls: the user is told; nothing is billed |
| 1 | S1 | `L start download`; `L start setup`; `L start hw`; `L ssh "cd /home/hotaisle/metalOps && bash env/session/vm.sh preflight"` | `PASS download` (95 s in round 3; up to ten minutes on a host without the model), `PASS setup` (483 s: the two venvs, the three patches, the Fleet build), `PASS hw` (158 s on a new host) | AUTO | setup: read `env/logs/setup.out`; the fix on the laptop, `L push`, `L start setup` again (an incremental build takes about a minute; a changed patch needs `FULL=1 L push` first) |
| 10 | S2 | `L start checks`; `L wait checks 5`; `L start reference` | `PASS checks` with 7 PASS lines (84 s); `PASS reference` (74 s; the compares need it) | AUTO | a check that passed in round 3: a machine difference, log it; reference FAIL: no compare can run, the queue guard says so |
| 13 | S3 | `L start kernels`; `L wait kernels 20`; `L start kernels nt`; `L wait kernels 10` | every suite 100 of 100 on both builds: round 3's nine (`mla_prep`, `mla_attend`, `mla_merge_uv`, `moe_router`, `copy`, `prefetch`, `prefetch_moe`, `mla_attend_scores`, `mla_attend_splits`) and the ten of this round (`linear_gemv`, `linear_gemv_norm`, `linear_gemv_res`, `moe_router4`, `mla_merge_uv_tile`, `mla_merge_uv_tile2`, `mla_merge_oproj`, `stream`, `gang_w13_gemv`, `gang_w2_gemv`; the last two through the `_xcd` build the stage makes); the first call builds six binaries (seconds each on 13 cores); the gang rows write 92 MB of expert slabs per trial, so the stage is about 8 minutes | RULE K1: a FAIL names the kernel; its flag is off for the round (the GEMV linear: `--gemv-linears`; w13: `--gemv-w13`; w2: the rows gain `--runtime-flags=-DMPK_W2_CK_TILE`; `moe_router4`: `--router-tasks`; a merge form: its flag). A FAIL in a round-3 suite means the kernel changed since round 3 (the router and merge headers): the round runs the stock rows only and the user is told | |
| 22 | S4 | `L start ktime nt`; `L wait ktime 6`; `L start ktime nt_b4`; `L wait ktime 5`; `L start ktime nt_b16`; `L wait ktime 5`; `L start ktime nt_strided`; `L wait ktime 5`; `L start ktime`; `L wait ktime 5` | `env/hw/<date>/ktime/ktime_*.txt`, per build: `TIME mla_attend`, `TIME mla_merge_uv`, `[SPIN]`, `TIME linear_gemv_norm ... mean_us=` and `TIME linear_gemv_res` (qkva's 96 tasks and o_proj's 64, the weight rotated over 27 copies), `TIME moe_router`, `TIME mla_merge_oproj`, and on the `nt` and plain builds `TIME gang_w13_gemv`, `TIME gang_w2_gemv` (each variant builds its binary first, about a minute) | DECIDE G0 (T1): the batch constant and the map; RULE: the exec counter of a GEMV class must differ from the record's CK class before any graph clock is read (the JIT trap, lesson 12) | a build without a `TIME` line: its binary did not run; the row is repeated once; a variant that fails to build: its point is missing, 8 stays |
| 32 | S5 | if the constant is not 8: `python3 env/session/queue_flag.py env/session/queue-f<k>.txt --runtime-flags=-DGEMV_BATCH=N --in-place` for f2 to f6, g1, g2 (and `-DROUTER_BATCH=M` when the router's differs; `-DGEMV_STRIDED` if the map won); `L push`. Then `L start queue env/session/queue-f2.txt` (G1, G2: 8 rows, about 13 minutes) | `compare=PASS` on the two `it1` rows, ids equal; `table=PASS` on every row; in `report_table.md`: the qkva and o_proj gaps of the `gv` rows against the CK rows', the `lnorm` class's exec and count (the GEMV tasks count there) | DECIDE G1 (T2), DECIDE G2 (T3), RULE T13 (the load policy) | the `gv` compare FAIL: `--gemv-linears` off, removed from f3 to f6, g1, g2; a `gv` row that faults: the same, and the address record kept |
| 45 | S6 | `L start queue env/session/queue-f3.txt` (G3, G4: 4 rows, about 7 minutes) | `compare=PASS`, ids equal; the w2 gap (the `w2silu` class now holds the w2 GEMV form and, in the G4 row, the w13 tiles too) and the w13 gap | DECIDE G3 (T4), DECIDE G4 (T5); H1 and H3 read the router and merge exec from the G4 row (T8) | G3 compare FAIL: `--runtime-flags=-DMPK_W2_CK_TILE` added to f4 to f6, g1, g2; G4 compare FAIL: `--gemv-w13` removed likewise |
| 53 | S7 | `L start queue env/session/queue-g1.txt` (H2, H4: 6 rows, about 10 minutes) | `compare=PASS`, the ids and the route log equal (`fleet_route_log.json` against the reference: the `output_ids` and `route_log` rows of the compare); the router gap with `--router-tasks`; the merge gap with `--merge-tasks` and `--merge-halves 2` | DECIDE H2 (T9), DECIDE H4 (T10) | a compare FAIL: the flag off, removed from g2 and f5 |
| 64 | S8 | `L start queue env/session/queue-g2.txt` (H5: 2 rows, about 4 minutes); then `L start tgcheck <the it1 run's name>` (the name is the first row's status line; `L2_it1_tile_at_fn1_fn2_fs_nt_nts_mfma_gv_w13_rt_mo` when nothing was removed or added) | `compare=PASS` with the `x_res` boundary the new operator's, ids equal; the `L{l}.o_proj` gap; the check prints `PASS` and a `task_types` count that matches the table below | DECIDE H5 (T11); RULE: the type counts | compare FAIL: `--merge-oproj` off, f5 gets `--merge-tasks --merge-halves 2` if H4 passed; a wrong type count: the flag off and the user told |
| 69 | S9 | `L pull` (the checkpoint); `L start queue env/session/queue-f4.txt` (G5, G6: 6 rows, about 9 minutes) | `table=PASS`; the stream rows' `report_table.md` carries the "Per-operator time and rate" block and the `stream probe: N GB/s` line; the head rows' `report_table.md` shows the head's chunk events | RULE G5 (T6, the ceiling written down), DECIDE G6 (T7) | a stream row FAIL: the probe is recorded as not run; nothing depends on it |
| 78 | S10 | `python3 env/session/queue_flag.py env/session/queue-f5.txt --remove <the flags that failed> --in-place` and, if G6 chose it, `--head-grid 320` added; `L push`; then `L start queue env/session/queue-f5.txt` (G7 = H6: 4 rows, about 7 minutes) | `output_ids PASS` and `compare=PASS` on the two compare rows; three per-token times within 2%; the `FWD_PASS` number of the fourth row (its `fwd` count is 29; the worker-timing rows show `fwd=0`, expected: their lines go to `fwd_pass.log`) | AUTO; the number against 4,500 us is reported with the levers it carries | a spread above 2%: three more runs |
| 86 | S11 | `L pull`; `L report --balance` | the record committed; the balance above $3 | RULE: below $3 the session ends now | |
| 90 | S12 | if time: `L start queue env/session/queue-f6.txt` (G9: 10 rows, about 17 minutes) | `compare=PASS` per knob; the per-token time against S5's `gv` row | RULE T12: a knob that passes and cuts the per-token time by more than 2% goes into a rerun of f5 | a knob's compare FAIL: off, its row recorded |
| 108 | S13 | if time: `L ssh "cd /home/hotaisle/metalOps && bash env/session/queue.sh bitdiff <the G1.1 run> <the G1.2 run>"` (G8; the two `it1` names from S5's status rows) | `record/bitdiff_<a>_<b>.md` with the differing elements and the max ULP per boundary | AUTO; the numerics paragraph of the results page | the run directories are gone (a rerun overwrote them): the two `it1` rows again, then the subcommand |
| 112 | S14 | if S12 kept a knob: `queue_flag.py env/session/queue-f5.txt --runtime-flags=-DNAME --in-place`, `L push`, then `L start queue env/session/queue-f5.txt` again | as S10 | AUTO | |
| 120 to 205 | S15 | reruns the rules asked for; nothing new is started after minute 190 | | AUTO | |
| end | end | `L pull`; `git push`; then, after the user's yes, `L delete --yes`; `L balance` shows `Hourly Rate: $0.00/hour` | | DECIDE (user: the deletion, asked every time) | if the rate is not $0.00, check the TUI by hand |

Hard stops: at 210 minutes of VM time the queue is stopped (`vm.sh kill
--all` after the `queue.sh run` loop, playbook), the record pulled and the
VM deleted whatever the state; at any point when the balance shows less
than $3.

### Thresholds for the RULE and DECIDE rows (baseline round 3, `../03-acceleration/08-results.md`)

| Rule | Row | Quantity | Round 3 | Rule |
|---|---|---|---|---|
| T1 | S4 | `TIME linear_gemv_norm` and `TIME linear_gemv_res` cold at 4, 8 and 16 rows per batch and with the strided map (`nt_b4`, `nt`, `nt_b16`, `nt_strided`); `TIME moe_router` the same | (new; the CK tile's 96 tasks 14.8 us in the graph, the router 16 us of exec) | the batch constant is the fastest of the three, 8 if within 5%; the strided map only if faster than the coalesced by more than 5%; the router's own constant by the same rule; the plain build's numbers beside them show what `--nt-streams` is worth standalone. The in-graph time is about twice the standalone (lesson 6), so a standalone GEMV grid above 8 us already says the kernel cannot beat 14.8 |
| T2 | S5 | the qkva and o_proj gaps of the `gv` row against the CK row's | 14.8 us each | a gap that fell by 3 us or more is the kernel's (G3 follows); within 1 us of 14.8: MAJ-8 (the runtime's per-task cost, 2.3 to 2.9 us per operator and 0.19 per task, lesson 9) bounds the linears and G2 confirms; the number goes into the report either way |
| T3 | S5 | the qkva gap at 96, 48 and 32 tasks | 14.8 us at 96 | a gap that tracks the task count (48 tasks about 9 us less by the ladder's line) is completion-bound: MAJ-8 enters the next round; one that tracks the bytes is the kernel's; the grid that measured best goes into f5 as `--linear-grid N` |
| T13 | S5 | the `gv` row with plain loads (G1.3) against the same with `--nt-streams` (G1.2); the stream rows without and with (S9) | (new) | `--nt-streams` stays unless plain wins both by more than 5%; then it is removed from f3 to f6, g1, g2 before they run (S6 waits for this reading) |
| T4 | S6 | the w2 gap with the GEMV multiply | 23.9 us (CK) | on if below 16 us; otherwise `--runtime-flags=-DMPK_W2_CK_TILE` joins the later files |
| T5 | S6 | the w13 gap with `--gemv-w13` | 42.5 us (two rounds) | on if below 33 us (the second round gone); the first round's number against G5's ceiling |
| T6 | S9 | the stream rows' GB/s per XCD (the gang row) and per device (the regular rows), with and without `--nt-streams` | (new; the machine's 5.3 TB/s peak, 660 GB/s per XCD) | the ceiling for every gang linear, written into `10-results.md`; the 152 and 304 KB rows request 5% more lines than their bytes (a partial last batch, the clamped rows re-read from L2), the 256 KB row is exact |
| T7 | S9 | the head's chunk events at the default grid (400 tasks of 256 rows) and at `--head-grid 320` (320 of 320) | 141 us | the grid with the lower head time goes into f5; the default if within 3% |
| T8 | S6 | the router and merge exec (`[TASK_TIME2]` router, merge) in the G4 row | router 16 us, merge 13.7 us of exec; gaps 19.8 and 18.1 | H1 and H3: on (they are the header) and reported as gains if the exec fell by 4 us or more; otherwise the router's depth is revisited from S4's `moe_router` curve |
| T9 | S7 | the router gap with `--router-tasks` | 19.8 us | on if below the G4 row's router gap by 2 us or more |
| T10 | S7 | the merge gap with `--merge-tasks`, then with `--merge-halves 2` | 18.1 us | the better of the two forms; on if within the spread of the G4 row's merge or better (H5 builds on the half form either way) |
| T11 | S8 | the `L{l}.o_proj` gap with `--merge-oproj` | merge 18.1 + o_proj 14.8 us | on if below 26 us; off otherwise |
| T12 | S12 | each knob: `compare=PASS` and the per-token time against S5's `gv` row | (round 3: no knob survived; the sleep and CAS knobs untested) | a knob that passes and cuts the per-token time by more than 2% joins the finals' rerun |
| G7 | S10, S14 | the final per-token time, event clock, and `FWD_PASS` | 4,571 to 4,600 us; `FWD_PASS` 4,584 and 4,590 | the number against 4,500; the three clocks in `report_table.md` beside it |

### How the numbers are read

- `report_table.md` (the `table` word runs `measure.py --run` on the run):
  the per-operator gaps from the event clock (event i fires when operator
  i - 1 completes; `measure.py` names them right since round 3, lesson 1),
  the per-token median over the iterations, and the "Exec time per task
  by class" table from the worker timing (`[TASK_TIME2]`).
- The classes of `[TASK_TIME2]` (the patch's switch): `prep` the per-head
  prep; `attend` the gang and the regular attention; `merge` the gang
  merge, the regular merge (205) and the o_proj fold (207); `router` the
  one-task and the four-task router; `copy`; `w2silu` the w2 GEMV form
  and, under `--gemv-w13`, the w13 tiles; `lnorm` the fused-norm CK tile
  and the GEMV linear (195); `prefetch` the prefetch and the stream tasks.
  A class's exec per task is the kernel's own time; the operator's gap is
  what the graph pays. Both are read; a lever that moves the wrong row is
  a measurement finding (lesson 3).
- `fwd=0` on a worker-timing row is expected (its `FWD_PASS` lines go to
  `fwd_pass.log`); the ids row of the compare is the correctness check of
  a model row, the boundaries of an `it1` row the check of a 2-layer one.
- A stream row's `report_table.md` has the "Per-operator time and rate"
  block and the `stream probe: N GB/s` median; the gang row's number is
  per XCD (37 tiles of 304 KB per XCD per operator), the regular rows'
  per device.

### The task-graph check (S8)

`L start tgcheck <run>` prints the side-task verdict (none this round) and
`task_types`, the task count per type. Expected on the 2-layer graph
(layer 0 dense, layer 1 MoE; both with attention), with every lever on:

| Type | Task | Count on 2 layers | With the head |
|---|---|---|---|
| 195 | the GEMV linear (qkva 96 + o_proj 64 per layer with `--merge-oproj` off; qkva only where the fold replaces o_proj) | 192 with the fold (96 per layer), 320 without | + 400 (or 320 at `--head-grid 320`) |
| 196 | the gang w13 GEMV, 37 tiles per XCD | 296 (the MoE layer) | |
| 204 | the four-task router | 4 (the MoE layer) | |
| 207 | the merge with o_proj folded in, 32 per layer | 64 | |
| 205 | the regular merge (only under `--merge-tasks` without the fold) | 32 or 64 | |

A count that differs names a plan defect (a flag not applied, a layer
skipped); the row's flag is off until it is understood. The JSON stays on
the VM (the record excludes `build/`); the printed verdict is pulled.

## The queue files

| File | Rows | Used at |
|---|---|---|
| `env/session/queue-f2.txt` | G1, G2: round 3's build as the reference, the CK build under `--nt-streams`, the GEMV linears (streaming and plain loads), the grid at 48 and 32 | S5 |
| `env/session/queue-f3.txt` | G3, G4: the w2 GEMV form, w13 in one round | S6 |
| `env/session/queue-g1.txt` | H2, H4: the router in four tasks, the merge as regular tasks (one and two per head) | S7 |
| `env/session/queue-g2.txt` | H5: the merge with o_proj folded in | S8 |
| `env/session/queue-f4.txt` | G5, G6: the stream probe (four rows), the head at two grids | S9 |
| `env/session/queue-f5.txt` | G7 = H6: the finals at 30, 31, 32 iterations and the `FWD_PASS` row, with the whole stack; edited by `queue_flag.py` before it runs | S10, S14 |
| `env/session/queue-f6.txt` | G9: the sleep and CAS knobs, each with the compare | S12 |

One `run_fleet.py` argument line per row with the optional trailing words
`compare`, `table`, `measure`, `continue`; a row is split on whitespace
without shell quoting, so a define travels as `--runtime-flags=-DNAME`,
one per option. Every queue file is pushed before its `L start queue`
(a file written after the push fails the stage in a second, lesson 13).
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
| `L wait <stage> [min]` | polls `vm.sh check` every 30 s until the stage's own PASS or FAIL row (not a queue's DONE line, lesson 13) |
| `L report [--balance]` | the minute mark, the cost so far, the last status rows, optionally the balance |
| `L start kernels [nt\|mfma]` | the suites against `kernel_tests`, `kernel_tests_nt` or `kernel_tests_mfma`, the gang GEMV rows through the variant's `_xcd` build; results under `fleet/tasks/results[_variant]` and the record |
| `L start ktime [nt\|nt_b4\|nt_b16\|nt_strided\|mfma] [launches] [copies]` | the standalone attention, merge, GEMV forms, router, o_proj fold and (through the `_xcd` build of the plain and `nt` variants) the gang GEMV forms (`KT_TIME` 50 launches, `KT_COLD` over 27 copies), into `env/hw/<date>/ktime/`; the round-4 variants are built on demand |
| `L start tgcheck <run name>` | the side-task verdict and the task count per type of the run's `task_graph_rank0.json`; the verdict into the run's record |
| `python3 env/session/queue_flag.py <queue> [--remove] <flags> --in-place` | a lever out of, or a flag or define into, the rows of a later queue (idempotent, keywords and comments kept); push the file afterwards |
| `L ssh "cd /home/hotaisle/metalOps && bash env/session/queue.sh bitdiff <a> <b>"` | the bit-diff of two runs' `fleet_boundaries.safetensors` into the record (G8) |
| `L ssh "cd /home/hotaisle/metalOps && rm -f fleet/tasks/build/kernel_tests*"` | after a kernel edit pushed mid-session: the `kernels` and `ktime` stages do not rebuild an existing binary |
| `L push` then `L start setup` | after a kernel edit pushed mid-session: the JIT reads the fork's installed copy of the headers, which only setup refreshes (73 s; lesson 12) |
| `L ssh "... pgrep -f 'queue.sh run'"` then `L ssh "... vm.sh kill --all"` | stops a queue: the loop first, then the graph run by anchored pid (a `KILLED` row); killing the stage's shell alone leaves the loop alive (lesson 14) |
| `run_fleet.py --gemv-linears [--linear-grid N] [--head-grid N]`, `--gemv-w13`, `--router-tasks`, `--merge-tasks [--merge-halves 2]`, `--merge-oproj`, `--graph stream --ops M --tasks N --kb K [--gang]`, `--runtime-flags=-DMPK_W2_CK_TILE`, `--runtime-flags=-DGEMV_BATCH=N`, `--nt-streams` | the flags of this round; every one off by default (`05-local-preparation.md`); the w2 GEMV form and the deeper router and merge are the header |

## Failure playbook (rounds 1 to 3 and the laptop's own findings, one line each)

| If you see | Do |
|---|---|
| a `gv` row whose `lnorm` class shows the CK tile's exec counter (about 18 us per task) or whose gaps equal the CK row's to the microsecond | the JIT reused the round-3 build's headers: `L start setup` (it re-installs them), then the row again; no clock is read until the counter moved (lesson 12) |
| the `gang_w13_gemv` and `gang_w2_gemv` suites SKIP | `kernel_tests_xcd` is missing: the `kernels` stage builds it; `rm -f fleet/tasks/build/kernel_tests_xcd` and the stage again |
| a `--router-tasks` or `--merge-oproj` run hangs at its first layer | the counter never reached its last task: a fence knob in the row (`run_fleet.py` refuses it; a hand-edited row could carry one), or the gang path lost the type (setup again); `vm.sh kill --all`, the flag off |
| `--probe-before L0.o_proj` fails the plan under `--merge-oproj` | the label is the fold's operator; probe `L0.norm2` |
| a stream row with `compare` | the guard refuses it: the probe has no boundaries; `table` only |
| the head row at 2 layers shows `output_ids FAIL` | expected: a 2-layer graph's logits cannot match the reference; the head is judged by G7's ids and compare (G6's rows carry no compare word) |
| `fwd=0` on a worker-timing row | not a failure: those builds write their `FWD_PASS` lines to `fwd_pass.log` |
| `no queue file` from a `queue` stage | the file was edited after the last push: `L push`, then the stage again |
| `L push` after the first push of a session overwrote the VM's fork | it cannot: `push` excludes `repos/` unless `FULL=1`; if the JIT loses a task type anyway, `L start setup` re-applies the patches |
| the `kernels` stage says PASS for a binary built before a kernel edit | it does not rebuild an existing binary: remove `fleet/tasks/build/kernel_tests*` on the VM first |
| `setup.sh` fails on a patch | the three patches apply in order `gfx942`, `new_tasks`, `sched_xcd` on the pristine commit; a changed patch needs `FULL=1 L push` first (lesson 17); `env/preflight.sh` checks the order on the laptop before every push |
| a queue row FAIL with `fault=1` before any `FWD_PASS` | a new flag faulted: the row's flag is off for the round; the address record is in `fleet_run_meta.json` |
| two graph runs at once | never: the JIT directory is shared; the queue serialises; no `ktime` beside a queue; a stopped queue is checked with `pgrep -f 'queue.sh run'` before the next start (lesson 14) |
| a repeated final overwrote its run directory | the finals run at 30, 31 and 32 iterations (distinct names); the queue moves an existing record directory aside before a rerun (lesson 15) |
| a queue row with a quoted define | the row is split on whitespace: `--runtime-flags=-DNAME`, no quotes |
| the delete check does not find the rate | the team page prints `Hourly Rate:       $0.00/hour` with several spaces; read it by hand |

## What is pulled back and committed

`L pull` does it: `env/logs/*.out` and the status files (into
`env/hw/<date>/logs/`); `env/hw/<date>/` (`runs/<name>/` for every queue
row with `plan.json`, `wall.json`, `fleet_run_meta.json`, `fwd_pass.log`
with the timing and spin lines, `clock.log`, `correctness_report.md` and
`.json`, `report_table.md`, `metrics.json`, `event_timing.json`,
`task_graph_check.txt`, `bitdiff_*.md`; `kernel_tests[_variant]/`;
`ktime/`); `harness/ref/` JSON files; `env/check_day1.log`. Then one
commit on the branch. Not pulled: the tensors, the build directories
(the task-graph JSON stays on the VM; its verdict is pulled), files above
400 KB. A run name that ran twice keeps both directories in the record
(`<name>.prev-<utc>`).

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
- The batch constant is not 8: a define in the later files, no push of a
  header; the suites already ran the default, and the `ktime` row that
  chose it is the evidence.
- The balance is short: the session ends at $3 with what it has; the
  finals are in hand by minute 86 in the plan above.
- `grab.sh` finds no VM within an hour: the user is told; nothing is
  billed.

## Expected outputs (round 3, to recognise a deviation at once)

| Stage or row | Expected |
|---|---|
| `checks` | 7 PASS lines (SPX+NPS1; `import mirage`; the schedulers on their XCDs; the agent-scope fences; the CK FMHA negative; the counter names; 304 CUs) |
| `reference` | 32 ids, the calibration floors of round 2 (router 3.59e-3, scores 2.61e-3, logits 4.1e-2, norm 4.5e-2, layer 4.3e-3) |
| `kernels` | 19 suites, 100 of 100 (round 3's nine and the ten of this round); the same for `nt` |
| `ktime nt` | the attention grid about 34 us cold, the merge 11.5, the spin about 2,100 MHz; the GEMV grids (qkva's 96 tasks of 38 rows, o_proj's 64 of 32, cold) are the question; the router, w13, w2 and o_proj-fold rows new |
| a 2-layer row | about 1.6 minutes of wall time with the JIT; the `it1` compare: 16 boundaries PASS, the route log PASS |
| the round-3 build on the model | 4,570 to 4,600 us per token on the event clock; per operator in a MoE layer: attention about 60 us, merge 18.1, o_proj 14.8, w2 23.9, w13 42.5, qkva 14.8, router 19.8, head 141 |
| the finals | the stack of `01` and `03`: 0.57 ms certain and 1.25 possible from the GEMV items, 0.37 and 0.8 from the router and merge, against 4,500 |
| the session's cost | about $4.50 for the plan's 90 minutes; $10.50 at the hard stop |

## What the results page (`10-results.md`) needs from the session

- The final table: the three final runs and the `FWD_PASS` row with their
  names, per-token times and ids, against round 3's 4,571 to 4,600.
- The per-operator table of a MoE layer from the finals' `report_table.md`
  beside round 3's (attention, merge, o_proj or the fold, w2, w13, qkva,
  router, the head).
- The standalone table from S4: every `TIME` line per build, the batch
  curve, the map, and the plain-against-streaming pairs.
- The decisions: every T-row's measured value and choice, in order.
- The ceiling: the stream rows' GB/s per XCD and per device, the policy
  that won.
- The numerics: the bit-diff table of G8 and the compare rows.
- The evidence of the JIT trap avoided: the `lnorm` and `w2silu` exec
  counters of the first `gv` and `w13` rows against round 3's.
