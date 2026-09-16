# 02 - Session plan: two sessions on a 1x MI300X for $27

**Ran on 2026-09-16.** The rows below are the plan as written before the
sessions; what happened is `03-session-log.md`, the numbers `04-results.md`,
the one-page outcome `07-summary.md`. Session B ran inside session A's VM on
the user's decision at minute 58; B3 was not measured (rocprofv3 on the torch
wheel); the whole round took 149 minutes and $7.33 instead of the 9 hours
budgeted.

Rewritten 2026-09-16 after the preparation (`01-preparation.md`, P1 to P8
and the verification pass). Every row below is one literal command of
`env/session/laptop.sh`, the text that decides PASS, whether the row runs
on its own (AUTO) or needs a decision (DECIDE, with the rule and who
decides), and what to do on FAIL. The exact commands the scripts expand to
are in `05-rehearsal.md`, generated in DRY mode; the run log goes to
`03-session-log.md` and the numbers to `04-results.md`, both prepared as
skeletons with the 2026-09-15 baseline already in their cells.

## Budget

| | |
|---|---|
| Balance | $27.56 (read from the team page on 2026-09-16, before any session; the gate of row A0) |
| Shape | 1x MI300X, $2.99 per hour, one-hour minimum, per-minute after |
| Hours | 9.2 |
| Session A | up to 4.0 hours of VM time ($11.96): the image, the fault, M4, the growth curve, the attribution check, the first timings |
| Session B | up to 3.5 hours ($10.47): the measurements and the second lever |
| Reserve | 1.5 hours ($4.49): a rescue if a session ends with an unsynced record or an unpushed image |

Running out of balance deletes the VM with everything on it, so the
record is pulled every 30 minutes and the balance is read before the
image push and at minute 180.

## Decisions taken by the user on 2026-09-16

| Decision | Choice |
|---|---|
| Image build in session A | yes: build and push in the background from minute 2, with the A9 cutoff |
| Session A length | 4.0 hours of VM time, hard stop at minute 240 |
| A7b, rocgdb after the four fixes fail | spend up to 60 minutes only if the bisection named a label in layer 7 (one of our kernels' offsets); a head label goes to the fallback |
| A4, the P6 kernels fail their suites | the rule: prefetch depth back to 1, continue with the old kernels, P6 becomes session B work |
| B5, the second lever | chosen by the user from the A8 numbers when they exist |
| Deletion of the VM | asked every time, at the end of each session; never on the agent's own judgment |
| Reporting | the protocol below, unchanged |
| Commits and pushes | `laptop.sh pull` commits when it brings more than logs (a new run, a changed record, reference files); a logs-only pull stages and does not commit; the branch is pushed at the end of each session |

## Reporting protocol

- Before `laptop.sh provision`: the balance and the GHCR check, and a
  one-line "provisioning now" to the user. Nothing is billed before that
  command.
- After every stage row: the `PASS`/`FAIL` line of `env/logs/session.status`
  pasted verbatim, with the minute mark and the balance estimate.
- At every DECIDE row: the rule, the measured value, and the choice, before
  the next command runs. The user can follow from another device on those
  three lines alone.
- At the end: the last `laptop.sh status`, the `Hourly Rate: $0.00` line,
  and the commit hash of the pulled record.

## What changes with one GPU

- The reference run, the calibration, the kernel tests and the probes ran
  on GPU 1 last time; they now run in the `reference` and `kernels` stages
  before the queue, about 4 minutes on the same GPU.
- The "not the clocks" check of MAJ-7 (a bandwidth probe on the other GPU
  during a timing run) cannot be repeated; it is recorded and cited.
- `HIP_VISIBLE_DEVICES=0` is the only device; the scripts set it.
- The host has fewer cores than the 2x (26 last time), so `setup.sh` and
  the image build, which both compile Fleet, take longer when they
  overlap. They overlap anyway: the GPU is idle until `setup.sh` ends.
- The hardware record is re-collected only if the host name or the ROCm
  version differs (the `hw` stage decides by itself, one minute).

## Session A (up to 4.0 hours of VM time)

Every command is run from the repository root on the laptop. `L` stands
for `bash env/session/laptop.sh`. Status rows are read with `L status`;
the "PASS when" column is the row that must appear in it.

| Minute | Row | Command | PASS when | Mode | On FAIL |
|---|---|---|---|---|---|
| -5 | A0 | `L balance`; `gh api /user/packages?package_type=container` | balance about $27, `No virtual machines`; the package list read | DECIDE (user: go) | a VM already listed: stop, nothing is provisioned |
| 0 | A0 | `L provision` | prints an address; `L push` completes in about a minute | AUTO | `provision` refuses if a VM exists; a TUI timeout: retry once, then stop |
| 2 | A1 | `L start download`; `L login`; `L start image`; `L start setup`; `L start hw` | `PASS download` (about 2 min); `PASS hw` (1 min, or `skipped: ... already recorded`) | AUTO | download: `df -h /` needs 100 GB; image: it is background, keep going; setup: read `env/logs/setup.out` on the VM, apply the manual fix from `env/docker/README.md`, `L start setup` again |
| 25 | A2 | after `PASS setup`: `L start checks` | `PASS checks` with 7 `PASS` lines in `env/logs/check_day1.out` | AUTO | a check that passed on 2026-09-15: a machine difference, log it in `03`; continue unless it is check 2 (import) |
| 27 | A3 | `L start reference` | `PASS reference` (about 3 min) | AUTO | FAIL: no `compare` row can run; the queue guard will say so; fix from the log or run the session without compares |
| 30 | A4 | `L start kernels` | `PASS kernels`: 7 suites, 100 trials each, in `fleet/tasks/results/kernel_tests.json` | DECIDE (agent: rule below) | a FAIL in `mla_attend`, `mla_merge_uv` or `mla_attend_splits` is the P6 kernels: set `PF` back to 1 in the two kernels, `L push`, `L start kernels`; if the old kernels pass, session A runs on them and P6 goes to session B |
| 33 | A5 | `L start queue env/session/queue-a.txt` (A5.1 the fault reproduces; A5.2 the address shift, four runs) | one status row per run; A5.1 `FAIL ... fault=1`; A5.2 read against the tree below | DECIDE (agent: the tree) | the queue continues past every row (all are `continue`) |
| 42 | A6 | `L start bisect env/session/queue-fault.txt -- --layers 8 --head --iters 2` | `BISECT first-fault=<label>` in `env/logs/bisect.result`, about 5 runs | AUTO | `no-fault-up-to=head.argmax_reduce`: the fault needs the untruncated graph; skip to A7 with all four fix rows |
| 50 | A7 | `L start queue env/session/queue-fix.txt` (the four pre-baked fixes, in order) | the first row with `PASS ... fwd=2` | DECIDE (agent: first PASS wins; user is told which) | all four FAIL: time box; record the frontier, the addresses and the label, and go to A8 with the 27-layer graph at 2 iterations instead of 32 (the fallbacks) |
| 60 | A7b | only if A7 found nothing and A6 named a label in layer 7: 60 minutes at most of `rocgdb` with `MPK_EXTRA_HIPCC_FLAGS=-gline-tables-only` on that label (the recipe in `01-preparation.md`, P1) | the faulting source line | AUTO (decided 2026-09-16: layer-7 label yes, head label no) | past the box, or a head label: fallbacks |
| 90 | A8 | `L pull` (30-minute checkpoint), then `L start queue env/session/queue-a2.txt` with the winning flag added to its rows by `queue_flag.py` (A8.1 the fixed 8-layer graph; A8.2 M4; A8.3 the growth curve; A8.4 B0 attribution, two runs; A8.5 the 2-layer timing with and without `--tile-linears`; A8.6 the 27-layer baseline of this machine) | A8.2 `compare=PASS` and 32 ids equal to `harness/ref/ref_output_ids.json`; A8.3 27 per-layer errors under the threshold; A8.6 the per-iteration time in `report_table.md` | AUTO to A8.3, DECIDE at A8.4 and A8.5 (rules below) | A8.2 a mismatch at token k: the run stands, the index goes to `04`; A8.3 the first layer above threshold is named |
| 120 | A9 | `L pull`; `L balance`; image status in `L status` | `PASS image` (build, push, logout); balance above $16 | DECIDE (agent: the push cutoff) | not built: read `env/logs/image.out`, fix on the laptop, `L push`, `L start image` only if more than 60 minutes remain; a push slower than 30 MB in the first 5 minutes is stopped (`docker save` is not attempted) |
| 150 | A10 | `L pull` | a commit on the branch | AUTO | |
| 180 | A11 | `L pull`; `L balance` | balance above $13 (session B needs $10.47 plus the reserve) | DECIDE (agent: hard stop at $13) | below: stop the queue now |
| 200 to 235 | A12 | whatever `queue-a2.txt` has left; nothing new is started after minute 220 | | AUTO | |
| 240 | end | `L pull`; `git push`; then, after the user's yes, `L delete --yes`; `L balance` shows `Hourly Rate: $0.00` | | DECIDE (user: the deletion, asked every time) | the delete dialog needs `y`; if the rate is not $0.00, check the TUI by hand |

Hard stops: at 4 hours of VM time the queue is stopped, the record pulled
and the VM deleted whatever the state; at any point when the balance shows
less than $13 before session B, the same.

### The fault, as a decision tree (A5 to A7)

| What A5 and A6 said | Meaning | Next |
|---|---|---|
| A5.1 passes on this machine | the fault was machine-specific | log it; A8 directly, no fix flag |
| A5.1 faults, any A5.2 pad run passes, or the 16-layer pad run faults | address-dependent | A6 for the label, then A7 in order: `--align-alloc 65536`, `--workspaces-first`, both, 2 MiB |
| A5.1 faults, every A5.2 run faults and the 16-layer pad run passes | not the address; the layer count itself | A6 for the label; A7 anyway (cheap); then A7b with the label, reading its pointer offsets in the generated `kernel_0.cu` |
| A6 finds a label in layer 7 | the operator whose task faults | its kernel's offsets against the addresses in `fleet_run_meta.json` (`addresses`), the only thing the fix has to explain |
| A6 finds a head label | the head path | the same, on `lm_head` / `argmax_partial` / `argmax_reduce` |

### Thresholds for the timing DECIDE rows (baseline 2026-09-15, `runs/L27_it32` and `L2_it32`)

| Row | Quantity | Baseline | Decision |
|---|---|---|---|
| A8.4 (B0) | trace time of the 2-layer graph stopped after `L0.qkva` minus after `L0.norm1`, per iteration | event gap said 4.4 us for the 15 MB `qkva` | near 4 us: the per-operator table is trusted as read; near 40 us: the event gaps shift time to the successor, every per-operator number is re-read before anything is optimized |
| A8.5 | `o_proj` (event 7) with `--tile-linears` | 37 to 40 us (gang, residual variant) | under 10 us: session B extends per-tile to the MoE linears and the elementwise ops; above: the per-tile path is not the lever, session B goes to the boundary fusions |
| A8.5 | `mla_attend` (event 5), `mla_merge_uv` (event 6) with the P6 kernels | 211 us, 64 us | under 60 us and under 20 us: P6 holds; above: session B's B1 decides between the column-mapped score and the MFMA path |
| A8.6 | per-iteration time, 27 layers, 32 iterations | 15.6 ms | the number every session B variant is compared with |

## Session B (up to 3.5 hours)

Starts from the image if it was pushed (`L login`; the `setup` stage is
skipped; `L start checks` at about minute 8) and from `setup.sh`
otherwise (minute 25 as in A). Then `L start reference`, `L start kernels`,
and `L start queue env/session/queue-b.txt`; the flags that won in session
A are added to its rows with `queue_flag.py` before the push.

| Minute | Row | PASS when | Mode | On FAIL |
|---|---|---|---|---|
| 8 or 25 | `checks`, `reference`, `kernels` | three PASS rows | AUTO | as in A |
| 15 or 32 | B1: the 2-layer timing with the P6 kernels | `table=PASS`; the thresholds above | DECIDE (agent) | |
| 30 | B2: 27 layers, 32 iterations, event timing | the median and P95 per iteration in `report_table.md` | AUTO | |
| 45 | B3: the `measure` row: kernel trace and the four PMC pairs, then `measure.py` | bytes per iteration, achieved bandwidth, `3 megakernel of N dispatches` | AUTO | the guard fails the row if rocprofv3 is missing |
| 60 | B4: correctness with every flag on: 27 layers, head, 32 iterations, `compare` | 32 ids equal, every boundary PASS | AUTO | a mismatch: the flag that broke it is bisected over the rows of B4 (one flag at a time) |
| 75 | B5: the second lever if time remains: per-tile MoE linears and elementwise ops, or `--nt-weights` (the design's experiment E2, MAJ-6), each as a 2-layer timing then a 27-layer one | `report_table.md` per variant | DECIDE (user: which lever, from the A8 numbers) | |
| 150 | `L pull`; `git push`; after the user's yes, `L delete --yes`; the rate is $0.00 | | DECIDE (user: the deletion) | |

Every timing row is the same command with one flag changed, so
`04-results.md` is a table of variants against the 15.6 ms baseline and
the 1.15 to 1.35 ms band of the design.

## The queue files

| File | Rows | Used at |
|---|---|---|
| `env/session/queue-a.txt` | the fault reproduced; the four address shifts | A5 |
| `env/session/queue-fault.txt` | the 16 labels of layer 7 and the head, in plan order (the bisection's input, not a queue) | A6 |
| `env/session/queue-fix.txt` | the four pre-baked fixes | A7 |
| `env/session/queue-a2.txt` | the fixed graph, M4, the growth curve, B0, the two 2-layer timings, the 27-layer baseline | A8 |
| `env/session/queue-b.txt` | B1 to B5 | session B (as planned; the rows actually run are below) |
| `env/session/queue-fault-all.txt` | every label of the 8-layer graph (the layer-7 list faulted at its first label) | A6, the wide bisection |
| `env/session/queue-b0.txt` | the stop-after ladder of layer 0 at 32 iterations | A8.4, B0 from the host clock |
| `env/session/queue-b2.txt` | B2, B4, the E2 lever on 2 and 27 layers | session B in the same VM |
| `env/session/queue-fix2.txt` | the plan-side fault fix without any flag | after B5 |
| `env/session/queue-b3.txt`, `queue-b4.txt`, `queue-b5.txt` | E2 and per-tile linears together; 61 splits; the attention as regular tasks | B5 |

One `run_fleet.py` argument line per row with the optional trailing words
`compare`, `table`, `measure`, `continue`. The guards of `queue.sh` fail a
row before it runs when `--iters` is above 32, `--debug` has more than one
iteration, a compare has no reference tensors, or a measure has no
profiler. `05-rehearsal.md` shows every row expanded.

## Helpers (so nothing is typed by hand on the clock)

| Command | Use |
|---|---|
| `L wait <stage> [min]` | polls `vm.sh check` every 30 s until the stage's PASS or FAIL row of its last start; the way to wait for `setup`, `kernels`, a queue |
| `L report [--balance]` | the status message of the protocol: minute since provisioning, cost so far, the last status rows, the bisection result, optionally the balance |
| `L ssh "cd /home/hotaisle/metalOps && bash env/session/vm.sh preflight"` | ten seconds at minute 2: GPU visible, disk, docker, hipcc, rocprofv3, rocgdb, the venvs, the model, the reference |
| `L ssh "... vm.sh kill <pattern>\|--all"` | stops a hung graph run by anchored pid (never `pkill -f`) and writes a `KILLED` row |
| `L ssh "... vm.sh gdb <label>"` | row A7b: compiles with line tables (`MPK_EXTRA_HIPCC_FLAGS`), runs the truncated graph under rocgdb, saves the backtrace to `env/logs/gdb_<label>.out` and the record |
| `bash env/session/pf.sh 1` | row A4's fallback: prefetch depth 1 in both attention kernels (then `L push`, `L start kernels`); `pf.sh 4` restores |
| `python3 env/session/queue_flag.py <queue> <flags> --in-place` | row A8: the winning fix flag into every run row of `queue-a2.txt` and `queue-b.txt` (idempotent, keywords and comments kept) |
| `python3 env/session/addr_diff.py <A>/fleet_run_meta.json <B>/fleet_run_meta.json` | row A5.2: which tensors moved between two runs, by how much, and whether a 4 GiB boundary was crossed |
| `bash env/session/grab.sh` (added on the day) | polls the provisioning list every 18 s and provisions the first 1x MI300X; the log is `env/logs/grab.log` |
| `KT_TIME=N [KT_COLD=K] fleet/tasks/build/kernel_tests mla_attend <trial dir>` (added on the day) | the standalone time of the attention or merge grid, warm or over K copies of the cache (`fleet/tasks/README.md`) |
| `run_fleet.py --split N`, `--attend-tasks` (added on the day) | positions per attention split (61 splits at 17); the attention as one regular task per split |

## Failure playbook (round 1, one line each)

| If you see | Do |
|---|---|
| `setup.sh` stops at the first pip install: the venv has no pip | `sudo apt-get update && sudo apt-get install python3.12-venv`, then `L start setup` again (step 2 does this itself now) |
| `import mirage` fails on `libz3.so.5.1` | the build ran with pip isolation; `setup.sh` uses `--no-build-isolation`; if it recurs, `pip install --no-deps z3-solver==5.1.0.0` in `.venv-fleet` and `LD_LIBRARY_PATH` in `activate` |
| the JIT fails with `paged_attention_minimal_decode` undeclared | the gfx942 patch is not applied: `setup.sh` step 5 resets and re-applies; check `git -C repos/fleet-chiplet-megakernel status` |
| `check_day1` check 3: worker mod 8 is not the XCD | the placement offset is per launch; the check is offset-tolerant; a FAIL means `sched_xcd.patch` is missing |
| a queue row FAIL with `fault=1` before any `FWD_PASS` | the M4 fault; the decision tree above |
| two graph runs at once | never: the JIT directory is shared; the queue serialises; do not start a second queue |
| an ssh command hangs or the session drops | a `pkill -f` matched its own command line, or a background job kept a descriptor: the scripts detach with `setsid nohup ... </dev/null`; kill by pid from `pgrep -f "^python harness/run_fleet.py"` |
| `rsync` nested the record into itself | a relative destination; `laptop.sh pull` uses absolute paths only |
| the record directory changed name mid-session | it cannot: `common.sh` pins it in `env/logs/record.dir` on the first call |
| `--iters 64` asserts on the cos table | the tables hold 1,056 positions; 32 is the ceiling (the guard refuses) |

## What is pulled back and committed

`L pull` does it: `env/logs/*.out` and the two status files (into
`env/hw/<date>/logs/` through `vm.sh snapshot-logs`); `env/hw/<date>/`
(the record if `hw` ran; `runs/<name>/` for every queue row with
`plan.json`, `wall.json`, `fleet_run_meta.json`, `fwd_pass.log`,
`correctness_report.md` and `.json`, `report_table.md`, `metrics.json`,
`event_timing.json`, the profiler CSVs under `prof/`); `harness/ref/`
JSON files; `env/check_day1.log`; `fleet/tasks/results/`. Then one commit
on the branch. Not pulled: the tensors, the build directories, files above
400 KB.

## Fallbacks decided now

- M4 not fixed in session A: session B runs B2 to B5 on the 27-layer
  graph without the head (it runs 32 iterations) and the head at 2
  iterations, and the write-up reports the frontier, the bisection label
  and the addresses.
- The image fails a fifth time: session B pays the 20 minutes of
  `setup.sh`; the Dockerfile is not debugged on VM time beyond reading
  the log.
- The P6 kernels fail their tests: `PF = 1` restores the old kernels (the
  loops are the same with one load in flight); P6 becomes session B work
  (decided by the user, 2026-09-16).
- The per-tile flag does not help (A8.5): session B's second lever is the
  boundary fusions of `docs/design-doc/09-expected-performance.md`,
  reductions 1 and 2.
- The balance is short before session B: session B is cut to B2, B3 and
  B4, about 1.5 hours.

## Expected outputs (2026-09-15, to recognise a deviation at once)

| Stage or row | Expected |
|---|---|
| `checks` | 7 PASS lines: SPX+NPS1; `import mirage`; 8 schedulers on their XCDs, 296 workers, offset 5; `buffer_wbl2 sc1` x14 and `buffer_inv sc1` x4; CK FMHA 576/512 does not instantiate (PASS as the expected negative); all counter names present; 304 CUs |
| `reference` | 32 ids, `generate` agrees with the loop; calibration floors router 3.59e-3, scores 2.61e-3, logits 4.1e-2, norm 4.5e-2, layer 4.3e-3, gemv 3.2e-3, rope 5.3e-3, attention 1.0e-2, expert 1.1e-2; route overlap 0.21 |
| `kernels` | 7 suites, 100 of 100: `mla_prep`, `mla_attend`, `mla_merge_uv`, `moe_router`, `copy`, `mla_attend_scores`, `mla_attend_splits` |
| a 2-layer run with `compare` | all 16 boundaries PASS, top-k of layer 1 exact `[2, 19, 39, 40, 47, 49]`, route log PASS |
| 27 layers with the head, 1 and 2 iterations | ids `[25]`, then `[25, 16228]` |
| the fault | 8 layers with the head at 2 iterations: `AcceleratorError` before the first `FWD_PASS` |
| 27 layers, 32 iterations, event timing | 15.6 ms per iteration; per operator `mla_attend` 211 us, `mla_merge_uv` 64, `o_proj` 40, `down` 37, `moe_silu_mul` 42, norms 14 to 50, `qkva` 4.4, `w13` 25, `w2` 5, `moe_mul_sum_add` 24, `mla_prep` 14 to 25, router 3.7 |
| the first session's cost | $12.56 for 4.5 hours on the 2x; about $6 for the same on the 1x |
