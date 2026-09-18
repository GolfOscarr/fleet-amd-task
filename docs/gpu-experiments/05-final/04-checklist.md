# 04 - Local preparation checklist

The progress record for `03-local-preparation.md`. One row per
deliverable; a box is ticked only when its check has run on the laptop
and the result is written in the Status line with the date and, once
committed, the commit. The order is the order of work (`03`, Part 3).
Rows marked "VM" are ticked on the machine.

The commands that close every item:

```
.venv/bin/python -m pytest harness/tests fleet/tests env/hw/tests -q     # the suite (237 pass after round 4)
bash fleet/tasks/check_syntax.sh                                          # 15 PASS after round 4
bash env/offline_gfx942/run.sh                                            # every variant, 15 min
SHELLCHECK=1 bash env/preflight.sh                                        # the gate, 9 PASS
```

## F3. The `--final` preset

- [x] `run_fleet.py --final`: the thirteen flags and `-DMPK_W2_CK_TILE` unless named; `--no-event-timing`, `--no-nt-streams`, `--no-gemv-linears`; the run name's `_final`
- [x] the `--worker-timing` help text: correct as it stands (the defect an earlier draft saw was `--pad-alloc`'s line, read across two joined ranges); nothing changed
- [x] `test_run_fleet_and_measure.py`: `--final` equals the spelled-out stack (246 operators, 6,386 tasks: the record's `plan.json`); the overrides; the run name
- [x] `harness/README.md`, `fleet/tasks/README.md`

Status: done 2026-09-18. `FINAL_STACK` and `FINAL_DEFINE` in `run_fleet.py`; `apply_final(args, argv)` sets a stack flag only when neither the option nor its `--no-` form is named in `argv`, appends the define unless a `MPK_W2_CK_TILE` define is present, and does nothing on a synthetic graph; `parse_args(argv)` is the one entry (`main`, `queue.sh`'s `name_of`, `test_queue_files.py`), so the name the queue computes before a row is the name the run writes. The name: `L27_head_it30_final_tile_at_fn1_fn2_fs_nt_nts_mfma_rf_w2cktile_gv_lg48_mt_mh2` (the stack's slugs after `_final`, so a name still reads without the preset's table). Checks: 240 tests (three new: the preset equals the spelled-out stack field by field, every named flag and `--no-` form keeps its value and the define is not doubled, the stack's dry-run plan has 246 operators and 6,386 tasks); `DRY=1 queue.sh run` on a `--final --no-event-timing` row prints the expected name; the preflight 9 PASS with shellcheck.

## F1. The route log's tie rule

- [x] `run_reference.py`: `w_all` per (step, MoE layer), the 64 softmax weights from the gate's input and weight
- [x] `compare_route_log`: tie, cascade, disagreement; the counts in the report; FAIL on a disagreement only; the old format keeps the exact rule
- [x] `test_compare.py`: the four cases (tie, disagreement, cascade, old format)
- [x] the replay on the fifteen round-4 finals: zero disagreements
- [x] `harness/README.md`

Status: done 2026-09-18. `register_route_hooks` adds a pre-hook on each gate's input; `route_weights` computes the softmax over the 64 routed experts exactly as the model's gate does (`F.linear` and the softmax in float32; the checkpoint's config: `scoring_func` softmax, `topk_method` greedy, `norm_topk_prob` false, `routed_scaling_factor` 1.0), and `route_entry` raises if the gate's own top-k weights are not those values at their ids (the formula guard), then stores `w_all` rounded to 7 decimals beside `idx` and `w`. `compare_route_log(ref, fleet, router_floor)`: with `w_all` in every entry the tie rule (`tol_rel` = 4 x the calibrated router floor, `ROUTE_TOL_FALLBACK` 0.015 without a calibration; a single swap within the tolerance is a tie, any mismatch after a tie or cascade a cascade, the rest disagreements; only ties seed cascades), otherwise the exact rule with the result's `rule` field saying which; the report line carries the counts and the tolerance. Checks: 243 tests (three new: the tie, cascade and disagreement classes with a calibration file and with the fallback; the exact rule on a reference without weights; the replay over the fifteen 48-task finals of the record, every mismatch a single swap or a multi-expert difference after one); the smoke reference run writes `w_all` of length 8 whose values at the top-k ids equal the gate's weights, and the self-check ran on every step. The tolerance's fit to the real reference is read on the VM (R1: one tie expected at step 0, MoE layer 4).

## F2. Iteration-aware boundaries

- [x] `compare.run`: the iteration from the meta; `NOT_COMPARABLE` rows when the reference has no file for it; the overall by the ids and the route log; the report's header names the iteration
- [x] `test_compare.py`: the it32 fixture, the it1 fixture
- [ ] the right form, if the box allows: `ref_boundaries_step31.safetensors` from `run_reference.py`, the file choice in `compare.run`, the test. Deferred: the file choice and its test are in (a `ref_boundaries_step{N}` beside the step-0 file is used when present); the reference's dump of a later step needs the generated token stream through `capture_step0`'s path, and the session's compare rows are one-iteration rows anyway (R1), the finals judged by the ids (R2)
- [x] `harness/README.md`
- [x] the layers the reference did not capture: `NOT_CAPTURED`, reported and not counted (found from the record while writing the tests)

Status: done 2026-09-18 (the right form deferred with its reason). `common.boundary_iteration(key, iters)` is the rule: the cache rows at the handover position and the first token come from iteration 0, every other boundary from `iters - 1` (`run_fleet.boundary_dump`'s note says the same). `compare.run` reads `iters` from `fleet_run_meta.json`, loads `ref_boundaries_step{iters - 1}.safetensors` when the reference has it, and `compare_boundaries` compares a later-iteration boundary against that file or reports it `NOT_COMPARABLE`; a boundary of a layer the reference never captured (a 27-layer run dumps layers 2 to 26's rows; the reference has 0 and 1) reads `NOT_CAPTURED`; neither counts as a failure, and a report with nothing compared is not a PASS. The report's header names the iteration and the file; the Overall line carries both counts. Checks: 247 tests (four new: the rule; the it32 fixture with the head not comparable, the cache rows and the token compared, the verdict the ids' and a failing id still FAIL, the same fixture at one iteration failing the head as before; the step-31 file present and used; the uncaptured layers reported and a missing key inside a captured layer still failing). On the round-4 record the it1 model rows would now read 7 PASS, 64 NOT_CAPTURED and the route log by the tie rule, and the it32 rows the head NOT_COMPARABLE; the record's safetensors are not pulled, so that replay is R1's.

## F7. The head's event count

- [x] `graph_plan.py`: `argmax_slices` (default 50); `run_fleet.py --argmax-slices N`
- [x] `test_graph_plan.py`: the counts at 50 and 8; the dry run
- [x] `fleet/tasks/README.md`

Status: done 2026-09-18. `build_plan(..., argmax_slices=ARGMAX_SLICES)` threads the count through `dry_run` and `build` to the head's `argmax_partial` operator and the two `amax_*` shapes (the reduce's chunk parameter follows from the grid, 12,800 at 8); `run_fleet.py --argmax-slices N` passes it to both plan call sites and names the run `_as{N}` (no slug at the default). The stock kernels loop over their slice with a stride (`argmax_mi300.cuh`), so any divisor of the vocabulary is legal; a non-divisor fails the plan's assert. Checks: 249 tests (two new: at 8 the argmax has 8 tasks, `(1, 8)` outputs, the reduce's chunk 12,800, the operator count unchanged and the task count 42 lower, the gcd arithmetic 50 and 8 at 400 tasks and 10 and 8 at 320, a non-divisor raising; the flag's run name and its reach into the plan); the suite's dry runs. The event count itself is the runtime's and is read on the VM (R5: the head's events from 50 to 8 in the report table).

## F8. The session tooling

- [x] `queue-h1.txt` (R1), `queue-h2.txt` (R2), `queue-h3.txt` (R3, if F4), `queue-h4.txt` (R4), `queue-h5.txt` (R5); `queue-h6.txt` besides (the optional finals sets: batch 4, 8 head events)
- [x] `test_queue_files.py`: the round-5 set, the rows build their plans, no duplicate run names
- [x] `rehearse.sh`: the round-5 rows; `06-rehearsal.md` regenerated
- [x] `05-session-plan.md`: R0 to R5 with minute marks, the budget and the $3 rule's decision, the protocol, the playbook

Status: done 2026-09-18. Six queue files, 31 rows, every one on `--final` (the round-5 existence test asserts it); the preset gained one rule on the way (`--no-gemv-linears` drops the grid, which the plan asserts needs the GEMV linears; R4's rows would have failed their plan). `test_queue_files.py`: `ROUND5` beside `ROUND4`, the existence test, the plan-building test over both rounds with `argmax_slices` in its key, the no-duplicate-name check over every file. `rehearse.sh`: `ROUND=5` writes the round-5 transcript (R0 to the end, the six queue files, the `ktime nt` line) and leaves round 4's path as it was; `06-rehearsal.md` generated: 31 `run_fleet.py` lines, no guard line (33 after F5 rewrote `queue-h4.txt` and added `queue-h7.txt`: seven files, 33 rows). `05-session-plan.md`: the rows with minute marks from round 4's durations (R0 about 18 minutes with the `nt` suites only, R1 at 18, R2 at 21, the checkpoint pull at 30, R5, R4, R3, the optional sets at 42, the end), the budget against $4.69 with the balance decision as an open DECIDE row of S0, the DECIDE table (R1, A1, A2, R5, R4, the number), the playbook with round 4's additions (the watchdog, the anchored kill, the record's commit after the pull, the reading of a model compare). Checks: 250 tests, `bash -n`, the preflight 9 PASS with shellcheck, the rehearsal.

## F5. The half-merge fault (MIN-36)

- [x] the stock `register_linear_task` (`with_residual`) input map and `graph_plan`'s o_proj call read against the merge tile registration's whole-tensor maps
- [x] the dry runs of the 2-layer and 27-layer graphs with the task types printed, compared
- [x] the tile form's store offset at `halves = 2` read against the registration's output map
- [x] the fix and the suite row, if a reading finds the defect; otherwise the locator rows in `queue-h4.txt` (and the layer bisect in `queue-h7.txt`) and the readings attached to MIN-36

Status: done 2026-09-18, no defect found by reading; the fault is located on the VM (R4). The readings, each against the source or the record:

1. The maps. The stock `linear_with_residual_layer` (`persistent_kernel.py`) declares the input whole, the weight on dim 0, the residual and the output on dim 1; `build_graph.linear_gemv_layer` declares the same four maps. The runtime (`runtime.cc`, step 2.1) builds events only between consecutive operators, from the tensor the previous operator outputs and this one inputs, with one event per gcd of the two partitions: the tile merge's `attn` is whole (partition 1), the gang merge's is on dim 1 over 8 tasks, and the stock consumer's input is whole, so both give one event, with 32 triggers from the tile merge and 8 from the gang. The stock and the GEMV o_proj get the same event structure, and the record agrees: 297 events and 7,684 tasks in the faulting run and in its GEMV twin.
2. The task types. The dry runs of the 27-layer and 2-layer plans of the faulting configuration list the same fifteen methods (`argmax_partial`, `argmax_reduce`, `embed`, `gang_linear_silu`, `gang_moe_w13_linear`, `gang_moe_w2_silu_linear`, `linear_norm` at 400 and 96 tasks, `linear_with_residual` at 64, `mla_attend` at 33, `mla_merge_uv_tile` at 32, `mla_prep`, `moe_mul_sum_add`, `moe_router`, `rmsnorm`); the GEMV plan differs only by `linear_gemv` in place of the two stock linears. No type is registered at 27 layers that is not at 2, so the worker union's LDS is not it.
3. The store. `mla_merge_uv_head` stores at `attn + h * D_V + half * D_V / HALVES + wave * rows_per_wave + b * W_BATCH + lane`, at most `h * D_V + D_V - 1`: inside the head's 128 columns for either half; the registration asserts `attn` whole with `nh * d_v` columns and the grid `nh * halves`.
4. The layout, from the record. `fleet_run_meta.json` carries every tensor's address: in the faulting run `attn` sits 65,536 bytes before `act` (the 16-row backing of `new_workspace` is there; the stock tile reads 16 rows), and the 351 tensors have the same successors and distances as in the round-3 stack's run at 27 layers except four large inputs at block boundaries (`W_embed`, `W_uv_2`, `W13_3`, `c_kv_22`), none read by the stock linears. The stock CK tile's own over-reads (64 weight rows where o_proj's task holds 32) are the same in round 3.
5. The premise. Every 2-layer merge row of round 4 carried the GEMV linears; the only run of the faulting configuration is the 27-layer one, so "runs at 2 layers" (an earlier draft of `01` and MIN-36's entry) had no row behind it. The layer bisect rested on it, so R4 is redesigned: `queue-h4.txt` locates at 2 layers (the configuration; the graph cut after `L0.o_proj` and after `L0.mla_merge_uv`, whose verdict is the run's rc; `--merge-halves 1`), and `queue-h7.txt` bisects the layer count only if the 2-layer row passes. Checks: `test_queue_files.py` (the round-5 set now seven files; the plan-building key gained `stop_after`, so the cut rows' labels are asserted on the laptop), the rehearsal regenerated, the suite.

## F4. The worker-timing hang (MIN-35)

- [x] `run.sh`: the `unionT` variant; its resource line against `union`'s recorded here
- [x] the worker's epilogue read in the disassembly (the three `printf` calls, the spills)
- [x] the fix if found (the `printf` split, or the device buffer and the host printer), the patch regenerated on the pristine fork, `preflight.sh`
- [x] otherwise the readings attached to MIN-35 and R3 dropped (not needed: the fix is in; R3 runs, its first row the check)

Status: done 2026-09-18, the buffer form written; the VM's first `queue-h3` row verifies it (R3). The readings:

1. The resource line is not it. `unionT` (the round-4 union with `MPK_ENABLE_TIMING`, the build that hung) compiles, and its worker line against `union`'s, old form: 256 VGPRs both, AGPRs 226 against 171, scratch 80 bytes both, 8 VGPR spills both, SGPR spills 139 against 131, LDS 2,512 both. Nothing the launch has to size grew.
2. The record names the phase. The hung row's log (`env/hw/20260918/runs/L2_it32_tile_at_fn1_fn2_fs_nt_mfma_wt/fwd_pass.log`) has the host's three sanity lines, then 286 `[WORKER_XCD]` lines and a cut one, no `[SCHED_XCD]` line, no forward pass, no `[TASK_TIME2]`: the kernel stopped in its first phase. In the stock worker that line is a device `printf` (a hostcall) issued right before `threadfence_gpu(); atomicAdd(worker_xcd_ready_count)`, the barrier the scheduler waits on before dispatching its first task, and the timing hunk (round 3's I1) made all 304 workers print there instead of eight. Round 3's own timing log has 296 of those 304 lines, so the path was lossy already; the round-4 header added nothing to it (every round-4 hunk of the worker loop is a knob guard), and the class switch is not it (the same fifteen types run without the define).
3. The fix is the buffer form of `03`, step 2: the timing build has no device `printf`. `RuntimeConfig::worker_timing_buffer` (`WORKER_TIMING_WORDS` 40 per worker; `runtime_header.h`) is allocated beside the event timing buffer, cleared before every launch, freed at finalize; at its terminate task each worker's thread 0 writes its XCD, block, the seven `[TIMING]` counters, the six stock classes and our eight as (cycles, count) pairs, then the written flag; after both launch forms have synchronised, `print_worker_timing()` copies the slots to the host and prints `[WORKER_XCD]`, `[TIMING]`, `[TASK_TIME]` and `[TASK_TIME2]` per terminated worker in the formats `harness/measure.py` parses, then one `[TIMING_MISSING]` line if any slot is unwritten (a hang diagnostic for the next reader), and flushes stdout so the lines land in `fwd_pass.log`. The start line is back to the stock eight workers on every build. The non-timing build is unchanged. `new_tasks.patch` regenerated on the pristine fork (apply `gfx942.patch`, a temporary base commit, apply, edit, `git diff`, reset to 51dce4f with zero dirty tracked lines).
4. Checks: the offline pass with the new patch, 22 of 22 compiles exit 0 (`unionT` included; the host syntax check of `graph.cc`, `runtime.cc`, `task_register.cc`); the timing worker's printf hostcall references in the `unionT` disassembly 39 before, 21 after (the stock's remain); `unionT`'s new worker line 256 VGPRs, 234 AGPRs, scratch 80, 8 spills; the preflight 9 PASS with the three patches applying in order; `check_syntax.sh` 15 PASS; 251 tests, one new: the four format strings read from the patch, rendered with sample values, parse with `measure.py`'s regexes field by field, the worker's terminate region has no `printf` call, the missing line is not a record. Docs: `01` L1, `03` F4, `05` R3 (runs; the RULE is the watchdog), `02` R3, MIN-35, the patches' README, the offline README, the help texts of `run_fleet.py` and `measure.py`.

## F6. The merge's standalone 5 us

- [x] the `kt_r3` scratch variant from round 3's file; the wait sequences and the `attn_s` stores side by side
- [x] the `if constexpr` on the store if it is the cost; the suites' merge rows bit-exact (not the cost: no kernel change)
- [x] the cause, or the regression, written into `../04-kernels/10-results.md`'s standalone table

Status: done 2026-09-18, the cause read, no kernel change. `env/offline_gfx942/kt_r3.sh` extracts round 3's launcher tree (`fleet/tasks` at 8946804, the launcher's `k_mla_merge_uv` and its timing loop unchanged since) and disassembles it with `run.sh`'s launcher line into `dev_kt_r3.s`, beside `dev_kt.s` of the head. The two `k_mla_merge_uv` bodies:

| | round 3 | round 4 |
|---|---|---|
| instructions | 1,775 | 2,733 |
| `vmcnt` waits (the sequence) | 15: `0 1 0 15 14 13 12 8 4 0 14 12 8 4 0` (the lse, the partials batch, two `W_uv` batches retiring load by load) | 11: `0 0 0 0 0 0 1 0 0 0 0` (the partials batch with the lse words, two `W_uv` batches, each a full drain) |
| dependent HBM trips | 4 | 3 |
| `global_load` | 57 | 36 |
| `ds_read` / `ds_write` | 143 / 7 (the o values re-read per FMA) | 25 / 11 (the o values in registers) |
| `ds_bpermute` (shuffles) | 13: one `wave_max`, one `wave_sum`, in wave 0, once | 17: the halving butterfly's four dependent stages per `W_uv` batch |
| `v_exp_f32` sites | 1 (one weight per lane) | 18 (per thread: the maximum loop, the total loop, one per row of the batch) |
| `v_cndmask` | 14 | 518 |
| `s_cbranch` | 31 | 54 |
| VGPRs, scratch | 114, 0 | 124, 0 |

So the round trips are not it: the deeper form makes one trip fewer. The cost is the weights phase (c) of `mla_merge_uv_head`: every thread runs `for j < live` twice over `lse_s` with a dependent LDS read and `expf` per step (66 steps at the run's 33 splits, unrolled by eight) and once more per row of the FMA batch, where round 3 read one lse per lane of wave 0 and reduced with two 6-stage shuffles; the `attn_s` null test is a wave-uniform branch around the store, not a per-value select, and is not it. At the launcher's 16 workgroups (8 x 2, one wave per SIMD) those serial VALU and LDS chains are exposed, in the graph's 32 or 64 regular merge tasks they are hidden under the other tasks (22.0 us gap both rounds, 15.0 with the tile form). The fix would be round 3's weights phase under round 4's batch (one lse per lane, `wave_max`, one `expf`, `wave_sum`, the weights through LDS, about fifteen lines, the divisor's last bit back to round 3's tree order) and gains nothing in the graph, so it is not made in the final stage; recorded in `../04-kernels/10-results.md`'s standalone table. Checks: `kt_r3.sh` exits 0 (10 kernel headers of the round-3 tree); the summariser over both assemblies; no kernel file changed, so the suites and the offline pass stand as of F4.

## F9. The checklist and the gate

- [x] every box above ticked or explicitly deferred with its reason (one deferral: F2's right form, the reference's later-step dump, with its reason in F2)
- [x] the suite, `check_syntax.sh`, the offline compile of every variant (`unionT` included), `preflight.sh` 9 PASS with shellcheck
- [x] the fork at zero dirty tracked lines after the patch check
- [x] `07-final-numbers.md` drafted with the round-4 numbers and the round-5 rows empty
- [x] the memory note

Status: done 2026-09-18. The gate on the final tree (head bb59318 plus this commit): 251 tests; `check_syntax.sh` 15 PASS, 0 FAIL; the offline pass of F4 on the current patch, 22 of 22 units exit 0 (`unionT` the 18th megakernel variant, the host syntax check, the four launcher builds), no kernel or patch changed since; `SHELLCHECK=1 env/preflight.sh` 9 PASS (the three patches apply in order on a clean worktree); the fork at 51dce4f with zero dirty tracked lines (the untracked kernel copies of the day-1 setup stay). `07-final-numbers.md` drafted: the number's table with round 3's and round 4's rows filled and A, B and the optional sets empty; the compare rows with round 4's reading and the expected counts; the head's events against round 4's 49 events and 111 us; the fault's locator rows; the timing build and the merge's `ktime` line; the DECIDE table with the round-4 column filled. The set's README (the `07` row, the status row of the day), `PROGRESS.md`, the root README's pointer row and the memory note updated. The counts of the checklist's header (237 tests, 15 PASS) are the round-4 figures the page was drafted with; the current ones are 251 and 15.

## Before the VM

- [x] the user's decision on the balance ($4.69: the $3 rule waived for the last session, or credits added): taken 2026-09-18, the whole balance and no credits; `05-session-plan.md` revised with the hard stop (minutes 70, 78, 82), the optional blocks split by their gates (`queue-h6` batch 4, `queue-h8` the 8-event set, the plain suites, `queue-h9` A's set again)
- [x] the user's go for the VM: 2026-09-18 11:08 UTC ("Poll VM proceed right away"); the 13-core host provisioned by hand at 11:09
- [x] `FULL=1 L push`, `L start setup`, `V preflight`, `V checks`, `V reference` (F1's weights), `V kernels`: all PASS by minute 21 (`08-session-log.md`)
- [x] the first report at R1's DECIDE row (the compares green), before the finals: minute 25, zero disagreements on both model rows, 19 boundaries PASS at 2 layers

Status: the session ran 2026-09-18 on `gpu/round-5`, 77 minutes, $3.74; the results in `07-final-numbers.md`, the log in `08-session-log.md`.
