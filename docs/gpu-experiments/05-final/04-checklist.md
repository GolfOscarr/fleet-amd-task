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

- [ ] `graph_plan.py`: `argmax_slices` (default 50); `run_fleet.py --argmax-slices N`
- [ ] `test_graph_plan.py`: the counts at 50 and 8; the dry run
- [ ] `fleet/tasks/README.md`

Status: open.

## F8. The session tooling

- [ ] `queue-h1.txt` (R1), `queue-h2.txt` (R2), `queue-h3.txt` (R3, if F4), `queue-h4.txt` (R4), `queue-h5.txt` (R5)
- [ ] `test_queue_files.py`: the round-5 set, the rows build their plans, no duplicate run names
- [ ] `rehearse.sh`: the round-5 rows; `06-rehearsal.md` regenerated
- [ ] `05-session-plan.md`: R0 to R5 with minute marks, the budget and the $3 rule's decision, the protocol, the playbook

Status: open.

## F5. The half-merge fault (MIN-36)

- [ ] the stock `register_linear_task` (`with_residual`) input map and `graph_plan`'s o_proj call read against the merge tile registration's whole-tensor maps
- [ ] the dry runs of the 2-layer and 27-layer graphs with the task types printed, compared
- [ ] the tile form's store offset at `halves = 2` read against the registration's output map
- [ ] the fix and the suite row, if a reading finds the defect; otherwise the bisect rows in `queue-h4.txt` and the readings attached to MIN-36

Status: open.

## F4. The worker-timing hang (MIN-35)

- [ ] `run.sh`: the `unionT` variant; its resource line against `union`'s recorded here
- [ ] the worker's epilogue read in the disassembly (the three `printf` calls, the spills)
- [ ] the fix if found (the `printf` split, or the device buffer and the host printer), the patch regenerated on the pristine fork, `preflight.sh`
- [ ] otherwise the readings attached to MIN-35 and R3 dropped

Status: open.

## F6. The merge's standalone 5 us

- [ ] the `kt_r3` scratch variant from round 3's file; the wait sequences and the `attn_s` stores side by side
- [ ] the `if constexpr` on the store if it is the cost; the suites' merge rows bit-exact
- [ ] the cause, or the regression, written into `../04-kernels/10-results.md`'s standalone table

Status: open.

## F9. The checklist and the gate

- [ ] every box above ticked or explicitly deferred with its reason
- [ ] the suite, `check_syntax.sh`, the offline compile of every variant (`unionT` included), `preflight.sh` 9 PASS with shellcheck
- [ ] the fork at zero dirty tracked lines after the patch check
- [ ] `07-final-numbers.md` drafted with the round-4 numbers and the round-5 rows empty
- [ ] the memory note

Status: open.

## Before the VM

- [ ] the user's decision on the balance ($4.69: the $3 rule waived for the last session, or credits added)
- [ ] the user's go for the VM
- [ ] `FULL=1 L push`, `L start setup`, `V preflight`, `V checks`, `V reference` (F1's weights), `V kernels`
- [ ] the first report at R1's DECIDE row (the compares green), before the finals
