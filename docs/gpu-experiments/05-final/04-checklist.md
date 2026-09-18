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

- [ ] `run_fleet.py --final`: the thirteen flags and `-DMPK_W2_CK_TILE` unless named; `--no-event-timing`, `--no-nt-streams`; the run name's `_final`
- [ ] the `--worker-timing` help text corrected (MIN-35 named)
- [ ] `test_run_fleet_and_measure.py`: `--final` equals the spelled-out stack (298 operators, the finals' task count from the record's `plan.json`); the two overrides; the run name
- [ ] `harness/README.md`, `fleet/tasks/README.md`

Status: open.

## F1. The route log's tie rule

- [ ] `run_reference.py`: `w_all` per (step, MoE layer), the 64 softmax weights from the gate's input and weight
- [ ] `compare_route_log`: tie, cascade, disagreement; the counts in the report; FAIL on a disagreement only; the old format keeps the exact rule
- [ ] `test_compare.py`: the four cases (tie, disagreement, cascade, old format)
- [ ] the replay on the fifteen round-4 finals: zero disagreements
- [ ] `harness/README.md`

Status: open.

## F2. Iteration-aware boundaries

- [ ] `compare.run`: the iteration from the meta; `NOT_COMPARABLE` rows when the reference has no file for it; the overall by the ids and the route log; the report's header names the iteration
- [ ] `test_compare.py`: the it32 fixture, the it1 fixture
- [ ] the right form, if the box allows: `ref_boundaries_step31.safetensors` from `run_reference.py`, the file choice in `compare.run`, the test
- [ ] `harness/README.md`

Status: open.

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
- [ ] the worker's epilogue read in the disassembly (the two `printf` calls, the spills)
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
