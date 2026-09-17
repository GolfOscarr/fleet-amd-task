# 02 - The work split: laptop first, then the VM

Written 2026-09-17 from `01-gemv-ideas.md` after its double-check (the
section "What the double-check changed" there: K2 is off, the batch depth
is set by the unroll pragma and its register cost is measured, w13's first
round is at the XCD's rate and that rate is unmeasured, the bit-diff needs
a dump row). The rule of rounds 2 and 3 holds: no minute of VM time goes
to work that can be done on the laptop. Every item below has a
deliverable, a check that runs on the laptop, a time box and the VM row
that consumes it; every VM row names the laptop items it needs, the PASS
text and the decision it feeds. This page covers the GEMV linear only; the
router and the merge get their own ideas page and join the session plan
after it.

## What carries over from round 3

- The session scripts (`env/session/`: `laptop.sh`, `vm.sh`, `queue.sh`,
  `grab.sh`) and their rules: push before `L start queue`, one graph run
  at a time, count DONE lines, a 2-layer compare needs `--iters 1`, the
  finals at distinct iteration counts, `fleet_env` installs the headers
  before every stage, a changed `new_tasks.patch` needs `FULL=1 L push`
  then `L start setup` (73 s; 12 minutes on a fresh host with the
  download and the hardware census: 95 + 158 + 483 s in round 3).
- The checks: the tests (`fleet/tests`, `harness/tests`, `env/hw/tests`),
  `fleet/tasks/check_syntax.sh`, the dry run (`DRY=1`), the offline gfx942
  compile (`env/offline_gfx942/run.sh`) and now the probe method
  (`env/offline_gfx942/gemv_probe/`).
- The reporting rules: report before the GPU is touched and at every
  DECIDE and RULE row; no VM, no deletion and no new milestone without
  the user's word; commit pulls only with meaningful progress; no names,
  no emoji, `docs/report` never committed.

## Local part (the laptop, before the VM)

Ordered by what the VM rows need first. L1 to L5 are the kernel and its
forms; L6 and L7 are the two instruments the double-check asked for; L8
and L9 are the session tooling and the gate.

| Item | Deliverable | Laptop check | Time box | Feeds |
|---|---|---|---|---|
| **L1. The GEMV tile kernel** (K1, K5, K6, K7, K8 without MOE) | `fleet/tasks/mi300/linear_gemv_mi300.cuh`: `linear_gemv_mi300_task_impl<T, K, NORM, RESIDUAL>(x, w_norm, W, residual, out, rows, o_stride, eps)`; a wave owns `rows / 4` rows (the uneven split for 38), a lane four 8-element chunks of x at `8 l + 512 i` (K9: every wave-load one contiguous KB; `GEMV_STRIDED` keeps the router's map for the A/B) in 32 FP32 VGPRs (from the NORM prologue's LDS row or from global), the first batch's loads and the residual's issued before the prologue, 8 rows per batch under `#pragma unroll 1` with the batch as a constant (`GEMV_BATCH`), the weight loads through `StreamSrc` (`sc1 nt`), the FMA on raw words in ascending k, `wave_sum`, the epilogue with the residual add and `nt_store`; the tail rows and columns masked; no scratch row | `check_syntax.sh` (the host stub); the suite rows of L1b; the offline variant of L1c | 5 h | G0, G1 |
| **L1b. Suite rows** (I4) | `k_linear_gemv` in `kernel_tests_mi300.cu` and `kernel_tests.py` entries `linear_gemv`, `linear_gemv_norm`, `linear_gemv_res` against `numpy_ref.linear_norm` and two one-line references; the `ktime` mode for the tile alone (cold cache, `KT_COLD`) at `GEMV_BATCH` 4, 8, 16 and once with `GEMV_STRIDED` (four suite binaries, as the `nt` and `mfma` builds are) | the suite's dry run; the numpy references' tests | 3 h | G0 (M5) |
| **L1c. The union, offline** (I3) | `MK_GEMV` in `env/offline_gfx942/mk_tu.cu` instantiating the qkva (38 rows, NORM), o_proj (32, RESIDUAL) and head (256, NORM) forms at the model's dims; `run.sh` gains the variant; `resources.txt` the lines; the `s_waitcnt vmcnt` sequence of the tile loop read from `dev_gemv.s` and written into the checklist | the compile exits 0; VGPRs at most 256, no scratch growth over 64 bytes per lane, the wait sequence shows 32 loads in flight (`vmcnt(28) (24) ...` as the probe's `nh8`) | 1 h | the gate |
| **L2. The task type and the plan flag** (I1, I2) | `TASK_LINEAR_GEMV_MI300` (195) in `new_tasks.patch`: `register_linear_gemv_mi300_task` with the input count set by the flags (x; w_norm; W; residual), the weight partitioned on dim 0 by the grid and the output on dim 1 (the rule of `register_linear_norm_mi300_task`), params `[norm, residual, eps_bits]`; `graph_plan.py --gemv-linears` issues qkva, o_proj, layer 0's down and the head as `linear_gemv_layer` calls with no scratch tensors, `--linear-grid N` overrides `grid_for_linear` for S2; `build_graph.py` the method; the counts in `test_graph_plan.py` and `graph_counts.py` | the patch applies on the pristine fork (the regeneration recipe of round 3); the dry run's counts; `task_graph_check.py`; the tests | 4 h | G1, G2, G6 |
| **L3. The w2 form** (K7's 1,408 map, K8 SILU and MOE) | the multiply of `gang_moe_w2_silu_mi300.cuh` replaced by the GEMV loop with the activation row computed into LDS (no scratch write, no `s_waitcnt 0`, no sc0 read); the 176-chunk lane map; the stock epilogue kept; the registration keeps its scratch output until the plan drops it under the flag | `check_syntax.sh`; a suite row `gang_w2_silu_gemv` (the existing w2 reference in `numpy_ref`, one expert, one slot); the offline variant (`MK_CK_GANG` becomes the GEMV form) | 4 h | G3 |
| **L4. The w13 form and one round per XCD** (K8 MOE, S1) | `gang_moe_w13_gemv_mi300.cuh` (type 196, gang): the stock decode of expert and slot, the tile's rows from `tile_idx` by the 33 x 76 + 4 x 77 arithmetic (a constant `TILES_PER_XCD` = 37 and `ROWS` = 2,816 in the template), the 19- or 20-row wave split, the scatter epilogue; the registration takes `n_tiles` = 37 (`_gang_moe` gains a `tiles` argument), asserts the worker count per XCD | `check_syntax.sh`; a suite row `gang_w13_gemv` (the stock w13 reference, one expert); the dry run's tile counts; the offline variant; a test that the 37 tiles cover 2,816 rows exactly once | 4 h | G4 |
| **L5. The head's passes** (S3) | the 256-row task as four passes of the tile with the norm computed once (part of L1's template: `rows` up to 256, the batch loop over passes); `--linear-grid` accepts 320 for the head | the suite row at 256 rows; the offline variant's registers (the passes must not multiply the live registers) | 2 h | G6 |
| **L6. The stream probe** (M7; decision 5 of `01`) | a plan mode `--graph stream --kb N --gang` on the empty-graph machinery of round 3: a gang task type `TASK_STREAM_MI300` (197) whose tile reads N KB with the GEMV's load loop into an XOR word (the prefetch kernel's device against elision), 37 tiles per XCD, and the regular form of 296 tasks; `measure.py` prints GB/s per operator from the event gap | the dry run; a test of the arithmetic; the offline variant | 3 h | G5 |
| **L7. The bit-diff** (I5) | `harness/bitdiff.py a b`: for every key of two `fleet_boundaries.safetensors`, the count of differing elements and the max ULP distance; a queue action `bitdiff <run-a> <run-b>` in `queue.sh` that runs it on the VM's `harness/fleet_out` directories and writes the result into the record | a test on two synthetic files | 2 h | G8 |
| **L8. The session tooling** | `queue-f1.txt` (G0's suites are stages, not rows), `queue-f2.txt` (G1, G2), `queue-f3.txt` (G3, G4), `queue-f4.txt` (G5, G6, G8), `queue-f5.txt` (G7), `queue-f6.txt` (G9: round 3's knob rows); the `ktime` stage gains the four GEMV binaries; the session plan `05-session-plan.md` with the rows, PASS texts, DECIDE rows, the fallbacks (the CK build is one flag away at every row) and the budget; the rehearsal `06-rehearsal.md` from `DRY=1 bash env/session/rehearse.sh` | the queue-file tests (`harness/tests/test_queue_files.py`), the rehearsal parses every row | 3 h | every G row |
| **L9. The checklist and the gate** | `04-checklist.md`: one box per deliverable above, ticked only when its laptop check has run; the double-check of the whole preparation (the patch regenerates from the pristine fork, the full test suite, the preflight, the offline compile of every variant) | the suite green, the preflight PASS, the offline resources within the bounds of L1c | 2 h | the go |

About 33 hours of laptop work; L1 to L2 (13 h) are enough for a first
session that answers M1 to M4, and L3 to L4 (8 h) add the two gang
linears whose gain is the certain part of the stack.

## The VM part

One session of about two hours, gains first and the attribution rows
after, as round 3. Each row: the command shape (`L` is `laptop.sh`, `V`
is `vm.sh` on the VM through `L ssh`), the PASS text, the rule or decision.

| Row | Needs | Command shape | PASS text | RULE or DECIDE |
|---|---|---|---|---|
| **G0. Setup and the suites** | L1 to L5, L9 | `L push`, `L start setup` (fresh host: download, hw, setup, about 12 min), `V preflight`, `V checks`, `V reference`, `V kernels` (the GEMV rows among the 100-trial suites), `V ktime` (the four GEMV binaries, cold) | every suite 100 of 100; `ktime` prints the tile's cold time at 4, 8 and 16 rows per batch and with the strided map | DECIDE: the batch constant for the graph rows (the fastest of the three; 8 if within noise); RULE: the exec counter of a GEMV class must differ from the record's CK class before any clock is read (the JIT trap of round 3) |
| **G1. The per-tile GEMV in the graph** (M1, M2, M3) | L1, L2, L8 | `queue-f2`: `--layers 2 --iters 1 --gemv-linears ... compare`, then `--iters 32 ... --worker-timing table` with the round-3 flags kept (`--tile-linears --nt-weights --fuse-norm2 --fuse-silu --fuse-norm1 --mfma-attend --attend-tasks`) | the step-0 compare 0 FAIL, the ids equal; the table's qkva and o_proj gaps and the GEMV exec counters | DECIDE: the gaps against 14.8 us. The ladder's 64- and 96-task points (12 and 18 us with empty tasks) overstate the floor, since 14.8 is already below 18, so the reading is by movement: a gap that fell by 3 us or more is the kernel's (proceed to G3, G2 measures the rest); a gap within 1 us of 14.8 is MAJ-8's (G2 confirms, the runtime enters the round) |
| **G2. The grid sweep** (M4) | L2 | `--layers 2 --iters 32 --gemv-linears --linear-grid 48`, then `32`, the table row each | the gap of qkva at 96, 48 and 32 tasks | DECIDE: a gap that tracks the task count is completion-bound (MAJ-8 enters the round, its plan in a new ideas page); a gap that tracks the bytes is the kernel's (S2 picks the grid that measured best) |
| **G3. The w2 form** | L3 | `--layers 2 --iters 1 --gemv-linears --gemv-w2 compare`, then the `it32 wt table` | 0 FAIL, ids equal; the w2 gap and exec against 23.9 | DECIDE: on if below 16 us; the row is a flag away from the CK form otherwise |
| **G4. The w13 form, one round** | L4 | the same pair with `--gemv-w13` | 0 FAIL, ids equal; the w13 gap against 42.5 | DECIDE: on if below 33 us (the second round gone); the first round's number against G5's ceiling |
| **G5. The stream probe** (M7) | L6 | `--graph stream --kb 305 --gang`, `--kb 152` regular 96 tasks, `--kb 256` regular 296 | GB/s per XCD (the gang rows) and per device (the regular rows) | the ceiling for S1 and every gang linear; written into `08-results.md` as the machine's number |
| **G6. The head** (S3) | L5 | `--layers 2 --head --iters 1 --gemv-linears compare` (the logits boundary), the table row, then `--linear-grid 320` for the head | 0 FAIL; the head's chunk events against 141 us | DECIDE: the head's grid |
| **G7. The finals** (M6) | G1 to G4 on | `queue-f5`: `--layers 27 --head` at 30, 31, 32 iterations with every lever that passed, `--event-timing`; one row at 29 without it for `FWD_PASS` | ids 32 of 32 equal in every row; the event-clock median and the `FWD_PASS` number | the round's number against 4,500; RULE: three distinct names, the record keeps all |
| **G8. The bit-diff** (I5, if time) | L7 | the CK build's `L2_it1` row rerun with its tensors kept, then `bitdiff` against G1's run | the count of differing elements and the max ULP distance per boundary | the numerics story for the results page |
| **G9. The knob rows** (MIN-33, if time) | round 3's knob queue | the six 2-layer rows of the sleep and CAS knobs (`MPK_POLL_SLEEP` 8, 32, 127; `MPK_NO_BCAST_CAS`, `MPK_NO_LOCAL_CAS`), run alone this time | the step-0 compare and the iteration time per knob | the gang broadcast's cost, which is part of what S1's number is measured against |

After G7 the session continues into the router and merge rows if their
ideas page and laptop work are ready by then; otherwise the VM is deleted
at the user's word, as in round 3.

## The budget

| | |
|---|---|
| Balance after round 3 | $13.01 at $2.99 per hour: 4.3 hours |
| The session above | G0 about 20 minutes (a fresh host), G1 to G6 about 25 graph rows at 1.5 to 2.5 minutes each (the JIT rebuilds per flag set), G7 15 minutes: about 100 minutes, $5 |
| Slack | an hour for reruns and the router and merge rows |

## The dependency graph

    L1 (kernel) -> L1b (suite) -> L1c (offline) -> L2 (type, flag) -> G0 -> G1 -> G2
                                                 -> L5 (head)      -> G6
    L3 (w2)  ----------------------------------------------------------> G3
    L4 (w13) ----------------------------------------------------------> G4
    L6 (stream probe) ------------------------------------------------> G5
    L7 (bit-diff) ---------------------------------------------------> G8
    L8 (tooling) needs L2 to L7's flags; L9 (the gate) needs all       -> G0
    G1 to G4 on -> G7 (the finals)

L3, L4, L6 and L7 are independent of each other and of L2 once L1's
template exists; the order above is by the size of the gain they feed.

## What this split does not cover

- The router and the merge: their ideas page (`03`) follows this one and
  joins the session plan.
- MAJ-8, the runtime's per-XCD completion hierarchy: G2 decides whether
  it enters the round.
- S4, layer 0's dense MLP: after the MoE layers' linears are in.
