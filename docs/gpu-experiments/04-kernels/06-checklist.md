# 06 - Local preparation checklist

The progress record for `05-local-preparation.md`. One row per
deliverable; a box is ticked only when its check has run on the laptop
(the test suite, the syntax check, the dry run, the offline compile, or
the named script) and the result is written in the Status line with the
date and, once committed, the commit. The order is the order of work
(`05`, Part 3). Rows marked "VM" are ticked on the machine.

The commands that close every item:

```
.venv/bin/python -m pytest harness/tests fleet/tests env/hw/tests -q     # 184 pass today
bash fleet/tasks/check_syntax.sh                                          # 9 PASS today
bash env/offline_gfx942/run.sh                                            # every variant, 15 min
```

## L1. The GEMV tile kernel (5 h)

- [x] `fleet/tasks/mi300/linear_gemv_mi300.cuh`: `linear_gemv_mi300_task_impl<T, K, NORM, RESIDUAL>(x, w_norm, W, residual, out, rows, o_stride, eps)`; the wave's rows (`ceil(rows / 4)`, the tail masked), batches of `GEMV_BATCH` = 8 under `#pragma unroll 1`, the K9 map (`8 l + 512 i`) for the rows and the x slice, the loads through `StreamSrc`
- [x] the first batch's loads and the residual's values (one 2-byte load per lane: the 38-row task's columns are not 16-byte aligned) issued before the prologue; NORM through `rmsnorm_row` with `out = nullptr` and the LDS row; the null guard added to the helper
- [x] the halving butterfly (`butterfly_sum` in `mla_common_mi300.cuh`), lane l < 8 holding row `r0 + l`; the epilogue (the residual add in FP32, `bf16r`, `nt_store`, the column tail)
- [x] 256 rows (the head) as eight batches per wave with the norm once
- [x] `check_syntax.sh` with the three instantiations

Status: done 2026-09-18, commit 9d52661 (cherry-picked as the L1 commit on `local/round-4`; wave 1, agent A). The wave split is the formula's (10, 10, 10, 8 for 38 rows; the page's "10, 10, 9, 9" was a slip, corrected). The epilogue stores through `st` (2 bytes per lane), not the CK header's `nt_store`, which would put the file outside the host stub; the non-temporal hint on the output is open for the offline pass. `o_stride` is unused at one token and kept for the registration's parity. Rows past the wave's range skip their loads and FMAs through wave-uniform branches. Live registers at depth 8: 128 raw words across the prologue, about 168 in the multiply (the x slice held for the task). Checks: 10 PASS, the offline line and wait sequence follow with wave 1's compile.

## L1b. The suite rows (3 h)

- [x] `k_linear_gemv`, `k_linear_gemv_norm`, `k_linear_gemv_res` in `kernel_tests_mi300.cu` with `rows` and `o_stride` as arguments; `KT_TIME` and `KT_COLD` (the weight's rotation) over 96 blocks of 38 rows
- [x] `kernel_tests.py`: `linear_gemv`, `linear_gemv_norm`, `linear_gemv_res` (tensors, make, ref, check); `RUNS`; the `--dry-run` passes
- [x] `numpy_ref.py`: the two one-line references; `test_numpy_ref` rows
- [ ] `vm.sh`: the four binaries (`_gemv4`, `_gemv8`, `_gemv16`, `_gemv8s`) in the `kernels` stage; the `ktime` stage's `linear_gemv` grid
- [x] `fleet/tasks/README.md`: the kernel's row

Status: rows done 2026-09-18, commit 122220d (cherry-picked; wave 1, agent A). Three wrappers with `rows` and `o_stride` as arguments; the weight's `KT_COLD` rotation modelled on the attention's (the merge has none); the numpy rows in `harness/tests/test_numpy_ref.py` (the page's path corrected). The dry run at `--n 2` (a trial writes the 15 MB weight; 100 trials would be 4.5 GB of files): the three rows PASS. 185 tests. The `vm.sh` binaries and the `ktime` grid are the L8 pass.

## L1c. The union, offline (1 h)

- [ ] `mk_tu.cu`: `MK_GEMV` with the three instantiations as variants 0 to 2 of type 195; `run.sh`: `compile gemv`, `dev_gemv.s`
- [ ] the compile exits 0; the worker line (VGPRs, AGPRs, scratch, spills) recorded here; the batch loop's `s_waitcnt vmcnt` sequence recorded here (32 loads in flight)

Status:

## L2. The task type and the plan flag (5 h, with L5's grid options)

- [ ] `new_tasks.patch`: `TASK_LINEAR_GEMV_MI300 = 195`, the name map, `register_linear_gemv_mi300_task` (params `[norm, residual, eps_bits]`; the inputs by flag; the weight on dim 0 by the grid, the output on dim 1), the dispatcher case; the patch regenerated on the pristine fork, the fork reset to zero dirty lines
- [ ] `build_graph.py`: `linear_gemv_layer`; `OUTPUT_ARGS`
- [ ] `graph_plan.py`: `--gemv-linears` (qkva, o_proj, `L0.down`, the head; no scratch tensors), `--linear-grid N`, `--head-grid N`; the dry run's counts with and without; `task_graph_check.py` on the dry run's graph
- [ ] `test_graph_plan.py`: the flag's rows and the grid overrides; `graph_counts.py`'s note; `fleet/tasks/README.md`
- [ ] tests; the offline compile

Status:

## N1. The router, one level deeper (4 h)

- [x] `moe_router_mi300.cuh`: two batches of `ROUTER_BATCH` = 8 rows per wave under `#pragma unroll 1`; the K9 map for the rows and the LDS slice; the gate weight through `StreamSrc`; the first batch before the norm; the butterfly
- [x] phase 5: the initialisations and the slot writes over the lanes
- [ ] `ROUTER_BATCH` and `ROUTER_STRIDED` as defines; the binaries in `vm.sh`; the `ktime` grid
- [x] `check_syntax.sh`; the suite's `moe_router` rows dry run (ids, log, mask exact)
- [ ] the offline build: `k_moe_router` (at most 180 VGPRs, no scratch) and the worker line recorded here; `run.sh` gains `dev_kt.s` and the router loop's wait sequence recorded here

Status: kernel done 2026-09-18, commit ef10e6e on `local/round-4` (wave 1, agent B; cherry-picked). Two file-local helpers (`router_chunk`, `router_load_batch`); the initialisations run before the GEMV (one entry per thread, ordered against the slot writes by the GEMV's barrier) instead of after the top-k, which would race on the forced ids; the slot pick is an unrolled select so `ids[]` stays in registers. The batch and map defines (`ROUTER_BATCH` 4, 8, 16; `ROUTER_STRIDED`) compile under the host stub; the binaries in `vm.sh` and the `ktime` grid are open (the third box). Emulated numerics: logits within 2.5e-6 relative of the old order, ids and weights identical. Checks: 9 PASS, 184 tests, the suite's dry run. The offline line and the wait sequence follow with wave 1's compile.

## N3. The merge, one level deeper (5 h)

- [ ] `mla_merge_uv_mi300.cuh`: `W_uv` batch 0 issued first (whole-row wave-loads, 16 per lane); the partials batch at `4 q` and `256 + 4 q` with `ROWS_IN_FLIGHT` = 9 and the lse words for q = 0; one barrier for `lse_s`, every thread's M, weights and total; the FMAs and `o_s` as today
- [ ] `W_uv` batch 1 issued when the partials registers are free; the lane's eight o values in registers; the FMAs, the butterfly over 16 rows, lanes 0 to 15 storing `attn`
- [ ] `check_syntax.sh`; the suite's `mla_merge_uv` rows dry run
- [ ] the offline build: `k_mla_merge_uv` (at most 200 VGPRs) and the worker line recorded here; the wait sequence from `dev_kt.s`

Status:

## L3. The w2 form (4 h)

- [ ] `gang_moe_w2_silu_mi300.cuh`: the activation row into LDS; the GEMV over 64 rows with the 176-chunk map (three loads per lane, two for lanes 48 to 63); the scatter epilogue; the CK multiply kept under `MPK_W2_CK_TILE`
- [ ] the suite row `gang_w2_gemv` (`-DKT_FAKE_XCD`, the `(32, 8)` launch); `numpy_ref.moe_w2`; `kernel_tests.py`
- [ ] `check_syntax.sh`; `run.sh`: the `ckgang` variant on the GEMV form, `w2ck` on the CK form; the registers and the wait sequence recorded here

Status:

## L4. The w13 form, one round per XCD (4 h)

- [ ] `gang_moe_w13_gemv_mi300.cuh`: type 196 (gang); the expert decode; the tile's rows by arithmetic (`static_assert(4 * 77 + 33 * 76 == 2816)`); 19- or 20-row waves in batches of eight; x from `h` with the K9 map; the scatter epilogue
- [ ] `new_tasks.patch`: the type in `is_gang_task_type`, in the runtime's gang list (the `tiles_per_xcd` lookup, `n_tile_start = 0`) and in the Python API's gang wrapper; `register_gang_moe_w13_gemv_mi300_task` with `tiles_per_expert` = 37 (the recorded count 9 x 37); the dispatcher case in `_execute_gang_task`; regenerated
- [ ] `build_graph._gang_moe(tiles=...)`, `gang_moe_w13_gemv_layer`; `graph_plan.py --gemv-w13` with the worker-count assert; the counts
- [ ] the range test (37 ranges cover 2,816 once); the suite row `gang_w13_gemv` (the `(37, 8)` launch); `numpy_ref.moe_w13`
- [ ] `check_syntax.sh`; `mk_tu.cu` (the gang instantiation under `MK_GEMV`); the offline registers recorded here

Status:

## L7. The bit-diff (2 h)

- [x] `harness/bitdiff.py` (differing elements, max ULP, max abs per key; a markdown table); `harness/tests/test_bitdiff.py` on two synthetic files
- [x] `queue.sh bitdiff <a> <b>` writing `record/bitdiff_<a>_<b>.md`; the dry run; `env/hw/tests/test_session_scripts.py`

Status: done 2026-09-18, commit 4072de4 on `local/round-4` (wave 1, agent D; cherry-picked). The subcommand activates the fleet venv itself (`run` and `bisect` get it from the stage wrapper); a `BITDIFF` override for the tests; a usage section in `harness/README.md`. Checks: 193 tests (nine new), the dry run prints the command.

## L6. The stream probe (3 h)

- [ ] `stream_mi300.cuh`: type 197, the regular and the gang form (the K9 load loop, the XOR into a dummy word); `new_tasks.patch` (both registrations, the gang lists); regenerated
- [ ] `graph_plan.py --graph stream --tasks N --kb K [--gang]`; `build_graph.py`; `run_fleet.py`'s run name; `measure.py`'s GB/s column; the tests (the counts, the arithmetic on a fixture)
- [ ] `check_syntax.sh`; `mk_tu.cu`; the offline variant

Status:

## N4. The merge as regular tasks (3 h)

- [ ] `mla_merge_uv_mi300.cuh`: `HALVES`, the regular form's head and half from `expert_offset`; `new_tasks.patch`: type 201 in the `expert_offset` list, `register_mla_merge_uv_tile_mi300_task` (whole imaps, params `[split, n_splits, halves]`), the dispatcher case; regenerated
- [ ] `build_graph.py`; `graph_plan.py --merge-tasks [--merge-halves 2]`; the counts; `task_graph_check.py`; a test on the plan
- [ ] `check_syntax.sh`; `mk_tu.cu`; the offline variant

Status:

## N2. The router in four tasks (4 h)

- [ ] `moe_router_mi300.cuh`: `SPLIT` and `part`; the 16-expert batch before the norm; the logits to the tensor; the release fence, thread 0's acq-rel atomic at agent scope, the last task's acquire fence by every thread, top-k, writes and reset
- [ ] `new_tasks.patch`: type 200 in the `expert_offset` list, `register_moe_router_norm4_mi300_task` (the counter input, grid 4), the dispatcher case; regenerated
- [ ] `build_graph.py`; `graph_plan.py --router-tasks` (`router_counter`); the counts
- [ ] the suite row `k_moe_router4` (four blocks, a zeroed counter, bit-exact against the one-task row)
- [ ] `check_syntax.sh`; `mk_tu.cu`; the offline variant

Status:

## N5. The merge with o_proj folded in (8 h)

- [ ] `mla_merge_oproj_mi300.cuh`: type 202; N3's and N4's merge; `attn` stored; the `W_o` slice (eight lanes per row, eight rows per wave-load, two batches of 32 under `#pragma unroll 1`), the three-step row reduction, the partial vector to `workspace[t]`; the release, the counter; the last task's acquire, the fixed-order sum with the residual into `x_res`, the reset
- [ ] `new_tasks.patch`: the type in the `expert_offset` list, the registration (inputs `partials, W_uv, W_o, x_res, counter`; outputs `x_res, attn, workspace`; grid 32), the dispatcher case; regenerated
- [ ] `build_graph.py`; `graph_plan.py --merge-oproj` (`oproj_ws`, `oproj_counter`; the merge and o_proj replaced by one call labelled `L{l}.o_proj`); the counts; `compare.py`'s boundary list checked (the `x_res` and `attn` rows keep their keys)
- [ ] the suite row `mla_merge_oproj` (32 blocks, a zeroed counter; `x_res` within the linear tolerance, `attn` bit-exact)
- [ ] `check_syntax.sh`; `mk_tu.cu`; the offline variant (at most 200 VGPRs for the kernel; the worker line recorded here)

Status:

## L8 and N6. The session tooling (4 h)

- [ ] `queue-f2.txt` (G1, G2), `queue-f3.txt` (G3, G4), `queue-f4.txt` (G5, G6, G8), `queue-f5.txt` (G7), `queue-f6.txt` (G9), `queue-g1.txt` (H1 to H4), `queue-g2.txt` (H5); `test_queue_files.py` on every file
- [ ] `vm.sh`: the `kernels` and `ktime` stages extended (L1b, N1)
- [ ] `07-session-plan.md`: the rows, the PASS texts, the DECIDE rows, the fallbacks, the budget
- [ ] `08-rehearsal.md` regenerated (`DRY=1 bash env/session/rehearse.sh`); every row and stage parses

Status:

## L9. The gate (2 h)

- [ ] the three patches apply on the pristine fork and regenerate cleanly (zero dirty lines after the reset)
- [ ] the full test suite; `check_syntax.sh`; `env/preflight.sh` (8 PASS)
- [ ] the offline compile of every variant exits 0; the worker within L1c's bounds; the standalone lines of every changed kernel recorded above
- [ ] every box above ticked or explicitly deferred with its reason; the memory note updated

Status:

## Before the VM

- [ ] the user's go for the VM (a fresh host: about 12 minutes of setup; the balance $13.01)
- [ ] `FULL=1 L push` (the pristine fork, the changed patch), `L start setup`, `V preflight`, `V checks`, `V reference`, `V kernels`, `V ktime` (G0, H0)
- [ ] the first report at G0's DECIDE row (the batch constant), before any graph row
