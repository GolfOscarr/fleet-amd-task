# 06 - Local preparation checklist

The progress record for `05-local-preparation.md`. One row per
deliverable; a box is ticked only when its check has run on the laptop
(the test suite, the syntax check, the dry run, the offline compile, or
the named script) and the result is written in the Status line with the
date and, once committed, the commit. The order is the order of work
(`05`, Part 3). Rows marked "VM" are ticked on the machine.

The commands that close every item:

```
.venv/bin/python -m pytest harness/tests fleet/tests env/hw/tests -q     # 234 pass (2026-09-18)
bash fleet/tasks/check_syntax.sh                                          # 15 PASS (2026-09-18)
bash env/offline_gfx942/run.sh                                            # every variant, 15 min
```

## L1. The GEMV tile kernel (5 h)

- [x] `fleet/tasks/mi300/linear_gemv_mi300.cuh`: `linear_gemv_mi300_task_impl<T, K, NORM, RESIDUAL>(x, w_norm, W, residual, out, rows, o_stride, eps)`; the wave's rows (`ceil(rows / 4)`, the tail masked), batches of `GEMV_BATCH` = 8 under `#pragma unroll 1`, the K9 map (`8 l + 512 i`) for the rows and the x slice, the loads through `StreamSrc`
- [x] the first batch after the prologue (the pre-load form measured at 60 to 75 registers more and removed); the residual's values read once per wave (one 2-byte load per lane: the 38-row task's columns are not 16-byte aligned) and shuffled to the batch's lanes at the store; NORM through `rmsnorm_row` with `out = nullptr` and the LDS row; the null guard added to the helper
- [x] the halving butterfly (`butterfly_sum` in `mla_common_mi300.cuh`), lane l < 8 holding row `r0 + l`; the epilogue (the residual add in FP32, `bf16r`, `nt_store`, the column tail)
- [x] 256 rows (the head) as eight batches per wave with the norm once
- [x] `check_syntax.sh` with the three instantiations

Status: done 2026-09-18, commit 1038c01 on `local/round-4` (the cherry-pick of the agent's 9d52661; wave 1, agent A). The wave split is the formula's (10, 10, 10, 8 for 38 rows; the page's "10, 10, 9, 9" was a slip, corrected). The epilogue stores through `st` (2 bytes per lane), not the CK header's `nt_store`, which would put the file outside the host stub; the non-temporal hint on the output is open for the offline pass. `o_stride` is unused at one token and kept for the registration's parity. Rows past the wave's range are loaded as the range's last row (a clamped index, no branch; the guarded form was rejected in L1c) and their sums dropped by the store's mask; the residual is read once per wave before the prologue (lane l at the clamped row `r_begin + l`) and shuffled to the storing lane at each batch: the final check of 2026-09-18 found every in-loop form sunk by the compiler into the store's branch, where it waited `vmcnt(0)` after the butterfly once per batch (the guarded form also cost the residual form 82 callee-saved scratch stores per task, 256 VGPRs against 248). Live registers at depth 8: 128 raw words across the prologue, about 168 in the multiply (the x slice held for the task). Checks: 10 PASS, the offline line and wait sequence follow with wave 1's compile.

## L1b. The suite rows (3 h)

- [x] `k_linear_gemv`, `k_linear_gemv_norm`, `k_linear_gemv_res` in `kernel_tests_mi300.cu` with `rows` and `o_stride` as arguments; `KT_TIME` and `KT_COLD` (the weight's rotation) over 96 blocks of 38 rows
- [x] `kernel_tests.py`: `linear_gemv`, `linear_gemv_norm`, `linear_gemv_res` (tensors, make, ref, check); `RUNS`; the `--dry-run` passes
- [x] `numpy_ref.py`: the two one-line references; `test_numpy_ref` rows
- [x] `vm.sh`: the depth and map builds (`nt_b4`, `nt_b16`, `nt_strided` beside `nt`, built by the `ktime` stage) and the `ktime` rows of the two GEMV forms
- [x] `fleet/tasks/README.md`: the kernel's row

Status: rows done 2026-09-18, commit a5d4cb7 on `local/round-4` (the cherry-pick of the agent's 122220d; wave 1, agent A). Three wrappers with `rows` and `o_stride` as arguments; the weight's `KT_COLD` rotation modelled on the attention's (the merge has none); the numpy rows in `harness/tests/test_numpy_ref.py` (the page's path corrected). The dry run at `--n 2` (a trial writes the 15 MB weight; 100 trials would be 4.5 GB of files): the three rows PASS. 185 tests. L8 (2026-09-18): the `ktime` stage times `linear_gemv_norm` and `linear_gemv_res` with the weight rotated over the cache copies, on the plain build and on the variants `nt`, `nt_b4`, `nt_b16` and `nt_strided` (the batch and the map, under the graph rows' load policy); the `kernels` stage builds `kernel_tests_xcd` besides.

## L1c. The union, offline (1 h)

- [x] `mk_tu.cu`: `MK_GEMV` with the three instantiations as variants 0 to 2 of type 195; `run.sh`: `compile gemv`, `dev_gemv.s`, and `w2ck` and `dev_kt.s` (the launcher's device code) besides
- [x] the compile exits 0; the worker line (VGPRs, AGPRs, scratch, spills) recorded here; the batch loop's `s_waitcnt vmcnt` sequence recorded here (32 loads in flight)

Status: done 2026-09-18 (commits 5bbf16d, e75806d, the resources refreshed). The worker union (`resources.txt`): `ours` 256 VGPRs, 100 AGPRs, 8 spilled, scratch 64 bytes per lane (64 AGPRs before wave 1); `gemv` 256, 102 AGPRs, scratch 332 (the call's 67 callee-saved stores); `ckgang` (the w2 GEMV form) 256, 111, scratch 92 (its 22 saves); `w2ck` (the CK fallback) 256, 100; `mfma` 256, 126; `cklinear` 256, 104.

Final check (2026-09-18, the lead's own offline pass): the call-form kernels received their arguments in VGPRs, so every buffer load under `MLA_NT_STREAMS` ran inside a `v_readfirstlane` waterfall loop (3,277 readfirstlanes and 1,133 exec-mask loops in `dev_gemvnt.s`, 3,613 and 1,197 in the union; the plain build 641 and 516). With the arguments made wave-uniform on entry (`uniform_ptr`, `uniform_int`; `StreamSrc` applies it to its base) the counts are 743 and 488 (the union 764 and 425), the same as the plain build's. The union with every round-4 task: `gemv` 256 VGPRs, 129 AGPRs, scratch 80; `gemvnt` 256, 144, 80; `union` (GEMV, the w2 form, MFMA, streams) 256, 171, 80 (181 and 256 before the fix); `ckgang` 256, 113, 76. The kernels' own lines in the streaming build: the norm and residual GEMV forms 248 VGPRs, 32 AGPRs, 16 and 8 bytes of scratch (212 and 228 before; the residual read once per wave and shuffled), w13 248 and none, the w2 form 248 and 76, the o_proj fold 228 and 80. The standalone launcher under `MLA_NT_STREAMS`: `k_linear_gemv` 248 and no scratch, `_norm` 16 bytes, `_res` none; `k_gang_w13_gemv` 248, none; `k_gang_w2_gemv` 248, 76; `k_moe_router` 190, none; `k_mla_merge_oproj` 248, 80. One thing the pass could not settle offline: the batch loop's first wait under `MLA_NT_STREAMS` is `vmcnt(0)` after 8 to 12 of the 32 loads (the rest then stream with counted waits, 11 to 18 in flight); the cause is neither the descriptor nor the per-batch store (a form with one store per 64-row window kept the wait and added 220 bytes of scratch, and was dropped), so the VM's A/B of `--nt-streams` and the batch constant (G0, G1) decides. Three forms were rejected on the way: the per-row guards (four loads in flight of 32), the first batch before the prologue (the union at 208 AGPRs; standalone 250, 238 and 244 against 172, 170 and 166) and the inlined body (222 AGPRs against 102). The batch loop's wait sequence at eight rows: `vmcnt(27) (26) (25) (24) (19) ... (0)` under `MLA_NT_STREAMS` (32 buffer loads per batch, consumed row by row); the plain build's call form makes 36 flat loads and waits `vmcnt(0)` per batch, so the graph rows run with `--nt-streams`. Standalone (`kernel_tests.log`, -O2, the call form): `k_linear_gemv` 248 VGPRs, 12 bytes of scratch.

## L2. The task type and the plan flag (5 h, with L5's grid options)

- [x] `new_tasks.patch`: `TASK_LINEAR_GEMV_MI300 = 195`, the name map, `register_linear_gemv_mi300_task` (params `[norm, residual, eps_bits]`; the inputs by flag; the weight on dim 0 by the grid, the output on dim 1), the dispatcher case; the patch regenerated on the pristine fork, the fork reset to zero dirty lines
- [x] `build_graph.py`: `linear_gemv_layer`; `OUTPUT_ARGS`
- [x] `graph_plan.py`: `--gemv-linears` (qkva, o_proj, the head; no scratch tensors; `L0.down` stays the stock per-tile linear, its K past the kernel's bound), `--linear-grid N`, `--head-grid N`; the dry run's counts with and without; `task_graph_check.py` deferred to the VM (it reads the compiled graph, which the laptop cannot produce)
- [x] `test_graph_plan.py`: the flag's rows and the grid overrides; `graph_counts.py`'s note; `fleet/tasks/README.md`
- [x] tests; the offline compile

Status: done 2026-09-18, commits efe462d and b20353a (wave 2, agent F; cherry-picked): the registration as the hunk file `fleet/patches/hunks/L2-linear-gemv.md`, folded into `new_tasks.patch` by the integration (fcaad6e). The residual is partitioned on dim 1 like the output (the kernel indexes it by the task-local row); the name branch lives in `graph.cc`; `--linear-grid` is per operator (48 divides qkva's rows, not o_proj's); `--gemv-linears` implies the fused-norm form and is off under `--debug`; layer 0's dense down (K 11,264) stays the stock linear, the kernel bounds K at 4,096. Counts: 298 operators and 6,593 tasks, the same as `--fuse-norm1 --tile-linears`; 5,297 at `--linear-grid 48`, 6,513 at `--head-grid 320`. Checks: 201 tests, the dry runs, `task_graph_check.py` needs the VM's compiled graph.

## N1. The router, one level deeper (4 h)

- [x] `moe_router_mi300.cuh`: two batches of `ROUTER_BATCH` = 8 rows per wave under `#pragma unroll 1`; the K9 map for the rows and the LDS slice; the gate weight through `StreamSrc`; the first batch before the norm; the butterfly
- [x] phase 5: the initialisations and the slot writes over the lanes
- [x] `ROUTER_BATCH` and `ROUTER_STRIDED` as defines; the `ktime` variants `nt_b4`, `nt_b16`, `nt_strided` carry them with the GEMV's; the `moe_router` row of the `ktime` stage (L8, 2026-09-18)
- [x] `check_syntax.sh`; the suite's `moe_router` rows dry run (ids, log, mask exact)
- [x] the offline build: `k_moe_router` (at most 180 VGPRs, no scratch) and the worker line recorded here; `run.sh` gains `dev_kt.s` and the router loop's wait sequence recorded here

Offline (2026-09-18): `k_moe_router` 170 VGPRs, no scratch (238 with the first batch before the norm, a form since removed; 162 at four rows); the batch's waits `vmcnt(12) (10) (9) (8) (7)`, then `(15) (11) (10) ...` (32 loads per batch, consumed as they land); the union above.

Status: kernel done 2026-09-18, commit ef10e6e on `local/round-4` (wave 1, agent B; cherry-picked). Two file-local helpers (`router_chunk`, `router_load_batch`); the initialisations run before the GEMV (one entry per thread, ordered against the slot writes by the GEMV's barrier) instead of after the top-k, which would race on the forced ids; the slot pick is an unrolled select so `ids[]` stays in registers. The batch and map defines (`ROUTER_BATCH` 4, 8, 16; `ROUTER_STRIDED`) compile under the host stub; the binaries in `vm.sh` and the `ktime` grid are open (the third box). Emulated numerics: logits within 2.5e-6 relative of the old order, ids and weights identical. Checks: 9 PASS, 184 tests, the suite's dry run. The offline line and the wait sequence follow with wave 1's compile.

## N3. The merge, one level deeper (5 h)

- [x] `mla_merge_uv_mi300.cuh`: `W_uv` batch 0 issued first (whole-row wave-loads, 16 per lane); the partials batch at `4 q` and `256 + 4 q` with `ROWS_IN_FLIGHT` = 9 and the lse words for q = 0; one barrier for `lse_s`, every thread's M, weights and total; the FMAs and `o_s` as today
- [x] `W_uv` batch 1 issued when the partials registers are free; the lane's eight o values in registers; the FMAs, the butterfly over 16 rows, lanes 0 to 15 storing `attn`
- [x] `check_syntax.sh`; the suite's `mla_merge_uv` rows dry run
- [x] the offline build: `k_mla_merge_uv` (at most 200 VGPRs) and the worker line recorded here; the wait sequence from `dev_kt.s`

Offline (2026-09-18): `k_mla_merge_uv` 166 VGPRs, no scratch (244 with the `W_uv` batches before the partials, a form since removed; 205 at a batch of 8 with it); the partials batch waits `vmcnt(0)` after the lse words (in-order completion: the batch is in flight together), the `W_uv` batches `vmcnt(11) (10) (9) ...`; the union above.

Status: kernel done 2026-09-18, commit 60ad996 on `local/round-4` (wave 1, agent C; cherry-picked). The phases as the page orders them; the `W_uv` batches as a two-slot pipeline so `-DMERGE_W_BATCH=8` (four batches) issues and consumes them in a loop, the default 16 running the page's schedule; `ROWS_IN_FLIGHT` a literal 9 with a `static_assert` against `N_SPLITS_MAX` (33) because `pf.sh` greps the literal, `PF_W` gone (`pf.sh` prints one merge line; its test needs one). The total of the weights is summed in ascending j by every thread (deterministic; the divisor may differ in its last FP32 bit from round 3's wave sum, the suite compares `attn` against the reference within tolerance), 42 `expf` per thread. `W_uv` through `StreamSrc` (byte-identical without `MLA_NT_STREAMS`). A host emulation of the six phases with the butterfly's exact order matched `numpy_ref.mla_merge_uv` on 128 of 128 outputs for three heads at batch 8, 16 and 32 (the agent's scratch script); that is a sample outcome within the BF16 tolerance, not an identity (the reference sums in einsum order, the kernel in its own FP32 order; the double-check of 2026-09-18 found one BF16 flip at step 0, batch 16), and the suite row compares within tolerance. The identity that holds is between the two forms: the tile form equals the gang form bit for bit. LDS 10,496 bytes; about 153 raw registers in either phase. Checks: 11 PASS, 194 tests.

## L3. The w2 form (4 h)

- [x] `gang_moe_w2_silu_mi300.cuh`: the activation row into LDS; the GEMV over 64 rows with the 176-chunk map (three loads per lane, two for lanes 48 to 63); the scatter epilogue; the CK multiply kept under `MPK_W2_CK_TILE`
- [x] the suite row `gang_w2_gemv` (`-DKT_FAKE_XCD`, the `(32, 8)` launch); `numpy_ref.moe_w2`; `kernel_tests.py` (wave 2, agent G, commit 57b7b6b: eight expert slabs per trial with a mask naming ids 0 to 7, since the model's 66 would be 761 MB; the row SKIPs without the `kernel_tests_xcd` binary, which L8's `vm.sh` builds)
- [x] `check_syntax.sh` (the entry `gang_moe_w2_silu_mi300`, 11 PASS); `run.sh`: the `ckgang` variant on the GEMV form, `w2ck` on the CK form; the registers and the wait sequence recorded here

Offline (2026-09-18): both variants compile; the GEMV form's function (a `__noinline__` call) 248 VGPRs and 32 AGPRs with 92 bytes of scratch (its 22 callee-saved stores), its 24 loads per batch issued in sequence before one `vmcnt(0)` (flat loads in the plain build; the streaming build makes them buffer loads); `W2_BATCH=4` takes the union to 101 AGPRs and 64 bytes.

Status: kernel done 2026-09-18, commit 69fe6b1 on `local/round-4` (wave 1, agent E; cherry-picked, the syntax-check entry merged with L1's). The default path includes no CK header: the XCD read is a local copy named `_gang_moe_w2_xcd_id` (the stock name would be a redefinition in the runtime unit), `-DKT_FAKE_XCD` takes the XCD from `blockIdx.y` there (the suite row of L4's agent must not add it again), `fast_silu` comes from the fork's header when on the include path and a local definition under the stub, and the BF16 rounding is a local round-to-nearest-even with the NaN quieting that reproduces `__float2bfloat16` bit for bit (the suite row and the B11 and B12 boundaries are the checks). A row past a short tile is loaded as the last valid row and its store dropped (no short tile at N = 2,048). Registers at `W2_BATCH` 8: 24 x-values, 96 raw words, 8 accumulators; LDS: the 2,816-byte row. The CK fallback path is unverified here (no CK headers on the laptop): the offline `w2ck` variant is its check. Checks: 11 PASS, 194 tests.

## L4. The w13 form, one round per XCD (4 h)

- [x] `gang_moe_w13_gemv_mi300.cuh`: type 196 (gang); the expert decode; the tile's rows by arithmetic (`static_assert(4 * 77 + 33 * 76 == 2816)`); 19- or 20-row waves in batches of eight; x from `h` with the K9 map; the scatter epilogue
- [x] `new_tasks.patch`: the type in `is_gang_task_type`, in the runtime's gang list (the `tiles_per_xcd` lookup, `n_tile_start = 0`); no Python hunk (`gang_task_tiles_per_xcd` is a C++ map that `graph.cc` fills); `register_gang_moe_w13_gemv_mi300_task` with `tiles_per_expert` = 37 (the recorded count 9 x 37); the dispatcher case in `_execute_gang_task`; regenerated
- [x] `build_graph._gang_moe(tiles=...)`, `gang_moe_w13_gemv_layer`; `graph_plan.py --gemv-w13` with the worker-count assert; the counts
- [x] the range test (37 ranges cover 2,816 once); the suite row `gang_w13_gemv` (the `(37, 8)` launch); `numpy_ref.moe_w13`
- [x] `check_syntax.sh`; `mk_tu.cu` (the gang instantiation under `MK_GEMV`); the offline registers recorded here

Status: done 2026-09-18, commit 57b7b6b (wave 2, agent G; cherry-picked), the hunk folded in at fcaad6e. The wave split of a 77-row tile is the formula's 20, 20, 20, 17; `gang_task_tiles_per_xcd` is a C++ map filled by `Graph::register_task` (no Python hunk); the layer's argument is `tiles_per_expert`; `NUM_WORKERS = 296` is asserted in the plan and `num_workers` in `build`; the `(37, 8)` launch does not reach the early return for the inactive expert slots (the runtime's loop does). Offline: `k_gang_w13_gemv` 248 VGPRs, 30 AGPRs, no scratch; the union with every round-4 task 256 VGPRs and 125 AGPRs. Checks: 14 PASS, 214 tests, the dry run moves the 26 w13 operators only.

## L7. The bit-diff (2 h)

- [x] `harness/bitdiff.py` (differing elements, max ULP, max abs per key; a markdown table); `harness/tests/test_bitdiff.py` on two synthetic files
- [x] `queue.sh bitdiff <a> <b>` writing `record/bitdiff_<a>_<b>.md`; the dry run; `env/hw/tests/test_session_scripts.py`

Status: done 2026-09-18, commit 4072de4 on `local/round-4` (wave 1, agent D; cherry-picked). The subcommand activates the fleet venv itself (`run` and `bisect` get it from the stage wrapper); a `BITDIFF` override for the tests; a usage section in `harness/README.md`. Checks: 193 tests (nine new), the dry run prints the command.

## L6. The stream probe (3 h)

- [x] `stream_mi300.cuh`: type 197, the regular and the gang form (the K9 load loop, the XOR into a dummy word); `new_tasks.patch` (both registrations, the gang lists); regenerated
- [x] `graph_plan.py --graph stream --tasks N --kb K [--gang]`; `build_graph.py`; `run_fleet.py`'s run name; `measure.py`'s GB/s column; the tests (the counts, the arithmetic on a fixture)
- [x] `check_syntax.sh`; `mk_tu.cu`; the offline variant

Status: done 2026-09-18, commit a3edf62 (wave 2, agent J; cherry-picked), the hunk folded in at fcaad6e with two corrections by the integrator (the gang form is 206, not 203; it belongs on both partition lists of `runtime.cc`). `--kb` must be a multiple of 4 (whole 4 KB rows): the gang row runs at 304 KB, w13's shape to 0.3%; the rows are cut into whole batches round-robin over the waves so no partial batch re-loads rows; the chain's dummy is a real input of every stream operator; the regular form has a suite row checking the XOR exactly. `queue.sh`'s `ref_cache` guard must learn `--graph stream` (L8). Checks: 12 PASS, 205 tests, the three dry runs.

## N4. The merge as regular tasks (3 h)

- [x] `mla_merge_uv_mi300.cuh`: `HALVES`, the regular form's head and half from `expert_offset`; `new_tasks.patch`: type 205 in the `expert_offset` list, `register_mla_merge_uv_tile_mi300_task` (whole imaps, params `[split, n_splits, halves]`), the dispatcher case; regenerated
- [x] `build_graph.py`; `graph_plan.py --merge-tasks [--merge-halves 2]`; the counts; `task_graph_check.py` deferred to the VM (the compiled graph); a test on the plan
- [x] `check_syntax.sh`; `mk_tu.cu`; the offline variant

Status: done 2026-09-18, commit 0f6eec3 (wave 2, agent I; cherry-picked). The shared body is `mla_merge_uv_head<T, NH, D_V, D_C, HALVES>`, the gang entry point unchanged, the tile entry point new (type 205 after the integration); `--merge-halves` needs `--merge-tasks`; the suite's `mla_merge_uv_tile` and `mla_merge_uv_tile2` rows compare `attn` bit for bit against the gang row. Counts: 2,501 tasks at 16 per layer, 2,933 at 32. Checks: 219 tests.

## N2. The router in four tasks (4 h)

- [x] `moe_router_mi300.cuh`: `SPLIT` and `part`; the 16-expert batch before the norm; the logits to the tensor; the release fence, thread 0's acq-rel atomic at agent scope, the last task's acquire fence by every thread, top-k, writes and reset
- [x] `new_tasks.patch`: type 204 in the `expert_offset` list, `register_moe_router_norm4_mi300_task` (the counter input, grid 4), the dispatcher case; regenerated
- [x] `build_graph.py`; `graph_plan.py --router-tasks` (`router_counter`); the counts
- [x] the suite row `k_moe_router4` (four blocks, a zeroed counter, bit-exact against the one-task row)
- [x] `check_syntax.sh`; `mk_tu.cu`; the offline variant

Status: done 2026-09-18, commit 3f406e6 (wave 2, agent I; cherry-picked), type 204 after the integration. The split form keeps `butterfly_sum<ROUTER_BATCH>` over eight rows with the unused rows zero, so the bits equal the one-task form's (a batch of four would change the association); `--router-tasks` implies the fused norm; the stub header gained the fence, atomic and scope macros for the host check; the suite's `moe_router4` row compares every output bit for bit and the counter's reset. Registers about 64 below the one-task form. Counts: 300 operators, 2,337 tasks. Checks: 224 tests.

## N5. The merge with o_proj folded in (8 h)

- [x] `mla_merge_oproj_mi300.cuh`: type 207; N3's and N4's merge; `attn` stored; the `W_o` slice (eight lanes per row, eight rows per wave-load, two batches of 32 under `#pragma unroll 1`), the three-step row reduction, the partial vector to `workspace[t]`; the release, the counter; the last task's acquire, the fixed-order sum with the residual into `x_res`, the reset
- [x] `new_tasks.patch`: the type in the `expert_offset` list, the registration (inputs `partials, W_uv, W_o, x_res, counter`; outputs `x_res, attn, workspace`; grid 32), the dispatcher case; regenerated
- [x] `build_graph.py`; `graph_plan.py --merge-oproj` (`oproj_ws`, `oproj_counter`; the merge and o_proj replaced by one call labelled `L{l}.o_proj`); the counts; `compare.py`'s boundary list checked (the `x_res` and `attn` rows keep their keys)
- [x] the suite row `mla_merge_oproj` (32 blocks, a zeroed counter; `x_res` within the linear tolerance, `attn` bit-exact)
- [x] `check_syntax.sh`; `mk_tu.cu`; the offline variant (at most 200 VGPRs for the kernel; the worker line recorded here)

Status: done 2026-09-18, commit 9231083 (wave 2, agent K; cherry-picked), type 207 (f49dba2), the hunk folded in at d91a1fe (every variant, the host check and the launchers clean). Offline: `k_mla_merge_oproj` 248 VGPRs, 32 AGPRs, 112 bytes of scratch, no spills (the page's "at most 200" was an estimate; the three phases' peak is 248 under the cap); the union with every round-4 task 256 VGPRs and 126 AGPRs (151 AGPRs and 368 bytes of scratch under `MLA_NT_STREAMS`, the build the graph rows make; round 3's own streaming build was 179). The review of 2026-09-18 (a Fable agent) found the weight phase's batch of 4 wave-loads (sixteen round trips for the 256 KB slice) and it is 16 now (248 registers at 4, 8 and 16; 32 spills 144 bytes); the release fence is followed by a barrier before the atomic in both last-task kernels (the page's order); the review's notes for L8: `vm.sh` must build `kernel_tests_xcd` or the gang rows SKIP, `KT_TIME` repeats the in-place `x_res` accumulation (timing only), `--probe-before L{l}.o_proj` does not apply under the fold, and the acquire-fence knob must not be paired with the counter forms. The shared merge body gained an optional LDS copy of the attn values and a constexpr LDS size; `x_res` is one argument (input 3 and output 0 share the address); `--merge-oproj` always uses two halves and overrides `--merge-tasks`; not gated by `--debug`. Counts: 299 operators and 2,717 tasks alone, 245 and 5,565 with every wave-2 flag. An emulation of phases 4 and 5 matched the residual linear within the BF16 tolerance (a sample outcome: the double-check of 2026-09-18 found one BF16 flip in 1 of 6 trials of 2,048 elements, the reference summing in BLAS order and the kernel in its fixed task order; the suite row compares within tolerance). Checks: 15 PASS, 232 tests, the suite's `mla_merge_oproj` row.

## L8 and N6. The session tooling (4 h)

- [x] `queue-f2.txt` (G1, G2), `queue-f3.txt` (G3, G4), `queue-f4.txt` (G5, G6; G8 is the `bitdiff` subcommand, not a row), `queue-f5.txt` (G7 = H6), `queue-f6.txt` (G9), `queue-g1.txt` (H2, H4; H1 and H3 read from G4's row, the deeper kernels being the header), `queue-g2.txt` (H5); `test_queue_files.py` on every file, building the plan of every model row
- [x] `queue.sh`: a compare on `--graph empty` or `stream` refused; a stream row runs without the reference tensors; the two cases in `test_session_scripts.py`
- [x] `vm.sh`: the `kernels` stage builds `kernel_tests_xcd` and passes the variant's `_xcd` build to the gang rows; the `ktime` stage times the two GEMV forms, the router, the o_proj fold and the gang forms, and builds the variants `nt_xcd`, `nt_b4`, `nt_b16`, `nt_strided` on demand (L1b, N1)
- [x] `07-session-plan.md`: the rows S0 to the end with the PASS texts, the RULE and DECIDE rows and their thresholds (T1 to T12), the queue files, the helpers, the playbook, the fallbacks, the budget, the decisions to confirm
- [x] `08-rehearsal.md` regenerated (`bash env/session/rehearse.sh`); every row and stage parses (40 `run_fleet.py` lines, no guard or parse failure)

Status: done 2026-09-18, commits 7067ac4 (the queue files, the guards, the tests), ca90be7 (the stages) and 2043a1b (the plan, the rehearsal, the gate); the final check of the same day gave the norm and residual GEMV runners the `KT_TIME` block the plain one had (one shared helper, the `TIME` line naming the form), which the `ktime` rows of S4 need. Every graph row carries `--nt-streams`; the CK build's rows and G1's hold the w2 multiply on CK with `-DMPK_W2_CK_TILE` so they differ in the linears alone; the head's rows are timing rows (a 2-layer graph's logits cannot match the reference; the head form is judged by the finals' ids); the fence knobs are not in G9 (the counter forms). The `ktime` variants carry both the GEMV and the router defines, so G0 and H0 read one curve each from four builds. Checks: 234 tests (the round-4 queue rows build their plans), `bash -n`, the DRY runs of the two stages, the rehearsal.

## L9. The gate (2 h)

- [x] the three patches apply on the pristine fork and regenerate cleanly (zero dirty lines after the reset)
- [x] the full test suite; `check_syntax.sh`; `env/preflight.sh` (8 PASS)
- [x] the offline compile of every variant exits 0; the worker within L1c's bounds; the standalone lines of every changed kernel recorded above
- [x] every box above ticked or explicitly deferred with its reason; the memory note updated

Status: passed 2026-09-18 after L8. `preflight.sh` 8 PASS (the suite 234 passed, the syntax check 15 PASS, the dry run 326 operators and 2,285 tasks, the patches apply on a clean worktree); the fork's tracked files at zero dirty lines after the reset (`new_tasks.patch` regenerated on it this day, the timing classes); the offline compile of every variant exit 0 with the disassembly of `gemv`, `gemvnt`, `union`, `ckgang` and the launcher read (the worker union 256 VGPRs and 8 spills in every variant, as L1c bounds it; `union` 171 AGPRs and 80 bytes of scratch); the kernels' lines recorded in L1c, N1, N3, L3, L4 and N5. Deferred to the VM by design: `task_graph_check.py` on the compiled graph (S8 of `07`), the standalone timings (S4), the CK fallback path of w2 (the `w2ck` variant compiles it; the VM runs it only if G3 fails). The memory note names the four compiler conventions and the state.

## Before the VM

- [ ] the user's go for the VM (a fresh host: about 12 minutes of setup; the balance $13.01)
- [ ] `FULL=1 L push` (the pristine fork, the changed patch), `L start setup`, `V preflight`, `V checks`, `V reference`, `V kernels`, `V ktime` (G0, H0)
- [ ] the first report at G0's DECIDE row (the batch constant), before any graph row
