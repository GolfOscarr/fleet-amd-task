# 04 - Local preparation checklist

The progress record for `03-local-preparation.md`. One row per deliverable;
a box is ticked only when its check has run on the laptop (the test suite,
the dry run, the offline compile, or the named script) and the result is
written in the Status column with the date and, once committed, the commit.
The order is the order of work. Items marked "VM" are ticked on the machine.

Two commands close every item:

```
.venv/bin/python -m pytest harness/tests fleet/tests env/hw/tests -q     # 155 pass today
bash env/offline_gfx942/run.sh                                            # the gfx942 compile
```

## O0. The duplicate task-type value (0.5 h)

- [x] `TASK_MLA_ATTEND_TILE_MI300` renumbered to 190 in the enum hunk and the name map of `new_tasks.patch`; the patch regenerated against the pristine fork
- [x] tests pass; the offline compile passes; `env/preflight.sh` 8 PASS

Status: done 2026-09-17, commit 654b931 on `local/round-3`. Found on the way: `check_syntax.sh` had failed since session B (the suite's timing loops use HIP events the stub header did not declare); fixed in the same commit, preflight back to 8 PASS. The name map is in `profiler_persistent.py`, not `persistent_kernel.py`.

## O1. The router folds the post-attention norm (3 h)

- [x] `numpy_ref.moe_router` takes `x_res` and `w_norm` and applies `rmsnorm` first; `test_numpy_ref` compares the composition against the tiny model's norm and router modules
- [x] `moe_router_mi300.cuh`: the norm prologue (FP32 sum of squares over the row, the reference's rounding order, 16-byte stores of `h`), one more input (`w_norm`), `h` as the first output, `eps` as a float bit pattern
- [x] `new_tasks.patch`: `register_moe_router_mi300_task` with the new pointer count and the emitted call; `build_graph.moe_router_layer` pointer order
- [x] `graph_plan.py --fuse-norm2`: `L{l}.norm2` dropped for MoE layers, the router reads `x_res`; the dry run reports 300 operators and no chain violation; a test
- [x] the kernel suite's router case covers the norm (`kernel_tests_mi300.cu`, `kernel_tests.py --dry-run`)
- [x] `compare.py`: B7 (`h` after the norm) still resolves to its last writer
- [x] `check_syntax.sh`; tests; the offline compile (`mk_tu.cu` call updated)
- [ ] VM: router suite 100 of 100; `L2_it32` compare PASS; 27-layer ids equal

Status: laptop part done 2026-09-17 (commit below on `local/round-3`). The kernel keeps one template with a `NORM` flag; the un-fused registration is byte-for-byte the old task (its emitted call gained two `nullptr`s and `0.0f`), the fused one is `moe_router_norm_mi300` (3 inputs, 6 outputs, 7 params) under the same task type; `MAX_OUTPUTS_PER_TASK` 5 to 6. The plan flag `--fuse-norm2` is default off: 27 layers give 300 operators and 1,854 tasks with it, 326 and 1,880 without. `L{l}.norm2` is not a boundary (B7 is the norm's input), so the compare is unchanged; the dump keeps the `L{l}.norm2` key from the router as last writer. Checks: 158 tests; the suite's dry run (the `h` row exact); check_syntax 7 PASS (both `NORM` instantiations); the offline compile of all five variants exit 0: the worker union 234 to 237 VGPRs, no VGPR spills, static LDS +256 B (the sixth output pointer over the 16-entry descriptor buffer); preflight 8 PASS.

## O2. The silu-mul into the expert down projection's prologue (4 h)

- [x] `gang_moe_w2_silu_mi300.cuh`: the copy of the w2 kernel with the prologue (silu-mul of the slot's row from `mid` into the tile's scratch row, BF16 rounding as the stock silu, `fast_silu`), the workgroup-scope release, `__syncthreads()`, the workgroup-scope acquire, then the CK pipeline on the scratch row
- [x] `new_tasks.patch`: the task type, `register_gang_moe_w2_silu_task` taking K from the weight and the input row stride from `mid`, the gang-type lists in `runtime.cc` and `persistent_kernel.cuh`, `gang_moe_w2_silu_linear_layer` in the Python API
- [x] `build_graph.py`: the wrapper and the scratch workspace `[256, 1408]` BF16 with `ROW_SLACK`; `graph_plan.py --fuse-silu`: `L{l}.silu` dropped, `w2` reads `mid`; under `--debug` the stock silu stays; the dry run reports 274 operators; a test
- [x] the offline compile: the disassembly shows `buffer_inv sc0` between the scratch stores and the first buffer load of the pipeline (if the compiler emits nothing for the workgroup-scope acquire, the CK view takes the L1-bypassing coherence value instead; record which)
- [x] `check_syntax.sh`; tests; `mk_tu.cu` updated
- [ ] VM: `L2_it32` compare PASS with B12 (`out8`) within threshold; the 27-layer ids; the event clock

Status: laptop part done 2026-09-17 (commit below on `local/round-3`). `fleet/tasks/mi300/gang_moe_w2_silu_mi300.cuh` is the stock w2 kernel with the prologue; task type 191 (gang) in every runtime list the stock w2 is in; registration `gang_moe_w2_silu_linear_mi300` with K from the weight and the input stride asserted to be 2K; `--fuse-silu` default off: with `--fuse-norm2` 274 operators and 1,646 tasks at 27 layers. The visibility sequence in the disassembly of the fused kernel: `flat_store_dwordx4` (the scratch row), `s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)`, `s_barrier`, then the A loads as `buffer_load_dwordx4 ... sc0` (the workgroup-scope release alone had lowered to `lgkmcnt(0)` plus the barrier, so an explicit `s_waitcnt 0` was added). The `ckgang` variant of `env/offline_gfx942/run.sh` instantiates the CK small-tile pipeline for gfx942 offline (exit 0; the fused kernel 114 VGPRs, the worker union 235 VGPRs and 4 AGPRs, no VGPR spills), which also answers O3's open question. Checks: 160 tests (two new), check_syntax 7 PASS (the CK header is outside its reach), preflight 8 PASS, the three patches apply, the offline compile of all six variants exit 0.

## O5. The probe before an operator (1 h)

- [x] `graph_plan.py --probe-before <label>`: a one-task `copy` of the producer's output into a twin tensor, the named operator reading the twin; the chain check passes; a test
- [x] `run_fleet.py` accepts the flag and suffixes the run name (`_probe_<label>`)
- [ ] VM: `L2_it32 --probe-before L0.o_proj --event-timing`: the `o_proj` gap with a one-task predecessor

Status: laptop part done 2026-09-17 (commit below on `local/round-3`). `Plan.insert_probe(label)` is a post-pass on the built plan: it takes the [1, N] BF16 tensor the operator shares with its predecessor, declares the twin `<name>_probe`, inserts `copy_layer` as `<prefix>.probe_<op>` and points the operator's input at the twin; applied before `--stop-after`; it refuses a tensor the copy task cannot take (e.g. `out8` before `combine`). `run_fleet.py --probe-before LABEL`, suffix `_probe_<label>`, in the run meta. Checks: 161 tests (one new), preflight 8 PASS; no kernel or patch change.

## O6. Non-temporal loads for our cache streams (2 h)

- [x] `mla_common_mi300.cuh`: `load8_nt` (the raw buffer load with coherence 18, or the inline `sc1 nt` form; whichever the disassembly honours)
- [x] `mla_attend_mi300.cuh`: the p x V pass loads under `-DMLA_NT_STREAMS`; `mla_merge_uv_mi300.cuh`: the partials loads
- [x] `run_fleet.py --nt-streams` passes the define through `MPK_EXTRA_HIPCC_FLAGS`; the suite builds with and without
- [x] the offline compile: the `nt` bit on the loads in the disassembly, no other change in the code
- [ ] VM: suites unchanged; `L2_it32` event clock with and without

Status: laptop part done 2026-09-17 (commit below on `local/round-3`). `StreamSrc`, `load8_from` and `ldf_from` in `mla_common_mi300.cuh`: raw buffer loads with CK's policy value 18 (the assembler prints it `nt sc1`) through `__builtin_amdgcn_raw_buffer_load_b128/_b32` and `__builtin_amdgcn_make_buffer_rsrc`, the idiom CK itself uses; plain loads without the define or on the host. Applied to the attention's p x V pass (the row's second and last read; the scores pass keeps the default so that read hits L2) and to the merge's three partials reads. `run_fleet.py --nt-streams` sets `-DMLA_NT_STREAMS` through `MPK_EXTRA_HIPCC_FLAGS` (suffix `_nts`, the flags in the run meta); `vm.sh` builds a third suite binary `kernel_tests_nt` for `KT_TIME` against the plain one; `run.sh` gained the `ntstreams` variant and the `_nt` launcher. Disassembly of the worker: without the define 0 loads carry `nt sc1` (the runtime's own 4 `global_load_dwordx2 nt` are in both builds), with it 10 `buffer_load_dwordx4` (the attention) and 12 `buffer_load_dword` (the merge) carry `nt sc1`; the worker union 230 VGPRs with the define. Checks: 161 tests (the run-name test extended), check_syntax 7 PASS, preflight 8 PASS, the offline compile of all eight variants exit 0.

## O3. The input norm as a prologue of the per-tile linear (5 h)

- [x] `mla_common_mi300.cuh`: `rmsnorm_row` shared with O1
- [x] `linear_norm_mi300.cuh`: the prologue into the task's scratch row, the fences of O2, then the batch-1 tier of `linear_kernel_ck` copied with the `sc0` A view (not a call: the view's coherence bits are inside that function), no residual
- [x] `new_tasks.patch`: `register_linear_norm_mi300_task` (from `register_linear_task`: inputs `x`, `w_norm`, `W`; outputs `out`, the scratch partitioned on dim 0), the emitted call, task type 192, `linear_norm_layer` in `build_graph.py` (where the other new layer methods live)
- [x] `build_graph.py`: the wrapper; two scratch workspaces `[96, 2048]` and `[400, 2048]`; `graph_plan.py --fuse-norm1`: `L{l}.norm1` and `head.norm` dropped, `qkva` reads `x_res`, `lm_head` reads the head's input; under `--debug` the stock norms stay; the dry run reports 246 operators; two tests
- [x] `test_numpy_ref`: `rmsnorm` then the linear against the tiny model's modules (`numpy_ref.linear_norm`, against `input_layernorm` and `q_proj` of layers 0 and 1)
- [x] the offline compile: `mk_tu.cu` instantiates the fused linear (the CK small-tile linear at K = 2048) for gfx942 under `MK_CK_LINEAR` (the `cklinear` variant of `run.sh`, exit 0); the disassembly shows the O2 sequence and the `sc0` A loads
- [ ] VM: `L2_it32` compare PASS (B2 `qkva`, the logits); the ids; the event clock

Status: laptop part done 2026-09-17 (commit below on `local/round-3`). `fleet/tasks/mi300/linear_norm_mi300.cuh`: the `rmsnorm_row` prologue into the task's scratch row, `s_waitcnt 0`, the workgroup-scope release, `__syncthreads()`, then the batch-1 tier of `linear_kernel_ck` copied with the `sc0` A view (the view's coherence bits are inside that function, so a call could not carry them; the residual path left out). Task type 192 (regular); registration `linear_norm_mi300` (inputs `x`, `w_norm`, `W`; outputs `out`, `scratch`; K from `x`, the output size and stride as the stock per-tile linear, the scratch's rows asserted equal to the grid); `linear_norm_layer` in `build_graph.py` with the stock linear's imaps and the scratch on dim 0; `--fuse-norm1` default off: with `--fuse-norm2 --fuse-silu` 246 operators (5,954 tasks with `--tile-linears`, 4,386 without: the fused linear is per-tile either way); off under `--debug`. Disassembly of the fused function: `flat_store_short` (the norm row), `s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)`, `s_barrier`, then `buffer_load_dwordx4 ... sc0` (six sites per instantiation, twelve in the worker with the two registrations of `mk_tu.cu`). Registers: the K = 2048 small tile is the stock linear's register-heavy tier; offline, with only our tasks in the unit, the worker union is 256 VGPRs and 23 AGPRs (`__forceinline__`, the stock form; 35 AGPRs as a `__noinline__` call, so the stock form is kept), no VGPR spills; the VM's day-1 megakernel with the stock linears was already 248 VGPRs and 48 AGPRs at one wave per SIMD (`01-bringup/04-session-log.md`), the regime the persistent kernel runs in anyway. `numpy_ref.linear_norm` and its test against the tiny model's `input_layernorm` and `q_proj`. Checks: 164 tests (three new), check_syntax 7 PASS (the CK header is outside its reach), preflight 8 PASS, the three patches apply, the offline compile of all nine variants exit 0.

## O7. The MFMA attention (12 h)

- [x] `fleet/tests/test_mfma_layout.py`: the 16 x 16 x 16 product assembled from the per-lane fragments of the assumed operand layout equals the plain product (five tests: one instruction, the 36-step score block split over four waves, the p x V blocks and the accumulator placement, a zero-filled tail pass, the LDS row stride)
- [x] `mla_common_mi300.cuh`: `mfma_16x16x16_bf16` (the builtin on the device, the accumulator unchanged on the host syntax check), the `bf16x4_t` and `f32x4_t` fragment types, `load16_from` (the raw 16-byte tile load, streaming under O6's define); the fragments are read in the kernel by pointer casts (`ds_read2_b64` in the disassembly)
- [x] `mla_attend_mfma_mi300.cuh`: `TILE = 16`; the cooperative tile load (five 16-byte chunks per thread, issued before use), the next pass prefetched into registers; the scores as 36 steps split nine per wave and reduced through LDS (the doc's "one wave, 36 steps" spread over the four); the softmax by the 16 lanes of a head with the reference's rounding points; p x V on four waves (8 column blocks each, the V fragment read as four strided `ds_read_u16` from the staged row-major tile: no transposed copy); `o = acc / l`, `lse`; the tile decode, the empty-split fill, the partials layout and the debug scores unchanged; selected by `-DMLA_ATTEND_MFMA` from `mla_attend_mi300.cuh`
- [x] the suite builds both variants (`kernel_tests_mfma`, `kernel_tests_mfma_debug` in `vm.sh`; `kernel_tests.py --bin`); `run_fleet.py --mfma-attend` (suffix `_mfma`, through the extra-flags hook)
- [x] the offline compile: 17 `v_mfma_f32_16x16x16_bf16` in the worker (nine score steps, eight column blocks), the accumulators in AGPRs, no VGPR spills; LDS 42,176 B by the kernel's `static_assert` (the resource print shows the static 2,512 B only); the worker union 253 VGPRs and 32 AGPRs (237 and 0 for the VALU build); the standalone suite kernel 140 VGPRs and 32 AGPRs (the VALU one 124), no spills
- [x] `check_syntax.sh` (8 PASS, the MFMA header through the wrapper's host path); tests; `mk_tu.cu` unchanged (the same emitted call; the `mfma` variant of `run.sh` sets the define)
- [ ] VM: the suites (1 split, 33 splits, scores) 100 of 100 against `kernel_tests_mfma`; `KT_TIME` standalone against 34 us; `L2_it32 --mfma-attend` compare and event clock; the 27-layer ids

Status: laptop part done 2026-09-17 (commit below on `local/round-3`). The kernel is `fleet/tasks/mi300/mla_attend_mfma_mi300.cuh`, the same template and signature as the VALU kernel, so nothing in the registration, the plan or the launcher changes; the flag is off by default. Per pass of 16 rows: the tile `[16][576]` is loaded as 16-byte chunks by all threads (rows past the split zero-filled) with the next pass in flight, written to LDS rows of stride 584 (16-byte aligned; a 1,152-byte stride would put all 16 rows on the same banks); the scores are 36 MFMA steps, nine per wave, both fragments four consecutive elements of a row (`ds_read2_b64`), reduced through LDS by thread (head, row), scaled and masked; the online softmax by the head's 16 lanes (xor shuffles of width 16) with the reference's rounding points (`l` from the unrounded probabilities, the probabilities BF16 into LDS as the next A operand); p x V eight column blocks per wave with the V fragment as four strided `ds_read_u16` of the staged tile, the accumulator (32 FP32 per lane, in AGPRs) rescaled by alpha before each step. The lane arithmetic is the layout CK encodes for the instruction (`kAMLane = kBNLane = 16`, `kABKLane = kABKPerLane = 4`, `kCMLane = 4`, `kCNLane = 16`, `kCM1PerLane = 4`) and is tested by emulation in `test_mfma_layout.py`; the instruction itself runs first on the VM. Checks: 169 tests (five new, the run-name test extended), check_syntax 8 PASS, preflight 8 PASS, the offline compile of all eleven variants exit 0.

## O4. Per-tile elementwise operators (2 h, only without O2)

- [ ] `graph_plan.py --tile-moe`: `combine` at 64 tasks through the stock registration; the silu through `moe_silu_mul_tile_mi300` (our variant with the full-row stride)
- [ ] the dry run; the wrappers' assertions; a test
- [ ] VM: `L2_it32` event clock

Status:

## O8. The prefetch operator (8 h, only if time remains)

- [ ] `new_tasks.patch`, `runtime.cc`: the side-operator branch (dependent events from the following operator, not made `pre_op`, triggers the end-of-graph event with `num_triggers` raised); `prefetch_layer` in the Python API
- [ ] `prefetch_mi300.cuh` and its registration; `graph_plan.py --prefetch` (the expert prefetch on the router's event, the linears' on the previous operator's); `build_graph.py`
- [ ] the plan's chain check skips side operators; a host compile of the patched `runtime.cc` (`clang++ -fsyntax-only`); the offline compile of the task; a test that reads a VM dry run's `task_graph_0.json`
- [ ] VM: `task_graph_0.json` shows the side tasks' events; `L2_it32` compare; the event clock without and with E2 on the consuming linears

Status:

## Instruments (Part 2 of `03`; detailed on the next pass)

- [ ] I1 worker timing: the patch hunk (every worker, our types), `--worker-timing`, `measure.py` and `worker_timing.json`, a fixture test
- [ ] I2 the SCLK spin: the `copy` task's spin mode, `KT_SPIN` in the suite, `measure.py` reports MHz
- [ ] I3 the empty-task ladder: `--graph empty --ops M --tasks N`, the placement read from the timing lines, `queue-c3.txt` (12 rows), tests
- [ ] I4 the fence knobs: the four defines in `gfx942.patch`, `--runtime-flags`, the offline compile per flag with the fence counts
- [ ] I5 the vLLM stage: `vm.sh vllm` in the background, the per-token latency and the ids, `DRY=1`, shellcheck, a test
- [ ] I6 the clock helper: `amd-smi metric --clock` loop into the record
- [ ] I7 `05-session-plan.md` and the rehearsal

Status:

## Before the VM

- [ ] every ticked item is on `main` (or the round's branch) with its test
- [ ] `env/preflight.sh` 8 PASS; `OFFLINE_COMPILE=1` adds the compile
- [ ] the README's status paragraph corrected (it still says the GPU days have not started)
- [ ] the balance read; the image on GHCR listed; `grab.sh` ready
