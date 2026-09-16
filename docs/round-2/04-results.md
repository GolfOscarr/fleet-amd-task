# 04 - Results

Filled after the sessions from the record (`env/hw/<date>/runs/`). The
baseline column holds the 2026-09-15 numbers so every cell is a
comparison, not a bare value. A cell is left `-` until the run exists;
nothing is estimated.

## Correctness evidence

| Claim | Evidence | Run | Result |
|---|---|---|---|
| M2: layer 1 (MoE) validated end to end | 16 boundaries PASS, top-k exact, route log PASS | `runs/L2_it1` (2026-09-15) | PASS |
| M3: 27 layers run 32 iterations | `FWD_PASS` 32 times | `runs/L27_it32` (2026-09-15) | PASS |
| M3: growth curve, 27 per-layer errors | `compare.py` growth curve under the layer threshold (4.3e-3) | `runs/L27_it1_al65536` (A8.3, 2026-09-16) | FAIL by the rule: layers 0 to 4 at 3.7e-3 to 6.4e-3, then 2.97e-2 at layer 5, 2.3e-2, 2.0e-2, 1.8e-2, and a monotone decay to 7.1e-3 at layer 26; the jump is the top-k flip of step 0 at MoE index 4 (model layer 5: expert 49 in the reference, 2 in Fleet, a tie within the router floor 3.6e-3); the ids are unaffected |
| M4: 32 token ids equal to the reference | `fleet_output_ids.json` = `harness/ref/ref_output_ids.json` | `runs/L27_head_it32_al65536` (A8.2, 2026-09-16 18:04:47) | PASS: `output_ids PASS`, 32 of 32 equal, `head.B16.token fleet 25 ref 25`; the route log differs in 60 of 832 (step, layer) top-k sets, one expert in most, growing after step 22 (BF16 drift, ids unchanged); the head norm and logits rows compare the last iteration with the step-1 reference and are not evidence |
| The fault is fixed | 8 layers with the head, 2 iterations, `fwd=2` | `runs/L8_head_it2_al65536` (A7); `runs/L8_head_it2` and `runs/L27_head_it32` after commit ce3a317 (no flag) | PASS with the flag (`fwd=2`, also `_wsfirst` and the combinations); PASS without any flag once the plan backs every single-row activation with 16 rows: `L8_head_it2 PASS fwd=2`, `L27_head_it32 PASS fwd=31 output_ids PASS` |
| The kernels after P6 | 7 suites, 100 of 100 | `fleet/tasks/results/kernel_tests.json` (A4) | PASS: 7 suites, 100 of 100 each, twice (before and after the runtime rebuild) |
| Every flag on, still correct | B4 `compare=PASS` | `runs/L27_head_it32_tile_al65536` (B4) | `output_ids PASS`, 32 of 32 equal with `--tile-linears --align-alloc 65536`; the route log differs as in A8.2 (drift, not a wrong id) |

## The fault (M4)

| Question | Answer | Where |
|---|---|---|
| Reproduces on this machine | yes: `L8_head_it2 FAIL rc=1 mpk=0 fault=1 fwd=0 wall=37s`, `AcceleratorError: an illegal memory access` before the first forward pass | A5.1 status row |
| Address-dependent | a uniform shift of every buffer by 1, 2 or 4 GiB does not change it (`_pad1`, `_pad2`, `_pad4` all fault); the 16-layer graph runs; aligning every buffer to 64 KiB fixes it: the low address bits decide, not the position | A5.2 rows: pads 1, 2, 4 GiB; the 16-layer pad run; A7 |
| First faulting label | `L0.gate_up` (`gang_linear_silu_layer`, 8 tasks, 22 tiles, the fused gate-up linear of the dense layer 0): stopped after `L0.norm2` the graph runs two iterations, stopped after `L0.gate_up` it faults; the layer-7 list of the plan faulted at its first label | `logs/bisect.result`; the wide list `env/session/queue-fault-all.txt` |
| The fix that passed | `--align-alloc 65536` (the first row); `--workspaces-first` and the combinations pass as well; the fix at the cause is in the plan since ce3a317 (`ROW_SLACK = 16` in `build_graph.new_workspace`), verified without any flag | A7 row; `queue-fix2.txt` |
| Why (the offset that overflowed or the buffer that moved) | the silu gang kernel over-reads its input. `gang_linear_silu_kernel` (`gang_linear_mi300.cuh`, line 359) builds its own CK tile GEMM with `MPerBlock = 16` and loops `LoopM` M-tiles with no `num_active_tokens` mask, so at batch 1 it reads 16 rows of A, 64 KB, from a 4 KB `[1, 2048]` buffer; the plain and the residual gang kernels go through the shared masked path and read one row. Whether the 60 KB past the buffer are mapped depends on the layout: in the failing run the input `h` sits at `0x7116cb3f7600`, the highest activation of the tensor set with 35 KB to the end of its 2 MiB segment and nothing recorded above it; in the passing 16-layer run the same input has 93 KB of slack; with `--align-alloc 65536` every buffer is over-allocated and `h` has 256 KB. The 92 MB weight and the output are read and written in bounds. A uniform shift keeps the slack, so the pads changed nothing | `fleet_run_meta.json` `addresses` of `runs/L8_head_it2_L0.gate_up`, `runs/L16_head_it2_pad1`, `runs/L8_head_it2_al65536`; the kernel source |

## Time per iteration, 27 layers, 32 iterations

| Variant | Command difference | Median us | P95 us | Trace us | Against baseline |
|---|---|---|---|---|---|
| Baseline, 2x host, gang, old kernels | `runs/L27_it32` (2026-09-15) | 15,572 | - | - | 1.00 |
| Baseline, this host, gang, P6 kernels, `--align-alloc 65536` | A8.6, `runs/L27_head_it32_al65536` (with the head) | 15,019.5 (host clock, mean of 32) | - | - | 0.96 |
| `--tile-linears` | B2, `runs/L27_head_it32_tile_al65536` (2 layers in A8.5: 1,735.8 to 1,595.9) | 14,384.2 (host clock) | - | - | 0.92 |
| `--nt-weights` (E2) | B5, `runs/L27_head_it32_nt_al65536` (2 layers: 1,497.3) | 12,977.8 (host clock) | - | - | 0.83; `mla_attend` 215 to 150 us, `w13` 26 to 22 us |
| E2 and per-tile linears, no flag (the plan fix in place) | `runs/L27_head_it32_tile_nt` | 12,401.6 (host clock) | - | - | 0.83; 32 ids equal |
| the same with 61 splits (`--split 17`) | `runs/L27_head_it32_tile_nt_s17` | 12,634.9 (host clock) | - | - | 0.84; `mla_attend` 149 us as with 33 splits: the per-tile cost is fixed, not per row; 32 ids equal |
| Design band | `09-expected-performance.md` | 1,148 to 1,349 + 326 t_b | | | |

## Per-operator time (event gaps, mean over iterations; the 2-layer graph, 32 iterations)

Read with the A8.5 decision: the P6 prefetch changed nothing in `mla_attend`
(211 to 215 us) and `mla_merge_uv` moved with the neighbouring linear, not
with its own code (61 us gang, 46 us when `o_proj` is per-tile); the residual
linears sit on a 25 to 35 us floor whatever their size (`o_proj` 8 MB, `down`
46 MB); the plain and the silu gang linears read 92 MB in under 5 us, which no
memory system delivers, so the event gaps do not measure them. The `--tile-linears`
alone column was not run: the tile run carries the P6 kernels too.

Standalone (the suite binary, `KT_TIME=50`, the graph's grid of 8 x 5 tiles at step
1032 with all 33 splits live) the attention grid takes 38.4 us and the merge grid
11.5 us, against 145 to 215 us and 46 to 61 us inside the megakernel. The kernels
are not the floor; the gang dispatch of the runtime is, by 100 to 175 us per
attention operator (27 per iteration) and 35 to 50 us per merge. The residual
gang linears' 25 to 35 us floor is the same mechanism at a smaller size.

| Operator | Event | Baseline us (2026-09-15) | P6 kernels (`runs/L2_it32_al65536`) | `--tile-linears` alone | Both (`runs/L2_it32_tile_al65536`) | Bandwidth floor us |
|---|---|---|---|---|---|---|
| `qkva` (gang linear, 15 MB) | 3 | 4.1 | 4.3 | - | 3.9 (per-tile `linear_layer`, 96 tasks) | 3.5 |
| `mla_prep` | 4 | 21.5 | 22.3 | - | 24.8 | about 1 |
| `mla_attend` | 5 | 211.2 | 215.0 | - | 216.3 | 9 (36 KB per tile) |
| `mla_merge_uv` | 6 | 51.4 | 61.3 | - | 45.8 | about 1 |
| `o_proj` (residual gang, 8 MB) | 7 | 36.9 | 28.6 | - | 24.6 (per-tile `linear_with_residual_layer`, 64 tasks) | 2 |
| `norm2` | 8 | 15.3 | 14.1 | - | 13.2 | under 1 |
| `gate_up` (silu gang, 92 MB) | 9 | 4.5 | 4.7 | - | 4.1 | 21 |
| `down` (residual gang, 46 MB) | 10 | 34.6 | 34.2 | - | 33.9 | 11 |
| `moe_silu_mul` | 20 | 40.2 | 42.2 | - | 40.4 | under 1 |
| `moe_mul_sum_add` | 22 | 21.5 | 21.2 | - | 21.1 | under 1 |
| B0 attribution: `qkva` alone from the host clock | A8.4 | 4.4 (event gap) | +7.3 (the 32-iteration ladder, `runs/L2_it32_L0.qkva_al65536` minus `..._L0.norm1_al65536`); the ladder's other steps swing by -85 to +265 us, so its resolution is about 100 us; a graph with `norm1` alone costs 670 us per iteration | | | the persistent kernel is one dispatch, so a kernel trace would give the same host-side number |

## Traffic and bandwidth (B3, the megakernel's dispatches only)

| Quantity | Predicted | Measured | Where |
|---|---|---|---|
| bytes read per iteration (MiB) | 4,710 weights and cache; about 4,772 with activations | - | `metrics.json` traffic |
| bytes written per iteration (MiB) | | - | |
| achieved read bandwidth (TB/s) | 3.66 to 4.3 | - | |
| L2 hit rate | 16 to 17% (Fleet's batch-1 figure) | - | |
| launches per generation | 3 | - | `3 megakernel of N dispatches` |
| tokens per second | | - | |

## What remains open after round 2

| Item | State | Next |
|---|---|---|
| MAJ-7, the per-operator floor | measured: the gang path costs 100 to 175 us per attention operator and 25 to 50 us per gang linear or merge on top of the kernels (38 us and 11.5 us standalone) | move the attention and the merge off the gang path (one regular task per tile, the runtime's per-task partitions), then the residual linears; the design band (1.15 to 1.35 ms) needs the operators at their kernel time |
| E2 (`--nt-weights`) | the largest lever measured: 15.0 to 13.0 ms; `mla_attend` 215 to 150 us | keep on; understand why non-temporal weight loads change an attention kernel that reads the cache, not the weights (the runtime toggles the loads of every task) |
| B3, the counters | rocprofv3 cannot attach to the torch wheel's bundled runtime | a wheel built against the system ROCm, or the counters from a standalone binary (`copy_bytes` of round 1) |
| the route log at 32 steps | 60 of 832 top-k sets differ, ids equal | a tolerance rule for ties within the router floor |
