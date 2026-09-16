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
| The fault is fixed | 8 layers with the head, 2 iterations, `fwd=2` | `runs/L8_head_it2_al65536` (A7) | PASS: `fwd=2`, and the same for `_wsfirst`, `_al65536_wsfirst`, `_al2097152_wsfirst`; then 27 layers with the head at 32 iterations |
| The kernels after P6 | 7 suites, 100 of 100 | `fleet/tasks/results/kernel_tests.json` (A4) | PASS: 7 suites, 100 of 100 each, twice (before and after the runtime rebuild) |
| Every flag on, still correct | B4 `compare=PASS` | `runs/L27_head_it32_<flags>` (B4) | - |

## The fault (M4)

| Question | Answer | Where |
|---|---|---|
| Reproduces on this machine | yes: `L8_head_it2 FAIL rc=1 mpk=0 fault=1 fwd=0 wall=37s`, `AcceleratorError: an illegal memory access` before the first forward pass | A5.1 status row |
| Address-dependent | a uniform shift of every buffer by 1, 2 or 4 GiB does not change it (`_pad1`, `_pad2`, `_pad4` all fault); the 16-layer graph runs; aligning every buffer to 64 KiB fixes it: the low address bits decide, not the position | A5.2 rows: pads 1, 2, 4 GiB; the 16-layer pad run; A7 |
| First faulting label | `L0.gate_up` (`gang_linear_silu_layer`, 8 tasks, 22 tiles, the fused gate-up linear of the dense layer 0): stopped after `L0.norm2` the graph runs two iterations, stopped after `L0.gate_up` it faults; the layer-7 list of the plan faulted at its first label | `logs/bisect.result`; the wide list `env/session/queue-fault-all.txt` |
| The fix that passed | `--align-alloc 65536` (the first row); `--workspaces-first` and the combinations pass as well | A7 row |
| Why (the offset that overflowed or the buffer that moved) | open. In the failing run `act` (the 22,528-byte output) sits at `0x7114e0b21200`, 512-byte aligned in the small pool right after `k_pe_7`; `h` at `0x7116cb3f7600`; `W_gu_shuffled` at `0x711634c00000`; no tensor overlaps another. In the passing 16-layer run `act` is `0x786b0edf1a00`, also 512-byte aligned, next to the other activations. The generated code embeds the addresses. A kernel that assumes more than 512-byte alignment of one of its pointers, or a per-XCD partition offset that lands the last XCD past the end of a 512-byte-aligned buffer, fits the evidence; the stock `gang_linear_silu_kernel` is the place to read | `fleet_run_meta.json` `addresses` of `runs/L8_head_it2_L0.gate_up` and `runs/L16_head_it2_pad1`; `runs/L8_head_it2_al65536` |

## Time per iteration, 27 layers, 32 iterations

| Variant | Command difference | Median us | P95 us | Trace us | Against baseline |
|---|---|---|---|---|---|
| Baseline, 2x host, gang, old kernels | `runs/L27_it32` (2026-09-15) | 15,572 | - | - | 1.00 |
| Baseline, this host, gang, P6 kernels, `--align-alloc 65536` | A8.6, `runs/L27_head_it32_al65536` (with the head) | 15,019.5 (host clock, mean of 32) | - | - | 0.96 |
| `--tile-linears` | A8.5 (2 layers): 1,735.8 to 1,595.9 us per iteration; 27 layers in B2 | - | - | - | 0.92 on the 2-layer graph |
| `--nt-weights` | B5 | - | - | - | |
| Design band | `09-expected-performance.md` | 1,148 to 1,349 + 326 t_b | | | |

## Per-operator time (event gaps, mean over iterations; the 2-layer graph, 32 iterations)

Read with the A8.5 decision: the P6 prefetch changed nothing in `mla_attend`
(211 to 215 us) and `mla_merge_uv` moved with the neighbouring linear, not
with its own code (61 us gang, 46 us when `o_proj` is per-tile); the residual
linears sit on a 25 to 35 us floor whatever their size (`o_proj` 8 MB, `down`
46 MB); the plain and the silu gang linears read 92 MB in under 5 us, which no
memory system delivers, so the event gaps do not measure them. The `--tile-linears`
alone column was not run: the tile run carries the P6 kernels too.

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
| B0 attribution: `qkva` alone from the host clock | A8.4 | 4.4 (event gap) | at 4 iterations: -29 (the 15 ms launch cost hides it); the 32-iteration ladder of `queue-b0.txt` follows | | | rocprofv3 aborts on the torch wheel's second HIP runtime, so the trace column is empty |

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
| | | |
