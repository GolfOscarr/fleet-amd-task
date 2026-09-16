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
| M3: growth curve, 27 per-layer errors | `compare.py` growth curve under the layer threshold (4.3e-3) | `runs/L27_it1` with `--debug` (A8.3) | - |
| M4: 32 token ids equal to the reference | `fleet_output_ids.json` = `harness/ref/ref_output_ids.json` | `runs/L27_head_it32` (A8.2) | - |
| The fault is fixed | 8 layers with the head, 2 iterations, `fwd=2` | `runs/L8_head_it2_<flag>` (A7) | - |
| The kernels after P6 | 7 suites, 100 of 100 | `fleet/tasks/results/kernel_tests.json` (A4) | - |
| Every flag on, still correct | B4 `compare=PASS` | `runs/L27_head_it32_<flags>` (B4) | - |

## The fault (M4)

| Question | Answer | Where |
|---|---|---|
| Reproduces on this machine | | A5.1 status row |
| Address-dependent | | A5.2 rows: pads 1, 2, 4 GiB; the 16-layer pad run |
| First faulting label | | `logs/bisect.result` |
| The fix that passed | | A7 row |
| Why (the offset that overflowed or the buffer that moved) | | `fleet_run_meta.json` `addresses` of the failing and the passing run |

## Time per iteration, 27 layers, 32 iterations

| Variant | Command difference | Median us | P95 us | Trace us | Against baseline |
|---|---|---|---|---|---|
| Baseline, 2x host, gang, old kernels | `runs/L27_it32` (2026-09-15) | 15,572 | - | - | 1.00 |
| Baseline, this host, gang, P6 kernels | A8.6 | - | - | - | |
| `--tile-linears` | A8.5 then B2 | - | - | - | |
| `--nt-weights` | B5 | - | - | - | |
| Design band | `09-expected-performance.md` | 1,148 to 1,349 + 326 t_b | | | |

## Per-operator time (event gaps, mean over iterations; the 2-layer graph, 32 iterations)

| Operator | Event | Baseline us (2026-09-15) | P6 kernels | `--tile-linears` | Both | Bandwidth floor us |
|---|---|---|---|---|---|---|
| `qkva` (gang linear, 15 MB) | 3 | 4.1 | | | | 3.5 |
| `mla_prep` | 4 | 21.5 | | | | about 1 |
| `mla_attend` | 5 | 211.2 | | | | 9 (36 KB per tile) |
| `mla_merge_uv` | 6 | 51.4 | | | | about 1 |
| `o_proj` (residual gang, 8 MB) | 7 | 36.9 | | | | 2 |
| `norm2` | 8 | 15.3 | | | | under 1 |
| `gate_up` (silu gang, 92 MB) | 9 | 4.5 | | | | 21 |
| `down` (residual gang, 46 MB) | 10 | 34.6 | | | | 11 |
| `moe_silu_mul` | 20 | 40.2 | | | | under 1 |
| `moe_mul_sum_add` | 22 | 21.5 | | | | under 1 |
| B0 attribution: `qkva` alone from the trace | A8.4 | 4.4 (event gap) | | | | |

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
