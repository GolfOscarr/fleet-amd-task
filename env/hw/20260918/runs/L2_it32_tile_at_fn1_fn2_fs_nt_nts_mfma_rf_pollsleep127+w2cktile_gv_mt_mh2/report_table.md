# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | 575.4 |
| time per iteration from host wall clock (us) |  | 1130.3 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2107 / 2108 over 35 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | iteration_start | 157.55 | 13.88 | 210.08 | 30 |
| 2 | embed_layer | 17.95 | 11.83 | 21.39 | 31 |
| 3 | linear_gemv_layer | 25.74 | 20.04 | 32.64 | 31 |
| 4 | mla_prep_layer | 14.44 | 13.13 | 16.28 | 31 |
| 5 | mla_attend_layer | 17.84 | 17.28 | 18.36 | 31 |
| 6 | mla_merge_uv_tile_layer | 21.39 | 20.93 | 21.79 | 31 |
| 7 | linear_gemv_layer | 14.28 | 13.60 | 14.76 | 31 |
| 8 | rmsnorm_layer | 8.30 | 6.49 | 10.65 | 31 |
| 9 | gang_linear_silu_layer | 35.12 | 34.61 | 36.08 | 31 |
| 10 | linear_with_residual_layer | 55.88 | 55.26 | 56.55 | 31 |
| 11 | linear_gemv_layer | 17.20 | 16.72 | 17.83 | 31 |
| 12 | mla_prep_layer | 13.51 | 12.56 | 14.81 | 31 |
| 13 | mla_attend_layer | 17.58 | 17.17 | 18.47 | 31 |
| 14 | mla_merge_uv_tile_layer | 21.02 | 20.53 | 21.46 | 31 |
| 15 | linear_gemv_layer | 13.83 | 13.31 | 14.20 | 31 |
| 16 | moe_router_layer | 19.82 | 17.61 | 21.57 | 31 |
| 17 | gang_moe_w13_linear_layer | 43.70 | 42.75 | 44.78 | 31 |
| 18 | gang_moe_w2_silu_linear_layer | 26.21 | 25.08 | 27.07 | 31 |
| 19 | moe_mul_sum_add_layer | 9.64 | 8.63 | 10.15 | 31 |
