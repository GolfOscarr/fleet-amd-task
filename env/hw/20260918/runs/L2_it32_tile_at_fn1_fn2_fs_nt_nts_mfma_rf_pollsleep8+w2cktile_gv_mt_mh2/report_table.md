# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | 492.4 |
| time per iteration from host wall clock (us) |  | 1044.9 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2106 / 2107 over 35 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | iteration_start | 153.21 | 8.40 | 204.04 | 30 |
| 2 | embed_layer | 11.36 | 10.48 | 13.65 | 31 |
| 3 | linear_gemv_layer | 24.89 | 22.07 | 27.44 | 31 |
| 4 | mla_prep_layer | 9.18 | 8.72 | 9.81 | 31 |
| 5 | mla_attend_layer | 12.73 | 11.86 | 13.40 | 31 |
| 6 | mla_merge_uv_tile_layer | 15.18 | 14.44 | 15.88 | 31 |
| 7 | linear_gemv_layer | 7.79 | 7.11 | 8.48 | 31 |
| 8 | rmsnorm_layer | 4.06 | 3.22 | 4.95 | 31 |
| 9 | gang_linear_silu_layer | 34.92 | 32.69 | 36.21 | 31 |
| 10 | linear_with_residual_layer | 50.55 | 49.20 | 52.04 | 31 |
| 11 | linear_gemv_layer | 11.36 | 10.68 | 12.55 | 31 |
| 12 | mla_prep_layer | 8.39 | 7.57 | 9.21 | 31 |
| 13 | mla_attend_layer | 12.35 | 11.72 | 12.83 | 31 |
| 14 | mla_merge_uv_tile_layer | 15.03 | 14.47 | 15.52 | 31 |
| 15 | linear_gemv_layer | 6.94 | 6.53 | 7.39 | 31 |
| 16 | moe_router_layer | 15.79 | 14.33 | 17.89 | 31 |
| 17 | gang_moe_w13_linear_layer | 41.07 | 38.28 | 43.19 | 31 |
| 18 | gang_moe_w2_silu_linear_layer | 21.85 | 20.40 | 24.20 | 31 |
| 19 | moe_mul_sum_add_layer | 2.87 | 2.69 | 3.68 | 31 |
