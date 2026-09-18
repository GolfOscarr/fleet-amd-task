# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | 498.9 |
| time per iteration from host wall clock (us) |  | 1050.5 |
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
| 1 | iteration_start | 148.98 | 7.56 | 230.40 | 30 |
| 2 | embed_layer | 12.53 | 10.16 | 15.17 | 31 |
| 3 | linear_gemv_layer | 24.83 | 21.72 | 27.00 | 31 |
| 4 | mla_prep_layer | 9.63 | 9.25 | 10.63 | 31 |
| 5 | mla_attend_layer | 12.59 | 11.88 | 13.24 | 31 |
| 6 | mla_merge_uv_tile_layer | 15.09 | 14.60 | 15.64 | 31 |
| 7 | linear_gemv_layer | 7.75 | 7.24 | 8.31 | 31 |
| 8 | rmsnorm_layer | 3.87 | 3.20 | 4.68 | 31 |
| 9 | gang_linear_silu_layer | 35.62 | 33.32 | 36.97 | 31 |
| 10 | linear_with_residual_layer | 50.63 | 49.51 | 52.12 | 31 |
| 11 | linear_gemv_layer | 11.33 | 10.79 | 12.34 | 31 |
| 12 | mla_prep_layer | 9.07 | 8.49 | 9.67 | 31 |
| 13 | mla_attend_layer | 12.65 | 12.12 | 13.12 | 31 |
| 14 | mla_merge_uv_tile_layer | 14.91 | 14.35 | 15.41 | 31 |
| 15 | linear_gemv_layer | 6.89 | 6.49 | 7.25 | 31 |
| 16 | moe_router_layer | 16.30 | 14.52 | 17.32 | 31 |
| 17 | gang_moe_w13_linear_layer | 41.01 | 38.39 | 45.75 | 31 |
| 18 | gang_moe_w2_silu_linear_layer | 21.71 | 20.48 | 23.29 | 31 |
| 19 | moe_mul_sum_add_layer | 2.99 | 2.71 | 3.88 | 31 |
