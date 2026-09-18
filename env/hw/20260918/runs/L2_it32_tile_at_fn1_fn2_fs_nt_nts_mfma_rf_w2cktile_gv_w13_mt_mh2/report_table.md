# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | 498.4 |
| time per iteration from host wall clock (us) |  | 1052.5 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2107 / 2107 over 34 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | iteration_start | 175.19 | 7.52 | 247.16 | 30 |
| 2 | embed_layer | 11.12 | 10.29 | 13.24 | 31 |
| 3 | linear_gemv_layer | 25.49 | 23.28 | 27.64 | 31 |
| 4 | mla_prep_layer | 8.57 | 8.24 | 8.80 | 31 |
| 5 | mla_attend_layer | 12.95 | 12.04 | 13.92 | 31 |
| 6 | mla_merge_uv_tile_layer | 15.17 | 14.53 | 15.76 | 31 |
| 7 | linear_gemv_layer | 7.69 | 7.17 | 8.44 | 31 |
| 8 | rmsnorm_layer | 3.98 | 3.07 | 5.15 | 31 |
| 9 | gang_linear_silu_layer | 35.30 | 33.17 | 37.00 | 31 |
| 10 | linear_with_residual_layer | 50.52 | 48.92 | 52.52 | 31 |
| 11 | linear_gemv_layer | 11.32 | 10.21 | 12.36 | 31 |
| 12 | mla_prep_layer | 8.94 | 8.18 | 9.94 | 31 |
| 13 | mla_attend_layer | 12.86 | 11.78 | 13.53 | 31 |
| 14 | mla_merge_uv_tile_layer | 15.02 | 14.49 | 15.35 | 31 |
| 15 | linear_gemv_layer | 6.91 | 6.64 | 7.52 | 31 |
| 16 | moe_router_layer | 15.51 | 14.30 | 16.46 | 31 |
| 17 | gang_moe_w13_gemv_layer | 36.75 | 35.71 | 38.67 | 31 |
| 18 | gang_moe_w2_silu_linear_layer | 22.04 | 20.61 | 23.36 | 31 |
| 19 | moe_mul_sum_add_layer | 3.16 | 2.85 | 4.04 | 31 |
