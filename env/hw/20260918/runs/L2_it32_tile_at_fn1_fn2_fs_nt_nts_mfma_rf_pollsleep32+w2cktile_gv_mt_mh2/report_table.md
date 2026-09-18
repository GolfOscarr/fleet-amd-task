# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | 510.7 |
| time per iteration from host wall clock (us) |  | 1066.0 |
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
| 1 | iteration_start | 159.34 | 8.36 | 213.80 | 30 |
| 2 | embed_layer | 12.52 | 10.31 | 14.59 | 31 |
| 3 | linear_gemv_layer | 24.68 | 22.23 | 27.07 | 31 |
| 4 | mla_prep_layer | 9.91 | 9.41 | 10.89 | 31 |
| 5 | mla_attend_layer | 13.41 | 12.25 | 14.14 | 31 |
| 6 | mla_merge_uv_tile_layer | 15.83 | 15.29 | 16.30 | 31 |
| 7 | linear_gemv_layer | 8.77 | 8.22 | 9.25 | 31 |
| 8 | rmsnorm_layer | 4.25 | 3.61 | 4.75 | 31 |
| 9 | gang_linear_silu_layer | 34.11 | 33.37 | 34.70 | 31 |
| 10 | linear_with_residual_layer | 50.86 | 50.44 | 51.68 | 31 |
| 11 | linear_gemv_layer | 12.17 | 11.64 | 13.30 | 31 |
| 12 | mla_prep_layer | 9.47 | 8.82 | 10.79 | 31 |
| 13 | mla_attend_layer | 13.02 | 12.49 | 13.73 | 31 |
| 14 | mla_merge_uv_tile_layer | 15.65 | 14.85 | 16.06 | 31 |
| 15 | linear_gemv_layer | 8.07 | 7.65 | 8.34 | 31 |
| 16 | moe_router_layer | 15.67 | 14.75 | 16.54 | 31 |
| 17 | gang_moe_w13_linear_layer | 42.98 | 42.17 | 45.05 | 31 |
| 18 | gang_moe_w2_silu_linear_layer | 23.68 | 22.80 | 24.57 | 31 |
| 19 | moe_mul_sum_add_layer | 4.15 | 3.94 | 4.34 | 31 |
