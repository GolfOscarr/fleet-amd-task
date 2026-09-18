# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | 498.7 |
| time per iteration from host wall clock (us) |  | 1049.4 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2107 / 2107 over 35 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | iteration_start | 161.51 | 7.88 | 216.64 | 30 |
| 2 | embed_layer | 11.49 | 10.60 | 13.19 | 31 |
| 3 | linear_gemv_layer | 25.10 | 23.44 | 26.64 | 31 |
| 4 | mla_prep_layer | 9.69 | 9.26 | 10.54 | 31 |
| 5 | mla_attend_layer | 13.22 | 12.24 | 14.04 | 31 |
| 6 | mla_merge_uv_tile_layer | 15.24 | 14.63 | 15.90 | 31 |
| 7 | linear_gemv_layer | 7.83 | 7.28 | 8.36 | 31 |
| 8 | rmsnorm_layer | 4.04 | 3.28 | 5.20 | 31 |
| 9 | gang_linear_silu_layer | 31.47 | 30.04 | 33.92 | 31 |
| 10 | linear_with_residual_layer | 50.57 | 49.15 | 51.84 | 31 |
| 11 | linear_gemv_layer | 11.38 | 10.69 | 12.27 | 31 |
| 12 | mla_prep_layer | 9.38 | 8.74 | 9.86 | 31 |
| 13 | mla_attend_layer | 12.97 | 12.38 | 13.42 | 31 |
| 14 | mla_merge_uv_tile_layer | 15.07 | 14.51 | 15.56 | 31 |
| 15 | linear_gemv_layer | 6.94 | 6.47 | 7.42 | 31 |
| 16 | moe_router_layer | 16.87 | 14.55 | 17.56 | 31 |
| 17 | gang_moe_w13_linear_layer | 42.77 | 41.21 | 45.71 | 31 |
| 18 | gang_moe_w2_silu_linear_layer | 22.25 | 20.43 | 24.72 | 31 |
| 19 | moe_mul_sum_add_layer | 2.96 | 2.72 | 3.76 | 31 |
