# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | - |
| time per iteration from host wall clock (us) |  | 3374.2 |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | - |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | embed_layer | 135.73 | 8.12 | 207.96 | 6 |
| 2 | rmsnorm_layer | 13.82 | 12.96 | 14.71 | 7 |
| 3 | gang_linear_layer | 4.19 | 3.93 | 4.49 | 7 |
| 4 | mla_prep_layer | 23.74 | 23.23 | 24.59 | 7 |
| 5 | mla_attend_layer | 208.36 | 205.13 | 210.97 | 7 |
| 6 | mla_merge_uv_layer | 50.75 | 50.52 | 51.04 | 7 |
| 7 | gang_linear_with_residual_layer | 39.52 | 39.12 | 40.12 | 7 |
| 8 | rmsnorm_layer | 13.58 | 13.48 | 13.68 | 7 |
| 9 | gang_linear_silu_layer | 5.24 | 4.39 | 6.95 | 7 |
| 10 | gang_linear_with_residual_layer | 35.53 | 34.28 | 36.48 | 7 |
| 11 | rmsnorm_layer | 50.51 | 49.30 | 50.92 | 7 |
| 12 | gang_linear_layer | 4.49 | 3.79 | 4.87 | 7 |
| 13 | mla_prep_layer | 13.76 | 13.52 | 14.52 | 7 |
| 14 | mla_attend_layer | 209.46 | 207.16 | 212.08 | 7 |
| 15 | mla_merge_uv_layer | 50.91 | 50.73 | 51.05 | 7 |
| 16 | gang_linear_with_residual_layer | 36.13 | 35.87 | 36.63 | 7 |
| 17 | rmsnorm_layer | 13.59 | 13.48 | 13.76 | 7 |
| 18 | moe_router_layer | 3.74 | 3.59 | 3.95 | 7 |
| 19 | gang_moe_w13_linear_layer | 25.15 | 23.69 | 26.85 | 7 |
| 20 | moe_silu_mul_layer | 41.01 | 39.92 | 42.24 | 7 |
| 21 | gang_moe_w2_linear_layer | 5.80 | 4.16 | 6.28 | 7 |
| 22 | moe_mul_sum_add_layer | 20.53 | 19.76 | 21.60 | 7 |
| 23 | event_23 | 2.96 | 2.84 | 3.29 | 7 |
