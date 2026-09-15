# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | - |
| time per iteration from host wall clock (us) |  | 2213.8 |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | - |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | embed_layer | 777.93 | 8.24 | 3776.72 | 30 |
| 2 | rmsnorm_layer | 14.66 | 13.40 | 15.66 | 31 |
| 3 | gang_linear_layer | 4.06 | 3.60 | 4.84 | 31 |
| 4 | mla_prep_layer | 21.49 | 19.56 | 23.40 | 31 |
| 5 | mla_attend_layer | 211.19 | 207.82 | 215.34 | 31 |
| 6 | mla_merge_uv_layer | 51.42 | 50.99 | 51.63 | 31 |
| 7 | gang_linear_with_residual_layer | 36.85 | 35.23 | 37.68 | 31 |
| 8 | rmsnorm_layer | 15.31 | 13.60 | 17.28 | 31 |
| 9 | gang_linear_silu_layer | 4.45 | 3.72 | 6.06 | 31 |
| 10 | gang_linear_with_residual_layer | 34.62 | 32.28 | 37.36 | 31 |
| 11 | rmsnorm_layer | 50.16 | 49.50 | 51.24 | 31 |
| 12 | gang_linear_layer | 4.65 | 3.61 | 5.13 | 31 |
| 13 | mla_prep_layer | 13.38 | 12.55 | 14.48 | 31 |
| 14 | mla_attend_layer | 203.96 | 178.87 | 214.59 | 31 |
| 15 | mla_merge_uv_layer | 51.08 | 50.81 | 51.29 | 31 |
| 16 | gang_linear_with_residual_layer | 37.13 | 36.32 | 37.72 | 31 |
| 17 | rmsnorm_layer | 13.50 | 13.24 | 13.76 | 31 |
| 18 | moe_router_layer | 3.80 | 3.67 | 4.07 | 31 |
| 19 | gang_moe_w13_linear_layer | 25.30 | 23.58 | 26.34 | 31 |
| 20 | moe_silu_mul_layer | 40.15 | 37.96 | 43.44 | 31 |
| 21 | gang_moe_w2_linear_layer | 5.03 | 3.64 | 6.64 | 31 |
| 22 | moe_mul_sum_add_layer | 21.48 | 20.16 | 23.20 | 31 |
| 23 | event_23 | 2.95 | 2.62 | 3.34 | 31 |
