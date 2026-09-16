# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | - |
| time per iteration from host wall clock (us) |  | 1676.6 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | - |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | embed_layer | 145.21 | 8.36 | 182.76 | 30 |
| 2 | rmsnorm_layer | 14.38 | 13.12 | 15.43 | 31 |
| 3 | gang_linear_layer | 4.21 | 3.71 | 4.75 | 31 |
| 4 | mla_prep_layer | 21.06 | 19.63 | 22.59 | 31 |
| 5 | mla_attend_layer | 210.25 | 205.80 | 214.11 | 31 |
| 6 | mla_merge_uv_layer | 64.75 | 64.01 | 65.13 | 31 |
| 7 | gang_linear_with_residual_layer | 28.71 | 26.04 | 30.04 | 31 |
| 8 | rmsnorm_layer | 13.80 | 13.36 | 15.29 | 31 |
| 9 | gang_linear_silu_layer | 4.49 | 3.76 | 5.92 | 31 |
| 10 | gang_linear_with_residual_layer | 34.13 | 32.30 | 35.96 | 31 |
| 11 | rmsnorm_layer | 50.33 | 49.44 | 50.88 | 31 |
| 12 | gang_linear_layer | 3.79 | 3.48 | 4.88 | 31 |
| 13 | mla_prep_layer | 14.48 | 13.60 | 15.00 | 31 |
| 14 | mla_attend_layer | 210.07 | 206.27 | 213.83 | 31 |
| 15 | mla_merge_uv_layer | 64.60 | 64.03 | 64.92 | 31 |
| 16 | gang_linear_with_residual_layer | 25.45 | 24.76 | 26.16 | 31 |
| 17 | rmsnorm_layer | 13.78 | 13.44 | 14.16 | 31 |
| 18 | moe_router_layer | 3.90 | 3.75 | 4.19 | 31 |
| 19 | gang_moe_w13_linear_layer | 24.91 | 23.35 | 26.23 | 31 |
| 20 | moe_silu_mul_layer | 42.76 | 41.21 | 45.42 | 31 |
| 21 | gang_moe_w2_linear_layer | 5.16 | 3.78 | 6.53 | 31 |
| 22 | moe_mul_sum_add_layer | 21.19 | 19.52 | 22.80 | 31 |
| 23 | event_23 | 3.10 | 2.85 | 3.43 | 31 |
