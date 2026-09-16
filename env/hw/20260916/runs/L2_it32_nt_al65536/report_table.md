# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | - |
| time per iteration from host wall clock (us) |  | 1497.3 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | - |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | embed_layer | 147.68 | 7.72 | 185.68 | 30 |
| 2 | rmsnorm_layer | 11.76 | 10.79 | 12.76 | 31 |
| 3 | gang_linear_layer | 3.83 | 3.48 | 4.08 | 31 |
| 4 | mla_prep_layer | 24.10 | 22.84 | 24.96 | 31 |
| 5 | mla_attend_layer | 144.85 | 140.21 | 148.57 | 31 |
| 6 | mla_merge_uv_layer | 56.83 | 54.16 | 58.28 | 31 |
| 7 | gang_linear_with_residual_layer | 24.13 | 21.63 | 25.68 | 31 |
| 8 | rmsnorm_layer | 13.78 | 13.36 | 15.17 | 31 |
| 9 | gang_linear_silu_layer | 5.06 | 3.80 | 7.17 | 31 |
| 10 | gang_linear_with_residual_layer | 32.55 | 31.15 | 33.80 | 31 |
| 11 | rmsnorm_layer | 50.39 | 49.44 | 51.36 | 31 |
| 12 | gang_linear_layer | 4.64 | 3.68 | 5.03 | 31 |
| 13 | mla_prep_layer | 13.64 | 12.76 | 14.84 | 31 |
| 14 | mla_attend_layer | 147.49 | 144.76 | 150.88 | 31 |
| 15 | mla_merge_uv_layer | 59.21 | 58.57 | 60.29 | 31 |
| 16 | gang_linear_with_residual_layer | 22.05 | 21.16 | 23.04 | 31 |
| 17 | rmsnorm_layer | 13.63 | 13.40 | 13.96 | 31 |
| 18 | moe_router_layer | 3.64 | 3.39 | 3.95 | 31 |
| 19 | gang_moe_w13_linear_layer | 19.83 | 18.93 | 21.29 | 31 |
| 20 | moe_silu_mul_layer | 40.82 | 38.51 | 42.75 | 31 |
| 21 | gang_moe_w2_linear_layer | 5.62 | 3.56 | 6.56 | 31 |
| 22 | moe_mul_sum_add_layer | 20.34 | 19.00 | 22.28 | 31 |
| 23 | event_23 | 3.11 | 3.00 | 3.45 | 31 |
