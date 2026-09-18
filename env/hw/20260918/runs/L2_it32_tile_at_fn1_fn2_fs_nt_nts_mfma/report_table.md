# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | 548.2 |
| time per iteration from host wall clock (us) |  | 1107.8 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2107 / 2108 over 34 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | iteration_start | 172.96 | 7.96 | 215.52 | 30 |
| 2 | embed_layer | 11.35 | 10.22 | 13.02 | 31 |
| 3 | linear_norm_layer | 28.88 | 26.53 | 30.93 | 31 |
| 4 | mla_prep_layer | 10.21 | 9.74 | 11.09 | 31 |
| 5 | mla_attend_layer | 11.36 | 10.55 | 11.68 | 31 |
| 6 | mla_merge_uv_layer | 22.11 | 21.20 | 23.80 | 31 |
| 7 | linear_with_residual_layer | 13.21 | 11.95 | 15.24 | 31 |
| 8 | rmsnorm_layer | 4.59 | 3.47 | 5.04 | 31 |
| 9 | gang_linear_silu_layer | 35.12 | 34.00 | 36.93 | 31 |
| 10 | linear_with_residual_layer | 50.81 | 49.53 | 52.56 | 31 |
| 11 | linear_norm_layer | 13.98 | 13.15 | 14.67 | 31 |
| 12 | mla_prep_layer | 10.36 | 9.86 | 10.97 | 31 |
| 13 | mla_attend_layer | 11.42 | 10.95 | 11.99 | 31 |
| 14 | mla_merge_uv_layer | 21.59 | 21.04 | 22.28 | 31 |
| 15 | linear_with_residual_layer | 13.00 | 11.40 | 14.92 | 31 |
| 16 | moe_router_layer | 14.30 | 13.86 | 14.88 | 31 |
| 17 | gang_moe_w13_linear_layer | 42.50 | 40.92 | 43.85 | 31 |
| 18 | gang_moe_w2_silu_linear_layer | 34.57 | 32.41 | 37.04 | 31 |
| 19 | moe_mul_sum_add_layer | 3.23 | 2.90 | 4.16 | 31 |
