# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | 498.3 |
| time per iteration from host wall clock (us) |  | 1072.2 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2107 / 2108 over 33 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | iteration_start | 154.77 | 8.24 | 190.56 | 30 |
| 2 | embed_layer | 12.45 | 11.42 | 13.67 | 31 |
| 3 | linear_gemv_layer | 17.03 | 15.83 | 18.40 | 31 |
| 4 | mla_prep_layer | 9.42 | 8.95 | 10.17 | 31 |
| 5 | mla_attend_layer | 11.35 | 10.81 | 11.69 | 31 |
| 6 | mla_merge_uv_layer | 21.91 | 21.18 | 22.93 | 31 |
| 7 | linear_gemv_layer | 8.26 | 7.33 | 10.08 | 31 |
| 8 | rmsnorm_layer | 4.80 | 4.28 | 5.46 | 31 |
| 9 | gang_linear_silu_layer | 34.50 | 32.75 | 35.62 | 31 |
| 10 | linear_with_residual_layer | 50.19 | 48.68 | 51.84 | 31 |
| 11 | linear_gemv_layer | 13.51 | 13.11 | 14.03 | 31 |
| 12 | mla_prep_layer | 9.66 | 9.40 | 10.14 | 31 |
| 13 | mla_attend_layer | 11.21 | 10.54 | 11.75 | 31 |
| 14 | mla_merge_uv_layer | 21.77 | 21.22 | 23.19 | 31 |
| 15 | linear_gemv_layer | 7.65 | 6.70 | 10.12 | 31 |
| 16 | moe_router_layer | 16.59 | 14.94 | 17.14 | 31 |
| 17 | gang_moe_w13_linear_layer | 40.84 | 38.19 | 43.00 | 31 |
| 18 | gang_moe_w2_silu_linear_layer | 34.79 | 33.59 | 37.96 | 31 |
| 19 | moe_mul_sum_add_layer | 3.03 | 2.78 | 3.35 | 31 |
