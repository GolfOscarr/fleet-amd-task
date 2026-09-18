# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | 518.8 |
| time per iteration from host wall clock (us) |  | 1090.4 |
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
| 1 | iteration_start | 167.41 | 8.40 | 222.60 | 30 |
| 2 | embed_layer | 12.02 | 11.05 | 13.61 | 31 |
| 3 | linear_gemv_layer | 24.23 | 22.57 | 26.77 | 31 |
| 4 | mla_prep_layer | 9.78 | 9.45 | 10.23 | 31 |
| 5 | mla_attend_layer | 11.36 | 10.93 | 11.95 | 31 |
| 6 | mla_merge_uv_layer | 22.03 | 21.05 | 23.52 | 31 |
| 7 | linear_gemv_layer | 8.20 | 7.39 | 9.96 | 31 |
| 8 | rmsnorm_layer | 4.64 | 4.16 | 5.63 | 31 |
| 9 | gang_linear_silu_layer | 34.40 | 33.26 | 35.80 | 31 |
| 10 | linear_with_residual_layer | 50.72 | 49.60 | 52.32 | 31 |
| 11 | linear_gemv_layer | 11.24 | 10.47 | 12.04 | 31 |
| 12 | mla_prep_layer | 9.79 | 9.22 | 10.24 | 31 |
| 13 | mla_attend_layer | 11.23 | 10.85 | 11.73 | 31 |
| 14 | mla_merge_uv_layer | 21.79 | 21.07 | 23.59 | 31 |
| 15 | linear_gemv_layer | 8.22 | 6.63 | 10.04 | 31 |
| 16 | moe_router_layer | 16.70 | 16.06 | 17.42 | 31 |
| 17 | gang_moe_w13_linear_layer | 41.15 | 40.48 | 42.54 | 31 |
| 18 | gang_moe_w2_silu_linear_layer | 34.42 | 32.75 | 36.12 | 31 |
| 19 | moe_mul_sum_add_layer | 3.01 | 2.85 | 3.45 | 31 |
