# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | 505.9 |
| time per iteration from host wall clock (us) |  | 1055.4 |
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
| 1 | iteration_start | 160.67 | 7.48 | 216.00 | 30 |
| 2 | embed_layer | 10.88 | 10.06 | 12.85 | 31 |
| 3 | linear_gemv_layer | 25.03 | 22.38 | 26.34 | 31 |
| 4 | mla_prep_layer | 8.64 | 8.12 | 10.43 | 31 |
| 5 | mla_attend_layer | 11.30 | 10.76 | 11.68 | 31 |
| 6 | mla_merge_uv_layer | 21.69 | 21.00 | 23.60 | 31 |
| 7 | linear_gemv_layer | 8.16 | 7.05 | 9.68 | 31 |
| 8 | rmsnorm_layer | 3.61 | 3.25 | 4.80 | 31 |
| 9 | gang_linear_silu_layer | 34.91 | 32.66 | 36.30 | 31 |
| 10 | linear_with_residual_layer | 50.81 | 49.56 | 52.56 | 31 |
| 11 | linear_gemv_layer | 11.08 | 10.56 | 11.88 | 31 |
| 12 | mla_prep_layer | 8.63 | 8.30 | 9.20 | 31 |
| 13 | mla_attend_layer | 11.02 | 10.69 | 11.57 | 31 |
| 14 | mla_merge_uv_layer | 21.90 | 21.17 | 23.96 | 31 |
| 15 | linear_gemv_layer | 7.80 | 6.76 | 10.24 | 31 |
| 16 | moe_router_norm4_layer | 13.92 | 12.65 | 15.40 | 31 |
| 17 | gang_moe_w13_gemv_layer | 37.09 | 35.04 | 40.33 | 31 |
| 18 | gang_moe_w2_silu_linear_layer | 22.27 | 20.79 | 23.60 | 31 |
| 19 | moe_mul_sum_add_layer | 3.15 | 2.86 | 4.20 | 31 |
