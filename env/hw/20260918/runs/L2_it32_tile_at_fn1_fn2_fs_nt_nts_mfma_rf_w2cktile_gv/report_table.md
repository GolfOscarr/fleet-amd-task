# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | 495.4 |
| time per iteration from host wall clock (us) |  | 1052.4 |
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
| 1 | iteration_start | 138.54 | 7.36 | 191.72 | 30 |
| 2 | embed_layer | 12.71 | 11.28 | 14.16 | 31 |
| 3 | linear_gemv_layer | 23.85 | 22.49 | 26.92 | 31 |
| 4 | mla_prep_layer | 9.88 | 9.35 | 10.27 | 31 |
| 5 | mla_attend_layer | 11.41 | 11.06 | 11.85 | 31 |
| 6 | mla_merge_uv_layer | 22.16 | 20.97 | 23.88 | 31 |
| 7 | linear_gemv_layer | 8.21 | 7.40 | 10.08 | 31 |
| 8 | rmsnorm_layer | 4.57 | 3.29 | 5.25 | 31 |
| 9 | gang_linear_silu_layer | 33.02 | 31.95 | 35.91 | 31 |
| 10 | linear_with_residual_layer | 50.73 | 49.52 | 52.08 | 31 |
| 11 | linear_gemv_layer | 11.21 | 10.67 | 11.90 | 31 |
| 12 | mla_prep_layer | 9.58 | 9.04 | 11.44 | 31 |
| 13 | mla_attend_layer | 11.30 | 10.78 | 11.67 | 31 |
| 14 | mla_merge_uv_layer | 22.16 | 21.08 | 24.56 | 31 |
| 15 | linear_gemv_layer | 8.05 | 6.53 | 10.08 | 31 |
| 16 | moe_router_layer | 16.18 | 14.13 | 17.28 | 31 |
| 17 | gang_moe_w13_linear_layer | 40.70 | 38.25 | 43.53 | 31 |
| 18 | gang_moe_w2_silu_linear_layer | 21.40 | 20.47 | 23.07 | 31 |
| 19 | moe_mul_sum_add_layer | 3.10 | 2.91 | 4.24 | 31 |
