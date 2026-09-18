# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | 532.3 |
| time per iteration from host wall clock (us) |  | 1099.0 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2107 / 2107 over 33 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | iteration_start | 167.54 | 7.76 | 224.44 | 30 |
| 2 | embed_layer | 11.70 | 10.58 | 13.18 | 31 |
| 3 | linear_gemv_layer | 18.10 | 16.40 | 19.52 | 31 |
| 4 | mla_prep_layer | 9.70 | 9.10 | 11.90 | 31 |
| 5 | mla_attend_layer | 11.27 | 10.96 | 11.64 | 31 |
| 6 | mla_merge_uv_layer | 21.77 | 21.12 | 23.40 | 31 |
| 7 | linear_gemv_layer | 10.88 | 10.31 | 12.41 | 31 |
| 8 | rmsnorm_layer | 4.08 | 3.28 | 4.97 | 31 |
| 9 | gang_linear_silu_layer | 33.64 | 32.24 | 35.16 | 31 |
| 10 | linear_with_residual_layer | 50.58 | 49.14 | 51.88 | 31 |
| 11 | linear_gemv_layer | 15.93 | 15.55 | 16.32 | 31 |
| 12 | mla_prep_layer | 9.61 | 9.26 | 10.00 | 31 |
| 13 | mla_attend_layer | 11.26 | 10.92 | 11.65 | 31 |
| 14 | mla_merge_uv_layer | 21.42 | 20.87 | 22.91 | 31 |
| 15 | linear_gemv_layer | 10.84 | 9.58 | 12.84 | 31 |
| 16 | moe_router_layer | 16.79 | 16.08 | 17.44 | 31 |
| 17 | gang_moe_w13_linear_layer | 40.43 | 38.36 | 42.71 | 31 |
| 18 | gang_moe_w2_silu_linear_layer | 34.68 | 32.65 | 36.74 | 31 |
| 19 | moe_mul_sum_add_layer | 3.06 | 2.81 | 4.32 | 31 |
