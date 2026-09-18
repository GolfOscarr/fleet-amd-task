# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | 650.6 |
| time per iteration from host wall clock (us) |  | 1789.7 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2107 / 2107 over 36 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | iteration_start | 198.88 | 190.80 | 210.80 | 30 |
| 2 | embed_layer | 14.25 | 13.15 | 15.11 | 31 |
| 3 | linear_gemv_layer | 23.90 | 22.35 | 26.36 | 31 |
| 4 | mla_prep_layer | 9.62 | 9.14 | 11.22 | 31 |
| 5 | mla_attend_layer | 11.49 | 11.01 | 12.01 | 31 |
| 6 | mla_merge_uv_layer | 21.74 | 20.79 | 23.88 | 31 |
| 7 | linear_gemv_layer | 8.14 | 7.20 | 10.04 | 31 |
| 8 | rmsnorm_layer | 3.80 | 3.44 | 4.57 | 31 |
| 9 | gang_linear_silu_layer | 35.33 | 33.48 | 36.93 | 31 |
| 10 | linear_with_residual_layer | 50.21 | 49.16 | 51.36 | 31 |
| 11 | linear_gemv_layer | 11.15 | 10.43 | 11.82 | 31 |
| 12 | mla_prep_layer | 9.82 | 9.18 | 10.32 | 31 |
| 13 | mla_attend_layer | 11.40 | 11.05 | 12.00 | 31 |
| 14 | mla_merge_uv_layer | 21.76 | 20.94 | 23.54 | 31 |
| 15 | linear_gemv_layer | 7.80 | 6.61 | 10.48 | 31 |
| 16 | moe_router_layer | 15.82 | 13.96 | 17.33 | 31 |
| 17 | gang_moe_w13_linear_layer | 41.33 | 38.67 | 43.96 | 31 |
| 18 | gang_moe_w2_silu_linear_layer | 22.82 | 21.37 | 24.56 | 31 |
| 19 | moe_mul_sum_add_layer | 5.94 | 3.91 | 7.21 | 31 |
| 20 | linear_gemv_layer | 0.17 | 0.00 | 0.69 | 31 |
| 21 | argmax_partial_layer | 0.25 | 0.00 | 1.29 | 31 |
| 22 | argmax_reduce_layer | 2.62 | 0.00 | 76.00 | 31 |
| 23 | event_23 | 0.22 | 0.00 | 1.73 | 31 |
| 24 | event_24 | 7.30 | 0.00 | 74.99 | 31 |
| 25 | event_25 | 0.09 | 0.00 | 0.69 | 31 |
| 26 | event_26 | 2.58 | 0.00 | 76.29 | 31 |
| 27 | event_27 | 5.03 | 0.00 | 78.40 | 31 |
| 28 | event_28 | 0.21 | 0.00 | 1.63 | 31 |
| 29 | event_29 | 0.13 | 0.00 | 0.60 | 31 |
| 30 | event_30 | 7.27 | 0.00 | 74.30 | 31 |
| 31 | event_31 | 4.98 | 0.00 | 74.72 | 31 |
| 32 | event_32 | 2.52 | 0.00 | 73.59 | 31 |
| 33 | event_33 | 9.71 | 0.00 | 75.94 | 31 |
| 34 | event_34 | 2.60 | 0.00 | 73.80 | 31 |
| 35 | event_35 | 2.56 | 0.00 | 74.43 | 31 |
| 36 | event_36 | 2.60 | 0.00 | 75.78 | 31 |
| 37 | event_37 | 0.36 | 0.00 | 1.69 | 31 |
| 38 | event_38 | 2.48 | 0.00 | 71.93 | 31 |
| 39 | event_39 | 2.48 | 0.00 | 72.12 | 31 |
| 40 | event_40 | 5.04 | 0.00 | 75.74 | 31 |
| 41 | event_41 | 0.21 | 0.00 | 1.33 | 31 |
| 42 | event_42 | 4.89 | 0.00 | 74.23 | 31 |
| 43 | event_43 | 4.89 | 0.00 | 74.63 | 31 |
| 44 | event_44 | 0.23 | 0.00 | 0.83 | 31 |
| 45 | event_45 | 0.16 | 0.00 | 0.60 | 31 |
| 46 | event_46 | 4.87 | 0.00 | 72.97 | 31 |
| 47 | event_47 | 0.17 | 0.00 | 0.72 | 31 |
| 48 | event_48 | 0.21 | 0.00 | 1.24 | 31 |
| 49 | event_49 | 0.20 | 0.00 | 1.01 | 31 |
| 50 | event_50 | 0.19 | 0.00 | 1.16 | 31 |
| 51 | event_51 | 2.54 | 0.00 | 73.00 | 31 |
| 52 | event_52 | 0.74 | 0.04 | 2.43 | 31 |
| 53 | event_53 | 0.91 | 0.00 | 3.52 | 31 |
| 54 | event_54 | 0.67 | 0.00 | 3.12 | 31 |
| 55 | event_55 | 0.65 | 0.00 | 2.54 | 31 |
| 56 | event_56 | 0.46 | 0.00 | 1.96 | 31 |
| 57 | event_57 | 1.21 | 0.00 | 29.57 | 31 |
| 58 | event_58 | 3.90 | 0.00 | 28.36 | 31 |
| 59 | event_59 | 4.66 | 0.01 | 28.16 | 31 |
| 60 | event_60 | 2.83 | 0.00 | 28.76 | 31 |
| 61 | event_61 | 2.15 | 0.00 | 28.59 | 31 |
| 62 | event_62 | 2.13 | 0.02 | 29.28 | 31 |
| 63 | event_63 | 3.00 | 0.00 | 28.57 | 31 |
| 64 | event_64 | 2.23 | 0.00 | 29.00 | 31 |
| 65 | event_65 | 1.29 | 0.00 | 28.20 | 31 |
| 66 | event_66 | 1.21 | 0.00 | 27.84 | 31 |
| 67 | event_67 | 1.25 | 0.00 | 28.44 | 31 |
| 68 | event_68 | 2.91 | 0.00 | 28.44 | 31 |
| 69 | event_69 | 2.96 | 0.01 | 27.44 | 31 |
| 70 | event_70 | 5.23 | 4.97 | 5.59 | 31 |
| 71 | event_71 | 5.02 | 4.84 | 5.16 | 31 |
