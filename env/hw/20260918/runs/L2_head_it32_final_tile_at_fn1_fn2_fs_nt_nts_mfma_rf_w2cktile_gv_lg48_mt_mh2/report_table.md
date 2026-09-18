# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | - |
| time per iteration, P95 (us) |  | - |
| time per iteration from event timing, median (us) |  | 615.2 |
| time per iteration from host wall clock (us) |  | 1744.5 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2107 / 2108 over 35 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | iteration_start | 185.38 | 176.72 | 205.76 | 30 |
| 2 | embed_layer | 14.13 | 12.42 | 15.42 | 31 |
| 3 | linear_gemv_layer | 15.66 | 14.25 | 17.14 | 31 |
| 4 | mla_prep_layer | 9.30 | 8.78 | 9.78 | 31 |
| 5 | mla_attend_layer | 11.99 | 11.40 | 13.12 | 31 |
| 6 | mla_merge_uv_tile_layer | 15.11 | 14.33 | 15.64 | 31 |
| 7 | linear_gemv_layer | 7.71 | 7.23 | 8.31 | 31 |
| 8 | rmsnorm_layer | 4.04 | 3.41 | 5.02 | 31 |
| 9 | gang_linear_silu_layer | 34.59 | 32.67 | 36.19 | 31 |
| 10 | linear_with_residual_layer | 50.32 | 49.16 | 51.64 | 31 |
| 11 | linear_gemv_layer | 13.48 | 12.99 | 14.03 | 31 |
| 12 | mla_prep_layer | 9.58 | 9.00 | 10.21 | 31 |
| 13 | mla_attend_layer | 12.31 | 11.85 | 12.84 | 31 |
| 14 | mla_merge_uv_tile_layer | 15.09 | 14.36 | 15.70 | 31 |
| 15 | linear_gemv_layer | 6.95 | 6.59 | 7.33 | 31 |
| 16 | moe_router_layer | 15.01 | 14.11 | 16.76 | 31 |
| 17 | gang_moe_w13_linear_layer | 42.40 | 39.93 | 45.74 | 31 |
| 18 | gang_moe_w2_silu_linear_layer | 23.25 | 21.01 | 25.60 | 31 |
| 19 | moe_mul_sum_add_layer | 5.75 | 4.04 | 7.17 | 31 |
| 20 | linear_gemv_layer | 0.19 | 0.00 | 1.57 | 31 |
| 21 | argmax_partial_layer | 2.49 | 0.00 | 70.18 | 31 |
| 22 | argmax_reduce_layer | 0.27 | 0.00 | 1.30 | 31 |
| 23 | event_23 | 2.56 | 0.00 | 70.90 | 31 |
| 24 | event_24 | 2.44 | 0.00 | 70.50 | 31 |
| 25 | event_25 | 4.85 | 0.00 | 73.43 | 31 |
| 26 | event_26 | 7.11 | 0.00 | 73.24 | 31 |
| 27 | event_27 | 2.62 | 0.00 | 73.80 | 31 |
| 28 | event_28 | 0.19 | 0.00 | 0.80 | 31 |
| 29 | event_29 | 0.27 | 0.00 | 1.08 | 31 |
| 30 | event_30 | 0.19 | 0.00 | 1.19 | 31 |
| 31 | event_31 | 0.27 | 0.00 | 1.28 | 31 |
| 32 | event_32 | 0.27 | 0.00 | 1.44 | 31 |
| 33 | event_33 | 2.68 | 0.00 | 72.97 | 31 |
| 34 | event_34 | 2.49 | 0.00 | 70.86 | 31 |
| 35 | event_35 | 4.93 | 0.00 | 73.89 | 31 |
| 36 | event_36 | 0.17 | 0.00 | 0.82 | 31 |
| 37 | event_37 | 2.53 | 0.00 | 73.88 | 31 |
| 38 | event_38 | 2.56 | 0.00 | 72.36 | 31 |
| 39 | event_39 | 4.95 | 0.00 | 74.74 | 31 |
| 40 | event_40 | 7.19 | 0.00 | 74.07 | 31 |
| 41 | event_41 | 0.27 | 0.00 | 1.44 | 31 |
| 42 | event_42 | 0.18 | 0.00 | 1.00 | 31 |
| 43 | event_43 | 0.22 | 0.00 | 1.81 | 31 |
| 44 | event_44 | 4.92 | 0.00 | 73.73 | 31 |
| 45 | event_45 | 4.97 | 0.00 | 73.64 | 31 |
| 46 | event_46 | 0.20 | 0.00 | 0.85 | 31 |
| 47 | event_47 | 4.78 | 0.00 | 71.79 | 31 |
| 48 | event_48 | 2.52 | 0.00 | 70.92 | 31 |
| 49 | event_49 | 0.18 | 0.00 | 0.80 | 31 |
| 50 | event_50 | 0.24 | 0.00 | 2.28 | 31 |
| 51 | event_51 | 2.65 | 0.00 | 73.91 | 31 |
| 52 | event_52 | 0.19 | 0.00 | 0.95 | 31 |
| 53 | event_53 | 2.61 | 0.00 | 73.36 | 31 |
| 54 | event_54 | 2.55 | 0.00 | 71.23 | 31 |
| 55 | event_55 | 2.77 | 0.00 | 73.55 | 31 |
| 56 | event_56 | 0.26 | 0.00 | 2.76 | 31 |
| 57 | event_57 | 0.51 | 0.00 | 1.95 | 31 |
| 58 | event_58 | 6.02 | 0.04 | 32.79 | 31 |
| 59 | event_59 | 4.10 | 0.00 | 31.71 | 31 |
| 60 | event_60 | 1.16 | 0.00 | 22.72 | 31 |
| 61 | event_61 | 1.32 | 0.01 | 29.24 | 31 |
| 62 | event_62 | 4.18 | 0.00 | 30.08 | 31 |
| 63 | event_63 | 3.37 | 0.00 | 30.58 | 31 |
| 64 | event_64 | 2.22 | 0.00 | 30.53 | 31 |
| 65 | event_65 | 1.32 | 0.00 | 26.80 | 31 |
| 66 | event_66 | 1.39 | 0.00 | 28.56 | 31 |
| 67 | event_67 | 1.39 | 0.00 | 29.16 | 31 |
| 68 | event_68 | 5.32 | 0.00 | 30.35 | 31 |
| 69 | event_69 | 2.51 | 0.00 | 30.55 | 31 |
| 70 | event_70 | 5.41 | 4.90 | 5.77 | 31 |
| 71 | event_71 | 5.10 | 4.92 | 5.27 | 31 |
