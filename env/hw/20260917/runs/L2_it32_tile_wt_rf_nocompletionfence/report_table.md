# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | 3134.0 |
| time per iteration, P95 (us) |  | 9340.0 |
| time per iteration from event timing, median (us) |  | 626.8 |
| time per iteration from host wall clock (us) |  | 92237.3 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | 319.1 |
| workers with tasks (of those reporting) |  | 296 of 296 |
| tasks per XCD (placement) |  | 0:5285 1:5281 2:5284 3:5254 4:5252 5:5283 6:5285 7:5284 |
| shader clock from the spin (MHz) |  | - |
| exec cycles per task, busy workers |  | 42175.7 |
| exec us per task (at the spin's SCLK) |  | - |
| dep-wait us per iteration per busy worker |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2103 / 2104 over 38 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Exec time per task by class (worker timing, I1)

| Class | tasks | cycles per task | us per task |
|---|---|---|---|
| attend | 2368 | 96070 | - |
| linear | 11776 | 39658 | - |
| linear_res | 6144 | 46860 | - |
| merge | 960 | 37507 | - |
| prep | 955 | 15892 | - |
| rms | 128 | 3775 | - |
| router | 32 | 32030 | - |
| silu | 256 | 2404 | - |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | iteration_start | 127.03 | 8.20 | 167.32 | 30 |
| 2 | embed_layer | 14.79 | 13.52 | 15.94 | 31 |
| 3 | rmsnorm_layer | 3.96 | 3.60 | 4.40 | 31 |
| 4 | linear_layer | 23.55 | 22.00 | 25.64 | 31 |
| 5 | mla_prep_layer | 11.11 | 10.61 | 12.75 | 31 |
| 6 | mla_attend_layer | 69.27 | 66.59 | 71.47 | 31 |
| 7 | mla_merge_uv_layer | 24.82 | 23.80 | 25.28 | 31 |
| 8 | linear_with_residual_layer | 13.43 | 13.18 | 13.72 | 31 |
| 9 | rmsnorm_layer | 3.95 | 3.65 | 4.23 | 31 |
| 10 | gang_linear_silu_layer | 34.55 | 27.17 | 35.93 | 31 |
| 11 | linear_with_residual_layer | 51.12 | 49.63 | 52.76 | 31 |
| 12 | rmsnorm_layer | 4.14 | 3.70 | 4.61 | 31 |
| 13 | linear_layer | 13.24 | 12.79 | 13.68 | 31 |
| 14 | mla_prep_layer | 12.26 | 11.40 | 13.33 | 31 |
| 15 | mla_attend_layer | 64.03 | 63.51 | 66.03 | 31 |
| 16 | mla_merge_uv_layer | 21.87 | 20.61 | 23.56 | 31 |
| 17 | linear_with_residual_layer | 13.41 | 12.93 | 14.69 | 31 |
| 18 | rmsnorm_layer | 3.68 | 3.47 | 3.91 | 31 |
| 19 | moe_router_layer | 18.29 | 16.79 | 20.43 | 31 |
| 20 | gang_moe_w13_linear_layer | 40.18 | 38.04 | 42.24 | 31 |
| 21 | moe_silu_mul_layer | 4.38 | 3.60 | 6.36 | 31 |
| 22 | gang_moe_w2_linear_layer | 22.47 | 20.62 | 23.84 | 31 |
| 23 | moe_mul_sum_add_layer | 3.09 | 2.86 | 3.65 | 31 |
