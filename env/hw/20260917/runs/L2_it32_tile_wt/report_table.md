# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | 2185.5 |
| time per iteration, P95 (us) |  | 8724.0 |
| time per iteration from event timing, median (us) |  | 618.3 |
| time per iteration from host wall clock (us) |  | 61831.3 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | 457.6 |
| workers with tasks (of those reporting) |  | 296 of 296 |
| tasks per XCD (placement) |  | 0:5289 1:5285 2:5285 3:5254 4:5250 5:5284 6:5280 7:5281 |
| shader clock from the spin (MHz) |  | - |
| exec cycles per task, busy workers |  | 42446.0 |
| exec us per task (at the spin's SCLK) |  | - |
| dep-wait us per iteration per busy worker |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2103 / 2104 over 37 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Exec time per task by class (worker timing, I1)

| Class | tasks | cycles per task | us per task |
|---|---|---|---|
| attend | 2496 | 93107 | - |
| linear | 11776 | 38313 | - |
| linear_res | 6144 | 46834 | - |
| merge | 1024 | 38004 | - |
| prep | 955 | 16363 | - |
| rms | 128 | 3758 | - |
| router | 32 | 32037 | - |
| silu | 256 | 2294 | - |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | iteration_start | 126.35 | 8.60 | 164.24 | 30 |
| 2 | embed_layer | 14.91 | 13.56 | 16.83 | 31 |
| 3 | rmsnorm_layer | 4.28 | 3.83 | 4.51 | 31 |
| 4 | linear_layer | 23.52 | 21.65 | 26.17 | 31 |
| 5 | mla_prep_layer | 12.46 | 12.16 | 13.36 | 31 |
| 6 | mla_attend_layer | 63.39 | 57.12 | 65.24 | 31 |
| 7 | mla_merge_uv_layer | 24.83 | 21.56 | 25.68 | 31 |
| 8 | linear_with_residual_layer | 13.33 | 12.79 | 14.04 | 31 |
| 9 | rmsnorm_layer | 4.75 | 4.00 | 5.63 | 31 |
| 10 | gang_linear_silu_layer | 33.81 | 32.36 | 34.93 | 31 |
| 11 | linear_with_residual_layer | 51.44 | 50.30 | 52.92 | 31 |
| 12 | rmsnorm_layer | 4.41 | 3.94 | 4.94 | 31 |
| 13 | linear_layer | 13.53 | 12.72 | 14.14 | 31 |
| 14 | mla_prep_layer | 12.65 | 12.07 | 13.67 | 31 |
| 15 | mla_attend_layer | 63.02 | 58.32 | 64.39 | 31 |
| 16 | mla_merge_uv_layer | 21.99 | 21.33 | 24.08 | 31 |
| 17 | linear_with_residual_layer | 13.62 | 13.04 | 14.75 | 31 |
| 18 | rmsnorm_layer | 4.01 | 3.77 | 4.27 | 31 |
| 19 | moe_router_layer | 18.19 | 17.14 | 19.62 | 31 |
| 20 | gang_moe_w13_linear_layer | 42.35 | 40.52 | 44.87 | 31 |
| 21 | moe_silu_mul_layer | 5.63 | 3.83 | 6.39 | 31 |
| 22 | gang_moe_w2_linear_layer | 21.46 | 20.03 | 23.76 | 31 |
| 23 | moe_mul_sum_add_layer | 3.44 | 3.18 | 4.22 | 31 |
