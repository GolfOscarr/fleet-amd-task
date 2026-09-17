# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | 3539.0 |
| time per iteration, P95 (us) |  | 10222.2 |
| time per iteration from event timing, median (us) |  | 741.5 |
| time per iteration from host wall clock (us) |  | 61285.8 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | 282.6 |
| workers with tasks (of those reporting) |  | 296 of 296 |
| tasks per XCD (placement) |  | 0:5287 1:5282 2:5283 3:5252 4:5253 5:5283 6:5283 7:5285 |
| shader clock from the spin (MHz) |  | - |
| exec cycles per task, busy workers |  | 37205.0 |
| exec us per task (at the spin's SCLK) |  | - |
| dep-wait us per iteration per busy worker |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2103 / 2104 over 37 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Exec time per task by class (worker timing, I1)

| Class | tasks | cycles per task | us per task |
|---|---|---|---|
| attend | 2496 | 93013 | - |
| linear | 11776 | 32443 | - |
| linear_res | 6144 | 46349 | - |
| merge | 1024 | 37723 | - |
| prep | 985 | 15645 | - |
| rms | 128 | 3788 | - |
| router | 32 | 32445 | - |
| silu | 256 | 2289 | - |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | iteration_start | 135.68 | 11.04 | 177.80 | 30 |
| 2 | embed_layer | 20.95 | 16.70 | 24.58 | 31 |
| 3 | rmsnorm_layer | 9.32 | 4.31 | 11.43 | 31 |
| 4 | linear_layer | 20.96 | 19.73 | 29.04 | 31 |
| 5 | mla_prep_layer | 18.09 | 16.48 | 19.80 | 31 |
| 6 | mla_attend_layer | 74.08 | 71.69 | 76.45 | 31 |
| 7 | mla_merge_uv_layer | 27.53 | 26.68 | 28.18 | 31 |
| 8 | linear_with_residual_layer | 19.93 | 19.41 | 20.46 | 31 |
| 9 | rmsnorm_layer | 9.63 | 7.93 | 11.51 | 31 |
| 10 | gang_linear_silu_layer | 36.14 | 35.37 | 37.61 | 31 |
| 11 | linear_with_residual_layer | 56.70 | 56.16 | 57.10 | 31 |
| 12 | rmsnorm_layer | 9.36 | 7.76 | 11.20 | 31 |
| 13 | linear_layer | 19.68 | 19.34 | 20.08 | 31 |
| 14 | mla_prep_layer | 17.95 | 16.08 | 20.49 | 31 |
| 15 | mla_attend_layer | 68.87 | 66.88 | 70.22 | 31 |
| 16 | mla_merge_uv_layer | 27.59 | 26.64 | 28.14 | 31 |
| 17 | linear_with_residual_layer | 20.09 | 19.51 | 20.44 | 31 |
| 18 | rmsnorm_layer | 9.21 | 7.32 | 11.28 | 31 |
| 19 | moe_router_layer | 22.96 | 20.90 | 25.10 | 31 |
| 20 | gang_moe_w13_linear_layer | 43.81 | 42.89 | 45.36 | 31 |
| 21 | moe_silu_mul_layer | 10.14 | 8.74 | 10.80 | 31 |
| 22 | gang_moe_w2_linear_layer | 25.26 | 24.59 | 26.21 | 31 |
| 23 | moe_mul_sum_add_layer | 9.98 | 9.18 | 10.53 | 31 |
