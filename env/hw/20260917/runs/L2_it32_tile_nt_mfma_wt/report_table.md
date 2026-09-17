# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | 1568.0 |
| time per iteration, P95 (us) |  | 9461.3 |
| time per iteration from event timing, median (us) |  | 778.7 |
| time per iteration from host wall clock (us) |  | 87223.3 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | 637.8 |
| workers with tasks (of those reporting) |  | 296 of 296 |
| tasks per XCD (placement) |  | 0:5156 1:5156 2:5164 3:5158 4:5157 5:5153 6:5152 7:5152 |
| shader clock from the spin (MHz) |  | - |
| exec cycles per task, busy workers |  | 37739.3 |
| exec us per task (at the spin's SCLK) |  | - |
| dep-wait us per iteration per busy worker |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2102 / 2104 over 39 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Exec time per task by class (worker timing, I1)

| Class | tasks | cycles per task | us per task |
|---|---|---|---|
| attend | 2432 | 16274 | - |
| linear | 11776 | 40341 | - |
| linear_res | 6144 | 46944 | - |
| merge | 960 | 36282 | - |
| prep | 55 | 301548 | - |
| rms | 128 | 3243 | - |
| router | 32 | 38460 | - |
| silu | 256 | 2566 | - |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | embed_layer | 128.81 | 9.00 | 155.64 | 30 |
| 2 | rmsnorm_layer | 13.04 | 12.21 | 15.25 | 31 |
| 3 | linear_layer | 3.89 | 3.71 | 4.11 | 31 |
| 4 | mla_prep_layer | 23.94 | 22.06 | 26.48 | 31 |
| 5 | mla_attend_layer | 142.01 | 139.42 | 145.37 | 31 |
| 6 | mla_merge_uv_layer | 12.86 | 12.09 | 16.20 | 31 |
| 7 | linear_with_residual_layer | 25.42 | 21.79 | 26.53 | 31 |
| 8 | rmsnorm_layer | 13.41 | 12.69 | 14.16 | 31 |
| 9 | gang_linear_silu_layer | 4.06 | 3.58 | 4.71 | 31 |
| 10 | linear_with_residual_layer | 35.58 | 34.69 | 36.47 | 31 |
| 11 | rmsnorm_layer | 51.61 | 49.88 | 53.12 | 31 |
| 12 | linear_layer | 3.90 | 3.60 | 4.76 | 31 |
| 13 | mla_prep_layer | 13.71 | 12.80 | 14.16 | 31 |
| 14 | mla_attend_layer | 147.12 | 144.00 | 151.08 | 31 |
| 15 | mla_merge_uv_layer | 12.26 | 11.92 | 12.60 | 31 |
| 16 | linear_with_residual_layer | 22.17 | 21.53 | 23.08 | 31 |
| 17 | rmsnorm_layer | 13.47 | 12.99 | 14.04 | 31 |
| 18 | moe_router_layer | 3.96 | 3.68 | 4.19 | 31 |
| 19 | gang_moe_w13_linear_layer | 21.75 | 20.02 | 23.50 | 31 |
| 20 | moe_silu_mul_layer | 39.51 | 37.68 | 41.49 | 31 |
| 21 | gang_moe_w2_linear_layer | 5.40 | 3.94 | 6.53 | 31 |
| 22 | moe_mul_sum_add_layer | 20.77 | 19.25 | 22.45 | 31 |
| 23 | event_23 | 3.24 | 3.06 | 3.65 | 31 |
