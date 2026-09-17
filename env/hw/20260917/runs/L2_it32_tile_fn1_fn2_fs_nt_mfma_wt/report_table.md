# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | 1624.0 |
| time per iteration, P95 (us) |  | 10152.2 |
| time per iteration from event timing, median (us) |  | 807.2 |
| time per iteration from host wall clock (us) |  | 56048.6 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | 615.8 |
| workers with tasks (of those reporting) |  | 296 of 296 |
| tasks per XCD (placement) |  | 0:5123 1:5123 2:5088 3:5088 4:5088 5:5120 6:5131 7:5135 |
| shader clock from the spin (MHz) |  | - |
| exec cycles per task, busy workers |  | 40546.9 |
| exec us per task (at the spin's SCLK) |  | - |
| dep-wait us per iteration per busy worker |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2102 / 2103 over 39 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Exec time per task by class (worker timing, I1)

| Class | tasks | cycles per task | us per task |
|---|---|---|---|
| attend | 2304 | 16594 | - |
| linear | 5632 | 58641 | - |
| linear_res | 6123 | 46748 | - |
| lnorm | 5481 | 30192 | - |
| merge | 960 | 37059 | - |
| prep | 64 | 307726 | - |
| rms | 32 | 3700 | - |
| router | 32 | 47114 | - |
| w2silu | 8448 | 32143 | - |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | embed_layer | 159.96 | 9.24 | 213.40 | 30 |
| 2 | linear_norm_layer | 12.75 | 11.65 | 14.72 | 31 |
| 3 | mla_prep_layer | 30.46 | 28.73 | 32.08 | 31 |
| 4 | mla_attend_layer | 146.84 | 142.87 | 149.68 | 31 |
| 5 | mla_merge_uv_layer | 12.70 | 12.17 | 15.96 | 31 |
| 6 | linear_with_residual_layer | 24.66 | 21.93 | 25.64 | 31 |
| 7 | rmsnorm_layer | 13.61 | 13.07 | 13.92 | 31 |
| 8 | gang_linear_silu_layer | 4.63 | 3.99 | 5.46 | 31 |
| 9 | linear_with_residual_layer | 34.29 | 32.58 | 36.02 | 31 |
| 10 | linear_norm_layer | 51.61 | 50.16 | 53.08 | 31 |
| 11 | mla_prep_layer | 18.25 | 17.52 | 18.81 | 31 |
| 12 | mla_attend_layer | 148.01 | 145.64 | 150.60 | 31 |
| 13 | mla_merge_uv_layer | 13.08 | 12.20 | 13.67 | 31 |
| 14 | linear_with_residual_layer | 22.26 | 22.03 | 22.59 | 31 |
| 15 | moe_router_layer | 12.82 | 12.25 | 13.64 | 31 |
| 16 | gang_moe_w13_linear_layer | 25.49 | 23.00 | 27.76 | 31 |
| 17 | gang_moe_w2_silu_linear_layer | 41.40 | 38.51 | 44.31 | 31 |
| 18 | moe_mul_sum_add_layer | 23.33 | 21.45 | 24.68 | 31 |
| 19 | event_19 | 3.64 | 3.49 | 3.86 | 31 |
