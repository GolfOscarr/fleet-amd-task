# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | 1709.0 |
| time per iteration, P95 (us) |  | 11875.2 |
| time per iteration from event timing, median (us) |  | 849.0 |
| time per iteration from host wall clock (us) |  | 60733.0 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | 585.1 |
| workers with tasks (of those reporting) |  | 296 of 296 |
| tasks per XCD (placement) |  | 0:5121 1:5122 2:5098 3:5096 4:5095 5:5124 6:5120 7:5120 |
| shader clock from the spin (MHz) |  | - |
| exec cycles per task, busy workers |  | 42748.4 |
| exec us per task (at the spin's SCLK) |  | - |
| dep-wait us per iteration per busy worker |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2102 / 2103 over 39 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Exec time per task by class (worker timing, I1)

| Class | tasks | cycles per task | us per task |
|---|---|---|---|
| attend | 2304 | 76540 | - |
| linear | 5632 | 49479 | - |
| linear_res | 6144 | 47127 | - |
| lnorm | 5462 | 30407 | - |
| merge | 896 | 35734 | - |
| prep | 64 | 302680 | - |
| rms | 32 | 3554 | - |
| router | 32 | 45315 | - |
| w2silu | 8416 | 30577 | - |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | embed_layer | 130.77 | 8.24 | 161.24 | 30 |
| 2 | linear_norm_layer | 11.49 | 10.72 | 14.51 | 31 |
| 3 | mla_prep_layer | 31.97 | 29.49 | 34.89 | 31 |
| 4 | mla_attend_layer | 141.34 | 138.02 | 152.73 | 31 |
| 5 | mla_merge_uv_layer | 47.37 | 47.10 | 47.66 | 31 |
| 6 | linear_with_residual_layer | 26.02 | 23.43 | 27.28 | 31 |
| 7 | rmsnorm_layer | 13.58 | 13.16 | 14.04 | 31 |
| 8 | gang_linear_silu_layer | 4.32 | 3.82 | 5.18 | 31 |
| 9 | linear_with_residual_layer | 31.67 | 30.15 | 32.46 | 31 |
| 10 | linear_norm_layer | 51.69 | 50.24 | 53.32 | 31 |
| 11 | mla_prep_layer | 18.32 | 17.61 | 18.78 | 31 |
| 12 | mla_attend_layer | 148.26 | 145.54 | 151.48 | 31 |
| 13 | mla_merge_uv_layer | 47.24 | 46.98 | 47.58 | 31 |
| 14 | linear_with_residual_layer | 22.52 | 21.96 | 23.44 | 31 |
| 15 | moe_router_layer | 12.99 | 12.60 | 13.84 | 31 |
| 16 | gang_moe_w13_linear_layer | 25.59 | 23.16 | 26.70 | 31 |
| 17 | gang_moe_w2_silu_linear_layer | 41.31 | 38.62 | 43.94 | 31 |
| 18 | moe_mul_sum_add_layer | 22.65 | 21.20 | 24.71 | 31 |
| 19 | event_19 | 3.54 | 3.40 | 4.64 | 31 |
