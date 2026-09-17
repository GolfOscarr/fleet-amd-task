# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | 1710.0 |
| time per iteration, P95 (us) |  | 4742.6 |
| time per iteration from event timing, median (us) |  | 850.4 |
| time per iteration from host wall clock (us) |  | 64043.4 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | 584.8 |
| workers with tasks (of those reporting) |  | 296 of 296 |
| tasks per XCD (placement) |  | 0:5163 1:5158 2:5166 3:5121 4:5120 5:5152 6:5152 7:5152 |
| shader clock from the spin (MHz) |  | - |
| exec cycles per task, busy workers |  | 42706.1 |
| exec us per task (at the spin's SCLK) |  | - |
| dep-wait us per iteration per busy worker |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2103 / 2104 over 39 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Exec time per task by class (worker timing, I1)

| Class | tasks | cycles per task | us per task |
|---|---|---|---|
| attend | 2240 | 76005 | - |
| linear | 5632 | 55090 | - |
| linear_res | 6144 | 46697 | - |
| lnorm | 5646 | 30109 | - |
| merge | 896 | 35790 | - |
| prep | 64 | 301063 | - |
| rms | 64 | 3371 | - |
| router | 32 | 36498 | - |
| silu | 256 | 2352 | - |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | embed_layer | 138.55 | 8.36 | 154.60 | 30 |
| 2 | linear_norm_layer | 12.48 | 11.50 | 14.06 | 31 |
| 3 | mla_prep_layer | 31.83 | 29.56 | 33.26 | 31 |
| 4 | mla_attend_layer | 144.14 | 140.84 | 148.53 | 31 |
| 5 | mla_merge_uv_layer | 47.44 | 47.15 | 47.88 | 31 |
| 6 | linear_with_residual_layer | 24.95 | 22.59 | 26.05 | 31 |
| 7 | rmsnorm_layer | 13.48 | 12.96 | 13.92 | 31 |
| 8 | gang_linear_silu_layer | 4.43 | 3.87 | 5.48 | 31 |
| 9 | linear_with_residual_layer | 33.37 | 32.33 | 34.16 | 31 |
| 10 | linear_norm_layer | 51.64 | 50.41 | 52.92 | 31 |
| 11 | mla_prep_layer | 18.04 | 17.40 | 18.68 | 31 |
| 12 | mla_attend_layer | 143.71 | 139.99 | 148.60 | 31 |
| 13 | mla_merge_uv_layer | 47.41 | 47.13 | 47.65 | 31 |
| 14 | linear_with_residual_layer | 22.12 | 21.80 | 22.37 | 31 |
| 15 | rmsnorm_layer | 12.85 | 12.39 | 13.64 | 31 |
| 16 | moe_router_layer | 3.77 | 3.56 | 4.01 | 31 |
| 17 | gang_moe_w13_linear_layer | 21.78 | 20.16 | 22.24 | 31 |
| 18 | moe_silu_mul_layer | 39.24 | 38.16 | 41.64 | 31 |
| 19 | gang_moe_w2_linear_layer | 5.15 | 3.88 | 6.43 | 31 |
| 20 | moe_mul_sum_add_layer | 21.23 | 19.65 | 22.96 | 31 |
| 21 | event_21 | 3.33 | 3.18 | 3.69 | 31 |
