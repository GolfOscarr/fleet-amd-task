# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | 1015.0 |
| time per iteration, P95 (us) |  | 1970.1 |
| time per iteration from event timing, median (us) |  | 1012.6 |
| time per iteration from host wall clock (us) |  | 61554.9 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | 985.2 |
| workers with tasks (of those reporting) |  | 296 of 296 |
| tasks per XCD (placement) |  | 0:7042 1:7040 2:7008 3:7008 4:7008 5:7040 6:7041 7:7069 |
| shader clock from the spin (MHz) |  | - |
| exec cycles per task, busy workers |  | 37226.0 |
| exec us per task (at the spin's SCLK) |  | - |
| dep-wait us per iteration per busy worker |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2103 / 2104 over 38 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Exec time per task by class (worker timing, I1)

| Class | tasks | cycles per task | us per task |
|---|---|---|---|
| attend | 2432 | 95266 | - |
| linear | 5632 | 57467 | - |
| linear_res | 6144 | 47120 | - |
| lnorm | 5890 | 31891 | - |
| merge | 960 | 45154 | - |
| prefetch | 14741 | 16687 | - |
| prep | 64 | 400148 | - |
| rms | 32 | 3925 | - |
| router | 32 | 56856 | - |
| w2silu | 9088 | 27030 | - |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | embed_layer | 157.46 | 9.96 | 181.52 | 30 |
| 2 | linear_norm_layer | 14.43 | 13.59 | 15.51 | 31 |
| 3 | mla_prep_layer | 31.42 | 29.88 | 33.67 | 31 |
| 4 | mla_attend_layer | 192.30 | 187.64 | 209.78 | 31 |
| 5 | mla_merge_uv_layer | 64.59 | 64.04 | 65.52 | 31 |
| 6 | linear_with_residual_layer | 27.86 | 25.70 | 29.48 | 31 |
| 7 | rmsnorm_layer | 13.82 | 13.00 | 14.60 | 31 |
| 8 | gang_linear_silu_layer | 4.54 | 4.23 | 4.87 | 31 |
| 9 | linear_with_residual_layer | 33.86 | 33.01 | 36.68 | 31 |
| 10 | linear_norm_layer | 52.44 | 50.83 | 54.16 | 31 |
| 11 | mla_prep_layer | 18.75 | 17.65 | 19.79 | 31 |
| 12 | mla_attend_layer | 193.02 | 189.58 | 210.48 | 31 |
| 13 | mla_merge_uv_layer | 64.82 | 64.20 | 65.31 | 31 |
| 14 | linear_with_residual_layer | 25.43 | 24.65 | 26.13 | 31 |
| 15 | moe_router_layer | 12.86 | 12.51 | 13.40 | 31 |
| 16 | gang_moe_w13_linear_layer | 29.86 | 28.53 | 31.96 | 31 |
| 17 | gang_moe_w2_silu_linear_layer | 42.53 | 39.95 | 45.39 | 31 |
| 18 | moe_mul_sum_add_layer | 27.74 | 26.01 | 29.40 | 31 |
| 19 | event_19 | 3.64 | 3.52 | 3.77 | 31 |
