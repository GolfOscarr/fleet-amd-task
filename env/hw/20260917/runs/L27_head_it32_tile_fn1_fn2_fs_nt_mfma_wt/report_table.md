# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | 8982.0 |
| time per iteration, P95 (us) |  | 8997.5 |
| time per iteration from event timing, median (us) |  | 8982.0 |
| time per iteration from host wall clock (us) |  | 70516.7 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | 111.3 |
| workers with tasks (of those reporting) |  | 296 of 296 |
| tasks per XCD (placement) |  | 0:88705 1:88704 2:88704 3:88704 4:88704 5:88736 6:88767 7:88704 |
| shader clock from the spin (MHz) |  | - |
| exec cycles per task, busy workers |  | 37831.6 |
| exec us per task (at the spin's SCLK) |  | - |
| dep-wait us per iteration per busy worker |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2102 / 2103 over 53 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Exec time per task by class (worker timing, I1)

| Class | tasks | cycles per task | us per task |
|---|---|---|---|
| attend | 33696 | 15710 | - |
| linear | 5632 | 61197 | - |
| linear_res | 57344 | 23805 | - |
| lnorm | 93478 | 44441 | - |
| merge | 12960 | 40525 | - |
| prep | 864 | 312773 | - |
| rms | 32 | 3533 | - |
| router | 801 | 50847 | - |
| w2silu | 241280 | 30745 | - |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | embed_layer | 199.88 | 165.36 | 218.32 | 30 |
| 2 | linear_norm_layer | 14.94 | 13.59 | 16.43 | 31 |
| 3 | mla_prep_layer | 30.03 | 28.07 | 32.84 | 31 |
| 4 | mla_attend_layer | 150.28 | 147.25 | 154.97 | 31 |
| 5 | mla_merge_uv_layer | 12.90 | 12.14 | 16.28 | 31 |
| 6 | linear_with_residual_layer | 29.26 | 26.11 | 31.40 | 31 |
| 7 | rmsnorm_layer | 13.53 | 12.83 | 14.16 | 31 |
| 8 | gang_linear_silu_layer | 4.64 | 3.89 | 5.33 | 31 |
| 9 | linear_with_residual_layer | 35.50 | 33.69 | 37.74 | 31 |
| 10 | linear_norm_layer | 53.30 | 50.62 | 55.60 | 31 |
| 11 | mla_prep_layer | 19.15 | 18.52 | 20.07 | 31 |
| 12 | mla_attend_layer | 150.80 | 147.99 | 154.18 | 31 |
| 13 | mla_merge_uv_layer | 12.37 | 11.97 | 12.61 | 31 |
| 14 | linear_with_residual_layer | 25.55 | 24.77 | 27.12 | 31 |
| 15 | moe_router_layer | 12.91 | 12.14 | 14.16 | 31 |
| 16 | gang_moe_w13_linear_layer | 28.80 | 26.92 | 29.90 | 31 |
| 17 | gang_moe_w2_silu_linear_layer | 43.09 | 39.07 | 47.46 | 31 |
| 18 | moe_mul_sum_add_layer | 23.85 | 21.96 | 25.46 | 31 |
| 19 | linear_norm_layer | 5.42 | 4.71 | 5.77 | 31 |
| 20 | mla_prep_layer | 19.08 | 18.01 | 20.24 | 31 |
| 21 | mla_attend_layer | 149.71 | 146.67 | 156.59 | 31 |
| 22 | mla_merge_uv_layer | 12.59 | 12.28 | 12.85 | 31 |
| 23 | linear_with_residual_layer | 25.38 | 24.14 | 27.60 | 31 |
| 24 | moe_router_layer | 13.69 | 12.94 | 15.98 | 31 |
| 25 | gang_moe_w13_linear_layer | 28.35 | 26.86 | 29.89 | 31 |
| 26 | gang_moe_w2_silu_linear_layer | 44.35 | 40.72 | 47.59 | 31 |
| 27 | moe_mul_sum_add_layer | 25.44 | 23.13 | 27.64 | 31 |
| 28 | linear_norm_layer | 5.40 | 4.65 | 6.06 | 31 |
| 29 | mla_prep_layer | 19.15 | 18.40 | 20.32 | 31 |
| 30 | mla_attend_layer | 149.13 | 146.03 | 153.42 | 31 |
| 31 | mla_merge_uv_layer | 12.41 | 12.10 | 12.70 | 31 |
| 32 | linear_with_residual_layer | 25.16 | 24.29 | 26.29 | 31 |
| 33 | moe_router_layer | 13.46 | 12.74 | 14.04 | 31 |
| 34 | gang_moe_w13_linear_layer | 27.77 | 26.10 | 31.70 | 31 |
| 35 | gang_moe_w2_silu_linear_layer | 44.25 | 41.10 | 48.16 | 31 |
| 36 | moe_mul_sum_add_layer | 23.93 | 22.76 | 25.46 | 31 |
| 37 | linear_norm_layer | 5.29 | 4.80 | 5.76 | 31 |
| 38 | mla_prep_layer | 19.22 | 18.19 | 20.40 | 31 |
| 39 | mla_attend_layer | 149.71 | 146.78 | 153.19 | 31 |
| 40 | mla_merge_uv_layer | 12.35 | 11.93 | 12.71 | 31 |
| 41 | linear_with_residual_layer | 25.14 | 23.98 | 26.15 | 31 |
| 42 | moe_router_layer | 13.51 | 12.61 | 14.20 | 31 |
| 43 | gang_moe_w13_linear_layer | 27.85 | 26.20 | 33.37 | 31 |
| 44 | gang_moe_w2_silu_linear_layer | 41.80 | 38.85 | 44.68 | 31 |
| 45 | moe_mul_sum_add_layer | 24.10 | 22.53 | 25.52 | 31 |
| 46 | linear_norm_layer | 5.42 | 4.85 | 6.04 | 31 |
| 47 | mla_prep_layer | 18.73 | 18.08 | 19.80 | 31 |
| 48 | mla_attend_layer | 150.30 | 147.00 | 153.84 | 31 |
| 49 | mla_merge_uv_layer | 12.46 | 12.05 | 12.84 | 31 |
| 50 | linear_with_residual_layer | 25.17 | 24.65 | 25.77 | 31 |
| 51 | moe_router_layer | 13.46 | 12.52 | 14.08 | 31 |
| 52 | gang_moe_w13_linear_layer | 28.19 | 26.22 | 33.35 | 31 |
| 53 | gang_moe_w2_silu_linear_layer | 42.54 | 39.31 | 45.04 | 31 |
| 54 | moe_mul_sum_add_layer | 23.67 | 22.45 | 25.24 | 31 |
| 55 | linear_norm_layer | 5.42 | 4.86 | 5.75 | 31 |
| 56 | mla_prep_layer | 18.88 | 17.97 | 19.80 | 31 |
| 57 | mla_attend_layer | 149.85 | 146.82 | 156.90 | 31 |
| 58 | mla_merge_uv_layer | 12.47 | 12.13 | 12.89 | 31 |
| 59 | linear_with_residual_layer | 29.64 | 26.62 | 31.32 | 31 |
| 60 | moe_router_layer | 13.38 | 12.45 | 14.20 | 31 |
| 61 | gang_moe_w13_linear_layer | 27.82 | 26.93 | 33.92 | 31 |
| 62 | gang_moe_w2_silu_linear_layer | 42.67 | 39.47 | 45.09 | 31 |
| 63 | moe_mul_sum_add_layer | 23.56 | 21.90 | 26.16 | 31 |
| 64 | linear_norm_layer | 5.30 | 4.61 | 8.54 | 31 |
| 65 | mla_prep_layer | 19.25 | 18.52 | 20.15 | 31 |
| 66 | mla_attend_layer | 149.48 | 146.47 | 152.87 | 31 |
| 67 | mla_merge_uv_layer | 12.47 | 12.02 | 16.28 | 31 |
| 68 | linear_with_residual_layer | 26.17 | 25.18 | 27.20 | 31 |
| 69 | moe_router_layer | 13.53 | 12.99 | 14.44 | 31 |
| 70 | gang_moe_w13_linear_layer | 28.25 | 26.68 | 32.34 | 31 |
| 71 | gang_moe_w2_silu_linear_layer | 42.04 | 38.66 | 47.29 | 31 |
| 72 | moe_mul_sum_add_layer | 23.86 | 22.47 | 25.50 | 31 |
| 73 | linear_norm_layer | 5.25 | 4.67 | 6.04 | 31 |
| 74 | mla_prep_layer | 19.15 | 18.15 | 20.10 | 31 |
| 75 | mla_attend_layer | 149.83 | 146.82 | 153.84 | 31 |
| 76 | mla_merge_uv_layer | 12.41 | 12.02 | 12.89 | 31 |
| 77 | linear_with_residual_layer | 24.32 | 23.09 | 26.48 | 31 |
| 78 | moe_router_layer | 13.34 | 12.69 | 14.32 | 31 |
| 79 | gang_moe_w13_linear_layer | 26.66 | 25.88 | 28.21 | 31 |
| 80 | gang_moe_w2_silu_linear_layer | 44.37 | 41.21 | 47.31 | 31 |
| 81 | moe_mul_sum_add_layer | 23.79 | 21.84 | 26.44 | 31 |
| 82 | linear_norm_layer | 5.31 | 4.57 | 7.32 | 31 |
| 83 | mla_prep_layer | 18.94 | 18.24 | 19.64 | 31 |
| 84 | mla_attend_layer | 150.35 | 147.35 | 153.84 | 31 |
| 85 | mla_merge_uv_layer | 12.30 | 12.01 | 12.76 | 31 |
| 86 | linear_with_residual_layer | 24.14 | 23.08 | 26.83 | 31 |
| 87 | moe_router_layer | 13.35 | 12.50 | 14.00 | 31 |
| 88 | gang_moe_w13_linear_layer | 27.76 | 26.16 | 29.36 | 31 |
| 89 | gang_moe_w2_silu_linear_layer | 41.11 | 38.64 | 47.15 | 31 |
| 90 | moe_mul_sum_add_layer | 24.52 | 22.28 | 27.40 | 31 |
| 91 | linear_norm_layer | 6.31 | 4.91 | 7.97 | 31 |
| 92 | mla_prep_layer | 18.40 | 17.86 | 18.96 | 31 |
| 93 | mla_attend_layer | 150.08 | 146.97 | 153.43 | 31 |
| 94 | mla_merge_uv_layer | 12.33 | 11.78 | 12.72 | 31 |
| 95 | linear_with_residual_layer | 23.98 | 23.19 | 26.66 | 31 |
| 96 | moe_router_layer | 13.37 | 12.60 | 16.16 | 31 |
| 97 | gang_moe_w13_linear_layer | 28.39 | 27.12 | 29.77 | 31 |
| 98 | gang_moe_w2_silu_linear_layer | 43.15 | 39.52 | 45.33 | 31 |
| 99 | moe_mul_sum_add_layer | 23.42 | 22.16 | 25.10 | 31 |
| 100 | linear_norm_layer | 5.61 | 4.63 | 7.11 | 31 |
| 101 | mla_prep_layer | 19.11 | 18.45 | 19.63 | 31 |
| 102 | mla_attend_layer | 149.25 | 146.17 | 152.71 | 31 |
| 103 | mla_merge_uv_layer | 12.30 | 11.82 | 12.70 | 31 |
| 104 | linear_with_residual_layer | 23.85 | 23.13 | 25.89 | 31 |
| 105 | moe_router_layer | 13.38 | 12.89 | 13.88 | 31 |
| 106 | gang_moe_w13_linear_layer | 27.37 | 26.33 | 30.88 | 31 |
| 107 | gang_moe_w2_silu_linear_layer | 42.98 | 39.92 | 47.52 | 31 |
| 108 | moe_mul_sum_add_layer | 23.55 | 21.71 | 25.53 | 31 |
| 109 | linear_norm_layer | 5.12 | 4.48 | 5.68 | 31 |
| 110 | mla_prep_layer | 18.80 | 18.11 | 19.53 | 31 |
| 111 | mla_attend_layer | 149.82 | 146.86 | 153.18 | 31 |
| 112 | mla_merge_uv_layer | 12.36 | 11.97 | 12.74 | 31 |
| 113 | linear_with_residual_layer | 24.00 | 23.16 | 27.04 | 31 |
| 114 | moe_router_layer | 13.68 | 12.69 | 17.52 | 31 |
| 115 | gang_moe_w13_linear_layer | 26.88 | 25.17 | 28.77 | 31 |
| 116 | gang_moe_w2_silu_linear_layer | 40.85 | 38.31 | 43.03 | 31 |
| 117 | moe_mul_sum_add_layer | 23.85 | 22.28 | 25.84 | 31 |
| 118 | linear_norm_layer | 5.25 | 4.76 | 6.94 | 31 |
| 119 | mla_prep_layer | 19.04 | 18.30 | 19.61 | 31 |
| 120 | mla_attend_layer | 150.11 | 147.04 | 153.77 | 31 |
| 121 | mla_merge_uv_layer | 12.36 | 11.97 | 12.93 | 31 |
| 122 | linear_with_residual_layer | 26.42 | 23.60 | 27.41 | 31 |
| 123 | moe_router_layer | 13.55 | 12.85 | 15.17 | 31 |
| 124 | gang_moe_w13_linear_layer | 27.41 | 25.91 | 28.86 | 31 |
| 125 | gang_moe_w2_silu_linear_layer | 42.67 | 38.48 | 47.09 | 31 |
| 126 | moe_mul_sum_add_layer | 23.33 | 21.68 | 25.10 | 31 |
| 127 | linear_norm_layer | 5.33 | 4.69 | 5.67 | 31 |
| 128 | mla_prep_layer | 18.65 | 18.29 | 19.39 | 31 |
| 129 | mla_attend_layer | 149.22 | 146.00 | 152.79 | 31 |
| 130 | mla_merge_uv_layer | 12.54 | 12.25 | 12.77 | 31 |
| 131 | linear_with_residual_layer | 24.31 | 23.12 | 26.58 | 31 |
| 132 | moe_router_layer | 14.10 | 13.03 | 16.13 | 31 |
| 133 | gang_moe_w13_linear_layer | 29.42 | 27.27 | 30.76 | 31 |
| 134 | gang_moe_w2_silu_linear_layer | 40.67 | 38.80 | 43.43 | 31 |
| 135 | moe_mul_sum_add_layer | 23.66 | 22.00 | 25.59 | 31 |
| 136 | linear_norm_layer | 5.33 | 4.76 | 7.13 | 31 |
| 137 | mla_prep_layer | 18.86 | 18.20 | 19.38 | 31 |
| 138 | mla_attend_layer | 149.06 | 145.89 | 152.40 | 31 |
| 139 | mla_merge_uv_layer | 12.52 | 12.26 | 12.74 | 31 |
| 140 | linear_with_residual_layer | 23.51 | 23.05 | 24.08 | 31 |
| 141 | moe_router_layer | 13.29 | 12.90 | 13.88 | 31 |
| 142 | gang_moe_w13_linear_layer | 28.25 | 26.29 | 29.12 | 31 |
| 143 | gang_moe_w2_silu_linear_layer | 40.48 | 38.43 | 43.14 | 31 |
| 144 | moe_mul_sum_add_layer | 23.43 | 21.96 | 25.58 | 31 |
| 145 | linear_norm_layer | 6.35 | 4.90 | 7.98 | 31 |
| 146 | mla_prep_layer | 19.29 | 18.59 | 19.88 | 31 |
| 147 | mla_attend_layer | 149.61 | 146.39 | 153.11 | 31 |
| 148 | mla_merge_uv_layer | 12.34 | 11.94 | 12.98 | 31 |
| 149 | linear_with_residual_layer | 23.73 | 22.81 | 25.92 | 31 |
| 150 | moe_router_layer | 13.41 | 12.85 | 14.28 | 31 |
| 151 | gang_moe_w13_linear_layer | 27.89 | 25.74 | 28.57 | 31 |
| 152 | gang_moe_w2_silu_linear_layer | 40.71 | 38.60 | 43.71 | 31 |
| 153 | moe_mul_sum_add_layer | 24.47 | 22.60 | 27.64 | 31 |
| 154 | linear_norm_layer | 5.39 | 4.75 | 6.63 | 31 |
| 155 | mla_prep_layer | 18.63 | 18.01 | 19.35 | 31 |
| 156 | mla_attend_layer | 150.20 | 146.76 | 153.44 | 31 |
| 157 | mla_merge_uv_layer | 12.33 | 11.97 | 12.61 | 31 |
| 158 | linear_with_residual_layer | 23.93 | 23.29 | 26.03 | 31 |
| 159 | moe_router_layer | 12.78 | 12.09 | 16.06 | 31 |
| 160 | gang_moe_w13_linear_layer | 27.63 | 26.18 | 29.74 | 31 |
| 161 | gang_moe_w2_silu_linear_layer | 42.52 | 39.31 | 46.72 | 31 |
| 162 | moe_mul_sum_add_layer | 23.93 | 22.40 | 25.73 | 31 |
| 163 | linear_norm_layer | 5.39 | 4.59 | 8.22 | 31 |
| 164 | mla_prep_layer | 18.86 | 18.31 | 19.37 | 31 |
| 165 | mla_attend_layer | 149.27 | 146.27 | 152.91 | 31 |
| 166 | mla_merge_uv_layer | 12.49 | 12.28 | 12.69 | 31 |
| 167 | linear_with_residual_layer | 24.13 | 23.20 | 26.21 | 31 |
| 168 | moe_router_layer | 13.33 | 12.64 | 14.54 | 31 |
| 169 | gang_moe_w13_linear_layer | 27.55 | 26.37 | 29.35 | 31 |
| 170 | gang_moe_w2_silu_linear_layer | 42.08 | 39.15 | 45.97 | 31 |
| 171 | moe_mul_sum_add_layer | 23.96 | 21.90 | 27.04 | 31 |
| 172 | linear_norm_layer | 5.58 | 4.79 | 7.24 | 31 |
| 173 | mla_prep_layer | 18.88 | 18.36 | 19.54 | 31 |
| 174 | mla_attend_layer | 149.14 | 145.87 | 152.91 | 31 |
| 175 | mla_merge_uv_layer | 12.36 | 12.02 | 12.66 | 31 |
| 176 | linear_with_residual_layer | 24.35 | 23.64 | 27.29 | 31 |
| 177 | moe_router_layer | 13.94 | 12.99 | 17.72 | 31 |
| 178 | gang_moe_w13_linear_layer | 28.14 | 26.02 | 30.38 | 31 |
| 179 | gang_moe_w2_silu_linear_layer | 40.85 | 38.82 | 44.26 | 31 |
| 180 | moe_mul_sum_add_layer | 24.32 | 22.66 | 25.42 | 31 |
| 181 | linear_norm_layer | 5.23 | 4.84 | 6.37 | 31 |
| 182 | mla_prep_layer | 19.31 | 18.36 | 19.84 | 31 |
| 183 | mla_attend_layer | 149.60 | 146.37 | 153.08 | 31 |
| 184 | mla_merge_uv_layer | 12.36 | 12.02 | 12.74 | 31 |
| 185 | linear_with_residual_layer | 26.30 | 23.59 | 27.32 | 31 |
| 186 | moe_router_layer | 13.37 | 12.88 | 15.97 | 31 |
| 187 | gang_moe_w13_linear_layer | 27.53 | 26.05 | 28.69 | 31 |
| 188 | gang_moe_w2_silu_linear_layer | 42.97 | 39.26 | 48.00 | 31 |
| 189 | moe_mul_sum_add_layer | 24.26 | 22.23 | 26.36 | 31 |
| 190 | linear_norm_layer | 5.44 | 4.71 | 5.85 | 31 |
| 191 | mla_prep_layer | 18.44 | 18.11 | 19.04 | 31 |
| 192 | mla_attend_layer | 150.37 | 147.36 | 153.99 | 31 |
| 193 | mla_merge_uv_layer | 12.30 | 12.05 | 12.57 | 31 |
| 194 | linear_with_residual_layer | 24.90 | 23.44 | 26.54 | 31 |
| 195 | moe_router_layer | 13.81 | 12.98 | 15.74 | 31 |
| 196 | gang_moe_w13_linear_layer | 27.28 | 25.95 | 29.69 | 31 |
| 197 | gang_moe_w2_silu_linear_layer | 41.53 | 38.80 | 44.54 | 31 |
| 198 | moe_mul_sum_add_layer | 23.41 | 21.64 | 24.60 | 31 |
| 199 | linear_norm_layer | 5.23 | 4.57 | 5.87 | 31 |
| 200 | mla_prep_layer | 18.66 | 17.91 | 19.35 | 31 |
| 201 | mla_attend_layer | 149.48 | 146.10 | 153.52 | 31 |
| 202 | mla_merge_uv_layer | 12.32 | 11.97 | 12.65 | 31 |
| 203 | linear_with_residual_layer | 23.56 | 23.03 | 24.20 | 31 |
| 204 | moe_router_layer | 13.38 | 12.86 | 14.08 | 31 |
| 205 | gang_moe_w13_linear_layer | 26.91 | 26.08 | 28.45 | 31 |
| 206 | gang_moe_w2_silu_linear_layer | 41.94 | 39.51 | 43.97 | 31 |
| 207 | moe_mul_sum_add_layer | 24.06 | 22.32 | 26.28 | 31 |
| 208 | linear_norm_layer | 5.34 | 4.74 | 7.13 | 31 |
| 209 | mla_prep_layer | 19.09 | 18.52 | 19.61 | 31 |
| 210 | mla_attend_layer | 149.03 | 145.83 | 152.83 | 31 |
| 211 | mla_merge_uv_layer | 12.46 | 12.22 | 12.70 | 31 |
| 212 | linear_with_residual_layer | 23.86 | 23.00 | 26.35 | 31 |
| 213 | moe_router_layer | 13.28 | 12.54 | 14.12 | 31 |
| 214 | gang_moe_w13_linear_layer | 28.29 | 26.85 | 30.97 | 31 |
| 215 | gang_moe_w2_silu_linear_layer | 41.55 | 38.93 | 45.16 | 31 |
| 216 | moe_mul_sum_add_layer | 24.82 | 22.40 | 27.56 | 31 |
| 217 | linear_norm_layer | 5.29 | 4.73 | 7.26 | 31 |
| 218 | mla_prep_layer | 19.23 | 18.48 | 19.85 | 31 |
| 219 | mla_attend_layer | 149.84 | 146.95 | 153.62 | 31 |
| 220 | mla_merge_uv_layer | 12.55 | 12.09 | 15.88 | 31 |
| 221 | linear_with_residual_layer | 23.77 | 23.02 | 26.79 | 31 |
| 222 | moe_router_layer | 13.45 | 12.81 | 16.22 | 31 |
| 223 | gang_moe_w13_linear_layer | 27.21 | 25.45 | 29.12 | 31 |
| 224 | gang_moe_w2_silu_linear_layer | 41.66 | 38.31 | 45.22 | 31 |
| 225 | moe_mul_sum_add_layer | 23.63 | 22.28 | 25.61 | 31 |
| 226 | linear_norm_layer | 5.41 | 4.52 | 6.66 | 31 |
| 227 | mla_prep_layer | 18.80 | 18.28 | 19.44 | 31 |
| 228 | mla_attend_layer | 150.10 | 146.80 | 153.92 | 31 |
| 229 | mla_merge_uv_layer | 12.50 | 11.93 | 12.84 | 31 |
| 230 | linear_with_residual_layer | 23.55 | 23.04 | 23.97 | 31 |
| 231 | moe_router_layer | 13.37 | 12.83 | 14.61 | 31 |
| 232 | gang_moe_w13_linear_layer | 26.91 | 25.39 | 28.10 | 31 |
| 233 | gang_moe_w2_silu_linear_layer | 41.99 | 39.30 | 44.18 | 31 |
| 234 | moe_mul_sum_add_layer | 23.45 | 22.49 | 24.88 | 31 |
| 235 | linear_norm_layer | 5.11 | 4.56 | 5.58 | 31 |
| 236 | mla_prep_layer | 18.77 | 18.04 | 19.55 | 31 |
| 237 | mla_attend_layer | 149.78 | 146.35 | 153.47 | 31 |
| 238 | mla_merge_uv_layer | 12.27 | 11.93 | 12.64 | 31 |
| 239 | linear_with_residual_layer | 23.69 | 23.11 | 25.15 | 31 |
| 240 | moe_router_layer | 13.76 | 12.91 | 16.52 | 31 |
| 241 | gang_moe_w13_linear_layer | 27.24 | 26.89 | 28.53 | 31 |
| 242 | gang_moe_w2_silu_linear_layer | 42.38 | 39.89 | 45.12 | 31 |
| 243 | moe_mul_sum_add_layer | 23.56 | 22.18 | 25.40 | 31 |
| 244 | linear_norm_layer | 6.00 | 4.83 | 9.36 | 31 |
| 245 | argmax_partial_layer | 0.18 | 0.00 | 0.61 | 31 |
| 246 | argmax_reduce_layer | 3.06 | 0.00 | 88.88 | 31 |
| 247 | event_247 | 0.37 | 0.00 | 1.96 | 31 |
| 248 | event_248 | 0.37 | 0.00 | 2.67 | 31 |
| 249 | event_249 | 0.34 | 0.00 | 1.76 | 31 |
| 250 | event_250 | 0.23 | 0.00 | 0.83 | 31 |
| 251 | event_251 | 3.06 | 0.00 | 88.15 | 31 |
| 252 | event_252 | 0.35 | 0.00 | 2.14 | 31 |
| 253 | event_253 | 2.99 | 0.00 | 87.82 | 31 |
| 254 | event_254 | 0.27 | 0.00 | 1.49 | 31 |
| 255 | event_255 | 0.24 | 0.00 | 1.40 | 31 |
| 256 | event_256 | 0.30 | 0.00 | 1.96 | 31 |
| 257 | event_257 | 0.27 | 0.00 | 1.30 | 31 |
| 258 | event_258 | 0.28 | 0.00 | 1.30 | 31 |
| 259 | event_259 | 0.17 | 0.00 | 0.79 | 31 |
| 260 | event_260 | 0.18 | 0.00 | 1.58 | 31 |
| 261 | event_261 | 0.14 | 0.00 | 0.99 | 31 |
| 262 | event_262 | 0.21 | 0.00 | 0.88 | 31 |
| 263 | event_263 | 3.18 | 0.00 | 87.36 | 31 |
| 264 | event_264 | 0.39 | 0.00 | 2.90 | 31 |
| 265 | event_265 | 0.19 | 0.00 | 1.53 | 31 |
| 266 | event_266 | 0.18 | 0.00 | 0.72 | 31 |
| 267 | event_267 | 0.13 | 0.00 | 0.75 | 31 |
| 268 | event_268 | 0.12 | 0.00 | 0.56 | 31 |
| 269 | event_269 | 0.32 | 0.00 | 2.41 | 31 |
| 270 | event_270 | 0.22 | 0.00 | 0.92 | 31 |
| 271 | event_271 | 0.24 | 0.00 | 1.38 | 31 |
| 272 | event_272 | 0.30 | 0.00 | 3.04 | 31 |
| 273 | event_273 | 0.22 | 0.00 | 0.88 | 31 |
| 274 | event_274 | 0.19 | 0.00 | 0.89 | 31 |
| 275 | event_275 | 8.73 | 0.00 | 88.14 | 31 |
| 276 | event_276 | 11.50 | 0.00 | 89.64 | 31 |
| 277 | event_277 | 14.19 | 0.00 | 88.14 | 31 |
| 278 | event_278 | 17.10 | 0.00 | 88.32 | 31 |
| 279 | event_279 | 16.91 | 0.00 | 87.04 | 31 |
| 280 | event_280 | 0.28 | 0.00 | 1.59 | 31 |
| 281 | event_281 | 8.69 | 0.00 | 89.16 | 31 |
| 282 | event_282 | 2.82 | 0.00 | 39.24 | 31 |
| 283 | event_283 | 6.63 | 0.00 | 40.62 | 31 |
| 284 | event_284 | 0.57 | 0.01 | 3.00 | 31 |
| 285 | event_285 | 3.34 | 0.00 | 42.11 | 31 |
| 286 | event_286 | 6.70 | 0.00 | 42.01 | 31 |
| 287 | event_287 | 5.33 | 0.00 | 39.94 | 31 |
| 288 | event_288 | 0.33 | 0.00 | 1.96 | 31 |
| 289 | event_289 | 5.45 | 0.00 | 41.12 | 31 |
| 290 | event_290 | 6.58 | 0.00 | 40.46 | 31 |
| 291 | event_291 | 4.09 | 0.00 | 40.27 | 31 |
| 292 | event_292 | 0.43 | 0.00 | 1.88 | 31 |
| 293 | event_293 | 0.92 | 0.00 | 3.96 | 31 |
| 294 | event_294 | 1.68 | 0.00 | 39.28 | 31 |
| 295 | event_295 | 6.23 | 5.95 | 6.56 | 31 |
| 296 | event_296 | 5.43 | 5.20 | 5.73 | 31 |
