# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | 12487.0 |
| time per iteration, P95 (us) |  | 12524.5 |
| time per iteration from event timing, median (us) |  | 12486.7 |
| time per iteration from host wall clock (us) |  | 82440.4 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | 80.1 |
| workers with tasks (of those reporting) |  | 296 of 296 |
| tasks per XCD (placement) |  | 0:132225 1:132224 2:132224 3:132224 4:132224 5:132256 6:132287 7:132224 |
| shader clock from the spin (MHz) |  | - |
| exec cycles per task, busy workers |  | 34922.0 |
| exec us per task (at the spin's SCLK) |  | - |
| dep-wait us per iteration per busy worker |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2103 / 2104 over 53 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Exec time per task by class (worker timing, I1)

| Class | tasks | cycles per task | us per task |
|---|---|---|---|
| attend | 31968 | 92203 | - |
| linear | 5632 | 66256 | - |
| linear_res | 57344 | 24821 | - |
| lnorm | 86753 | 50675 | - |
| merge | 12960 | 44928 | - |
| prefetch | 315194 | 19819 | - |
| prep | 780 | 437363 | - |
| rms | 32 | 4363 | - |
| router | 770 | 58488 | - |
| w2silu | 223808 | 26403 | - |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | embed_layer | 213.48 | 191.40 | 230.12 | 30 |
| 2 | linear_norm_layer | 16.25 | 15.30 | 17.02 | 31 |
| 3 | mla_prep_layer | 31.54 | 30.40 | 32.79 | 31 |
| 4 | mla_attend_layer | 212.27 | 208.22 | 215.86 | 31 |
| 5 | mla_merge_uv_layer | 64.65 | 61.74 | 65.54 | 31 |
| 6 | linear_with_residual_layer | 29.38 | 25.73 | 31.20 | 31 |
| 7 | rmsnorm_layer | 13.60 | 12.85 | 14.76 | 31 |
| 8 | gang_linear_silu_layer | 5.16 | 4.12 | 5.76 | 31 |
| 9 | linear_with_residual_layer | 36.60 | 35.24 | 37.96 | 31 |
| 10 | linear_norm_layer | 53.73 | 51.38 | 56.36 | 31 |
| 11 | mla_prep_layer | 19.56 | 18.44 | 21.16 | 31 |
| 12 | mla_attend_layer | 211.84 | 208.50 | 216.02 | 31 |
| 13 | mla_merge_uv_layer | 64.13 | 60.46 | 65.44 | 31 |
| 14 | linear_with_residual_layer | 26.49 | 25.20 | 27.80 | 31 |
| 15 | moe_router_layer | 13.21 | 12.52 | 14.12 | 31 |
| 16 | gang_moe_w13_linear_layer | 33.00 | 30.84 | 36.71 | 31 |
| 17 | gang_moe_w2_silu_linear_layer | 43.99 | 42.25 | 46.45 | 31 |
| 18 | moe_mul_sum_add_layer | 29.70 | 26.40 | 31.86 | 31 |
| 19 | linear_norm_layer | 5.70 | 4.94 | 6.54 | 31 |
| 20 | mla_prep_layer | 19.59 | 18.50 | 21.22 | 31 |
| 21 | mla_attend_layer | 211.22 | 207.28 | 214.98 | 31 |
| 22 | mla_merge_uv_layer | 64.03 | 59.38 | 65.08 | 31 |
| 23 | linear_with_residual_layer | 26.47 | 25.18 | 27.64 | 31 |
| 24 | moe_router_layer | 15.12 | 12.41 | 18.12 | 31 |
| 25 | gang_moe_w13_linear_layer | 33.18 | 30.78 | 37.04 | 31 |
| 26 | gang_moe_w2_silu_linear_layer | 44.34 | 42.33 | 48.26 | 31 |
| 27 | moe_mul_sum_add_layer | 29.69 | 27.80 | 32.36 | 31 |
| 28 | linear_norm_layer | 6.00 | 4.86 | 6.72 | 31 |
| 29 | mla_prep_layer | 19.18 | 18.36 | 20.98 | 31 |
| 30 | mla_attend_layer | 210.91 | 206.41 | 214.93 | 31 |
| 31 | mla_merge_uv_layer | 64.62 | 63.61 | 65.05 | 31 |
| 32 | linear_with_residual_layer | 26.51 | 25.28 | 27.48 | 31 |
| 33 | moe_router_layer | 13.14 | 12.30 | 15.20 | 31 |
| 34 | gang_moe_w13_linear_layer | 32.09 | 30.52 | 33.10 | 31 |
| 35 | gang_moe_w2_silu_linear_layer | 45.80 | 42.66 | 49.96 | 31 |
| 36 | moe_mul_sum_add_layer | 29.67 | 27.46 | 31.72 | 31 |
| 37 | linear_norm_layer | 5.77 | 5.09 | 6.64 | 31 |
| 38 | mla_prep_layer | 19.55 | 18.20 | 21.08 | 31 |
| 39 | mla_attend_layer | 211.55 | 208.62 | 214.30 | 31 |
| 40 | mla_merge_uv_layer | 64.60 | 61.88 | 65.54 | 31 |
| 41 | linear_with_residual_layer | 26.39 | 25.26 | 27.19 | 31 |
| 42 | moe_router_layer | 13.60 | 12.34 | 14.60 | 31 |
| 43 | gang_moe_w13_linear_layer | 32.22 | 30.24 | 33.26 | 31 |
| 44 | gang_moe_w2_silu_linear_layer | 43.64 | 41.80 | 46.68 | 31 |
| 45 | moe_mul_sum_add_layer | 29.89 | 27.72 | 32.24 | 31 |
| 46 | linear_norm_layer | 6.07 | 5.04 | 6.56 | 31 |
| 47 | mla_prep_layer | 18.77 | 18.00 | 20.20 | 31 |
| 48 | mla_attend_layer | 211.47 | 208.64 | 215.62 | 31 |
| 49 | mla_merge_uv_layer | 64.39 | 62.80 | 65.18 | 31 |
| 50 | linear_with_residual_layer | 27.71 | 25.14 | 30.80 | 31 |
| 51 | moe_router_layer | 13.86 | 12.84 | 17.28 | 31 |
| 52 | gang_moe_w13_linear_layer | 33.06 | 32.00 | 36.27 | 31 |
| 53 | gang_moe_w2_silu_linear_layer | 44.29 | 42.05 | 46.88 | 31 |
| 54 | moe_mul_sum_add_layer | 30.82 | 26.81 | 35.88 | 31 |
| 55 | linear_norm_layer | 5.73 | 4.82 | 6.47 | 31 |
| 56 | mla_prep_layer | 19.56 | 18.73 | 20.61 | 31 |
| 57 | mla_attend_layer | 210.95 | 207.50 | 214.18 | 31 |
| 58 | mla_merge_uv_layer | 64.22 | 60.98 | 65.16 | 31 |
| 59 | linear_with_residual_layer | 26.41 | 25.00 | 27.23 | 31 |
| 60 | moe_router_layer | 13.45 | 12.74 | 14.50 | 31 |
| 61 | gang_moe_w13_linear_layer | 33.46 | 30.80 | 39.58 | 31 |
| 62 | gang_moe_w2_silu_linear_layer | 42.54 | 40.74 | 44.46 | 31 |
| 63 | moe_mul_sum_add_layer | 29.09 | 27.12 | 31.52 | 31 |
| 64 | linear_norm_layer | 5.72 | 4.66 | 6.45 | 31 |
| 65 | mla_prep_layer | 19.36 | 18.40 | 21.16 | 31 |
| 66 | mla_attend_layer | 210.83 | 207.29 | 214.01 | 31 |
| 67 | mla_merge_uv_layer | 64.47 | 63.44 | 64.93 | 31 |
| 68 | linear_with_residual_layer | 26.50 | 25.36 | 27.48 | 31 |
| 69 | moe_router_layer | 13.50 | 12.80 | 15.00 | 31 |
| 70 | gang_moe_w13_linear_layer | 31.37 | 30.36 | 33.04 | 31 |
| 71 | gang_moe_w2_silu_linear_layer | 45.88 | 42.36 | 47.79 | 31 |
| 72 | moe_mul_sum_add_layer | 30.90 | 27.92 | 32.76 | 31 |
| 73 | linear_norm_layer | 5.50 | 4.65 | 6.36 | 31 |
| 74 | mla_prep_layer | 19.49 | 18.38 | 20.98 | 31 |
| 75 | mla_attend_layer | 211.31 | 207.04 | 214.22 | 31 |
| 76 | mla_merge_uv_layer | 64.83 | 63.62 | 65.40 | 31 |
| 77 | linear_with_residual_layer | 26.43 | 25.26 | 28.68 | 31 |
| 78 | moe_router_layer | 13.55 | 12.58 | 14.36 | 31 |
| 79 | gang_moe_w13_linear_layer | 30.46 | 28.57 | 34.34 | 31 |
| 80 | gang_moe_w2_silu_linear_layer | 46.32 | 43.46 | 51.72 | 31 |
| 81 | moe_mul_sum_add_layer | 32.38 | 29.21 | 34.98 | 31 |
| 82 | linear_norm_layer | 5.59 | 4.62 | 6.26 | 31 |
| 83 | mla_prep_layer | 19.48 | 18.22 | 20.64 | 31 |
| 84 | mla_attend_layer | 211.28 | 207.62 | 214.89 | 31 |
| 85 | mla_merge_uv_layer | 64.62 | 63.55 | 65.12 | 31 |
| 86 | linear_with_residual_layer | 26.63 | 25.33 | 27.32 | 31 |
| 87 | moe_router_layer | 13.30 | 12.52 | 14.84 | 31 |
| 88 | gang_moe_w13_linear_layer | 31.15 | 29.67 | 35.19 | 31 |
| 89 | gang_moe_w2_silu_linear_layer | 48.13 | 44.76 | 53.65 | 31 |
| 90 | moe_mul_sum_add_layer | 29.49 | 27.05 | 32.44 | 31 |
| 91 | linear_norm_layer | 5.67 | 4.84 | 6.40 | 31 |
| 92 | mla_prep_layer | 19.35 | 18.30 | 20.60 | 31 |
| 93 | mla_attend_layer | 210.57 | 207.98 | 214.21 | 31 |
| 94 | mla_merge_uv_layer | 64.13 | 57.64 | 65.08 | 31 |
| 95 | linear_with_residual_layer | 27.17 | 25.27 | 30.56 | 31 |
| 96 | moe_router_layer | 13.82 | 12.62 | 18.40 | 31 |
| 97 | gang_moe_w13_linear_layer | 32.08 | 30.48 | 37.34 | 31 |
| 98 | gang_moe_w2_silu_linear_layer | 44.72 | 41.89 | 48.10 | 31 |
| 99 | moe_mul_sum_add_layer | 31.00 | 27.88 | 32.50 | 31 |
| 100 | linear_norm_layer | 5.87 | 4.78 | 6.48 | 31 |
| 101 | mla_prep_layer | 19.33 | 18.48 | 20.76 | 31 |
| 102 | mla_attend_layer | 210.63 | 207.25 | 214.25 | 31 |
| 103 | mla_merge_uv_layer | 64.27 | 61.08 | 65.17 | 31 |
| 104 | linear_with_residual_layer | 26.34 | 24.97 | 27.27 | 31 |
| 105 | moe_router_layer | 13.79 | 12.70 | 16.32 | 31 |
| 106 | gang_moe_w13_linear_layer | 31.93 | 30.34 | 33.30 | 31 |
| 107 | gang_moe_w2_silu_linear_layer | 46.78 | 43.35 | 51.70 | 31 |
| 108 | moe_mul_sum_add_layer | 30.00 | 27.36 | 32.84 | 31 |
| 109 | linear_norm_layer | 5.72 | 4.84 | 6.54 | 31 |
| 110 | mla_prep_layer | 19.22 | 18.48 | 20.40 | 31 |
| 111 | mla_attend_layer | 211.09 | 207.62 | 214.80 | 31 |
| 112 | mla_merge_uv_layer | 64.32 | 61.22 | 65.32 | 31 |
| 113 | linear_with_residual_layer | 26.75 | 25.72 | 27.52 | 31 |
| 114 | moe_router_layer | 13.88 | 12.38 | 17.68 | 31 |
| 115 | gang_moe_w13_linear_layer | 31.06 | 28.96 | 32.18 | 31 |
| 116 | gang_moe_w2_silu_linear_layer | 43.94 | 41.74 | 47.52 | 31 |
| 117 | moe_mul_sum_add_layer | 29.93 | 26.16 | 33.28 | 31 |
| 118 | linear_norm_layer | 5.76 | 4.96 | 6.52 | 31 |
| 119 | mla_prep_layer | 19.23 | 18.16 | 20.36 | 31 |
| 120 | mla_attend_layer | 211.21 | 207.24 | 215.40 | 31 |
| 121 | mla_merge_uv_layer | 64.39 | 61.31 | 65.24 | 31 |
| 122 | linear_with_residual_layer | 26.96 | 25.53 | 30.60 | 31 |
| 123 | moe_router_layer | 13.90 | 12.57 | 16.68 | 31 |
| 124 | gang_moe_w13_linear_layer | 30.37 | 28.91 | 32.12 | 31 |
| 125 | gang_moe_w2_silu_linear_layer | 45.73 | 43.28 | 51.53 | 31 |
| 126 | moe_mul_sum_add_layer | 29.97 | 28.34 | 32.50 | 31 |
| 127 | linear_norm_layer | 5.58 | 4.62 | 6.46 | 31 |
| 128 | mla_prep_layer | 19.26 | 18.30 | 20.36 | 31 |
| 129 | mla_attend_layer | 210.36 | 207.24 | 214.64 | 31 |
| 130 | mla_merge_uv_layer | 64.33 | 61.47 | 65.16 | 31 |
| 131 | linear_with_residual_layer | 26.42 | 25.48 | 27.48 | 31 |
| 132 | moe_router_layer | 13.70 | 12.74 | 14.78 | 31 |
| 133 | gang_moe_w13_linear_layer | 33.04 | 31.24 | 33.86 | 31 |
| 134 | gang_moe_w2_silu_linear_layer | 43.85 | 41.54 | 46.37 | 31 |
| 135 | moe_mul_sum_add_layer | 29.52 | 28.23 | 31.94 | 31 |
| 136 | linear_norm_layer | 5.44 | 4.80 | 6.40 | 31 |
| 137 | mla_prep_layer | 19.74 | 18.58 | 20.52 | 31 |
| 138 | mla_attend_layer | 210.50 | 206.35 | 213.83 | 31 |
| 139 | mla_merge_uv_layer | 64.44 | 61.44 | 65.09 | 31 |
| 140 | linear_with_residual_layer | 26.45 | 25.29 | 28.55 | 31 |
| 141 | moe_router_layer | 13.50 | 12.50 | 14.92 | 31 |
| 142 | gang_moe_w13_linear_layer | 31.66 | 30.08 | 32.69 | 31 |
| 143 | gang_moe_w2_silu_linear_layer | 44.64 | 42.34 | 49.84 | 31 |
| 144 | moe_mul_sum_add_layer | 30.34 | 25.58 | 32.96 | 31 |
| 145 | linear_norm_layer | 6.30 | 4.91 | 6.64 | 31 |
| 146 | mla_prep_layer | 18.56 | 17.87 | 20.08 | 31 |
| 147 | mla_attend_layer | 211.08 | 208.38 | 214.15 | 31 |
| 148 | mla_merge_uv_layer | 64.69 | 60.97 | 65.46 | 31 |
| 149 | linear_with_residual_layer | 26.87 | 25.48 | 27.85 | 31 |
| 150 | moe_router_layer | 13.60 | 12.68 | 14.76 | 31 |
| 151 | gang_moe_w13_linear_layer | 30.98 | 29.42 | 32.56 | 31 |
| 152 | gang_moe_w2_silu_linear_layer | 44.56 | 42.12 | 46.70 | 31 |
| 153 | moe_mul_sum_add_layer | 29.73 | 28.33 | 31.58 | 31 |
| 154 | linear_norm_layer | 5.54 | 4.86 | 6.58 | 31 |
| 155 | mla_prep_layer | 19.29 | 18.32 | 20.51 | 31 |
| 156 | mla_attend_layer | 211.38 | 207.52 | 215.18 | 31 |
| 157 | mla_merge_uv_layer | 64.55 | 63.38 | 65.50 | 31 |
| 158 | linear_with_residual_layer | 26.50 | 25.37 | 27.24 | 31 |
| 159 | moe_router_layer | 13.06 | 12.26 | 14.84 | 31 |
| 160 | gang_moe_w13_linear_layer | 32.04 | 29.36 | 33.01 | 31 |
| 161 | gang_moe_w2_silu_linear_layer | 45.22 | 42.77 | 49.01 | 31 |
| 162 | moe_mul_sum_add_layer | 30.63 | 27.61 | 33.20 | 31 |
| 163 | linear_norm_layer | 5.74 | 4.78 | 6.56 | 31 |
| 164 | mla_prep_layer | 19.28 | 18.46 | 20.44 | 31 |
| 165 | mla_attend_layer | 210.36 | 206.88 | 214.26 | 31 |
| 166 | mla_merge_uv_layer | 64.53 | 63.92 | 65.14 | 31 |
| 167 | linear_with_residual_layer | 26.97 | 25.40 | 30.44 | 31 |
| 168 | moe_router_layer | 14.32 | 12.58 | 19.00 | 31 |
| 169 | gang_moe_w13_linear_layer | 32.30 | 30.02 | 32.94 | 31 |
| 170 | gang_moe_w2_silu_linear_layer | 44.42 | 42.77 | 47.26 | 31 |
| 171 | moe_mul_sum_add_layer | 30.90 | 27.79 | 33.36 | 31 |
| 172 | linear_norm_layer | 5.42 | 4.64 | 6.42 | 31 |
| 173 | mla_prep_layer | 19.38 | 18.34 | 20.16 | 31 |
| 174 | mla_attend_layer | 210.97 | 208.11 | 214.63 | 31 |
| 175 | mla_merge_uv_layer | 64.46 | 61.23 | 64.99 | 31 |
| 176 | linear_with_residual_layer | 26.60 | 25.37 | 29.62 | 31 |
| 177 | moe_router_layer | 14.27 | 12.54 | 17.66 | 31 |
| 178 | gang_moe_w13_linear_layer | 31.03 | 29.30 | 32.14 | 31 |
| 179 | gang_moe_w2_silu_linear_layer | 44.68 | 42.19 | 48.60 | 31 |
| 180 | moe_mul_sum_add_layer | 29.68 | 27.92 | 32.74 | 31 |
| 181 | linear_norm_layer | 6.16 | 4.96 | 6.52 | 31 |
| 182 | mla_prep_layer | 18.98 | 18.40 | 20.14 | 31 |
| 183 | mla_attend_layer | 210.88 | 206.64 | 214.76 | 31 |
| 184 | mla_merge_uv_layer | 64.48 | 61.32 | 65.42 | 31 |
| 185 | linear_with_residual_layer | 26.78 | 25.46 | 30.40 | 31 |
| 186 | moe_router_layer | 14.10 | 12.52 | 17.70 | 31 |
| 187 | gang_moe_w13_linear_layer | 31.07 | 29.34 | 32.38 | 31 |
| 188 | gang_moe_w2_silu_linear_layer | 44.16 | 41.74 | 47.12 | 31 |
| 189 | moe_mul_sum_add_layer | 31.30 | 28.78 | 33.10 | 31 |
| 190 | linear_norm_layer | 5.62 | 4.67 | 6.29 | 31 |
| 191 | mla_prep_layer | 19.01 | 18.22 | 20.30 | 31 |
| 192 | mla_attend_layer | 212.12 | 209.06 | 215.70 | 31 |
| 193 | mla_merge_uv_layer | 64.82 | 63.30 | 65.66 | 31 |
| 194 | linear_with_residual_layer | 26.25 | 25.22 | 27.30 | 31 |
| 195 | moe_router_layer | 13.90 | 12.68 | 14.96 | 31 |
| 196 | gang_moe_w13_linear_layer | 31.77 | 30.35 | 33.01 | 31 |
| 197 | gang_moe_w2_silu_linear_layer | 44.32 | 41.77 | 48.81 | 31 |
| 198 | moe_mul_sum_add_layer | 30.15 | 28.00 | 32.82 | 31 |
| 199 | linear_norm_layer | 5.59 | 4.46 | 6.24 | 31 |
| 200 | mla_prep_layer | 19.18 | 18.20 | 20.18 | 31 |
| 201 | mla_attend_layer | 210.68 | 206.18 | 214.50 | 31 |
| 202 | mla_merge_uv_layer | 64.06 | 60.30 | 64.83 | 31 |
| 203 | linear_with_residual_layer | 26.44 | 25.16 | 30.40 | 31 |
| 204 | moe_router_layer | 13.94 | 12.48 | 17.72 | 31 |
| 205 | gang_moe_w13_linear_layer | 31.79 | 30.52 | 33.38 | 31 |
| 206 | gang_moe_w2_silu_linear_layer | 47.54 | 44.42 | 50.80 | 31 |
| 207 | moe_mul_sum_add_layer | 29.58 | 27.40 | 33.04 | 31 |
| 208 | linear_norm_layer | 5.51 | 4.60 | 6.32 | 31 |
| 209 | mla_prep_layer | 19.16 | 18.05 | 20.20 | 31 |
| 210 | mla_attend_layer | 210.77 | 206.84 | 214.72 | 31 |
| 211 | mla_merge_uv_layer | 64.88 | 62.05 | 65.72 | 31 |
| 212 | linear_with_residual_layer | 26.88 | 25.34 | 30.04 | 31 |
| 213 | moe_router_layer | 14.25 | 12.49 | 18.92 | 31 |
| 214 | gang_moe_w13_linear_layer | 31.36 | 29.56 | 32.34 | 31 |
| 215 | gang_moe_w2_silu_linear_layer | 44.57 | 41.94 | 47.52 | 31 |
| 216 | moe_mul_sum_add_layer | 30.79 | 27.88 | 33.16 | 31 |
| 217 | linear_norm_layer | 6.11 | 5.14 | 6.44 | 31 |
| 218 | mla_prep_layer | 18.58 | 18.06 | 19.64 | 31 |
| 219 | mla_attend_layer | 211.14 | 207.48 | 215.24 | 31 |
| 220 | mla_merge_uv_layer | 64.99 | 64.24 | 65.54 | 31 |
| 221 | linear_with_residual_layer | 26.37 | 25.38 | 27.60 | 31 |
| 222 | moe_router_layer | 13.85 | 12.60 | 16.96 | 31 |
| 223 | gang_moe_w13_linear_layer | 29.59 | 28.46 | 31.10 | 31 |
| 224 | gang_moe_w2_silu_linear_layer | 44.25 | 41.78 | 47.44 | 31 |
| 225 | moe_mul_sum_add_layer | 30.61 | 26.59 | 33.10 | 31 |
| 226 | linear_norm_layer | 5.54 | 4.72 | 6.39 | 31 |
| 227 | mla_prep_layer | 19.30 | 18.48 | 20.24 | 31 |
| 228 | mla_attend_layer | 211.80 | 208.14 | 214.52 | 31 |
| 229 | mla_merge_uv_layer | 64.36 | 63.80 | 64.80 | 31 |
| 230 | linear_with_residual_layer | 26.35 | 25.16 | 27.36 | 31 |
| 231 | moe_router_layer | 13.80 | 12.55 | 16.04 | 31 |
| 232 | gang_moe_w13_linear_layer | 30.97 | 29.41 | 32.47 | 31 |
| 233 | gang_moe_w2_silu_linear_layer | 45.00 | 42.59 | 49.28 | 31 |
| 234 | moe_mul_sum_add_layer | 30.33 | 28.20 | 33.00 | 31 |
| 235 | linear_norm_layer | 5.54 | 4.54 | 6.34 | 31 |
| 236 | mla_prep_layer | 19.19 | 18.42 | 20.52 | 31 |
| 237 | mla_attend_layer | 210.91 | 207.46 | 214.46 | 31 |
| 238 | mla_merge_uv_layer | 64.35 | 59.82 | 65.32 | 31 |
| 239 | linear_with_residual_layer | 26.24 | 25.28 | 29.16 | 31 |
| 240 | moe_router_layer | 13.48 | 12.54 | 14.88 | 31 |
| 241 | gang_moe_w13_linear_layer | 30.73 | 30.00 | 31.36 | 31 |
| 242 | gang_moe_w2_silu_linear_layer | 45.92 | 43.38 | 49.54 | 31 |
| 243 | moe_mul_sum_add_layer | 29.80 | 28.38 | 31.36 | 31 |
| 244 | linear_norm_layer | 5.67 | 4.68 | 7.32 | 31 |
| 245 | argmax_partial_layer | 0.36 | 0.00 | 2.87 | 31 |
| 246 | argmax_reduce_layer | 0.35 | 0.00 | 2.67 | 31 |
| 247 | event_247 | 3.62 | 0.00 | 99.47 | 31 |
| 248 | event_248 | 0.40 | 0.00 | 3.19 | 31 |
| 249 | event_249 | 0.24 | 0.00 | 1.49 | 31 |
| 250 | event_250 | 9.90 | 0.00 | 100.20 | 31 |
| 251 | event_251 | 0.33 | 0.00 | 0.96 | 31 |
| 252 | event_252 | 3.39 | 0.00 | 99.76 | 31 |
| 253 | event_253 | 0.28 | 0.00 | 1.19 | 31 |
| 254 | event_254 | 0.25 | 0.00 | 1.40 | 31 |
| 255 | event_255 | 6.67 | 0.00 | 102.45 | 31 |
| 256 | event_256 | 0.25 | 0.00 | 1.29 | 31 |
| 257 | event_257 | 0.27 | 0.00 | 1.22 | 31 |
| 258 | event_258 | 6.72 | 0.00 | 99.00 | 31 |
| 259 | event_259 | 0.23 | 0.00 | 1.90 | 31 |
| 260 | event_260 | 3.27 | 0.01 | 93.51 | 31 |
| 261 | event_261 | 0.27 | 0.00 | 0.74 | 31 |
| 262 | event_262 | 0.20 | 0.00 | 0.53 | 31 |
| 263 | event_263 | 0.25 | 0.00 | 0.76 | 31 |
| 264 | event_264 | 3.56 | 0.00 | 100.26 | 31 |
| 265 | event_265 | 0.22 | 0.00 | 0.92 | 31 |
| 266 | event_266 | 6.63 | 0.00 | 97.82 | 31 |
| 267 | event_267 | 0.20 | 0.00 | 0.72 | 31 |
| 268 | event_268 | 0.23 | 0.00 | 0.87 | 31 |
| 269 | event_269 | 0.15 | 0.00 | 1.18 | 31 |
| 270 | event_270 | 0.17 | 0.00 | 0.55 | 31 |
| 271 | event_271 | 3.45 | 0.00 | 97.64 | 31 |
| 272 | event_272 | 0.20 | 0.00 | 1.08 | 31 |
| 273 | event_273 | 0.24 | 0.00 | 1.76 | 31 |
| 274 | event_274 | 6.97 | 0.00 | 99.96 | 31 |
| 275 | event_275 | 0.36 | 0.00 | 2.30 | 31 |
| 276 | event_276 | 3.53 | 0.00 | 96.48 | 31 |
| 277 | event_277 | 6.53 | 0.00 | 101.44 | 31 |
| 278 | event_278 | 6.78 | 0.02 | 100.73 | 31 |
| 279 | event_279 | 22.59 | 0.00 | 101.32 | 31 |
| 280 | event_280 | 3.52 | 0.00 | 99.88 | 31 |
| 281 | event_281 | 6.76 | 0.00 | 100.36 | 31 |
| 282 | event_282 | 1.77 | 0.00 | 38.82 | 31 |
| 283 | event_283 | 1.69 | 0.00 | 37.81 | 31 |
| 284 | event_284 | 9.36 | 0.00 | 40.94 | 31 |
| 285 | event_285 | 1.89 | 0.00 | 39.87 | 31 |
| 286 | event_286 | 5.62 | 0.00 | 40.84 | 31 |
| 287 | event_287 | 6.59 | 0.00 | 39.80 | 31 |
| 288 | event_288 | 1.90 | 0.00 | 40.32 | 31 |
| 289 | event_289 | 0.42 | 0.01 | 1.67 | 31 |
| 290 | event_290 | 2.91 | 0.00 | 39.07 | 31 |
| 291 | event_291 | 1.72 | 0.04 | 37.48 | 31 |
| 292 | event_292 | 5.37 | 0.00 | 38.96 | 31 |
| 293 | event_293 | 1.81 | 0.00 | 38.38 | 31 |
| 294 | event_294 | 4.05 | 0.00 | 38.71 | 31 |
| 295 | event_295 | 6.56 | 6.17 | 6.98 | 31 |
| 296 | event_296 | 5.81 | 5.66 | 6.18 | 31 |
