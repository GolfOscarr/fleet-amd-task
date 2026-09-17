# Measurement report

| Quantity | Predicted | Measured |
|---|---|---|
| bytes per iteration (read, MiB) | 4709.9 weights + cache; about 4772.0 with activations | - |
| time per iteration, median (us) | 1148.5-1349.4 + 326 t_b + T_serial | 10224.0 |
| time per iteration, P95 (us) |  | 10296.5 |
| time per iteration from event timing, median (us) |  | 10223.6 |
| time per iteration from host wall clock (us) |  | 101311.8 |
| time per iteration from the kernel trace (us) |  | - |
| achieved read bandwidth (TB/s) | 3.66-4.3 over T_bw | - |
| launches per generation | 3 | - |
| L2 hit rate | 16-17% (Fleet's batch-1 figure) | - |
| tokens per second |  | 97.8 |
| workers with tasks (of those reporting) |  | 296 of 296 |
| tasks per XCD (placement) |  | 0:88705 1:88704 2:88704 3:88704 4:88704 5:88736 6:88767 7:88704 |
| shader clock from the spin (MHz) |  | - |
| exec cycles per task, busy workers |  | 41524.4 |
| exec us per task (at the spin's SCLK) |  | - |
| dep-wait us per iteration per busy worker |  | - |
| GFX clock from amd-smi during the run, median / max (MHz) | 2100 max (D2 of round 1) | 2102 / 2103 over 53 samples |
| memory clock from amd-smi, median (MHz) |  | 900 |

## Exec time per task by class (worker timing, I1)

| Class | tasks | cycles per task | us per task |
|---|---|---|---|
| attend | 33696 | 89279 | - |
| linear | 5632 | 59772 | - |
| linear_res | 57344 | 23750 | - |
| lnorm | 93478 | 45184 | - |
| merge | 12960 | 40303 | - |
| prep | 833 | 312941 | - |
| rms | 32 | 3673 | - |
| router | 832 | 50884 | - |
| w2silu | 240448 | 30567 | - |

## Per-operator time (event gaps, mean over iterations after the first)

| Event | Operator | mean us | min us | max us | n |
|---|---|---|---|---|---|
| 1 | embed_layer | 212.74 | 167.40 | 300.04 | 30 |
| 2 | linear_norm_layer | 14.31 | 12.49 | 15.93 | 31 |
| 3 | mla_prep_layer | 31.36 | 28.48 | 33.44 | 31 |
| 4 | mla_attend_layer | 150.03 | 146.79 | 161.07 | 31 |
| 5 | mla_merge_uv_layer | 59.96 | 59.72 | 60.24 | 31 |
| 6 | linear_with_residual_layer | 25.90 | 23.05 | 26.84 | 31 |
| 7 | rmsnorm_layer | 13.41 | 13.08 | 13.96 | 31 |
| 8 | gang_linear_silu_layer | 5.31 | 3.97 | 5.57 | 31 |
| 9 | linear_with_residual_layer | 34.32 | 33.36 | 36.43 | 31 |
| 10 | linear_norm_layer | 53.17 | 50.12 | 55.60 | 31 |
| 11 | mla_prep_layer | 18.77 | 17.97 | 19.60 | 31 |
| 12 | mla_attend_layer | 150.26 | 146.79 | 154.64 | 31 |
| 13 | mla_merge_uv_layer | 60.43 | 60.09 | 60.80 | 31 |
| 14 | linear_with_residual_layer | 23.14 | 22.61 | 23.48 | 31 |
| 15 | moe_router_layer | 12.75 | 12.24 | 13.76 | 31 |
| 16 | gang_moe_w13_linear_layer | 28.45 | 26.93 | 33.48 | 31 |
| 17 | gang_moe_w2_silu_linear_layer | 43.32 | 40.11 | 47.68 | 31 |
| 18 | moe_mul_sum_add_layer | 23.88 | 22.39 | 25.67 | 31 |
| 19 | linear_norm_layer | 5.25 | 4.48 | 5.64 | 31 |
| 20 | mla_prep_layer | 19.13 | 18.09 | 20.03 | 31 |
| 21 | mla_attend_layer | 150.22 | 147.31 | 153.88 | 31 |
| 22 | mla_merge_uv_layer | 60.57 | 60.33 | 60.93 | 31 |
| 23 | linear_with_residual_layer | 23.25 | 22.76 | 23.44 | 31 |
| 24 | moe_router_layer | 13.28 | 12.75 | 13.96 | 31 |
| 25 | gang_moe_w13_linear_layer | 28.18 | 26.09 | 33.68 | 31 |
| 26 | gang_moe_w2_silu_linear_layer | 44.09 | 40.75 | 46.71 | 31 |
| 27 | moe_mul_sum_add_layer | 23.22 | 22.08 | 25.32 | 31 |
| 28 | linear_norm_layer | 5.66 | 4.67 | 9.48 | 31 |
| 29 | mla_prep_layer | 19.00 | 18.23 | 19.85 | 31 |
| 30 | mla_attend_layer | 149.91 | 146.76 | 153.33 | 31 |
| 31 | mla_merge_uv_layer | 60.47 | 60.24 | 60.72 | 31 |
| 32 | linear_with_residual_layer | 23.48 | 23.16 | 23.88 | 31 |
| 33 | moe_router_layer | 13.57 | 12.84 | 16.31 | 31 |
| 34 | gang_moe_w13_linear_layer | 28.00 | 26.04 | 32.61 | 31 |
| 35 | gang_moe_w2_silu_linear_layer | 42.00 | 39.07 | 45.80 | 31 |
| 36 | moe_mul_sum_add_layer | 23.96 | 22.51 | 26.52 | 31 |
| 37 | linear_norm_layer | 5.38 | 4.60 | 5.84 | 31 |
| 38 | mla_prep_layer | 19.17 | 18.35 | 20.68 | 31 |
| 39 | mla_attend_layer | 149.44 | 146.12 | 153.67 | 31 |
| 40 | mla_merge_uv_layer | 60.51 | 60.17 | 60.65 | 31 |
| 41 | linear_with_residual_layer | 23.33 | 22.96 | 23.72 | 31 |
| 42 | moe_router_layer | 13.35 | 12.59 | 13.92 | 31 |
| 43 | gang_moe_w13_linear_layer | 28.86 | 27.08 | 33.45 | 31 |
| 44 | gang_moe_w2_silu_linear_layer | 41.41 | 39.07 | 44.11 | 31 |
| 45 | moe_mul_sum_add_layer | 23.85 | 22.12 | 25.33 | 31 |
| 46 | linear_norm_layer | 5.18 | 4.51 | 5.73 | 31 |
| 47 | mla_prep_layer | 18.74 | 17.88 | 20.01 | 31 |
| 48 | mla_attend_layer | 150.15 | 146.83 | 155.23 | 31 |
| 49 | mla_merge_uv_layer | 60.37 | 59.89 | 60.80 | 31 |
| 50 | linear_with_residual_layer | 23.41 | 22.93 | 23.68 | 31 |
| 51 | moe_router_layer | 13.22 | 12.63 | 13.96 | 31 |
| 52 | gang_moe_w13_linear_layer | 28.11 | 26.16 | 32.76 | 31 |
| 53 | gang_moe_w2_silu_linear_layer | 42.96 | 39.88 | 46.35 | 31 |
| 54 | moe_mul_sum_add_layer | 24.14 | 22.97 | 25.57 | 31 |
| 55 | linear_norm_layer | 5.13 | 4.52 | 5.92 | 31 |
| 56 | mla_prep_layer | 18.99 | 18.05 | 20.08 | 31 |
| 57 | mla_attend_layer | 150.42 | 147.31 | 154.16 | 31 |
| 58 | mla_merge_uv_layer | 60.55 | 60.33 | 60.93 | 31 |
| 59 | linear_with_residual_layer | 23.23 | 22.72 | 24.84 | 31 |
| 60 | moe_router_layer | 14.56 | 12.63 | 17.36 | 31 |
| 61 | gang_moe_w13_linear_layer | 27.04 | 25.60 | 29.57 | 31 |
| 62 | gang_moe_w2_silu_linear_layer | 41.88 | 39.04 | 44.59 | 31 |
| 63 | moe_mul_sum_add_layer | 23.40 | 21.89 | 25.31 | 31 |
| 64 | linear_norm_layer | 5.30 | 4.60 | 6.31 | 31 |
| 65 | mla_prep_layer | 19.08 | 18.12 | 19.72 | 31 |
| 66 | mla_attend_layer | 150.38 | 146.76 | 153.45 | 31 |
| 67 | mla_merge_uv_layer | 59.71 | 59.35 | 59.95 | 31 |
| 68 | linear_with_residual_layer | 23.25 | 22.84 | 24.25 | 31 |
| 69 | moe_router_layer | 13.40 | 12.80 | 14.00 | 31 |
| 70 | gang_moe_w13_linear_layer | 27.91 | 26.32 | 29.52 | 31 |
| 71 | gang_moe_w2_silu_linear_layer | 41.75 | 38.79 | 44.75 | 31 |
| 72 | moe_mul_sum_add_layer | 23.49 | 22.08 | 25.29 | 31 |
| 73 | linear_norm_layer | 5.24 | 4.69 | 6.04 | 31 |
| 74 | mla_prep_layer | 18.96 | 18.13 | 19.72 | 31 |
| 75 | mla_attend_layer | 149.14 | 146.35 | 152.23 | 31 |
| 76 | mla_merge_uv_layer | 59.96 | 59.64 | 60.16 | 31 |
| 77 | linear_with_residual_layer | 23.38 | 22.89 | 23.69 | 31 |
| 78 | moe_router_layer | 13.14 | 12.55 | 13.68 | 31 |
| 79 | gang_moe_w13_linear_layer | 26.92 | 26.29 | 28.33 | 31 |
| 80 | gang_moe_w2_silu_linear_layer | 44.43 | 41.95 | 47.83 | 31 |
| 81 | moe_mul_sum_add_layer | 23.73 | 22.29 | 25.65 | 31 |
| 82 | linear_norm_layer | 5.04 | 4.40 | 5.92 | 31 |
| 83 | mla_prep_layer | 18.76 | 18.08 | 19.36 | 31 |
| 84 | mla_attend_layer | 149.69 | 146.75 | 152.75 | 31 |
| 85 | mla_merge_uv_layer | 59.84 | 59.64 | 60.29 | 31 |
| 86 | linear_with_residual_layer | 23.32 | 22.88 | 23.57 | 31 |
| 87 | moe_router_layer | 13.24 | 12.71 | 13.68 | 31 |
| 88 | gang_moe_w13_linear_layer | 27.55 | 25.65 | 28.81 | 31 |
| 89 | gang_moe_w2_silu_linear_layer | 41.61 | 38.68 | 46.23 | 31 |
| 90 | moe_mul_sum_add_layer | 23.29 | 21.65 | 25.05 | 31 |
| 91 | linear_norm_layer | 5.91 | 4.56 | 7.55 | 31 |
| 92 | mla_prep_layer | 18.91 | 18.23 | 19.64 | 31 |
| 93 | mla_attend_layer | 150.55 | 147.08 | 153.83 | 31 |
| 94 | mla_merge_uv_layer | 59.94 | 59.73 | 60.21 | 31 |
| 95 | linear_with_residual_layer | 23.51 | 22.48 | 25.96 | 31 |
| 96 | moe_router_layer | 13.28 | 12.68 | 15.20 | 31 |
| 97 | gang_moe_w13_linear_layer | 27.51 | 26.08 | 29.00 | 31 |
| 98 | gang_moe_w2_silu_linear_layer | 42.58 | 40.07 | 46.92 | 31 |
| 99 | moe_mul_sum_add_layer | 24.02 | 22.49 | 25.45 | 31 |
| 100 | linear_norm_layer | 5.43 | 4.59 | 6.00 | 31 |
| 101 | mla_prep_layer | 18.66 | 17.88 | 19.28 | 31 |
| 102 | mla_attend_layer | 150.17 | 147.37 | 153.56 | 31 |
| 103 | mla_merge_uv_layer | 60.56 | 60.40 | 60.76 | 31 |
| 104 | linear_with_residual_layer | 23.29 | 22.80 | 26.72 | 31 |
| 105 | moe_router_layer | 13.56 | 12.40 | 16.39 | 31 |
| 106 | gang_moe_w13_linear_layer | 27.34 | 25.97 | 29.69 | 31 |
| 107 | gang_moe_w2_silu_linear_layer | 42.68 | 39.71 | 46.72 | 31 |
| 108 | moe_mul_sum_add_layer | 23.65 | 22.16 | 25.60 | 31 |
| 109 | linear_norm_layer | 5.25 | 4.61 | 5.64 | 31 |
| 110 | mla_prep_layer | 18.72 | 17.97 | 19.51 | 31 |
| 111 | mla_attend_layer | 149.21 | 146.08 | 152.40 | 31 |
| 112 | mla_merge_uv_layer | 59.87 | 59.64 | 60.04 | 31 |
| 113 | linear_with_residual_layer | 23.63 | 22.89 | 26.60 | 31 |
| 114 | moe_router_layer | 13.33 | 12.68 | 17.40 | 31 |
| 115 | gang_moe_w13_linear_layer | 26.85 | 26.33 | 30.48 | 31 |
| 116 | gang_moe_w2_silu_linear_layer | 42.17 | 39.39 | 44.92 | 31 |
| 117 | moe_mul_sum_add_layer | 23.74 | 21.97 | 26.00 | 31 |
| 118 | linear_norm_layer | 5.03 | 4.65 | 6.84 | 31 |
| 119 | mla_prep_layer | 18.92 | 18.09 | 19.25 | 31 |
| 120 | mla_attend_layer | 149.59 | 146.59 | 152.79 | 31 |
| 121 | mla_merge_uv_layer | 59.96 | 59.68 | 60.24 | 31 |
| 122 | linear_with_residual_layer | 23.26 | 22.88 | 23.64 | 31 |
| 123 | moe_router_layer | 13.27 | 12.57 | 13.88 | 31 |
| 124 | gang_moe_w13_linear_layer | 27.04 | 25.89 | 28.76 | 31 |
| 125 | gang_moe_w2_silu_linear_layer | 43.42 | 38.80 | 47.08 | 31 |
| 126 | moe_mul_sum_add_layer | 22.85 | 21.80 | 24.76 | 31 |
| 127 | linear_norm_layer | 5.41 | 4.56 | 9.24 | 31 |
| 128 | mla_prep_layer | 18.45 | 17.72 | 19.25 | 31 |
| 129 | mla_attend_layer | 150.38 | 147.39 | 153.63 | 31 |
| 130 | mla_merge_uv_layer | 59.87 | 59.68 | 60.13 | 31 |
| 131 | linear_with_residual_layer | 23.45 | 22.92 | 24.88 | 31 |
| 132 | moe_router_layer | 13.68 | 12.68 | 17.52 | 31 |
| 133 | gang_moe_w13_linear_layer | 26.85 | 25.77 | 28.57 | 31 |
| 134 | gang_moe_w2_silu_linear_layer | 42.20 | 39.72 | 44.24 | 31 |
| 135 | moe_mul_sum_add_layer | 22.99 | 21.68 | 24.84 | 31 |
| 136 | linear_norm_layer | 5.81 | 4.75 | 7.39 | 31 |
| 137 | mla_prep_layer | 18.69 | 18.08 | 19.24 | 31 |
| 138 | mla_attend_layer | 150.01 | 147.28 | 153.05 | 31 |
| 139 | mla_merge_uv_layer | 59.99 | 59.67 | 60.35 | 31 |
| 140 | linear_with_residual_layer | 23.48 | 22.80 | 26.08 | 31 |
| 141 | moe_router_layer | 13.41 | 12.60 | 15.93 | 31 |
| 142 | gang_moe_w13_linear_layer | 27.21 | 26.28 | 28.93 | 31 |
| 143 | gang_moe_w2_silu_linear_layer | 41.99 | 38.55 | 45.64 | 31 |
| 144 | moe_mul_sum_add_layer | 23.68 | 22.12 | 25.48 | 31 |
| 145 | linear_norm_layer | 5.62 | 4.72 | 7.12 | 31 |
| 146 | mla_prep_layer | 18.95 | 18.35 | 19.59 | 31 |
| 147 | mla_attend_layer | 149.23 | 146.39 | 152.15 | 31 |
| 148 | mla_merge_uv_layer | 60.33 | 59.81 | 60.68 | 31 |
| 149 | linear_with_residual_layer | 23.21 | 22.81 | 23.60 | 31 |
| 150 | moe_router_layer | 13.45 | 12.96 | 14.96 | 31 |
| 151 | gang_moe_w13_linear_layer | 26.99 | 26.56 | 28.60 | 31 |
| 152 | gang_moe_w2_silu_linear_layer | 42.20 | 39.96 | 44.59 | 31 |
| 153 | moe_mul_sum_add_layer | 24.01 | 22.33 | 25.92 | 31 |
| 154 | linear_norm_layer | 5.48 | 4.93 | 8.56 | 31 |
| 155 | mla_prep_layer | 18.78 | 18.27 | 19.44 | 31 |
| 156 | mla_attend_layer | 149.67 | 146.47 | 152.84 | 31 |
| 157 | mla_merge_uv_layer | 60.37 | 60.13 | 60.61 | 31 |
| 158 | linear_with_residual_layer | 23.32 | 22.84 | 23.64 | 31 |
| 159 | moe_router_layer | 12.67 | 12.07 | 15.59 | 31 |
| 160 | gang_moe_w13_linear_layer | 27.91 | 26.25 | 29.61 | 31 |
| 161 | gang_moe_w2_silu_linear_layer | 42.02 | 38.43 | 47.80 | 31 |
| 162 | moe_mul_sum_add_layer | 23.83 | 22.39 | 25.92 | 31 |
| 163 | linear_norm_layer | 5.62 | 4.57 | 9.64 | 31 |
| 164 | mla_prep_layer | 18.38 | 17.73 | 19.00 | 31 |
| 165 | mla_attend_layer | 150.04 | 146.96 | 153.27 | 31 |
| 166 | mla_merge_uv_layer | 60.60 | 60.37 | 60.81 | 31 |
| 167 | linear_with_residual_layer | 23.15 | 22.60 | 23.56 | 31 |
| 168 | moe_router_layer | 13.55 | 12.75 | 15.67 | 31 |
| 169 | gang_moe_w13_linear_layer | 26.67 | 25.24 | 29.08 | 31 |
| 170 | gang_moe_w2_silu_linear_layer | 42.10 | 39.07 | 46.64 | 31 |
| 171 | moe_mul_sum_add_layer | 23.77 | 22.27 | 26.04 | 31 |
| 172 | linear_norm_layer | 5.33 | 4.71 | 5.73 | 31 |
| 173 | mla_prep_layer | 18.50 | 18.04 | 19.28 | 31 |
| 174 | mla_attend_layer | 150.02 | 147.01 | 152.97 | 31 |
| 175 | mla_merge_uv_layer | 60.14 | 59.83 | 60.39 | 31 |
| 176 | linear_with_residual_layer | 23.34 | 22.91 | 23.84 | 31 |
| 177 | moe_router_layer | 13.67 | 12.76 | 15.68 | 31 |
| 178 | gang_moe_w13_linear_layer | 28.02 | 26.20 | 28.76 | 31 |
| 179 | gang_moe_w2_silu_linear_layer | 41.21 | 38.80 | 44.28 | 31 |
| 180 | moe_mul_sum_add_layer | 23.76 | 21.88 | 25.20 | 31 |
| 181 | linear_norm_layer | 5.24 | 4.56 | 6.88 | 31 |
| 182 | mla_prep_layer | 18.82 | 18.24 | 19.81 | 31 |
| 183 | mla_attend_layer | 148.98 | 145.75 | 152.44 | 31 |
| 184 | mla_merge_uv_layer | 59.88 | 59.52 | 60.17 | 31 |
| 185 | linear_with_residual_layer | 23.33 | 22.81 | 23.93 | 31 |
| 186 | moe_router_layer | 13.29 | 12.59 | 14.65 | 31 |
| 187 | gang_moe_w13_linear_layer | 28.38 | 26.96 | 29.80 | 31 |
| 188 | gang_moe_w2_silu_linear_layer | 44.13 | 40.07 | 48.35 | 31 |
| 189 | moe_mul_sum_add_layer | 23.49 | 22.24 | 25.48 | 31 |
| 190 | linear_norm_layer | 5.36 | 4.52 | 8.52 | 31 |
| 191 | mla_prep_layer | 18.29 | 17.53 | 19.16 | 31 |
| 192 | mla_attend_layer | 149.67 | 146.87 | 153.08 | 31 |
| 193 | mla_merge_uv_layer | 59.84 | 59.61 | 60.17 | 31 |
| 194 | linear_with_residual_layer | 23.21 | 22.80 | 23.63 | 31 |
| 195 | moe_router_layer | 13.69 | 12.64 | 16.29 | 31 |
| 196 | gang_moe_w13_linear_layer | 27.75 | 26.24 | 29.04 | 31 |
| 197 | gang_moe_w2_silu_linear_layer | 41.32 | 38.80 | 44.03 | 31 |
| 198 | moe_mul_sum_add_layer | 23.75 | 22.07 | 25.80 | 31 |
| 199 | linear_norm_layer | 5.52 | 4.59 | 7.21 | 31 |
| 200 | mla_prep_layer | 18.64 | 18.12 | 19.33 | 31 |
| 201 | mla_attend_layer | 150.30 | 147.07 | 152.99 | 31 |
| 202 | mla_merge_uv_layer | 60.62 | 60.25 | 60.89 | 31 |
| 203 | linear_with_residual_layer | 23.17 | 22.64 | 24.15 | 31 |
| 204 | moe_router_layer | 13.68 | 12.80 | 17.16 | 31 |
| 205 | gang_moe_w13_linear_layer | 26.92 | 25.25 | 29.09 | 31 |
| 206 | gang_moe_w2_silu_linear_layer | 43.84 | 41.19 | 48.31 | 31 |
| 207 | moe_mul_sum_add_layer | 23.72 | 21.88 | 25.97 | 31 |
| 208 | linear_norm_layer | 5.41 | 4.65 | 7.64 | 31 |
| 209 | mla_prep_layer | 18.53 | 17.93 | 19.36 | 31 |
| 210 | mla_attend_layer | 150.15 | 147.25 | 153.12 | 31 |
| 211 | mla_merge_uv_layer | 60.02 | 59.63 | 60.27 | 31 |
| 212 | linear_with_residual_layer | 23.27 | 22.80 | 24.47 | 31 |
| 213 | moe_router_layer | 13.29 | 12.64 | 15.41 | 31 |
| 214 | gang_moe_w13_linear_layer | 27.85 | 26.25 | 29.12 | 31 |
| 215 | gang_moe_w2_silu_linear_layer | 41.59 | 38.83 | 44.60 | 31 |
| 216 | moe_mul_sum_add_layer | 23.03 | 21.72 | 24.93 | 31 |
| 217 | linear_norm_layer | 5.28 | 4.52 | 6.97 | 31 |
| 218 | mla_prep_layer | 18.99 | 18.41 | 19.73 | 31 |
| 219 | mla_attend_layer | 149.01 | 145.83 | 152.08 | 31 |
| 220 | mla_merge_uv_layer | 60.16 | 59.73 | 60.53 | 31 |
| 221 | linear_with_residual_layer | 23.16 | 22.64 | 23.60 | 31 |
| 222 | moe_router_layer | 13.37 | 12.91 | 13.84 | 31 |
| 223 | gang_moe_w13_linear_layer | 26.39 | 26.00 | 27.00 | 31 |
| 224 | gang_moe_w2_silu_linear_layer | 42.53 | 40.99 | 47.80 | 31 |
| 225 | moe_mul_sum_add_layer | 23.49 | 21.88 | 24.80 | 31 |
| 226 | linear_norm_layer | 5.23 | 4.40 | 7.01 | 31 |
| 227 | mla_prep_layer | 18.70 | 18.01 | 19.41 | 31 |
| 228 | mla_attend_layer | 149.78 | 146.59 | 152.83 | 31 |
| 229 | mla_merge_uv_layer | 60.39 | 60.01 | 60.65 | 31 |
| 230 | linear_with_residual_layer | 23.09 | 22.60 | 23.40 | 31 |
| 231 | moe_router_layer | 13.75 | 12.80 | 17.20 | 31 |
| 232 | gang_moe_w13_linear_layer | 27.11 | 25.36 | 28.24 | 31 |
| 233 | gang_moe_w2_silu_linear_layer | 41.00 | 38.35 | 44.04 | 31 |
| 234 | moe_mul_sum_add_layer | 23.96 | 22.64 | 25.64 | 31 |
| 235 | linear_norm_layer | 5.17 | 4.43 | 6.29 | 31 |
| 236 | mla_prep_layer | 18.68 | 17.96 | 19.56 | 31 |
| 237 | mla_attend_layer | 150.34 | 147.28 | 153.47 | 31 |
| 238 | mla_merge_uv_layer | 59.91 | 59.65 | 60.08 | 31 |
| 239 | linear_with_residual_layer | 23.08 | 22.57 | 23.32 | 31 |
| 240 | moe_router_layer | 13.20 | 12.60 | 14.69 | 31 |
| 241 | gang_moe_w13_linear_layer | 26.94 | 25.32 | 28.08 | 31 |
| 242 | gang_moe_w2_silu_linear_layer | 41.48 | 38.96 | 44.32 | 31 |
| 243 | moe_mul_sum_add_layer | 23.24 | 21.88 | 24.76 | 31 |
| 244 | linear_norm_layer | 6.30 | 4.49 | 7.48 | 31 |
| 245 | argmax_partial_layer | 0.24 | 0.00 | 1.23 | 31 |
| 246 | argmax_reduce_layer | 0.23 | 0.00 | 0.92 | 31 |
| 247 | event_247 | 0.24 | 0.00 | 1.39 | 31 |
| 248 | event_248 | 0.24 | 0.00 | 1.00 | 31 |
| 249 | event_249 | 0.29 | 0.00 | 1.40 | 31 |
| 250 | event_250 | 0.26 | 0.00 | 1.87 | 31 |
| 251 | event_251 | 0.14 | 0.00 | 0.60 | 31 |
| 252 | event_252 | 0.18 | 0.00 | 0.83 | 31 |
| 253 | event_253 | 0.23 | 0.00 | 2.23 | 31 |
| 254 | event_254 | 0.19 | 0.00 | 0.80 | 31 |
| 255 | event_255 | 0.27 | 0.00 | 0.97 | 31 |
| 256 | event_256 | 0.24 | 0.00 | 1.35 | 31 |
| 257 | event_257 | 0.20 | 0.00 | 0.64 | 31 |
| 258 | event_258 | 0.28 | 0.00 | 1.01 | 31 |
| 259 | event_259 | 0.21 | 0.00 | 0.91 | 31 |
| 260 | event_260 | 0.19 | 0.00 | 0.80 | 31 |
| 261 | event_261 | 0.30 | 0.00 | 2.29 | 31 |
| 262 | event_262 | 0.20 | 0.00 | 1.08 | 31 |
| 263 | event_263 | 0.28 | 0.00 | 1.99 | 31 |
| 264 | event_264 | 0.29 | 0.00 | 1.92 | 31 |
| 265 | event_265 | 5.86 | 0.03 | 87.77 | 31 |
| 266 | event_266 | 3.13 | 0.00 | 87.09 | 31 |
| 267 | event_267 | 0.20 | 0.00 | 1.03 | 31 |
| 268 | event_268 | 0.19 | 0.00 | 0.68 | 31 |
| 269 | event_269 | 0.19 | 0.00 | 0.60 | 31 |
| 270 | event_270 | 0.30 | 0.00 | 1.36 | 31 |
| 271 | event_271 | 0.34 | 0.00 | 3.20 | 31 |
| 272 | event_272 | 0.26 | 0.00 | 1.61 | 31 |
| 273 | event_273 | 0.32 | 0.00 | 1.28 | 31 |
| 274 | event_274 | 0.29 | 0.00 | 1.12 | 31 |
| 275 | event_275 | 3.20 | 0.00 | 87.20 | 31 |
| 276 | event_276 | 3.42 | 0.00 | 88.65 | 31 |
| 277 | event_277 | 17.25 | 0.00 | 89.55 | 31 |
| 278 | event_278 | 14.15 | 0.00 | 88.96 | 31 |
| 279 | event_279 | 28.48 | 0.00 | 89.48 | 31 |
| 280 | event_280 | 6.05 | 0.00 | 87.09 | 31 |
| 281 | event_281 | 8.78 | 0.00 | 88.19 | 31 |
| 282 | event_282 | 3.02 | 0.04 | 40.05 | 31 |
| 283 | event_283 | 5.56 | 0.00 | 40.56 | 31 |
| 284 | event_284 | 1.81 | 0.00 | 40.16 | 31 |
| 285 | event_285 | 1.71 | 0.00 | 40.64 | 31 |
| 286 | event_286 | 7.90 | 0.00 | 40.44 | 31 |
| 287 | event_287 | 11.70 | 0.00 | 42.79 | 31 |
| 288 | event_288 | 1.53 | 0.00 | 40.76 | 31 |
| 289 | event_289 | 1.64 | 0.00 | 40.20 | 31 |
| 290 | event_290 | 1.81 | 0.00 | 40.88 | 31 |
| 291 | event_291 | 0.34 | 0.00 | 1.37 | 31 |
| 292 | event_292 | 1.71 | 0.00 | 40.56 | 31 |
| 293 | event_293 | 3.11 | 0.04 | 40.55 | 31 |
| 294 | event_294 | 2.91 | 0.00 | 38.88 | 31 |
| 295 | event_295 | 6.20 | 5.80 | 6.52 | 31 |
| 296 | event_296 | 5.62 | 5.39 | 5.75 | 31 |
