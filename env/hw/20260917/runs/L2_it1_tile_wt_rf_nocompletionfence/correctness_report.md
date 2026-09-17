# Correctness report

Thresholds: 4 x calibrated floor.

| Key | Boundary | Class | max_abs_err | rel_err | cos_sim | floor | threshold | Result |
|---|---|---|---|---|---|---|---|---|
| `L0.B3.c_kv` | B3 | norm | 1.398 | 1.000 | 0.00000 | 0.045 | 0.181 | **FAIL** |
| `L0.B3.k_pe` | B3 | rope | 21.625 | 1.000 | 0.00000 | 5.294e-03 | 0.021 | **FAIL** |
| `L1.B2.q` | B2 | gemv | 9.720 | 0.932 | 0.36377 | 3.235e-03 | 0.013 | **FAIL** |
| `L1.B3.c_kv` | B3 | norm | 1.148 | 1.168 | 0.42101 | 0.045 | 0.181 | **FAIL** |
| `L1.B3.k_pe` | B3 | rope | 12.188 | 0.640 | 0.91260 | 5.294e-03 | 0.021 | **FAIL** |
| `L1.B4.q_pe` | B4 | rope | 9.720 | 0.943 | 0.35760 | 5.294e-03 | 0.021 | **FAIL** |
| `L1.B6.attn` | B6 | attention | 0.215 | 1.070 | 0.04369 | 0.010 | 0.040 | **FAIL** |
| `L1.B8.router_logits` | B8 | router | 5.346 | 2.225 | 0.74893 | 3.593e-03 | 0.014 | **FAIL** |
| `L1.B9.topk_idx` | B9 | exact | - | - | - | - | exact | **FAIL** fleet [1, 2, 8, 19, 28, 39] ref [2, 19, 39, 40, 47, 49] |
| `L1.B10.topk_w` | B10 | router | - | - | - | - | exact | **FAIL** expert sets differ: fleet [1, 2, 8, 19, 28, 39] ref [2, 19, 39, 40, 47, 49] |
| `L1.B11.expert_1` | B11 | expert | - | - | - | - | exact | **MISSING_REF**  |
| `L1.B11.expert_19` | B11 | expert | 0.484 | 1.000 | 0.00000 | 0.011 | 0.044 | **FAIL** |
| `L1.B11.expert_2` | B11 | expert | 0.879 | 1.000 | 0.00000 | 0.011 | 0.044 | **FAIL** |
| `L1.B11.expert_28` | B11 | expert | - | - | - | - | exact | **MISSING_REF**  |
| `L1.B11.expert_39` | B11 | expert | 1.820 | 1.000 | 0.00000 | 0.011 | 0.044 | **FAIL** |
| `L1.B11.expert_8` | B11 | expert | - | - | - | - | exact | **MISSING_REF**  |
| `L1.B12.shared` | B12 | expert | 0.311 | 1.000 | 0.00000 | 0.011 | 0.044 | **FAIL** |
| `L1.B13.layer_out` | B13 | layer | 1.586 | 0.686 | 0.75421 | 4.334e-03 | 0.017 | **FAIL** |

Output ids: **SKIP**, truncated graph: no head, ids are placeholders.
Route log: **FAIL**, 1 steps compared, 1 mismatching (step, layer) pairs.

Overall: **FAIL** (0 pass, 16 fail, 3 missing).
