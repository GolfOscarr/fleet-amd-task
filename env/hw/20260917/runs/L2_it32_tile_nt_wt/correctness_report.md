# Correctness report

Thresholds: 4 x calibrated floor.

| Key | Boundary | Class | max_abs_err | rel_err | cos_sim | floor | threshold | Result |
|---|---|---|---|---|---|---|---|---|
| `L0.B3.c_kv` | B3 | norm | 1.953e-03 | 2.612e-03 | 1.00000 | 0.045 | 0.181 | **PASS** |
| `L0.B3.k_pe` | B3 | rope | 0.031 | 1.385e-03 | 1.00000 | 5.294e-03 | 0.021 | **PASS** |
| `L1.B2.q` | B2 | gemv | 9.688 | 1.116 | 0.40409 | 3.235e-03 | 0.013 | **FAIL** |
| `L1.B3.c_kv` | B3 | norm | 6.409e-03 | 7.161e-03 | 0.99997 | 0.045 | 0.181 | **PASS** |
| `L1.B3.k_pe` | B3 | rope | 0.125 | 5.868e-03 | 0.99999 | 5.294e-03 | 0.021 | **PASS** |
| `L1.B4.q_pe` | B4 | rope | 9.516 | 1.003 | 0.55894 | 5.294e-03 | 0.021 | **FAIL** |
| `L1.B6.attn` | B6 | attention | 0.526 | 1.449 | 0.05248 | 0.010 | 0.040 | **FAIL** |
| `L1.B8.router_logits` | B8 | router | 3.783 | 1.231 | -0.08254 | 3.593e-03 | 0.014 | **FAIL** |
| `L1.B9.topk_idx` | B9 | exact | - | - | - | - | exact | **FAIL** fleet [4, 24, 25, 35, 55, 56] ref [2, 19, 39, 40, 47, 49] |
| `L1.B10.topk_w` | B10 | router | - | - | - | - | exact | **FAIL** expert sets differ: fleet [4, 24, 25, 35, 55, 56] ref [2, 19, 39, 40, 47, 49] |
| `L1.B11.expert_24` | B11 | expert | - | - | - | - | exact | **MISSING_REF**  |
| `L1.B11.expert_25` | B11 | expert | - | - | - | - | exact | **MISSING_REF**  |
| `L1.B11.expert_35` | B11 | expert | - | - | - | - | exact | **MISSING_REF**  |
| `L1.B11.expert_4` | B11 | expert | - | - | - | - | exact | **MISSING_REF**  |
| `L1.B11.expert_55` | B11 | expert | - | - | - | - | exact | **MISSING_REF**  |
| `L1.B11.expert_56` | B11 | expert | - | - | - | - | exact | **MISSING_REF**  |
| `L1.B12.shared` | B12 | expert | 0.129 | 1.171 | 0.64083 | 0.011 | 0.044 | **FAIL** |
| `L1.B13.layer_out` | B13 | layer | 1.707 | 1.092 | 0.26110 | 4.334e-03 | 0.017 | **FAIL** |

Output ids: **SKIP**, truncated graph: no head, ids are placeholders.
Route log: **FAIL**, 32 steps compared, 31 mismatching (step, layer) pairs.

Overall: **FAIL** (4 pass, 9 fail, 6 missing).
