# Correctness report

Thresholds: 4 x calibrated floor.

| Key | Boundary | Class | max_abs_err | rel_err | cos_sim | floor | threshold | Result |
|---|---|---|---|---|---|---|---|---|
| `L0.B3.c_kv` | B3 | norm | 0.000 | 0.000 | 1.00000 | 0.045 | 0.181 | **PASS** |
| `L0.B3.k_pe` | B3 | rope | 0.000 | 0.000 | 1.00000 | 5.294e-03 | 0.021 | **PASS** |
| `L1.B2.q` | B2 | gemv | 0.062 | 4.630e-03 | 0.99999 | 3.235e-03 | 0.013 | **PASS** |
| `L1.B3.c_kv` | B3 | norm | 3.967e-03 | 6.172e-03 | 0.99998 | 0.045 | 0.181 | **PASS** |
| `L1.B3.k_pe` | B3 | rope | 0.125 | 5.517e-03 | 0.99999 | 5.294e-03 | 0.021 | **PASS** |
| `L1.B4.q_pe` | B4 | rope | 0.062 | 4.273e-03 | 0.99999 | 5.294e-03 | 0.021 | **PASS** |
| `L1.B6.attn` | B6 | attention | 1.953e-03 | 6.965e-03 | 0.99998 | 0.010 | 0.040 | **PASS** |
| `L1.B8.router_logits` | B8 | router | 9.604e-03 | 3.606e-03 | 0.99999 | 3.593e-03 | 0.014 | **PASS** |
| `L1.B9.topk_idx` | B9 | exact | - | - | - | - | exact | **PASS** fleet [2, 19, 39, 40, 47, 49] ref [2, 19, 39, 40, 47, 49] |
| `L1.B10.topk_w` | B10 | router | 2.371e-04 | 1.886e-03 | 1.00000 | 3.593e-03 | 0.014 | **PASS** |
| `L1.B11.expert_19` | B11 | expert | 3.418e-03 | 9.173e-03 | 0.99996 | 0.011 | 0.044 | **PASS** |
| `L1.B11.expert_2` | B11 | expert | 3.906e-03 | 8.414e-03 | 0.99996 | 0.011 | 0.044 | **PASS** |
| `L1.B11.expert_39` | B11 | expert | 5.859e-03 | 6.287e-03 | 0.99998 | 0.011 | 0.044 | **PASS** |
| `L1.B11.expert_40` | B11 | expert | 3.418e-03 | 8.895e-03 | 0.99996 | 0.011 | 0.044 | **PASS** |
| `L1.B11.expert_47` | B11 | expert | 2.197e-03 | 8.852e-03 | 0.99996 | 0.011 | 0.044 | **PASS** |
| `L1.B11.expert_49` | B11 | expert | 3.906e-03 | 8.259e-03 | 0.99997 | 0.011 | 0.044 | **PASS** |
| `L1.B12.shared` | B12 | expert | 1.465e-03 | 8.971e-03 | 0.99996 | 0.011 | 0.044 | **PASS** |
| `L1.B13.layer_out` | B13 | layer | 3.906e-03 | 4.645e-03 | 0.99999 | 4.334e-03 | 0.017 | **PASS** |

Output ids: **SKIP**, truncated graph: no head, ids are placeholders.
Route log: **PASS**, 1 steps compared, 0 mismatching (step, layer) pairs.

Overall: **PASS** (19 pass, 0 fail, 0 missing).
