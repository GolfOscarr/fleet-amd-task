# Correctness report

Thresholds: 4 x calibrated floor.

| Key | Boundary | Class | max_abs_err | rel_err | cos_sim | floor | threshold | Result |
|---|---|---|---|---|---|---|---|---|
| `L0.B3.c_kv` | B3 | norm | 1.953e-03 | 2.612e-03 | 1.00000 | 0.045 | 0.181 | **PASS** |
| `L0.B3.k_pe` | B3 | rope | 0.031 | 1.385e-03 | 1.00000 | 5.294e-03 | 0.021 | **PASS** |
| `L1.B2.q` | B2 | gemv | 0.062 | 5.426e-03 | 0.99999 | 3.235e-03 | 0.013 | **PASS** |
| `L1.B3.c_kv` | B3 | norm | 6.409e-03 | 7.161e-03 | 0.99997 | 0.045 | 0.181 | **PASS** |
| `L1.B3.k_pe` | B3 | rope | 0.125 | 5.868e-03 | 0.99999 | 5.294e-03 | 0.021 | **PASS** |
| `L1.B4.q_pe` | B4 | rope | 0.031 | 4.632e-03 | 0.99999 | 5.294e-03 | 0.021 | **PASS** |
| `L1.B6.attn` | B6 | attention | 2.930e-03 | 7.079e-03 | 0.99998 | 0.010 | 0.040 | **PASS** |
| `L1.B8.router_logits` | B8 | router | 8.409e-03 | 3.649e-03 | 0.99999 | 3.593e-03 | 0.014 | **PASS** |
| `L1.B9.topk_idx` | B9 | exact | - | - | - | - | exact | **PASS** fleet [2, 19, 39, 40, 47, 49] ref [2, 19, 39, 40, 47, 49] |
| `L1.B10.topk_w` | B10 | router | 6.458e-04 | 3.712e-03 | 0.99999 | 3.593e-03 | 0.014 | **PASS** |
| `L1.B11.expert_19` | B11 | expert | 3.906e-03 | 0.010 | 0.99995 | 0.011 | 0.044 | **PASS** |
| `L1.B11.expert_2` | B11 | expert | 4.395e-03 | 8.463e-03 | 0.99996 | 0.011 | 0.044 | **PASS** |
| `L1.B11.expert_39` | B11 | expert | 7.812e-03 | 7.006e-03 | 0.99998 | 0.011 | 0.044 | **PASS** |
| `L1.B11.expert_40` | B11 | expert | 3.906e-03 | 9.209e-03 | 0.99996 | 0.011 | 0.044 | **PASS** |
| `L1.B11.expert_47` | B11 | expert | 2.441e-03 | 9.480e-03 | 0.99996 | 0.011 | 0.044 | **PASS** |
| `L1.B11.expert_49` | B11 | expert | 3.906e-03 | 8.947e-03 | 0.99996 | 0.011 | 0.044 | **PASS** |
| `L1.B12.shared` | B12 | expert | 9.766e-04 | 0.011 | 0.99995 | 0.011 | 0.044 | **PASS** |
| `L1.B13.layer_out` | B13 | layer | 3.906e-03 | 4.907e-03 | 0.99999 | 4.334e-03 | 0.017 | **PASS** |

Output ids: **SKIP**, truncated graph: no head, ids are placeholders.
Route log: **PASS**, 1 steps compared, 0 mismatching (step, layer) pairs.

Overall: **PASS** (19 pass, 0 fail, 0 missing).
