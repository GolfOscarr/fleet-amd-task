# Correctness report

Thresholds: 4 x calibrated floor.

| Key | Boundary | Class | max_abs_err | rel_err | cos_sim | floor | threshold | Result |
|---|---|---|---|---|---|---|---|---|
| `L0.B3.c_kv` | B3 | norm | 1.953e-03 | 2.612e-03 | 1.00000 | 0.045 | 0.181 | **PASS** |
| `L0.B3.k_pe` | B3 | rope | 0.031 | 1.385e-03 | 1.00000 | 5.294e-03 | 0.021 | **PASS** |
| `L1.B3.c_kv` | B3 | norm | 6.409e-03 | 7.161e-03 | 0.99997 | 0.045 | 0.181 | **PASS** |
| `L1.B3.k_pe` | B3 | rope | 0.125 | 5.868e-03 | 0.99999 | 5.294e-03 | 0.021 | **PASS** |
| `L2.B2.q` | B2 | gemv | - | - | - | - | exact | **MISSING_REF**  |
| `L2.B3.c_kv` | B3 | norm | - | - | - | - | exact | **MISSING_REF**  |
| `L2.B3.k_pe` | B3 | rope | - | - | - | - | exact | **MISSING_REF**  |
| `L2.B4.q_pe` | B4 | rope | - | - | - | - | exact | **MISSING_REF**  |
| `L2.B6.attn` | B6 | attention | - | - | - | - | exact | **MISSING_REF**  |
| `L2.B8.router_logits` | B8 | router | - | - | - | - | exact | **MISSING_REF**  |
| `L2.B9.topk_idx` | B9 | exact | - | - | - | - | exact | **MISSING_REF**  |
| `L2.B10.topk_w` | B10 | router | - | - | - | - | exact | **MISSING_REF**  |
| `L2.B11.expert_2` | B11 | expert | - | - | - | - | exact | **MISSING_REF**  |
| `L2.B11.expert_30` | B11 | expert | - | - | - | - | exact | **MISSING_REF**  |
| `L2.B11.expert_40` | B11 | expert | - | - | - | - | exact | **MISSING_REF**  |
| `L2.B11.expert_45` | B11 | expert | - | - | - | - | exact | **MISSING_REF**  |
| `L2.B11.expert_54` | B11 | expert | - | - | - | - | exact | **MISSING_REF**  |
| `L2.B11.expert_55` | B11 | expert | - | - | - | - | exact | **MISSING_REF**  |
| `L2.B12.shared` | B12 | expert | - | - | - | - | exact | **MISSING_REF**  |
| `L2.B13.layer_out` | B13 | layer | - | - | - | - | exact | **MISSING_REF**  |

Output ids: **SKIP**, truncated graph: no head, ids are placeholders.
Route log: **PASS**, 1 steps compared, 0 mismatching (step, layer) pairs.

Overall: **FAIL** (5 pass, 0 fail, 16 missing).
