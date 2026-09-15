# Correctness report

Thresholds: 4 x calibrated floor.

| Key | Boundary | Class | max_abs_err | rel_err | cos_sim | floor | threshold | Result |
|---|---|---|---|---|---|---|---|---|
| `L0.B3.c_kv` | B3 | norm | 1.953e-03 | 2.612e-03 | 1.00000 | 0.045 | 0.181 | **PASS** |
| `L0.B3.k_pe` | B3 | rope | 0.031 | 1.385e-03 | 1.00000 | 5.294e-03 | 0.021 | **PASS** |
| `L0.B6.attn` | B6 | attention | 4.883e-03 | 7.995e-03 | 0.99997 | 0.010 | 0.040 | **PASS** |
| `L0.B13.layer_out` | B13 | layer | 3.906e-03 | 3.721e-03 | 0.99999 | 4.334e-03 | 0.017 | **PASS** |
| `L1.B1.norm1` | B1 | norm | 0.047 | 4.287e-03 | 0.99999 | 0.045 | 0.181 | **PASS** |
| `L1.B2.q` | B2 | gemv | 0.062 | 5.426e-03 | 0.99999 | 3.235e-03 | 0.013 | **PASS** |
| `L1.B3.c_kv` | B3 | norm | 6.409e-03 | 7.161e-03 | 0.99997 | 0.045 | 0.181 | **PASS** |
| `L1.B3.k_pe` | B3 | rope | 0.125 | 5.868e-03 | 0.99999 | 5.294e-03 | 0.021 | **PASS** |
| `L1.B4.q_pe` | B4 | rope | 0.031 | 4.632e-03 | 0.99999 | 5.294e-03 | 0.021 | **PASS** |
| `L1.B5.scores` | B5 | scores | 0.089 | 5.380e-03 | 0.99999 | 2.606e-03 | 0.010 | **PASS** |

Output ids: **SKIP**, truncated graph: no head, ids are placeholders.
Route log: **FAIL**, 1 steps compared, 1 mismatching (step, layer) pairs.

Overall: **FAIL** (10 pass, 1 fail, 0 missing).
