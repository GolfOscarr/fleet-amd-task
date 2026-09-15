# Correctness report

Thresholds: 4 x calibrated floor.

| Key | Boundary | Class | max_abs_err | rel_err | cos_sim | floor | threshold | Result |
|---|---|---|---|---|---|---|---|---|
| `L0.B1.norm1` | B1 | norm | 0.016 | 2.309e-03 | 1.00000 | 0.045 | 0.181 | **PASS** |
| `L0.B2.q` | B2 | gemv | 0.062 | 1.974e-03 | 1.00000 | 3.235e-03 | 0.013 | **PASS** |

Output ids: **FAIL**, 0 of 32 matched, first divergence at index 0.

Overall: **FAIL** (2 pass, 1 fail, 0 missing).
