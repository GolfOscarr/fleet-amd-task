# Correctness report

Thresholds: 4 x calibrated floor.

| Key | Boundary | Class | max_abs_err | rel_err | cos_sim | floor | threshold | Result |
|---|---|---|---|---|---|---|---|---|
| `L0.B3.c_kv` | B3 | norm | 0.000 | 0.000 | 1.00000 | 0.045 | 0.181 | **PASS** |
| `L0.B3.k_pe` | B3 | rope | 0.000 | 0.000 | 1.00000 | 5.294e-03 | 0.021 | **PASS** |
| `L1.B3.c_kv` | B3 | norm | 3.967e-03 | 6.172e-03 | 0.99998 | 0.045 | 0.181 | **PASS** |
| `L1.B3.k_pe` | B3 | rope | 0.125 | 5.517e-03 | 0.99999 | 5.294e-03 | 0.021 | **PASS** |
| `L2.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L2.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L3.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L3.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L4.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L4.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L5.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L5.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L6.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L6.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L7.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L7.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L8.B2.q` | B2 | gemv | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L8.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L8.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L8.B4.q_pe` | B4 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L8.B6.attn` | B6 | attention | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L8.B8.router_logits` | B8 | router | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L8.B9.topk_idx` | B9 | exact | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L8.B10.topk_w` | B10 | router | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L8.B11.expert_1` | B11 | expert | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L8.B11.expert_11` | B11 | expert | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L8.B11.expert_2` | B11 | expert | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L8.B11.expert_28` | B11 | expert | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L8.B11.expert_61` | B11 | expert | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L8.B11.expert_9` | B11 | expert | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L8.B12.shared` | B12 | expert | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L8.B13.layer_out` | B13 | layer | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |

Output ids: **SKIP**, truncated graph: no head, ids are placeholders.
Route log: **PASS**, 1 steps compared, 1 mismatching (step, layer) pairs: 1 ties, 0 cascades, 0 disagreements (the tie rule, tol 0.0144 of the larger weight).

Overall: **PASS** (5 pass, 0 fail, 0 missing, 28 of layers the reference did not capture).
