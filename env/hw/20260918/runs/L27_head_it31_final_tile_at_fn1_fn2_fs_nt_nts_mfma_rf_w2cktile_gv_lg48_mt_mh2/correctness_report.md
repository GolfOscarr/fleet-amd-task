# Correctness report

Thresholds: 4 x calibrated floor.
The run made 31 iterations: the boundaries other than the cache rows and the first token were dumped from iteration 30 and are not comparable against the reference's step 0 (NOT_COMPARABLE below); the output ids and the route log carry the verdict.

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
| `L8.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L8.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L9.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L9.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L10.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L10.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L11.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L11.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L12.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L12.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L13.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L13.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L14.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L14.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L15.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L15.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L16.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L16.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L17.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L17.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L18.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L18.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L19.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L19.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L20.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L20.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L21.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L21.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L22.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L22.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L23.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L23.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L24.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L24.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L25.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L25.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L26.B2.q` | B2 | gemv | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L26.B3.c_kv` | B3 | norm | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L26.B3.k_pe` | B3 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L26.B4.q_pe` | B4 | rope | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L26.B6.attn` | B6 | attention | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L26.B8.router_logits` | B8 | router | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L26.B9.topk_idx` | B9 | exact | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L26.B10.topk_w` | B10 | router | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L26.B11.expert_14` | B11 | expert | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L26.B11.expert_15` | B11 | expert | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L26.B11.expert_19` | B11 | expert | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L26.B11.expert_2` | B11 | expert | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L26.B11.expert_23` | B11 | expert | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L26.B11.expert_50` | B11 | expert | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L26.B12.shared` | B12 | expert | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `L26.B13.layer_out` | B13 | layer | - | - | - | - | exact | **NOT_CAPTURED** the reference captures layers 0, 1 |
| `head.B15.logits` | B15 | logits | - | - | - | - | exact | **NOT_COMPARABLE** dumped from iteration 30; the reference has no ref_boundaries_step30 |
| `head.B16.token` | B16 | exact | - | - | - | - | exact | **PASS** fleet 25 ref 25 |

Output ids: **PARTIAL**, 31 of 32 matched.
Route log: **PASS**, 31 steps compared, 63 mismatching (step, layer) pairs: 31 ties, 32 cascades, 0 disagreements (the tie rule, tol 0.0144 of the larger weight).

Overall: **PASS** (7 pass, 0 fail, 0 missing, 1 not comparable, 64 of layers the reference did not capture).
