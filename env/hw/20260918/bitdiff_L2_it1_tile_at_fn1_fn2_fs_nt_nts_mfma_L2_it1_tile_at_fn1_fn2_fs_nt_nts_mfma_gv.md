# Bit-diff report

| Key | Dtype | N | Differing | Max ULP | Max abs diff |
|---|---|---|---|---|---|
| `L0.B3.c_kv` | bfloat16 | 512 | 0 | 0 | 0.000 |
| `L0.B3.k_pe` | bfloat16 | 64 | 0 | 0 | 0.000 |
| `L1.B10.topk_w` | float32 | 6 | 6 | 13992 | 7.825e-05 |
| `L1.B11.expert_19` | bfloat16 | 2048 | 1163 | 32920 | 1.953e-03 |
| `L1.B11.expert_2` | bfloat16 | 2048 | 1127 | 33131 | 1.953e-03 |
| `L1.B11.expert_39` | bfloat16 | 2048 | 1028 | 32844 | 3.906e-03 |
| `L1.B11.expert_40` | bfloat16 | 2048 | 1079 | 681 | 1.953e-03 |
| `L1.B11.expert_47` | bfloat16 | 2048 | 1141 | 32906 | 9.766e-04 |
| `L1.B11.expert_49` | bfloat16 | 2048 | 968 | 590 | 1.953e-03 |
| `L1.B12.shared` | float32 | 2048 | 1594 | 3087007744 | 1.526e-04 |
| `L1.B13.layer_out` | bfloat16 | 2048 | 841 | 32500 | 1.953e-03 |
| `L1.B2.q` | bfloat16 | 3072 | 2 | 2 | 2.441e-04 |
| `L1.B3.c_kv` | bfloat16 | 512 | 0 | 0 | 0.000 |
| `L1.B3.k_pe` | bfloat16 | 64 | 0 | 0 | 0.000 |
| `L1.B4.q_pe` | bfloat16 | 1024 | 0 | 0 | 0.000 |
| `L1.B6.attn` | bfloat16 | 2048 | 18 | 6 | 1.221e-04 |
| `L1.B8.router_logits` | float32 | 64 | 64 | 181008 | 1.690e-03 |
| `L1.B9.topk_idx` | int32 | 6 | 0 | 0 | 0.000 |
| `L1.kva` | bfloat16 | 576 | 0 | 0 | 0.000 |
| `L1.norm2` | bfloat16 | 2048 | 137 | 10 | 7.812e-03 |
| `L1.ql_nope` | bfloat16 | 8192 | 4 | 2 | 9.766e-04 |

