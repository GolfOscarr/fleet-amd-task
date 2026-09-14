# 99 — Open Questions

## Q1 — Does runtime reassociation avoid the MLA dequantization requirement? `resolved` (2026-09-14)

**Resolved: yes, and the checkpoint's scale layout makes it trivial.**

Read from the FP8 checkpoint's safetensors header:

```
q_proj.weight          F8_E4M3  [3072, 2048]
q_proj.weight_scale    F32      []          <- scalar, per-tensor
kv_b_proj.weight       F8_E4M3  [4096, 512]
```

The scales are **scalars**, not per-channel or per-block. So for the per-head
product our reassociated path needs:

```
q_nope[h] @ W_UK_fp8[h] * scale_kv_b
```

the scale factors straight out of the matmul. vLLM's constraint arises only
because it *materialises* `W_UQ @ W_UK` at load time, where two independently
scaled quantized matrices cannot be recombined into one meaningfully quantized
product. We never form that product.

**Residual:** confirm numerically alongside `../deepseek-v2-lite` Q1. The
fallback if something unexpected appears — keep `kv_b_proj` in BF16 — costs
54 MiB/token = **10.7 us**, about 2% of an FP8 roofline.

## Q2 — MFMA or VALU dot-product for an M=1 GEMV? `open`

**Why.** `03-kernel-craft.md` argues VALU wins at M=1 on register pressure,
which is the binding constraint on megakernel occupancy. But Fleet's own MI300
kernels use `ck_tile` MFMA templates even at bs=1, which suggests the simple
argument is missing something.

**Check.** Microbenchmark both for `[1,2048] × [2048,1408]`: time, achieved
bandwidth, and VGPR count from `-Rpass-analysis=kernel-resource-usage`.

---

## Q3 — Accuracy of the RedHatAI FP8 checkpoint on our task `open`

**Why.** Red Hat published gsm8k numbers for the *Instruct* variant via vLLM PR
#13181; nothing comparable is published for **Base**, which is our model.

**Check.** If we pursue FP8: run our 32-token greedy generation on both BF16 and
FP8 and compare token IDs, plus per-boundary errors. Note that exact token
match is a *stricter* test than gsm8k and may legitimately fail while quality is
preserved — decide in advance how we report a mismatch.

---

## Q4 — Does the FP8 checkpoint load without vLLM? `resolved` (2026-09-14)

**Resolved: yes, and it is simple.** From the safetensors header:

| Tensor class | dtype | Scale |
|---|---|---|
| Linear weights (attn, experts, shared, dense MLP) | `F8_E4M3` | `weight_scale` F32, shape `[]` |
| | | `input_scale` F32, shape `[]` |
| RMSNorm weights, `kv_a_layernorm` | `BF16` | none |
| **`mlp.gate.weight` (router)** | **`BF16`** | none |
| `embed_tokens`, `model.norm`, `lm_head` | `BF16` | none |

Per-tensor **scalar** scales — no block or group structure, so dequantisation is
a single multiply. Our loader reads `weight` as `F8_E4M3` bytes and one FP32
scalar; nothing from vLLM or compressed-tensors is needed.

Two useful specifics:

- The **router stays BF16**, which preserves the FP32-router requirement in
  `../deepseek-v2-lite/03-moe.md`. Red Hat made the same judgement we would have.
- The 110 unquantized tensors decompose exactly: 54 layer norms + 27
  `kv_a_layernorm` + 26 routers + `embed_tokens` + `model.norm` + `lm_head`.

**Correction to `01-precision.md`:** the roofline there counted the router as
quantizable. It is not, which adds 3.25 MiB/token — "FP8 as shipped" is
**508.7 us**, not 508.1. Immaterial, but recorded.

## Q5 — Is the ~100 µs split-KV estimate right? `open`

**Why.** `02-decode-parallelism.md` derives it from a crude CU-count ratio.
If the real cost of head-limited attention is much smaller, split-KV drops in
priority; if larger, it rises above some FP8 work.

**Check.** On the machine, time the attention step with and without splitting at
S=1024. Alternatively bound it from Fleet's "attention = 5% of decode time".

---

## Q6 — Non-temporal weight loads: does the KV slice survive? `open`

Inherited from `../mi300x/99-open-questions.md` Q3, now with a concrete target:
does marking weight loads `sc1=1 nt=1` keep the 144 KiB per-XCD latent slice
resident across a layer's 158.5 MiB of expert streaming? Measure `TCC_HIT` /
`TCC_REQ` with and without.
