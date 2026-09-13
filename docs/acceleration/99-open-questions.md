# 99 — Open Questions

## Q1 — Does runtime reassociation avoid the MLA dequantization requirement? `open`

**Why.** vLLM must dequantize `q_proj`, `kv_b_proj`, and `o_proj` for MLA
because it *materializes* the absorbed weight products at load time. We chose
runtime reassociation, so we never form a product — each matrix should be able
to stay quantized and be dequantized per-use in-kernel. This is a structural
argument, not a tested one.

**Check.** Implement the reassociated attention against the FP8 checkpoint on
CPU in PyTorch, compare to the BF16 oracle per
`../deepseek-v2-lite/08-correctness.md`. **No GPU needed.**

**Fallback cost if wrong:** keep `kv_b_proj` in BF16 — 54 MiB/token, **10.7 µs**,
about 2% of an FP8 roofline. Not a blocker either way.

---

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

## Q4 — Does the FP8 checkpoint load without vLLM? `open`

**Why.** It ships `weight_scale` and `input_scale` per tensor in
compressed-tensors/fp8 format. Our loader must parse that itself.

**Check.** Read the scales from safetensors, dequantize one tensor, compare
against the BF16 checkpoint's corresponding tensor. **No GPU needed.**

---

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
