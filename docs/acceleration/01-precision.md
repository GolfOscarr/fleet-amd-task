# 01 — Precision

The only lever that moves the roofline. Everything else closes the gap to it.

## gfx942 has full FP8 — verified from the ISA

Grepped directly from `../mi300x/sources/cdna3-isa.pdf`:

**Matrix instructions** (dense):
```
V_MFMA_F32_16X16X32_{FP8,BF8}_{FP8,BF8}    opcodes 112-115
V_MFMA_F32_32X32X16_{FP8,BF8}_{FP8,BF8}    opcodes 116-119
```
and sparse variants `V_SMFMAC_F32_{16X16X64,32X32X32}_*` (opcodes 120-123).

**Conversion instructions**:
```
V_CVT_PK_FP8_F32   V_CVT_PK_BF8_F32     pack FP32 -> FP8/BF8, RNE
V_CVT_SR_FP8_F32   V_CVT_SR_BF8_F32     stochastic rounding variants
V_CVT_F32_FP8      V_CVT_F32_BF8        unpack
V_CVT_PK_F32_FP8   V_CVT_PK_F32_BF8
```

"FP8" is E4M3 and "BF8" is E5M2 in AMD's naming. The ROCm microarchitecture
table gives Matrix FP8 at 4096 FLOPS/clock/CU, 2,614.9 peak TFLOPS — but for
batch-1 decode the compute rate is irrelevant. **What matters is that a weight
byte halves.**

### The gfx942 / gfx950 incompatibility

Per `../mi300x/`, the LLVM generic target `gfx9-4-generic` spans gfx942 and
gfx950 but explicitly excludes "FP8 and BF8 instructions, FP8 and BF8 conversion
instructions, as well as instructions with XF32 format support". A generic
target can only contain the common subset, so **FP8 differs between CDNA 3 and
CDNA 4**. Any FP8 kernel written for gfx950 — including anything recent in
AITER — must be assumed not to port to us until proven.

## An FP8 checkpoint for our exact model exists

`RedHatAI/DeepSeek-Coder-V2-Lite-Base-FP8` — the **Base** model, not Instruct.

| | |
|---|---|
| `quant_method` | `fp8` |
| `activation_scheme` | `static` (W8A8) |
| `ignored_layers` | `["lm_head"]` |
| Total size | 16,129,490,408 B = **16.13 GB** (0.513× the BF16 31.41 GB) |
| Tensors | 15,653 = 5,291 weights + 5,181 `weight_scale` + 5,181 `input_scale` |

Per-tensor scales (5,181 = 5,291 minus the 110 norm/embedding tensors that stay
BF16). **Every linear layer is quantized, including all four attention
projections.** Verified from the checkpoint's own `config.json` and
`model.safetensors.index.json`, both in `sources/`.

Sibling checkpoints exist for Instruct (`RedHatAI/...-Instruct-FP8`, 124k
downloads) and AWQ (`TechxGenus/DeepSeek-Coder-V2-Lite-Base-AWQ`).

**This removes the main practical obstacle.** We do not need to run
llm-compressor ourselves, and we inherit whatever validation Red Hat did.

## The MLA / quantization interaction

vLLM PR #13181 ("Expand MLA to support most types of quantization", merged Feb
2025) opens with:

> "For MLA, aside from the FP8 case, we need to have access to the unquantized
> weights for the decode kernel."

Before that PR, vLLM refused MLA unless `q_proj`, `kv_b_proj`, and `o_proj` were
unquantized or FP8, erroring with "Only FP8 and UnquantizedLinearMethod are
supported for MLA, please run with `VLLM_MLA_DISABLE=1`". The PR generalizes by
**dequantizing at load time** — running an identity matrix through
`layer.quant_method.apply()` to recover transposed unquantized weights, noted in
the code as `O(N^3)` and "should only be used offline".

The PR asserts the requirement but never explains the mechanism. The reason is
visible from our own analysis: **vLLM materializes the absorbed weights**
(`W_UQ' = W_UQ @ W_UK`, `W_O' = W_O @ blockdiag(W_UV)`) at load time. You cannot
multiply two separately-quantized matrices and get a meaningful quantized
product, so both must be dequantized first.

**We chose runtime reassociation instead** (`../deepseek-v2-lite/02-mla.md`),
for bandwidth reasons entirely unrelated to quantization. That choice appears to
sidestep this constraint: we never form a weight product, so each matrix can
stay in its own quantized form and be dequantized per-use inside the kernel.
**Unverified** — it is a structural argument, not a tested one, and it is Q1
below.

The fallback costs almost nothing anyway: keeping `kv_b_proj` in BF16 across all
27 layers is 54 MiB/token of extra traffic = **10.7 µs**, about 2% of an FP8
roofline. Worth knowing, not worth worrying about.

## Scenarios

All computed from `config.json`; reproduce with `sources/fp8_roofline.py`.

All TPOT figures below are at **theoretical** 5.3 TB/s — the hard floor. At
the realistic 4.3/3.66 TB/s band, multiply by 1.23 / 1.45
(`../mi300x/07-achievable-bandwidth.md`).

| Scenario | Traffic/token | TPOT | tok/s | vs BF16 |
|---|---|---|---|---|
| BF16 everywhere (task baseline) | 4,705.9 MiB | 931.0 µs | 1,074 | 1.00× |
| **RedHatAI FP8 as shipped** (`lm_head` + router BF16) | 2,571.4 MiB | 508.7 µs | 1,966 | **1.83×** |
| ... + `kv_b_proj` dequantized (vLLM behaviour) | 2,622.1 MiB | 518.8 µs | 1,928 | 1.79× |
| ... + `lm_head` also FP8 | 2,422.1 MiB | 479.2 µs | 2,087 | 1.94× |
| FP8 weights + FP8 latent KV cache | 2,406.9 MiB | 476.2 µs | 2,100 | 1.96× |

Per-component, BF16 → FP8:

| Component | BF16 MiB | FP8 MiB | Saved |
|---|---|---|---|
| routed experts (26 layers × 6) | 2,574.0 | 1,287.0 | **1,287.0** |
| shared experts | 858.0 | 429.0 | 429.0 |
| `lm_head` | 400.0 | 200.0 | 200.0 |
| attn `q_proj` | 324.0 | 162.0 | 162.0 |
| attn `o_proj` | 216.0 | 108.0 | 108.0 |
| dense MLP (layer 0) | 128.25 | 64.1 | 64.1 |
| attn `kv_b_proj` | 108.0 | 54.0 | 54.0 |
| attn `kv_a_proj` | 60.75 | 30.4 | 30.4 |
| router | 6.5 | 3.25 | 3.25 |

**Quantizing only the experts** (routed + shared) captures 1,716 of the 2,338
MiB available — **73% of the benefit from one change**, and experts are the
part with no MLA subtlety at all. That is the obvious first increment.

Quantizing the **KV cache** saves only 15 MiB/token at S=1024 — not worth the
accuracy risk here, though it dominates at long context.

## W8A8 vs weight-only — a simplification worth taking

The shipped checkpoint is `activation_scheme: static`, i.e. W8A8. But **for a
memory-bound batch-1 GEMV, activation quantization buys nothing**: the
activation is a single 2048-wide vector (4 KB) against megabytes of weights.

So we can ignore the `input_scale` tensors entirely and run **weight-only FP8**:
load FP8 weights, dequantize in-register during the GEMV, accumulate in FP32.
This gets the full bandwidth win, avoids activation-scale plumbing, avoids
FP8 MFMA entirely (useless at M=1 anyway — see `03-kernel-craft.md`), and is
strictly more accurate than W8A8.

The cost is that we cannot use the FP8 matrix instructions. At batch 1 that
costs nothing we can measure.

## Sequencing against the task

The task says "Precision: **BF16 initially**". That wording invites FP8 as a
follow-on, but BF16 must work first, and every boundary in
`../deepseek-v2-lite/08-correctness.md` must be validated in BF16 before a
second precision is introduced. Given five days and MLA still to write, FP8 is
realistically a **documented next step with the arithmetic attached**, not
something we ship — unless the BF16 path lands early.

Recommended posture: design the weight-loading path so precision is a parameter
from the start (a per-tensor dtype tag + optional scale pointer), so FP8 is a
loader change plus a GEMV variant rather than a rewrite. That costs little now
and keeps the option open.
