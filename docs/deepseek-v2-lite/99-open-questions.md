# 99 — Open Questions

Claims our design depends on that are not settled by the checkpoint, with the
check that settles each. Status: `open` | `resolved` | `blocked`.

Unlike `../mi300x/99-open-questions.md`, most of these can be answered
**without a GPU** — they are CPU reference runs and arithmetic. Do them locally
before MI300X time is on the clock.

---

## Q1 — Does runtime reassociation stay within tolerance? `open`

**Why.** `02-mla.md` chooses runtime reassociation of MLA. It is algebraically
exact but reorders BF16 accumulation. If the error exceeds our B5/B6 thresholds,
the whole cache-layout decision needs revisiting.

**Check.** Implement both the naive and reassociated attention in NumPy/PyTorch
on CPU, run on the reference's real layer-1 inputs, and compare against the HF
oracle. Sweep the accumulation precision (BF16 vs FP32) for the per-head
`q_nope @ W_UK` product. **No GPU needed.**

---

## Q2 — Confirm the absorption cost analysis `resolved` (2026-09-13)

**Resolved.** Materialized weight fusion is nearly 2× worse than doing nothing
at S=1024 (1,927 MB vs 979 MB of attention traffic per token); runtime
reassociation is best at 739 MB. Full table in `05-weights.md`. A first draft of
these notes claimed materialized fusion cost only +20 MB/layer; the correct
figure is **+44 MB/layer**, and the conclusion reverses. Arithmetic is in
`sources/roofline.py`.

**Residual item.** Find the context length at which materialized fusion breaks
even, and state it in the design spec — it is the honest way to present "we did
not use the conventional absorbed form".

---

## Q3 — Can the latent KV cache actually stay resident in L2? `open`

**Why.** `07-roofline.md` observes that the 30.4 MB latent cache nearly fits the
32 MB aggregate L2, and that per layer it is only 1.125 MB against one XCD's
4 MB. If true in practice, this is the core Fleet win for this model. If the
158.5 MB/layer of streaming expert weights evicts it, the observation is
worthless.

**Check.** On the machine: measure L2 hit rate (`TCC_HIT`/`TCC_REQ`, see
`../mi300x/06-profiling.md`) for cache reads with expert weights loaded (a)
normally and (b) with non-temporal cache-policy bits. The delta is the answer.
**Needs GPU.**

---

## Q4 — Is expert routing temporally correlated across decode steps? `open`

**Why.** Static expert→XCD affinity only pays if a re-selected expert is still
warm. Also tells us whether routing is skewed enough that some experts dominate.

**Check.** Run the HF reference on the real 1,024-token prompt, log the 6 chosen
expert indices for all 26 MoE layers × 32 decode steps, and compute: per-layer
expert frequency distribution, and step-to-step overlap (how many of step `t`'s
6 experts were also chosen at `t-1`). **No GPU needed.** Falls out of the
`08-correctness.md` B9 logging for free.

---

## Q5 — What is the reference's actual kernel-launch count per token? `open`

**Why.** `04-tensor-flow.md` estimates 800–1,000 launches/token statically. It
frames the headline "GPU launches" metric.

**Check.** `rocprofv3 --kernel-trace` on 32 decode steps of the HF reference;
divide. One run. **Needs GPU**, but trivially cheap.

---

## Q6 — Calibrate the BF16 noise floor `open`

**Why.** The thresholds in `08-correctness.md` are reasoned, not measured. A
threshold that is too tight wastes days chasing normal BF16 behaviour; too loose
hides real bugs.

**Check.** Run the HF reference twice with different accumulation orders (CPU vs
GPU, or different batch groupings) and measure per-boundary error. Set our
thresholds at a small multiple. **No GPU needed** for the CPU-vs-CPU variant.

---

## Q7 — Are the weight shards downloadable without gating? `open`

**Why.** `config.json` and the safetensors index fetched anonymously, but weight
shards are sometimes gated separately. Blocking discovery on day 1 of GPU access
would be expensive.

**Check.** `huggingface-cli download deepseek-ai/DeepSeek-Coder-V2-Lite-Base
--include "*.safetensors"` — or just an anonymous HTTP range request on a shard,
which we already did successfully for shard 1's header. That partial success is
good evidence but not proof for the full file.

---

## Q8 — Which prompt? `open`

**Why.** The task fixes 1,024 input tokens but not their content. Routing
behaviour, and therefore everything in Q4, depends on the prompt. A code prompt
is the natural choice for a Coder model.

**Check.** Pick one, tokenize to exactly 1,024 tokens, commit the **token IDs**,
and use it everywhere. Decide once and never change it, or every measurement
becomes incomparable.

---

## Q9 — Does `lm_head` argmax need full logits? `open`

**Why.** `lm_head` is 400 MB/token, 8.5% of traffic, and greedy decoding needs
only the **argmax**, not the 102,400 logits. A chiplet-parallel partial-argmax
reduction (`04-tensor-flow.md`) avoids materializing the logit vector.

**Check.** Arithmetic plus an implementation decision — but note the correctness
protocol wants B15 logits for comparison. Keep a debug path that materializes
them and a fast path that does not. **No GPU needed to decide.**
