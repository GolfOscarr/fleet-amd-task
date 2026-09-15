# 99 — Open Questions

Claims our design depends on that are not settled by the checkpoint, with the
check that settles each. Status: `open` | `resolved` | `blocked`.

Unlike `../mi300x/99-open-questions.md`, most of these can be answered
**without a GPU** — they are CPU reference runs and arithmetic. Do them locally
before MI300X time is on the clock.

---

## Q1 — Does runtime reassociation stay within tolerance? `resolved` (2026-09-14)

**Resolved: yes, and the attention floor is set by score magnitude, not by
reassociation.** `harness/reassoc_check.py` at the real attention shapes on
CPU (random weights, 1,024 positions, three seeds): B5 `rel_err` 3.2e-3; B6
2.0e-2 to 2.7e-2 at score magnitude 37 and 3.8e-2 to 7.3e-2 at 140, while
the reference's own arithmetic recomputed in another summation order sits at
2.5e-2 to 3.0e-2 and 4.8e-2 to 8.7e-2 on the same inputs, because the
reference rounds its scores to BF16 before the softmax. The threshold for B6
is therefore the calibrated floor (Q6), not a fixed number. Results in
`harness/results/reassoc_check.json`. The original entry follows.

**Why.** `02-mla.md` chooses runtime reassociation of MLA. It is algebraically
exact but reorders BF16 accumulation. If the error exceeds our B5/B6 thresholds,
the whole cache-layout decision needs revisiting.

**Check.** Implement both the naive and reassociated attention in NumPy/PyTorch
on CPU, run on the reference's real layer-1 inputs, and compare against the HF
oracle. Sweep the accumulation precision (BF16 vs FP32) for the per-head
`q_nope @ W_UK` product. **No GPU needed.**

```python
# W_UK = kv_b_proj.view(16, 256, 512)[:, :128, :]
naive = (q @ k.transpose(-1, -2)) * softmax_scale          # decompressed path
reassoc = (torch.einsum('hd,hdc->hc', q_nope, W_UK) @ c_KV.T
           + q_pe @ k_pe.T) * softmax_scale                 # latent path
rel = (naive - reassoc).norm() / naive.norm()
```
Accumulate the einsum in FP32; report `rel` against the B5 threshold.

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

## Q3 — Can the latent KV cache actually stay resident in L2? `open, narrowed` (2026-09-15)

**Narrowed: L2 is almost certainly not the tier that holds it, but a tier
that could hold it does exist.** Two results from `env/hw/20260915/` bear on this.
First, every task's acquire lowers to `buffer_inv sc1` on this hipcc
(`../mi300x/99-open-questions.md` Q4), and each operator boundary executes
one, so L2 lines are invalidated between operators and residency across a
layer is not something to count on; what survives depends on what that
invalidate actually touches for plain device memory, which is DQ4 in
`../design-doc/99-open-questions.md` and still open. Second, pointer-chase
latency shows a tier between L2 and HBM: 81 ns at 1 MiB, 258 ns at 64 MiB,
342 ns at 1 GiB (`../mi300x/99-open-questions.md` Q13). The 30.4 MB latent
cache is smaller than the 64 MiB working set that hit that tier, so it would
fit, and being memory-side it is the one level the acquire is not expected
to invalidate (`../design-doc/03-synchronization.md`). The tier's capacity
is not measured, so "would fit" is an inference from one data point.

So the question is unchanged in form but has moved one level down the
hierarchy: the win, if there is one, is the memory-side cache rather than
L2, and the experiment that decides it is the non-temporal weight-load
comparison (DQ5, `OPEN-PROBLEMS.md` MAJ-6) rather than an L2 hit rate alone.
The check below is still the right check and still needs the GPU. The
original question follows.

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

**Check.** Hook every `MoEGate`, log `topk_idx` for all 26 MoE layers x 32
decode steps, then compute the per-layer expert histogram and the step-to-step
overlap:

```python
overlap = [len(set(idx[t]) & set(idx[t-1])) / 6 for t in range(1, 32)]
```

Uniform routing gives mean overlap ~ 6/64 = **0.094**. If the measured mean is
materially higher, expert->XCD affinity is worth building; if not, drop the idea
and say so. **No GPU needed** — falls out of the `08-correctness.md` B9 logging
for free.

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
GPU, or a reordered expert summation) and measure per-boundary error. Set our
thresholds at **3-5x** the observed floor. Commit the calibration output — it is
what makes the thresholds defensible rather than arbitrary. **No GPU needed**
for the CPU-vs-CPU variant.

---

## Q7 — Are the weight shards downloadable without gating? `resolved` (2026-09-14)

**Resolved.** An anonymous HTTP range request on
`model-00001-of-000004.safetensors` returns **HTTP 206** with content. No token,
no gating, no license acceptance. The full 31 GB download on the MI300X host
needs nothing beyond network access.

---

## Q8 — Which prompt? `resolved` (2026-09-14)

**Resolved:** `harness/prompt_ids.json` holds BOS (100000) followed by the
first 1,023 tokens of `repos/fleet-chiplet-megakernel/python/mirage/mpk/split_linear_tasks.py`
at submodule commit `51dce4f`, tokenized with the checkpoint's tokenizer
(sha256 of `tokenizer.json` in `harness/prompt_meta.json`). BOS is prepended
explicitly because transformers versions differ on whether the fast
tokenizer honours `add_bos_token`. `harness/make_prompt.py` recomputes and
compares by default and refuses to overwrite. The original recipe follows.

**Why.** The task fixes 1,024 input tokens but not their content. Routing
behaviour, and therefore everything in Q4, depends on the prompt. A code prompt
is the natural choice for a Coder model.

**Check / recipe.** Pick a real source file (this is a Coder model), tokenize,
slice to exactly 1,024, and commit the **IDs**, not the text:

```python
ids = tok(open("fixture.py").read())["input_ids"][:1024]
assert len(ids) == 1024
json.dump(ids, open("bench/prompt_1024.json", "w"))
```

Committing IDs rather than text means a tokenizer version change cannot silently
alter the experiment. A source file from this repo works and is self-documenting.
Decide once and never change it, or every measurement becomes incomparable.

---

## Q9 — Does `lm_head` argmax need full logits? `resolved` (2026-09-14)

**Resolved: do the chiplet-parallel argmax, but for parallelism, not traffic.**

Skipping the logit buffer saves almost nothing: `[1, 102400]` in BF16 is 200 KB,
so writing and re-reading it costs ~0.08 µs. The 400 MiB of `lm_head` weights
must be read either way.

The real reason to split is **device utilisation**: partition the 102,400 rows
across 8 XCDs, argmax locally, then reduce 8 (value, index) pairs. That spreads
the single largest per-token read across the whole machine instead of
concentrating it.

Keep a debug path that materialises full logits, since the B15 boundary check in
`08-correctness.md` compares them.
