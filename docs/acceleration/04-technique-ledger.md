# 04 — Technique Ledger

Every technique considered, its lever, its estimated value against the 931 µs
BF16 roofline, and whether it is in scope. This is the file that stops us
optimizing a 0.6% line item while a 55% one sits untouched.

Lever **A** = read fewer bytes (moves the roofline). Lever **B** = waste less
time not reading (closes the gap to it).

| Technique | Lever | Est. value | Cost | Status |
|---|---|---|---|---|
| **FP8 weights** (experts + MLP + projections) | A | **−423 µs** (931→508) | 2–3 d | **Stretch goal**; checkpoint exists |
| ... FP8 `lm_head` as well | A | −29 µs further | +0.5 d | Stretch |
| **Persistent megakernel** (Fleet) | B | eliminates ~800–1,000 launches/token | the project | **Core** |
| **Chiplet-task dispatch** (8 tasks/GEMM vs 96–256) | B | Fleet measures 1.16× at bs=1 | inherited | **Core** |
| **Split-KV attention** | B | ~100 µs (rough) | 1 d | **Required** — 16 units vs 304 CUs |
| **Runtime MLA reassociation** | A | −240 MiB/token (−47 µs) | folded into MLA task | **Decided** |
| **Non-temporal weights to protect KV slice** | B | unknown, cheap to test | hours | **High priority** |
| **Fused SiLU into gate+up** | B | Fleet: L2 hit 9.4%→17.4% at bs=1 | inherited | Core |
| **Hierarchical two-level sync** | B | `buffer_wbl2` ~24K→~300/step | inherited | Core |
| **Chiplet-parallel `lm_head` + argmax** | B | 400 MiB spread over 8 XCDs | 0.5 d | Worthwhile |
| **Weight-only (skip activation quant)** | — | simplifies FP8 by a lot | negative cost | **Decided** |
| FP8 latent KV cache | A | −15 MiB/token (−3 µs) | 1 d | Not worth it at S=1024 |
| INT4/AWQ weights | A | −1,170 µs vs BF16 in principle | high | Out — accuracy risk, no time |
| HIP Graphs | B | partial launch-overhead removal | 0.5 d | **Fallback only** if megakernel stalls |
| Materialized MLA weight fusion | A | **+996 MiB/token — worse** | — | **Rejected**, see `../deepseek-v2-lite/05-weights.md` |
| Continuous batching | — | — | — | Excluded by task |
| Speculative decoding | — | — | — | Excluded by task |
| Tensor parallelism | — | — | — | Excluded by task |
| Cooperative weight tiling (Fleet bs≥32) | A | 0 at batch 1 | — | Unavailable — needs `m_tiles ≥ 2` |
| Prefill optimization | — | — | — | Out of scope |

## Reading the ledger

**The three things that matter, in order:**

1. **FP8** is worth more than everything else combined (−423 µs, 1.83×). It is
   also the only item that changes the roofline rather than chasing it.
2. **The megakernel itself** is the task, and its value is bounded by whatever
   launch overhead actually measures — currently unknown, and Q5 in
   `../fleet/99-open-questions.md`.
3. **Split-KV** is not optional; without it attention is head-limited to ~5% of
   the device.

**The cheapest experiment** is the non-temporal cache policy — hours of work,
and it is the one place our model's structure (a 144 KiB per-XCD latent slice)
gives us something Fleet's evaluation could not exploit.

**The most important rejection** is materialized MLA weight fusion, which the
conventional "use absorbed MLA" advice would have led us straight into and which
is 2× worse than doing nothing at 1,024 context.

## Honest expectation

Stacking what is realistically in scope for five days — megakernel, chiplet
dispatch, split-KV, runtime reassociation, fused activations, non-temporal
policy — the destination is **the BF16 roofline of 931 µs, approached but not
reached**. HazyResearch's megakernel reportedly achieves 78% of peak bandwidth
at batch 1 (Fleet §7), which for us would be **~1.19 ms/token**. That is a
reasonable target to state in the design document.

FP8, if reached, moves the target to ~508 µs roofline / ~650 µs realistic. It
should be written up with the arithmetic regardless of whether we implement it,
since "recommended next steps" is a required deliverable.
