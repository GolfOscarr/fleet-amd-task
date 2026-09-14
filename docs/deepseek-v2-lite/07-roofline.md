# 07 — Roofline and Memory Budget

Everything here is arithmetic on `01-config.md` and the verified shapes. No GPU
needed — this is the analysis that lets the design deliverable state expected
performance with a number behind it.

Reproduce with `sources/roofline.py`.

## Parameter accounting (cross-check)

| Component | Elements | Bytes |
|---|---|---|
| `embed_tokens` | 102400 × 2048 | 400 MB |
| `lm_head` | 102400 × 2048 | 400 MB |
| Attention × 27 layers | 27 × 13,762,560 | 708.75 MB |
| Dense MLP (layer 0) | 3 × 10944 × 2048 | 128.25 MB |
| MoE × 26 layers (all experts) | 26 × (gate + shared + 64 experts) | 28,275.5 MB |
| Norms | 27 × (2 × 2048 + 512) + 2048 | ~0.26 MB |
| **Total** | **15,706,484,224** | **31,412,968,448 B = 31.41 GB** |

The index file reports `total_size: 31412968448`. Our independent count from
`config.json` alone gives **exactly the same number** — including the easily
forgotten `kv_a_layernorm` (27 × 512). This is a strong check that our
understanding of the architecture matches the actual checkpoint: had we
misunderstood any projection shape or the dense/MoE split, it would not close.

Active parameters per token: **2.45 B** — consistent with the model card's
"2.4B active".

## Per-token memory traffic

| Component | MB | Share |
|---|---|---|
| Layer 0 attention | 26.25 | 0.6% |
| Layer 0 dense MLP | 128.25 | 2.7% |
| Layers 1–26 attention | 682.5 | 14.5% |
| Layers 1–26 router | 6.5 | 0.1% |
| Layers 1–26 shared experts | 858.0 | 18.2% |
| **Layers 1–26 routed experts** | **2,574.0** | **54.7%** |
| `lm_head` | 400.0 | 8.5% |
| MLA latent KV cache | 30.4 | 0.6% |
| **Total** | **4,705.9 MB** | 100% |

**Routed experts are 55% of all traffic.** Any meaningful speedup at batch 1 is
an expert-loading speedup. Shared experts plus routed experts together are 73%.

## Roofline

MI300X peak HBM bandwidth: **5.3 TB/s** (`../mi300x/01-architecture.md`,
primary).

```
4,705.9 MiB = 4,934,467,584 B
4,934,467,584 / 5.3e12 = 931.0 µs per token
                       → 1,074 tokens/second
```

For 32 output tokens: **~29.8 ms** of pure decode, excluding prefill.

### ...but 5.3 TB/s is not achievable

Nothing reaches theoretical peak. Per `../mi300x/07-achievable-bandwidth.md`,
MI300X measures **4.3 TB/s** (81%) on BabelStream, and AMD's own read-only
acceptance threshold is **3.66 TB/s** (69%). Our traffic is ~98% reads, and a
GEMV is structurally a dot product, so the Dot figure is the closest proxy.

| | @5.3 (theoretical) | @4.3 (measured) | @3.66 (conservative) |
|---|---|---|---|
| Per token | **931 µs** | **1,148 µs** | **1,348 µs** |
| Tokens/s | 1,074 | 871 | 742 |
| 32 tokens | 29.8 ms | 36.7 ms | 43.1 ms |

**Treat 931 µs as the hard floor and 1.15-1.35 ms as the realistic band.**

### How to read this number

This is a **hard lower bound under one assumption**: that every active weight is
read from HBM exactly once per token. That assumption is sound here — 4.7 GB of
per-token traffic against 32 MB of L2 and (if present) 256 MB of Infinity Cache
means weights cannot be resident between tokens. There is no reuse to exploit at
batch 1.

So:

- A real implementation achieving 60–80% of peak bandwidth lands at
  **1.2–1.6 ms/token, 630–860 tok/s**.
- Anything reporting **below ~931 µs/token is measuring something wrong** —
  a warm cache, a skipped layer, or a mis-timed window. This number is our
  sanity check on our own results.
- Conversely, landing at 1.2-1.4 ms is **success, not failure** — it is at or
  near the achievable ceiling.
- Kernel-launch overhead, which Fleet exists to remove, is *on top of* this.
  With ~800–1,000 launches per token (`04-tensor-flow.md`) at a few µs each, an
  eager baseline could plausibly spend as much time launching as streaming —
  which is precisely the gap Fleet targets.

### With the naive KV cache, for contrast

Using the HF decompressed cache (270 MB instead of 30.4 MB):

```
4,945.5 MiB = 5,185,843,200 B / 5.3e12 = 978.4 µs per token → 1,022 tok/s
```

Only **5% worse**, because at 1,024 context the cache is a small share of total
traffic either way. **The absorbed form is not primarily a bandwidth win at this
context length** — it is a *cache-residency* win, which is a different and
subtler argument (below). At 32K context the picture inverts completely: the
naive cache becomes 8.4 GB and dominates everything.

Worth being honest about this in the design: the headline 8.89× cache reduction
translates to only 5% of end-to-end traffic at S=1024.

## The L2 residency question

| Working set | Size | vs 32 MB aggregate L2 |
|---|---|---|
| MLA latent cache, all 27 layers | 30.4 MB | **fits, barely** |
| MLA latent cache, one layer | 1.125 MB | fits in a single XCD's 4 MB |
| HF decompressed cache, all layers | 270 MB | 8.4× over |
| One routed expert | 16.5 MB | 4× over one XCD's L2 |
| One MoE layer's active weights | 158.5 MB | 5× over aggregate |

Two observations that should drive the design:

1. **The whole latent KV cache fits in L2 — but only if nothing evicts it.**
   Each layer streams 158.5 MB of expert weights through the same L2 that we
   want to hold 30.4 MB of cache. Unless those weights are loaded with a
   non-temporal / streaming cache policy (`nt` / `sc` bits, see
   `../mi300x/03-memory-model.md`), they will evict the cache we are trying to
   keep. **Marking weight loads non-temporal while leaving KV-cache loads
   cached is likely the single highest-value micro-optimization available**, and
   it is cheap to try.
2. **Per-layer, the latent cache is only 1.125 MB** — comfortably inside one
   XCD's 4 MB L2. So a per-layer, per-chiplet cache shard is very plausible;
   it does not require the whole 30.4 MB to stay resident simultaneously.

Both are hypotheses from arithmetic, not measurements. Q3 in
`99-open-questions.md`.

## What this means for milestones

| Milestone | Traffic/token | @5.3 theo | @4.3 meas | @3.66 consv |
|---|---|---|---|---|
| One MoE layer (layer 1) | 159.6 MiB | **31.6 µs** | 38.9 µs | 45.7 µs |
| 4 consecutive MoE layers | 638.5 MiB | 126 µs | 156 µs | 183 µs |
| All 27 layers, no head | 4,305.9 MiB | 852 µs | 1,050 µs | 1,234 µs |
| Full decode incl. `lm_head` | 4,705.9 MiB | **931 µs** | **1,148 µs** | **1,348 µs** |

(All "MB" in these notes means MiB = 1024². The roofline divides bytes by
5.3e12 B/s.)

The single-layer milestone has a **30 µs** roofline. That is small enough that
launch overhead and synchronization cost will be clearly visible in the
measurement — which makes layer 1 a genuinely informative first target, not just
a box to tick. If our single-layer Fleet path takes 300 µs, we will know
immediately that the task-graph overhead, not bandwidth, is the problem.
