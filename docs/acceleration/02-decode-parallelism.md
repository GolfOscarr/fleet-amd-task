# 02 — Decode Parallelism

A lever-B technique, but a structural one: without it a large part of the
machine sits idle regardless of how good the kernels are.

## The problem

At batch 1 the natural parallelism in MLA attention is **16 heads × 1 query
token = 16 units**. MI300X has **304 CUs**. Sixteen workgroups cannot saturate
5.3 TB/s — they are spread over at most 16 CUs, roughly **5% of the device**.

The GEMMs do not have this problem: `q_proj` is `[3072,2048]`, and N-splitting
its output columns across 8 chiplets and then across workers within each gives
plenty of tiles. It is the **score and weighted-sum steps** — the parts that
read the KV cache — where parallelism is head-limited.

## Split-KV

Partition the 1,024 cached positions across workers. Each computes a partial
attention result over its slice, then a merge step combines them.

Per worker slice we need three values: the running **max**, the **sum of
exponentials**, and the **weighted accumulator**. The merge rescales:

```
m   = max(m_a, m_b)
s   = s_a * exp(m_a - m) + s_b * exp(m_b - m)
acc = acc_a * exp(m_a - m) + acc_b * exp(m_b - m)
```

This is the standard FlashDecoding / split-KV reduction, and it is numerically
safe provided the rescaling is done in FP32.

## Sizing for us

| Split | Units of work | Positions per unit | Latent bytes per unit |
|---|---|---|---|
| none | 16 | 1024 | 1.125 MiB |
| 8-way (one per XCD) | 128 | 128 | 144 KiB |
| 16-way | 256 | 64 | 72 KiB |

**Superseded.** The table above assumes parallelism = heads × splits. It does
not, because all 16 heads share one KV read (the MQA property) and belong in a
single block — `BLOCK_H = 16`. Parallelism therefore comes from **splits alone**,
and the sizing has to weigh partial-buffer traffic against block count.

`../mla-decode/04-our-kernel-spec.md` works this through and lands on
**`P_split` = 32** (4 per XCD, 32 positions each), not 8. For contrast, vLLM's
own heuristic would give **2** at S=1024.

## What it is worth

Per layer the latent cache read is 1.125 MiB — 0.22 µs at full bandwidth. But
if only 16 CUs participate, the achievable bandwidth is a fraction of peak, and
that step stretches by roughly the ratio of CUs used to CUs available. A crude
bound: 16/304 of the machine implies up to ~4 µs per layer instead of ~0.2 µs,
so **on the order of 100 µs per token across 27 layers** — around 10% of the
BF16 roofline, and proportionally more of an FP8 one.

That estimate is deliberately rough: per-CU bandwidth does not scale linearly
and latency hiding complicates it. The direction is what matters — this is worth
doing, and it is not optional if we want the attention step to disappear into
the noise rather than show up as a visible per-layer cost.

Fleet's own measurement is consistent: their Table 2 puts attention at **5% of
decode time** against 95% for linear ops, on a model whose attention is GQA and
whose cache is larger than ours.

## Interaction with the task graph

Split-KV turns one attention node into `N` Chiplet-tasks plus a merge CU-task —
tasks 7 and 8 in `../fleet/06-our-task-graph.md`. Fleet already has
`gang_attention_merge_mi300.cuh`, though shaped for GQA; whether it generalizes
is Q8 there.

Note that split-KV is a **reduction-dimension split (K-split)** in Fleet's
vocabulary, which the paper says "excels at bs≥32" while N-split dominates at
bs=1–16. That guidance is about *GEMM* partitioning, where N-split is available;
for attention at batch 1 there is no N dimension to split, so K-split over the
sequence is the only option. The two statements are not in conflict.
