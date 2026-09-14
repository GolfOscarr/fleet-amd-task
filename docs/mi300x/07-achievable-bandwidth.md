# 07 — Achievable Bandwidth and Memory-Level Parallelism

Two questions with one answer between them: **what bandwidth can we actually
get**, and **can a megakernel at 1 wave/SIMD reach it?**

Everything else in these notes divides by 5.3 TB/s theoretical. Nothing achieves
that, so this file establishes the realistic band and propagates it.

---

## Part 1 — Achievable bandwidth

### AMD's own acceptance thresholds (primary)

From the *AMD Instinct Customer Acceptance Test Guide*, BabelStream pass
thresholds for MI300X. These are AMD's published minimum for a **healthy**
system — a floor, not a ceiling.

| Kernel | Threshold | % of 5.3 TB/s | Reads : writes |
|---|---|---|---|
| Copy | ≥ 4,177,285 MB/s ≈ 4.18 TB/s | 78.8% | 1:1 |
| Mul | ≥ 4,067,069 MB/s ≈ 4.07 TB/s | 76.7% | 1:1 |
| Add | ≥ 3,920,853 MB/s ≈ 3.92 TB/s | 74.0% | 2:1 |
| Triad | ≥ 3,885,301 MB/s ≈ 3.89 TB/s | 73.3% | 2:1 |
| **Dot** | **≥ 3,660,781 MB/s ≈ 3.66 TB/s** | **69.1%** | **2:0 — read-only** |

Run configuration AMD specifies: HIP BabelStream, `-n 50 -s 268435456`, one MPI
rank per GPU. No clock pinning, no partition-mode qualifier — AMD does not say
which SPX/NPS configuration these were measured in, and gives no basis for
adjusting them if you change it.

The percentages are ours; AMD publishes only MB/s and never mentions 5.3 TB/s.

### Independent measurement (secondary)

Ambati & Diep, *AMD MI300X GPU Performance Analysis* (arXiv:2510.27583):
MI300X reaches "approximately 81% of its theoretical peak of 5.3 TB/s",
"saturating at around 4.3 TB/s", with bandwidth plateauing at 64–128 MiB array
size. For comparison they report NVIDIA A100/H100/H200 reaching "up to 90%".

The paper does **not** say which BabelStream kernel produced that number, so
4.3 TB/s cannot be treated as read-only.

### Which figure applies to us

Our decode traffic is **~98% reads** — weights stream in, the only write is a
4 KB activation. That argues for the read-only figure.

**Dot is also structurally the closest match**: it streams two input arrays and
reduces to a scalar, which is exactly what a GEMV does per output element. It is
also the *lowest* of the five thresholds, because the reduction adds overhead
that pure streaming does not.

So we adopt a **band**, not a point:

| Label | Bandwidth | % of peak | Basis |
|---|---|---|---|
| Theoretical | 5.3 TB/s | 100% | ROCm spec (primary) |
| **Measured** | **4.3 TB/s** | **81%** | BabelStream peak, arXiv:2510.27583 |
| **Conservative** | **3.66 TB/s** | **69%** | AMD Dot acceptance threshold (primary) |

Quote the theoretical figure as the hard floor on latency, and the 4.3 / 3.66
pair as the realistic range.

### Resulting roofline band

| Workload | Traffic | @5.3 (theo) | @4.3 (meas) | @3.66 (consv) |
|---|---|---|---|---|
| **BF16 full decode** | 4,705.9 MiB | 931 µs | **1,148 µs** | **1,348 µs** |
| BF16, tokens/s | | 1,074 | **871** | **742** |
| Layer 1 (MoE) milestone | 159.6 MiB | 31.6 µs | 38.9 µs | 45.7 µs |
| 27 layers, no head | 4,305.9 MiB | 852 µs | 1,050 µs | 1,234 µs |
| FP8 as shipped | 2,571.4 MiB | 509 µs | 627 µs | 737 µs |
| FP8, tokens/s | | 1,966 | 1,595 | 1,357 |

**Realistic BF16 target: 1.15–1.35 ms per token, 740–870 tok/s.**

### Correction: the 78%-of-peak target was wrong

`../fleet/` and `../acceleration/` adopted "78% of peak bandwidth at bs=1" from
Fleet §7's description of the HazyResearch megakernel, giving ~1.19 ms/token.

That was a mistake. HazyResearch's 78% is **of theoretical peak, on H100**,
where measured peak reaches ~90% of theoretical. On MI300X, where measured peak
is 81%, hitting 78% of theoretical would mean achieving **96% of what
BabelStream itself achieves** — essentially the hardware ceiling, for a kernel
far more complex than a streaming benchmark.

A defensible target is **60–70% of theoretical**, i.e. **1.33–1.55 ms/token**,
with 1.15 ms as a stretch. Stating 1.19 ms would have been over-promising.

---

## Part 2 — Can 1 wave/SIMD saturate it?

Fleet §8 reports that compiling all task types into one kernel makes the
register footprint the union of every task's, "limit[ing] occupancy to a single
wave per SIMD, eliminating latency hiding from wave switching." For a
memory-bound workload that sounds alarming. It is not — for a specific reason.

### The hardware limit (CDNA3 ISA, primary)

> `VMCNT` — Vector memory instruction count — **6** bits — "Counts the number of
> VMEM instructions issued but not yet completed."

Six bits ⇒ **up to 63 outstanding vector-memory instructions per wave**.
(`IB_STS` Table 20 splits `VM_CNT` across bits 23:22 and 3:0 — six bits total,
consistent.)

That is the key fact: the *thread-level* parallelism we lose to 1 wave/SIMD can
be replaced by *memory-level* parallelism **inside** each wave.

### Little's Law

```
bytes_in_flight_required = bandwidth × latency
bytes_in_flight_available = waves × outstanding_loads_per_wave × bytes_per_load
```

With 8 CUs reserved for schedulers: **296 worker CUs × 4 SIMDs × 1 wave =
1,184 waves**. A `global_load_dwordx4` moves 16 B × 64 lanes = **1 KiB per wave
per instruction**, so each outstanding load across the device is **1.16 MiB** in
flight.

Required prefetch depth `N` per wave:

| Bandwidth | L=250 ns | L=500 ns | L=1 µs | L=2 µs |
|---|---|---|---|---|
| 5.3 TB/s (theo) | 1.1 | 2.2 | 4.4 | 8.7 |
| **4.3 TB/s (meas)** | 0.9 | **1.8** | **3.6** | 7.1 |
| 3.66 TB/s (consv) | 0.8 | 1.5 | 3.0 | 6.0 |

At 2 waves/SIMD every figure halves.

### Verdict: not a structural ceiling — a code-structure requirement

**We do not know MI300X's HBM latency**, and we could not find a published
figure (see below). But the conclusion is robust across the whole plausible
range:

- At any latency from 250 ns to 2 µs, the required depth is **N ≈ 1–9**.
- The hardware allows **63**.
- Inverting: at 4.3 TB/s with `VMCNT=63` fully used, 1 wave/SIMD could tolerate
  **17.8 µs** of memory latency. HBM latency is not remotely near that.

So 1 wave/SIMD does **not** prevent saturating HBM. What it requires is that the
inner loop **issue several independent loads before the first `s_waitcnt`** —
i.e. unroll by 4–8 and software-pipeline. If we write a naive
`load → waitcnt → use` loop, we will be latency-bound at roughly `1/N` of peak,
and that will look like a bandwidth problem when it is a scheduling problem.

### The register price is small

Each outstanding `dwordx4` needs 4 VGPRs for its destination:

| Prefetch depth | VGPRs | of 512 |
|---|---|---|
| 2 | 8 | 1.6% |
| 4 | 16 | 3.1% |
| **8** | **32** | **6.3%** |
| 16 | 64 | 12.5% |

Depth 8 costs 32 VGPRs — cheap in isolation. But in a megakernel the footprint
is the **union across all tasks**, so this is additive with everything else and
feeds directly into `../fleet/99-open-questions.md` Q3.

### What this changes

`MAJ-4` in `../../OPEN-PROBLEMS.md` was "occupancy is 1 wave/SIMD, no latency
hiding — is that fatal?" It is now: **prefetch depth is a first-class design
parameter, target 4–8, and budget ~32 VGPRs for it.** That is actionable, and it
should be a stated requirement on every GEMV and MLA inner loop we write.

---

## Unresolved: HBM latency

We could not find a published load-to-use latency for MI300X HBM.

- Chips and Cheese, *Testing AMD's Giant MI300X*: gives **Infinity Cache ≈ 218 ns**
  and a **TLB miss penalty of 47.1 ns**, and cross-workgroup atomic latency of
  116–202.5 ns — but **no HBM number**; DRAM latency appears only in chart images.
  It does say VRAM latency is "comparable to H100's" once TLB misses are reduced.
- *The MALL is Open* (SC'25 Workshops, doi 10.1145/3731599.3767487) almost
  certainly has it, but ACM returned HTTP 403 — not even the abstract.
- arXiv:2510.27583 contains no latency measurement at all.

**This does not block the analysis** — the verdict holds across 250 ns–2 µs, an
8× range. But it is worth measuring on the machine with a pointer-chase
microbenchmark, both to confirm and because the MALL (if the 256 MB figure is
real — `99-open-questions.md` Q13) may put a meaningfully lower latency tier
between L2 and HBM.

---

## Reproduce

`sources/bandwidth_analysis.py` regenerates every number in this file.
