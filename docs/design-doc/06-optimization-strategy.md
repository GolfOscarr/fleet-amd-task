# 06 - Optimization strategy

How the Fleet execution model is applied to this model on this GPU to make a
decode step fast, ranked by expected value against the 931.8 us floor and the
1,148-1,349 us achievable band (`01-execution-flow.md`), with the measurement
that decides each item and what happens if it goes the other way. The
ledger this extends is `docs/acceleration/04-technique-ledger.md`; the
numbers Fleet's own paper supports are in `docs/fleet/07-gap-analysis.md`.

## The frame

Batch-1 decode of this model moves 4.7 GiB per token at 0.5 FLOP per byte.
There are two levers and only two: read fewer bytes, or spend less time not
reading. Everything below is one or the other, and each item says which.

What the Fleet runtime contributes is the second lever: one kernel launch per
generation instead of roughly 800-1,000 per token, tasks dispatched on the
device, no L2 flush at kernel boundaries. Fleet's own batch-1 measurement on
Qwen3-8B is the honest reference for what to expect from it: L2 hit rate
16.4% to 16.9%, HBM reads 0.98x of the baseline, and a 1.16x speedup that
comes from dispatch, not bandwidth (`docs/fleet/07-gap-analysis.md`). The
design promises that, and states the bandwidth band as the target.

## Ranked list

| Rank | Technique | Lever | Expected value | Cost | Decided by |
|---|---|---|---|---|---|
| 1 | **Persistent megakernel with on-device task graph** (the Fleet runtime) | time | removes ~800-1,000 launches per token; the only path to the band at all | the project | M2 evidence; the launch count from `rocprofv3 --kernel-trace` |
| 2 | **Chiplet-task placement of every GEMV** (8 gang tasks per op, one per XCD, N-split) | time | inherited; Fleet measures 1.16x at bs=1 from this and rank 1 together | inherited | per-op bandwidth from event timing versus 4.3 TB/s |
| 3 | **Forced shared experts** (top-8-of-66, `00-decisions.md` D6) | time | the 99 MiB routed phase and the 33 MiB shared phase become one 132 MiB phase on 8 XCDs with none idle; removes 2 ops and 2 boundaries per layer; the largest single line item (73% of bytes) runs at full width | 0 (packing) | per-XCD busy time during the expert ops (event timing); compared against a 6-routed + separate-shared build |
| 4 | **Runtime MLA reassociation** (D4) | bytes | 240 MiB per token less than the naive cache, 47 us; and 44 MiB per layer less than materialized absorption | folded into `mla_prep` and `mla_merge_uv` | B5/B6 correctness; no measurement needed for the byte count |
| 5 | **Split-KV attention** with `BLOCK_H = 16` and 33 splits (D11) | time | attention is 0.7% of bytes but would be 16 workers wide without splits; with 33 splits it is 33 workers and off the critical path (order of 100 us per token at stake, `docs/acceleration/02-decode-parallelism.md`) | in `mla_attend` | `P_split` sweep: tiles per XCD 3, 5, 9 (splits of 64, 32, 16 positions); event timing of `mla_attend` versus `o_proj` |
| 6 | **CK FMHA instantiation for `mla_attend`** (D12) | schedule | not a speedup; it converts days of kernel writing into a wrapper, which is what protects ranks 1-5 | 0.5 d probe | day-1 probe (Q11) |
| 7 | **Non-temporal weight loads** (`USE_NT_WEIGHTS=1`, `sc1 nt`: L2 stream, MALL no-allocate) | time | the one residency experiment left after `03-synchronization.md` removed L2 from consideration: if the Infinity Cache exists and the 31.3 MiB latent cache stays in it while 4.7 GiB of weights bypass it, the 30 MiB per token of cache reads come from a faster level; at most 6 us per token, realistically the measurement is the value | one env var | `TCC_HIT`/`TCC_MISS` and per-layer `mla_attend` time with and without the flag; a second build is all it costs |
| 8 | **Fused SiLU into the gate/up GEMV** (`gang_linear_silu`, layer 0) | time | one op and 22 KiB of intermediate less; Fleet reports L2 hit 9.4% to 17.4% at bs=1 for this fusion on their model | inherited | none needed |
| 9 | **Chiplet-parallel `lm_head` and 50-way argmax** (D13) | time | 400 MiB (8.5% of bytes) spread over 8 XCDs and 296 workers instead of a single-op tail | inherited | event timing of the head versus 400 MiB / 4.3 TB/s = 97 us |
| 10 | **Two-level event counting** (inherited) | time | for our 8-task ops it cannot batch (1,838 flushes per iteration, `03-synchronization.md`); its value here is that the mechanism is correct, not that it is cheap | inherited | `MPK_DISABLE_THREADFENCE` timing-only build gives the fence cost bound |
| 11 | **Prefetch depth 4-8 per wave** (D19) | time | required to reach the band at one wave per SIMD; CK's pipelines are assumed to have it | verification | disassembly of a `linear` instantiation and of `mla_attend`: count of `global_load` between `s_waitcnt vmcnt` |
| 12 | **FP8 weights, weight-only** (D26) | bytes | the roofline moves from 932 to 509 us (1.83x); the band to 630-740 us; the checkpoint exists | 2-3 d | not scheduled; the arithmetic is the recommendation |

Items 1, 2, 8, 10 are inherited from the runtime; 3, 4, 5, 9 are design
choices in this document; 6, 7, 11 are day-1 checks; 12 is the next step
after the week.

## What each item is worth, in the units of the band

The band is 1,148 us (81% of peak) to 1,349 us (69%). The items above do not
add: 1 and 2 are what puts the design in the band at all; 3 keeps the
largest phase at full width; 4 and 5 keep attention out of the critical
path; 7 is bounded by 30 MiB of cache reads (6 us at peak). So the honest
expectation for the BF16 design is **the band itself**, with the position
inside it decided by three things the design cannot compute:

1. per-op achieved bandwidth of the CK linears at M = 1 (the 4.3 versus 3.66
   TB/s question),
2. the per-boundary latency of 326 release-acquire boundaries per iteration
   (`09-expected-performance.md`),
3. how much of `mla_attend`, `mla_prep` and the router is hidden behind the
   bandwidth-bound linears (they are on the chain, so none of it is
   overlapped; their absolute time is what counts).

## Items considered and not taken

| Item | Why not |
|---|---|
| Overlapping the shared-expert branch with routing as two concurrent ops | not expressible: the runtime's dependency model is a chain (`docs/fleet/03-runtime.md`); replaced by rank 3, which gets the same effect by construction |
| Materialized MLA absorption (`W_UQ @ W_UK`, `W_O @ W_UV`) | +44 MiB per layer, worse than doing nothing at S = 1024 (`docs/deepseek-v2-lite/05-weights.md`) |
| Pinning the latent cache in L2 | every task's acquire invalidates the XCD's L2 (`03-synchronization.md`); the runtime cannot keep it |
| VALU GEMVs instead of the CK MFMA linears | rewriting validated kernels for an ALU saving in a bandwidth-bound op (D18); reconsidered only if a linear measures under the band |
| Expert-to-XCD affinity across decode steps | with 8 active experts on 8 XCDs by mask order, an expert that repeats lands on whichever XCD its mask position gives; a warm expert is 16.5 MiB against a 4 MB L2 that is invalidated anyway; the route log will say whether repetition even occurs |
| Cooperative weight tiling (Fleet's bs >= 32 mechanism) | needs `m_tiles >= 2`; at batch 1 there is one M row |
| HIP Graphs as a cheaper launch-overhead fix | fallback for a runtime that does not build, not an optimization of one that does |
| FP8 activations, INT4, FP8 KV cache | activation quantization buys nothing at batch 1; INT4 is an accuracy risk with no time; the FP8 cache saves 15 MiB per token (3 us) |

## Measurement matrix for the strategy

| Experiment | Build or flag | Metric | Decision |
|---|---|---|---|
| E1 baseline | default | per-iteration time, bytes, launches | the reported numbers |
| E2 non-temporal weights | `USE_NT_WEIGHTS=1` | E1 metrics, `TCC_HIT`/`TCC_MISS`, `mla_attend` time | keep the flag if faster or equal |
| E3 split count | tiles per XCD 3 / 5 / 9 | `mla_attend` and `mla_merge_uv` time | pick the minimum of the sum |
| E4 fence cost | `MPK_DISABLE_THREADFENCE` (timing only) | per-iteration time delta | quantifies rank 10; informs the dispatch-overhead term |
| E5 forced versus separate shared experts | two graphs | layer time, per-XCD busy | confirms rank 3 |
| E6 `o_proj` and `qkv_a` tile size | `tile_n` 16 / 32 / 64 | op time | worker count versus per-worker granularity |

E1-E3 are scheduled (`08-milestones.md`, day 5); E4-E6 if time allows, in
that order.
