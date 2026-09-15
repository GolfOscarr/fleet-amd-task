# 99 - Open questions at the design level

Unknowns that the design states rather than resolves, each with the
observation that settles it. Questions inherited from the discovery sets
keep their original ids; `OPEN-PROBLEMS.md` at the repository root holds the
consolidated view.

## DQ1 - Per-boundary latency `t_b` `open, narrowed` - decisive, day 3

**Why.** `09-expected-performance.md`: 326 chain boundaries per iteration,
none overlapped. At 1 us each the boundary term is 28% of the band; at 5 us
it exceeds it. No source we read measures a release-flush plus
cross-XCD-atomic plus acquire round trip on MI300X.

**First measured input (2026-09-15).** Two microbenchmarks in
`env/hw/20260915/` put a floor under `t_b`. A release store on XCD 0 with an
acquire poll on XCD 1, over 10,000 rounds, costs 703 ns one way. The fence
work itself adds 115 ns on release and 137 ns on acquire uncontended, 317 ns
on acquire with all eight XCDs participating
(`../mi300x/99-open-questions.md` Q5). Release plus hop plus acquire is
therefore about 0.9 to 1.0 us per boundary before any task-dispatch or
wake-up cost is counted.

At that rate 326 boundaries cost roughly 300 us, which is 25 to 30% of the
1,148 to 1,349 us band — the 1 us row of the sensitivity table in
`09-expected-performance.md`, reached from a lower bound rather than a
guess. The consequence is decided: the boundary-count reductions in that
file are worth doing rather than contingent. Reductions 1 and 2, the
`rmsnorm` and `moe_silu_mul` fusions, take 326 boundaries to 245 and are
justified by this number alone. Reduction 3, folding `mla_prep` in, keeps
its own gate of 3 us per boundary, which 0.9 to 1.0 us does not meet. The question stays open because these numbers
are a microbenchmark floor, not `t_b`; the layer-1 measurement below is
still the calibration, and it is what says how much dispatch and poll
latency sits on top.

**Check.** Layer 1 alone in a one-iteration graph: `(T_layer1 - T_bw - T_serial) / 12`
from `[FWD_PASS]` and the event-timing gaps across the two norm-to-gang
boundaries, which carry almost no work. If above 2 us, apply the
boundary-count reductions before M3.

## DQ2 - `mla_prep` as a single workgroup `open` - day 3

**Why.** It streams the 2 MiB `W_uk` with one workgroup, serial on the
chain, estimated 20-40 us per layer, which would be the largest term after
bandwidth (`09-expected-performance.md`).

**Check.** Event timing of `mla_prep` at layer 1. Above 5 us: move the
`W_uk` product into `mla_attend` as a per-XCD phase A or make `mla_prep` a
16-tile gang op; both are local changes (`02-task-graph.md`).

## DQ3 - CK split-KV FMHA at (576, 512) `resolved, negative` - 2026-09-14 (= `docs/fleet` Q11)

**Why.** Decides whether `mla_attend` is a wrapper or a kernel
(`00-decisions.md` D12).

**Check.** Grep the ROCm `ck_tile` headers; compile a one-file instantiation
at `kM0 = 16, kQKHeaddim = 576, kN1 = 512`; read the LDS `static_assert`.

**Answer.** Not instantiable. Both split-KV pipelines carry
`static_assert(kSubQKHeaddim <= 256, "hdim bigger than 256 is not suitable
for this pipeline!")` (`block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp:48` and
the `nwarp_sshuffle` variant), at the CK commit Fleet pins (`d8ee107a`) and
at `rocm-7.2.4`; `ck_tile/ops/fmha` has no MLA pipeline. D12's reversal
condition is met: `mla_attend` is the spec kernel already written in
`fleet/tasks/mi300/mla_attend_mi300.cuh`, and the day-1 probe
(`env/probe_ck_fmha_576_512.cpp`, `check_day1.sh` check 5) is kept only to
confirm the machine's CK says the same.

## DQ4 - What `buffer_inv sc1` invalidates for plain device memory in SPX + NPS1 `open` - day 1

**Why.** `03-synchronization.md` assumes the latent cache lines are among
the "non-coherently cached lines" the acquire invalidates, so L2 residency is
not counted on. If device memory in SPX carries an MTYPE the invalidate
leaves alone, the cache read of 1.125 MiB per layer could hit L2 across
operator boundaries, and the residency experiment changes meaning.

**Check.** `TCC_HIT`/`TCC_MISS` over a layer-1 graph where `mla_attend` runs
twice on the same rows (a debug graph with the op duplicated): a second-pass
hit rate near zero confirms the invalidate; near one refutes it.

## DQ5 - Does the Infinity Cache exist and does `nt` control its allocation `open, narrowed` - day 5

**First half answered (2026-09-15): it exists.** Pointer-chase latency on
the VM is 81 ns at a 1 MiB working set, 258 ns at 64 MiB and 342 ns at 1 GiB
(`env/hw/probes/chase.cu`, `env/hw/20260915/`), so there is a tier between
L2 and HBM, and it is no longer `secondary` in our sources. Its size is not
measured: 64 MiB hits it, 1 GiB does not, and `rocminfo` lists no L3 row
(`../mi300x/99-open-questions.md` Q13). The 30 MiB of cache reads per token
would fit. The second half, whether `nt` controls allocation into it, is
untouched and E2 below is still the experiment.

**Why.** The only residency layer left to the cache reads
(`03-synchronization.md`); its existence on MI300X is `secondary` in our
sources (`docs/mi300x/99-open-questions.md` Q13), and the runtime's comment
that `NT = 1` means MALL no-allocate is a comment.

**Check.** E2 in `06-optimization-strategy.md`: `USE_NT_WEIGHTS=1` versus
default, `mla_attend` time and TCC counters. A change in `mla_attend` time
with unchanged L2 counters points at a memory-side cache.

## DQ6 - Forced shared experts versus separate ops `open` - day 5 (= `docs/fleet` Q4)

**Why.** D6 is chosen on structure (chain model, 8 on 8); the measured
comparison is the evidence.

**Check.** E5: two graphs, layer time and per-XCD busy time.

## DQ7 - Register union after `mla_attend` `open` - day 3 (= `docs/fleet` Q3)

**Why.** One wave per SIMD is the occupancy; spills would be fatal to the
bandwidth term.

**Check.** `-Rpass-analysis=kernel-resource-usage` before and after; the
fallback is the two-pass `V` accumulation in `04-memory-plan.md`.

## DQ8 - Tile size of the small gang linears `open` - day 5

**Why.** `o_proj` and `qkv_a_proj` at `tile_n = 32` and 24 put 64 and 152
workers on 8 and 14 MiB; fewer, larger tiles mean fewer workers streaming,
more, smaller tiles mean more per-tile overhead. Unknown which side of the
knee we are on.

**Check.** E6: `tile_n` 16 / 32 / 64 for `o_proj`; event timing.

## DQ9 - Does online mode need any meta tensor we have not traced `open` - day 2

**Why.** D14 sets `step`, `num_new_tokens`, `qo_indptr_buffer`; the paged
buffers are left at zero because none of our tasks read them. The reused
kernels' emitted call code was read for `gang_linear`, `embedding`,
`rmsnorm`, `moe_*`, `argmax_*`; a runtime read of another buffer would show
as a wrong `num_active_tokens` or an out-of-range index.

**Check.** The M1 run of `qkv_a_proj`: output non-zero and equal to B2. If
zero, `qo_indptr_buffer[1]` was not seen; if garbage, look for a paged read.

## DQ10 - `argmax_partial` slice count `open` - day 5

**Why.** 50 tasks of 2,048 elements is a guess between too many tiny tasks
and too little parallelism on a 200 KiB read.

**Check.** 25 / 50 / 100 slices; head time from event timing. Cheap, last.

## Resolved into decisions during the design

| Question | Now |
|---|---|
| shared-expert branch overlapping the router | not expressible; D6 |
| `P_split` = 32 from a traffic model (`OPEN-PROBLEMS.md` MIN-23) | 33 splits of 32 positions over `S_max`; the sweep is E3 |
| top-6 over 8 XCDs leaves two idle (MIN-11) | D6 puts 8 active experts on 8 XCDs; measured by DQ6 |
| MFMA versus VALU at M = 1 (MIN-14) | D18: reuse the CK MFMA linears; VALU only if a linear measures under the band |
| how per-token latency is measured under one launch | `[FWD_PASS]` and the event-timing buffer (D23) |
| how the prefill hands over (`docs/deepseek-v2-lite` Q8 recipe) | D15, D16: capture from module outputs, boundary at 1023 |
| stock router correctness (`docs/fleet` Q13) | D7: fused FP32 router replaces it |
