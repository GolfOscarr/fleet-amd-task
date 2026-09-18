# 10 - Results of round 4

Filled from the record of 2026-09-18 (`env/hw/20260918/runs/`, the session
in `09-session-log.md`). Every number is from the runtime's event clock
(the median over the iterations after the first) unless marked `FWD_PASS`
(the megakernel's own per-iteration report, the it29 rows without the
event timing). The baseline is round 3's 4,571 to 4,600 us per token
(`../03-acceleration/08-results.md`); the target is the production vLLM
figure quoted for this model on this machine, 4.5 ms per token.

## The number

| Configuration | Run | us per token | ids |
|---|---|---|---|
| round 3's number (its record) | `../03-acceleration`, `L27_head_it30/31/32_tile_at_fn1_fn2_fs_nt_mfma` | 4,571 to 4,600; `FWD_PASS` 4,584 to 4,590 | PASS |
| the round-4 stack: the GEMV linears at 48 tasks, the half merge, the CK w2 under the define, `--nt-streams` (S10) | `L27_head_it30/31/32_tile_at_fn1_fn2_fs_nt_nts_mfma_rf_w2cktile_gv_lg48_mt_mh2` | 4,287.5, 4,325.3, 4,322.5; rerun 4,305.9, 4,339.7, 4,307.9 | PASS |
| the same, `FWD_PASS` clock | `L27_head_it29_..._gv_lg48_mt_mh2` | 4,305; rerun 4,310 | 28 of 28 |
| the same at 96 tasks (the grid A/B) | `L27_head_it30/31/32_..._gv_mt_mh2` | 4,458.0, 4,427.8, 4,429.9; `FWD_PASS` 4,289 | PASS |
| plus `MPK_NO_LOCAL_CAS` (S14) | `..._rf_w2cktile+nolocalcas_gv_lg48_mt_mh2` | 4,261.8, 4,341.3, 4,289.2; `FWD_PASS` 4,303 | PASS |
| plus `MPK_POLL_SLEEP=8`: **the round's number** | `..._rf_w2cktile+nolocalcas+pollsleep8_gv_lg48_mt_mh2` | **4,307.0, 4,265.0, 4,293.5; `FWD_PASS` 4,267** | PASS |
| plus `GEMV_BATCH=4` | `..._rf_w2cktile+nolocalcas+pollsleep8+gemvbatch4_gv_lg48_mt_mh2` | 4,323.4, 4,300.0, 4,291.6; `FWD_PASS` 4,289 | PASS |

5 to 7% faster than round 3 on the event clock over the fifteen 48-task
runs (4,262 to 4,341 us; the medians 4,306 against round 3's 4,586, 6%),
3.5 to 5.3% below the 4.5 ms target; `FWD_PASS` 4,267 to 4,310. The two
runtime knobs and the batch constant are within the run-to-run spread on
the model (their 2-layer gains of 1 to 7% do not carry), so the round's
number is the 48-task stack, with or without the knobs. The 96-task set is
3% slower on the event clock and equal on `FWD_PASS`; the event clock is
the plan's and the 48-task grid stays in the stack.

Correctness of the number's configuration: the 32 output ids match the
reference on every final; the step-0 boundaries of the 2-layer graph pass
the compare with every lever (19 boundaries and the route log, `L2_it1_*`
rows); on the model the it32 compare rows fail `head.B15.logits` (the
boundary is captured after the reference's step) and the route log (one
tail-expert flip at step 0, MoE layer 4, then 53 to 67 of 832 pairs over
32 steps), exactly as round 3's own it32 final does in its record (0.836
and 61 mismatches). The flips are MIN-32 (a tie rule the compare lacks);
the header's changed summation order moves the step-0 flip into layer 4.

## Where the time goes, per token (27 layers with the head)

The event gaps summed over the layers, the round's number's it31 run
(4,265 us) beside round 3's last table (the worker-timing run, 4,713;
its non-timing equivalent 4,590).

| Operator | round 3, us (per layer) | round 4, us (per layer) | what changed |
|---|---|---|---|
| w13 (gang, 92 MB) | 1,104 (42.5) | 1,102 (42.4) | untouched: the stock CK gang kernel (the GEMV form measured 36.6 and stays off, below) |
| w2 (gang, 33 MB) | 621 (23.9) | 602 (23.2) | round 3's CK multiply, restored under the define |
| qkva, o_proj, the head, layer 0's down (per-tile) | 831 (14.8 each) | 638 (the GEMV rows 588: qkva 13.5, o_proj 8.1 at 48 tasks; the stock down 50) | **the batch-1 GEMV linear** (layer 0's dense down stays the CK tile, its K past the kernel's bound) |
| router | 516 (19.8) | 424 (16.3) | **the deeper router** (the header) |
| merge | 489 (18.1) | 406 (15.0) | **the merge as regular tasks, two halves per head** |
| attention | 338 (12.5) | 337 (12.5) | unchanged |
| mla_prep | 287 (10.6) | 265 (9.8) | unchanged |
| iteration start | 181 | 192 | the prelaunch of the iteration's tasks |
| combine (`mul_sum_add`) | 147 | 130 | unchanged |
| the head's chunks (49 events of 8 tasks) | 141 | 111 | unchanged (2.3 us per event) |
| embed, layer 0's dense silu linear, the argmax, the last norm | 58 | 62 | unchanged |
| **total (the sum of the gaps; the median per iteration 4,265)** | **4,713** | **4,268** | |

By family: the CK gang linears (w13, w2) 1.7 ms; our GEMV linears 0.6 ms;
the single-task and per-head kernels (router, merge, prep, attention)
1.4 ms; the iteration start, the combine, the head's chunks and the small
operators 0.5 ms. The bandwidth view: the stream probe (below) puts the
megakernel's task-shaped reads at 2.25 TB/s device-wide, and w13 sits on
that line.

## The kernels, standalone (`ktime`, the suite binaries, 50 launches)

The weight rotated over 27 copies (`KT_COLD`) for the GEMV forms and the
router; the merge and the fold warm.

| Kernel | plain | `nt` (batch 8) | `nt_b4` | `nt_b16` | `nt_strided` |
|---|---|---|---|---|---|
| `linear_gemv_norm` (qkva: 96 tasks of 38 rows) | 10.76 | 9.39 | **8.71** | 16.28 | 13.24 |
| `linear_gemv_res` (o_proj: 64 of 32) | 6.17 | **5.86** | 6.72 | 12.50 | 8.58 |
| `moe_router` | 11.53 | 11.46 | 12.57 | **11.29** | 14.71 |
| `gang_w13_gemv` (37 x 8 tiles of 304 KB) | 24.53 | 23.92 | | | |
| `gang_w2_gemv` (32 x 8) | 21.67 | 18.82 | | | |
| `mla_merge_uv` (round 3: 11.4 to 11.5) | 16.47 | 16.26 | 16.31 | 16.30 | 16.29 |
| `mla_merge_oproj` | 39.05 | 39.25 | 39.27 | 39.19 | 39.30 |
| `mla_attend` (MFMA build not run; the VALU grid) | 34.05 | 33.81 | | | |

The streaming loads beat the plain ones on every kernel standalone (the
norm form by 13%, w2 by 13%). Batch 16 spills; the strided map loses by
40%. The deeper merge is 5 us slower standalone than round 3's but equal
in the graph (22.0 us gap both rounds); the fold's 39 us warm foretold its
42.5 us gap. The 5 us was read offline in round 5 (F6 of
`../05-final/04-checklist.md`, the two launchers' `k_mla_merge_uv`
disassembled side by side): not the round trips (three dependent HBM
trips against round 3's four) but the weights phase, where every thread
walks the live splits through LDS twice with a dependent `expf` per step
(the maximum and the total, 66 steps at 33 splits) and once more per row
of the batch, against round 3's one weight per lane in a single wave with
two wave reductions; with the halving butterfly's dependent shuffle chain
per `W_uv` batch and 518 conditional selects against 14. A known
regression of the standalone form that the graph does not pay.

## The ceiling (the stream probe, G5)

| Row | Bytes per operator | GB/s | us per operator |
|---|---|---|---|
| gang shape, 37 tiles of 304 KB per XCD, `--nt-streams` | 87.9 MiB | 2,242 | 41 |
| the same, plain loads | 87.9 MiB | 2,257 | 41 |
| 96 tasks of 152 KB (qkva's shape) | 14.3 MiB | 775 | 19 |
| 296 tasks of 256 KB | 74.0 MiB | 1,283 | 60 |

The megakernel's task-shaped reads reach 42% of the 5.3 TB/s peak in the
gang shape and less in the regular shapes: the bytes in flight per CU (one
wave per SIMD, 32 loads of 16 bytes per lane) are a quarter of what
Little's law asks at 5.3 TB/s. The stock w13 at 42.4 us and the GEMV form
at 36.6 both sit on the gang line; nothing in this round's forms moves it.

## The levers, one by one

| Lever | Row | Verdict |
|---|---|---|
| the GEMV linear (`--gemv-linears`) | G1, G2 | **on**: o_proj 13.0 to 8.2 us, qkva 14.0 to 11.2 on the 2-layer graph; 831 to 588 us per token; the grid at 48 tasks 4% better per 2-layer iteration and 3% on the event clock of the finals |
| `--nt-streams` on the GEMV rows | G1.3 | **kept**: 1.3% better than plain loads with the GEMV linears (the CK build prefers plain by 2%) |
| the batch constant | G0, queue-g3, g4 | **8**: batch 4 wins the 2-layer graph by 7% (463 against 499 us) and nothing on the model (4,292 to 4,323 against 4,265 to 4,307) |
| the w2 GEMV form (the header's default) | G1, queue-f7 | **off**: 34.4 us against the CK multiply's 21.4; round 3's CK file restored as the define's path |
| w13 in one round (`--gemv-w13`) | G4 | **off** by the threshold: 36.6 us against the stock's 40.7 and the 33 us line, the iteration unchanged |
| the deeper router (the header) | G1 to G4 | **on**: 25.2 to 14.3 to 16.7 us per layer on the 2-layer graph; 516 to 424 per token |
| the four-task router (`--router-tasks`) | H2 | **off**: 13.9 against 15.5 us, the iteration 2% slower; the last task's serial top-k |
| the deeper merge (the header) | G1 | no change in the graph (22.0 us both rounds; 5 us slower standalone) |
| the merge as regular tasks (`--merge-tasks --merge-halves 2`) | H4 | **on**: 22.0 to 15.0 us per layer; 489 to 406 per token |
| the merge with o_proj folded in (`--merge-oproj`) | H5 | **off**: 42.5 us against 21.9; the last-task reduction costs more than the boundary it removes |
| the head at 320 tasks (`--head-grid 320`) | G6 | **off**: 661 against 651 us per 2-layer iteration |
| `MPK_NO_LOCAL_CAS`, `MPK_POLL_SLEEP=8` | G9, S14 | every knob's compare PASS; -2.3% and -1.3% at 2 layers, within the spread on the model; `NO_BCAST_CAS` 0, `POLL_SLEEP=32` +2.4%, `=127` +15% (MIN-33 closed) |

## Numerics (G8 and the compares)

The bit-diff of the CK-linear and GEMV-linear step-0 runs
(`env/hw/20260918/bitdiff_L2_it1_..._L2_it1_..._gv.md`): `q` differs in 2
of 3,072 elements by 2 ULP, `attn` in 18 of 2,048 by 6 ULP, `norm2` in 137
by 10 ULP, the router logits in all 64 by up to 1.7e-3 absolute, the
top-k ids in none; the expert outputs and `layer_out` differ in about half
their elements by one BF16 ULP (the accumulation order). Every 2-layer
compare row passed on 19 boundaries and the route log with every lever
(the fold's `x_res` included). On the model the one-iteration compares
pass every boundary (the head's logits at 0.013 to 0.028 against 0.164)
and fail the route log by one tail-expert flip (MIN-32).

## Provenance of the numbers

The run names carry the flags (`_gv` the GEMV linears, `_lg48` the grid,
`_mt_mh2` the half merge, `_rf_w2cktile` the CK w2 define, `_nts` the
streaming loads, `+nolocalcas`, `+pollsleep8`, `+gemvbatch4` the knobs);
each directory holds `report_table.md` (the per-operator gaps), the
correctness report, `plan.json`, `fleet_run_meta.json` and the run's
output. A rerun of a name keeps the earlier directory as
`<name>.prev-<utc>`. The `ktime` tables are `env/hw/20260918/ktime/`; the
task-graph check of the fold's run is in its directory; the probes of the
hangs (`probe_p1` to `p4`) stayed on the VM, their verdicts are in `09`.

## What remains for 4.0 ms and below

1. **Bytes in flight for the weight streams** (w13, w2, qkva, o_proj:
   2.3 ms per token). The stream probe caps the task-shaped reads at 2.25
   TB/s; direct-to-LDS loads (`01-gemv-ideas.md`, the staging form) hold
   more bytes in flight without VGPRs. At 4 TB/s the same bytes cost about
   1 ms less.
2. **The per-operator fixed cost** (MAJ-8): prep 9.8 us, `mul_sum_add`
   5.0, the norms 4 to 5 are the boundary; fusing pairs without a serial
   tail (prep into the attention, `mul_sum_add` into w2's epilogue, the
   router's gate GEMV into the norm that streams x) saves 3 to 5 us per
   pair per layer, about 0.3 ms for three.
3. **The iteration start**, 192 us per token, the runtime's per-iteration
   host work.
4. **The router** at 16.3 us for 2 MB is latency: eight tasks writing the
   logits with the top-k in the next operator's first task, instead of the
   four-task form's serial tail.
