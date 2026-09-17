# 01 - Ideas for a batch-1 GEMV linear

Written 2026-09-17 after round 3 (`../03-acceleration/`), for the fourth GPU
round. The decode stands at 4,571 to 4,600 us per token on the event clock
(`../03-acceleration/08-results.md`); the target is 4,500. The CK linears
hold 2.7 ms of the 4.6, and round 3 read their cost as a latency-bound
K loop (`09-lessons.md`, part 3, rank 2). This page lists every idea for
replacing that loop with a kernel of our own, what the source and the
round-3 record say about each, what it is worth and what it costs. It does
not choose; the split into laptop work and VM work is the next page. Every
number names its run or its source line.

## What round 3 established that this builds on

- **Boundaries are cheap, round trips are not.** An operator boundary costs
  2.3 to 2.9 us, a regular task 0.19 us on top (the empty-task ladder,
  `08-results.md`); a dependent memory round trip inside a task costs about
  1 to 2 us in the graph, twice what the same load costs in the standalone
  loop, because the producer's write-back has left the data in HBM
  (`08`, the kernels standalone).
- **The remedy that worked four times is loads in flight.** Prep, the
  router, the merge and the norm helper went 2x or better by issuing every
  load of a batch before the first multiply, keeping the words raw and
  converting on use, and keeping the FMA order (`09-lessons.md`, lesson 6).
- **The register file is the budget.** The worker kernel is the union of
  every task; the offline build reports 242 to 256 VGPRs, and the variant
  with the CK linear reports 256 VGPRs plus 35 AGPRs and 8 VGPRs spilled
  (`env/offline_gfx942/resources.txt`, `mk_cklinear`). The deeper router and
  merge batches were 1% slower for this reason (lesson 7).
- **The non-temporal weight loads are worth 20%** (round 2, E2: sc1 and nt
  on the CK weight loads, 12.3 to 10.2 ms); our own kernels have the same
  policy behind `MLA_NT_STREAMS` (`mla_common_mi300.cuh`, `StreamSrc`).
- **Unloaded latencies** (round 2, `env/hw/probes/chase.cu`): 81 ns in L2,
  258 ns in the memory-side cache, 342 ns at HBM.

## The linears today

Every linear runs the same CK tile: 16 x 64 x 256, one `v_mfma_16x16x32`
per wave per K step, `GemmPipelineAGmemBGmemCRegV2` with one LDS buffer
(`repos/fleet-chiplet-megakernel/.../linear_ck_mi300.cuh`; the pipeline in
`env/offline_gfx942/work/ck/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v2.hpp`).
The loop body is: sync, read the LDS step, eight MFMAs, sync, write the
next step from registers to LDS, issue the global loads of the step after.
One K step of global loads is in flight at a time, 32 KB of weight per
workgroup (the A rows beyond row 0 are out of the buffer's range and cost
no traffic). Every K step is therefore one dependent round trip plus two
barriers and an LDS pass.

| Linear | Tasks or tiles | Rows per task | Bytes per task | Per token, us (`08`, last column) | Per layer | HBM floor per layer at 4.3 TB/s |
|---|---|---|---|---|---|---|
| w13 (gang; 8 active experts, one per XCD, W13 `[66, 2816, 2048]`) | 44 tiles per XCD over 37 workers | 64 | 256 KB, 8 K steps | 1,104 | 42.5 | 21.5 (92 MB) |
| w2 with the silu prologue (gang, ours) | 32 tiles per XCD | 64 | 180 KB, 11 K steps of 128 | 621 | 23.9 | 10.7 (46 MB) |
| qkva with the norm prologue (per-tile, `linear_norm`) | 96 tasks | 38 | 152 KB, 8 K steps | 415 | 14.8 | 3.5 (15 MB) |
| o_proj (per-tile, residual; layer 0's down the same) | 64 tasks | 32 | 128 KB, 8 K steps | 416 | 14.8 | 2.0 (8 MB) |
| lm_head with the final norm (per-tile) | 400 tasks, 49 events of 8 | 256 (4 tiles) | 1 MB, 32 K steps | 141 | | 98 (420 MB) |
| layer 0's dense gate-up (stock silu gang) and down | 2 x 10,944 rows | | 90 MB and 45 MB | not in the MoE table | | 31 |

Correction to `08-results.md`: its w13 and w2 rows say 66 MB and 33 MB;
those count the six routed experts in MiB. The gang serves eight (the two
shared experts are packed as experts 64 and 65, `pack_weights.py`
`pack_moe`), 92 MB and 46 MB per layer.

### What bounds each one

Three regimes, and each linear sits in a different one:

1. **The latency chain.** A task whose loads are one K step at a time takes
   8 x (round trip + barriers) whatever the bandwidth. The per-tile linears
   measure 14.8 us for 8 steps: about 1.4 us per step, and 15 MB in 15 us
   is 1 TB/s of a 4.3 TB/s machine. The runtime's per-task cost (MAJ-8) is
   the other reading of the same number (`08`, what remains); the ladder
   predicts 12 to 18 us for 64 to 96 empty tasks. Which of the two binds is
   the first thing the VM must answer (M4 below).
2. **The XCD's share of HBM.** w13's 37 workers each keep 32 KB in flight:
   1.2 MB per XCD, more than the 540 KB Little's law needs for the XCD's
   540 GB/s share at 1 us. So w13's first round is bandwidth-bound
   (9.5 MB in about 18 us), and its 42.5 us is the second round (7 workers
   with a second tile, latency-bound, about 10 us) plus the gang's
   overhead. A deeper pipeline alone does not fix w13; the tile count does
   (S1).
3. **Between the two.** w2's 32 workers keep 16 KB each in flight
   (K steps of 128): 512 KB per XCD, at the edge of the requirement, so its
   11 steps run at about 1.3 us each plus the prologue: 24 us against a
   10.7 us floor.

Little's law per CU, for the bytes a task must keep in flight to stream
at its fair share of the machine (296 workers: 14.5 GB/s each at 4.3 TB/s;
a 96-task operator: 45 GB/s each):

| Latency assumed | Bytes in flight per CU for 45 GB/s | 16-byte loads per lane (4 waves) |
|---|---|---|
| 0.35 us (unloaded HBM) | 16 KB | 4 |
| 1 us (in-graph, the round-3 reading) | 45 KB | 11 |
| 2 us | 90 KB | 22 |

So 16 loads per lane in flight (64 VGPRs) covers the per-tile linears at
the in-graph latency, and 32 (128 VGPRs) covers a 2 us latency. A wave may
track 63 outstanding vector loads (`docs/mi300x/07-achievable-bandwidth.md`).
The CK tile's 32 KB per workgroup is 8 loads per lane, and it waits for all
of them before the next step's are issued.

## Group K: the kernel

The common shape (K1 to K3 differ in the multiply): a task or tile owns R
rows of the weight (R = 32 to 64; 256 for the head in four passes) and the
whole K. A wave owns R / 4 rows; a lane owns a 32-element slice of K
(K = 2048 over 64 lanes), held in registers for the whole task; each row is
four 16-byte loads per lane; B rows' loads are issued before the first
multiply (B = 8 or 16); a wave sum per row; the epilogue as the stock
kernel's (BF16 store, residual add, the expert scatter, the padded tail).
The router is this shape at R = 64, B = 4 (`moe_router_mi300.cuh`).

| Idea | What | Why, and the evidence | Worth | Cost, risk |
|---|---|---|---|---|
| **K1. VALU FMA on raw words** | the router's loop: `uint4` words, `(k & 1) ? (w & 0xffff0000) : (w << 16)`, `__uint_as_float`, one FMA per element in ascending k, `wave_sum` per row | proven in three kernels this round; exact products (BF16 x BF16 in FP32); the compiler places one `s_waitcnt vmcnt(n)` per use, so a batch issued in order and consumed in order overlaps naturally | the kernel core | two VALU ops per element: 1,024 per lane per 64-row tile at K = 2048, about 2 us of VALU per tile, on the same order as the memory time; must overlap with the next batch's loads (K6) |
| **K2. Packed `v_dot2_f32_bf16`** | the same loop with the input slice held as packed BF16 pairs and one `v_dot2_f32_bf16` per two elements: no conversion, half the FMAs | CDNA3 lists `v_dot2_f32_bf16` (VOP3P); whether hipcc 7.0 exposes a builtin for it on gfx942 is the offline compile's first check; the input slice costs 16 VGPRs instead of 32 | the VALU time of K1 to a quarter | the rounding point changes: two products and the accumulator summed in one instruction with the instruction's rounding, not FMA's; the suite row against the reference tolerates it, the boundary compare must confirm it. Not needed if K1 overlaps |
| **K3. MFMA from registers** | the attention kernel's form (`mla_attend_mfma_mi300.cuh`): lane l holds W row `l % 16` at k = 4 (l / 16) + i, x as A with only row 0 non-zero, `v_mfma_f32_16x16x16_bf16` straight from the loaded words; a wave covers 16 rows per instruction sequence | 16 x the VALU throughput per instruction even at 1/16 utilization: 128 MFMAs per wave per 16 rows (about 1,000 cycles) against 4,000 VALU cycles; the accumulator stays in the MFMA's FP32 order, the order the CK path uses today | removes the VALU term entirely | the 16-byte load gives a lane k = 8g..8g+7 and the MFMA wants k = 4g..4g+3 per group: feed two MFMAs from the halves and the sum's grouping differs from CK's at the ULP level (a permuted K order of the weight rows, done once in `pack_weights.py`, would make the instruction stream identical to CK's and the output bit-identical); AGPRs enter the union again |
| **K4. Direct-to-LDS staging** | `buffer_load_dword ... lds` (CK's `llvm_amdgcn_raw_buffer_load_lds`, gfx942: one dword per lane per instruction) into a 32 KB double buffer, the multiply reads LDS | bytes in flight without VGPRs: 64 KB of LDS instead of 128 VGPRs; the LDS read-back of a 256 KB tile is about 1 us at 128 B per clock | the register budget question disappears | four times the instructions of 16-byte loads (256 per wave per tile against the 63 in flight); the `lds` form needs the M0 register and inline asm as CK does it; CK's own async pipeline is gfx950-only. Last resort if K1 to K3 do not fit the union |
| **K5. The load policy** | the weight loads through `StreamSrc` (`sc1 nt`, the round-2 lever), the input row and the residual through plain loads, the output through `nt_store_u64` as the stock epilogue | the 20% of round 2 came from keeping the weight stream out of the caches the attention re-reads; the per-tile linears already run with `--nt-weights` | keeps the round-2 gain | none; `MLA_NT_STREAMS` is a build flag, off is the same math |
| **K6. Batch depth and the software pipeline** | B rows' loads issued, then the multiplies consume them in order while the next B rows' loads are issued (two batches alternating in registers), or the router's simpler form (a batch loaded, then consumed, then the next) | the compiler's per-use `vmcnt` waits make the first form free of explicit waits; the second form leaves the memory idle during the multiplies (K1's 2 us of VALU per tile would then add to the memory time instead of hiding under it) | the difference between a 4 us and a 6 us tile | registers: B = 16 rows is 64 VGPRs per batch, two batches 128; the union decides (I3); measured per step, as lesson 7 says |
| **K7. The input row** | the lane's 32-element slice of x in 32 FP32 VGPRs (K1) or 16 packed VGPRs (K2, K3), loaded once from the LDS copy the prologue writes (the norm's `out_s`) or from global for the plain linears | the router does this; the alternative of re-reading x from LDS per row costs 256 KB of LDS reads per tile (about 1 us, on the order of the memory time) | | K = 1,408 (w2) is 22 elements per lane, not a multiple of 8: chunks of 8 over the lanes with a bound (lanes 0 to 47 take three, 48 to 63 two), the same reason the stock w2 uses K steps of 128 |
| **K8. The prologues and epilogues in one kernel** | one template with flags: NORM (rmsnorm into LDS, no scratch row), SILU (silu times up into LDS, no scratch row), RESIDUAL (the add in the epilogue), MOE (the expert and slot decode and the scatter of the stock w13 and w2 kernels), the padded-tail store | the scratch rows of O2 and O3 existed only because the CK A window reads global memory; with the row in LDS the global write, the `s_waitcnt 0` and the sc0 re-read go (one round trip per task of qkva, lm_head and w2) | about 1 to 2 us per fused task | the registrations keep their scratch outputs until the plan drops them (I1); the layer-0 silu gang kernel's shuffled weight layout is its own case (S4) |

## Group S: shaping the work

| Idea | What | Why | Worth | Cost, risk |
|---|---|---|---|---|
| **S1. w13 in one round per XCD** | 2,816 rows over the XCD's 37 workers: 33 tiles of 76 rows and 4 of 77 (uneven tiles the kernel decodes from `tile_idx` and a row table), instead of 44 tiles of 64 | the second round of 7 tiles is latency-bound on 7 CUs; with every worker holding one tile of 305 KB the operator is bandwidth-bound end to end: about 21.5 us at the XCD's share plus the gang's overhead | 42.5 to about 26 us per layer: about 0.4 ms per token, the largest single item | the gang path dispatches `tiles_per_xcd` tiles to `min(tiles, workers)` workers (`persistent_kernel.cuh`, the gang loop); the tile count is a registration parameter, the worker count per XCD is 37 with `MAX_WORKER_PER_SCHEDULER=38`; a tile of 76 rows is 19 rows per wave, an uneven wave split |
| **S2. Rows per task for the per-tile linears** | qkva at 96, 48 or 32 tasks (38, 76, 114 rows); o_proj at 64 or 32 | if the kernel is the cost, more rows per task cost nothing (the bytes stream either way) and fewer tasks pay less of the ladder's 0.19 us; if MAJ-8 is the cost, the sweep shows it directly | up to 0.5 ms per token across the 55 per-tile operators, or a clean attribution | `grid_for_linear` in `graph_plan.py` is the knob; the weight and output partitions follow the grid |
| **S3. The head's grid** | 400 tasks of 256 rows in two rounds over 296 workers, or 320 tasks of 320 rows | the head is one operator per token at 141 us against a 98 us floor | about 40 us per token | the runtime chunks the 400 tasks into events of 8; `size // 256` is the demo's rule |
| **S4. Layer 0's dense MLP** | the same kernel for the 90 MB gate-up (the stock silu gang kernel reads a shuffled weight, `W_gu_shuffled`) and the 45 MB down | one layer, 135 MB, not in the per-layer table; floor 31 us | unknown until measured (a 2-layer run with `--layers 1`) | the shuffled layout must be undone in `pack_weights.py` or decoded in the kernel; later |

## Group I: integration

| Idea | What | Why | Cost |
|---|---|---|---|
| **I1. Task types** | one regular type `TASK_LINEAR_GEMV_MI300` with template flags for NORM and RESIDUAL (replacing `linear`, `linear_with_residual`, `linear_norm` in the plan), one gang type for w13 (`gang_moe_w13_gemv`) and the multiply of `gang_moe_w2_silu` swapped in place; registrations in `new_tasks.patch` after the pattern of `register_linear_norm_mi300_task` (the weight partitioned on dim 0 by the grid, the output on dim 1) | the plan's pointer conventions and the runtime's per-task offsets stay as they are; the CK kernels leave the JIT unit when no operator registers them, which is what shrinks the union | a day; the counts in `fleet/tests/test_graph_plan.py` and `graph_counts.py` follow |
| **I2. A flag, default off** | `--gemv-linears` in `graph_plan.py` and `build_graph.py`, as every round-3 lever was | the VM validates each lever with a step-0 compare before it becomes the default | an hour |
| **I3. The union, measured offline** | `env/offline_gfx942/run.sh` with a variant that instantiates the GEMV in place of the CK linears: VGPRs, AGPRs, spills, and the disassembly's `s_waitcnt vmcnt` placement between the batch's loads | lesson 7: a spill costs more than a round trip saves; the disassembly is the only laptop proof that the loads are in flight together | 15 minutes per variant in Docker |
| **I4. Suite rows** | `k_linear_gemv` in `kernel_tests_mi300.cu` for the plain, norm, residual and MoE forms against `numpy_ref.linear_norm` and new one-line references; a `ktime` row for the tile alone, cold cache | the suite has no CK linear today, so the first linear rows are new; the standalone time against the graph's exec counter is the 2x check of lesson 6 | half a day on the laptop, minutes on the VM |
| **I5. The reference for bit-comparison** | keep the round-3 boundary dumps of `L2_it1_tile_at_fn1_fn2_fs_nt_mfma` and diff the new run's qkva and o_proj boundaries against them element by element, beside the tolerance compare | tells the ULP-level story of K1 against K3 (and K3 with the permuted K order should be zero diff) | a script over the existing dump format |

## Group M: measurement on the VM

| Row | What it answers |
|---|---|
| **M1.** the 2-layer graph, `--gemv-linears`, `--iters 1`, the step-0 compare and the ids | correctness of every fused form |
| **M2.** the same with the worker timing: the exec counter of the GEMV class against the CK class of the S5 record | the kernel's own time in the graph (the 2x rule) |
| **M3.** the per-operator event gaps of qkva, o_proj, w13, w2 against the S5 columns | the operator-level gain, and the gang overhead of w13 by subtraction |
| **M4.** S2's sweep: qkva at 96, 48, 32 tasks | whether the per-tile linears are kernel-bound or completion-bound (MAJ-8); decides whether the runtime change is this round's or the next's |
| **M5.** `ktime` of the tile, cold, at B = 8, 16, 32 | the batch depth's own curve, without the graph |
| **M6.** the 27-layer finals at 30, 31, 32 iterations, `FWD_PASS` without instrumentation | the round's number |

## Not chosen

| Route | Why not |
|---|---|
| CK `GemmPipelineAgBgCrMem` with a custom policy | its prefetch depth is `clamp(32 KB / ((M + N) x 2 x K), 2, 8)`: 32 KB in flight per workgroup, the same as today; a smaller K block gives more stages of the same total (`gemm_pipeline_ag_bg_cr_mem.hpp`, lines 40 to 48) |
| CK's async (direct-to-LDS) pipelines | the 16-byte LDS load is gfx950-only (`amd_buffer_addressing.hpp`, `#if defined(__gfx950__)`); gfx942 has the dword form, which is K4 by hand |
| a larger CK tile (M = 16 padded rows are wasted anyway) | the tile's M is padding; the K loop's depth is the problem, not the tile |
| FP8 weights | bytes, not latency; Group F of `../03-acceleration/01-ideas.md`, unchanged |

## The stack

On the last worker-timing run's 4,713 us (the finals 4,590):

| Item | Today, us per token | Reachable | Gain | Certainty |
|---|---|---|---|---|
| w13: S1 (one round) with K1 to K3 | 1,104 | about 650 | 450 | high: the second round and the gang overhead are visible in the arithmetic |
| w2: the deep stream with the silu row in LDS | 621 | about 340 | 280 | high: 512 KB per XCD in flight today is at the edge of Little's law |
| qkva, o_proj, down: the deep stream, no scratch row | 831 | about 330 | 500 | open: MAJ-8 may bind (M4) |
| lm_head: S3 | 141 | about 100 | 40 | medium |
| layer 0's dense MLP: S4 | unmeasured | | | later |
| **total** | | | **770 certain, 1,270 possible** | |

The target needs 90 us. The two gang items alone clear it by the arithmetic;
the per-tile item is the one the round learns about.

## To decide before the split

1. K1 first (proven, exact), with K3 as the second step if the VALU time
   shows in M5; or K3 first for the bit-identity with the CK path.
2. The batch depth to build for: B = 16 (64 VGPRs, the safe union) with
   B = 32 as a compile-time constant to sweep on the VM.
3. Whether S1's uneven tiles are done by a row table in the registration
   parameters or by the kernel's arithmetic from `tile_idx`.
4. Whether MAJ-8 (the runtime's completion hierarchy) enters this round at
   all, or waits for M4's answer.
