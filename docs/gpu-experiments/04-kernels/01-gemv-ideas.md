# 01 - Ideas for a batch-1 GEMV linear

Written 2026-09-17 after round 3 (`../03-acceleration/`), for the fourth GPU
round; expanded and double-checked the same day (the section "What the
double-check changed" and the probe in `env/offline_gfx942/gemv_probe/`).
The decode stands at 4,571 to 4,600 us per token on the event clock
(`../03-acceleration/08-results.md`); the target is 4,500. The CK linears
hold 2.7 ms of the 4.6, and round 3 read their cost as a latency-bound
K loop (`09-lessons.md`, part 3, rank 2). This page lists every idea for
replacing that loop with a kernel of our own, what the source, the round-3
record and the offline compiler say about each, what it is worth and what
it costs. It does not choose; the split into laptop work and VM work is
`02-local-gpu-split.md`. Every number names its run or its source line.

## What the double-check changed

Each idea of the first draft was checked against the runtime's source, the
CK headers we pin, the round-3 record and, where a claim was about the
compiler, against hipcc 7.0 for gfx942 in the offline Docker image
(`env/offline_gfx942/gemv_probe/`, 30 s per variant).

| Claim of the draft | What the check says | Consequence |
|---|---|---|
| K2, a packed `v_dot2_f32_bf16` halves the VALU work | the builtin "needs target feature dot12-insts" and the inline instruction is "not supported on this GPU" for gfx942; both compile for gfx950 (`gemv_probe/results.txt`) | **K2 is off** for the MI300X; the VALU form's conversion cost stays, and K3 (MFMA) is the only way to cut it |
| K6, the batch depth B is set by the loop constant | the compiler unrolls the wave's 16 rows and hoists every load across the batches: BATCH = 8 gives 30 loads in flight and 175 VGPRs, BATCH = 16 gives 60 in flight and 256 VGPRs plus 28 AGPRs; `#pragma unroll 1` on the batch loop restores the constant (4 rows: 110 VGPRs, 16 in flight; 8 rows: 158, 32 in flight) | the depth is controlled by the unroll pragma, not the constant; the register cost per depth is now measured (the table under K6); the round-3 router's loop was not hoisted (124 VGPRs standalone: one batch live), so its four round trips are real |
| "the register file is the budget", the CK tile fills it | the arch VGPR limit is 256 per wave at one wave per SIMD; the excess goes to AGPRs (`v_accvgpr` moves, 28 of them at 60 loads in flight, no scratch) and a scratch spill appears only past that (the invalid 32-row probe: 68 spilled, 276 bytes per lane); the union without any CK linear (`mk_ours`) was 249 VGPRs before round 3 and is 256 VGPRs plus 64 AGPRs with 8 VGPRs spilled with round 3's batched kernels (the refreshed offline build of 2026-09-17 evening, `resources.txt`) | the CK tile leaving the union frees nothing by itself; the union is already past the arch limit and round 3's batches put it there; what matters is that the GEMV's own form fits under 256 without a scratch spill, which 32 loads in flight does (158) and 60 does with AGPRs (256 + 28); layer 0's dense gate-up keeps the CK silu kernel in the unit until S4 replaces it |
| regime 2, w13's first round is bandwidth-bound at the XCD's share (17.5 us) | 42.5 us per layer less a latency-bound second round of about 10 us and a boundary of about 3 leaves about 28 us for the first round, not 17.5: the XCD moved 9.5 MB at about 340 GB/s with 1.2 MB in flight, which is Little's law at a loaded latency of about 3.5 us per K step, or a fabric limit below the 1/8 share; the record cannot tell which | S1 (one round) and the deep stream are both needed for w13, and the per-XCD ceiling is unmeasured: a stream probe on the VM (M7) sets w13's reachable number (21 to 30 us) |
| K3, the MFMA at 8 cycles per instruction | from the part's peak (1,307 TFLOPs BF16 dense, 304 CUs, 4 SIMDs, 2.1 GHz) a `16x16x16` MFMA is 16 cycles per SIMD | 128 MFMAs per wave per 16 rows is about 1 us against about 2 us of VALU (512 FMAs and 555 conversions, the probe's disassembly): MFMA halves the ALU time, it does not remove it |
| I5, a bit-diff against round 3's boundary dumps | the record keeps `compare.out` and the reports, not the tensors (`env/hw/20260917/runs/L2_it1_*`) | the CK build's boundaries are dumped once on the VM before the GEMV build runs (a queue row), then diffed |
| K1's lane map, the router's (a lane owns 32 consecutive elements) | a wave-load then touches 32 cache lines of 128 bytes and uses 32 bytes of each; the other three quarters are re-fetched by the next three loads, from L1 if the streaming policy kept the lines and from L2 if not. CK's own B distribution gives consecutive lanes consecutive 16-byte chunks (`linear_ck_mi300.cuh`, `MakeBDramTileDistribution`: K0 = 32 lanes along K), 8 full lines per wave-load. The merge's `W_uv` phase (two lanes per row) is worse still | **K9 added**: the lane map that makes every wave-load 1 KB contiguous, at no cost; the same finding goes to the router and merge page |
| K3, the head as 16 passes per wave | 256 rows over 4 waves is 64 rows per wave, four passes of 16 | corrected |
| the fresh-host setup in `02`, "about 8 minutes" | round 3's log: download 95 s, hardware census 158 s, setup 483 s | 12 minutes |
| everything else | the CK loop body (`gemm_pipeline_agmem_bgmem_creg_v2.hpp`, lines 251 to 279), the A view of one row (`linear_norm_mi300.cuh`, the w2 kernel, w13 at batch 1: `make_tuple(index_t(1), ...)`), the gang tile loop (`persistent_kernel.cuh`, lines 1099 to 1109: `for (t = rank; t < n_tile_count; t += workers_on_xcd)`), the eight experts per layer (`W13_{l}` is `[66, 2816, 2048]`, the mask holds 8 slots), the head's 400 tasks of 256 rows, the 37 workers per XCD, 63 outstanding loads per wave, the 57 KiB of dynamic LDS, 2,816 = 33 x 76 + 4 x 77 | stand as written |

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
- **The register wall is real, and round 3's kernels stand on it.** The
  worker kernel is the union of every task at one wave per SIMD: 256 arch
  VGPRs, then AGPRs, then scratch. Before round 3 the union was 249
  VGPRs and no AGPRs; with the batched prep, router, merge and norm it is
  256 VGPRs, 64 AGPRs and 8 VGPRs spilled to them (the refreshed
  `env/offline_gfx942/resources.txt`; 134 AGPRs with the MFMA attention,
  154 with the CK linear variant). The deeper router and merge batches of
  round 3 were 1% slower (lesson 7): AGPR traffic, not a scratch spill
  (the scratch size is the same 64 bytes per lane in every variant). Every
  depth this round is therefore measured on the union's line, not only on
  the kernel's own.
- **The non-temporal weight loads are worth 20%** (round 2, E2: sc1 and nt
  on the CK weight loads, 12.3 to 10.2 ms); our own kernels have the same
  policy behind `MLA_NT_STREAMS` (`mla_common_mi300.cuh`, `StreamSrc`).
- **Unloaded latencies** (round 2, `env/hw/probes/chase.cu`): 81 ns in L2,
  258 ns in the memory-side cache, 342 ns at HBM. Loaded, inside the
  graph, the round trip is 1 to 2 us (above) and, by the w13 arithmetic
  below, up to 3.5 us per K step when 37 CUs of an XCD stream at once.

## The linears today

Every linear runs the same CK tile: 16 x 64 x 256, one `v_mfma_16x16x32`
(two `16x16x16` on gfx942) per wave per K step,
`GemmPipelineAGmemBGmemCRegV2` with one LDS buffer
(`repos/fleet-chiplet-megakernel/.../linear_ck_mi300.cuh`; the pipeline in
`env/offline_gfx942/work/ck/include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v2.hpp`).
The loop body (lines 251 to 279) is: sync, read the LDS step, the MFMAs,
sync, write the next step from registers to LDS, issue the global loads of
the step after. One K step of global loads is in flight at a time: 32 KB of
weight per workgroup, 8 loads per lane (the A view has one row, so the
padded rows 1 to 15 are out of the buffer's range and cost no traffic).
Every K step is therefore one dependent round trip plus two barriers and an
LDS pass. The offline disassembly of the CK variant shows 192 MFMAs and 122
AGPR moves (`env/offline_gfx942/work/out/dev_cklinear.s`).

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
2. **The XCD's rate.** w13's 37 workers each keep 32 KB in flight, 1.2 MB
   per XCD, and the first round of 37 tiles takes about 28 us (42.5 less
   the second round and the boundary): 9.5 MB at about 340 GB/s per XCD,
   2.7 TB/s over the eight. Either the loaded round trip is about 3.5 us
   at this concurrency (Little's law: 1.2 MB in flight at 340 GB/s) or an
   XCD cannot draw its 1/8 share of 540 GB/s; the record cannot separate
   the two, and M7 does. In both readings the second round (7 tiles on
   7 CUs, about 10 us) is pure loss.
3. **Between the two.** w2's 32 workers keep 16 KB each in flight
   (K steps of 128): 512 KB per XCD, at the edge of the requirement even
   at 1 us, so its 11 steps run at about 1.3 us each plus the prologue:
   24 us against a 10.7 us floor.

Little's law per CU, for the bytes a task must keep in flight to stream
at its fair share of the machine (296 workers: 14.5 GB/s each at 4.3 TB/s;
a 96-task operator: 45 GB/s each):

| Latency assumed | Bytes in flight per CU for 45 GB/s | 16-byte loads per lane (4 waves) |
|---|---|---|
| 0.35 us (unloaded HBM) | 16 KB | 4 |
| 1 us (in-graph, the round-3 reading) | 45 KB | 11 |
| 2 us | 90 KB | 22 |
| 3.5 us (w13's first round, regime 2) | 158 KB | 39 |

So 16 loads per lane in flight covers the per-tile linears at the in-graph
latency, 32 covers 2 us, and the whole 64-row tile (64 loads, the probe's
`hoist16`) covers the worst reading. A wave may track 63 outstanding
vector loads (`docs/mi300x/07-achievable-bandwidth.md`). The CK tile's
32 KB per workgroup is 8 loads per lane, and it waits for all of them
before the next step's are issued.

## Group K: the kernel

The common shape (K1 and K3 differ in the multiply): a task or tile owns R
rows of the weight (R = 32 to 64; 256 for the head in four passes) and the
whole K. A wave owns R / 4 rows; a lane owns a 32-element slice of K
(K = 2048 over 64 lanes), held in registers for the whole task; each row is
four 16-byte loads per lane; B rows' loads are issued before the first
multiply; a wave sum per row; the epilogue as the stock kernel's (BF16
store, residual add, the expert scatter, the padded tail). The router is
this shape at R = 64, B = 4 (`moe_router_mi300.cuh`); the probe
`env/offline_gfx942/gemv_probe/gemv_probe.cu` is the shape at R = 64,
K = 2048 with B and the unroll control as compile-time switches.

### K1. VALU FMA on raw words (the core)

The router's loop: `uint4` words, `(k & 1) ? (w & 0xffff0000) : (w << 16)`,
`__uint_as_float`, one FMA per element in ascending k, `wave_sum` per row.

- **Thread map.** Wave w owns rows `w * R/4 .. (w+1) * R/4 - 1`; lane l
  owns 32 elements of every row in four 16-byte chunks, one per load. The
  router places them at `32 l + 8 i` (contiguous per lane); K9 places
  them at `8 l + 512 i` (contiguous per wave-load), which is the map to
  build. Each lane's slice of x sits in 32 FP32 VGPRs (K7), loaded with
  the same map.
  For R = 38 (qkva) the waves take 10, 10, 9, 9 rows; for R = 32, 8 each.
- **The multiply.** Per row per lane: 32 conversions (a shift or a mask)
  and 32 `v_fmac_f32`, then a 6-step `wave_sum`; the probe's disassembly
  counts 512 FMAs and 555 conversions per wave per 16 rows. About 2 us of
  VALU per 64-row tile at K = 2048 (1,067 ops x 4 cycles per wave64 op at
  2.1 GHz), on the same order as the memory time of the tile at the
  per-CU share (256 KB at 45 GB/s is 5.7 us; at 14.5 GB/s, 17.6 us), so it
  hides under the loads if the batches overlap (K6) and adds to them if
  they do not.
- **Evidence.** Three round-3 kernels use exactly this loop; the products
  are exact (BF16 x BF16 fits FP32); the accumulation order is a serial
  chain per lane then a butterfly tree, different from the MFMA's, which
  is within the tolerance the boundary compare applies to the CK path's
  own order today (the numerics under K3).
- **Worth.** The kernel core; every other idea builds on it.
- **Cost, risk.** The conversion ops are the price of BF16 on gfx942 (no
  packed BF16 FMA, no dot2: the double-check); a `v_pk_fma_f32` form
  (two FMAs per op on register pairs) would take the count to about 770
  ops per 16 rows if the compiler emits it, which the offline
  disassembly shows.

### K2. Packed `v_dot2_f32_bf16` (off)

The instruction is gfx950's (dot12-insts); hipcc rejects both the builtin
and the inline form for gfx942. Kept here so it is not proposed again.

### K3. MFMA from registers

The attention kernel's form (`mla_attend_mfma_mi300.cuh`, the layout of
`mla_common_mi300.cuh` and `fleet/tests/test_mfma_layout.py`): the weight
tile is the B operand straight from the loaded words, x is the A operand
with only row 0 non-zero, `v_mfma_f32_16x16x16_bf16` accumulates in FP32.

- **Thread map.** A wave owns 16 rows; lane l holds row `l % 16` at
  k = `32 g + 8 (l / 16) + 0..7` for K group g (one 16-byte load per
  group: 64 loads per lane for K = 2048, the same bytes as K1). The
  instruction wants B[k][n] at lane l as k = `4 (l / 16) + i`, n = `l % 16`
  (n is the weight row here), so the load's eight values feed two MFMAs:
  the first with the lane's k = 8 (l/16) + 0..3, the second with + 4..7.
  The A operand, x, is needed only in the four lanes with `l % 16 == 0`
  (A[m][k] at lane l is m = `l % 16`): four BF16 values per MFMA from an
  LDS copy of x (`ds_read_b64`, 128 per lane per tile), zero elsewhere.
  The result D[0][n] lands in lanes 0 to 15, register 0: the 16 rows'
  outputs, no wave sum.
- **The multiply.** 128 MFMAs per wave per 16 rows; at 16 cycles each
  (the double-check) about 1 us per 64-row tile, half of K1's VALU time,
  and no conversion instructions.
- **Numerics.** Each MFMA sums 16 products into the accumulator in the
  hardware's order, the same instruction the CK path uses. The grouping
  of k into instructions differs from CK's (`WarpGemmMfmaBf16Bf16F32M16N16K32`
  on gfx942 splits K = 32 into two `K16` instructions whose lane layout is
  CK's, ours takes 8 consecutive k per lane from one load), so the result
  differs from today's CK output at the ULP level. A permuted K order of
  the weight rows, done once in `pack_weights.py` so that the lane's
  16-byte load holds CK's k groups, would make the instruction stream
  identical and the output bit-identical to the CK path (checked by I5).
  Not needed for correctness; a way to keep the ids provably unchanged.
- **Worth.** Removes half the ALU time and the conversions; the natural
  second step if M5 shows the VALU term.
- **Cost, risk.** AGPRs hold the accumulators (the MFMA attention put 53
  in the union); the x operand's LDS reads and the lane masking are new
  code; the head's 256 rows are four passes of 16 rows per wave. The
  instruction fixes the lane map: lanes 0 to 15 read 16 different rows
  at one k offset, so a wave-load touches 16 lines and uses 64 bytes of
  each (the next K group uses the other half), two lines per KB against
  K9's one; the same L1 question as K1's strided map, at half the
  weight. A `v_mfma_f32_4x4x4_16b` form (16 independent 4 x 4 blocks per
  instruction, so a quarter of the work is padding instead of fifteen
  sixteenths) would halve the ALU time again at the cost of an intricate
  layout; a step after K3 only if the ALU term still shows.

### K4. Direct-to-LDS staging (last resort)

`buffer_load_dword ... lds` (CK's `llvm_amdgcn_raw_buffer_load_lds` in
`amd_buffer_addressing.hpp`; gfx942 has the one-dword form, gfx950 adds
three and four dwords): the loads land in LDS at M0 plus the lane's
offset without touching VGPRs, so 64 KB can be in flight per CU from the
LDS budget alone, and the multiply reads the rows back from LDS
(256 KB per tile at 128 bytes per clock, about 1 us).

- **Why last.** Four times the instruction count of 16-byte loads (256
  per wave per tile against the 63 a wave may have outstanding, so four
  waits per tile), the M0 setup and inline asm CK uses only for gfx950,
  and the probe shows the VGPR route reaches 32 loads in flight at 158
  VGPRs without it.
- **When.** If the union cannot take 158 (I3 says), or for the head's
  1 MB tasks where the LDS ring would let the loads run ahead of the
  passes.

### K5. The load policy

The weight loads through `StreamSrc` (`sc1 nt`: the round-2 lever, 20%),
the x row and the residual through plain loads, the output through
`nt_store_u64` as the stock epilogue. `MLA_NT_STREAMS` is a build flag; off
is the same math. The probe uses `__builtin_nontemporal_load` (the `nt`
bit alone); the buffer form with `sc1` is what the kernels use.

### K6. Batch depth and the software pipeline

- **Control.** The batch constant does not set the depth by itself: the
  compiler unrolls the wave's row loop and hoists the loads (the probe:
  BATCH = 8 unrolled gives 30 loads in flight and 175 VGPRs). `#pragma
  unroll 1` on the batch loop makes the constant the depth; the register
  cost is then:

  | Rows per batch (NOHOIST) | Loads in flight per lane | Bytes in flight per CU | VGPRs | AGPRs |
  |---|---|---|---|---|
  | 4 | 16 | 64 KB | 110 | 0 |
  | 8 | 32 | 128 KB | 158 | 0 |
  | 16 (the whole wave's tile) | 64 | 256 KB | 256 | 28 |
  | the CK tile today | 8 | 32 KB | (256 + 35 in the union) | |

- **Two forms.** The router's: issue the batch, wait, multiply, next
  batch (the memory idles during the multiply; the compiler's per-use
  `vmcnt` waits still overlap the multiply of row u with the arrival of
  rows u + 1 ..). The pipelined: two batches alternate, the loads of batch
  b + 1 issued before the multiplies of batch b (twice the raw registers
  of one batch: 8 rows pipelined costs what 16 rows unrolled costs).
- **What the numbers say.** With 32 loads in flight per lane a 64-row
  tile is two batches: at a 2 us round trip about 4 us of memory and 2 us
  of VALU in sequence (the router's form) or about 4 us overlapped. So the
  simple form at 8 rows is within 2 us of the pipelined one per tile, and
  the first build should be the simple form at 8 rows with the depth as a
  constant to sweep (M5); the pipelined form is the step after, if the
  ALU shows.

### K7. The input row

The lane's 32-element slice of x in 32 FP32 VGPRs (K1) or as 16 packed
words read from LDS per MFMA (K3), loaded once. The plain linears read x
from global memory (4 KB, one load per lane); the fused ones read the
prologue's LDS copy (`rmsnorm_row` already writes `out_s`; the silu
prologue writes the activation row the same way). Re-reading x from LDS
per row instead would cost 256 KB of LDS reads per tile, about 1 us.

K = 1,408 (w2) is 22 elements per lane, not a multiple of 8: the row is
176 chunks of 8, lanes 0 to 47 take three chunks and 48 to 63 two (the
chunk index `lane + 64 j`, j < 3, bounded by 176), and the x slice is 24
VGPRs (lanes 48 to 63 use 16 of them); the same irregularity made the stock w2 use K
steps of 128.

### K8. The prologues and epilogues in one kernel

One template with flags:

| Flag | What the task does before or after the multiply | Today |
|---|---|---|
| NORM | `rmsnorm_row` of x into LDS with the layer's norm weight (FP32 statistics, the normalised value rounded to BF16, the BF16 weight multiply: the reference's order) | the same, plus a write of the row to a `[grid, K]` scratch tensor, `s_waitcnt 0`, a release fence, and the CK A window reading it back with sc0 (`linear_norm_mi300.cuh`) |
| SILU | `silu(gate) * up` of the slot's row into LDS, BF16-rounded element by element as `silu_mul_task_impl` | the same, through a `[8 x 32, 1408]` scratch row (`gang_moe_w2_silu_mi300.cuh`) |
| RESIDUAL | the epilogue adds the residual's BF16 value in FP32 before the BF16 store | the stock 16 x 64 epilogue |
| MOE | the expert and slot decode of the stock w13 and w2 kernels (`d_mask`, `expert_routing`, the XCD from `s_getreg`), the scatter into `[1, 8, N]` | the stock kernels |
| the tail | rows past the task's count masked (the 38-row qkva task); columns past N masked as the stock epilogue does | |

Dropping the scratch rows removes one global round trip per fused task
(the write, the wait, the read back) and two tensors from the plan; the
registrations keep the scratch outputs until the plan drops them (I1).

Two more round trips hide for free: the first weight batch's loads do not
depend on x, so they are issued before the prologue (the norm's read of
x and its block reduction, or the silu row, then run under the batch's
latency; the batch's raw registers are live across the prologue, whose
own need is small), and the residual's 32 elements per lane are loaded
with the first batch, not in the epilogue. The router page will use the
same trick for its norm.

### K9. The coalesced lane map

A wave-load is 64 lanes x 16 bytes. With the router's map (lane l at
`32 l + 8 i`) the 64 addresses of one load are 64 bytes apart: 32 cache
lines of 128 bytes touched, 32 bytes used from each, the rest re-fetched
by the next three loads of the row, from L1 if the line survived and
from L2 if the streaming policy (`sc1 nt`: L1 miss-evict) dropped it.
With lane l at `8 l + 512 i` the 64 addresses of one load are one
contiguous KB: 8 full lines, nothing re-fetched. The lane's four chunks
are then k = 8 l + 512 i + 0..7, and its x slice is loaded with the same
map, so the products are the same and only the summation order within
the lane changes (four runs of eight instead of one of 32). CK's B
distribution is of this kind (K0 = 32 lanes along K, two rows per
wave-load); the round-3 kernels are not: the router's rows are strided as
above, the merge's partials phase reads 32-byte chunks per lane and its
`W_uv` phase two lanes per row, 32 to 64 lines per wave-load.

- **Worth.** Unknown until measured; it is the difference between 8 and
  32 line requests per instruction on the L1 and L2 request queues, which
  is the resource the `docs/mi300x/07` analysis names as the unproven
  limit. Free to adopt; a compile switch (`GEMV_STRIDED`) keeps the
  router's map for one `ktime` A/B (M5).
- **The K-split variant.** Wave w could own a quarter of K for every row
  (a wave-load then reads bytes `1024 w + 16 l` of a row: also
  contiguous) with the four partial sums added through LDS: the x slice
  drops to 8 VGPRs (24 more loads in flight at the same register cost),
  but every row then needs a wave sum in each wave (64 per wave instead
  of 16, about 290 extra shuffle ops per tile against 512 FMAs). Not
  first; a variant if the register budget, not the ALU, turns out to bind.

## Group S: shaping the work

### S1. w13 in one round per XCD

2,816 rows over the XCD's 37 workers: 33 tiles of 76 rows and 4 of 77
instead of 44 tiles of 64. The kernel decodes the tile's rows from
`tile_idx` by arithmetic (tile t < 4 starts at 77 t with 77 rows, else at
308 + 76 (t - 4) with 76), no table; a wave takes 19 rows (20 in the last
wave of a 77-row tile). The registration's `n_tiles` becomes 37 (the
`_gang_moe` helper in `build_graph.py` computes `weight.dim(1) // 64`
today; the w13 form takes the worker count instead) and
`total_tiles_per_xcd` follows.

- **Why.** The gang loop hands tile t to the worker of rank `t mod 37`
  (`persistent_kernel.cuh`), so 44 tiles are two rounds for 7 workers;
  with 37 tiles every worker holds one tile of about 305 KB and the
  operator ends when the XCD's stream ends.
- **Worth.** The second round (about 10 us per layer, 0.26 ms per token)
  goes in every reading of regime 2; the first round's 28 us goes to the
  XCD's rate over 11.5 MB, 21.5 us at the 1/8 share or about 28 at the
  observed 340 GB/s, which M7 settles. 42.5 to 26 to 33 us per layer:
  0.25 to 0.43 ms per token.
- **Cost, risk.** The worker count per XCD is 37 for `num_workers = 296`
  and `MAX_WORKER_PER_SCHEDULER = 38`; a different count runs the 37
  tiles in more rounds, correct but slower (an assert in the plan against
  the worker count). The uneven wave split is a bound in the row loop.

### S2. Rows per task for the per-tile linears

qkva at 96, 48 or 32 tasks (38, 76, 114 rows); o_proj at 64 or 32.
`grid_for_linear` in `graph_plan.py` is the knob; the weight and output
partitions and the scratch's row count follow the grid. If the kernel is
the cost, more rows per task cost nothing (the bytes stream either way)
and fewer tasks pay less of the ladder's 0.19 us; if MAJ-8 is the cost,
the sweep shows it directly: an operator whose time tracks the task count
rather than the bytes is completion-bound. Worth up to 0.5 ms per token
across the 55 per-tile operators, or a clean attribution.

### S3. The head's grid

400 tasks of 256 rows in two rounds over 296 workers (the runtime chunks
them into events of 8), or 320 tasks of 320 rows. The head is one
operator per token at 141 us against a 98 us floor: about 40 us. The
GEMV's 256-row task is four passes of the 64-row form with the norm
computed once.

### S4. Layer 0's dense MLP

The 90 MB gate-up runs the stock silu gang kernel (`gang_linear_silu`)
on a shuffled weight (`W_gu_shuffled`, `pack_weights.py`), the 45 MB down
the residual per-tile linear. One layer, 135 MB, a 31 us floor, not in
the per-layer table (a `--layers 1` run measures it). The same GEMV with
the silu epilogue serves it once the shuffled layout is undone in the
packer or decoded in the kernel; it is also what keeps a CK tile in the
JIT unit after every other linear is replaced. Later in the round.

## Group I: integration

| Item | What | Why | Cost |
|---|---|---|---|
| **I1. Task types** | one regular type `TASK_LINEAR_GEMV_MI300` with the K8 flags as template parameters, replacing `linear`, `linear_with_residual` and `linear_norm` in the plan; one gang type for w13 (`gang_moe_w13_gemv`, S1's tiles); the multiply of `gang_moe_w2_silu` swapped in place (its type 191 stays); registrations in `new_tasks.patch` after `register_linear_norm_mi300_task` (the weight partitioned on dim 0 by the grid, the output on dim 1, the residual whole) | the plan's pointer conventions and the runtime's per-task offsets stay as they are; a CK kernel leaves the JIT unit when no operator registers it | a day; the counts in `fleet/tests/test_graph_plan.py` and `docs/design-doc/sources/graph_counts.py` follow |
| **I2. A flag, default off** | `--gemv-linears` in `graph_plan.py` and `build_graph.py`, as every round-3 lever; with it the fused forms drop their scratch tensors | the VM validates each lever with a step-0 compare before it becomes the default | an hour |
| **I3. The union, measured offline** | `env/offline_gfx942/run.sh` with a variant `MK_GEMV` that instantiates the GEMV forms at the model's dims in `mk_tu.cu` (as `MK_CK_LINEAR` and `MK_CK_GANG` do for the CK forms): VGPRs, AGPRs, scratch, spills; the `s_waitcnt vmcnt` sequence of the tile loop in the disassembly (the probe's method) | lesson 7: a scratch spill costs more than a round trip saves; the disassembly is the laptop's only proof that the loads are in flight together | 15 minutes per variant in Docker |
| **I4. Suite rows** | `k_linear_gemv` in `kernel_tests_mi300.cu` for the plain, norm, residual and MoE forms against `numpy_ref.linear_norm` and one-line references (a dot product per row, the residual add, the scatter by a routing vector); a `ktime` row for the tile alone, cold cache, at the depths of K6 | the suite has no CK linear today, so the first linear rows are new; the standalone time against the graph's exec counter is the 2x check of lesson 6 | half a day on the laptop, minutes on the VM |
| **I5. The reference for bit-comparison** | a queue row of the CK build that keeps its boundary tensors (`fleet_out/`, today discarded after the compare), and `harness/bitdiff.py` that reports, per boundary, the count of differing elements and the max ULP distance against the GEMV build's tensors | tells the ULP-level story of K1 against K3 (and K3 with the permuted K order should be zero diff) beside the tolerance compare | a script; a queue flag |

## Group M: measurement on the VM

| Row | What it answers |
|---|---|
| **M1.** the 2-layer graph, `--gemv-linears`, `--iters 1`, the step-0 compare and the ids | correctness of every fused form |
| **M2.** the same with the worker timing: the exec counter of the GEMV class against the CK class of the S5 record | the kernel's own time in the graph (the 2x rule) |
| **M3.** the per-operator event gaps of qkva, o_proj, w13, w2 against the S5 columns | the operator-level gain, and the gang overhead of w13 by subtraction |
| **M4.** S2's sweep: qkva at 96, 48, 32 tasks | whether the per-tile linears are kernel-bound or completion-bound (MAJ-8); decides whether the runtime change is this round's or the next's |
| **M5.** `ktime` of the tile, cold, at 4, 8 and 16 rows per batch | the batch depth's own curve, without the graph, and whether the VALU term shows (K3's trigger) |
| **M6.** the 27-layer finals at 30, 31, 32 iterations, `FWD_PASS` without instrumentation | the round's number |
| **M7.** a stream probe: a gang operator whose 37 tiles per XCD read 305 KB each with the GEMV's loads and no multiply, and a regular operator of 296 such tasks | the XCD's achievable rate and the device's, the ceiling for S1 and for every gang linear |

## Not chosen

| Route | Why not |
|---|---|
| CK `GemmPipelineAgBgCrMem` with a custom policy | its prefetch depth is `clamp(32 KB / ((M + N) x 2 x K), 2, 8)`: 32 KB in flight per workgroup, the same as today; a smaller K block gives more stages of the same total (`gemm_pipeline_ag_bg_cr_mem.hpp`, lines 40 to 48) |
| CK's async (direct-to-LDS) pipelines | the 16-byte LDS load is gfx950-only (`amd_buffer_addressing.hpp`, `#if defined(__gfx950__)`); gfx942 has the dword form, which is K4 by hand |
| a larger CK tile (M = 16 padded rows are wasted anyway) | the tile's M is padding; the K loop's depth is the problem, not the tile |
| `v_dot2_f32_bf16` (K2) | not on gfx942 (the double-check) |
| FP8 weights | bytes, not latency; Group F of `../03-acceleration/01-ideas.md`, unchanged |

## The stack

On the last worker-timing run's 4,713 us (the finals 4,590):

| Item | Today, us per token | Reachable | Gain | Certainty |
|---|---|---|---|---|
| w13: S1 (one round) with the deep stream | 1,104 | 680 to 860 | 250 to 430 | high for the second round; the first round's number is M7's |
| w2: the deep stream with the silu row in LDS | 621 | about 340 | 280 | high: 512 KB per XCD in flight today is at the edge of Little's law |
| qkva, o_proj, down: the deep stream, no scratch row | 831 | about 330 | 500 | open: MAJ-8 may bind (M4) |
| lm_head: S3 | 141 | about 100 | 40 | medium |
| layer 0's dense MLP: S4 | unmeasured | | | later |
| **total** | | | **570 certain, 1,250 possible** | |

The target needs 90 us. The two gang items alone clear it by the
arithmetic in either reading of w13; the per-tile item is the one the
round learns about.

## To decide before the split

1. K1 first (proven, exact products, the probe's registers) with K9's
   map, K3 as the second step if M5 shows the VALU term; K2 is gone.
2. The batch to build for: 8 rows with `#pragma unroll 1` (32 loads in
   flight, 158 VGPRs, no AGPRs), the constant swept at 4 and 16 on the VM.
3. S1's uneven tiles by the kernel's arithmetic from `tile_idx` (no
   table).
4. Whether MAJ-8 (the runtime's completion hierarchy) enters this round at
   all, or waits for M4's answer.
5. Whether M7 (the stream probe) is built: it is the only way to know
   w13's reachable number before the kernel is written, and it is about
   three hours of laptop work.
