# 08 - Implementation milestones

Five GPU days after the local work in `10-local-work.md`. Each milestone
has an entry condition, an exit criterion that is evidence rather than a
feeling, and a fallback. Two gates decide the shape of the remaining days
(`00-decisions.md` D25).

## The ladder

| Milestone | Exit criterion | Evidence artifact |
|---|---|---|
| **M0** environment | Fleet builds for gfx942 and runs a trivial graph; model downloaded; reference run and capture done on the machine | `env/check_day1.log`, `ref_*` artifacts of `07-correctness.md` |
| **M1** one operator through the Fleet path | `qkv_a_proj` (reused gang linear) and then each new kernel validated in isolation and in a truncated graph | `correctness_report.md` rows B2; kernel tests |
| **M2** layer 1 validated (**required**) | truncated graph through layer 1, one iteration: B1-B13 within threshold, B9 exact | `correctness_report.md` for layers 0 and 1 |
| **M3** N consecutive persistent layers | layers 0..N-1 in one graph, one iteration, B13 at every layer | growth curve |
| **M4** end-to-end decode | full graph, 32 iterations, 32 ids exact | `fleet_output_ids.json` == `ref_output_ids.json`; latency and traffic numbers |
| **M5** FP8 (stretch) | weight-only FP8 experts, same correctness protocol | not scheduled |

## Day by day

### Day 1: build, verify the machine, decide (gate 1)

In order; each item is short and each gates the next.

1. `env/setup.sh`: ROCm version, PyTorch-ROCm, `rocminfo`, `amd-smi` partition
   query (expect SPX + NPS1); capture into `env/check_day1.log`.
2. Apply the known gfx942 fixes (`OPEN-PROBLEMS.md` MIN-27): drop the
   `paged_attention_decode_minimal_mi300.cuh` include; then
   `AMDGPU_TARGETS=gfx942 pip install -e . -v`. Fix what fails next; the
   `16x16x32` warp GEMM selection is the likely second item.
3. Run the smallest shipped graph (`cpp_examples` or `tests`) and confirm
   `[WORKER_XCD]`, `[SCHED_XCD]` output: 296 workers, 8 schedulers,
   `sched_id == xcd`, `worker mod 8 == xcd` (`03-synchronization.md`).
4. Disassemble: `buffer_wbl2 sc1` in `threadfence_gpu`, `buffer_inv sc1`
   in the dependency check, `sc1` on the counter poll (MAJ-3).
5. Probe CK: `grep` the ROCm `ck_tile/ops/fmha` headers for a 576/512
   configuration; compile a one-file instantiation at
   `(kM0 = 16, kQKHeaddim = 576, kN1 = 512)` (Q11). Record the answer; it
   picks the `mla_attend` implementation for day 3.
6. Download the model (31 GB); run `run_reference.py` and `calibrate.py`
   in the background while 2-5 proceed.
7. `rocprofv3 --list-avail`: confirm the counter names in
   `docs/mi300x/06-profiling.md`.

**Gate 1 (end of day 1).** Fleet builds and runs a graph on gfx942, or it
does not. If not: switch to the fallback runtime (below) for the rest of the
week and aim at M2 only.

### Day 2: M0 closed, M1

1. Weight packing on the GPU (`pack_weights.py`): fused `W_qkva`, padded
   layer-0 MLP, 66-expert `W13`/`W2`, `W_uk`/`W_uv` views; checksum against
   the checkpoint sums.
2. Builder script runs and generates `task_graph_0.json` and `kernel_0.cu`
   for a truncated graph of `embed` + `rmsnorm` + `qkv_a_proj`; compile;
   run one iteration; B2 within threshold. **M1 for a reused kernel.**
3. `moe_router_mi300`: kernel tests (random inputs versus NumPy), then in a
   truncated graph through layer 1's router (attention ops stubbed by
   copying the reference's `x_res` after layer 0 into the residual); B8,
   B9 exact, B10.
4. `mla_prep_mi300`: kernel tests; in the truncated attention graph; B3 and
   B4 at layer 0.
5. Register-usage check after each new kernel (`-Rpass-analysis=kernel-resource-usage`).

### Day 3: M2 (gate 2)

1. `mla_attend_mi300`: the CK instantiation if day-1 item 5 succeeded,
   otherwise the spec kernel with one split first. Kernel tests: 1 split
   versus 33 splits, and versus NumPy. Debug-scores build for B5.
2. `mla_merge_uv_mi300`: kernel tests; B6 on the truncated graph.
3. `o_proj`, norm, then the MoE ops of layer 1 with the forced experts;
   `moe_silu_mul` and `moe_mul_sum_add` with `k = 8`.
4. Full truncated graph through layer 1, one iteration:
   `compare.py` for B1-B13. **M2.**
5. First measurements of layer 1 alone: `[FWD_PASS]`, event timing,
   `TCC_EA0_RDREQ` bytes versus the 159.6 MiB prediction
   (`09-expected-performance.md`).

**Gate 2 (end of day 3).** M2 validated, or not. If not: days 4 and 5 go
to finishing M2, and the report documents the exact boundary at which the
path diverges and why. Do not start M3 with M2 red.

### Day 4: M3, M4

1. Layers 0..3 in one graph (M3 at N = 4): growth curve, B3 at each layer.
2. All 27 layers plus the head; the `argmax_reduce` variant; one iteration:
   B15, B16.
3. 32 iterations: the 32 ids. **M4** if exact. If a later id diverges, the
   route log shows the first (layer, step) at which routing differed, which
   localizes it to a layer and a boundary class.
4. The full measurement matrix (`09-expected-performance.md`): five
   generations, median and P95 per token, bytes, bandwidth, L2 hit rate,
   occupancy, VGPR/LDS per task, launches.

### Day 5: measurement, write-up, and one optimization

1. Re-run every measurement from a clean process; commit the raw outputs.
2. The two cheapest experiments from `06-optimization-strategy.md`:
   `USE_NT_WEIGHTS=1` and the `P_split` sweep (tile count per XCD 3, 5, 9).
3. The report: milestone reached, correctness rows, metrics, Fleet-native
   versus new, known failures, recommended next steps with the FP8
   arithmetic.
4. FP8 (M5) only if M4 was closed on day 4 with time to spare: it is the
   loader's precision parameter plus the CK pipeline's FP8 types, and it
   is not planned.

## Fallback runtime (if gate 1 fails)

A minimal persistent kernel of our own, targeting M2 only:

- One launch, 296 worker workgroups plus one scheduler; a static task list
  for layer 1 (12 ops, 68 tasks) with a per-op counter; workers spin on the
  op counter with the `buffer_inv sc1` acquire and signal with
  `buffer_wbl2 sc1` release, as in `docs/mi300x/03-memory-model.md`.
- Kernels: our four new ones plus straightforward GEMV, norm, SiLU and
  weighted-sum kernels written for M = 1 with VALU (no CK dependency).
- Evidence: the same `compare.py` rows for layer 1.
- The report states that the Fleet runtime did not build, with the exact
  error and the fixes attempted.

This costs the whole week and yields M2 only; it is the floor, not the plan.

## What is deliberately not on the schedule

- Prefill in Fleet, batching, speculative decoding, multi-GPU: out of scope.
- Tuning the CK linears or writing VALU GEMVs: only if a linear measures
  below the band (`00-decisions.md` D18).
- Expert-affinity placement across steps: the route log will say whether it
  could matter; it is a next-step recommendation, not a day-5 item.
