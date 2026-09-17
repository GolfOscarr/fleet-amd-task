# 02 - The work split: laptop first, then the VM

Written 2026-09-17 from `01-ideas.md` after a double-check of every idea
against the runtime's source and our code. The rule of round 2 holds: no
minute of VM time goes to work that can be done on the laptop. Every item
below has a deliverable, a check that runs on the laptop, a time box and the
VM row that consumes it; every VM row names the laptop item it needs, the
PASS text and the decision it feeds. The details of each item are filled in
on the next pass; this page fixes the shape.

## What the double-check changed

| Idea in `01` | What the source says | Consequence |
|---|---|---|
| B1, the norm fused into the following linear with the runtime's kernel | `rmsnorm_linear` registers `norm_linear_task_impl` from `tasks/ampere/norm_linear.cuh`, which the MI300 task header does not include; `gang_rmsnorm_linear_mi300.cuh` exists as a file and is not included either | B1 is our work: the post-attention norm folds into our router (it writes the normalised row), the input norm into our variant of the per-tile linear |
| B3, the weighted sum by atomics into `x_res` | `x_res` is BF16; the runtime's own accumulate pattern (`splitk_linear_res_atomic`) uses an FP32 workspace and a done-counter | the per-tile combine (C4) goes first; B3 needs the workspace pattern and a rounding check |
| B5, the last split merges | one CU streaming 3 MB is 30 to 60 us, not a gain over today's 46 to 59 | B5 holds only with `W_uv` applied per split and 8 splits, which needs the MFMA kernel (C1); it moves behind C1 |
| D1, a prefetch operator beside operator k | the runtime chains every operator to its immediate predecessor (`runtime.cc`: the event creation walks `pre_op` to `cur_op` and asserts a shared tensor; one dependent and one trigger event per task) | D1 needs a runtime patch (about 40 lines); the dry run and a host compile check it here, the VM builds it |
| A1 and A3, `MPK_TIMING=1` as the instrument | the print covers workers 0 to 7 and the per-class slots cover the stock types; `clock64` counts shader cycles, `s_memrealtime` a fixed 100 MHz | a patch prints every worker and adds slots for our types; the ratio of the two counters over a task gives the SCLK (A2) without `amd-smi` |
| A3, the empty-task graph | our `copy` task on a 256-element tensor is an empty task; alternating two tensors satisfies the chain rule | no new task type; a plan mode in `graph_plan.py` |
| the layer table | a one-task operator after a one-task operator costs 3.6 us, after an 8-XCD operator 13.6; 8 tasks after 8 gang tasks 20 to 41; 40 tasks after one 147 | the boundary cost depends on the spread of producer and consumer over XCDs; the empty-task graph varies both |

Everything else in `01` stands as written.

## Local part (the laptop, before the VM)

Ordered by what the VM rows need first. L1 to L4 are the instruments and
the empty graphs for Group A; L5 to L8 are the certain fusions of Group B
and C4; L9 to L11 are prepared so the second VM session has them if Group A
allows; L12 to L14 are the session tooling. The checks are the ones round 2
used: the tests, the dry run, the syntax check, the offline gfx942 compile.

| Item | Deliverable | Laptop check | Time box | Feeds |
|---|---|---|---|---|
| **L1. Worker timing in the harness** (A1, A3) | `run_fleet.py --worker-timing` sets `MPK_TIMING=1`; a patch hunk prints the `[TIMING]` and `[TASK_TIME]` lines for every worker and adds slots for `TASK_MLA_PREP/ATTEND/ATTEND_TILE/MERGE_UV/MOE_ROUTER/COPY`; `measure.py` reduces the lines per task class into `worker_timing.json` (exec, poll, dependency and signal cycles, per class and in total) and prints exec us per task class against the standalone numbers | the patch applies on the pristine fork; `measure.py` parses a fixture of the lines; a test | 3 h | G2, G3 |
| **L2. The SCLK from inside the kernel** (A2) | a spin mode in our `copy` task (`params[1] = spin iterations`) that records `clock64` and `s_memrealtime` at entry and exit into the output tensor; the same loop in the kernel suite (`KT_SPIN`); `measure.py` reports the ratio as MHz | the suite's dry run; the host syntax check; a test of the ratio arithmetic | 2 h | G2 |
| **L3. The empty-task graph** (A3, A4, A6) | `graph_plan.py --graph empty --ops M --tasks N --spread {one,all}`: M operators of N `copy` tasks over two alternating 256-element tensors, the spread over XCDs read back from the per-worker timing lines (the scheduler places the tasks, not the plan); `run_fleet.py` accepts it with `--event-timing`; a queue file with N in 1, 8, 40, 296 and both spreads, 12 rows | the dry run's counts; the chain check; a test per mode | 3 h | G3 |
| **L4. The fence knobs** (A4) | `gfx942.patch` hunks under `MPK_NO_COMPLETION_FENCE`, `MPK_NO_ACQUIRE_FENCE`, `MPK_NO_BCAST_CAS` that remove one site each, plus the poll sleep as `MPK_POLL_SLEEP` (default 1); `run_fleet.py --runtime-flags "..."` passes them through `MPK_EXTRA_HIPCC_FLAGS`; the correctness check for these runs is the boundary compare of layer 1, not the ids alone | the patch applies; the offline compile of the worker with each flag (`env/offline_gfx942/`) and the fence count read from the disassembly per flag | 3 h | G4 |
| **L5. The router writes the normalised row** (B1, the cheap half) | `moe_router_mi300.cuh` reads `x_res`, computes the RMS norm with the post-attention norm weight in FP32, writes `h` (BF16) and routes from it; `numpy_ref.py` gains the same; the plan drops `L{l}.norm2` for MoE layers (layer 0 keeps its norm); the kernel suite covers the new output; the boundary B7 (the normalised input) is still compared, now against the router's output | the suite's dry run, the numpy test, the dry run's op count (326 to 300), the offline compile | 3 h | G5 |
| **L6. Per-tile silu and combine** (C4, the first step of B2 and B3; skipped 2026-09-17: L7 is in) | `moe_silu_mul_layer` with a grid of 66 tasks and `moe_mul_sum_add_layer` with 64, under `--tile-moe`; the imaps checked against the kernels' indexing (the runtime hands each task an offset pointer, as the per-tile linears showed) | the dry run; the wrappers' assertions; a test | 2 h | G5 |
| **L7. The silu epilogue in the w13 gang kernel** (B2) | a copy of `gang_moe_w13_linear_kernel` under our patch that applies silu-mul to its output tile before the store, writing `act8` directly; the plan drops `L{l}.silu`; `numpy_ref.py` unchanged (the math is the same) | the patch applies; the offline compile; the syntax check; a test on the plan | 4 h | G5 |
| **L8. The norm prologue of the per-tile linear** (B1, the other half) | our registration `linear_norm_mi300` of the stock per-tile linear with a prologue: each task normalises the 4 KB input row into a private scratch row (a `[96, 2048]` workspace) and runs the CK linear on it; used for `L{l}.qkva` and `head.lm_head`; the plan drops `L{l}.norm1` and `head.norm` | the offline compile; the dry run's op count (300 to 272); the numpy check of the composition; a test | 5 h | G5 |
| **L9. The residual linear read** (C3) | a reading of `linear_kernel_ck` with `residual_add = true` against `false` (`linear_ck_mi300.cuh`): what the residual path does twice; if the cause is visible, a variant under our patch; if not, a queue row with `gang_splitk_linear_with_residual_layer` as the drop-in | the notes in this set; the offline compile if a variant is written | 3 h | G6 |
| **L10. The MFMA attention** (C1) | `mla_attend_mi300.cuh` with the 16 x 16 x 16 BF16 MFMA for the scores (M = 16 heads, K = 576 in 36 steps, N = the rows of the pass) and for p x V (N = 512 in 32 tiles), the online softmax unchanged; the numpy reference unchanged; a build flag selects VALU or MFMA | the offline compile with the MFMA lowering read from the disassembly (`v_mfma_f32_16x16x16_bf16`), no spills; the suite's dry run | 1.5 days | G7 |
| **L11. Non-temporal loads for our cache streams** (C2) | the `c_kv` and `k_pe` loads of `mla_attend` and the partials loads of `mla_merge_uv` under the same policy the stock linears use with `-DMPK_NT_WEIGHT_LOADS` (read the intrinsic they use, apply it under `-DMLA_NT_STREAMS`) | the offline compile; the `nt` bit visible in the disassembly | 2 h | G6 |
| **L12. The vLLM row** (G3 of `01`) | a session stage `vllm`: pull `rocm/vllm`, serve the model, one request with our 1,024-token prompt and 32 greedy tokens, the per-token latency from the server's metrics and the wall clock, the ids compared with `harness/ref/ref_output_ids.json`; the image is large, so the stage runs in the background from minute 2 as the docker build did | `DRY=1` prints the commands; shellcheck; a test of the status row | 3 h | G1 |
| **L13. The session plan of this round** | `03-session-plan.md` in the shape of round 2: the rows, the PASS text, AUTO or DECIDE, the thresholds, the queue files, the budget; `05-rehearsal.md` regenerated by `rehearse.sh` | the rehearsal runs in DRY mode; the queue guards' tests | 3 h | every G row |
| **L14. The prefetch runtime patch** (D1), only if time remains before the VM | the flagged operator in `runtime.cc` (dependent events copied from the following operator, skipped as `pre_op`, its tasks counted into the end-of-graph event), a `prefetch_layer` in the Python API, our prefetch task, the plan's `--prefetch` mode | the dry run; a host compile of `runtime.cc` with the stub headers; a test | 1 day | G8 |

About six working days for L1 to L13 with L10 the largest; L14 is a day
more. L1 to L4 first (one day), L5 to L8 second (two days), then L12 and L13,
then L10 and L11, then L9; L14 last.

## GPU part (the VM)

Two sessions on one 1x MI300X, as in round 2, with the image
`ghcr.io/golfoscarr/fleet-amd-task:20260916` (no Fleet build). Session C is
the measurements and the certain fusions; session D is the remedies that
depend on session C's answer. Each row is a queue file; the PASS text is the
status row of `queue.sh`; DECIDE rows are reported before the next row.

### Session C (about 2.5 hours)

| Row | What | Needs | PASS | Decision |
|---|---|---|---|---|
| **G0** | provision by `grab.sh`, `preflight`, `setup` (the new patches), `reference`, `kernels` (the suites with L5's router output) | L1 to L8 | the status rows as in round 2; 7 suites 100 of 100 | AUTO |
| **G1** | the vLLM stage in the background from minute 2 | L12 | `PASS vllm`: the per-token latency and the ids | records the target on our clock; DECIDE nothing |
| **G2** | the baseline with worker timing: `L2_it32` and `L27_head_it32` with `--tile-linears --nt-weights --worker-timing --event-timing`, and the spin task standalone and in a graph | L1, L2 | `worker_timing.json`: exec us per class; the SCLK in both settings | **DECIDE A1 against A2 to A4**: attention exec about 34 us means the time is around the task; about 140 means the kernel; an SCLK gap of more than 10% means the clock |
| **G3** | the empty-task ladder: 12 runs, `--event-timing --worker-timing` | L3 | per-operator and per-task cost against the task count and the placement | **DECIDE the remedy**: a per-operator cost above 10 us or a per-task cost above 1 us points at the runtime (G4); otherwise at the kernels (G7) |
| **G4** | the fence knobs, one at a time, on the 2-layer graph with the layer-1 boundary compare; the poll sleep at 8, 32, 127 | L4 | `compare=PASS` at every boundary and the event gaps | which site carries the time, if any; a knob that passes the compare and cuts the gaps stays on |
| **G5** | the fusions: `--tile-moe`, then the router norm, then the silu epilogue, then the norm prologue, each on `L2_it32` with compare and on `L27_head_it32` with the ids | L5 to L8 | `output_ids PASS`, the boundary compare within threshold, the event clock per token | each lever stays on if correct and faster; expected 9.58 to about 7.5 ms |
| **G6** | the residual variant (split-K residual as the drop-in or our variant) and the non-temporal streams of our kernels | L9, L11 | compare and the event clock | keep if faster |
| end | pull, snapshot, push the branch, delete on the user's yes | | | |

### Session D (about 2.5 hours, after the laptop has acted on session C)

| Row | What | Needs | PASS | Decision |
|---|---|---|---|---|
| **G7** | the MFMA attention: the suites, then `L2_it32` and `L27_head_it32` | L10; G2's answer | 100 of 100; compare PASS; the attention's exec and gap | if the kernel is the floor (A1) this is the fix; else it is the standalone gain |
| **G7b** | the remedy of Group A that session C named: a fence removed for good, a poll constant, a `noinline` attention with its own register budget, or a scheduler change | the laptop's work between the sessions | the per-token time on the event clock | the number that decides whether 4.5 ms is in reach: with G5 in place, an attention near 40 us is about 4 ms |
| **G8** | B5 (the last-split merge with `W_uv` per split, 8 splits) and, if L14 exists, the prefetch | L10, L14 | compare PASS; the event clock | the last 0.5 to 1.5 ms |
| **G9** | the final `L27_head_it32` with every lever on, three clocks, the vLLM number beside it; counters on the suite binary for the traffic row | | | the report |
| end | pull, push, delete on the user's yes | | | |

### Budget

| | |
|---|---|
| Credit | $20.23 at $2.99 per hour billed per minute: 6.7 hours |
| Session C | 2.5 hours, $7.5 |
| Session D | 2.5 hours, $7.5 |
| Reserve | 1.7 hours for a re-provisioning, a rebuild, or a third short session |
| The image pull | 25 GB, about 10 minutes, inside G0 |
| The vLLM image | several GB more, in the background |

## Dependencies at a glance

```
laptop                                   VM
L1 worker timing  ─┐
L2 SCLK spin      ─┼──────────────► G2 ─┐
L3 empty graphs   ─┼──────────────► G3 ─┼─► DECIDE the cause ─► laptop: the remedy ─► G7b
L4 fence knobs    ─┘──────────────► G4 ─┘
L5 router norm    ─┐
L6 per-tile MoE   ─┼──────────────► G5 (certain; no answer needed)
L7 silu epilogue  ─┤
L8 norm prologue  ─┘
L9 residual read  ─┬──────────────► G6
L11 nt streams    ─┘
L12 vLLM stage    ────────────────► G1
L10 MFMA attention ───────────────► G7 ─► G8 (B5 needs it)
L14 prefetch patch ───────────────► G8
L13 session plan  ────────────────► every row
```

## What is not in this round

FP8 (F1), the shared-expert overlap (D2), the attention with more splits
(D3), and anything on the list of `01`, "Not worth doing again".
