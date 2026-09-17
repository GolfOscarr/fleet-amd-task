# 01 - Ideas for getting under 4.5 ms per token

Written 2026-09-17, after round 2 (`../02-validation/`), for the third GPU
round. The goal has changed: the required milestone and the end-to-end decode
are done, and the target is now a production vLLM baseline of **4.5 ms per
token** (222 tokens per second) on the same model and machine, quoted by the
recruiter; the figure is not in the task document and has not been measured
by us. Our best steady state is **9.58 ms** on the runtime's event clock
(`env/hw/20260916/runs/L27_head_it32_tile_nt`), so the gap is 2.13x. The
design band is 1.15 to 1.35 ms, so the target sits well above the bandwidth
floor: this is an overhead problem, not a bytes problem.

This page lists every idea worth considering, what round 2 says about it,
what it is worth, and what it costs. It does not choose; the split into
laptop work and VM work is the next page. Every number names its run.

## The arithmetic of the target

One MoE layer with both round-2 levers on (`--tile-linears --nt-weights`),
event gaps of the 2-layer graph, `runs/L2_it32_tile_al65536` and
`runs/L2_it32_nt_al65536`, from `../02-validation/04-results.md`:

| Operator | us | Bandwidth floor us | What the time is |
|---|---|---|---|
| input norm | 13.6 | under 1 | one task; almost all boundary overhead |
| Q and latent-KV projection, 15 MB | 4.6 | 3.5 | at the floor |
| `mla_prep` (norm, RoPE, append, q W_uk) | 13.6 | about 1 | ours, one task |
| `mla_attend`, 33 splits | 147.5 | 9 | ours; 34 us standalone, warm or cold |
| `mla_merge_uv` | 45.7 to 59.2 | about 1 | ours; 11.5 us standalone |
| `o_proj` with residual, 8 MB | 22.1 | 2 | the residual floor |
| post-attention norm | 13.6 | under 1 | one task |
| router | 3.6 | under 1 | ours |
| expert gate and up, 66 MB | 19.8 | 15 | near the floor |
| `moe_silu_mul` | 40.8 | under 1 | elementwise, 8 tasks |
| expert down, 33 MB | 5.6 | 8 | under its floor: mis-attributed to its neighbour |
| `moe_mul_sum_add` | 20.3 | under 1 | elementwise, 8 tasks |
| **one MoE layer** | **364** | **about 40** | 26 of these, plus layer 0 and the head |

26 x 364 us is 9.5 ms: the layer table accounts for the whole iteration.

The budget for 4.5 ms: the head (400 MB of `lm_head` at 4 TB/s plus the
argmax) needs about 120 us, which leaves **(4,500 - 120) / 27 = 162 us per
layer**. The bandwidth term of a layer is 40 us, so the allowance for
everything else is about 120 us per layer over 12 operators, **10 us per
operator boundary on average**. Today a norm alone, a task with microseconds
of work, costs 13.6 us, and the attention 147.

Two consequences shape the list:

- **No single family of ideas is enough.** Removing the attention and merge
  overhead entirely (147 to 40, 50 to 12) takes the layer to about 220 us,
  6.1 ms per token. Fusing every elementwise and norm boundary away without
  touching the attention takes it to about 246 us, 6.8 ms. Both together give
  about 110 us, **3.2 ms**. The plan needs the overhead found *and* the
  boundary count cut.
- **Bytes are not the lever yet.** FP8 weights halve a 40 us term per layer;
  worth 1 ms per token only once the layer is near 160 us. It stays last.

## What round 2 fixed as facts

Everything below is measured (`../02-validation/04-results.md`, `06-lessons.md`).

1. The attention kernel is a 34 us kernel for the graph's grid (8 x 5 tiles,
   step 1032, 33 splits), warm or cold up to 300 MB of rotating cache. In the
   megakernel it costs 146 to 149 us with non-temporal weight loads and 211 to
   215 without.
2. That in-graph number does not move with the dispatch path (a regular task
   per split, `--attend-tasks`), the prefetch depth (P6), or the rows per tile
   (`--split 17`, 61 splits). It does move with what the *other* kernels do
   to the cache: the stock linears' loads made non-temporal took it from 215
   to 147 (`runs/L27_head_it32_nt_al65536`).
3. The merge is 11.5 us standalone and 46 to 61 in the graph; the residual
   linears sit at 22 to 34 us whatever their size; the norms at 13.6; the
   elementwise MoE operators at 20 and 41 for 8 tasks each.
4. The plain and silu gang linears show 4 to 5 us for 15 to 92 MB, which no
   memory system delivers: the event gaps credit a producer's time to the
   consumer's wait, so every per-operator number is the gap between two
   completion events, not a kernel time.
5. The runtime's per-task path (`persistent_kernel.cuh`, worker loop): load
   the task descriptors into LDS by `cp_async`; one acquire load of the event
   counter plus an agent-scope acquire fence (`buffer_inv sc1`); the task
   (gang tasks loop over tiles strided by the XCD's worker count); then an
   XCD-local atomic on the event, and on the last worker of the XCD a
   `threadfence_gpu` (`buffer_wbl2 sc1` and wait), a release atomic on the
   global counter, and for a fired event a push onto a scheduler queue: a
   release atomic, a store, a fence, and a CAS loop, on the cross-XCD
   broadcast queue when the event's tasks span XCDs. The scheduler is one
   lane per XCD; it hands a task to a worker by a store, a fence and an
   atomic per worker, sequentially.
6. The runtime has two instruments we have not used: `MPK_TIMING=1` builds
   per-worker cycle counters (poll, dependency wait, execute, signal) and
   `MPK_DEVICE_ACCUM=1` accumulates nanoseconds per task class. As shipped
   the timing print covers workers 0 to 7 only and the per-class slots cover
   the stock types; a patch extends both to every worker and to our types.
   `MPK_EVENT_TIMING=1` is the event clock we used. The worker's `clock64`
   counts shader cycles while `s_memrealtime` counts a fixed 100 MHz, so the
   ratio of the two over one task is the SCLK during that task: the clock
   (A2) is measurable from inside the kernel without `amd-smi`.
7. rocprofv3 cannot attach to the torch wheel; counters come from a
   standalone binary. The kernel suite times its grids (`KT_TIME`, `KT_COLD`).
8. Our three MLA kernels are VALU versions "correctness first"; the MFMA
   16 x 16 x 16 version of the spec has not been written
   (`fleet/tasks/mi300/mla_attend_mi300.cuh`, header comment).

## Group A: find the per-task overhead (the critical path)

The 110 us between the attention's standalone and in-graph time is the
largest single item and its cause is unknown. Six hypotheses, each with the
measurement that separates it. None needs the GPU for its preparation; all
need it for the answer.

| Hypothesis | Why it is plausible | The discriminating measurement | Where it runs |
|---|---|---|---|
| **A1. The kernel itself runs slower inside the worker** (code generation: the attention inlined into the worker's switch over every task type with a 234-VGPR union; conservative `s_waitcnt` placement; prefetch loads serialised) | the in-graph time moves with cache state (215 to 147) as a latency-bound loop would; the standalone build has 124 VGPRs and its own waitcnt schedule | `MPK_TIMING=1`: the attention tasks' `exec_cycles` against the 34 us standalone. If exec is about 34 us the time is around the task (A3 to A5); if it is about 140 the kernel is the problem. Offline: disassemble the worker's attention region (`env/offline_gfx942/`) and count the `vmcnt` waits per split loop against the standalone kernel's | offline compile now; one run on the VM |
| **A2. Clock.** 296 workers spin on their queues (`s_sleep 1`, 64 cycles) while 40 work; the persistent kernel's power draw may hold the busy CUs at a lower SCLK than a standalone launch reaches | the VM refuses `amd-smi set --perf-level`; the clocks were never read during a run | a spin task that records `clock64` and `s_memrealtime` over a fixed loop, run standalone (the kernel suite) and inside the megakernel: the ratio is the SCLK in each setting; `amd-smi metric --clock` every 100 ms as the outer check if the VF allows the read; then raise the poll sleep (`s_sleep 1` to 8, 32, 127; a constant in the poll loop) and re-time | VM, cheap |
| **A3. Dispatch latency.** The single-lane scheduler pushes a gang task to each of its 37 workers sequentially (store, fence, atomic each); the worker then loads its descriptor batch from global | per-tile linears (96 tasks) cost 4 us, so a task hand-off is cheap on its own; but gang and regular tasks were the same 146 us. The layer table has a pattern: a one-task operator after a one-task operator costs 3.6 us (the router after the norm), a one-task operator after an operator spread over 8 XCDs costs 13.6 (the norms), 8 tasks after 8 gang tasks cost 20 to 41 (combine, silu), 40 tasks after one cost 147 (the attention). The cost depends on the producer's and consumer's spread over XCDs, not on the task count alone | a graph of M operators of N empty tasks each (our `copy` task on a 256-element tensor, alternating two tensors to satisfy the chain rule), N in 1, 8, 40, 296, with the consumer's tasks placed on one XCD or spread: the per-operator and per-task costs as a small table. `MPK_TIMING` gives poll and dependency-wait cycles directly | VM, about 12 runs of a few seconds |
| **A4. Completion cost.** `threadfence_gpu` on the last worker per XCD, the global release atomic, and the broadcast-queue push with a CAS loop under contention from 8 XCDs | the 703 ns hop and the 115 to 317 ns fences measured in round 1 are far below 100 us, but they were measured uncontended and with an empty L2 | remove the sites one at a time under a build flag (a patch): the `threadfence_gpu` at completion, the acquire fence at dispatch, the broadcast CAS; check correctness by the boundary compare, not only by the ids | VM, 3 to 4 runs |
| **A5. Occupancy of the worker kernel.** One workgroup per CU for every worker, 58 KB of dynamic LDS; the attention's 22 KB standalone allows more | the grid of 40 tiles fits 40 CUs either way; occupancy should not matter for one tile per CU | covered by A1's exec_cycles; if exec is high and the disassembly is equal, this is next | VM |
| **A6. The measurement mis-attributes.** An operator's gap includes its predecessor's tail | the linears' 4 us for 92 MB prove it happens; the attention's predecessor is the 13 us `mla_prep` | the empty-task graph of A3 also calibrates the gap; `MPK_TIMING` exec cycles are per task and immune | VM |

**What A settles.** If A1 is true, the remedy is on the laptop: build the
attention so its loop is not at the mercy of the union (a `noinline`
device function, its own register budget through `__launch_bounds__`
attributes, or the MFMA rewrite of Group C which changes the loop anyway).
If A2, the remedy is one constant. If A3 or A4, the remedy is in the runtime
and generic: it helps every one of the 326 operators, which is what the
4.5 ms budget needs (10 us per boundary).

## Group B: fewer boundaries

Every boundary costs at least a norm's 13.6 us today, and the budget allows
10. The graph has 12 operators per MoE layer; the design listed the fusions
(`docs/design-doc/09-expected-performance.md`, reductions 1 to 3) and the
runtime ships the kernels for most of them.

| Idea | Removes per layer | Worth per token today | How | Cost |
|---|---|---|---|---|
| **B1. Norm into the following operator.** Corrected in the double-check of 2026-09-17: the runtime's fused `rmsnorm_linear` task calls `norm_linear_task_impl` from `tasks/ampere/`, which the MI300 task header does not include, and `gang_rmsnorm_linear_mi300.cuh` is not included either, so no fused norm-linear exists on the gfx942 path as shipped. The cheap half is ours: the post-attention norm folds into our router, which reads `x_res`, writes the normalised row `h` as a second output and the routing from it (the expert gate-up reads `h` as today); the input norm folds into the head of the per-tile Q/KV linear as our variant of the stock per-tile linear (each of the 96 tasks normalises the 4 KB row into a private scratch row, then runs the CK linear on it); `model.norm` into `lm_head` the same way | 2 boundaries, 27 us | 27 x 27 = **0.7 ms**; the router half alone 26 x 13.6 = 0.35 ms | the router change is 30 lines of ours; the linear variant is a registration and a prologue (a day); the layer-0 dense `gate_up` keeps its norm unless the silu gang kernel gets the same prologue | 0.5 day for the router half, 1 day for the linear half |
| **B2. `moe_silu_mul` into the expert gate-up epilogue.** The dense layer already does this (`gang_linear_silu`); the MoE variant `gang_moe_w13` writes `mid` and a separate 8-task op applies silu-mul, 41 us for microseconds of work | 1 boundary, 41 us | 26 x 41 = **1.1 ms** | a silu epilogue in `gang_moe_linear_mi300.cuh` (the w13 kernel writes act8 directly); or as a first step, per-tile silu (66 tasks instead of 8) if the elementwise cost is per task | 0.5 to 1 day |
| **B3. `moe_mul_sum_add` into the expert down epilogue.** Corrected 2026-09-17: the 8 expert outputs would be scaled and added into `x_res` by atomics, but `x_res` is BF16 and eight BF16 roundings in an unknown order replace the reference's one FP32 sum; the runtime's own pattern (`splitk_linear_res_atomic`) uses an FP32 workspace plus a done-counter, and the last arriver converts, which is a second pass on one worker. Plausible only with the FP32 workspace and the counter; the boundary compare decides | 1 boundary, 20 us | 26 x 20 = **0.5 ms** | an epilogue in the w2 gang kernel with FP32 atomics into a workspace, a counter, the last expert writes `x_res`; the per-tile combine of C4 is the safer first step | 1 day; after C4 |
| **B4. `mla_prep` into the Q/KV projection's tail or into the attention.** Design reduction 3: the `W_uk` product once per XCD in the attention's prologue (2 MB per XCD, 16 MB per layer, 4 us of bandwidth) | 1 boundary, 13.6 us | 27 x 13.6 = **0.4 ms** | the attention gains a phase A (ours); the norm and RoPE of the cache row need one writer, so the append stays a small task or is done by tile 0 with a flag | 1 day; only after Group A, since it lengthens the attention task |
| **B5. Merge into the attention: the last split merges.** Corrected 2026-09-17: as first written the last of the 33 splits would read 1.1 MB of partials and 2 MB of `W_uv` on one CU, 30 to 60 us at a single CU's streaming rate, no better than today's 46 to 59. The version that holds: apply `W_uv` per split before the merge (exact up to rounding, since the merge is linear: the weighted sum of `o_s W_uv` equals `(sum w_s o_s) W_uv`), so the partials shrink from 516 to 129 floats per head and the last split's merge is microseconds; the `W_uv` read then happens once per split, 2 MB each, so the split count must drop to 8 (one per XCD, 128 rows each), which is only sensible with the MFMA kernel of C1 whose per-tile cost does not grow four-fold with the rows | 1 boundary, 46 to 59 us | 27 x 50 = **1.3 ms** | a kernel change (ours) with a per-layer counter, an agent-scope release before the increment and an acquire in the last split; depends on C1 and on Group A | 1 day after C1 |
| **B6. Merge into `o_proj`.** Alternative to B5: the residual linear reads `partials` and does the merge in its prologue per tile | same boundary | same | the o_proj tile would repeat the 33-split merge for its 16 heads per tile (64 tiles x 1.1 MB of partials: 70 MB of extra reads, 17 us); worse than B5 | not preferred |

B1, B2 and C4 (the per-tile elementwise ops) need no answer from Group A and
no runtime change; together with B3 and B5 in their corrected forms they take a
layer from 12 operators to 7. What is certain today is B1 and B2, about
70 us per layer, **1.8 ms per token**; the rest depends on C1 or on a
workspace pattern. They compound with Group A: whatever a boundary costs
after A, there are fewer of them.

## Group C: the kernels themselves

| Idea | Evidence | Worth | Cost |
|---|---|---|---|
| **C1. MFMA attention.** The VALU tile does 16 heads x 576 x 26 rows of dot products by scalar FMA with BF16 to FP32 conversion, and the same again for p x V (16 x 26 x 512); 34 us standalone against a 9 us bandwidth floor. The spec (`docs/mla-decode/04-our-kernel-spec.md`) is a 16 x 16 x 16 MFMA per head group, M = 16 heads fills the tile exactly; vLLM ships the same instruction shape on gfx942 | the standalone number is the kernel's own; it is the floor of the attention once Group A is solved | 27 x (34 - about 10) = **0.65 ms**, only after A | 1.5 to 2 days: the kernel, the numpy check unchanged, the suite; the offline compile checks the MFMA lowering |
| **C2. Non-temporal loads for our kernels' streams.** The cache rows are read once per iteration; `c_kv` and `k_pe` loads with the non-temporal policy the stock linears use under `-DMPK_NT_WEIGHT_LOADS` stop them evicting `ql_nope` and the partials that the merge re-reads; E2 on the stock kernels gave 2 ms | MAJ-6: the E2 mechanism is exactly this | uncertain: 0 to 0.5 ms; cheap | 2 hours, one run |
| **C3. The residual linears' floor.** `o_proj` (8 MB) 22 us and `down` (46 MB) 34 us in every variant, plain linears of the same size 4 us. The residual path reads and writes `x_res` in place; suspect the second pass or the workspace of the CK residual epilogue | unexplained since round 1 | 27 x (22 - 5) + (34 - 12) = **0.5 ms** | read `gang_linear_mi300.cuh`'s residual variant against the plain one on the laptop; then the split-K atomic variant (`gang_splitk_linear_mi300.cuh`) as a drop-in test |
| **C4. Per-tile MoE elementwise ops as a fallback for B2 and B3.** If the fusions slip, 66 silu tasks and 8 x 8 combine tasks instead of 8 and 8; the per-tile linears showed the runtime spreads tasks well | 0.7 ms of the per-tile linears came this way | up to **1.0 ms** if the cost is per task rather than per operator | hours; a plan change only |
| **C5. `mla_merge_uv` with more tiles.** 8 gang tiles of 2 heads; 16 (one head each) or 33 (one split each with an atomic sum) | 11.5 us standalone against about 1 us of bandwidth | 27 x 10 = **0.3 ms** after A | hours |

## Group D: use the idle machine

The chain is serial: while the attention runs on 40 workers, 256 idle; while
a norm runs on one, 295 idle. Fleet's model has no overlap across a
boundary, but nothing stops a task that depends on the *previous* event from
running beside the current operator.

| Idea | What it does | Worth | Cost, risk |
|---|---|---|---|
| **D1. Weight prefetch into the infinity cache by idle workers.** MI300X has a 256 MB memory-side cache (measured in round 1: a tier at 258 ns between L2 and HBM, `MAJ-6`). A prefetch task for operator k+1 is dispatched with operator k: it streams k+1's weights (66 MB for the experts, 8 to 46 MB for the linears) with ordinary loads and discards them, so k+1 reads them from the cache at up to 17 TB/s instead of 4. The MoE weights are known only after the router; the prefetch of the expert weights starts with the router's event. Corrected 2026-09-17: the runtime chains every operator to its immediate predecessor (`runtime.cc`, the event creation walks `pre_op` to `cur_op` and asserts a shared tensor; a task has one dependent event and one trigger event), so a prefetch operator cannot be registered beside operator k through the API: it needs a runtime patch that gives a flagged operator the dependent events of the operator after it, skips it as `pre_op`, and counts its tasks into the end-of-graph event | turns the serial `T_bw + overhead` into a pipeline: up to the 1.2 ms bandwidth term hidden, realistically 0.5 to 0.8 ms | a runtime patch of about 40 lines in `runtime.cc` (ours, checked by the dry run and a host compile on the laptop, built on the VM); a trivial task type; the E2 non-temporal loads bypass the memory-side cache on allocation, so the consuming linears must read with the allocating policy again, which is what E2 turned off. The order of the two experiments matters: D1 without E2, then D1 with E2 on our kernels only (C2) |
| **D2. Overlap the shared experts with the router.** The reference runs the shared-expert MLP beside the router; the design folded them into the routed set as experts 64 and 65 (D6) so 8 experts fill 8 XCDs. Nothing to gain here unless the per-task overhead is per task and not per operator | 0 | none |
| **D3. Split the attention across all XCDs' workers with smaller tiles** was tested (61 splits, unchanged). Do not repeat until Group A has an answer | 0 today | none |

## Group E: runtime constants and build knobs (cheap, test once each)

| Knob | Where | Why |
|---|---|---|
| E1. The poll loop's `s_sleep 1` to 8, 32, 127 | worker and scheduler loops | A2; also fewer L2 loads from 296 pollers per XCD |
| E2. Flag parity between the megakernel build (`-O3`, `persistent_kernel.py`) and the kernel suite's build: fast-math, denormal handling, `-mcumode`, the unroll pragmas | build | A1: the two binaries of the same kernel should be compiled alike before their timings are compared |
| E3. `__launch_bounds__(256, 1)` on the worker kernel, and the VGPR limit of the union read from the disassembly (`env/offline_gfx942/resources.txt`) | build | A1, A5 |
| E4. The worker queue depth and the descriptor batch size (`per_worker_queue_len`, `TASK_SIZE` chunks of the `cp_async` load) | runtime config | A3 |
| E5. The number of schedulers: one lane per XCD today (`num_local_schedulers` is a parameter of the runtime; whether the gfx942 path takes more than one per XCD is to be read in `persistent_kernel.cuh`) | `persistent_kernel.py` | A3, if the empty-task graph shows a per-task dispatch cost above 1 us |

## Group F: bytes (later)

| Idea | Worth | Why later |
|---|---|---|
| F1. FP8 weights for the experts and the projections (`docs/acceleration/`, 1.83x on bytes; a quantised checkpoint exists for this model) | the 40 us bandwidth term per layer to 22: **0.5 ms** at today's layer, up to 1 ms once the layer is near 160 us | the loader takes precision as a parameter and the CK linears have FP8 paths, but every stock kernel we use must accept the format; 2 to 3 days; nothing until Groups A and B are done |
| F2. FP8 latent cache | 15 MB per token | not worth it at 1,024 tokens (`04-technique-ledger.md`) |

## Group G: measurement, so the round can decide on the clock

| Idea | Why |
|---|---|
| G1. `MPK_TIMING=1` in the harness: a `--worker-timing` flag, the per-worker counters pulled and reduced per task class in `measure.py` | the instrument that answers A1 in one run; unused so far |
| G2. The empty-task graph as a harness mode (`--graph empty --ops M --tasks N`) with a no-op task type (ours, 5 lines) | A3, A4, A6 |
| G3. A vLLM measurement on the same VM: `rocm/vllm` image, the same model, 1,024-token prompt, 32 output tokens, the server's per-token latency and the same prompt for the ids | the target is a quoted number; the round should own it, with the clock and the context named |
| G4. The clock reader (`amd-smi metric --clock` in a loop) as a session helper, written into the run's record | A2, and every timing row from now on carries the SCLK |
| G5. The counters from a standalone binary for the attention and merge (the kernel suite under rocprofv3, `TCC_EA0_RDREQ`, `TCC_HIT`) | closes the traffic row the task asks for; secondary to the target |

## The stack, and a ranking

What the ideas add up to if each delivers its estimate, on the 9.58 ms
baseline (both round-2 levers on):

| Step | Cut, ms | Running total, ms | Depends on |
|---|---|---|---|
| B2 silu epilogue (or C4 per-tile silu as the first step) | 1.1 | 8.5 | nothing |
| B1 norm fusions (the router half first) | 0.7 | 7.8 | nothing |
| C4 per-tile combine, then B3 with the FP32 workspace | 0.5 | 7.3 | nothing; B3 after C4 |
| C3 residual floor | 0.5 | 6.8 | reading the kernel |
| Group A: attention overhead 147 to about 40 | 2.9 | 3.9 | the cause; if it is generic (A3, A4) every remaining boundary drops as well, about 1 ms more |
| C1 MFMA attention | 0.65 | 3.2 | A |
| B5 last-split merge with `W_uv` per split | 1.3 | 1.9 | C1 and A |
| D1 prefetch | 0.5 to 0.8 | 1.1 to 1.4 | A, the runtime patch, and the E2 policy sorted out |
| F1 FP8 | 0.5 to 1.0 | about 1 | everything above |

The certain part of Group B with C3 and C4 reaches about 6.8 ms; Group A alone about 6.7; the target needs
both, and the order of work follows from the dependencies: **Group A's
measurements first (they are cheap and decide the rest), Group B's fusions
prepared on the laptop meanwhile (they need no answer from A), then C1 and
D1 with whatever VM time is left.** At $2.99 per hour and $20.23 of credit
the round has about 6.5 hours of VM time; Group A's runs are minutes each,
Group B's validation runs are a minute each plus the compare.

## Not worth doing again

From `../02-validation/06-lessons.md`, so the round does not spend on them:
the prefetch depth of the VALU attention (P6, no change), smaller attention
tiles (61 splits, no change), the attention as regular tasks (no change),
a cold-cache explanation of the standalone number (34 us cold and warm),
address padding for the fault (fixed at the cause), and rocprofv3 on the
torch wheel (cannot attach).
