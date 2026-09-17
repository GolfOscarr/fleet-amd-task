# 09 - Lessons of round 3

Written 2026-09-17 after the session (`07-session-log.md`, `08-results.md`).
Three parts: every approach with its verdict and evidence, the lessons by
kind, and the ranked list for the next round. Every number names its run
in `env/hw/20260917/runs/` or the log row that read it.

## The result

| | us per token | ids |
|---|---|---|
| round 2 (per-tile linears, nt weights) | 9,575 | equal |
| round 3, seven final runs (event clock) | 4,571 to 4,600 | equal |
| round 3, the megakernel's own clock, no instrumentation | 4,584, 4,590 | equal (29 steps) |
| the target (production vLLM, quoted) | 4,500 | |

2.09x in one session of 144 minutes ($7.12). The gap to the target is
1.6 to 2.2%; the lever that closes it is known and is a runtime change
(part 3).

## Part 1: the approaches, each with its verdict

| Approach | Hypothesis | Done | Evidence | Verdict |
|---|---|---|---|---|
| A. Find the attention's 110 us of "per-task overhead" (Group A of `01`) | the runtime costs about 100 us around every attention operator | the worker timing, the shader-clock spin, the empty-task ladder, the fence knobs (I1 to I4) | the exec counter of the attention was 36 to 43 us per task against a 145 us "gap"; the MFMA kernel cut the exec to 7.7 us and the named gap did not move while the next operator's did; the task graph showed event i is triggered by the (i - 1)-th operator | **the overhead did not exist**: the gap was `mla_prep`'s (one task, 143 us). The instruments were right; the table they were checked against was off by one |
| B. The boundary fusions (O1 to O3, 326 to 246 operators) | a boundary costs 13.6 to 40 us, so 80 fewer boundaries save 1.1 to 1.8 ms | `--fuse-norm2`, `--fuse-silu`, `--fuse-norm1`, each with a suite row, a step-0 compare, a 2-layer timing and a model row | all correct; the model 10,224 to 10,375 us against 10,251; the removed operators' true cost was 4 to 5 us each | **neutral**; kept (correct, fewer operators). A boundary costs 2.3 to 5 us |
| C. The MFMA attention (O7) | the VALU kernel's 34 us can be 10 | `--mfma-attend`, the layout tested by emulation on the laptop, the instruction first run on the VM | 9.25 us standalone against 33.8; the model 10,251 to 8,953 | **on**, as estimated |
| D. The weight prefetch by side operators (O8) | streaming the next linear's weight into the memory-side cache hides its bandwidth term | `--prefetch`, the runtime's side-operator branch, `task_graph_check.py` | the wiring PASS, correct, the model 12,487 against 10,224 | **off**: the prefetch tasks occupy the workers the linears need, and the linears were never bandwidth-bound |
| E. The per-head prep task | one task reading 2 MiB of `W_uk` is latency-bound; 16 tasks read 128 KiB each | the kernel takes its head from the runtime's task index; task 0 writes the cache rows; the suite launches 16 blocks | prep 143 to 10.6 us per layer; the model 8,904 to 5,208 | **on**: the largest single gain of the round (3.7 ms) |
| F. Batched loads in the router, the merge and the norm helper | each was one memory round trip per row or column, 11 to 16 round trips per thread | four expert rows per lane; a split batch and sixteen weight rows per thread with the FMAs in order; the norm's row read once with 16-byte loads | merge 20 to 13.7 us exec, router 24 to 16, the fused tiles 21 to 18; the model 5,000 to about 4,600 | **on** |
| G. Deeper batches (eight rows per lane, the weight half row at once) | fewer round trips still | the two constants | 1% slower on the model (4,621 to 4,654 against 4,578 to 4,600) | **reverted**: register pressure in the worker's union |
| H. The attention as regular per-split tasks (`--attend-tasks`) | the gang broadcast to 296 workers costs more than 33 regular tasks | one flag, the same MFMA kernel | 4,589 against 4,618 on the first pair, 4,597 against 4,576 on the rerun | **within the spread**; on |
| I. The fence knobs (I4) | a completion or acquire fence is the per-task cost | `MPK_NO_COMPLETION_FENCE`, `MPK_NO_ACQUIRE_FENCE` on the 2-layer graph with the step-0 compare | both fail the compare (stale data); no completion fence also no faster (626.8 against 618.3) | **off**; the sleep and CAS knobs not measured |
| J. The streaming loads (O6) | the attention's second pass and the merge's partials pollute the L2 | the `nt` suite build, `ktime` | 34.0 against 33.8 standalone | **not a lever**; not run in a graph |
| K. The probe (O5) | a one-task predecessor separates the producer's spread from the operator's cost | built, not run | the corrected attribution answered the question | **moot** |
| L. The CK memory pipeline for the linears | more K-steps in flight per tile | read, not built | `GemmPipelineAgBgCrMem` computes two prefetch stages for this tile, the same as the current pipeline; more stages need a policy rewrite | **next round, second** |

## Part 2: the lessons

### Measurement

1. **Verify the instrument's mapping before reading it.** `measure.py`
   named event i after operator i; the runtime fires event i when operator
   i - 1 completes. Every per-operator number of round 2 and of `01-ideas.md`
   was one row off, and a whole group of ideas (A) chased a cost that did
   not exist. The check that would have caught it costs one minute: dump
   the task graph and read which task type triggers each event
   (`fleet/task_graph_check.py` now does the side operators; the mapping
   check belongs beside it).
2. **Two clocks disagreeing is the signal.** The worker timing said prep's
   exec was 143 us while the table said 13.6; the table said the attention
   was 147 while its exec was 40. Either witness alone was explained away
   in round 2; together they were the answer in minutes.
3. **A lever that moves the wrong row is a measurement finding.** The MFMA
   attention left the "attention" gap unchanged and cut the "merge" gap by
   35 us. That pattern, not a hunch, exposed the shift.
4. **Correctness rows must match what the run holds.** A 32-iteration run's
   boundaries are iteration 31's; only a one-iteration run compares against
   the step-0 reference. Round 2 knew this (`L2_it1`) and the plan forgot
   it; the ids on the model row were the check that held throughout.
5. **The buffer capacity is not the count.** `num_events` in the event
   timing file is the runtime's allocation (498 for every graph); the
   iteration marker is the highest index that fires.

### Kernels

6. **At batch 1, single-task kernels are latency-bound, and the remedy is
   loads in flight, not bandwidth.** Prep, the router, the merge and the
   norm helper each issued one 2-byte or 16-byte load per row or column
   and waited; batching rows per lane with raw 16-byte words (converted
   on use, the FMA order kept) halved them, and splitting prep over the
   heads took it from 143 to 10 us. In a graph every round trip is an HBM
   one (the producer's write-back, the weights streamed once), so the
   in-graph time is about twice the standalone loop's.
7. **Registers bound the batch.** Eight rows per lane and a whole weight
   half row were 1% slower on the model than four and two batches: the
   worker's union sits at 253 VGPRs and a spill costs more than a round
   trip saves. Measure every step; do not extrapolate.
8. **Keep the accumulation order.** Every batched loop kept the FMA order
   of the scalar one, so the suites' bit-exact rows stayed exact and the
   ids never moved. The one place the order changed (the norm's per-thread
   grouping of eight elements before the block sum) was within the FP32
   noise the compare tolerates.

### The runtime

9. **The boundary floor is 2.3 to 2.9 us per operator, and 0.19 us per
   regular task on top.** The ladder measured it with empty tasks; the
   per-tile linears sit on that line (96 tasks 18 us predicted, 14.8
   measured; 64 tasks 12 predicted, 14.8 measured). They are not memory
   bound. The one event counter every task of an operator increments from
   eight XCDs is the serial cost; the gang path avoids it with per-XCD
   counters, and the runtime chunks the 400-task head into events of 8.
10. **More tasks is not more parallelism here.** Doubling the per-tile
    linears' grid would add 0.19 us per task; the gain must come from the
    completion path or from bytes in flight per task.
11. **Both runtime fences are needed.** Removing either fails the step-0
    compare, and removing the completion fence saves nothing: the atomic,
    not the fence, is the cost.

### Process and tooling

12. **The JIT reads the fork's installed copy of the task headers.** Seven
    graph rows ran old kernels after a push while the standalone suite
    (which reads the pushed files) passed with the new ones; the exec
    counters, byte-for-byte the old values, were the tell. Every stage now
    installs the headers first. Rule: after a kernel push, check that a
    counter moved before reading a clock.
13. **Push before the queue starts, and wait on the stage's own row.** A
    queue file written after the push fails the stage in a second; a wait
    on the queue's DONE line is satisfied by an earlier DONE of the same
    file. Count the lines, or read the last one.
14. **Never two graph runs at once, and know what a kill kills.** Killing
    the stage's shell left `queue.sh`'s loop alive, and the finals ran
    beside the knob rows for eight minutes; every row of both queues in
    that window was discarded and the finals rerun.
15. **Distinct names for repeated finals.** Three identical rows overwrite
    one run directory; the record kept only the last. Running the repeats
    at 30, 31 and 32 iterations kept all three.
16. **The plan's minute marks held because the rows were short and the
    rules were written.** The prepared levers' final was in hand at
    minute 47, the round's number at minute 100; the levers found on the
    machine took the second hour. Gains first, diagnostics after, was the
    right order: the diagnostics' answer came from the gains' rows.
17. **A changed runtime patch needs the pristine fork pushed first.**
    `setup.sh` refuses a changed patch over the old one; `FULL=1 L push`
    then `L start setup` re-applies the three patches in 73 s on 13 cores.

## Part 3: the next round, ranked

| Rank | Work | Where | Expected | Effort |
|---|---|---|---|---|
| 1 | a per-XCD completion hierarchy for regular tasks: eight local counters per event filled at prelaunch, the last task per XCD touches the global counter | `persistent_kernel.cuh` (the completion path, the prelaunch) | 82 per-tile operators at about 10 us each, about 0.8 ms per token; every multi-task operator benefits | a day with the empty-task ladder as the test (the per-op cost at N = 296 should fall from 56 us toward 10) |
| 2 | the CK linears with more K-steps in flight: a policy for `GemmPipelineAgBgCrMem` with the shuffled register distributions, KPerBlock 128 | `linear_norm_mi300.cuh` first (ours), then the stock tile and gang kernels via `gfx942.patch` | the 8-step loop at 1.4 us per step to about half; up to 1 ms across 2.7 ms of linears once rank 1 has removed the per-task cost | a day of CK work, compiled on the VM (seconds per try) |
| 3 | w13 in one round per XCD: 128-row tiles (22 per XCD) or the expert's rows over the 37 workers | the gang w13 kernel | 42 to about 25 us per layer, 0.4 ms | half a day |
| 4 | the router's GEMV over four tasks with the top-k as a second operator | the router kernel, the plan | 20 to about 10 us per layer, 0.25 ms | half a day |
| 5 | the iteration start (the prelaunch of 7,000 descriptors, 180 us) | the scheduler | follows rank 1 (fewer, wider operators) | |

The 4.5 ms target falls with rank 1 alone by the arithmetic above; the
design band of 1.2 to 1.4 ms needs ranks 1 to 3 and the bytes work of
`01-ideas.md` group F after them.

## Corrections to earlier documents

- `01-ideas.md`: the layer table and everything derived from it (Group A,
  the stack's "Group A" row) read one operator off; the note at its top
  gives the true table. The sums per layer stand.
- `02-local-gpu-split.md`, `03-local-preparation.md`: the expected gains of
  O1 to O3 (26 x 13.6 us) assumed the mis-attributed norm cost; the true
  cost was 4 us, hence neutral.
- `../02-validation/04-results.md`: the per-operator table's rows are
  shifted by one; its per-iteration numbers, the standalone timings and
  the correctness evidence are unaffected. Noted there.
- `05-session-plan.md`: three defects fixed in the session (the 2-layer
  compare at 32 iterations, the event clock's marker, the identical final
  names); the "as run" note at its top.
