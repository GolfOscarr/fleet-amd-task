# 11 - Lessons of round 4

Written 2026-09-18 after the session (`09-session-log.md`) from its record
(`env/hw/20260918/`) and the numbers page (`10-results.md`), in the shape
of round 3's `../03-acceleration/09-lessons.md`: the result, each approach
with its verdict, the lessons by area, the mistakes with what each cost,
the next round ranked, and the corrections to the earlier pages of this
set.

## The result

Round 3 ended at 4,571 to 4,600 us per token; round 4 ends at 4,262 to
4,341 (fifteen finals, the medians 4,586 against 4,306), 3.5 to 5.3% below
the 4.5 ms production baseline, every final's 32 ids equal to the
reference's. Three of the seven prepared kernels carry it: the batch-1
GEMV linear (qkva, o_proj, the head), the router one level deeper, and the
merge as regular tasks at two halves per head. The other four were
measured and are off. The session took 167 minutes and $8.22 of the 210
and $10.47 budgeted; about 35 of those minutes went to two hangs and one
wrong reading.

## Part 1: the approaches, each with its verdict

| Approach (page) | Prepared as | Measured | Verdict |
|---|---|---|---|
| the batch-1 GEMV linear in place of the CK tile (`01`, L1 to L3) | `--gemv-linears`, batches of 8 rows, 32 loads in flight per lane, the norm and the residual fused | o_proj 13.0 to 8.2 us, qkva 14.0 to 11.2 (2 layers); 831 to 638 us per token | **on**; the round's largest gain (0.19 ms per token) |
| the task count of the linears (`01`, MAJ-8's test; G2) | `--linear-grid 48`, `32` | 519 / 498 / 532 us per 2-layer iteration at 96 / 48 / 32; the per-layer qkva gap rises with fewer tasks (11.2, 13.5, 15.9) | 48 in the stack by the event clock (3% on the finals; `FWD_PASS` indifferent); the linears are not completion-bound at 96 |
| the load policy (`01`, L1c's open item; G1.3, G5) | `--nt-streams` on every round-4 row, one row with plain loads | 518.8 against 525.5 with the GEMV linears; standalone 13% faster on the norm form; the stream probe indifferent (2,242 against 2,257 GB/s) | **kept**; MIN-34's `vmcnt(0)` after 8 to 12 loads stays unexplained |
| the batch constant (`01`, M5; G0) | `-DGEMV_BATCH=4`, `16`, `-DGEMV_STRIDED` through `--runtime-flags` | standalone 4 wins the norm form, 8 the residual form, 16 spills, the strided map loses 40%; in the graph 4 is 7% faster at 2 layers and equal on the model | **8**; the 2-layer graph over-rewards the batch |
| the w2 GEMV form as the header's fused-silu default (`01`, L3) | the multiply over 64 rows from an LDS row, the CK multiply under `-DMPK_W2_CK_TILE` | 34.4 us against the CK multiply's 21.4 on the 2-layer graph; its inline CK fallback returned ids `[0]` and hung | **off**; round 3's CK file restored verbatim as the define's path; the form needs the bytes-in-flight work of Part 3 before it can win |
| w13 in one round per XCD (`01`, L4) | `--gemv-w13`, 37 tiles of 77 rows per expert, 296 tasks | 36.6 us against the stock's 40.7; the iteration unchanged (495.1 against 495.4); the stream probe's gang ceiling 39 to 41 us | **off** by the 33 us threshold; both forms sit on the streaming ceiling, so the tile count was not the cost |
| the router one level deeper (`03`, N1; the header) | two batches of 8 rows per wave through `StreamSrc`, the writes over the lanes | 25.2 to 14.3 to 16.7 us per layer (2 layers); 516 to 424 per token | **on**; a gain of the header, no flag |
| the router in four tasks (`03`, N2) | `--router-tasks`, a counter, the last task's top-k | 13.9 us against 15.5, the iteration 2% slower (505.9 against 495.1) | **off**; the last task's serial top-k and the counter's round trip eat the split |
| the merge one level deeper (`03`, N3; the header) | `W_uv` batch first, the partials batch with the lse words, one barrier | 16.3 us standalone against round 3's 11.4; 22.0 in the graph, the same as round 3 | no change in the graph; the standalone loss is unexplained (an LDS copy of the attn values, added for N5, is the suspect) |
| the merge as regular tasks (`03`, N4) | `--merge-tasks`, `--merge-halves 2` | 20.9 (one task per head), 15.0 (two halves) against 22.0; 489 to 406 per token | **on** at two halves; the gang path's per-XCD serialisation was the cost |
| the merge with o_proj folded in (`03`, N5) | `--merge-oproj`, per-head partial products, a last-task reduction | 42.5 us against 21.9 for the half merge and the GEMV o_proj; 39 us warm standalone; compare PASS on `x_res` | **off**; a last-task reduction costs more than the boundary it removes (twice now: N2 and N5) |
| the head at 320 tasks (`01`, G6) | `--head-grid 320` | 661 against 651 us per 2-layer iteration with the head | **off** |
| the stream probe (`01`, L6; G5) | `--graph stream`, regular and gang forms | 2,242 GB/s (the gang shape), 775 (96 x 152 KB), 1,283 (296 x 256 KB) | the ceiling of the round, Part 3's first item |
| the runtime knobs (MIN-33; G9) | `MPK_NO_LOCAL_CAS`, `NO_BCAST_CAS`, `POLL_SLEEP=8/32/127`, each with its compare | -2.3%, 0, -1.3%, +2.4%, +15% at 2 layers, every compare PASS; within the spread on the model | measured and closed; `NO_LOCAL_CAS` and `POLL_SLEEP=8` harmless, in the best set |
| the bit-diff (`02`, L7; G8) | `queue.sh bitdiff` of two runs' boundaries | the GEMV linears move q by 2 ULP in 2 of 3,072 elements, the router logits by up to 1.7e-3, the top-k ids by nothing | the numerics paragraph of `10` |

## Part 2: the lessons

### Measurement

1. The finals are judged by the ids. The it32 compare rows fail
   `head.B15.logits` (the boundary is captured after the reference's step)
   and the route log (MIN-32's tail-expert flips) in round 3's record
   exactly as in this one; the plan's PASS text for the finals
   (`compare=PASS on the two compare rows`) was never achievable and led
   to a wrong bisect (Part 4, mistake 3). The one-iteration model rows are
   the boundary check; the 2-layer rows the per-lever check.
2. The 2-layer graph over-rewards levers that shorten a layer's critical
   path: `GEMV_BATCH=4` (-7% at 2 layers, 0 on the model), the 48-task
   grid (-4% at 2 layers, 3% on the finals' event clock and 0 on
   `FWD_PASS`), `NO_LOCAL_CAS` (-2.3%, 0). A 2-layer gain under 5% needs
   the model before it is called a gain.
3. Two clocks disagree by 3% on the grid A/B (the event clock 4,288 to
   4,325 at 48 tasks against 4,428 to 4,458 at 96; `FWD_PASS` 4,305
   against 4,289). Both are reported; the plan names the event clock.
   Every set of three should be followed by its `FWD_PASS` row, as the
   queue files do.
4. The standalone `ktime` numbers predicted the graph well where the
   kernel is the cost (the fold's 39 us warm, the merge's equal standalone
   and graph deltas) and badly where the graph's shape is (the norm GEMV's
   9.4 us standalone against 11.2 in the graph; batch 4 against 8).
5. The stream probe is the round's most useful new instrument: one row
   per access shape, no model, 30 s each. It says the machine's task-shaped
   reads reach 2.25 TB/s and that the w13 forms sit on that line, which no
   per-kernel number could say.
6. The exec-per-class table (`[TASK_TIME2]`) was lost to MIN-35; the
   event gaps carried every decision. The JIT-trap rule (the exec counter
   must move before a clock is read) was replaced by the gaps differing
   from the CK row's, which held.

### Kernels

7. A weight stream in this runtime is bound by bytes in flight per CU,
   not by the kernel's inner loop: the GEMV linear (128 KB per workgroup
   in flight) beat the CK tile (32 KB) by a third; w13's GEMV form and the
   stock kernel both sit on the 2.25 TB/s line; batch 16 spilled. The
   next step is bytes in flight without VGPRs (direct-to-LDS loads), not
   deeper batches.
8. A last-task reduction (a counter, one task summing the others' work)
   loses twice: the four-task router (13.9 against 15.5 in the split, the
   iteration slower) and the o_proj fold (42.5 against 21.9). The serial
   tail plus the agent-scope counter costs more than the boundary it
   removes. Fusions must keep every task independent.
9. The gang path's per-XCD serialisation is a cost by itself: the same
   merge kernel as regular tasks at two halves per head drops from 22.0
   to 15.0 us with no change to the arithmetic.
10. A fallback that only compiled on the laptop is not a fallback. The
    CK w2 path was compile-checked (`w2ck`) and hung on the VM; the file
    that had actually run round 3 was the fallback. Keep the previous
    round's file verbatim under the define, and let the new form be the
    experiment.
11. The batch constant and the lane map are graph decisions: standalone,
    batch 4 and 8 split the two GEMV forms; in the graph 4 wins 2 layers
    and nothing on the model. The `--runtime-flags` define (no push, no
    rebuild) made that a 90 s question.

### The runtime

12. `--worker-timing` (`MPK_ENABLE_TIMING`) hangs the round-4 header
    (MIN-35): a timing build must be one of the offline variants and one
    of the `kernels`-stage builds before a session, since the plan's rows
    depended on it.
13. The task-shaped read ceiling (2.25 TB/s in the gang shape, 0.8 to 1.3
    in the regular shapes) is the runtime's: one wave per SIMD, one
    workgroup per CU, so the bytes in flight per CU are the kernel's
    registers. It bounds every weight-streaming kernel of this design.
14. The per-operator fixed cost (MAJ-8) is not what bounds the GEMV
    linears at 96 tasks (fewer, longer tasks cost more per layer), but it
    is what the small operators are: prep 9.8 us, `mul_sum_add` 5.0, the
    norms 4 to 5, the head's chunks 2.3 per event.
15. The half merge without the GEMV linears faults on 27 layers with the
    head (MIN-36): a configuration no row had run. Every flag pair the
    finals could carry needs its own one-iteration model row.

### Process and tooling

16. A queue row can hang, and the queue must own the watchdog: 20 minutes
    went to the first hang before `timeout ${ROW_TIMEOUT:-600}` existed.
    A hang looks like a running row (100% GFX activity, the host process
    at 190% CPU); the giveaway is the run log not moving.
17. A `pkill -f <pattern>` whose pattern appears in the ssh command line
    kills the ssh session (twice). Kill by anchored pid.
18. Probes with `timeout` outside the queue were the fastest diagnosis
    on the clock: four rows of 43 s each settled both hangs (the define,
    then the worker timing) where the queue had cost 30 minutes.
19. Read the previous round's record before blaming a lever for a compare
    result on a row shape that round also ran (Part 4, mistake 3).
20. The DRY rehearsal (`08`) cannot see a wrong directory argument (the
    o_proj `ktime` row); a `--n 1` run of each stage against a generated
    trial would have. The suites, the `ktime` variants and the queue rows
    all ran through the rehearsal and passed; the two hangs were in what
    the rehearsal cannot execute.
21. The queue files as they ran are the record: every mid-session edit
    was committed with the decision that caused it (queue-f2 to f9, g1 to
    g4), and a file's rows were restored to what ran when a later edit
    would have misrecorded it (queue-g1's `--gemv-w13`).
22. The mid-session `L pull` stages the record and the next `git commit`
    sweeps it into whatever is committed (mistake 5); commit the record
    right after the pull, before any other commit.

## Part 3: the next round, ranked

1. **Bytes in flight for the weight streams** (w13 1.1 ms, w2 0.6, the
   GEMV linears 0.6 per token). The stream probe's 2.25 TB/s ceiling
   against the machine's 5.3: the direct-to-LDS staging form of `01`
   (`buffer_load ... lds`) holds bytes in flight without VGPRs, so a
   workgroup can keep 256 KB or more moving. The probe rows measure the
   form before any kernel changes. Up to 1 ms per token.
2. **Fusions without a serial tail** (MAJ-8's boundaries): prep into the
   attention (9.8 us per layer), `mul_sum_add` into w2's epilogue (5.0),
   the router's gate GEMV into the norm task that already streams x (a
   boundary and a 2 MB read). About 0.3 ms per token for the three.
3. **The iteration start** (192 us per token, 4.5%): the runtime's
   per-iteration host work; round 3 measured 125 to 181.
4. **The router's latency** (16.3 us for 2 MB): eight tasks writing the
   logits with the top-k done by the following operator's first task, no
   counter.
5. The laptop items: MIN-35 (the timing build), MIN-36 (the fault), the
   merge's standalone 5 us, MIN-32's tie rule and a step-aware head
   boundary in `compare.py`.

## Part 4: the mistakes, with what each cost

| Mistake | Cost | The rule now |
|---|---|---|
| 1. The queue had no per-row watchdog; the plan's "1.6 minutes per row" was taken as protection | 20 minutes of VM time on the first hang, 10 on the second | `timeout ${ROW_TIMEOUT:-600}` on every row (`queue.sh`) |
| 2. `pkill -f 'queue.sh run'` and `pkill -f 'harness/run_fleet.py'` matched the ssh command itself | two dropped sessions, one probe rerun | kill by anchored pid; `pgrep -f '^bash env/session/queue.sh run'` |
| 3. The 48-task grid blamed for the finals' compare FAIL on a one-iteration bisect, before reading that round 3's it32 final fails the same way | one finals set (which became the grid A/B, so the record gained a number) | lesson 1 and 19; the plan's finals row now says "ids PASS" |
| 4. The CK w2 fallback path compile-checked only | a hang, the probes, the restoration on the clock (about 15 minutes) | lesson 10 |
| 5. The pulled record swept into the T11 queue-edit commit | a `reset --soft` and two commits | lesson 22 |
| 6. The `ktime` o_proj row's directory | a `vm.sh` fix and three hand reruns (3 minutes) | lesson 20 |
| 7. The plan's finals PASS text and its trust in `--worker-timing` | the FAIL rows of S10 read as defects; the exec table missing | lessons 1 and 12; the corrections below |

## Corrections to earlier pages of this set

- `07-session-plan.md`, S10 and S14: the finals' PASS text is `output_ids
  PASS` and three per-token times within 2%; the it32 compare rows fail
  `head.B15.logits` and the route log by construction (round 3's record
  too). The page carries a corrections note.
- `07`, S3 to S14: every graph row assumed `--worker-timing`; it hangs the
  round-4 header (MIN-35) and every row ran without it.
- `07`, G3 and T4: the CK multiply is round 3's file, not the inline path
  the header carried; the pair ran as `queue-f7.txt`.
- `02-local-gpu-split.md` and `01-gemv-ideas.md`: the stack "0.57 ms
  certain, 1.25 possible" for the GEMV items and "0.37 certain, 0.8
  possible" for the router and merge came to 0.19 (the linears) and 0.17
  (the router and the merge) per token; the certain figures assumed the
  w2 and w13 forms would win, and the stream probe says why they did not.
- `05-local-preparation.md`, L3: "the CK fallback path is unverified here
  (no CK headers on the laptop): the offline `w2ck` variant is its check"
  was not a check; the path hung.
- `06-checklist.md`, N3: the merge's offline line was 166 VGPRs; the
  session's standalone time is 16.3 us against round 3's 11.4, a
  regression the checklist's emulation could not see.
- `../03-acceleration/09-lessons.md`, Part 3: "the per-task cost hierarchy
  is the lever that reaches 4.5 ms" was not needed to reach it; the
  kernels did, and MAJ-8 remains for the small operators.
