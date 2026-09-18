# 01 - Ideas for the final stage

Written 2026-09-18 after round 4's session (`../04-kernels/09-session-log.md`,
`10-results.md`, `11-lessons.md`). The decode stands at 4,262 to 4,341 us
per token on the event clock (`FWD_PASS` 4,267 to 4,310), the 32 ids equal
on every final, below the 4,500 target. This is the last round and a
light one: nothing here is a new kernel. The page lists what round 4 left
open, what each item is worth, what it costs and how it is checked; the
split into laptop and VM work is `02-local-gpu-split.md`, the items in
detail `03-local-preparation.md` (whose Part 4 double-checked this page
against the source and changed A1, L2 and N3), the progress record
`04-checklist.md`. Every number names its run or its source line.

## What round 4 left

| | |
|---|---|
| The stack that made the number | `--tile-linears --nt-weights --event-timing --fuse-norm2 --fuse-silu --fuse-norm1 --mfma-attend --attend-tasks --nt-streams --gemv-linears --linear-grid 48 --merge-tasks --merge-halves 2 --runtime-flags=-DMPK_W2_CK_TILE` (thirteen flags and a define; `queue-f5.txt` as it ran) |
| The finals | fifteen 48-task runs, 4,262 to 4,341 us; the best set `NO_LOCAL_CAS` with `POLL_SLEEP=8`: 4,265 to 4,307, `FWD_PASS` 4,267 (`10-results.md`, "The number") |
| The two things tried at the end | the finals with `POLL_SLEEP=8` beside `NO_LOCAL_CAS` (`queue-g3.txt`), then with `-DGEMV_BATCH=4` (`queue-g4.txt`): both within the run-to-run spread on the model, both 1 to 7% better on the 2-layer graph |
| The checks that are not green | the it32 compare rows fail `head.B15.logits` (the boundaries are dumped from the last iteration, `run_fleet.py`, `boundary_dump`: "boundaries are from iteration iters - 1, not decode step 0") and the route log (MIN-32: 309 mismatches over the fifteen finals, 287 of them a single expert, 251 of those the lowest-weighted slot); round 3's finals fail the same way |
| The laptop items | MIN-35 (`--worker-timing` hangs the round-4 header), MIN-36 (the half merge without the GEMV linears faults on 27 layers), the merge's standalone 16.3 us against round 3's 11.4, MIN-32's tie rule |
| The balance | $4.69 at $2.99 per hour: 94 minutes of VM time, of which the $3 stop rule leaves 34 |

## Group A: the two things tried at the end of round 4

### A1. The runtime knobs `MPK_NO_LOCAL_CAS` and `MPK_POLL_SLEEP=8`

What they switch (`fleet/patches/new_tasks.patch`, the I4 knobs of round
3): a worker that finishes a task publishes the event's new ready position
to its scheduler's queue. For a same-XCD queue the stock path is an
`atom_add_local` followed by a CAS loop that waits until the slot before
it has been published (in-order publish); `MPK_NO_LOCAL_CAS` replaces the
loop by a plain store, with the patch's own caveat: "with concurrent
producers a later slot can be published before an earlier one".
`MPK_POLL_SLEEP` is the `s_sleep` argument of an idle worker's and
scheduler's poll loop, 1 unit of about 64 cycles by default; 8 makes an
idle wave sleep eight times longer between polls.

What round 4 measured (`10-results.md`, "The levers"): on the 2-layer
graph against the no-knob row (498.7 us per iteration) `NO_LOCAL_CAS`
487.2 (-2.3%), `POLL_SLEEP=8` 492.4 (-1.3%), `NO_BCAST_CAS` unchanged,
`POLL_SLEEP=32` +2.4%, `=127` +15%; every knob's compare PASS. On the
model, the finals with `NO_LOCAL_CAS` 4,262 to 4,341 and with both knobs
4,265 to 4,307, against 4,288 to 4,340 without: within the spread.

What remained to decide was not speed but safety, and the reading is
done (`03-local-preparation.md`, Part 4): the scheduler consumes its
queue by position (it loads the published position and reads every slot
up to it; the slots carry no flag of their own), and a producer reserves
a slot, writes its event index, fences, then publishes through the CAS
loop that waits until the position equals its slot. With the store form a
worker that reserved slot n+1 can publish n+2 before the worker at slot
n has written its event index, and the scheduler reads a stale slot. Two
of an XCD's 37 workers finishing tasks of the same operator together is
the common case, so the hazard is reachable; four finals and five 2-layer
rows not hitting it is what a rare race looks like. `POLL_SLEEP=8` has no
such question: a longer sleep only delays an idle wave's next poll.

Worth: at most 1% of the token, likely nothing. Cost: one VM set.
Decision: `NO_LOCAL_CAS` is unsafe by construction and stays a probe (its
round-4 numbers stand as measured); `POLL_SLEEP=8` enters the default
stack if the final session's interleaved sets confirm it is not slower
on either clock.

### A2. The batch constant 4

What it switches: `GEMV_BATCH` (`fleet/tasks/mi300/linear_gemv_mi300.cuh`,
`#ifndef`, default 8), the rows a wave holds in flight in the GEMV linear:
8 rows are 32 loads of 16 bytes per lane before the first wait, 4 rows
16. Round 4's G0 chose 8 standalone (batch 4 wins the norm form by 7%,
batch 8 the residual form by 15%); on the 2-layer graph batch 4 was 7%
faster per iteration (463.2 against 498.7 us; qkva 10.6 against 11.2,
o_proj 6.8 against 8.2); on the model 4,292 to 4,323 against 4,265 to
4,307 with the same knobs: equal.

Why the 2-layer gain vanishes on the model is the one question worth a
row: on 2 layers the linears are a third of the iteration, on 27 with the
head a seventh, and the shorter batch's saving is per task, not per byte
(fewer registers, an earlier first wait). The candidates are the
run-to-run spread (the model's sets differ by 1% among themselves) and
the event clock's own bias (lesson 3 of `11`: the two clocks disagreed by
3% on the grid A/B). One set of three with `FWD_PASS` rows on both
constants, interleaved, decides; the default stays 8 unless batch 4 wins
both clocks.

Worth: 0 to 1%. Cost: one VM set (5 minutes). Decision: the default stays
8 unless the interleaved set says otherwise.

## Group C: the compare, so the finals are green

### C1. The route log's tie rule (MIN-32)

The rule today (`harness/compare.py`, `compare_route_log`): exact set
equality of the top-6 ids per (step, MoE layer); any difference is FAIL.
The record says what the differences are: over the fifteen finals 309
mismatches out of 832 (step, layer) pairs per run, 287 a single expert
swapped, 251 of those the reference's lowest-weighted slot, 22 the
fifth, 13 the fourth, 1 the third; 22 mismatches differ in more than one
expert, every one of them at a step after a single swap in the same run
(the swapped expert changes the hidden state, and the steps after it
drift). At step 0 there is one swap (MoE layer 4: expert 49 at weight
0.0396 out, expert 2 in), present with the round-3 flags on the round-4
header, so it is the header's summation order at a near tie.

The rule to write: a mismatch is a tie when exactly one expert differs
and the reference's weight for the expert that left is within a tolerance
of the reference's weight for the expert that came in. The second weight
is not in the log today (the reference keeps its top-6 only), so the
reference capture (`harness/run_reference.py`, the route log) stores all
64 router weights per (step, layer): 64 x 26 x 32 floats, 213 KB as JSON,
or the logits before the softmax if the tolerance is to be the router
floor (3.59e-3 relative, `harness/ref/calibration.json`). The rule
counts three classes and reports them: ties (within the tolerance),
cascades (a mismatch at a step after a tie in the same run, judged by the
ids of that step instead) and disagreements (everything else); only the
third class fails. The output ids stay the check of the whole decode.

Worth: the finals' compare rows read PASS for the first time since round
2, with the reason on the page. Cost: two hours on the laptop, one
reference stage on the VM (74 s). Check: the rule on the fifteen finals'
records gives zero disagreements; a synthetic test with a swap outside
the tolerance fails.

### C2. Iteration-aware boundaries

`run_fleet.py` dumps the boundaries from the last iteration and says so in
the run's notes ("boundaries are from iteration N, not decode step 0");
`compare.py` compares them against the reference's step 0 anyway, so an
it32 row fails `head.B15.logits` by construction (relative error 0.8 to
1.2 against 0.164; the layer boundaries pass because layers 0 and 1 are
compared and their inputs at step 31 happen to be within tolerance of step
0's, which is luck, not a check). The fix is one of two:

- the cheap one: `compare.py` reads the run's notes and reports every
  boundary of a later iteration as "not comparable (iteration N)", neither
  PASS nor FAIL, and the row's verdict is the ids; the queue's `compare`
  word then means the same on every row;
- the right one: the reference captures the boundaries of the last
  decode step too (`ref_boundaries_step31.safetensors`; the head's logits
  and layer 1's boundaries at step 31 are the interesting ones), and the
  compare picks the file by the run's iteration. `run_reference.py` already
  runs the 32 steps; the capture is a second dump at the end.

Worth: the same as C1, the finals green. Cost: two hours for the cheap
form, four for the right one (a reference stage on the VM, 74 s, for
either). Check: the compare of a round-4 it32 record reports the head as
not comparable (cheap) or PASS against step 31 (right); the it1 records
unchanged.

## Group L: the laptop items of round 4

### L1. `--worker-timing` hangs the round-4 header (MIN-35)

The facts (`09-session-log.md`, 05:07): the round-3 stack on the round-4
header completes 32 iterations without `--worker-timing` and hangs at one
iteration with it; the round-3 header ran the timing build for a whole
session. The timing switch (`new_tasks.patch`, `[TASK_TIME2]`) has eight
classes in arrays of eight, so the array is not the bug. What the timing
build adds per task is a `clock64` pair and the class switch; what it adds
per worker at the end is a `printf` of seventeen arguments. Candidates,
in order of likelihood: the `printf`'s argument count or format under the
round-4 union (the worker is at 256 VGPRs with 8 spills in every variant;
a `printf` with more live values at the end of a 256-VGPR worker can spill
into scratch the launch did not size, and the round-3 union had room);
a class case for a type the round-4 dispatcher reaches through the gang
path (`TASK_GANG_MOE_W13_GEMV_MI300` sits in the class switch, the stock
gang path sets the type before the switch runs, but the gang w13 form
was off in every hung row, so this is unlikely); the timing buffer's size
against the round-4 task count (6,593 tasks per iteration against round
3's 6,593 with the same flags: no).

The diagnosis is offline: the `union` variant of `env/offline_gfx942/run.sh`
with `-DMPK_ENABLE_TIMING` (a new variant, `unionT`), its resource line
against the plain union's (scratch and spills), and the disassembly of the
worker's epilogue around the `printf`. If the scratch grew, the fix is the
`printf` split into two of eight arguments or the class totals written to
a device buffer the host prints (the cleaner form; the host already
parses the lines). The check is the offline compile and, on the VM, one
2-layer row with the timing build.

Worth: the exec-per-class table for the results page (the kernels' own
time beside their gaps), not speed. Cost: three hours. If the offline
pass finds nothing, the item is recorded as open and the round runs
without it, as round 4 did.

### L2. The half merge without the GEMV linears faults on 27 layers (MIN-36)

The facts: `L27_head_it1_..._rf_w2cktile_mt_mh2` (the round-3 CK linears
with `--merge-tasks --merge-halves 2`) faults with `hipErrorIllegalAddress`
before any forward pass; the same at 2 layers passes its compare; the same
with `--gemv-linears` runs every final. The plan builds all three on the
laptop without an assert (`build_graph.py --dry-run --layers 27
--tile-linears --merge-tasks --merge-halves 2`: 326 operators, 7,269
tasks; with `--gemv-linears` 298 and 7,241), so the plan's shapes are
legal and the fault is in an address a task computes. The first suspect, the
stock CK tile's 16-row read past a one-row `attn` (MIN-33), is out:
`attn` is a `[1, NH x D_V]` plan tensor and `build_graph.new_workspace`
backs every single-row activation with 16 rows. What differs between the
two builds that run and the one that faults is the consumer of `attn`:
the stock per-tile `linear_with_residual` in the faulting build, our GEMV
linear in the running one. The readings left (`03`, F5): the stock
linear's input map against the merge tile registration's whole-tensor
maps (the event structure both consumers get), the two graphs' task
types side by side, and the tile form's store offset at `halves = 2`
against the registration's output map; a `--layers 3 --merge-halves 1`
row on the VM separates the halves from the layer count.

Worth: nothing for the number (the finals do not use the configuration);
a correctness hole closed. Cost: two hours of reading, one VM row (45 s)
to confirm. If the readings find nothing, a layer bisect on the VM (3, 5,
9 and 14 layers at one iteration, 4 minutes) locates the first faulting
layer count for the record.

### L3. The merge's standalone regression

`mla_merge_uv` reads 16.3 us on every round-4 build against 11.4 to 11.5
on round 3's three (`10-results.md`, the standalone table), while its gap
in the graph is 22.0 us in both rounds. N3 changed the phases (the
`W_uv` batch first, the partials batch with the lse words) and N5 added an
optional LDS copy of the attn values (`attn_s`, a null pointer for the
gang and tile forms). Since the graph does not see the 5 us, the item is
for the record: the `ktime` row of round 3's kernel file (restored beside
the round-4 one under a define, as the w2 CK path was) on the same
binary would say whether the 5 us is the kernel's or the launcher's
(`KT_TIME` runs the gang launch of 8 workgroups: a per-XCD serialisation
the graph's tile form does not pay). The cheaper reading: the standalone
merge runs the gang form (8 workgroups, one per XCD, the heads looped
inside); the graph runs the tile form (32 or 64 tasks). Round 3's 11.4
was the gang form too, so the comparison is fair and the 5 us is N3's.

Worth: none for the number. Cost: an hour offline (the two files' wait
sequences side by side in `dev_kt.s`) and one `ktime` row. Decision: done
if the hour finds the cause; otherwise recorded as a known regression of
the standalone form that the graph does not pay.

## Group D: the defaults and the record

### D1. The final stack as the default

`run_fleet.py`'s flags are every one off by default (`../04-kernels/05-local-preparation.md`),
so the number's command is thirteen flags and a define, copied by hand into
every queue file and into the report. A single flag `--final` (or the
flags' defaults flipped, with `--no-...` forms to turn each off) makes the
number's configuration the one a reader can run: `run_fleet.py --layers
27 --head --iters 32 --final`. The flipped defaults are the cleaner form
(no second code path) but touch every test that asserts the stock
counts; the `--final` preset is one function that sets the thirteen flags
and the define unless the row names them, and one test. The queue files
of this round use it; the round-4 files stay as they ran.

Worth: the report's reproducibility. Cost: two hours. Check: the dry run
with `--final` gives the finals' counts (246 operators and 6,386 tasks,
the record's `plan.json`), `test_queue_files` on the round-5 files.

### D2. The results across the rounds, one page

The report needs one table from 9.58 ms (round 2) through 4.57 to 4.60
(round 3) to 4.26 to 4.34 (round 4) and this round's final, with the
configuration, the clock and the ids check of each, and the per-operator
table of a MoE layer across the rounds (`../03-acceleration/08-results.md`
and `../04-kernels/10-results.md` have the two halves). This page is
`07-final-numbers.md` of this set, drafted before the session with the
round-4 numbers in place and filled from the record after it; the report under `docs/report` (never committed) draws on it.

## Group N: light optimisations that fit a final stage

Listed for completeness; none is required for the number, and the budget
(34 minutes of VM time under the $3 rule, 94 without it) holds them only
if the rule is waived.

### N1. Three interleaved sets instead of three consecutive ones

The round-4 finals ran each configuration as three consecutive rows, and
the sets differ by 1% among themselves while the two clocks disagree by
3% on one A/B (`11`, lesson 3). The final session's finals interleave the
configurations (A, B, A, B, A, B at 30, 31, 32 iterations) so the drift
of the host across the minutes falls on both; the `FWD_PASS` rows the
same. Worth: a number with an honest spread. Cost: nothing, the queue
file's order.

### N2. `POLL_SLEEP=8` and the batch constant as defaults (A1, A2)

If the interleaved sets say so; otherwise the defaults stay.
`NO_LOCAL_CAS` is out by A1's reading.

### N3. The head's event count (a plan constant)

The head's 400 tasks run as 50 events (49 gaps of 2.3 us in the record,
111 us per token, `../04-kernels/10-results.md`). The runtime makes the
event count between two operators the gcd of the producer's and the
consumer's partitions per dimension (`runtime.cc`, "number of events is
the product of gcd of producer/consumer"); the head's consumer is
`argmax_partial` at `ARGMAX_SLICES = 50` tasks (`fleet/graph_plan.py`,
line 21, the design's D13), and gcd(400, 50) = 50. The constant is the
plan's. At 8 slices the pair makes 8 events (gcd(320, 8) = 8 as well at
`--head-grid 320`) and each argmax task scans 12,800 logits instead of
2,048 (the stock kernel takes `num_elements / num_tasks` per task). Worth:
up to 42 boundaries of 2.3 us, about 95 us per token (2%), against a
longer last argmax task of a few microseconds. Cost: one plan argument,
one flag (`--argmax-slices N`), one test, one 2-layer head row and one
model compare row (`03`, F7). The round's one speed item, and it is one
line of arithmetic once the gcd rule is read.

## Routes not taken, and why

- Bytes in flight by direct-to-LDS loads (`11`, Part 3, rank 1): a new
  kernel form for every weight stream, the offline pass, the suites, a
  session of its own. The largest remaining lever (about 1 ms) and out of
  a final stage's scope; the next reader starts there.
- The fusions without a serial tail (rank 2): prep into the attention,
  the combine into w2's epilogue, the gate GEMV into the norm; each a
  kernel change with a registration. Out of scope for the same reason.
- The iteration start (rank 3) and the router's latency (rank 4):
  runtime and kernel work respectively.
- MAJ-8's per-XCD completion hierarchy: the runtime's scheduler; a round
  of its own.
- The w2 GEMV form, w13 in one round, the four-task router, the o_proj
  fold: measured and off in round 4; nothing light changes their
  verdicts (`../04-kernels/11-lessons.md`, Part 1).
