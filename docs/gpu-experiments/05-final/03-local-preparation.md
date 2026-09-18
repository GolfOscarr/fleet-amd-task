# 03 - Local preparation: the final stage, item by item

Written 2026-09-18 from `01-ideas.md` and `02-local-gpu-split.md`, in the
shape of round 4's `../04-kernels/05-local-preparation.md`: for every
laptop item its direction (the decision it serves), its core approach at
the level of files, functions and rules, the files it touches, the checks
that run here, the time box and the VM row it feeds. Part 3 is the order
of work; Part 4 the double-check of this page against the source, done
while it was written (three of the ideas changed: A1 is settled by
reading, L2's first suspect is out, N3 is a plan constant and not a
runtime one).

The rules of the round: no new kernel; every change behind a flag or a
define with the stock behaviour kept; every deliverable checked on the
laptop before the VM; the checklist (`04-checklist.md`) ticked only when
the check has run; no names, no emoji, `docs/report` never committed.

## Part 1: the items

### F1. The route log's tie rule (C1 of `01`)

Direction. The finals' compare rows read FAIL on the route log in every
round since round 2 (MIN-32) while every output id matches; the record of
round 4 says the mismatches are near ties (309 over the fifteen finals,
287 a single expert swapped, 251 of those the reference's lowest slot,
the 22 multi-expert ones all after a swap in the same run). The rule
classifies instead of failing.

Approach.

1. The reference stores the weights the rule needs.
   `harness/run_reference.py`: `route_entry` reads `cap.store[f"route.L{l}"]`,
   the gate's output tuple (indices, weights) captured by
   `cap.out_tuple(model.model.layers[l].mlp.gate, ...)` (line 136), and
   keeps the ordered top-6. The gate module's forward computes the scores
   over all 64 experts before its top-k; the capture hooks the module's
   output, which is the top-k pair only. The cheapest source of all 64
   weights is the router logits the reference already computes for the
   boundaries (`L{l}.B8.router_logits`, `F.linear(h_in, w_gate)` at line
   297, layer 1 only): the same expression for every MoE layer at every
   step, its softmax as the weights. `route_entry` gains a third field per
   layer, `"w_all"`: the 64 softmax weights (rounded to 7 decimals; 64 x
   26 x 32 values, about 0.6 MB in `ref_route_log.json`; the log's readers
   (`route_analysis.py`, the compare) take the extra key without change).
   The hidden state the logits need is the layer's post-attention normed
   input, which the capture has for layer 1 at step 0; for every layer and
   step it is one more hook on the gate's input (`cap.inp(gate, key)`, the
   capture's forward pre-hook, as `L{l}.layer_in` and `B6.attn` use it),
   and the softmax of `h @ W_gate^T` in float32. The log's top-6 weights
   are that softmax unnormalised (they sum to 0.49 at step 0, layer 0), so
   `w_all` agrees with them element for element.
2. `compare_route_log(ref_log, fleet_log, floor)` in `harness/compare.py`
   classifies each mismatching (step, layer):
   - `tie`: exactly one expert differs, and
     `|w_all[out] - w_all[in]| <= tol`, with `out` the reference's expert
     that Fleet dropped, `in` the one it took, `tol` the router floor
     (`calibration.json`, `floor.router`, 3.59e-3 in round 4's reference)
     times the larger of the two weights, or an absolute 5e-4 when the
     calibration is absent (the record's step-0 swap: 0.0396 out, the
     incoming weight is not in the log today; the rule's tolerance is set
     on the first reference with `w_all` and written into the page);
   - `cascade`: the (step, layer) comes after a `tie` or another
     `cascade` at an earlier step of the same run; judged by that step's
     output id (the ids row already compares it);
   - `disagreement`: everything else (more than one expert differs at a
     step with no earlier tie; one expert differs by more than `tol`).
   The result carries the three counts and the first disagreement; the
   verdict is FAIL only on a disagreement; the report's line names the
   counts ("Route log: PASS, 32 steps, 1 tie, 3 cascades, 0
   disagreements").
3. The old reference format (no `w_all`) keeps the old exact rule, so the
   round-4 record still compares as it did; the report says which rule
   ran.

Files. `harness/run_reference.py` (`route_entry`, the capture hook),
`harness/compare.py` (`compare_route_log`, `write_report`, `run`),
`harness/common.py` if the weights file is separate, `harness/README.md`
(the rule), `harness/tests/test_compare.py`.

Checks. `test_compare.py`: a synthetic log with (a) one swap within `tol`
gives `tie`, (b) one swap outside gives `disagreement` and FAIL, (c) a
two-expert difference two steps after (a) gives `cascade`, (d) the old
format runs the exact rule. The replay on the record is a test over the
fifteen finals' `fleet_route_log.json` against the round-4 reference (no
weights, so the exact rule and its mismatch list): every mismatch is a
single swap or a multi-expert difference after a single swap of the same
run, which is the structure the tie rule expects; the tolerance itself is
first read on the VM (R1). `test_run_reference_smoke.py` with the new field.

Time box: 2 hours (3 with the capture hook). Feeds R1, R2.

### F2. Iteration-aware boundaries (C2)

Direction. `run_fleet.py` dumps the boundaries after the last iteration
(`boundary_dump`, the note "boundaries are from iteration N, not decode
step 0" in `fleet_run_meta.json`) and `compare.py` compares them against
`ref_boundaries_step0.safetensors`; an it32 row's head boundary fails by
construction. The check should say so instead of FAIL.

Approach (the cheap form, the round's deliverable). `compare.run` already
reads `fleet_run_meta.json`; it gains `iteration = run_meta["iters"] - 1`
from the meta (or the note's number). When `iteration > 0` and the
reference directory has no `ref_boundaries_step{iteration}.safetensors`,
every boundary row gets `result = "NOT_COMPARABLE"` with
`detail = f"dumped from iteration {iteration}, the reference is step 0"`;
the rows are listed, not counted in `n_fail`; the overall verdict is the
ids and the route log. The report's header names the iteration. The it1
records are untouched (`iteration == 0`).

The right form, if F1 finishes inside its box: `run_reference.py` dumps a
second boundary file after the last decode step
(`ref_boundaries_step31.safetensors`: the head's logits and every layer's
boundaries at step 31, the same capture as step 0's run once more at the
end of the loop), and `compare.run` picks the file by the run's iteration
when it exists. The record's it32 rows then compare against step 31.

Files. `harness/compare.py` (`run`, `write_report`, `compare_boundaries`'s
result vocabulary), `harness/run_fleet.py` (the meta carries `iters`
already, `meta_out` in `main`), `harness/run_reference.py` (the right form),
`harness/README.md`, `harness/tests/test_compare.py`.

Checks. `test_compare.py`: a fixture with `iters: 32` in the meta and no
step-31 reference reads every boundary as `NOT_COMPARABLE` and the
overall as the ids' result; with `iters: 1` unchanged; with a step-31
file present (the right form) the boundaries compare against it. The
replay on `env/hw/20260918/runs/L27_head_it32_*`: `compare.py --fleet`
against the record's `harness/ref/` reads the head as not comparable
and the overall by the ids (the boundaries' safetensors are not in the
record, so the replay is on the fixture and on the VM in R1).

Found while writing the tests, from the record: a 27-layer run's dump
also carries the cache rows of layers 2 to 25 and layer 26's sixteen
boundaries (the last writer of every workspace is the last layer), and
the reference captures layers 0 and 1 only, so every model compare of
rounds 3 and 4 had 64 `MISSING_REF` rows and could never read PASS,
whatever the route log did. Those rows are `NOT_CAPTURED` now (the
reference's layers named in the detail), reported and not counted; a key
absent inside a captured layer stays `MISSING_REF` and fails. The rule is
per key: `common.boundary_iteration(key, iters)`, the cache rows and the
first token at iteration 0, everything else at `iters - 1`.

Time box: 2 hours (4 with the right form). Feeds R1, R2.

### F3. The `--final` preset (D1)

Direction. The number's configuration is thirteen flags and a define
(`01`, "What round 4 left"); the report's command and every queue row
should be one flag.

Approach. `harness/run_fleet.py`, `build_parser`: `--final`
(`action="store_true"`) documented as "the round-4 finals' stack"; after
`parse_args`, a function `apply_final(args)` sets, for each flag the user
did not name on the command line, the stack's value: `tile_linears`,
`nt_weights`, `event_timing`, `fuse_norm2`, `fuse_silu`, `fuse_norm1`,
`mfma_attend`, `attend_tasks`, `nt_streams`, `gemv_linears`,
`linear_grid = 48`, `merge_tasks`, `merge_halves = 2`, and
`-DMPK_W2_CK_TILE` appended to `runtime_flags` unless a `MPK_W2_CK_TILE`
define is already there. "Not named" is read from `sys.argv` (the
parser's defaults are `False`, so a bare `--final --no-event-timing`
needs the `--no-` forms: add `--no-event-timing`, `--no-nt-streams` and
`--no-gemv-linears`, the three a row of this round wants off, as
`store_false` on the same destinations; the rest are turned off by not
using `--final`). The run name gains `_final`
before the per-flag slugs; `graph_plan.py` is untouched (it receives the
flags as today). `queue_flag.py` learns nothing new: `--final` is a
word like any other in a row.

Files. `harness/run_fleet.py` (`build_parser`, `apply_final`, `run_name`),
`harness/tests/test_run_fleet_and_measure.py`, `harness/README.md`,
`fleet/tasks/README.md` (the flags table), `env/session/queue_flag.py`
(only if the `--no-` forms need the idempotence rule extended).

Checks. The test: `--final` alone gives the same `Plan` as the thirteen
flags spelled out, and that plan has the finals' counts, 246 operators
and 6,386 tasks (the record's `env/hw/20260918/runs/L27_head_it30_..._gv_lg48_mt_mh2/plan.json`); `--final --no-nt-streams` drops one;
`--final --linear-grid 96` overrides; the run name of the finals' row is
`L27_head_it30_final_` followed by the stack's slugs (the test pins it).
The dry run of `build_graph.py` with the flags spelled out (it has no
`--final`; the preset lives in `run_fleet.py`). The `--worker-timing`
help text is correct as it stands: an earlier draft of this page read a
defect into it from two joined line ranges (Part 4, item 6).

Time box: 2 hours. Feeds R1 to R5.

### F4. The worker-timing hang (L1, MIN-35)

Direction. The timing build (`-DMPK_ENABLE_TIMING`) hangs the round-4
header at one iteration and ran a whole session on round 3's; the
exec-per-class table depends on it. The diagnosis is offline first.

Approach.

1. The offline variant `unionT`: `env/offline_gfx942/run.sh` gains
   `compile unionT -DMK_GEMV=1 -DMK_CK_GANG=1 -DMLA_ATTEND_MFMA=1
   -DMLA_NT_STREAMS=1 -DMPK_ENABLE_TIMING=1` beside `union`, and the
   resource line of the worker (`resources.txt`) for both. What the
   timing adds is in the patch (`new_tasks.patch`, the `MPK_ENABLE_TIMING`
   hunks): a `printf` of every worker's XCD at start (round 3's I1 made it
   every worker, not eight), two `clock64` reads around each task, the
   eight-class switch, and three `printf` calls of 8, 13 and 17
   arguments at the terminate task (`[TIMING]`, `[TASK_TIME]`,
   `[TASK_TIME2]`). `run.sh` has a `timing` variant already (round 3's,
   the base header without the round-4 tasks); `unionT` is the first
   compile of the timing build over the round-4 union.
   worker's scratch and spills in `unionT` against `union` (the round-4
   union is at 256 VGPRs with 8 spills already; a `printf` with seventeen
   live values at the end can push the epilogue's spill into scratch the
   launch does not size, and round 3's union had room); (b) the `printf`
   of every worker's XCD at start, 304 workers printing before any task
   (round 3 did the same with the same worker count, so unlikely); (c) a
   timing counter the round-4 dispatcher reaches with an out-of-range
   class (the arrays are eight, the classes eight: no).
2. If (a): the seventeen-argument `printf` is split into two, or the class
   totals go to a device buffer (`config.timing_buffer`, one row per
   worker, 8 x 2 words) the host reads after the run and prints in the
   same line format (`run_fleet.py` already parses the lines from the
   log; a host-side printer writes the same lines to `fwd_pass.log`). The
   buffer form is cleaner and removes the `printf` from the epilogue
   altogether; it is the form to write if the box allows.
3. If the offline pass finds nothing, the item ends as a recorded open
   problem with the disassembly attached, and R3 does not run.

Outcome (`04`): the record, not the resource line, named the phase: the
hung row's log ends among the workers' start lines with no scheduler
line, and those lines are device `printf` calls on the hostcall path
issued right before the ready-count barrier the scheduler waits on, by
every worker under the timing define. The buffer form of step 2 was
written (every timing line printed by the host from per-worker slots, no
device `printf` on the timing build); R3 runs, its first row the check.

Files. `env/offline_gfx942/run.sh` (`unionT`), `fleet/patches/new_tasks.patch`
(regenerated on the pristine fork; `env/preflight.sh` checks the order),
`harness/run_fleet.py` (the printer, if the buffer form), `fleet/tasks/README.md`.

Checks. `unionT` exits 0; its resource line against `union`'s; the patch
applies on a clean tree (`preflight.sh`); `check_syntax.sh` (the stub
headers see no timing code, so 15 PASS as before); if the buffer form,
`test_run_fleet_and_measure.py` parses a synthetic buffer into the
`[TASK_TIME2]` lines.

Time box: 3 hours. Feeds R3 (only if fixed).

### F5. The half-merge fault (L2, MIN-36)

Direction. `--merge-tasks --merge-halves 2` without `--gemv-linears`
faults on 27 layers with the head (`hipErrorIllegalAddress` in the first
launch) and runs with the GEMV linears; no row of the configuration has
run at 2 layers (the double-check below found every 2-layer merge row of
round 4 carrying the GEMV linears). The finals do not use the
configuration; the round closes the hole or records where it is.

Approach. The first suspect of `01` (the merge's `attn` output lacking the
16-row backing the stock CK tile over-reads, MIN-33) is out: `attn` is a
`[1, NH x D_V]` plan tensor (`graph_plan.py`, `p.t("attn", ...)`) and every
single-row activation is backed by `ROW_SLACK` rows in
`build_graph.new_workspace`. The remaining candidates, each a reading:

1. The event structure between the tile-form merge and the stock o_proj:
   the merge tile registration asserts every tensor whole
   (`input_map.x == -1` for both inputs and the output, `new_tasks.patch`,
   `register_mla_merge_uv_tile_mi300_task`), so the producer's partition
   is 1 on every dimension and the runtime makes one event with
   `NH x halves` triggers (`runtime.cc`, "number of events is the product
   of gcd of producer/consumer"); the stock per-tile `linear_with_residual`
   consumer at 64 tasks partitions the weight, the input whole. That is
   the same structure the GEMV o_proj gets, so the events are not it
   unless the stock consumer's `input_map` is not whole: read
   `register_linear_task` (`task_register.cc`, the `with_residual` form)
   and `graph_plan`'s `linear_with_residual_layer` call for `attn`.
2. The worker union's dynamic LDS: the tile-form merge asks for
   `mla_merge_uv_lds_floats()` (10.25 KiB) and the CK tile for its own;
   the runtime sizes the launch by the largest registered task
   (`runtime_header.h`, `MAX_DYNAMIC_SHARED_MEMORY_SIZE` 57 KiB) and both
   fit; the 2-layer graph registers the same types. Not it, unless the
   27-layer graph registers a type the 2-layer one does not (layer 0's
   dense MLP is in both). A dry run of both graphs listing the task types
   (`build_graph.py --dry-run` prints them) settles this in a minute.
3. An address the tile form computes from the task index that the gang
   form did not: `expert_offset` gives the head and the half; at 27 layers
   the same indices repeat per layer, so no. Unless the plan's `attn`
   tensor is shared across layers (it is: one `attn` for every layer) and
   the fault is a write past its end when `halves = 2` indexes rows by
   `half`: the tile form's store offset `half * (D_V / 2)` within the
   head's `D_V` columns against a stride assumption of the gang form.
   Read the tile form's store in `mla_merge_uv_mi300.cuh` against the
   registration's output map; a `--layers 3` row with `--merge-halves 1`
   on the VM (R4) separates the halves from the layers.
4. If the readings find nothing: R4 locates the fault at 2 layers first
   (the configuration; the graph cut after the first stock o_proj and
   after the first merge with `--stop-after`; the halves control), 45 s a
   row, and bisects the layer count (3, 5, 9, 14) only if the 2-layer row
   passes; MIN-36 stays open with the readings and the located operator
   attached.

Files. Reading only, unless 1 or 3 finds the defect: then
`fleet/tasks/mi300/mla_merge_uv_mi300.cuh` or `new_tasks.patch`, the suite
row `mla_merge_uv_tile2` (bit-exact against the gang row already: a store
defect that the suite does not see is one the suite's tensor layout
hides, so the suite gains a two-layer-sized `attn` if 3 is the cause).

Checks. The plan tests, the dry runs of both graphs with the task types
printed, `check_syntax.sh`, the offline compile; the suite row.

Time box: 2 hours. Feeds R4.

### F6. The merge's standalone 5 us (L3)

Direction. `mla_merge_uv` reads 16.3 us on every round-4 build against
11.4 to 11.5 on round 3's; the graph's gap is 22.0 us in both rounds. For
the record, not the number.

Approach. The `ktime` row runs the gang launch (8 workgroups, the heads
looped inside) in both rounds, so the comparison is fair. Offline:
`dev_kt.s` of the round-4 launcher against round 3's (regenerate round
3's from `git show 8946804:fleet/tasks/mi300/mla_merge_uv_mi300.cuh` into
a scratch variant, `compile kt_r3`), the wait sequences and the LDS
traffic of the gang form side by side: N3 moved the `W_uv` batch before
the partials and added the lse words to the partials batch; N5 added the
optional `attn_s` store (a null pointer in the gang form, so a branch per
value). The likely cost is the second: a predicated store per value under a
wave-uniform null test that the round-3 form did not have, or the `W_uv` batch's 16 wave-loads
per lane now issued before the partials and drained by the partials'
`vmcnt(0)`. If the disassembly names it, the fix is one `if constexpr`
on a template flag (`HAS_ATTN_S`) so the gang and tile forms compile
without the store; the suite rows and the offline union check it.

Files. `env/offline_gfx942/run.sh` (`kt_r3`, scratch), possibly
`fleet/tasks/mi300/mla_merge_uv_mi300.cuh` and `mla_merge_oproj_mi300.cuh`
(the caller with `attn_s`).

Checks. The offline compile of every variant; `check_syntax.sh`; the
suites' merge rows (`mla_merge_uv`, `_tile`, `_tile2` bit-exact against
each other); `ktime nt` on the VM (R3's `ktime` line).

Time box: 1 hour (2 with the fix). Feeds R3.

### F7. The head's event count (N3, now a plan constant)

Direction. The head's 400 tasks run as 50 events (49 gaps of 2.3 us in
the record, 111 us per token). The runtime makes the event count of an
operator pair the gcd of the producer's and the consumer's partitions
(`runtime.cc`, `event_dims[d] = gcd(producer_partition[d],
consumer_partition[d])`); the head's consumer is `argmax_partial` at
`ARGMAX_SLICES = 50` tasks (`graph_plan.py`, line 21, "D13"), and
gcd(400, 50) = 50. The constant is the plan's, not the runtime's.

Approach. `ARGMAX_SLICES` becomes a plan argument `argmax_slices`
(default 50, the round-4 behaviour), exposed as `run_fleet.py
--argmax-slices N`; `d.V % N == 0` asserted as today. At N = 8:
gcd(400, 8) = 8 events (and gcd(320, 8) = 8 at `--head-grid 320`), each
`argmax_partial` task scanning 12,800 logits instead of 2,048 (the stock
kernel takes `num_elements / num_tasks` per task, `task_register.cc`,
`register_argmax_partial_task`); at N = 10, 10 events of 40 tasks. The
saving is up to 42 boundaries of 2.3 us, about 95 us per token (2%),
against a longer last argmax task (12,800 elements at one row: a few
microseconds).

Files. `fleet/graph_plan.py` (`ARGMAX_SLICES` to an argument, the two
`p.t` shapes and the `p.op`), `harness/run_fleet.py` (the flag),
`fleet/tests/test_graph_plan.py`, `fleet/tasks/README.md`.

Checks. `test_graph_plan.py`: the counts at 50 and at 8 (the head's
operator unchanged, the argmax's task count and the `amax_*` shapes
follow); the dry run; `test_queue_files` on the R5 rows.

Time box: 1 hour. Feeds R5.

### F8. The session tooling

Direction. One short session; every row a literal command with its PASS
text, rehearsed in DRY mode.

Approach. The queue files (`env/session/`): `queue-h1.txt` (R1: the
one-iteration model compares with `--final`, with `--runtime-flags=-DMPK_POLL_SLEEP=8`,
and one 2-layer compare), `queue-h2.txt` (R2: the finals interleaved,
A30 B30 A31 B31 A32 B32 with A the stack and B the stack with
`POLL_SLEEP=8`, then A29 and B29 with `--no-event-timing`, then the six
with `-DGEMV_BATCH=4`), `queue-h3.txt` (R3: two 2-layer rows with
`--worker-timing --final`, only if F4 fixed the hang), `queue-h4.txt`
(R4: the fault's row with F5's fix, or the layer bisect at 3, 5, 9, 14
with `--merge-halves 2` and a `--layers 3 --merge-halves 1` control),
`queue-h5.txt` (R5: `--layers 2 --head --iters 32 --final --argmax-slices 8`
and the same at 10, plus one `--layers 27 --head --iters 1 --final
--argmax-slices 8 compare`). `harness/tests/test_queue_files.py`: a
`ROUND5 = glob("queue-h[0-9].txt")` set, the existence test, the rows
building their plans as round 4's test does. `env/session/rehearse.sh`:
the round-5 rows as a second section (or a `ROUND=5` switch), writing
`06-rehearsal.md`. `05-session-plan.md`: the rows R0 to R5 with minute
marks from round 4's durations (setup 453 s, checks 81, reference 71,
the suites 337 and 308, a 2-layer row 42 s, a model row 68 s), the
budget against the balance with the $3 rule's decision recorded, the
reporting protocol, the failure playbook carried over with the round-4
additions (the watchdog, the anchored kill, the record's commit right
after the pull).

Files. `env/session/queue-h1.txt` to `h5`, `harness/tests/test_queue_files.py`,
`env/session/rehearse.sh`, `05-session-plan.md`, `06-rehearsal.md`.

Checks. `test_queue_files` (every row parses, names a run, builds its
plan; no duplicate run name across files), the DRY runs of the stages,
the rehearsal regenerated without a guard failure.

Time box: 3 hours. Feeds R0 to R5.

### F9. The checklist and the gate

Direction. Nothing reaches the VM unchecked.

Approach. `04-checklist.md` holds one box per deliverable above, ticked
with the date and the commit when its check has run; the gate is
`SHELLCHECK=1 bash env/preflight.sh` at 9 PASS (the suite, the syntax
check, the dry run, the patches on a clean tree, shellcheck), the offline
compile of every variant of `run.sh` (17 today, `unionT` the 18th), the fork at zero dirty tracked lines after the
patch check, the memory note. The results page of the round
(`07-final-numbers.md`) is drafted before the session with the round-4
numbers in place and the round-5 rows empty, so the session fills a
table instead of writing a page.

Time box: 1 hour (2 with the draft). Feeds R0.

## Part 2: what is not done on the laptop

- The finals themselves and every DECIDE row (`02`, R1 to R5): the VM.
- The direct-to-LDS weight streams, the fusions, the iteration start,
  the router's latency, MAJ-8 (`01`, "Routes not taken").
- `NO_LOCAL_CAS` as a default: settled by reading (Part 4, A1), no work.

## Part 3: the order of work

| Order | Item | Hours | Why here |
|---|---|---|---|
| 1 | F3 the preset | 2 | every queue row of the round uses it; the help-text fix rides along |
| 2 | F1 the tie rule | 2 to 3 | the compare's verdict for R1; the reference stage of R0 needs its capture |
| 3 | F2 the boundaries | 2 (4) | the same verdict; the cheap form first |
| 4 | F7 the argmax slices | 1 | the round's one speed item, one flag |
| 5 | F8 the tooling | 3 | needs F1 to F3 and F7's flags |
| 6 | F5 the fault | 2 | a reading; R4 either way |
| 7 | F4 the timing hang | 3 | a reading and maybe a patch; R3 only if fixed |
| 8 | F6 the merge's 5 us | 1 (2) | for the record |
| 9 | F9 the gate | 1 (2) | last, with the draft of the numbers page |

17 hours (22 with every optional form); items 1 to 5 and 9 (11 hours)
are the session's minimum. F4 to F6 are dropped in the order F6, F4, F5
if the hours run short, with their rows.

## Part 4: the double-check of this page against the source

Done while the page was written, on 2026-09-18; each line names the
source read and what it changed in `01` or `02`.

1. **A1, the knob's safety, settled by reading.** The scheduler consumes
   its queue by position: it loads `sched_queue_last_ready_event_id`,
   then reads every slot from `cur_event_pos` up to it
   (`persistent_kernel.cuh`, the scheduler loop around line 1470: the
   poll on `last_ready`, then `ld_local_u64(&sched_queues[0][cur ...])`);
   the slots have no flag of their own. A producer reserves a slot with
   `atom_add_local` on `next_free`, stores its event index into the slot,
   fences, and then publishes with the CAS loop that waits until
   `last_ready == its slot` (around line 1345), which is what keeps a
   later slot from being published before an earlier one is written.
   With `MPK_NO_LOCAL_CAS` the publish is a plain store of `slot + 1`: a
   worker that reserved slot n+1 and finished its store first publishes
   n+2 while slot n's event index is not yet written, and the scheduler
   reads slot n stale. Reachable whenever two of an XCD's 37 workers
   complete tasks for the same scheduler queue together, which every
   multi-task operator does. **Verdict: the knob is unsafe by
   construction; it stays a probe and is not in the default stack.** `01`
   A1 and `02` R1 and R2 are revised accordingly (B is `POLL_SLEEP=8`
   alone; no `NO_LOCAL_CAS` row).
2. **L2, the first suspect out.** `attn` is a `[1, NH x D_V]` plan tensor
   and `build_graph.new_workspace` backs every single-row activation with
   `ROW_SLACK` (16) rows, so the stock CK tile's 16-row read stays inside
   the allocation. F5 keeps the other three readings.
3. **N3 is a plan constant.** The event count is the gcd of the two
   partitions (`runtime.cc`), and the head's consumer partition is
   `ARGMAX_SLICES = 50` in `graph_plan.py`; the item is a flag, not a
   runtime edit, and it moved from "only if one line" to F7.
4. **F1's weights.** The reference's route capture hooks the gate's output
   tuple (top-k only); the 64 weights come from the gate's input and
   weight as the boundary code already does for layer 1.
5. **F2's iteration.** `boundary_dump` writes the note and `run_fleet.py`
   records `iters` in the meta; `compare.run` reads the meta already
   (`run_meta.get("head")`, `get("stop_after")`), so the cheap form is a
   few lines in one function.
6. **The `--worker-timing` help text** in `run_fleet.py` is correct; the
   "dummy device allocation" line an earlier draft attributed to it is
   `--pad-alloc`'s, read across two joined `sed` ranges. Nothing to fix;
   the checklist's box is struck.
7. **The flag count.** The stack is thirteen flags and a define (`01`,
   "What round 4 left"); an earlier draft said twelve.
8. **The finals' plan counts** are 246 operators and 6,386 tasks (the
   record's `plan.json`), not the 298 of the dry run without the fusions;
   F3's test pins the record's numbers.
9. **The capture's hooks** are `out`, `inp` and `out_tuple` (`Capture` in
   `run_reference.py`); F1 uses `inp` on the gate. The timing build's
   epilogue has three `printf` calls (8, 13 and 17 arguments), and
   `run.sh` already compiles a `timing` variant of the base header, so
   `unionT` is the union's first timing compile.
