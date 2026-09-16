# 01 - Preparation: everything that runs on the laptop before the VM

Written 2026-09-16 from the artifacts of the first sessions
(`docs/gpu-bringup/`). The rule of this round: no minute of VM time goes to
work that can be done here. Each item below has a deliverable, a check that
runs on the laptop, a time box, and what it saves on the VM. Items are
ordered by what they unblock; P1 to P4 decide whether the first session
reaches M4, P5 to P8 decide what the second session can measure.

The hardware has changed: a 1x MI300X VM ($2.99 per hour) instead of the
2x ($5.98). Consequences are in `02-session-plan.md`; for the preparation
they mean one thing: every command that ran on GPU 1 last time (the
reference run, the calibration, the kernel tests, the probes) now queues
behind the graph runs on GPU 0, so the session script (P4) serialises them.

## What the artifacts say, condensed

| Fact | Where | What it forces |
|---|---|---|
| M0 to M3 reached; 27 layers run 32 iterations at 15.6 ms per iteration without the head | `env/hw/20260915/runs/L27_it32` | the performance work has a baseline command |
| 7, 8, 9 layers fault deterministically before the first iteration; 2, 3, 4, 16, 27 run; 27 with the head faults only at 32 iterations | `docs/gpu-bringup/03-lessons-and-ideas.md` item 13 | M4 is blocked on one fault |
| The gang model runs one workgroup per XCD per operator; `mla_attend` 211 us, `moe_silu_mul` 42 us, a norm 14 to 50 us | `OPEN-PROBLEMS.md` MAJ-7 | the performance lever is task granularity, not clocks |
| The image build failed four times; the fixes are in the tree and untested | `env/docker/README.md` | the fifth build is a session item that needs no GPU |
| `run_fleet.py --debug` aborts at registration | `03` lessons table | the growth curve needs a plan change |
| Two sessions cost $12.56 on the 2x shape, 4.5 hours | `docs/gpu-bringup/04-session-log.md` | $27 on the 1x shape is about 9 hours |

## The fault, from the plans (done here, 2026-09-16)

`python fleet/build_graph.py --dry-run --layers N --out p.json` writes the
plan without a GPU. Its summary for every layer count:

| Layers | Ops | Tasks | Input tensors | Input bytes (GB) | Ran on 2026-09-15 |
|---|---|---|---|---|---|
| 2 | 26 | 180 | 30 | 2.18 | ran (with and without the head, up to 32 iterations) |
| 3 | 38 | 248 | 42 | 3.35 | ran |
| 4 | 50 | 316 | 54 | 4.52 | ran |
| 5 | 62 | 384 | 66 | 5.69 | not tried |
| 6 | 74 | 452 | 78 | 6.86 | not tried |
| 7 | 86 | 520 | 90 | 8.03 | fault |
| 8 | 98 | 588 | 102 | 9.20 | fault (every configuration) |
| 9 | 110 | 656 | 114 | 10.37 | fault |
| 10 | 122 | 724 | 126 | 11.54 | not tried |
| 12 | 146 | 860 | 150 | 13.89 | not tried |
| 16 | 194 | 1,132 | 198 | 18.57 | ran |
| 27 | 326 | 1,880 | 330 | 31.45 | ran; with the head, faults at 32 iterations only |

Every quantity the plan derives is linear in the layer count (12 ops, 68
tasks, 12 input tensors and 1.17 GB per layer; 19 workspace tensors and 33
splits always). Nothing in the plan is special at 7 to 9, and 16 exceeds 9
in every count, so a count threshold in the runtime (queue depth, task
table, event table) is ruled out along with the queues already tested. What
is left is what the plan does not control: the addresses. The weights are
`torch` allocations made in layer order, the workspaces are `torch.zeros`
after them (`fleet/build_graph.py`, `make_tensors`), and PyTorch's caching
allocator aligns every allocation to 512 bytes, so the base of every buffer
moves with the total size in a way that is not monotonic in the layer
count. The 27-layer case fits the same picture: at 32 iterations the
sequence length grows from 1,026 to 1,056, the per-layer cache slices grow
by 30 rows, and every workspace allocated after them moves.

Two consequences for the plan:

- The 513-float partials row (2,052 bytes) is misaligned at every layer
  count, because the buffer base is always 512-byte aligned; it cannot by
  itself explain a fault that 2, 3, 4, 16 and 27 layers do not show. The
  padding to 516 floats is kept as cheap hygiene (P2), not as the fix.
- The discriminating GPU test is to move the addresses without changing
  the graph: a dummy allocation of X GB before the weights are packed
  (`--pad-alloc X`, P1). If 8 layers pass at some X, or 16 layers fault at
  some X, the fault is address-dependent and the bisection is over X, at
  about one minute per run of the 8-layer graph.

## Items

### P1. Fault tooling in `run_fleet.py` (2 hours) - done 2026-09-16

Deliverables:

1. `--pad-alloc <GB>`: allocate and keep a `torch.empty` of that size on
   the device before `pack_all`, so every later address shifts. Recorded
   in `fleet_run_meta.json`.
2. `--stop-after` bisection list for layer 7 of the 8-layer graph, in
   `env/session/queue-fault.txt` (P4 format): the 8-layer graph with the
   head runs when stopped after its first operator and faults when
   stopped after `L7.combine`, so the faulting operator is among the 15 of
   layer 7 or the head; a binary search over the label list
   `norm1 qkva mla_prep mla_attend mla_merge_uv o_proj norm2 gate_up silu
   down router w13 w2 combine` and the four head labels takes four to
   five runs.
3. Device-side location: `rocgdb` is on the VM image (`rocm-gdb`
   16.3 in `env/hw/20260915/raw/dpkg-rocm.txt`). The generated kernel is
   compiled with `-O3` and no `-g` (`persistent_kernel.py`, the
   `common_cmd` list), so a stack trace names the kernel, not the line. A
   one-line patch that appends `MPK_EXTRA_HIPCC_FLAGS` from the
   environment to that list goes into `fleet/patches/gfx942.patch`, so
   the session can compile with `-gline-tables-only` and run
   `rocgdb --batch -ex run -ex bt --args python harness/run_fleet.py ...`
   to read the faulting task's source line. Time-boxed to two runs; the
   bisection alone locates the operator if the trace is unreadable.

Check here: `harness/tests/test_run_fleet_and_measure.py` covers the
argument, the run name, the address record and the label list against
the 8-layer plan (three tests); the three patches apply on a clean tree
(`env/preflight.sh`). Done: `run_name`, `tensor_addresses` and the
`pad_alloc_gb`, `pad_addr`, `addresses` fields of `fleet_run_meta.json`;
the hunk is the last of `gfx942.patch`; the labels are
`env/session/queue-fault.txt`, in plan order, with the bisection rule in
its header for P4's `queue.sh bisect`. One trap found on the way: a tree
that already carries the previous `gfx942.patch` (the image, or a VM
after a code rsync) passes neither of `setup.sh`'s checks for the new
one; step 5 now resets the submodule's tracked files and re-applies all
three patches in that case (simulated here on a worktree).

Saves on the VM: the per-operator sweep of layer 7 (15 runs) becomes 5,
and the "is it the address" question is answered in two runs instead of
being open.

### P2. Partials row padded to 516 floats (1 hour, after P1) - done 2026-09-16

`D_C + 1` is the row length in `fleet/graph_plan.py` (the `partials`
tensor), `harness/numpy_ref.py` (`mla_attend`, `merge_partials`), the two kernels
`fleet/tasks/mi300/mla_attend_mi300.cuh` and `mla_merge_uv_mi300.cuh`, the
launcher `fleet/tasks/kernel_tests_mi300.cu`, `fleet/tasks/kernel_tests.py`
and the assertion in `fleet/build_graph.py` (`mla_attend_layer`). Replace
it with one constant `P_ROW = 516` (the lse stays at column `D_C`), keep
the sizes derived from it, and leave the semantics untouched.

Check here: the 111 tests, the kernel syntax check and the offline gfx942
compile (`OFFLINE_COMPILE=1 bash env/preflight.sh`).

Saves on the VM: nothing directly; removes one hypothesis from the fault
work and makes every partials row a 16-byte multiple for the later prefetch
work in `mla_attend` (P6).

Done: `partials_row(d_c)` in `fleet/graph_plan.py` (D_C + 1 rounded up to a
multiple of 4, so 513 -> 516 and every row is 16-byte aligned when the base
is) sizes the tensor; the two kernels and the launcher define a matching
`P_ROW` and use it as the row stride, with the lse still at column D_C and
columns 513 to 515 unused. The reference in `numpy_ref.py` stays logical
(513 wide) and `merge_partials` / `mla_merge_uv` take an explicit `d_c`; the
launcher's driver pads the device buffers and slices back for the compare
(`kernel_tests.py`, `rows_partials` gained a `d_c` argument). 141 tests and
the offline gfx942 compile pass (no VGPR spills). Standing caveat: P1
already showed this is not the M4 fault (the base is 512-byte aligned, so
513-float rows are misaligned at every layer count, yet 2, 3, 4, 16 and 27
layers run) and the kernels read partials with scalar loads today, so the
alignment matters only once P6 vectorises that access; this is hygiene that
makes P6 possible, not a fix.

### P3. Growth curve: re-wire the debug snapshot (1 hour) - done 2026-09-16

The runtime accepts a graph only if every operator reads a tensor the
previous operator wrote (`repos/fleet-chiplet-megakernel/src/kernel/runtime.cc`, `register_mugraph`,
`assert(num_shared_tensors >= 1)` on the consumer's inputs against the
producer's outputs). The debug snapshot `copy_layer` writes
`dbg_x_res_{l}` and the next operator (`norm1` of layer l+1, or the head's
norm) reads `x_res`, so the chain breaks there.

Fix in `fleet/graph_plan.py`: when `debug` is set, the operator after the
snapshot reads `dbg_x_res_{l}` instead of `x_res` (the copy holds the same
values; later readers of `x_res`, the residual adds, are not consecutive
with the snapshot and are unaffected). Add the same rule to the dry-run
recorder in `fleet/build_graph.py` (`FakeMPK`), so `--dry-run --debug`
fails here the way the runtime failed there, then passes after the fix.

Check here: `python fleet/build_graph.py --dry-run --layers 27 --debug`
prints the summary; a new test in `fleet/tests/` asserts the chain rule
over every plan variant (with and without the head, debug, stop-after).

Saves on the VM: one aborted run and the growth curve over 27 layers
(M3's remaining evidence) in a single run.

Done: `OUTPUT_ARGS` and `Plan.chain_violations()` in `fleet/graph_plan.py`
(the mapping `run_fleet.py` used for the boundary dump now lives there);
`dry_run()` and `build()` in `fleet/build_graph.py` assert an empty
violation list, so a broken graph fails here before the runtime sees it.
Before the fix the dry run reported exactly the three boundaries the VM
choked on (`L0.snapshot` to `L1.norm1`, and so on to `head.norm`); after
it, `norm1` of layer l reads `dbg_x_res_{l-1}` and the head's norm reads
the last snapshot when `--debug` is set, and nothing changes without it.
Three tests in `fleet/tests/test_graph_plan.py` (eight plan variants
clean, the wiring, the broken wiring caught).

### P4. Session scripts (2 hours) - done 2026-09-16

Three files under `env/session/`:

- `vm.sh`: runs on the VM; stages, each detached with `setsid nohup`,
  each writing `env/logs/<stage>.out` and one `PASS <stage>` or
  `FAIL <stage>: <reason>` line into `env/logs/session.status`:
  `download` (the model, from a `--without-pip` venv plus `get-pip.py`),
  `image` (the Docker build, then the push, then `docker logout`; runs in
  the background for the whole session), `setup` (`env/setup.sh` with
  `SKIP_DOWNLOAD=1`), `hw` (`collect_hw.sh`, only when the host name or
  the ROCm version differs from `env/hw/20260915/summary.md`), `checks`
  (`check_day1.sh`), `reference` (`run_reference.py --device cuda`,
  `calibrate.py --device cuda`, `route_analysis.py`), `kernels`
  (`kernel_tests.py`), then `queue`.
- `queue.sh`: reads a queue file (one `run_fleet.py` argument line per
  row, `#` comments, an optional `compare` or `measure` suffix), runs the
  rows one at a time, copies each run's `harness/fleet_out/<name>` into
  `env/hw/<date>/runs/`, and appends the outcome (`mpk()` seen,
  `AcceleratorError` seen, `FWD_PASS` count, wall time) to
  `env/logs/queue.status`. Never two graph runs at once (the shared JIT
  directory). Stops on the first `FAIL` unless the row is marked
  `continue`.
- `laptop.sh`: the driver: `provision`, `ip`, `push` (the rsync of the
  tree with the exclusion list from `06-agent-guide.md`), `start <stage>`,
  `status` (tails both status files), `pull` (rsync of `env/logs/`,
  `env/hw/<date>/`, `harness/fleet_out/*/` report files back, absolute
  paths, then `git add` and a commit), `delete`. The TUI calls are the
  recipes of `06-agent-guide.md` through `env/hotaisle/tui.py`.

Rules baked in, from the lessons: `HIP_VISIBLE_DEVICES=0` everywhere;
`--iters` at most 32; kill by pid from an anchored `pgrep`; no `pkill -f`
with a pattern in the caller's own command line; `</dev/null` on every
ssh; absolute rsync destinations; the record directory
`env/hw/$(date -u +%Y%m%d)` is created, never deleted.

Check here: `bash -n` on the three files in `env/preflight.sh`; a
`DRY=1` mode that prints every command instead of running it, exercised by
a test in `env/hw/tests/` that runs the queue on a fake `run_fleet.py`
which writes the expected files.

Saves on the VM: the hand-typed ssh commands of the first session (the
two dropped sessions, the self-matching wait, the nested rsync) and the
serial waits between stages; the image build starts at minute zero
without anyone watching it.

Done: `env/session/common.sh`, `vm.sh`, `queue.sh`, `laptop.sh`; the
queue files `queue-a.txt` (A1, A2), `queue-a2.txt` (A4 to A7 and the
baseline), `queue-b.txt` (B1 to B5) and the label file `queue-fault.txt`;
eleven tests in `env/hw/tests/test_session_scripts.py` (the queue on a
fake `run_fleet.py`: status rows, record copies, the stop rule; the
bisection for three fault positions, five runs each; the DRY modes of
every stage, the profiler rows and the laptop commands; every queue row
parses as `run_fleet.py` arguments). Two details that differ from the
sketch above: a fourth row keyword `table` runs `measure.py` on the
result alone (the per-operator table of an `--event-timing` run without
the profiler), and `laptop.sh login` pipes the laptop's `gh auth token`
into `docker login` on the VM so the `image` stage can push; the stage
builds and keeps the image local when no login exists. `laptop.sh delete`
needs `--yes`, since deletions are the user's call. The laptop's bash is
3.2, so the scripts use no associative arrays. Known limit for P7:
`measure.py` sums the PMC rows of every dispatch of the profiled run,
including the weight packing; P7 filters to the persistent kernel.

### P5. Per-tile linears behind a flag (2 hours) - done 2026-09-16

MAJ-7's largest lever. The stock runtime has a non-gang `linear_layer`
(`persistent_kernel.py`) that issues one task per weight tile; it exists
next to the gang variants whatever `USE_GANG` says (that variable only
compiles the gang code paths in). Our plan uses `gang_linear_layer` and
its two variants for every linear. Add `--tile-linears` to `run_fleet.py` and a
`tile_linears` switch in `fleet/graph_plan.py` that issues the four stock
linears of a layer (`qkva`, `o_proj`, `down` and `gate_up` in layer 0, the
head's `lm_head`) as `linear_layer` calls with the tile counts the
per-tile path expects, leaving the MoE linears and our kernels unchanged.
The dry-run recorder enforces the wrapper's shape assertions.

Check here: the dry run with and without the flag; the task count changes
from 8 per linear to the tile count; the 111 tests plus a test for the
new plan variant.

On the VM: the same 2-layer, 32-iteration event-timing run with and
without the flag; the target is `o_proj` moving from 38 us toward the
2 us its 8 MB of weights need at 4 TB/s. If it does, the second session
extends it to the MoE linears and the elementwise ops.

Done: `grid_for_linear()` and the `tile_linears` argument in
`fleet/graph_plan.py` switch `qkva`, `o_proj`, `down` and `lm_head` from
the 8-task gang to the stock `linear_layer` / `linear_with_residual_layer`,
which split the output columns across 96, 64, 64 and 400 tasks over all
296 workers (the demo's tested grid heuristic, `size // 256` for the large
`lm_head`). Both are real methods on the runtime, so only the dry-run
`FakeMPK` needed the two wrappers (with the stock shape asserts). Threaded
through `run_fleet.py` as `--tile-linears` (run-name suffix `_tile`,
recorded in `fleet_run_meta.json`) and `build_graph.py`; the
`queue-a2.txt` A7 row is enabled. Six tests: the flag flips exactly those
four ops and multiplies the task count, leaves the gang plan untouched
when off, the grids divide the output sizes, and the chain rule still
holds. Scope: `gate_up` is silu-fused (gang-only; a non-gang split would
add a boundary) and the MoE linears are expert-routed, so both stay gang;
they are session B's per-tile work if A7 pays.

### P6. Prefetch and more splits in `mla_attend` and `mla_merge_uv` (3 hours, second) - done 2026-09-16 (prefetch; splits deferred)

`mla_attend` is 5.7 ms of the 15.6 ms and `mla_merge_uv` 1.45 ms. Both are
ours. Two changes prepared here and validated only on the VM: a prefetch
depth of 4 in the split loop of `mla_attend` (the knee measured in
`env/hw/20260915/summary.md`, E2), and a split count that fills the
machine (the plan's 33 splits are 5 tiles per XCD; 64 or 128 splits give
8 to 16 per XCD at the same partials layout). Both are constants in
`fleet/graph_plan.py` (`SPLIT`, `n_splits`) and loops in the two kernels.

Check here: the syntax check, the offline compile with the resource usage
read from the disassembly (no VGPR spills, `env/offline_gfx942/`), and
`numpy_ref.py` unchanged because the semantics are unchanged.

On the VM: `kernel_tests.py --kernel mla_attend --kernel mla_attend_splits
--kernel mla_merge_uv` (100 trials each), then the 2-layer timing run.

Done: reading the kernels explained the 211 us. Each thread of an
`mla_attend` tile computed two scores by walking a 576-element cache row
with one 16-byte load at a time, 144 serialised loads per thread; at the
measured 1 to 1.5 us of HBM latency that is 150 to 216 us, the number in
`runs/L27_it32`, for a tile that moves 36 KB. Two changes, both keeping
the FP32 accumulation order and the BF16 probabilities: the column loops
of the score dot, the row loop of the PV product and the merge's two
loops now issue `PF = 4` loads before their FMAs (the E2 knee); and the
score mapping is lane-to-head (`h = e % NH, p = e / NH`) instead of
lane-to-row, so the 16 lanes of one row share its loads and a row is no
longer fetched 16 times through 64-line uncoalesced instructions. Expected
order: about 4x from the batching and less L2 traffic from the mapping;
B1 of the session plan asks for `mla_attend` under 60 us and
`mla_merge_uv` under 20 us. Cost in registers, offline: `k_mla_attend` 90
to 124 VGPRs, `k_mla_merge_uv` 32 to 75, no spills; the worker kernel's
union 182 to 234 VGPRs offline (the VM's union with the CK linears was
248, a maximum not a sum, so it should not move). Syntax check, the 142
tests and the offline compile pass; the kernel tests on the VM
(`kernels` stage, 100 trials, both binaries) are the correctness gate,
and the `mla_attend_splits` suite checks the 33-split merge against one
split end to end.

Deferred, with the reason: "more splits" is not a constant change.
`mla_merge_uv` maps one lane of a wave per split, so 64 splits is a hard
cap (the plan's 33 fit; 32 rows per split at 1,056 positions), and the
attend tile is 32 rows because 16 heads times 32 rows is two scores per
thread; a 16-row split needs a new score mapping and a merge over more
than one wave. With the batching in place the tile is no longer
latency-bound, so the case for more splits rests on B1's number: if
`mla_attend` lands near its 36 KB at 4 TB/s (about 9 us plus the boundary)
the split count stays; if it stays above 60 us, the next lever is the
column-mapped score (a wave reduces one row, coalesced 1 KB loads, 16
accumulators per lane) or the MFMA path of `docs/mla-decode/04`.

### P7. Measurement commands for the submission (1 hour) - done 2026-09-16

`harness/measure.py` reads a rocprofv3 kernel-trace CSV and PMC CSVs, but
no graph run has been profiled yet. Add to the queue format a `measure`
row type that runs the 27-layer, 32-iteration graph under
`rocprofv3 --kernel-trace --output-format csv` once and under `--pmc` once
per counter pair (`TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_32B_sum`,
`TCC_EA0_WRREQ_sum TCC_EA0_WRREQ_64B_sum`, `TCC_HIT_sum TCC_MISS_sum`,
`TCC_BUBBLE_sum TCC_EA0_RDREQ_sum`: the four `PMC_SETS` of
`env/collect_hw.sh`), then calls `measure.py` with all of them. The counter formulas in `measure.py` are the ones validated to
the byte on 2026-09-15 (MIN-19).

Check here: the queue test of P4 with a fake `rocprofv3`; the existing
`measure.py` tests.

On the VM: five runs of about two minutes, giving bytes per iteration,
achieved bandwidth, launches per generation and the per-operator table.

Done: `measure.py` filters the counters and the launch count to the
megakernel's dispatches (`prepare_kernel`, `worker_kernel`,
`scheduler_kernel`, `persistent_kernel`; `--kernel-filter ''` sums the whole
process), reports the megakernel's dispatches against the run's total, and
derives a per-iteration time from the trace's own timestamps. `--pmc` takes
a file, a comma-separated list or a directory of runs, merged per file with
the later file winning (the summarizer's rule): a bug found on the way is
that P4's `measure` row concatenated the four PMC runs, and the counter
present in two of them (`TCC_EA0_RDREQ_sum`, in the 32B pair and in the
BUBBLE pair) was double counted, 1.5 GiB for a 1 GiB copy; `queue.sh` now
passes the profile directory and copies the per-run CSVs into the record.
Checked against the 2026-09-15 record: filtered to `copy_kernel`, reads and
writes are 1 GiB within 1e-4 in both the directory and the list form. Four
tests. What P7 cannot do here: the megakernel filter matches no dispatch of
a probe run, so its first real use is session B's B3 row.

### P8. Documents for the 1x shape and this round (1 hour) - done 2026-09-16

- `docs/gpu-bringup/06-agent-guide.md` (the second GPU in three places),
  `05-next-session.md` (the reference run on GPU 1; the whole file is
  superseded by `02-session-plan.md` and says so at the top),
  `03-lessons-and-ideas.md` idea 7, `01-plan.md` line 331: one sentence
  each.
- `env/docker/README.md`: the fifth attempt is a session stage of P4.
- `PROGRESS.md`, `README.md`, `OPEN-PROBLEMS.md`: pointers to this section.
- `03-session-log.md` and `04-results.md` in this section are created at
  the start of the session and filled as it runs.

Done: the four second-GPU passages corrected (`06-agent-guide.md`, which
now also names the session scripts as the way every step is run;
`05-next-session.md`, marked superseded at the top; `03` idea 7 and its
lessons row; `01-plan.md`); `env/docker/README.md` points the fifth build
at the `image` stage; the `gpu-bringup` index marks 05 superseded. One
substantive addition found while reading the profile for P5, written into
`OPEN-PROBLEMS.md` MAJ-7, `03` idea 12 and the session plan: the plain
gang linear `qkva` shows 4.4 us for 15 MB (3.4 TB/s) and `gate_up` 4.8 us
for 92 MB, which no memory system delivers, so those rows are at the floor
or the event gaps mis-attribute; the time sits in the residual gang
variant (about 37 us whether 8 MB or 46 MB), `moe_silu_mul`, the norms and
our three kernels. Session B therefore opens with B0, two stop-after runs
timed from the kernel trace whose difference is one plain gang linear
(`queue-b.txt`, first two rows), before anything is optimized.

### P9. Closed or not worth doing here (checked 2026-09-16)

- `compare.py` already SKIPs the output ids and the route log on a
  truncated graph (`harness/compare.py`, the `run_meta` checks); nothing
  to do.
- The calibration floors for `norm` (4.5%) and `logits` (4.1%) come from
  the GPU reference run; they are stated in the write-up, not re-derived
  here.
- Testing the image build locally: torch for ROCm and the Fleet build
  under Rosetta would take hours; the build runs on the VM at minute
  zero instead (P4, `image` stage).
- The wall-clock 1.8% offset touches no timing path we report (event
  timing uses the runtime's own timestamps and the wall time is the
  host's); noted in the results, not fixed.

## Order and the gate before booking the VM

| Order | Item | Hours | Deliverable checked by |
|---|---|---|---|
| 1 | P1 fault tooling | 2 | tests, patches apply |
| 2 | P4 session scripts | 2 | `bash -n`, `DRY=1` queue test |
| 3 | P3 snapshot re-wire | 1 | dry run with `--debug`, chain-rule test |
| 4 | P5 per-tile linears | 2 | dry run, plan-variant test |
| 5 | P2 partials padding | 1 | tests, offline compile |
| 6 | P7 measurement rows | 1 | queue test with a fake profiler |
| 7 | P8 documents | 1 | link check, no emoji, no names |
| 8 | P6 attention prefetch and splits | 3 | offline compile, resources |

About 13 hours of laptop work; all eight items were done on 2026-09-16
(P6 with the split count deferred, see its section).

Gate, all PASS before `laptop.sh provision`: `bash env/preflight.sh` (the
8 checks plus the three new files), `OFFLINE_COMPILE=1` once after P2 and
P6, the branch pushed, `gh api /user/packages?package_type=container` read
so the session knows whether an image exists, and the balance read from
the TUI (`python3 env/hotaisle/tui.py 12`).
