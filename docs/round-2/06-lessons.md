# 06 - Lessons from round 2, and what the measurements say

The counterpart of `docs/gpu-bringup/03-lessons-and-ideas.md` for the sessions
of 2026-09-16 (`03-session-log.md` has the rows, `04-results.md` the
numbers). The first part is what went wrong or was believed wrongly, with the
fix and the rule that would have avoided it; the second is what the measured
numbers say the next round should do, in the order of the expected gain. Every
number names the run it comes from.

## What went wrong, and the fix

Three groups: what the preparation could not catch on the laptop, what went
wrong on the clock, and what was believed and measured false.

### Missed by the preparation

| What | What happened | Fix, and where it lives now | The rule |
|---|---|---|---|
| The runtime's asserts still wanted a 513-wide partials row | P2 padded the row to 516 in the plan, the kernels and the reference; two `assert`s in the registration glue of `new_tasks.patch` demanded `d_c + 1`; every graph run died with `rc=134` in 4 s | both asserts use the kernels' expression; verified by applying the three patches on the pristine fork (commit 9569302) | the offline compile covers the kernels, not the C++ of the patch: a change to a tensor shape is checked against `task_register.cc` by grep before the session |
| The Fleet fork travelled with every push | the laptop's fork is the pristine pinned commit; a push after it had been restored overwrote the VM's patched sources, and five bisection runs failed in the JIT (`TASK_MLA_PREP_MI300` undeclared) | `laptop.sh push` excludes `repos/` unless `FULL=1`; `setup` re-applied the patches in 65 s (commit d393dc8) | the VM owns its fork after the first push; the laptop sends patches, never sources |
| rocprofv3 cannot attach to the torch wheel | the wheel bundles its own HSA runtime and rocprofiler libraries; the second registration is fatal (`error code 16`); swapping only `libamdhip64.so` did not help | none on the wheel; B0 was answered from the host clock, B3 was not measured (`04`, the traffic table) | the counters come from a standalone binary (`copy_bytes` of round 1, the kernel suite), never from the Python graph |
| A faulting run left no address record | `fleet_run_meta.json` was written after the forward loop, so the runs the fault tooling was made for had none | the record with the addresses is written before the loop and completed after (`run_fleet.py`, commit d393dc8) | anything a failing run must leave behind is written before the thing that can fail |
| The bisection's label list was one layer wide | the plan's list held layer 7 and the head; the graph faulted at its first label, `L7.norm1`, so the list said nothing | `queue-fault-all.txt`, every label of the 8-layer graph, from the pulled `plan.json`; 7 runs named `L0.gate_up` | a bisection list starts at a label known to pass; the plan's did not |

### On the clock

| What | What happened | Fix, and where it lives now | The rule |
|---|---|---|---|
| The reference stage failed on the calibration file | the tracked `calibration.json` of 2026-09-15 rode along with the rsync and `calibrate.py` refuses to overwrite by design; the 38 tensors and the 32 ids had been written | the stage removes the file before calibrating, every machine records its own floor (`vm.sh`, commit 9569302) | a stage that "records once" deletes what the push carried |
| The bisection narrowed on JIT failures | `queue_bisect` took any FAIL for the fault and produced `first-fault=L0.mla_prep` from five 12-second failures | a FAIL without a fault line ends the bisection with `BISECT error=<label>`; a test with the fake harness (commits d393dc8, 8e6ba55) | a bisection distinguishes "faulted" from "did not run" |
| A pull reverted a test edit | `laptop.sh pull` rsynced the whole `env/hw/` tree from the VM, `tests/` included, and the VM's older copy won | the pull brings back the record only (`laptop.sh`, commit 8e6ba55) | the pull's source list is the record's directories, named, not a tree |
| The E2 runs overwrote their baselines | `run_name` had no suffix for `--nt-weights`, so `L2_it32_al65536` and `L27_head_it32_al65536` were rewritten on the VM (the laptop had them committed) | renamed `_nt` on the VM before the pull; `run_name` carries `_nt`, `_s<split>`, `_at` (commits a274325, fbb92a5) | every flag that changes a run changes its name, the day it is added |
| `delete --yes` reported a rate that was $0.00 | the check's pattern had one space after `Hourly Rate:`; the page pads it with several | `grep -qE 'Hourly Rate: +\$0\.00'` (`laptop.sh`, commit d3827df); the address files were removed by hand | a check against screen text is tested against the screen text once |
| The first provisioning failed | one 1x unit on the list, taken between the read and the request (`available VM matching requested specs not found`) | `env/session/grab.sh` polls the list every 18 s and provisions at once; 109 polls | the list is read and the request sent in one TUI session, not two |
| The image export took 47 minutes | the 13 build steps were done in 42 minutes; `exporting layers` of a 25 GB image took 2,824 s more, then the push 3 minutes | none needed: it ran in the background; the A9 cutoff was never in question | the image is built in the background from minute 2 or not at all |

### Believed, then measured false

| Belief | What killed it | What is true instead |
|---|---|---|
| The fault is address-dependent in the sense of position (a 32-bit offset overflow, a 4 GiB crossing) | pads of 1, 2 and 4 GiB moved every buffer and changed nothing (`runs/L8_head_it2_pad*`); 64 KiB alignment fixed it at once (`runs/L8_head_it2_al65536`) | the stock `gang_linear_silu_kernel` reads 16 rows of its `[1, 2048]` input at batch 1 (a CK tile GEMM with `MPerBlock = 16` and no active-token mask); whether the 60 KB past the buffer are mapped depends on the layout; the fix backs every single-row activation with 16 rows (`build_graph.new_workspace`, commit ce3a317) |
| The 211 us of `mla_attend` were 144 serialised loads per thread (P6) | the P6 kernels with four loads in flight measure 215 us in the graph (`runs/L2_it32_al65536`, event 5) against 211 before | the loads are not the time; the kernel costs 34 us standalone, warm or cold (`KT_TIME`, `KT_COLD`) |
| The attention runs as 8 workgroups (one per XCD) and per-tile tasks would parallelise it | the gang loop of the runtime already hands each tile to a different worker of the XCD (`persistent_kernel.cuh`, the `n_tile_count` loop strided by `block_workers_on_xcd`); built anyway as a regular task per split (`--attend-tasks`): 146 to 149 us, unchanged (`runs/L27_head_it32_tile_at_nt`) | the dispatch path is not the difference between 34 and 146 us |
| Smaller tiles would cut the per-tile time | 61 splits of 17 rows instead of 33 of 32: 149 us, unchanged (`runs/L27_head_it32_tile_nt_s17`) | the cost is per tile, not per row |
| The standalone 38 us is a warm-cache number and the graph pays cold reads | the suite loop rotating over 4 to 256 copies of the cache (300 MB, past the infinity cache): 33.7 to 34.0 us | the kernel is a 34 us kernel cold or warm; the 146 us is what the megakernel does around a task, paid 33 times per attention operator |
| The host mean is the per-token time | the runtime's own event clock gives 12,268 us where the host mean says 15,020, and 9,575 where it says 10,308 (`04`, the three-clock table); the "670 us per iteration of a one-operator graph" (the B0 ladder) was the launch cost over 32 iterations | the steady state is read from the event clock or the megakernel's `FWD_PASS` lines; the host mean carries the launch, the first iteration and the timing readback |

## What the measurements say for the next round

1. **The megakernel's per-task overhead is the floor, and it is measurable
   on its own.** The attention kernel costs 34 us standalone, warm or cold,
   and 146 us in the graph with E2 (215 without); the merge 11.5 us against
   46 to 61; the residual linears 25 to 35 us whatever their size (8 MB and
   46 MB); the norms 13 to 50 us for microseconds of work. Neither the
   dispatch path, the prefetch depth nor the tile size moved the attention.
   What a task pays in the megakernel that a launch does not: the
   dependency wait, the counter atomics, the completion fences (`buffer_wbl2
   sc1` at 14 sites, `buffer_inv sc1` at 4 in the worker), and the worker's
   register budget (the union of every task, 234 VGPRs, against the
   standalone kernel's 124). Measure it directly before touching any kernel:
   a graph of N empty tasks on the event clock, then the fences removed one
   at a time. The 27 attention operators alone are 27 x (146 - 34) = 3 ms of
   the 9.6 ms iteration; the same overhead on the other 300 operators is
   most of the rest.

2. **E2 is worth 2 ms per token and its mechanism points further.**
   `--nt-weights` is the compile flag `-DMPK_NT_WEIGHT_LOADS`: the stock
   linear, rmsnorm and silu kernels load their weights non-temporally, the
   weight streams stop evicting the L2, and the attention finds its cache
   rows and its q in L2 more often (215 to 150 us; 12,268 to 10,230 us per
   iteration, `runs/L27_head_it32_nt_al65536`). Our kernels read the cache
   rows once per iteration and would gain from the same treatment of their
   own streams; the MoE `w13` already moved from 26 to 22 us.

3. **Per-tile linears are worth 0.7 ms and are correct.** 12,268 to 11,514
   us on the event clock with `--tile-linears`, 32 ids equal (`runs/L27_head_it32_tile_al65536`,
   B4). With E2 together 9,575 us. Extending the flag to the MoE linears and
   the elementwise ops (`moe_silu_mul` 42 us, `moe_mul_sum_add` 21 us) is the
   next code lever after item 1, worth about 1.5 ms if they reach the plain
   linears' 4 us.

4. **The residual variant of the gang linear is a 25 to 35 us floor.**
   `o_proj` (8 MB) 24 to 29 us and `down` (46 MB) 34 us in every variant,
   while the plain gang linear reads 15 MB in 4 us and the silu one 92 MB in
   under 5 us (which no memory system delivers: the event gaps do not
   measure those two, and B0 could not settle it). The residual path's
   workspace and its second pass are the suspect; item 1's measurement
   separates the runtime's share from the kernel's.

5. **The launch costs about 20 ms and the first iteration is not the
   steady state.** The 2-layer graph's host mean is 1,736 us against an
   event-clock median of 1,048; the 27-layer graph's 15,020 against 12,268.
   A per-token figure is a steady-state median with the instrument named;
   the harness should print the event-clock median and P95 next to the
   host mean (`report_table.md` has the cells, `measure.py` leaves them
   empty).

6. **The route log needs a tie rule.** 60 of 832 (step, layer) top-k sets
   differ from the reference over 32 steps, one expert swapped in most,
   growing after step 22, with every id equal (`runs/L27_head_it32_al65536`).
   The router floor is 3.6e-3; a set that differs only by experts whose
   reference logits are within that floor of the sixth is a tie, not a
   mismatch. `compare.py` should say so instead of `FAIL`.

7. **The counters come from a standalone binary.** rocprofv3 cannot attach
   to the torch wheel; the kernel suite binary can be profiled (it ran under
   `KT_TIME`), so the bytes-per-operator arithmetic of B3 can be done on
   the suite's launches of our kernels, and `copy_bytes` of round 1 already
   proved the counters on this shape.

8. **The growth curve's one jump is a routing tie.** Layer 5's error of
   2.97e-2 (`runs/L27_it1_al65536`) is the step-0 flip at MoE index 4
   (expert 49 in the reference, 2 in Fleet), then a monotone decay to 7.1e-3
   at layer 26; the curve passes everywhere else. The tie rule of item 6
   settles this row too.
