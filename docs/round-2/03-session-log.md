# 03 - Session log

Filled during the sessions, one row per command, from the status files
(`env/logs/session.status`, `env/logs/queue.status`, pulled into
`env/hw/<date>/logs/`). The plan is `02-session-plan.md`; the row ids are
its. Times are UTC. Nothing here is written from memory: every result row
quotes a status line or a file in the record.

## Session A

| | |
|---|---|
| Date | 2026-09-16 (UTC) |
| VM | `enc1-gpuvm015`, 1x MI300X, 8 cores, 224 GiB, 13 TB disk; the shape lists a 1-minute minimum and bills per minute |
| ROCm, hipcc | 7.2.4, hipcc 7.2; rocprofv3 1.1.0; rocgdb present |
| Balance at provision, at deletion | $27.56 before; $25.12 at minute 51 (billed per minute, no 1-hour minimum); at deletion: see the end row |
| Record | `env/hw/20260916/`, branch `gpu/round-2` |
| Image | pushed: `ghcr.io/golfoscarr/fleet-amd-task:20260916`, 25 GB; `PASS image 3663s` at 18:23 (the 13 build steps 42 min, the export of layers 2,824 s of it, the push 3 min) |

### Timeline

| UTC | Row | Command | Status line or result | Note |
|---|---|---|---|---|
| 2026-09-16 (gate, before the session) | A0 | `laptop.sh balance`; `gh api /user/packages?package_type=container` | `No virtual machines`, `Available Balance: $27.56`, `Hourly Rate: $0.00/hour`; GHCR: no container package, no `fleet-amd-task` image | the plan's assumptions hold; the image is built in session A |
| 16:46 to 17:19 | A0 | `env/session/grab.sh` (109 polls of the list, one every 18 s) | the first `L provision` at 16:4x failed: `available VM matching requested specs not found` (the single unit was taken between the read and the request); the poller provisioned at poll 109 | the user authorized the poller: "check every 5 seconds; if there's a slot, reserve immediately" |
| 17:20:27 | A0 | `laptop.sh provision` (by the poller) | address saved; `L push` done at minute 1; team page `Hourly Rate: $2.99/hour` | |
| 17:21:56 to 17:22:09 | A1 | `start download`; `login`; `start image`; `start setup`; `start hw` | `PASS preflight 0s` (13 checks); `PASS download 53s` (the model was in the host cache); `PASS hw 65s` | |
| 17:29:26 | A1 | `setup` | `PASS setup 440s` | 8 cores and the image build overlapping: still faster than the plan's 25 minutes |
| 17:31:05 | A2 | `start checks` | `PASS checks 71s`; 7 PASS lines, the same readings as 2026-09-15 | |
| 17:32:26 | A3 | `start reference` | `FAIL reference 39s`: the 38 tensors and 32 ids were written, then `calibrate.py` refused to overwrite the tracked `calibration.json` the push had carried | fix 1 below |
| 17:34:34 | A3 | `start reference` again | `PASS reference 62s`; floors router 3.59e-3, layer 4.33e-3; route overlap 0.21 | |
| 17:35:24 | A4 | `start kernels` | `PASS kernels 28s`: 7 suites, 100 of 100 each | DECIDE A4 below |
| 17:37:07 | A5 | `start queue queue-a.txt` | five rows `FAIL rc=134 fault=0` in 4 to 57 s: `task_register.cc:4686 Assertion ... dim[2] == d_c + 1` | not the fault: fix 2 below |
| 17:40:55 | A5 | `start setup` (the fork reset, the patch fixed) | `PASS setup 70s` (incremental rebuild); `PASS kernels 23s` | |
| 17:45:15 | A5 | `start queue queue-a.txt` again | `L8_head_it2 FAIL rc=1 fault=1 fwd=0`; `_pad1`, `_pad2`, `_pad4` the same; `L16_head_it2_pad1 PASS fwd=2 wall=42s` (7.3 ms per iteration) | DECIDE A5 below |
| 17:48:27 | A6 | `start bisect queue-fault.txt -- --layers 8 --head --iters 2` | `BISECT first-fault=L7.norm1 runs=4`: every layer-7 label faults, the list is too narrow | the fault is in layers 0 to 6 |
| 17:49 | A6 | `L push` (the wide label list) | overwrote the VM's patched fork with the laptop's pristine copy | fix 3 below |
| 17:51:21 | A6 | `start bisect queue-fault-all.txt` (98 labels) | five rows `FAIL rc=1 fault=0` in 12 s (JIT: `TASK_MLA_PREP_MI300` undeclared), two PASS rows on stock operators; verdict void | fix 4 below |
| 17:54:01 | A6 | `start setup` | `PASS setup 65s`: the three patches applied again | |
| 17:58:07 | A6 | `start bisect queue-fault-all.txt` again | `BISECT first-fault=L0.gate_up runs=7`: `L0.norm2 PASS fwd=2`, `L0.gate_up FAIL fault=1`, `L0.o_proj PASS`, `L1.mla_prep`, `L2.mla_prep`, `L4.mla_prep` FAIL | the fused gate-up gang linear of the dense layer 0 |
| 18:01:32 | A7 | `start queue queue-fix.txt` | `L8_head_it2_al65536 PASS fwd=2`; `_wsfirst PASS`; `_al65536_wsfirst PASS`; `_al2097152_wsfirst PASS` | DECIDE A7 below |
| 18:08:26 | A8 | `start queue queue-a2.txt` (flag `--align-alloc 65536` on every row) | A8.1 `L8_head_it2_al65536 PASS fwd=2`; A8.2 `L27_head_it32_al65536 PASS fwd=31 compare=FAIL` (`output_ids PASS`: 32 ids equal; `route_log FAIL`; head rows compare the last iteration with the step-1 reference); A8.3 `L27_it1_al65536 PASS compare=FAIL` (`growth_curve FAIL`, max 0.0297 at layer 5); A8.4 two rows `measure=FAIL` (rocprofv3 aborts, fix 5); A8.5 `L2_it32_al65536 table=PASS`, `L2_it32_tile_al65536 table=PASS`; A8.6 `L27_head_it32_al65536 table=PASS`, 15,019.5 us per iteration | M4 reached at 18:04:47; DECIDE A8.4, A8.5 below |
| 18:12 to 18:17 | A8.4 | `start queue queue-b0.txt` (the stop-after ladder of layer 0 at 32 iterations) | nine rows PASS, `fwd` 28 to 32; per-iteration host clock 670 us with `norm1` alone, 1,075 us through `down` | B0 from the host clock: at 4 iterations the 15 ms launch cost hid the difference (-29 us); at 32 the ladder is noisy to about 100 us; the empty iteration costs about 670 us |
| 18:12 | A9 (early) | `report --balance` | minute 51; `Available Balance: $25.12`, `Hourly Rate: $2.99/hour`; the image still exporting layers | |
| 18:19 | A9 | the user's decision: session B runs on this VM | `queue-b2.txt`: B2, B4, the E2 lever on 2 and 27 layers; B1 is A8.5; B3 is out (rocprofv3, fix 5) | |
| 18:22:06 | B2 | `start queue queue-b2.txt` | `L27_head_it32_tile_al65536 PASS fwd=31 table=PASS`: 14,384.2 us per iteration (host clock) | 0.96 of A8.6 |
| 18:22:57 | B4 | (same queue) | `L27_head_it32_tile_al65536 PASS compare=FAIL`: `output_ids PASS`, `route_log FAIL` as in A8.2 | every flag on, the 32 ids equal |
| 18:23:32, 18:24:21 | B5 (E2) | (same queue) `--nt-weights` | 2 layers 1,497.3 us per iteration; 27 layers with the head 12,977.8 us; `mla_attend` 145 to 150 us instead of 215, `w13` 20 to 22 instead of 24 to 26 | the runs were named like their baselines and renamed `_nt` on the VM before the pull; `run_name` carries the suffix now |
| 18:23:06 | A9 | image | `PASS image 3663s`; `pushed ghcr.io/golfoscarr/fleet-amd-task:20260916` | session B of a later round can start from it |
| 18:3x | B5 | `start queue queue-b3.txt`: E2 and per-tile linears together, no flag (the plan fix in place) | `L27_head_it32_tile_nt PASS table=PASS` 12,401.6 us per iteration; `compare`: `output_ids PASS` | 0.83 of A8.6 |
| 18:4x | B5 | the user chose "mla_attend as per-tile tasks"; reading the runtime showed the premise false: `TASK_MLA_ATTEND_MI300` is gang-typed and the gang loop hands each tile to a different worker of the XCD (`persistent_kernel.cuh`, the `n_tile_count` loop strided by `block_workers_on_xcd`), so the 33 splits already run as 40 workgroups | no code change; the cheap test of the same intent instead: `--split 17` (61 splits, 8 tiles per XCD) | |
| 19:19 | B5 | `KT_TIME=50 fleet/tasks/build/kernel_tests mla_attend /tmp/kt/mla_attend/000` (the suite binary with an event-timed loop, 50 launches of the graph's grid of 8 x 5 tiles, step 1032, 33 splits live) | `TIME mla_attend launches=50 grid=8x5 mean_us=38.44`; `TIME mla_merge_uv launches=50 mean_us=11.53` | standalone the attention grid costs 38 us and the merge 11.5 us; in the megakernel 145 to 215 and 46 to 61: the difference is the runtime's gang path, not the kernels |
| 19:14 to 19:15 | B5 | `start queue queue-b4.txt` (`--split 17`) | `L2_it32_nt_s17 table=PASS` 1,511.2 us, `mla_attend` 145.2 us; `L27_head_it32_tile_nt_s17 table=PASS` 12,634.9 us, `mla_attend` 149.1 us; `compare`: `output_ids PASS` | the per-tile time does not move with the rows per tile (26 to 17): a fixed cost per tile, not the row loop; the prefetch depth did not move it either |
| 18:26 to 18:27 | fix | `start queue queue-fix2.txt` (the plan-side fix, no flag: `build_graph.new_workspace` backs every single-row activation with 16 rows, commit ce3a317) | `L8_head_it2 PASS fwd=2` (faulted at 17:42 without the fix); `L27_head_it32 PASS fwd=31`, `output_ids PASS`; `L2_it32 table=PASS` 1,676.6 us per iteration | the M4 fault is fixed at its cause; no flag needed from here on |

### Decisions (the DECIDE rows)

| Row | Rule | Measured | Chosen | Told the user at |
|---|---|---|---|---|
| A4 | the P6 kernels pass their suites | 7 suites, 100 of 100, `mla_attend`, `mla_merge_uv`, `mla_attend_splits` included | session A on the P6 kernels, no prefetch fallback | 17:36 |
| A5 | the fault tree | A5.1 faults; pads 1, 2, 4 GiB all fault; the 16-layer pad run passes | "not the address; the layer count itself" branch: A6 for the label, A7 anyway | 17:46 |
| A7 | first fix row that passes | all four pass; the first is `--align-alloc 65536` alone | `--align-alloc 65536` on every remaining row (`queue_flag.py`); a uniform shift keeps the low address bits, alignment changes them | 18:02 |
| A8.4 (B0) | near 4 us or near 40 us | the 32-iteration ladder of layer 0 (`runs/L2_it32_L0.<op>_al65536`, host clock per iteration): norm1 670.5, qkva +7.3, mla_prep +102, mla_attend -74, mla_merge_uv +265, o_proj -85, norm2 +133, gate_up -9, down +65 | `qkva` reads +7 us, near 4; but the ladder is noisy to about 100 us (a graph truncated after a different operator ends its iterations differently, `fwd` 28 to 32 of 32), so the event gaps stay the per-operator source with that caveat; the iteration itself costs about 670 us with one operator in it | 18:18 |
| A8.5 | `o_proj` under 10 us; attention under 60 and 20 us | `o_proj` (event 7) 28.6 us gang, 24.6 us per-tile; `mla_attend` 215 us; `mla_merge_uv` 61 us gang, 46 us with per-tile linears | per-tile is not the lever (above 10 us); P6 did not hold (215 and 61 us): session B's B1 question; the residual variant sits on a 25 to 35 us floor whatever its size (`down`, 46 MB, 34 us) | 18:10 |
| A9 | the push cutoff | | | |
| A11 | balance above $13 | $25.12 at minute 51 | continue | 18:12 |

### Failures and fixes

| Failure | Cause | Fix, and where it lives now |
|---|---|---|
| 1. `reference` FAIL: `calibration.json exists; the floor is recorded once` | the tracked 2026-09-15 file rides along with the rsync; `calibrate.py` refuses to overwrite by design | the stage removes the file before calibrating, every machine records its own floor (`vm.sh`, commit 9569302) |
| 2. every graph run `rc=134`: `register_mla_attend_mi300_task ... dim[2] == d_c + 1` | P2 padded the partials row to 516 in the plan, the kernels and the reference; the two registration asserts of `new_tasks.patch` still demanded 513; the offline compile covers the kernels, not the runtime | both asserts use the kernels' expression `((d_c + 1 + 3) / 4) * 4`; verified by applying the three patches on the pristine fork (`fleet/patches/new_tasks.patch`, commit 9569302) |
| 3. `L push` overwrote the VM's patched fork | the laptop's fork is the pristine pinned commit (restored after the patch check) and its files were newer; rsync sent them | `push` excludes `repos/` unless `FULL=1` (the first push of a session); `setup` re-applied the patches in 65 s (`laptop.sh`, commit d393dc8) |
| 4. the bisection narrowed on JIT failures | `queue_bisect` took any FAIL for the fault | a FAIL without a fault line ends the bisection with `BISECT error=<label>`; test with the fake harness (`queue.sh`, commits d393dc8 and 8e6ba55) |
| 5. `measure` rows: rocprofv3 aborts, `api registration failed with error code 16: Configuration request occurred outside of valid rocprofiler configuration period` | the torch wheel ships its own `libamdhip64.so` (soname without a version), loaded after the system HIP that rocprofv3 hooked; the second runtime registers again; `LD_PRELOAD` of the system library does not satisfy the unversioned soname | open: B0 answered from the host clock instead (`queue-b0.txt`); for B3 either torch's bundled HIP is replaced by a link to the system one or the counters are not measured |
| 6. a faulting run had no `fleet_run_meta.json` | the record was written after the forward loop | written with the addresses before the loop, completed after (`run_fleet.py`, commit d393dc8) |
| 7. `L pull` reverted a test edit | the pull rsynced the whole `env/hw/` tree from the VM, `tests/` included | the pull brings back the record only (`laptop.sh`, commit 8e6ba55) |
| 8. `L provision` failed with `available VM matching requested specs not found` | one unit, taken between the list read and the request | `env/session/grab.sh` polls the list and provisions at once (authorized by the user) |

## Session B

| | |
|---|---|
| Date | |
| VM, ROCm | |
| Balance at provision, at deletion | |
| Started from | the image / `setup.sh` |

### Timeline

| UTC | Row | Command | Status line or result | Note |
|---|---|---|---|---|
| | | | | |

### Decisions

| Row | Rule | Measured | Chosen | Told the user at |
|---|---|---|---|---|
| B1 | the thresholds of `02` | A8.5 stands for B1 | | |
| B5 | which lever | the user asked for session B in this VM; E2 (`--nt-weights`) ran as the cheap lever: 27 layers 12,977.8 us against 15,019.5 (0.86); `mla_attend` 215 to 150 us | E2 stays on; the per-tile MoE linears and the elementwise ops are the next code lever | 18:25 |

### Failures and fixes

| Failure | Cause | Fix, and where it lives now |
|---|---|---|
| | | |

## Artifacts

| Path | What it proves |
|---|---|
| `env/hw/<date>/runs/<name>/` | one directory per queue row (`plan.json`, `wall.json`, `fleet_run_meta.json`, `fwd_pass.log`, reports) |
| `env/hw/<date>/logs/` | the stage logs and the two status files |
| `env/logs/bisect.result` (in `logs/`) | the first faulting label |
| `fleet/tasks/results/kernel_tests.json` | the kernel suites |
| `harness/ref/*.json` | the reference ids, route log, calibration |
