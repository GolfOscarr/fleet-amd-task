# 04 - Session log: every run, every failure, every fix

The first two VM sessions, 2026-09-15 UTC, Hot Aisle `enc1-gpuvm005`
(2x MI300X VF, ROCm 7.2.4, hipcc 7.2.53211, Ubuntu 24.04, Python 3.12,
26 cores, 440 GiB RAM). One VM, about 3.5 hours, about $11. Every log is
under `env/hw/20260915/logs/`, every run's report under
`env/hw/20260915/runs/<name>/`, the hardware record in
`env/hw/20260915/summary.md`. The next session starts from
`05-next-session.md`.

## Timeline

| UTC | Step | Result |
|---|---|---|
| 15:10 | VM provisioned from the TUI (`n`, Enter, `y`) | ready in about 2 min; `ssh hotaisle@<ip>` |
| 15:15 | repo copied by rsync; model download started | 30 GB in 79 s |
| 15:17 | `bash env/collect_hw.sh` | 62 rows in about 1 min (+ BabelStream later): `summary.md` |
| 15:27 | `bash env/setup.sh` (first attempt) | aborted: venv without pip |
| 15:28 | python3.12-venv installed; `setup.sh` again | built in 15 min; gate 1 FAIL on `import mirage` (z3) |
| 15:43 | z3 matched, `LD_LIBRARY_PATH` set; `check_day1.sh` | import PASS; graph run FAIL (CK wrapper references the gfx950 kernel) |
| 15:52 | gfx942 hunk applied; `sched_xcd.patch` applied; `check_day1.sh` | all checks PASS: 8 schedulers on their XCDs, fences present |
| 15:55 | `run_reference.py --device cuda` on GPU 1 | 32 ids, generate agrees with the loop, cache captured |
| 15:57 | `calibrate.py --device cuda` | crashed on a GPU-versus-CPU operand; fixed; floors per class |
| 16:00 | `route_analysis.py` | overlap 0.21, affinity would not pay |
| 16:02 | `pack_weights.py --dry-run`, then real | 272 tensors, 31.42 GB, checks pass |
| 16:03 | kernel tests, all suites, GPU 1 | 6 PASS, `mla_attend_splits` 10 of 100 FAIL (bound too tight); fixed; 7 of 7 PASS |
| 16:03 | M1: `--layers 1 --stop-after L0.qkva` | B1, B2 PASS |
| 16:06 | `--layers 2 --stop-after L1.moe_router` | label wrong (`L1.router`); no run |
| 16:07 | `--layers 2 --stop-after L1.mla_prep` | layer 0 complete PASS, layer 1 through B4 PASS |
| 16:10 | M2: `--layers 2` | all 16 boundaries PASS, top-k exact, route log PASS |
| 16:12 | resource usage of the generated megakernel | 248 VGPRs, 48 AGPRs, 1 wave/SIMD, 0 VGPR spills, 150 SGPR spills |
| 16:14 | B5: `--stop-after L1.mla_attend --debug-scores` | scores rel 5.4e-3 PASS |
| 16:15 | timing: `--layers 2 --iters 8 --event-timing` | per-op times 10 to 100x bandwidth time |
| 16:16 | M4: `--layers 27 --head --iters 32 --event-timing` | illegal memory access |
| 16:20 | `--layers 3` | runs; layers 0 and 1 PASS, layer 2 has no reference |
| 16:21 | `--layers 2 --head` | runs; head boundaries fail by construction (2 of 27 layers) |
| 16:22 | `--iters 64` | assertion: `cos` table is 1,056 rows, 64 iterations need 1,088 |
| 16:23 | `--layers 27` | runs: 1,822 tasks, 71 boundary tensors, 27.8 ms; route log one near-tie at MoE layer 4 |
| 16:26 | `--layers 27 --head` | token 25 = reference; head norm rel 9.8e-3, logits 0.016 |
| 16:27 | `--layers 2 --head --iters 4` | runs |
| 16:28 | `--layers 2 --iters 32 --event-timing` | 2.21 ms per iteration; same per-op times |
| 16:30 | M4 without event timing | illegal memory access |
| 16:31 | `amd-smi metric --clock` during a run | 131 MHz reading; `--perf-level high` refused in the VF |
| 16:33 | GPU 1: bandwidth probe running alongside the timing run | identical per-op times: not the clocks |
| 16:35 | `--layers 27 --head --iters 2` | tokens [25, 16228] = reference |
| 16:36 | `--layers 27 --iters 4` | runs, 17.7 ms per iteration |
| 16:37 | `--layers 8 --head --iters 32` | illegal memory access before iteration 1 |
| 16:40 | queues 1,024 to 16,384; `--layers 8 --head --iters 32` | still faults; reverted |
| 16:43 | `--layers 8 --head --iters 8` and `16` | fault |
| 16:46 | `--layers 2 --head --iters 32` | runs all 32 |
| 16:50 | Docker image build, attempt 1 | failed: the submodule reset had no git metadata (`.git` excluded from the context) |
| 17:00 | image build, attempt 2 | failed: `rocblas/rocblas.h` missing in the ROCm dev base image |
| 17:05 | image build, attempt 3 | built, `import mirage` failed on `libz3.so.5.1` (pip build isolation), then `hatchling` missing without isolation |
| 17:12 | `--layers 27 --iters 32 --event-timing` (sequential) | 15.6 ms per iteration; the whole-model profile |
| 17:15 | frontier runs with the head, sequential: 8 layers at 2 fails; 4 at 4 and 8 pass; 16 at 2 passes; 7 and 9 at 2 fail; 8 without the head at 2, 16, 32 fails; 8 with the head stopped after the first operator passes, stopped after the last MoE operator fails | the fault depends on the layer count, not on the head or the sequence length |
| 17:20 | `--layers 27 --debug` (growth curve) | aborted: the snapshot copy shares no tensor with its successor (`register_mugraph` assertion) |
| 17:25 | image build, attempt 4 (hatchling added) | running at the end of the session; `env/docker/README.md` |

## Failures and fixes

| Failure | Cause | Fix, where |
|---|---|---|
| `setup.sh` aborted at the first pip install | the image's `python3 -m venv` has no `ensurepip`, so the venv has no pip; `apt-get install python3.12-venv` 404s without `apt-get update` | `setup.sh` step 2 installs `python3.12-venv` when `ensurepip` is missing (needs sudo, which the `hotaisle` user has) |
| `import mirage` failed after a successful build: `libz3.so.5.1` not found | pip's isolated build environment installed z3-solver 5.1 (unpinned in Fleet's `pyproject` build requirements) and the extension linked against it; the venv holds the pinned 4.15; neither `z3/lib` is on the loader path | `setup.sh` step 6: `PIP_CONSTRAINT` pins the build environment's z3 to the venv's; `export LD_LIBRARY_PATH=<venv>/z3/lib` appended to `.venv-fleet/bin/activate`; the gate check runs from the Fleet directory |
| The Qwen3 smoke graph's JIT failed: `use of undeclared identifier 'paged_attention_minimal_decode'` | the CK split-KV wrapper calls the gfx950-only minimal decode kernel whose include `gfx942.patch` hides; the offline compile had parsed the file without instantiating that wrapper; an `if (false)` guard was not enough, two-phase lookup still needs the name | a `#if defined(__gfx950__)` hunk in `gfx942.patch` sends the one-token step through the CK prefill pipeline; only the smoke graph takes this path |
| `check_day1.sh` check 3 FAIL: worker mod 8 != xcd | workgroup k lands on XCD (k + c) mod 8, c = 4 in the probe, 5 for the worker kernel and 6 for the scheduler kernel of one process; the runtime's scheduler read queue k by block id | `fleet/patches/sched_xcd.patch`: a local scheduler takes its queue index from `HW_REG_XCC_ID`; check 3 accepts any constant offset per kernel |
| `check_day1.sh` check 6 UNKNOWN although every counter is listed | `grep -q` closed the pipe early and `pipefail` failed the test; also `rocprofv3 --list-avail` exits 120 with a complete listing | here-strings instead of pipes; the exit code is ignored, the presence of `TCC_` decides |
| `calibrate.py` crashed: tensors on `cuda:0` and `cpu` | never run on a GPU before; the captured boundaries stayed on the device while the reference is loaded from disk | `compare.metrics` moves both operands to the CPU; `capture()` returns CPU tensors |
| `compare.py` reported the placeholder ids and the empty route log of a truncated graph as FAIL | no rule for `--stop-after` runs | SKIP for both on a truncated graph; the report prints SKIP entries; the route rule still misfires on a graph stopped before the router (placeholders count as routes), to be fixed |
| `mla_attend_splits` 10 of 100 trials FAIL at rel_err 5e-4 | the per-element bound allowed one ulp of the element plus 1e-5 of the maximum; a 33-way merge of cancelling partials rounds near-zero elements by an ulp of the largest term | the floor is one ulp of the tensor's largest element (`BF16_ABS_FLOOR = BF16_ULP`); the test that expected a two-ulp mid-element failure now uses the largest element |
| `--stop-after L1.moe_router` did nothing | the op is labelled `L1.router` | labels are listed in `fleet/graph_plan.py`: `norm1 qkva mla_prep mla_attend mla_merge_uv o_proj norm2 gate_up silu down router w13 w2 combine snapshot` |
| `--iters 64` asserted on the `cos` table shape | the tables and `max_seq_length` are sized for 1,024 + 32 positions | 32 iterations is the ceiling without re-sizing |
| The profiler byte arithmetic was off by 2x on reads and 3x on writes | reads are 128-byte requests counted by `TCC_BUBBLE`; the sums included the fill and warm-up kernels | `summarize.py` and `measure.py` use the decomposition with the 128-byte term and isolate the copy kernel; validated exact on 1 GiB each way |
| A laptop dry run deleted the real record | same UTC date directory; the dry-run cleanup removed it | committed immediately after every copy; `collect_hw.sh --out DIR`; nothing under `env/hw/2*/` is deleted |
| `pkill -f <pattern>` dropped the SSH session twice | the pattern matched the SSH command line itself | kill by pid from `pgrep -f "^python ..."`; never a pattern that appears in the caller's own command |
| Queued chains waited forever | `while pgrep -f run_fleet.py` matched the waiting shell itself | anchored patterns; or `setsid nohup` and poll a log line |
| The 27-layer graph with the head faults at 32 iterations | open; see `03-lessons-and-ideas.md` item 14: not the queues, not the clocks, not event timing; needs a deeper graph and a larger configured iteration count with the head | next session: verbose runtime, bisect on `--layers 2..27 --head --iters 8` |
| Every operator 10 to 100x slower than its bandwidth time | the gang model runs one workgroup per XCD per operator (MAJ-7) | per-tile tasks or multi-workgroup gangs, in the graph builder and the task glue; the next performance step |

## What each artifact proves

- `env/check_day1.log`: gate 1 (build, import, XCD placement, fences, counters, CU count).
- `harness/ref/ref_run_meta.json`, `ref_output_ids.json`: the reference on the real checkpoint; `calibration.json`: the floors; `harness/results/route_analysis.json`: the routing statistics.
- `fleet/tasks/results/kernel_tests.json`: 7 suites, 100 trials each, PASS.
- `env/hw/20260915/runs/L2_it1/correctness_report.md`: M2. `L2_it1_L1.mla_attend_scores`: B5. `L27_head_it1`, `L27_head_it2` (`logs/bis_l27h2.out`): the first two tokens exact. `L2_it32/report_table.md`: the per-operator times behind MAJ-7. `raw/m1_resources.txt`: the megakernel's registers.
