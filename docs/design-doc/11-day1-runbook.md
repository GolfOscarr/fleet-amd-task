# 11 - Day-1 run-book

What to type in the first GPU session, in order, with the line that means
PASS, the time box, and what to do on FAIL. It is `08-milestones.md` Day 1
after the offline checks of 2026-09-14 (`env/offline_gfx942/README.md`),
which moved three of the day's questions off the machine: the patched
megakernel compiles and links for gfx942 with the ROCm 7.0 hipcc; CK's
split-KV FMHA cannot take a 576-wide QK head, so `mla_attend` is the spec
kernel already in `fleet/tasks/mi300/` (DQ3 closed); and the agent-scope
fence lowers to the expected cache-control instructions in the gfx942
code object (MAJ-3, see the README). What stays for the machine is
whether the build system and the runtime behave, and every measurement.

Before the session, on the laptop: `bash env/preflight.sh` (all PASS) and
`OFFLINE_COMPILE=1 bash env/preflight.sh` if the patches or kernels changed.

## Session 1 (gate 1, about 4 hours)

| # | Command | PASS looks like | Box | On FAIL |
|---|---|---|---|---|
| 0 | `git clone --recursive <repo> && cd <repo>` | `repos/fleet-chiplet-megakernel` at `51dce4f` | 5 min | network; retry |
| 1 | `bash env/setup.sh` (`SKIP_DOWNLOAD=1` if the weights are already there) | `GATE 1: PASS - Fleet built for gfx942 and imports`; two venvs; `composable_kernel: d8ee107a...`; download running in the background | 45-90 min (the cmake and cargo builds) | read the first `error:` in `env/logs/build.*.log`. Known shapes: a cmake `find_package(Z3)` miss means the z3-solver wheel did not install into `.venv-fleet` (`pip install z3-solver==4.15`); a `cargo` miss means rustup needs `~/.cargo/bin` on PATH; a `hipcc` miss means `ROCM_PATH`. A device-code error is unexpected (the headers compiled offline); paste it into the log and compare the machine's hipcc version with 7.0.51831 |
| 2 | `bash env/check_day1.sh` | seven `PASS` lines in `env/check_day1.log`; check 3 shows 8 `[SCHED_XCD] sched_id=k xcd=k` lines and `worker mod 8 == xcd`; check 7 reports 296 workers, 8 schedulers | 30 min | check 1 FAIL: `amd-smi set --compute-partition SPX --memory-partition NPS1` (needs root; ask the admin). Check 3 FAIL with a shape or download error: try `Qwen/Qwen3-8B`; with a hang: `MPK_ENABLE_VERBOSE`, then the fallback runtime decision. Check 7 wrong CU count: `docs/fleet/99-open-questions.md` Q2 |
| 3 | `source .venv/bin/activate && python harness/run_reference.py --device cuda` | `ref_output_ids.json` and `ref_output_ids_generate.json` identical (32 ids); `ref_cache.safetensors` written; under 15 min | 20 min (after the download) | an import error in the modeling file means the venv has the wrong transformers (`4.46.3` expected); OOM means another process holds the GPU (`amd-smi process`) |
| 4 | `python harness/calibrate.py --device cuda` then `python harness/route_analysis.py` | `calibration.json` with a floor per class; `exact checks hold in every run`; route summary printed | 20 min | a differing exact check between orderings is itself the finding: record it, it lowers B9 to a near-tie report |
| 5 | `rocprofv3 --list-avail \| grep -E "TCC_EA0_RDREQ\|TCC_HIT\|TCC_MISS"` | the three counter names present | 5 min | use the names it prints; edit `harness/measure.py` `parse_pmc` keys |
| 6 | decision | gate 1 recorded in `PROGRESS.md`: build PASS, graph ran, reference and calibration done | 5 min | if the build or the graph run failed and an hour of fixing did not clear it: the fallback runtime of `08-milestones.md`, M2 only |

Commit `env/check_day1.log`, `harness/ref/ref_run_meta.json`,
`calibration.json`, `harness/results/route_analysis.json` and the setup log
summary before leaving the machine.

## Session 2 (M0 closed, M1: about 6 hours)

| # | Command | PASS looks like | Box | On FAIL |
|---|---|---|---|---|
| 1 | `source .venv-fleet/bin/activate && python fleet/pack_weights.py --model-dir $SNAP --dry-run`, then without `--dry-run --device cuda` | packed sizes as `04-memory-plan.md`; the shared-expert split and the padded layer-0 MLP pass their checks | 30 min | a shape error names the tensor; compare with `docs/deepseek-v2-lite/05-weights.md` |
| 2 | `hipcc ... fleet/tasks/kernel_tests_mi300.cu -o kernel_tests` (line in `fleet/tasks/README.md`), then `python fleet/tasks/kernel_tests.py --kernel copy` | `PASS copy` | 20 min | a launch failure is the wrapper, not the kernel: `hipGetLastError` text in the output |
| 3 | `python fleet/tasks/kernel_tests.py` (all kernels, 100 trials, 1 versus 33 splits) | every kernel PASS at its tolerance; results in `fleet/tasks/results/kernel_tests.json` | 60 min | a FAIL isolates one kernel against `numpy_ref.py`; the debug build (`-DMLA_ATTEND_DEBUG_SCORES`) prints the scores for `mla_attend` |
| 4 | `python harness/run_fleet.py --layers 1 --model-dir $SNAP --stop-after L0.qkva` then `python harness/compare.py --fleet harness/fleet_out/L1_it1_L0.qkva` | `L0.B2.q PASS`; `[FWD_PASS] iter=0` printed once | 60 min | zeros: `qo_indptr_buffer[1]` not seen (DQ9); garbage: a paged read, `--debug` |
| 5 | the same through `L1.moe_router` and `L1.mla_prep` | B8 within threshold, B9 exact, B3 and B4 at layer 0 | 90 min | per boundary, `07-correctness.md` "Localization" |
| 6 | `hipcc -Rpass-analysis=kernel-resource-usage` on the generated `kernel_0.cu` | `worker_kernel` VGPRs and spills recorded next to the offline numbers | 15 min | spills: the two-pass accumulation of `04-memory-plan.md` |

## Session 3 (M2, gate 2) and after

`08-milestones.md` Day 3 to Day 5 unchanged: `mla_attend` and `mla_merge_uv`
in the truncated graph, then layer 1 end to end (`harness/README.md`,
"Milestone runs"), the first layer-1 timing, then M3, M4 and the
measurement matrix.

## Minimum viable schedule

If only two sessions can be had: session 1 above (gate 1, reference,
calibration) and a merged session 2-3 (kernel tests, M1, M2). The required
milestone is M2; everything past it is depth.
