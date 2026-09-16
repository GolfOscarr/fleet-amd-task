# 02 - Session plan: two sessions on a 1x MI300X for $27

Written 2026-09-16, before the sessions. Assumes the preparation of
`01-preparation.md` is done and gated. The run log goes to
`03-session-log.md` and the numbers to `04-results.md`, both written
during the sessions.

## Budget

| | |
|---|---|
| Balance | $27.00 |
| Shape | 1x MI300X, $2.99 per hour, one-hour minimum, per-minute after |
| Hours | 9.0 |
| Session A | up to 4.5 hours ($13.46): the image, M4, the growth curve, the first per-tile timing |
| Session B | up to 3.5 hours ($10.47): performance, the submission measurements |
| Reserve | 1 hour ($2.99): a rescue if a session ends with an unsynced record or an unpushed image |

Running out of balance deletes the VM with everything on it, so the record
is pulled every 30 minutes (`laptop.sh pull`) and the balance is read
before any stage longer than 30 minutes (the image push).

## What changes with one GPU

- The reference run, the calibration, the kernel tests and the probes
  ran on GPU 1 last time. They now run in the `reference` and `kernels`
  stages before the queue, about 4 minutes in total, on the same GPU.
- The "not the clocks" check of MAJ-7 (a bandwidth probe on the other GPU
  during a timing run) cannot be repeated; it is recorded and cited.
- `HIP_VISIBLE_DEVICES=0` is the only device; the scripts set it.
- The host has fewer cores than the 2x (26 last time), so `setup.sh`
  and the image build, which both compile Fleet, take longer when they
  overlap. They overlap anyway: the GPU is idle until `setup.sh` ends.
- The hardware record is re-collected if the host name or the ROCm
  version differs (the `hw` stage, one minute): the placement offset, the
  counter access in the VF and the bandwidth numbers are the ones that
  could differ.

## Session A (up to 4.5 hours)

| Minute | Stage | PASS line | On FAIL |
|---|---|---|---|
| 0 | `laptop.sh provision`, `ip`, `push` (rsync, about 1 minute) | `ssh hotaisle@<ip>` answers | retry the TUI call once; otherwise stop, nothing is billed until the VM exists |
| 2 | `vm.sh download`, `vm.sh image`, `vm.sh setup` all started; `vm.sh hw` if the host is new | `PASS download` in about 2 minutes; `PASS hw` in 1 | download: check the balance and the disk (`df -h /`, needs 100 GB free); image: keep going, it is background; setup: read `env/logs/setup.out`, apply the manual fix from `env/docker/README.md`, rerun |
| 25 | `PASS setup`; `vm.sh checks` (`check_day1.sh`, 7 PASS lines); `vm.sh reference`; `vm.sh kernels` | `PASS checks`, `PASS reference`, `PASS kernels` in about 6 minutes | a check that passed on 2026-09-15 and fails now is a machine difference: record it in `03-session-log.md`, and continue if it is not gate 1 |
| 32 | `vm.sh queue env/session/queue-a.txt` | one status row per run | the queue stops on the first FAIL that is not marked `continue` |
| 32 | A1: `--layers 8 --head --iters 2` (the fault reproduces, expected FAIL, `continue`) | `AcceleratorError` in the log | if it passes on this machine, the fault was machine-specific: record it, skip to A5 |
| 34 | A2: `--layers 8 --head --iters 2 --pad-alloc 1`, `--pad-alloc 2`, `--pad-alloc 4` (`continue`) | any PASS says address-dependent | all fault: the address is not the variable; go to A3 |
| 40 | A3: the `--stop-after` binary search over layer 7 (`queue-fault.txt`, about 5 runs) | the first faulting label | none faults but `L7.combine` did: the fault is in the head, repeat over the four head labels |
| 50 | A4: the fix, coded on the laptop from A2 and A3, pushed by `laptop.sh push`; `--layers 8 --head --iters 2` again | `mpk()` seen, `FWD_PASS` 2 | time box 60 minutes from A3; past it, record the frontier and go to A5 with the 27-layer graph at 2 iterations |
| 80 | A5: M4: `--layers 27 --head --iters 32` then `compare.py` | 32 ids equal to `harness/ref/ref_output_ids.json`, route log PASS | a mismatch at token k: the run stands as evidence, the token index goes to `04-results.md` |
| 90 | A6: the growth curve: `--layers 27 --debug` then `compare.py` | 27 per-layer errors under the layer threshold | the first layer above the threshold is named; layers 0 and 1 are already validated boundary by boundary |
| 100 | A7: `--layers 2 --iters 32 --event-timing` with and without `--tile-linears` | `o_proj` under 10 us with the flag | above: the per-tile path is not the answer, MAJ-7 moves to the gang width; session B starts with P6 instead |
| 110 | `laptop.sh pull`; image status | `PASS image` (build, push, logout) | not built: read `env/logs/image.out`, fix the Dockerfile on the laptop, restart the stage if more than 40 minutes remain, else `docker save` to the laptop is not attempted (30 GB) and the fix waits for session B |
| 120 to 240 | the rest of `queue-a.txt`: `--layers 27 --head --iters 32 --event-timing` (the baseline of this machine), then the per-tile flag on 27 layers if A7 passed | the per-iteration time in `report_table.md` | |
| end | `laptop.sh pull`, commit, push; `docker logout`; `laptop.sh delete`; `tui.py 12` shows `Hourly Rate: $0.00` | | the delete dialog needs `y`; verify the rate, then stop |

Hard stops: at 4 hours of VM time the queue is stopped, the record pulled
and the VM deleted whatever the state; at any point when the balance shows
less than 1 hour, the same.

## Session B (up to 3.5 hours)

Starts from the image if it was pushed (`docker pull`, then `vm.sh` inside
the container: the `setup` stage is skipped and the GPU work starts at
about minute 8) and from `setup.sh` otherwise (minute 25 as in A).

| Minute | Stage | PASS line |
|---|---|---|
| 8 or 25 | `checks`, `reference`, `kernels` (the kernel tests now include the P6 kernels, 100 trials each) | three PASS rows |
| 12 or 29 | B0: attribution check, two runs of the 2-layer graph stopped after `L0.norm1` and after `L0.qkva`, timed from the kernel trace (`measure`); their difference is the cost of one plain gang linear (15 MB) with no successor to absorb it | the difference near 4 us confirms the table; near 40 us says the event gaps shift time to the successor and every per-operator number is re-read before anything is optimized |
| 15 or 32 | B1: 2-layer timing with the P6 attention (`mla_attend` under 60 us from 211, `mla_merge_uv` under 20 us from 64) | `report_table.md` |
| 30 | B2: 27-layer, 32-iteration event timing with every flag that passed | the per-iteration median and P95 in `report_table.md` |
| 45 | B3: the `measure` rows: kernel trace and the four PMC pairs over the same graph | bytes per iteration, achieved bandwidth, launches per generation |
| 60 | B4: correctness again with every flag on: `--layers 27 --head --iters 32`, `compare.py` | 32 ids equal, every boundary PASS |
| 75 | B5: the second lever if time remains: the MoE linears and the elementwise ops per tile, or `USE_NT_WEIGHTS=1` (the design's experiment E2, MAJ-6), each as a 2-layer timing then a 27-layer one | `report_table.md` per variant |
| 150 | `laptop.sh pull`, commit, push, delete, verify the rate | |

Every timing row is the same command with one flag changed, so
`04-results.md` is a table of variants against the 15.6 ms baseline and
the 1.15 to 1.35 ms band of the design.

## The queue files

`env/session/queue-a.txt` and `queue-b.txt` hold the rows above in order,
one `run_fleet.py` argument line per row with an optional trailing
`compare`, `measure` or `continue`; `queue-fault.txt` holds the binary
search. The scripts, the format and the status files are P4 of
`01-preparation.md`.

## What is pulled back and committed

`env/logs/*.out` and the two status files; `env/hw/<date>/` (the record if
`hw` ran, `runs/<name>/` for every queue row: `plan.json`, `wall.json`,
`fleet_run_meta.json`, `fwd_pass.log`, `correctness_report.md` and `.json`,
`report_table.md`, `metrics.json`, `event_timing.json`); `harness/ref/`
JSON files if the reference was re-run. Not pulled: the tensors, the
build directories, compile logs above 400 KB.

## Fallbacks decided now

- M4 not fixed in session A: session B still runs B2 to B5 on the
  27-layer graph without the head (it runs 32 iterations) and the head at
  2 iterations, and the write-up reports the frontier and the bisection.
- The image fails a fifth time: session B pays the 20 minutes of
  `setup.sh`; the Dockerfile is not debugged on VM time beyond reading
  the log.
- The per-tile flag does not help (A7): the performance work of session
  B is P6 plus the boundary fusions of `docs/design-doc/09-expected-performance.md`,
  reductions 1 and 2.
- The balance is short before session B: session B is cut to B2, B3 and
  B4, about 1.5 hours.
