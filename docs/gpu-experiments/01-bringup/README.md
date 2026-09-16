# GPU bring-up

Stage 4 of `PROGRESS.md`: the first sessions on the MI300X. The set
holds the plan and checklist for the hour of measurements that precedes
the build, what those measurements found, the log of every run of the
first sessions (gate 1, M1 to M3, the timing, the image builds) with
every failure and its fix, the lessons and ideas, and the quick start
for the next session. Written from 2026-09-15 on.

| File | What |
|---|---|
| [`01-plan.md`](01-plan.md) | why it runs before the build, what is prepared on the laptop first, the ten groups of measurements and the assumption each verifies, the procedure on the VM, what a VM can hide, the facts about the rented machine |
| [`02-checklist.md`](02-checklist.md) | the checks, one row each: command, expected value with its source, what it settles; the sign-off list |
| [`03-lessons-and-ideas.md`](03-lessons-and-ideas.md) | what went wrong on the first sessions and the fix for each; fourteen ideas the measurements suggest, each with the number it rests on |
| [`04-session-log.md`](04-session-log.md) | the timeline of every run of 2026-09-15, every failure with its cause and fix, what each artifact proves |
| [`05-next-session.md`](05-next-session.md) | superseded by [`docs/gpu-experiments/02-validation/02-session-plan.md`](../02-validation/02-session-plan.md); from a fresh VM to a running graph in ten minutes with the pushed image; what to do first |
| [`06-agent-guide.md`](06-agent-guide.md) | for the next agent: the Hot Aisle TUI from a script (`env/hotaisle/tui.py`), provisioning and deletion, reaching the VM, commands that survive the session, the traps, the session shape, what the user expects |

The script and the six probes are `env/collect_hw.sh` and
`env/hw/probes/`; each session's record goes to `env/hw/<date>/`, and the
first one is `env/hw/20260915/`.

## Results

The collection ran on 2026-09-15 on Hot Aisle VM `enc1-gpuvm005`, two AMD
Instinct MI300X VF devices under ROCm 7.2.4 and hipcc 7.2.53211, device 0
used throughout. The record is `env/hw/20260915/`, with the row-by-row
readings in its `summary.md` and the raw command output beside it. The
headline numbers: SPX + NPS1, 304 CUs over 8 XCDs, 64 KiB LDS, 4 MiB L2 per
XCD, 192 GB HBM and no L3 row; workgroups placed round-robin as
`xcd == (blockIdx.x + 4) mod 8`, perfectly balanced and stable across
launches and processes, which is not the mapping the Fleet runtime assumed;
3.943 TB/s streaming read with the knee at prefetch depth 4 and BabelStream
Triad at 4,133,844 MB/s, so the design's 3.66 to 4.3 TB/s band holds;
pointer-chase latency 81 ns in L2, 258 ns at 64 MiB and 342 ns at HBM, so
the memory-side cache tier is real; agent-scope fences lowering to
`buffer_wbl2 sc1` and `buffer_inv sc1` and costing 115 to 317 ns each; a
cross-XCD one-way release-to-acquire hop of 703 ns; one wave per SIMD at the
worker kernel's LDS footprint with all 304 blocks co-resident; and
`rocprofv3` PMC collection working inside the virtual function with 128 TCC
instances, validated to the byte against a 1 GiB copy.

Eleven questions closed on the strength of these: `docs/mi300x` Q1, Q3, Q4,
Q5, Q6, Q9, Q11, Q12 and Q14, and `docs/fleet` Q6 and Q10. Three were
narrowed rather than closed: `docs/mi300x` Q13, `docs/design-doc` DQ1 and DQ5. The
consolidated view is in `OPEN-PROBLEMS.md`.

Related: `docs/mi300x/99-open-questions.md` (Q1, Q3, Q4, Q5, Q6, Q11 to
Q14), `docs/fleet/99-open-questions.md` (Q6, Q10), `OPEN-PROBLEMS.md`
(MAJ-3, MAJ-4, MAJ-5, MAJ-6, MIN-16, MIN-21, MIN-22, MIN-24, MIN-25),
`docs/design-doc/11-day1-runbook.md` (the session this precedes).
