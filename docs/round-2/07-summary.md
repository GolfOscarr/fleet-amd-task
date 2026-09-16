# 07 - Round 2 in one page

The second and last GPU round of the take-home: one 1x MI300X VM on
2026-09-16, 149 minutes, two sessions run back to back. The log of every
command is `03-session-log.md`, the numbers are `04-results.md`, the lessons
`06-lessons.md`; this page is what a reader needs if they open nothing else.

## What was asked, and what the round set out to do

The task: a Fleet-style persistent megakernel that decodes 32 greedy tokens of
DeepSeek-Coder-V2-Lite-Base after a 1,024-token prompt on one MI300X, in
BF16, with the validated MoE layer (M2) as the required bar. Round 1 (2026-09-15)
reached M2 and M3 (27 layers, 32 iterations, without the head) and left two
open problems: a deterministic illegal memory access at 7 to 9 layers that also
stopped the 27-layer graph with the head at 32 iterations (M4), and a per-token
time of 15.6 ms against a design band of 1.15 to 1.35 ms (MAJ-7). Round 2 was
planned to reproduce and fix the fault, reach M4, and measure where the time
goes (`02-session-plan.md`).

## What was reached

- **M4.** The 27-layer graph with the head runs 32 iterations and the 32 ids
  equal the reference's, first with the alignment flag
  (`env/hw/20260916/runs/L27_head_it32_al65536`, 18:04 UTC), then without any
  flag once the cause was fixed in the plan (`runs/L27_head_it32`), and again
  with every performance flag on (`runs/L27_head_it32_tile_nt`).
- **The fault, named and fixed.** A bisection over the 98 operator labels of
  the 8-layer graph stopped at `L0.gate_up`: the stock fused gate-up kernel of
  the dense layer tiles its M dimension by 16 with no active-token mask, so at
  batch 1 it reads 16 rows, 64 KB, from a 4 KB activation. Whether the 60 KB
  past the buffer are mapped depends on the allocator's layout, which is why 8
  layers faulted, 16 ran, and a uniform shift of every buffer changed nothing
  while a 64 KiB alignment fixed it. The plan now backs every single-row
  activation with 16 rows (`fleet/build_graph.py`, `ROW_SLACK`), and no flag is
  needed (`04-results.md`, "The fault").
- **The image.** `ghcr.io/golfoscarr/fleet-amd-task:20260916`, 25 GB, built
  and pushed in the background of session A.

## The time per token

27 layers with the head, 32 iterations, the runtime's own event clock (the
median of 31 iteration intervals; the host mean over the 32 iterations, which
includes the launch and the first iteration, is 0.5 to 3 ms higher):

| Variant | us per token | Run |
|---|---|---|
| gang linears, this host | 12,268 | `runs/L27_head_it32_al65536` |
| per-tile linears (`--tile-linears`) | 11,514 | `runs/L27_head_it32_tile_al65536` |
| non-temporal weight loads (`--nt-weights`, E2) | 10,230 | `runs/L27_head_it32_nt_al65536` |
| both | 9,575 | `runs/L27_head_it32_tile_nt` |
| design band | 1,148 to 1,349 | `docs/design-doc/09-expected-performance.md` |

9.6 ms per token, 104 tokens per second, 7 to 8 times off the design. Every row
produces the reference's 32 ids. Two more variants, 61 attention splits and the
attention as regular tasks, changed nothing (9,837 and 9,858).

Where the time is: the attention kernel costs 34 us standalone, warm or cold,
and 146 to 215 us inside the megakernel; the merge 11.5 against 46 to 61; the
residual linears 25 to 35 us whatever their size; the norms 13 to 50 us. Neither
the dispatch path, the prefetch depth nor the tile size moved the attention.
The floor is what the megakernel does around every task, paid 33 times per
attention operator and once per operator elsewhere; measuring that overhead on
its own is the first item of the next round (`06-lessons.md`, item 1).

## What it cost

| | |
|---|---|
| VM | 1x MI300X, `enc1-gpuvm015`, provisioned 17:20 UTC by a poller after the single unit had been taken once, deleted 19:49 |
| Time | 149 minutes, billed per minute |
| Money | $7.33: balance $27.56 before, $20.23 after |
| Plan | session A's rows done by minute 58; session B run in the same VM on the user's decision, no second provisioning |

## What is open

| Item | State | Next measurement |
|---|---|---|
| MAJ-7, the per-operator floor | 34 us kernel, 146 us in the graph; the megakernel's per-task overhead | a graph of empty tasks on the event clock, then the completion fences one at a time |
| B3, traffic and bandwidth | not measured: rocprofv3 cannot attach to the torch wheel; 0.52 TB/s from bytes and time | the counters on the kernel suite's standalone launches |
| the route log at 32 steps | 60 of 832 top-k sets differ by one expert near a tie, ids equal | a tie rule within the router floor in `compare.py` |

## Where everything is

| What | Where |
|---|---|
| every command and status row, the decisions, the nine fixes | `03-session-log.md` |
| the correctness evidence, the fault, the three-clock timings, the per-operator table | `04-results.md` |
| what went wrong and what the numbers say next | `06-lessons.md` |
| the record: one directory per run with `plan.json`, `fleet_run_meta.json`, `report_table.md`, `correctness_report.md` | `env/hw/20260916/runs/` |
| the code of the day: the plan fix, `--attend-tasks`, `--split`, the session helpers | branch `gpu/round-2`, commits 9569302 to 178f33a |
| the image | `ghcr.io/golfoscarr/fleet-amd-task:20260916` |
