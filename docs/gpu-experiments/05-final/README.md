# Round 5: the final stage

The fifth and last round, a light one. Round 4 (`../04-kernels/`) took the
decode from 4.57 to 4.60 ms per token to 4.26 to 4.34 on the event clock,
below the 4.5 ms production figure, with three kernels of our own (the
GEMV linear, the deeper router, the merge as regular tasks). This round
does not open a new lever: it settles the two things tried at the end of
round 4's session (the runtime knobs `NO_LOCAL_CAS` and `POLL_SLEEP=8`, the
batch constant 4), closes the laptop items round 4 left (the compare's tie
rule and its iteration-aware boundaries, the worker-timing hang, the
half-merge fault, the merge's standalone regression), makes the winning
stack the default of `run_fleet.py`, and ends with one short session that
produces the final numbers with every check green.

| File | What |
|---|---|
| [`01-ideas.md`](01-ideas.md) | the state round 4 left; the two things tried at its end, each with what it switches, what it measured and the question that remains; the laptop items with their diagnosis and fix; the light optimisations that fit a final stage; the routes not taken (the deep ones, listed so the next reader knows why) |
| [`02-local-gpu-split.md`](02-local-gpu-split.md) | the laptop items F1 to F8 with deliverable, check, time box and the VM row each feeds; the VM rows R0 to R5 with PASS text and DECIDE rows; the budget against the balance left ($4.69); the dependency graph |

## Status

| Date | State |
|---|---|
| 2026-09-18 | branch `local/round-5` made from `gpu/round-4` (round 4's PR #11 open); the ideas and the split written; no VM |
