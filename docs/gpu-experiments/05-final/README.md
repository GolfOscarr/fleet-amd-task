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
produced the final numbers with every check green: 4,284 to 4,291 us per
token on the event clock, `FWD_PASS` 4,265 (`07-final-numbers.md`).

| File | What |
|---|---|
| [`01-ideas.md`](01-ideas.md) | the state round 4 left; the two things tried at its end, each with what it switches, what it measured and the question that remains; the laptop items with their diagnosis and fix; the light optimisations that fit a final stage; the routes not taken (the deep ones, listed so the next reader knows why) |
| [`02-local-gpu-split.md`](02-local-gpu-split.md) | the laptop items F1 to F9 with deliverable, check, time box and the VM row each feeds; the VM rows R0 to R5 with PASS text and DECIDE rows; the budget against the balance left ($4.69); the dependency graph |
| [`03-local-preparation.md`](03-local-preparation.md) | the laptop items in detail (F1 to F9): direction, approach at the level of files and rules, the files touched, the checks, the time box, the VM row; the order of work (17 hours, 11 for the session's minimum); the double-check of the ideas against the source (the knob unsafe by reading, the fault's first suspect out, the head's event count a plan constant) |
| [`04-checklist.md`](04-checklist.md) | the progress record: one box per deliverable, ticked only when its check has run; the boxes before the VM |
| [`05-session-plan.md`](05-session-plan.md) | the session: one command per row with its PASS text, the DECIDE rows (the balance, A1, A2, R1, R4, R5), the queue files, the protocol, the playbook with round 4's additions, the budget against the $4.69 balance and the decision it forces |
| [`06-rehearsal.md`](06-rehearsal.md) | every command of the session expanded in DRY mode by `ROUND=5 env/session/rehearse.sh`: the stages and the queue rows as `run_fleet.py` lines |
| [`07-final-numbers.md`](07-final-numbers.md) | the results page, filled from the session of 2026-09-18: the number (4,284 to 4,291 us per token, `FWD_PASS` 4,265, every compare row green), the compare made green with the tie's gap, the head's events at 50, 8 and 10, the fault located by halves, the timing build's hang in the buffer form, the decisions, what did not run |
| [`08-session-log.md`](08-session-log.md) | the session as it happened: one row per command with the minute, the status line and the decision; the cost; what it added to the laptop's list |

## Status

| Date | State |
|---|---|
| 2026-09-18 | branch `local/round-5` made from round 4's head and rebased onto `main` after PR #11 merged (51724a0); the ideas and the split written, then the preparation (`03`) and the checklist (`04`) with the double-check that revised the ideas (A1 settled by reading, L2's first suspect out, N3 a plan constant); no VM |
| 2026-09-18 | the laptop items done in the order of `03` Part 3: F3 the `--final` preset (13b3cc6), F1 the tie rule (the commit after it), F2 the iteration-aware boundaries (02b2ec3), F7 the head's event count (0d2b3fb), F8 the tooling and the session plan (e509fdc), F5 the fault read with R4 redesigned as a 2-layer locator (37ff3e0), F4 the timing hang fixed by the buffer form (e3c418c), F6 the merge's 5 us read (bb59318); F9 the gate: 251 tests, the syntax check, 22 offline units, the preflight at 9 PASS, the fork pristine; `07` drafted. The VM waits on the balance decision and the go |
| 2026-09-18 | the user held the VM session after F9 (no provisioning); the branch as pushed (7e06975), the plan `05` and the page `07` ready for a later go; the balance decision (S0 of `05`) still open |
| 2026-09-18 | the balance decided: the whole $4.69, no credits; `05` revised with the hard stop (no new queue after minute 70, the kill at 78, the deletion by 82) and the optional blocks split by their gates (`queue-h6` batch 4, `queue-h8` the 8-event set, the plain-build suites, `queue-h9` A's set again); nine queue files, 36 rows; the go still open |
| 2026-09-18 | the double-check of `06` and `07` (the rehearsal's marks aligned with `05`, the checkpoint pull shown, the optional sets named by their files) and the cross-repo pages updated (the root README, `PROGRESS.md`, `OPEN-PROBLEMS.md`, round 4's README); the laptop work ready for its PR |
| 2026-09-18 | PR #12 opened from `local/round-5` to `main` (the laptop work, 18 commits); the VM session follows the merge on `gpu/round-5` |
| 2026-09-18 | PR #12 merged (809d21d); the session on `gpu/round-5`: 77 minutes on the round-4 host, $3.74, every planned row run or skipped by its gate, the record in four commits; the number 4,284 to 4,291 us per token (`FWD_PASS` 4,265) with the ids equal and every compare row green; no knob and no constant entered the default (A1, A2, R5 and the last-minutes rows all lost on one clock or both); MIN-36 located by halves, MIN-35 still open (the buffer form hangs too); `07` filled, `08` written, the VM deleted at minute 77 with $0.85 left |
