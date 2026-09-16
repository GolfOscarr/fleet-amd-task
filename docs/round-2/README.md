# Round 2

The second round on the MI300X, planned from the artifacts of the first
(`docs/gpu-bringup/`): what is prepared on the laptop first, how the two
sessions on the 1x MI300X VM are spent, the log of what actually ran, and
the results. Written from 2026-09-16 on; branch `local/round-2`.

| File | What |
|---|---|
| [`01-preparation.md`](01-preparation.md) | what the artifacts say; the fault analysed from the plans (every plan quantity is linear in the layer count, so the addresses are the variable); items P1 to P9, each with its deliverable, its laptop check, its time box and what it saves on the VM; the gate before booking |
| [`02-session-plan.md`](02-session-plan.md) | the $27 budget on the 1x shape; what changes with one GPU; session A minute by minute (image, M4, growth curve, per-tile timing) and session B (attention kernels, measurements); the queue files; what is pulled back; the fallbacks decided in advance |
| `03-session-log.md` | written during the sessions: the timeline, every run, every failure and its fix |
| `04-results.md` | written after: the correctness evidence, the timing table of variants against the 15.6 ms baseline, the traffic and bandwidth from the counters, what remains open |

The scripts of this round are `env/session/` (the VM stages, the run
queue, the laptop driver; P4 of `01-preparation.md`) and the flags added to
`harness/run_fleet.py` (`--pad-alloc`, `--tile-linears`). The rules for
reaching the VM are unchanged from `docs/gpu-bringup/06-agent-guide.md`.

## Status

| Date | State |
|---|---|
| 2026-09-16 | plan written; preparation not started; no VM |
| 2026-09-16 | P1 done: `--pad-alloc` and the address record in `run_fleet.py`, the `MPK_EXTRA_HIPCC_FLAGS` hunk in `gfx942.patch`, `env/session/queue-fault.txt`; three tests |
| 2026-09-16 | P4 done: `env/session/{vm,queue,laptop}.sh`, the queue files for both sessions; eleven tests, preflight covers the scripts |
| 2026-09-16 | P3 done: the runtime's chain rule in the plan and the dry run; the debug snapshot re-wired; three tests |
