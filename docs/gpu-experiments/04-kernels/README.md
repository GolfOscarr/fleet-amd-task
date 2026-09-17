# Round 4: kernels

The fourth GPU round. Round 3 (`../03-acceleration/`) took the decode from
9.58 to 4.57 to 4.60 ms per token on the event clock; the target is the
production vLLM figure quoted for this model on this machine, 4.5 ms. This
round works the kernels that hold the remaining time: the CK linears
(2.7 ms per token, a latency-bound K loop), the router (0.5 ms) and the
merge (0.5 ms). The runtime's per-task completion cost (MAJ-8) stays a
separate item.

| File | What |
|---|---|
| [`01-gemv-ideas.md`](01-gemv-ideas.md) | every idea for a batch-1 GEMV linear of our own in place of the CK tile, double-checked against the source and the offline compiler (`env/offline_gfx942/gemv_probe/`): what bounds each linear today (the latency chain, the XCD's rate, or between), the bytes-in-flight arithmetic, the kernel forms (VALU on raw words, MFMA from registers, direct-to-LDS staging; the packed dot2 is not on gfx942), the load policy, the batch depth with its measured register cost, the fused prologues without scratch rows; w13 in one round per XCD and the rows per task; the integration, the VM rows, the routes not chosen; the stack (0.57 ms certain, 1.25 possible) and the decisions before the split |
| [`02-local-gpu-split.md`](02-local-gpu-split.md) | the laptop items L1 to L9 (the kernel, its suite rows and offline variant, the task type and flag, the w2 and w13 forms, the head, the stream probe, the bit-diff, the session tooling, the gate) with deliverable, check, time box and the VM row each feeds; the VM rows G0 to G8 with PASS text and DECIDE rows; the budget; the dependency graph |

## Status

| Date | State |
|---|---|
| 2026-09-17 | branch `local/round-4` made; the codebase and the round-3 record re-read; the ideas for the GEMV linear written, expanded and double-checked (`01`; the probe under `env/offline_gfx942/gemv_probe/` settled the dot2 question, the batch depth's control and its register cost); the laptop and VM split written (`02`); the router and merge ideas are next; no VM |
