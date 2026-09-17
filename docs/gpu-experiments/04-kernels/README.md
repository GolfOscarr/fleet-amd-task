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
| [`01-gemv-ideas.md`](01-gemv-ideas.md) | every idea for a batch-1 GEMV linear of our own in place of the CK tile: what bounds each linear today (the latency chain, the XCD's share of HBM, or between), the bytes-in-flight arithmetic, the kernel forms (VALU on raw words, packed dot2, MFMA from registers, direct-to-LDS staging), the load policy, the batch depth, the fused prologues without scratch rows; the shaping of w13 into one round per XCD and the rows per task; the integration, the VM rows, the routes not chosen; the stack (0.77 ms certain, 1.27 possible) and the decisions before the split |

## Status

| Date | State |
|---|---|
| 2026-09-17 | branch `local/round-4` made; the codebase and the round-3 record re-read; the ideas for the GEMV linear written (`01`); the local and GPU split is next; no VM |
