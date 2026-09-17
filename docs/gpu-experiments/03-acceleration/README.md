# Round 3: acceleration

The third GPU round. Rounds 1 and 2 (`../01-bringup/`, `../02-validation/`)
reached the required milestone and the end-to-end decode; this round is
about speed. The target is the production vLLM baseline quoted for this
model on this machine, 4.5 ms per token, against our 9.58 ms
(`../02-validation/04-results.md`, the event clock).

| File | What |
|---|---|
| [`01-ideas.md`](01-ideas.md) | every acceleration idea with the round-2 evidence behind it, its worth in ms per token, its cost and its dependencies, in seven groups: the per-task overhead (the critical path), fewer boundaries, the kernels, the idle machine, runtime knobs, bytes, measurement; the arithmetic of the target and the stack; corrected on 2026-09-17 where the double-check found the source said otherwise |
| [`02-local-gpu-split.md`](02-local-gpu-split.md) | what the double-check changed; the laptop items L1 to L14 with deliverable, check, time box and the VM row each feeds; the two VM sessions as rows with PASS text and decisions; the budget; the dependency graph |
| [`03-local-preparation.md`](03-local-preparation.md) | the laptop work in detail: eight optimizations (O1 to O8) with direction, core approach at the level of layout and kernel structure, files, laptop checks, time box and the VM row; the instruments in brief (I1 to I7); what the work requires; the double-check against the source and the assumptions that remain |
| [`04-checklist.md`](04-checklist.md) | the progress record of the local preparation: one box per deliverable, ticked only when its laptop check has run, with the VM rows that follow; the double-check of the whole preparation before the session plan |
| [`05-session-plan.md`](05-session-plan.md) | the VM session row by row (one session of up to four hours, gains first, the diagnostics after): the commands, the PASS text, the rule each row applies, the thresholds from round 2, the queue files, the helpers, the playbook, the fallbacks, the expected outputs; the user's decisions of 2026-09-17 and the budget ($20.13) |
| [`06-rehearsal.md`](06-rehearsal.md) | every command of the session expanded in DRY mode by `env/session/rehearse.sh`: the proof that each stage, helper, queue row and mid-session edit exists and parses before the VM is billed |

## Status

| Date | State |
|---|---|
| 2026-09-17 | ideas written and double-checked against the runtime's source (four corrections); the laptop and VM split written; the local preparation detailed (the optimizations first, the instruments in brief); the checklist written; a defect found on the way (two task types share the enum value 188, O0); the instruments' details and the session plan (05) are next; no VM |
| 2026-09-17 | laptop parts of O0, O1, O2, O5, O6, O3, O7 and O8 done on `local/round-3` (each a commit; the flags `--fuse-norm2`, `--fuse-silu`, `--fuse-norm1`, `--probe-before`, `--nt-streams`, `--mfma-attend`, `--prefetch`, all default off until the VM validates them): with the three fusions the graph is 246 operators instead of 326; the MFMA attention's lane arithmetic is tested by emulation, the instruction runs first on the VM; the prefetch side operators need the runtime patch's branch, checked here by a host syntax check and on the VM by `task_graph_check.py`; the instruments I1 to I4 and I6 (the per-worker timing, the shader-clock spin, the empty-task ladder and its queue, the fence knobs and their queue, the clock sampler) done; I5 skipped (no time for vLLM); the preparation double-checked (three small fixes); I7 the session plan (`05`) with its six queue files, three new VM stages and the rehearsal (`06`) done; the "Before the VM" gate is what remains; no VM |
