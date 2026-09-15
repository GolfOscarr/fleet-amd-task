# Hardware collection

The first hour of GPU access: every machine assumption the design makes,
measured on the Hot Aisle MI300X VM and committed before the VM is
deleted. Written 2026-09-15, after access was granted and before the
first VM was created.

| File | What |
|---|---|
| [`01-plan.md`](01-plan.md) | why it runs before the build, what is prepared on the laptop first, the ten groups of measurements and the assumption each verifies, the procedure on the VM, what a VM can hide, the facts about the rented machine |
| [`02-checklist.md`](02-checklist.md) | the checks, one row each: command, expected value with its source, what it settles; the sign-off list |

The script and the six probes will live in `env/collect_hw.sh` and
`env/hw/probes/` (the next local work item on this branch; the plan is
not runnable until they exist); each session's record goes to
`env/hw/<date>/`.

Related: `docs/mi300x/99-open-questions.md` (Q1, Q3, Q4, Q5, Q6, Q11 to
Q14), `docs/fleet/99-open-questions.md` (Q6, Q10), `OPEN-PROBLEMS.md`
(MAJ-3, MAJ-4, MAJ-5, MAJ-6, MIN-16, MIN-21, MIN-22, MIN-24, MIN-25),
`docs/design-doc/11-day1-runbook.md` (the session this precedes).
