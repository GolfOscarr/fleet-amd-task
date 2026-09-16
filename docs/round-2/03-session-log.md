# 03 - Session log

Filled during the sessions, one row per command, from the status files
(`env/logs/session.status`, `env/logs/queue.status`, pulled into
`env/hw/<date>/logs/`). The plan is `02-session-plan.md`; the row ids are
its. Times are UTC. Nothing here is written from memory: every result row
quotes a status line or a file in the record.

## Session A

| | |
|---|---|
| Date | |
| VM | `ssh hotaisle@<ip>`, host name from `env/hw/<date>/summary.md` or `hostname` |
| ROCm, hipcc | |
| Balance at provision, at deletion | |
| Record | `env/hw/<date>/` |
| Image | pushed / built only / failed (tag) |

### Timeline

| UTC | Row | Command | Status line or result | Note |
|---|---|---|---|---|
| | A0 | `laptop.sh balance` | | |
| | A0 | `laptop.sh provision` | | |
| | A1 | `start download` / `login` / `start image` / `start setup` / `start hw` | | |
| | A2 | `start checks` | | |
| | A3 | `start reference` | | |
| | A4 | `start kernels` | | |
| | A5 | `start queue queue-a.txt` | | |
| | A6 | `start bisect queue-fault.txt` | | |
| | A7 | `start queue queue-fix.txt` | | |
| | A8 | `start queue queue-a2.txt` | | |
| | A9 | image status, balance | | |
| | end | `pull`, `delete --yes`, rate | | |

### Decisions (the DECIDE rows)

| Row | Rule | Measured | Chosen | Told the user at |
|---|---|---|---|---|
| A4 | the P6 kernels pass their suites | | | |
| A5 | the fault tree | | | |
| A7 | first fix row that passes | | | |
| A8.4 (B0) | near 4 us or near 40 us | | | |
| A8.5 | `o_proj` under 10 us; attention under 60 and 20 us | | | |
| A9 | the push cutoff | | | |
| A11 | balance above $13 | | | |

### Failures and fixes

| Failure | Cause | Fix, and where it lives now |
|---|---|---|
| | | |

## Session B

| | |
|---|---|
| Date | |
| VM, ROCm | |
| Balance at provision, at deletion | |
| Started from | the image / `setup.sh` |

### Timeline

| UTC | Row | Command | Status line or result | Note |
|---|---|---|---|---|
| | | | | |

### Decisions

| Row | Rule | Measured | Chosen | Told the user at |
|---|---|---|---|---|
| B1 | the thresholds of `02` | | | |
| B5 | which lever | | | |

### Failures and fixes

| Failure | Cause | Fix, and where it lives now |
|---|---|---|
| | | |

## Artifacts

| Path | What it proves |
|---|---|
| `env/hw/<date>/runs/<name>/` | one directory per queue row (`plan.json`, `wall.json`, `fleet_run_meta.json`, `fwd_pass.log`, reports) |
| `env/hw/<date>/logs/` | the stage logs and the two status files |
| `env/logs/bisect.result` (in `logs/`) | the first faulting label |
| `fleet/tasks/results/kernel_tests.json` | the kernel suites |
| `harness/ref/*.json` | the reference ids, route log, calibration |
