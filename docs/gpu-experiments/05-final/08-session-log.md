# 08 - Session log: 2026-09-18, one MI300X, 77 minutes

The run of `05-session-plan.md` as it happened, one row per command,
with the minute mark from the provision (11:09:03 UTC), the result line
of `env/logs/session.status` and what was decided. The numbers are read
in `07-final-numbers.md`; the record is in `env/hw/20260918/runs/`
(the round-4 record of the same day sits beside it, the run names
distinguished by `_final`).

| | |
|---|---|
| Host | `enc1-gpuvm016` at 23.183.40.86, the 13-core 1x MI300X (Xeon Platinum 8470), ROCm 7.2.4: the same host as round 4, so the `hw` stage kept its record |
| Branch | `gpu/round-5`, made from `main` at 809d21d (PR #12) before the provision; the balance read and the queue-h3 header in the first commit; the record in three pulls (the checkpoint at minute 34, the interim at 62, the last at 74) plus the knob rows |
| Cost | $4.59 to $0.85: $3.74 for 75 billed minutes at $2.99 per hour; the deletion at minute 77 on the user's yes |
| Stopped by | every planned row run or skipped by its gate, then the user's last-minutes ask (`queue-h0`), then the deletion; no hard-stop mark was reached with a queue running |

## The rows

| Minute | UTC | Row | Command | Result | Decision |
|---|---|---|---|---|---|
| -4 | 11:05 | S0 | `L balance` | `No virtual machines`, $4.59 (ten cents under the plan's $4.69: round 4's last minute billed after its reading), $0.00/hour | DECIDE: the user's go at 11:08 ("Poll VM proceed right away"); the 13-core host listed (1 available) beside the 8-core type (2), provisioned by hand with round 4's `DOWN ENTER` sequence |
| 0 | 11:09 | R0 | the provision; `L ip`; `FULL=1 L push` | the address at 11:09; the first push failed on a stale host key for the reused address (`ssh-keygen -R`), the second completed; the fork at 51dce4f on the VM | |
| 2 | 11:11 | R0 | `L start download`; `L start setup`; `L start hw` | `PASS download 76s`; `PASS hw 0s` (skipped: the host and ROCm already in `env/hw/20260918/summary.md`); `PASS setup 491s` at 11:19 | |
| 11 | 11:20 | R0 | `V preflight`; `L start checks` | `PASS preflight` (the venvs, the model, ROCm 7.2.4); `PASS checks 81s` with 7 PASS lines at 11:21 | |
| 13 | 11:22 | R0 | `L start reference`; `L wait reference 6` | `PASS reference 70s`; the route log with `w_all` on all 832 entries; the router floor 0.003593 (the laptop's value), tolerance 0.0144 | |
| 15 | 11:24 | R0 | `L start kernels nt`; `L wait kernels` | `PASS kernels 338s`: 19 of 19 suites at 100 of 100, overall PASS | RULE: no FAIL, the header as pushed |
| 21 | 11:30 | R1 | `L start queue env/session/queue-h1.txt` | `PASS queue 186s`: the two model compares PASS (8 boundaries PASS, 64 not captured; the route log 1 tie, 0 cascades, 0 disagreements at tol 0.0144), the 2-layer compare PASS (19 boundaries, 0 mismatches) | DECIDE R1: zero disagreements, the compare claim stands |
| 25 | 11:34 | R2 | `L start queue env/session/queue-h2.txt` | `PASS queue 558s` at 11:43: eight rows PASS with compare and table; A 4,287.0 / 4,290.9 / 4,284.2, B 4,261.4 / 4,260.3 / 4,257.8; `FWD_PASS` A 4,265.0, B 4,274.0 (the it29 rows' lines in `run.out`, `fwd_pass.log` empty without the event timing, as in round 4) | DECIDE A1: B above A on the `FWD_PASS` clock, the stack stays A; A's spread 0.16%, `queue-h9` not needed |
| 34 | 11:43 | | `L pull`; the commit | "logs only: staged" with the runs staged: committed by hand as 8475160 (181 files), pushed | |
| 35 | 11:44 | R5 | `L start queue env/session/queue-h5.txt` | `PASS queue 204s`: the 2-layer head at 50, 8 and 10 slices 615.2, 623.6, 635.4 us per iteration, 50, 8 and 10 head events; the model row at 8 PASS | DECIDE R5: the median rose, the slices stay 50; `queue-h8` skipped |
| 39 | 11:48 | R4 | `L start queue env/session/queue-h4.txt` | `PASS queue 168s`: the halves-2 configuration runs at 2 layers (compare PASS), both cuts run, the halves-1 control faults (rc 1; the queue's `fault=2` counts the aperture-violation line and the `hipErrorIllegalAddress` line) | RULE: the first row passed, `queue-h7` runs |
| 43 | 11:52 | R4 | `L start queue env/session/queue-h7.txt` | `PASS queue 207s`: 3, 5, 9 and 14 layers run (compare PASS on each) | the halves-2 fault needs more than 14 layers or the head; into MIN-36 |
| 48 | 11:57 | R3 | `L start queue env/session/queue-h3.txt` | the first row (the timing build, one iteration) hung: at 12:01 the kernel at 100% GFX activity for 281 s with `fwd_pass.log` empty and no compile running | RULE R3: the hang stands in the buffer form; the second row skipped |
| 48 | 11:57 | | `L report --balance` | $2.24, runout 43 minutes (minute 91) | the marks hold |
| 53 | 12:02 | | the queue loop killed by pid (22571); `vm.sh kill --all` found no run (its anchored pattern misses the venv-path launch); the run and its `timeout` killed by pid (22587, 22585); GFX activity 0% | `FAIL queue 299s` on the row; the kill pattern fixed on the laptop after the session |
| 54 | 12:02 | R3 | `L start ktime nt` | `PASS ktime 8s` (the builds cached): `mla_merge_uv` 16.28 us | |
| 54 | 12:03 | opt 1 | `L start queue env/session/queue-h6.txt` | `PASS queue 278s` at 12:08: batch 4 at 4,293.3 / 4,275.8 / 4,270.6, `FWD_PASS` 4,294.0, compare PASS | DECIDE A2: not a win on both clocks, the constant stays 8 |
| 59 | 12:08 | opt 3 | `L start kernels` (the plain build) | `PASS kernels 307s` at 12:13: 19 of 19 at 100 of 100 | RULE K1: nothing to remove |
| 62 | 12:11 | | `L pull`; the commit | f011538 (221 files: R5, R4, the bisect, the hung row's log, `ktime`, batch 4), pushed | |
| 65 | 12:14 | | `L pull`; `L report --balance` | 28c457d (the plain suites' results and logs); $1.45, runout 28 minutes | DECIDE (user): "Hold, one more thing", then at minute 66 "try any methods to accelerate more; revert if it fails" |
| 68 | 12:17 | the last minutes | `queue-h0.txt` written, tested, pushed (`L push`); `L start queue env/session/queue-h0.txt` | `PASS queue 347s` at 12:23: sleep 8 with batch 4 4,287.8 (`FWD_PASS` 4,296.5), sleep 16 4,316.0 (`FWD_PASS` 4,300.5), sleep 4 4,284.0 | none wins both clocks: nothing enters the default. The user allowed a kernel change at minute 71; not taken (a JIT build, a possible fault and the pull did not fit before the deletion, and the one change with a read cause gains nothing in the graph) |
| 74 | 12:24 | end | `L pull` | 2b65418 (the five knob rows), pushed; the branch level with `origin/gpu/round-5` | |
| 76 | 12:25 | end | the deletion asked | DECIDE (user): "Delete now" | |
| 77 | 12:26 | end | `L delete --yes`; `L balance` | `No virtual machines`, `Hourly Rate: $0.00/hour`, $0.85 | |

## What the session added to the laptop's list

- `vm.sh kill` matched only a bare `python harness/run_fleet.py`; the
  queue launches the venv's interpreter by its full path under `timeout`,
  so `kill --all` reported no run while the hung kernel spun. Fixed after
  the session: the pattern accepts a path before `python`. The queue loop
  itself is still killed by its own pid (`pgrep -f '^bash env/session/queue.sh run'`).
- `L pull` commits the record itself when only logs and results changed
  and stages it when runs came in; the checkpoint at minute 34 needed the
  commit by hand, as the playbook says.
- The `hw` stage skips a host it has recorded; on a reused host the
  session gets no fresh hardware record, which was right here.
- A reused address carries a stale host key; `ssh-keygen -R <address>`
  before the first push.
- The first timing row's watchdog would have cost 10 minutes; reading
  the GFX activity and the process list at minute 4 of the row
  (`amd-smi metric -g 0 --usage`, `ps`) told the hang from a build and
  saved six.
