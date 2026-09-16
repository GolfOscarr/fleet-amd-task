# 06 - Agent guide: reaching the GPU and running commands on the VM

For the next agent session. Everything here was learned on 2026-09-15;
the recipes are the ones that worked, the traps are the ones that cost
time. Read `05-next-session.md` for what to do once the VM is up; this
file is about the mechanics of getting there and back.

## The account and the money

- Hot Aisle, https://admin.hotaisle.app/. The user's SSH key is on the
  account; the team belongs to the recruiter and is funded in small
  amounts ($20 to start). Billing is per minute after a one-hour minimum,
  only deletion ends it, and running out of balance deletes the VM with
  everything on it.
- Shapes seen: 2x MI300X VM at $5.98 per hour (26 cores, 448 GiB, 13 TB
  NVMe, the only one listed on 2026-09-15); the 1x at $2.99 is the shape
  of round 2 (`docs/round-2/02-session-plan.md`). On a 1x everything runs
  on GPU 0 in sequence: the reference run, the calibration, the kernel
  tests and the probes queue behind the graph runs (about 4 minutes in
  total). On a 2x the second GPU can take them, never a second graph run
  (see "one graph run at a time").
- The balance and the runout time are on the team page. Check them at the
  start and before any long upload. The first session cost $12.56 for
  about 4.5 hours.

## The admin TUI from a script

`ssh admin.hotaisle.app` is a full-screen terminal UI that refuses a
non-interactive session ("Requires an active PTY"). `env/hotaisle/tui.py`
drives it through a pseudo-terminal: it connects, waits, sends keys, hangs
up, and prints the last screen as text (rough: the screen is reconstructed
from the byte stream). Nothing persists between calls; every call is a
fresh login that lands on the team page.

| Want | Command | Read from the output |
|---|---|---|
| balance, VMs, runout | `python3 env/hotaisle/tui.py 12` | `Available Balance`, `Hourly Rate`, `Estimated Runout`, the VM list |
| what can be provisioned | `python3 env/hotaisle/tui.py 8 n WAIT4` | the `Available Resources` lines with price and specs |
| provision the listed VM | `python3 env/hotaisle/tui.py 8 n WAIT5 ENTER WAIT8 y WAIT25` | the dialog states the 1-hour minimum; `y` confirms it (Enter on the `yes` button does not); the VM appears on the team page about 10 s later |
| the VM's address | `python3 env/hotaisle/tui.py 10 ENTER WAIT8` | `SSH: ssh hotaisle@<ip>` on the VM page (Enter on the team page opens the selected VM) |
| delete the VM | `python3 env/hotaisle/tui.py 10 ENTER WAIT6 DOWN DOWN DOWN DOWN DOWN DOWN WAIT2 ENTER WAIT4 y WAIT20` | six `DOWN`s reach `Delete VM` in the operations list; the dialog offers `y` (normal, after the minimum hour), `f` (force, no refund), `n`; verify with the balance call: `No virtual machines`, `Hourly Rate: $0.00` |

If a rendering is unreadable, `env/hotaisle/raw.bin` holds the raw bytes of
the last call; strip the escape sequences and search for the phrase you
expect (the `python3 -` snippets in the session transcript did that).
Never send `n` twice in one call: a second provisioning dialog is one `y`
away from a second VM.

## Reaching the VM

- `ssh hotaisle@<ip>` with the same key; `StrictHostKeyChecking=accept-new`
  on the first connection. Port 22 only; the user has passwordless sudo.
- The VM has no GitHub credentials. The repo goes over by rsync from the
  laptop (25 MB plus the submodule):

```
rsync -az --exclude .venv --exclude .venv-fleet --exclude env/hw/build --exclude env/hw/probes/work \
  --exclude env/offline_gfx942/work --exclude docs/report --exclude .omc --exclude harness/fleet_out \
  --exclude env/logs --exclude '__pycache__' /Users/hyeonseop/Desktop/metalOps/ hotaisle@<ip>:/home/hotaisle/metalOps/
```

  and results come back the same way, always with an absolute destination
  (a relative one nested the record inside itself once).
- The image is Ubuntu 24.04, ROCm 7.2.4, hipcc 7.2, Python 3.12 without
  `ensurepip`, Docker, no cmake, no rocm-bandwidth-test. `env/setup.sh`
  handles the venv and the build; `env/collect_hw.sh` handles cmake.
- The model: 30 GB from Hugging Face in 79 s, from a throwaway venv made
  with `--without-pip` and `get-pip.py` (`05-next-session.md`, step 1).

## Running commands that survive the session

Every long command runs detached, writes its own log, and is polled by
reading that log; the agent's SSH call returns at once. The pattern:

```
ssh hotaisle@<ip> 'cd ~/metalOps && source .venv-fleet/bin/activate && \
  export MIRAGE_HOME=$HOME/metalOps/repos/fleet-chiplet-megakernel AMDGPU_TARGETS=gfx942 HIP_VISIBLE_DEVICES=0 && \
  (setsid nohup bash -c "python harness/run_fleet.py --layers 2 --model-dir $SNAP > env/logs/m2.out 2>&1; \
     python harness/compare.py --fleet harness/fleet_out/L2_it1 > env/logs/m2_compare.out 2>&1" \
     > /dev/null 2>&1 < /dev/null &); echo started' </dev/null
```

- `setsid nohup ... < /dev/null &` inside a subshell, plus `</dev/null` on
  the ssh itself: without them the SSH call hangs on the background job's
  open descriptors or the tool moves it to the background with no output.
- Poll with `grep` on the log: `grep -c "mpk()"` (a graph run finished),
  `grep -c AcceleratorError` (it faulted), `grep -c FWD_PASS` (iterations
  completed), `tail -2`. A watcher that loops until a phrase appears is
  fine inside one detached ssh call with `run_in_background`, and the
  harness re-invokes the agent when it returns.
- To wait for another run, poll for its log line, not for its process:
  `pgrep -f run_fleet.py` matches the shell that is doing the waiting (its
  own command line contains the pattern) and loops forever. If a process
  match is unavoidable, anchor it: `pgrep -f "^python harness/run_fleet.py"`.
- Never `pkill -f <pattern>` from an ssh command whose own line contains
  the pattern: it kills the ssh session (twice on 2026-09-15). Kill by pid
  from an anchored `pgrep`.
- One graph run at a time on the VM, whatever the GPU: the runtime JIT
  compiles every graph into one `permanent_output_dir` under the Fleet
  tree, and concurrent runs overwrite each other's generated kernel. The
  reference run, the calibration, the kernel tests and the probes do not
  use it; on a 2x they can share the other GPU, on a 1x they run before
  the queue.
- `--iters` is at most 32 (the RoPE tables are 1,056 rows); the op labels
  for `--stop-after` are in `fleet/graph_plan.py`.
- Logs live in `env/logs/` on the VM (gitignored there); the ones worth
  keeping are copied into `env/hw/<date>/logs/` and committed. Compile
  logs above 400 KB are dropped.

## The session shape that worked

Since round 2 every step below is a command of `env/session/laptop.sh`
(provision, ip, push, login, start, status, pull, delete) and a stage of
`env/session/vm.sh` (download, image, setup, hw, checks, reference,
kernels, queue, bisect), each detached with its own log and a PASS or FAIL
row in `env/logs/session.status`; the graph runs come from queue files
(`env/session/queue-*.txt`) through `queue.sh`, one at a time; `laptop.sh
wait` and `report`, `vm.sh check`, `preflight`, `kill` and `gdb`, `pf.sh`,
`queue_flag.py` and `addr_diff.py` are the helpers (the table in the plan).
The plan that uses them is `docs/round-2/02-session-plan.md`. The shape itself:

1. Provision; note the time and the balance.
2. rsync the tree; start the model download and the image build;
   `collect_hw.sh` if the machine or the ROCm version is new (a minute);
   otherwise skip.
3. Build or pull the environment (`env/docker/README.md`, or
   `env/setup.sh` with `SKIP_DOWNLOAD=1`); `check_day1.sh`.
4. Reference run, calibration and kernel tests: on a 2x on GPU 1 while
   the first graph runs on GPU 0; on a 1x before the queue.
5. The work of the session, one graph run at a time, each with its log.
6. Every 30 minutes and before anything risky: rsync the record and the
   logs back, commit, push. The record directory is `env/hw/<UTC date>/`;
   never run the laptop dry run of `collect_hw.sh` against it (`--out`).
7. `docker logout ghcr.io` if the registry was used; delete the VM; verify
   the rate is $0.00.

## What the user expects

- To be told right before the GPU is first touched, and progress reports
  at each step after that.
- No new VM, no deletion, no move into a new milestone without asking;
  everything reversible inside the session without asking.
- Commits on the branch as work lands, each one checkable on its own;
  documents without emoji; the recruiter brief in `docs/report/` never
  committed.
