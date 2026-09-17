# 07 - Session log: the round-3 VM session of 2026-09-17

One session on a 1x MI300X (Hot Aisle, `$2.99` per hour billed per minute),
run from `main` at 74acf6d and then on the branch `gpu/round-3`, following
`05-session-plan.md` rows S0 to S9 as written and then the levers the
measurements pointed at. Times are UTC; the minute mark counts from the
provision at 09:35:09. The numbers are in `08-results.md`; the record is
`env/hw/20260917/` (pulled and committed through the session).

| | |
|---|---|
| VM | 23.183.40.67, 1x MI300X, Xeon Platinum 8470 (13 cores, round 2 had 8), 224 GiB, 13 TB; ROCm 7.2.4 |
| Balance | $20.13 before; $14.90 at minute 107; $13.01 after the deletion (144 minutes, $7.12) |
| Branch | `gpu/round-3` (created at the user's request after the first commits had landed on `main`; `main` reset to origin) |

## Timeline

| UTC | Minute | Row | What happened |
|---|---|---|---|
| 09:34 | -1 | S0 | `grab.sh`: the single unit was taken on the first poll; the full push with the fork in under a minute |
| 09:36 | 1 | S1 | preflight PASS; download 95 s; hw 158 s (a new host); setup 483 s |
| 09:44 | 9 | S2 | checks 84 s, 7 PASS (SPX+NPS1, `import mirage`, the XCD placement, the fences, the CK FMHA negative, the counters, 304 CUs); reference 74 s |
| 09:46 | 11 | S3 | the nine suites on the plain, `nt` and `mfma` builds: 100 of 100 every time (the five suite binaries built in seconds on 13 cores) |
| 09:49 | 14 | S4 | `ktime`: the attention grid 33.8 us (plain), 34.0 (`nt`), **9.25 (`mfma`)**; the merge 11.5; the spin 2,107 MHz on all three |
| 09:50 | 15 | S5 | the baseline with the worker timing: 2 layers 880.3 us, the model 10,250.7 us, ids PASS. Three plan defects found and fixed on the spot (below) |
| 09:57 | 22 | S6 | the three fusions, each alone and stacked: every compare PASS, no gain (10,224 to 10,375 us); the gaps moved to the consumer |
| 10:10 | 35 | S7 | the MFMA attention: 8,952.8 us, ids PASS; the "attention" gap unchanged at 150 us while the "merge" gap fell 48 to 13: the event table's names were off by one (below) |
| 10:17 | 42 | S8 | the prefetch: wiring check PASS, ids PASS, the model slower (12,486.7 us): off |
| 10:22 | 47 | S9 | the final of the prepared levers: 8,904.0 us, ids PASS |
| 10:29 | 54 | S10 | the ladder: the runtime's boundary floor is 2.3 to 2.9 us per operator at one task, 0.19 us per task at 296; the shader clock inside a graph 2,044 to 2,091 MHz |
| 10:33 | 58 | | the user's go for the per-head prep task (prepared on the laptop meanwhile); setup 73 s; the prep suite 100 of 100 |
| 10:36 | 61 | | the per-head prep: step-0 boundaries PASS, 2 layers 508 us, the model **5,207.8 us**, ids PASS; the finals 5,006 to 5,114 |
| 10:42 | 67 | | the router's batched GEMV and a deeper merge batch, pushed: seven rows ran the old kernels (the JIT trap below), 4,984 to 5,109 |
| 10:53 | 78 | | the same rows with the kernels installed: the router 24 to 20.6 us, the model 4,989 to 4,997 |
| 11:01 | 86 | | a queue failed to start (its file written after the push); ten idle minutes |
| 11:07 | 92 | | the vectorized norm helper and the restructured merge: 2 layers 488 us, the model 4,717.8 (worker timing on), **4,601.0 with the attention as regular tasks**; merge 20 to 13.7 us, router to 16, the fused tiles 21 to 18 |
| 11:15 | 100 | | the finals: 4,599.5, 4,578.2, 4,589.3 us (regular attention), 4,618.1 (gang); the megakernel's own `FWD_PASS` clock without the event timing 4,584.0; ids PASS |
| 11:24 | 109 | | the router at eight rows per lane, the merge's weight half row in one batch (suites PASS); the knob queue started; the record pulled ($14.90) |
| 11:27 | 112 | S11 | the knob queue stopped after five rows (a misread of the wall clock, not a hang; the stop was incomplete, next row): `MPK_NO_COMPLETION_FENCE` and `MPK_NO_ACQUIRE_FENCE` both fail the step-0 compare (stale data), the first also no faster (626.8 against 618.3 us on the 2-layer graph); the CAS and sleep knobs not run |
| 11:27 | 112 | | the finals with the deeper batches started while the knob queue's loop was still alive (the kill took the stage's shell, not the queue's): two graph runs at once until 11:35; every row of both queues in that window is discarded (the sleep knob rows, the first finals) |
| 11:39 | 124 | | the finals rerun alone on the GPU with the deeper batches: 4,620.6, 4,639.4, 4,654.4 (regular attention), 4,666.3 (gang), `FWD_PASS` 4,606: 1% slower than 11:15, register pressure; the two constants reverted |
| 11:53 | 138 | | the record pulled, the branch pushed; the user chose to keep the VM for the CK pipeline; reading CK's memory pipeline: two stages for this tile by its own formula, the same as now; the ladder's 0.19 us per task explains the per-tile linears (`08`, what remains): a runtime change, next round |
| 11:59 | 144 | end | the user's yes; `laptop.sh delete --yes`; the team page: `No virtual machines`, `Hourly Rate: $0.00/hour`, balance $13.01 |
| 11:47 | 132 | | the finals with the reverted kernels: **4,583.1, 4,571.2, 4,596.8** (regular attention), 4,575.6 (gang), `FWD_PASS` 4,590.0; ids PASS; the record pulled |

## What the measurements said, in order

1. **The event table named every gap after the wrong operator.** `measure.py`
   assumed event i marks the completion of operator i (1-indexed calls);
   the task graph (`task_graph_rank0.json`, event 5 triggered by the prep
   task, type 185) shows event i marks the completion of operator i - 1,
   event 1 is the begin event and the last event the end of the graph.
   Round 2's "attention 147.5 us" was prep's 143 us; "merge 46 to 59" was
   the attention; "silu 41" was w13; "norm 13.6" was `o_proj`; "o_proj 22"
   was the merge; "combine 20" was w2; the boundary floor of a one-task
   operator is 3 to 5 us, not 13.6. The MFMA run made it visible (the named
   "attention" gap did not move, the named "merge" gap fell 35 us) and the
   worker-timing counter agreed (prep's exec 301k cycles = 143 us). Fixed in
   `measure.py` with the mapping verified against the JSON; every table of
   this session was re-measured.
2. **The fusions save nothing** because a boundary costs 3 to 5 us, not
   13 to 40: the operator removed was cheap and the time belonged to the
   producer. They stay on (correct, fewer operators, neutral).
3. **`mla_prep` was 4.0 ms of the 8.9**: one task, one CU reading 2 MiB of
   `W_uk` at latency. One task per head (16 tasks, the head from the
   runtime's `expert_offset`, task 0 writing the cache rows) took it to
   10 us per layer: 8,904 to 5,208 us.
4. **The remaining single-task and per-head kernels were latency bound**:
   the router at one expert row per round trip, the merge at one column of
   the partials per round trip and a quarter of a weight row per batch, the
   norm helper reading its row twice with 2-byte loads. Batching the loads
   (16-byte words kept raw, converted on use, the FMA order unchanged) took
   the merge from 20 to 13.7 us, the router from 24 to 16, the fused
   norm-linear tiles from 21 to 18: 5,208 to about 4,600 us.
5. **The attention as regular per-split tasks** (`--attend-tasks`, the same
   MFMA kernel) is 30 us per token faster than the gang broadcast; its
   iteration start is 70 us longer (more descriptors to prelaunch).
6. **The runtime's boundary floor** (the ladder): 2.3 to 2.9 us per operator
   at one task, about 0.19 us per task at 296 tasks; about 600 us of the
   4.6 ms is boundaries.

## Plan defects found at S5 and fixed in the session

| Defect | Fix |
|---|---|
| a 2-layer compare row at 32 iterations cannot pass (the boundaries hold iteration 31, the reference is step 0; round 2 only compared `L2_it1`) | every 2-layer compare row split into an `--iters 1` compare row and the `--iters 32` timing row (`queue-c4..d8`) |
| the event clock's median was empty: `num_events` is the runtime's buffer capacity (498 for every graph), not the graph's count | the iteration marker is the highest firing index (`measure.py`, a test) |
| three identical final rows overwrite one run name, so only the last number survives | the finals run at 30, 31 and 32 iterations (`queue-e1`, `e2`, `e5`) |
| `fwd=0` in the queue rows of the worker-timing builds | the `FWD_PASS` lines go to `fwd_pass.log` in those builds; not a failure |

## Traps found on the way

| Trap | Where it lives now |
|---|---|
| the megakernel's JIT compiles the fork's copy of our task headers (`env/setup.sh` step 5b copies them); a kernel pushed mid-session does not reach a graph run until it is installed again; seven rows ran old kernels before this was seen (their exec counters were byte-for-byte the old ones while the standalone suite, which reads the pushed files, had passed with the new) | `env/session/common.sh` `fleet_env` installs the headers before every stage |
| the record's copy of a run name that ran again did not always follow: the worker-timing run of 11:10 is in the record as its 10:45 version (the counters are the old kernels'); the live reads in this log are the source for those numbers (`08`, provenance) | not resolved; a run name per experiment (the iteration-count trick) avoids it |
| a queue file written after the push does not exist on the VM (`no queue file`) | push before `L start queue`; the `wait` should read the stage's own FAIL row, not only the queue's DONE |
| an `until ... DONE <queue>` wait is satisfied by an earlier DONE line of the same file | count the DONE lines, or read the last line |
| `kill` of the `vm.sh queue` shell leaves `queue.sh`'s loop alive: the next row starts after the current run is killed | kill the `queue.sh` process too (`pgrep -f 'queue.sh run'`), then `vm.sh kill --all`; the playbook's "never two graph runs at once" held for a reason |
| `setup.sh` refuses a changed `new_tasks.patch` on a tree that holds the old one | `FULL=1 L push` (the pristine fork) before `L start setup`; setup then re-applies all three patches and rebuilds incrementally (73 s) |

## Rules applied (the plan's RULE rows)

| Rule | Value | Choice |
|---|---|---|
| T1 | MFMA attention 9.25 us standalone (plain 33.8) | O7 the kernel fix if the graph agrees |
| T2 | attention exec 36 to 43 us per task against a named gap of 145 | read as "the runtime"; it was the naming (above) |
| T3 | 2,103 MHz in the graph (amd-smi), 2,107 standalone | no clock gap |
| G5 | every fusion correct, none faster than S5 | kept (the stacked row was the fastest; neutral by the true table) |
| T7 | 8,952.8 against 10,250.7 | MFMA attention on |
| T8 | wiring PASS, 12,486.7 against 10,223.6 | prefetch off |
| T4 | 2.3 to 2.9 us per operator at N = 1; 0.19 us per task at N = 296 | the kernels carry the time, not the runtime |
| T5 | both fence knobs fail the compare; no completion fence 626.8 against 618.3 us | off; the CAS and sleep knobs untested |
| S15 | the per-head prep, the batched loads, the regular attention (the user's go at minute 58) | the number went from 8,904 to about 4,590 |

## Not done

- S11 beyond the fence knobs (the CAS and sleep knobs): the queue was stopped on a misread of the clock, and the fence results made the family unpromising.
- S12 (the probes, the streaming loads): the corrected table made the
  probes' question moot; the streaming loads touch only the attention's
  second pass now.
- I5: the 4.5 ms figure stays the recruiter's; the session's clocks are the
  runtime's event clock and the megakernel's own `FWD_PASS` report, which
  agree within 0.3%.
