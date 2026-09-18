# 09 - Session log: the round-4 VM session of 2026-09-18

One session on a 1x MI300X (Hot Aisle, `$2.99` per hour billed per minute),
run from `main` at fa5919f on the branch `gpu/round-4`, following
`07-session-plan.md` rows S0 to S14 and then two rows the user asked for.
Times are UTC; the minute mark counts from the provision at 04:12:29. The
numbers are in `10-results.md`; the record is `env/hw/20260918/` (69 run
directories, the five `ktime` tables, the bit-diff, the task-graph check),
pulled and committed at 05:49, 06:33 and 06:58.

| | |
|---|---|
| VM | 23.183.40.86 (`enc1-gpuvm016`), 1x MI300X, Xeon Platinum 8470 (13 cores, chosen over the 8-core type by the user), 224 GiB, 13 TB; ROCm 7.2.4 |
| Balance | $12.91 before (the plan quoted $13.01); $6.08 at minute 140; $4.69 after the deletion (167 minutes, $8.22) |
| Branch | `gpu/round-4`, made from `main` before the provision; every queue edit committed as it was made, the record in three pulls |

## Timeline

| UTC | Minute | Row | What happened |
|---|---|---|---|
| 04:11 | -1 | S0 | the provisioning list read twice (two 1x types: 8 cores, 2 available; 13 cores, 1 available); the 13-core host provisioned by hand with `DOWN ENTER` (the same sequence as `grab.sh`, one key more); the address at 04:13; `FULL=1 L push` |
| 04:14 | 1 | S1 | preflight PASS (the model in the host cache); download 86 s; hw 130 s; setup 453 s |
| 04:22 | 9 | S2 | checks 81 s, 7 PASS; reference 71 s, the calibration floors of round 2 (router 3.59e-3, scores 2.61e-3, logits 4.1e-2, norm 4.5e-2, layer 4.3e-3) |
| 04:26 | 13 | S3 | the 19 suites on the plain build 337 s and on the `nt` build 308 s, 100 of 100 every one, the gang rows through the `_xcd` builds. Rule K1: nothing removed |
| 04:37 | 25 | S4 | `ktime` on `nt`, `nt_b4`, `nt_b16`, `nt_strided` and plain in 8 s each (the binaries were built by S3). The o_proj fold's row printed nothing: the stage pointed at the trial's parent directory instead of its `oproj` half; fixed in `vm.sh`, pushed, the row rerun by hand on every build (commit 4106cfa) |
| 04:40 | 28 | G0 | DECIDE T1: the batch constant stays 8 (batch 4 wins the norm form by 7%, batch 8 the residual form by 15%; the two linears of a layer together favour 8), the router's stays 8 (16 is 1.5% faster), the strided map loses everywhere (below). The norm form's 9.4 us is above T1's 8 us line; the deeper merge reads 16.3 us where round 3's build read 11.4; the o_proj fold 39.2 us warm |
| 04:41 | 29 | S5 | queue-f2 started; its first row, the round-3 stack with `-DMPK_W2_CK_TILE`, hung at 100% GFX activity with no forward pass. Killed by hand at 05:02 (20 minutes lost: the queue had no watchdog; the first `pkill` matched the ssh command itself) |
| 05:02 | 50 | | two direct probes under `timeout 300`: the same stack without the define runs (0.55 ms per iteration) and passes the compare on 19 boundaries and the route log (`probe_p1`); with the define it returns ids `[0]`, no `FWD_PASS` line and 17.5 ms for one iteration (`probe_p2`). The define removed from every row, the G3 pair dropped (now identical to G1.2), a per-row `timeout` (600 s) added to `queue.sh` (commit e79e0bc) |
| 05:07 | 54 | S5 | queue-f2 again: its first row hung again, now without the define, and the watchdog ended it as FAIL after 600 s; the one-iteration `--nt-streams` compare row passed in 42 s. Two more probes: 32 iterations without `--worker-timing` complete (`probe_p3`, 27 forward-pass lines as round 3's logs show), one iteration with it hangs (`probe_p4`). `--worker-timing` removed from every round-4 row (commit 2c4428c): the exec-per-class table is not available this round |
| 05:24 | 71 | S5 | queue-f2 a third time: 8 rows PASS in 335 s, every compare PASS. DECIDE T2: o_proj 13.0 to 8.2 us and qkva 14.0 to 11.2 on the 2-layer graph, the kernel's; T13: `--nt-streams` 518.8 against plain 525.5 per iteration, kept; T3: 498.3 at 48 tasks against 518.8 at 96 and 532.3 at 32. The header's w2 GEMV form read 34.4 us against round 3's 23.3 |
| 05:31 | 78 | G3 | the CK multiply restored: the define now includes round 3's file verbatim (`gang_moe_w2_silu_ck_mi300.cuh`; the inline copy of it in the round-4 header was what hung), queue-f7 (commit 65cb9ff): compare PASS, w2 21.4 us, 495.4 per iteration. DECIDE T4: the define joins every later file; T3's 48-task grid into the finals |
| 05:34 | 81 | S6 | queue-f3 (G4): compare PASS; w13 in one round 36.6 us against the stock's 40.7, the iteration unchanged (495.1 against 495.4). DECIDE T5: above the 33 us line, off |
| 05:36 | 83 | S7 | queue-g1 (H2, H4), 6 rows PASS, every compare PASS. DECIDE T9: the four-task router 13.9 us against 15.5, short of the 2 us margin and the iteration slower (505.9 against 495.1), off. DECIDE T10: the merge as regular tasks 20.9 us, at two halves per head 15.0 against 22.0, on |
| 05:41 | 88 | S8 | queue-g2 (H5): compare PASS with the fold's `x_res` boundary, but the fold's gap 42.5 us against 21.9 for the half merge and the GEMV o_proj, the iteration 535 against 498. DECIDE T11: off. The task-graph check PASS with 192 tasks of type 195 and 64 of 207, the expected counts |
| 05:49 | 97 | S9 | the checkpoint pull (the record swept into the T11 commit, split afterwards into cc53dc6 and 9bb6c63); queue-f4: the stream rows 2,242 GB/s (the gang shape with `--nt-streams`), 2,257 (without), 775 (96 tasks of 152 KB), 1,283 (296 of 256 KB); the head rows 650.6 us per iteration at the default grid against 661.2 at 320 tasks. RULE T6: the gang linear's ceiling is 39 us per 90 MB; DECIDE T7: the default grid |
| 05:54 | 101 | S10 | the finals (G7) with every decision applied: **4,287.5, 4,325.3, 4,322.5 us** per token on the event clock, `FWD_PASS` 4.305 ms, ids PASS on every row; the two compare rows FAIL on `head.B15.logits` and the route log |
| 06:00 | 107 | | queue-f8, a bisect of that FAIL at one iteration on the model: the finals' stack at grid 96 passes the head's logits, without the GEMV linears it faults (an illegal memory access with the half merge on 27 layers, a configuration no earlier row ran). The grid was blamed and the finals rerun at 96 (a wrong reading, corrected at 06:12 below) |
| 06:03 | 110 | | queue-f9, the route-log controls: the GEMV linears with the stock merge and the plain round-3 stack on today's header both fail the route log at one iteration, one (step, layer) pair each: MoE layer 4, the sixth expert (weight 0.0396) replaced by another; the 32 output ids match in every run |
| 06:07 | 115 | | the finals at grid 96: 4,458.0, 4,427.8, 4,429.9 us, `FWD_PASS` 4.289 ms, ids PASS |
| 06:12 | 120 | | round 3's own it32 final read from its record: the same two failures (`head.B15.logits` 0.836, the route log with 61 mismatches), ids PASS; round 3 judged its finals by the ids. So the it32 compare rows fail by construction (the head boundary captured after the reference's step) and the tail-expert flips predate this round (MIN-32). The 48-task finals stand; a second 48-task set for the grid A/B: 4,305.9, 4,339.7, 4,307.9, `FWD_PASS` 4.310 |
| 06:12 | 120 | S13 | the bit-diff (G8) of the CK-linear and GEMV-linear step-0 runs, `env/hw/20260918/bitdiff_*.md`: q differs in 2 of 3,072 elements by 2 ULP, attn in 18 by 6 ULP, the router logits in all 64 by up to 1.7e-3, the top-k ids in none |
| 06:19 | 126 | S12 | queue-f6 (G9) with a no-knob baseline row added (the rows carry the half merge): every knob's compare PASS; against 498.7 us per iteration `MPK_NO_LOCAL_CAS` 487.2, `POLL_SLEEP=8` 492.4, `NO_BCAST_CAS` 498.9, `POLL_SLEEP=32` 510.7, `POLL_SLEEP=127` 575.4. RULE T12: the one knob above 2% into a finals rerun |
| 06:27 | 135 | S14 | the finals with `NO_LOCAL_CAS`: 4,261.8, 4,341.3, 4,289.2 us, `FWD_PASS` 4.303 ms, ids PASS; the balance $6.08 at minute 140 |
| 06:33 | 141 | end | the record pulled and pushed; the user asked for two more rows before the deletion |
| 06:46 | 154 | | queue-g3: the GEMV linears at batch 4 on the 2-layer stack 463.2 us per iteration against 498.7 (compare PASS); the finals with `POLL_SLEEP=8` beside `NO_LOCAL_CAS` **4,307.0, 4,265.0, 4,293.5 us, `FWD_PASS` 4.267 ms**, ids PASS |
| 06:53 | 160 | | queue-g4: the finals with `-DGEMV_BATCH=4` besides: 4,323.4, 4,300.0, 4,291.6, `FWD_PASS` 4.289; the 2-layer gain does not carry to the model |
| 06:58 | 166 | end | the last pull (cdf8d61), the branch pushed; the deletion after the user's yes at 06:59; `Hourly Rate: $0.00/hour`, balance $4.69 |

## The decisions, in order

| Rule | Measured | Choice |
|---|---|---|
| T1 (G0) | norm form 8.71 / 9.39 / 16.28 us at batch 4 / 8 / 16, residual form 6.72 / 5.86 / 12.50, router 12.57 / 11.46 / 11.29; strided 13.24, 8.58, 14.71 | 8, the coalesced map, no define |
| K1 (S3) | 19 suites, 100 of 100 on both builds | nothing removed |
| T2 (G1) | qkva 14.0 to 11.2 us, o_proj 13.0 to 8.2 (2 layers) | `--gemv-linears` on |
| T13 (G1.3) | with the GEMV linears 518.8 (`--nt-streams`) against 525.5 (plain) | `--nt-streams` kept |
| T3 (G2) | 519 / 498 / 532 us per iteration at 96 / 48 / 32 tasks | `--linear-grid 48` into the finals |
| T4 (G3) | the CK multiply 21.4 us against the header's GEMV form 34.4 | `-DMPK_W2_CK_TILE` (round 3's file) everywhere |
| T5 (G4) | w13 36.6 against 40.7 us, the iteration unchanged | off (above 33) |
| T8 (G4) | router 15.5 and merge 22.0 us (gaps; no exec table this round) against round 3's 25.2 and 22.0 on the 2-layer graph | the deeper router is a gain of about 9 us; the deeper merge none |
| T9 (H2) | 13.9 against 15.5 us, the iteration 505.9 against 495.1 | `--router-tasks` off |
| T10 (H4) | 20.9 (one task per head), 15.0 (two halves) against 22.0 | `--merge-tasks --merge-halves 2` on |
| T11 (H5) | 42.5 against 21.9 us | `--merge-oproj` off |
| T6 (G5) | 2,242 and 2,257 GB/s (gang shape), 775 (96 x 152 KB), 1,283 (296 x 256 KB) | the ceilings in `10-results.md` |
| T7 (G6) | 650.6 against 661.2 us per iteration | the default head grid |
| G7 (S10) | 4,287.5 / 4,325.3 / 4,322.5 us, spread 0.9% | the number, against round 3's 4,571 to 4,600 |
| T12 (G9) | `NO_LOCAL_CAS` -2.3%, `POLL_SLEEP=8` -1.3%, the rest 0 to +15% | one finals rerun with `NO_LOCAL_CAS`; `POLL_SLEEP=8` added at the user's request |

## Defects found on the VM, with their fix or record

1. The `ktime` stage's o_proj row read the trial's parent directory (the suite writes `tile` and `oproj` halves); fixed on the clock (commit 4106cfa).
2. The inline copy of the CK w2 path under `-DMPK_W2_CK_TILE` returned ids `[0]` and hung at 32 iterations; the file that ran round 3 restored verbatim as the define's path (commit 65cb9ff). The laptop had compile-checked that path only.
3. `--worker-timing` (`MPK_ENABLE_TIMING`) hangs the round-4 header at one iteration (`probe_p4`); off for the round, open for the laptop (OPEN-PROBLEMS, MIN-35).
4. The queue had no per-row watchdog: 20 minutes lost on the first hang; `timeout ${ROW_TIMEOUT:-600}` on the run line since (commit e79e0bc).
5. The half merge without the GEMV linears faults on 27 layers with an illegal memory access (`L27_head_it1_..._rf_w2cktile_mt_mh2`); the finals do not use that configuration; open (MIN-36).
6. The it32 compare rows fail `head.B15.logits` and the route log by construction and by MIN-32, in round 3's record as in this one; the plan's PASS text for the finals was wrong, the ids are the check (the plan and the results page say so now).
7. A wrong reading on the clock: the 48-task grid was blamed for the head-logits FAIL on the strength of a one-iteration bisect (the it1 row passes what the it32 row cannot); found by reading round 3's record, cost one finals set (which became the grid A/B).

## Lessons

In `11-lessons.md`: the approaches with their verdicts (Part 1), the
lessons by area (Part 2), the next round ranked (Part 3), the mistakes of
this session with what each cost (Part 4), and the corrections to the
earlier pages of this set.
