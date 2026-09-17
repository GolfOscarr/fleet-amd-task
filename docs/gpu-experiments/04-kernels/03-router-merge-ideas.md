# 03 - Ideas for the router and the merge

Written 2026-09-17 after the GEMV pages (`01`, `02`), the same way: every
idea, what the source and the round-3 record say about it, what it is
worth and what it costs; the double-check against the source and the
offline compiler in its own section; the laptop and VM split in
`04-router-merge-split.md`. The two kernels are ours
(`fleet/tasks/mi300/moe_router_mi300.cuh`, `mla_merge_uv_mi300.cuh`),
batched in round 3 (`../03-acceleration/08-results.md`: the router 24 to 16
us of exec, the merge 20 to 13.7), and still about twice their potential:

| Operator | Per token, us (`08`, last column) | Per layer | Exec per task | Tasks |
|---|---|---|---|---|
| router with the norm (`moe_router_norm`) | 516 | 19.8 | 16 | one |
| merge (`mla_merge_uv`) | 489 | 18.1 | 13.7 per head | a gang of 8 x 2 tiles: 16 of 296 workers busy |
| o_proj (per-tile, residual; the merge's consumer) | 400 | 14.8 | | 64 |

## What the double-check changed

| Claim of the draft | What the check says | Consequence |
|---|---|---|
| the router's weight loads stream like the linears' | they are plain loads (`*reinterpret_cast<uint4 const *>(row + 8 * i)`), not `StreamSrc`: the 256 KB gate weight of every layer goes through L2 and the memory-side cache with the default policy, while the round-2 lever moved the CK weights out of them | R6 added: the same `sc1 nt` policy as the linears, one line |
| the router's batch of four rows is what runs | the batch loop has a compile-time trip count of four per wave, the case the GEMV probe showed the compiler can unroll and hoist (`01`, K6); the refreshed offline build says it did not here: `k_moe_router` is 124 VGPRs, one batch of 16 loads live (64 raw plus the 32-element slice), so the four round trips per wave are real | R2 stands: eight rows under `#pragma unroll 1` halve them, sixteen make one |
| the merge's loads are batched, so they are efficient | the partials phase reads two 16-byte words per lane 32 bytes apart (`row + q * 8` and `+ 4`), half a line per instruction; the `W_uv` phase gives two lanes to a row (`v = tid >> 1`), so a wave-load touches 32 rows, 16 bytes each: 32 to 64 lines per KB; the lse read is one 4-byte word per lane 33 KB apart (`((size_t)lane * NH + h) * P_ROW + D_C`) | M3 (the coalesced maps) and M1 (the lse read folded into the partials batch) |
| a "last task" reduction needs a counter the plan initialises | every `new` tensor of the plan is a `torch.zeros` buffer attached as an input (`build_graph.py`, line 263), so a counter starts at zero and the last task resets it; the fork's own `splitk_linear_res_atomic_kernel` (`linear_ck_mi300.cuh`, lines 889 to 920) is the pattern: an agent-scope release fence, `atomicAdd`, `is_last`, the reduction, the reset | R5 and M5 reuse it; the fork's reader has no acquire fence before it reads the other tasks' data, and ours adds one (`__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent")`: `buffer_inv sc1`, cheap) |
| the merge as regular tasks needs new pointer rules | the attention already has the switch (`per_tile=attend_tasks` in `graph_plan.py`, `mla_attend_tile_mi300` in `build_graph.py`) and prep takes its task index from `expert_offset`; with whole-tensor imaps a regular merge task needs neither `xcd_offset_dim0` nor the `local` flags of the gang registration | M6 is a flag and a second registration, not a new mechanism |
| the router and merge are register-light | the refreshed offline build (2026-09-17 evening, `resources.txt`): `k_moe_router` 124 VGPRs, `k_mla_merge_uv` 114, `k_mla_prep` 137, the VALU attention 122 to 124, the MFMA attention 160 plus 32 AGPRs (before round 3: 75, 75, 63, 90); the worker union 256 VGPRs, 64 AGPRs, 8 spilled (249 and none before round 3) | the union's owner is the MFMA attention plus the runtime's own registers, not these two; a router or merge at up to about 160 VGPRs does not raise it, and every depth is still read on the union's line (`01`, I3) |

## What round 3 established that this builds on

- The four latency-bound kernels went 2x by loads in flight
  (`09-lessons.md`, lesson 6); the deeper batches (eight router rows, the
  merge's half row) were 1% slower (lesson 7). The 1% is 46 us of 4,600,
  and the finals' spread was 4,571 to 4,600: the reading "register
  pressure" is one of two, the other being that the strided maps (above)
  put more half-used lines in flight. The K9 map settles it.
- A boundary costs 2.3 to 2.9 us and a regular task 0.19 us; a round
  trip inside a task 1 to 2 us in the graph (`08`).
- The router's consumers are w13 (`mask`, `routing`), w2 (`routing`,
  `mask`), the combine (`topk_w`) and the route log; the merge's is
  o_proj (`attn`), whose residual output is what the router normalises.

## The router today

`moe_router_mi300_task_impl<bf16, 2048, 64, 2, 6, 32, 26, NORM = true>`,
one task, 256 threads:

| Phase | What | Round trips |
|---|---|---|
| 1. the norm | `rmsnorm_row`: 8 elements of `x_res` and of `w_norm` per thread (two 16-byte loads), the sum of squares by `block_sum` (LDS, two barriers), the row rounded and multiplied, stored to `h` (global, for w13) and to LDS | 1 (x and w together), then the reductions |
| 2. the slice | each lane reads its 32 elements of the row from LDS | LDS only |
| 3. the GEMV | wave w owns experts 16 w .. 16 w + 15; four rows per batch, four 16-byte loads per row per lane (16 loads, 64 VGPRs raw), the FMAs on raw words, a wave sum per row into LDS | 4 per wave if the batches run in sequence, fewer if the compiler hoisted them |
| 4. softmax and top-k | wave 0: one expert per lane, `expf`, wave max and sum, six argmax rounds of a 6-step butterfly on value and index | none; about 100 shuffles |
| 5. the writes | lane 0 alone: 66 `routing` and 67 `mask` initialisations, then 8 of each of `topk_w`, `routing`, `mask`, the count, 8 of the route log; `logits` one per lane | about 170 sequential stores from one lane |

16 us of exec; the phases in sequence are about 2 (the norm's round trip
and reductions) + 8 (four batches at 2 us) + 1 + 1.

## The merge today

`mla_merge_uv_mi300_task_impl<bf16, 16, 128, 512>`, a gang of 8 slots x 2
tiles, one head per tile:

| Phase | What | Round trips |
|---|---|---|
| 1. the lse row | wave 0: one 4-byte word per lane from column 512 of the head's row in each live split (33 rows, 33 KB apart), wave max, `expf`, the weights and their sum into LDS; a barrier | 1 |
| 2. the partials | thread (q = tid % 64, s = tid / 64): chunk q of 8 floats from rows s, s + 4, ... (9 rows, two 16-byte words each, 18 loads in flight), the FMAs in ascending j, the four groups' sums added through LDS (a barrier), `o[c] = bf16r(sum / tot)` into LDS (a barrier) | 1 |
| 3. `W_uv` | thread pair per output v (`v = tid >> 1`, a half row of 256 elements each): 16 raw words per batch, two batches, the o values read from LDS in the FMA loop, a 2-lane shuffle, the BF16 store | 2 |

13.7 us of exec per head: four dependent round trips of about 2 us plus
the reductions and barriers, with the `W_uv` phase's loads the least
coalesced of any kernel we have (above).

## Group R: the router

### R1. The first weight batch before the norm

The gate weight does not depend on x. Issue the first batch's loads
(16 per lane at four rows, 32 at eight) before `rmsnorm_row`; the norm's
own round trip (x and w) and its two reductions then run under the
batch's latency, and the first multiply finds the rows in registers.
Cost: the batch's raw registers are live across the norm, whose need is
small (16 VGPRs for the row's 8 elements and the sums). Worth about 2 us
per layer (the norm phase hidden), 50 us per token. The same trick as K8
of `01` for the fused linears.

### R2. The depth, controlled

Eight rows per batch under `#pragma unroll 1` (32 loads in flight per
lane, 128 KB per CU, 158 VGPRs by the probe) makes the 256 KB weight two
round trips instead of four; sixteen rows (the wave's whole share, 64
loads per lane, 256 VGPRs plus 28 AGPRs by the probe) one round trip. The round-3 measurement of eight rows (1% slower)
was made without the pragma and with the strided map, so it does not
decide this; M5 of `01` (the cold `ktime` at 4, 8 and 16 rows) does, on
the router's suite binary. Worth 4 us per layer at eight rows, 100 us
per token.

### R3. The coalesced map (K9)

Lane l at `8 l + 512 i` instead of `32 l + 8 i`: every wave-load one
contiguous KB of the expert's row, 8 full lines instead of 32 quarter
lines. The x slice loaded with the same map from LDS. Free; the products
unchanged; the per-lane summation order changes (four runs of eight).

### R4. The writes in parallel

Lanes 0 to 65 write `routing`, 0 to 66 `mask` (the initialisations), the
eight slots by eight lanes, the log by eight lanes: about 170 sequential
store instructions from lane 0 become five per lane, and the task's
completion fence waits for fewer outstanding stores. Worth under 1 us;
free.

### R5. Four tasks and the last one routes

The GEMV over four regular tasks of 16 experts each (64 KB: one batch of
16 loads per lane), the task index from `expert_offset` as prep's is.
Every task computes the norm (4 KB of x, the reductions; task 0 writes
`h`), multiplies its 16 rows, writes 16 logits to the `logits` tensor,
releases (`__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent")` by all
threads, a barrier), and one thread adds one to a counter tensor. The
task that sees 3 acquires, reads the 64 logits, runs the softmax and the
top-k, writes the five outputs and the log as today (R4's map), and
resets the counter to zero. The counter is a `[1]` int32 tensor of the
plan (zeroed at allocation) per layer or shared by all layers (the chain
serialises them).

- **Worth.** The GEMV drops from two round trips (R2) to one; the norm's
  round trip hides under it (R1); the top-k and the writes stay: about
  16 to 6 or 7 us of exec, and four tasks cost 0.6 us more than one on
  the ladder. About 3 us per layer beyond R1 to R4, 80 us per token.
- **Cost, risk.** The cross-task visibility pattern is the fork's (the
  double-check); the top-k's inputs are then 64 floats read from L2 or
  HBM after the acquire (one round trip, which the four-way split has
  paid for by removing one). The `h` output is written by task 0 only,
  and w13 reads it after the operator's event, so no ordering issue. The
  route log, `logits` and the ids are unchanged bit for bit (the same
  arithmetic on the same logits in the same order).
- **Rank.** After R1 to R4: it buys the least per line of code and adds
  the only new synchronisation of the page.

### R6. The streaming policy

The gate weight loads through `StreamSrc` (`sc1 nt`), as the linears'
and the attention's. 256 KB per layer, 6.6 MB per token, kept out of the
memory-side cache the attention re-reads. One line; the round-2 lever's
last uncovered stream.

## Group M: the merge

### M1. The lse read folded into the partials batch

The weights `w_j = exp(lse_j - M)` are needed only at the FMAs, not
before the loads. Each thread loads its rows' chunks and, for its rows,
the lse word (column 512: one more 4-byte load per row, 9 per thread) in
the same batch; after the loads arrive the threads with q = 0 write
their rows' lse into LDS, one barrier, and every thread computes M, its
rows' weights and the total from the 33 values itself (33 `expf` per
thread, cheap; no second barrier since every thread holds the total).
Phase 1's round trip and its barrier disappear: about 2 us per head.
The FMA order and rounding are unchanged (the weights are the same
floats, the products the same).

### M2. `W_uv` issued first

The head's `W_uv` block (128 KB) does not depend on the partials. With
M3's map a wave owns 32 rows of it; the first 16 rows' loads (16 per
lane, 64 VGPRs) are issued before the partials batch, the second 16 right
after the partials' loads, before their FMAs. The `W_uv` phase's two
round trips then overlap the partials' one: the head is about two
dependent round trips instead of four. Registers: the partials batch (9
rows x 2 words = 72 VGPRs, plus the lse words) and 64 of `W_uv` live
together, about 160 with the accumulators; the second `W_uv` batch
replaces the first as it is consumed. Worth about 3 to 4 us per head.

### M3. The coalesced maps and o in registers

- Partials: lane q reads floats `4 q .. 4 q + 3` and `256 + 4 q ..` of the
  row (two 1 KB wave-loads) instead of `8 q .. 8 q + 7` (two half-line
  loads); the chunk it accumulates is then two runs of four columns, the
  reduction through LDS unchanged.
- `W_uv`: one row of 512 BF16 is exactly one wave-load (64 lanes x 8
  elements); wave w owns rows 32 w .. 32 w + 31, lane l the elements
  `8 l .. 8 l + 7` of every row, so the o values a lane multiplies are the
  same eight for all its rows: eight registers, read from LDS once
  instead of inside the FMA loop (today `o_s[c + 8 u + k]` is an LDS read
  per element). Per row: eight FMAs per lane and a wave sum (32 per wave,
  about 200 shuffle steps per tile against 4,096 FMAs); the BF16 store by
  lane 0 of each row, or the 32 results gathered and stored as 64 bytes.
  Every wave-load 8 full lines against 32 to 64 today.

Free; the products unchanged; the per-output summation order changes
(a 64-lane tree instead of two 256-element chains and a pair sum), the
same class of change as K9's, within the tolerance the o boundary's
compare applies today.

### M4. Two tasks per head

32 tasks, each merging the whole head (the partials batch is the same 66
KB) and multiplying 64 rows of `W_uv` (one batch of 16 loads per lane).
The partials traffic doubles (16 x 66 KB more per layer, from the L2 or
the memory-side cache: negligible) and the `W_uv` phase halves. After M1
to M3 the head is about two round trips and the `W_uv` batches are the
second; M4 removes the second `W_uv` batch, about 1 us. Its real value is
as the task shape of M5.

### M5. o_proj folded in: per-head partial products and a last-task reduction

Today the merge writes `attn` (16 x 128 BF16) and o_proj (64 per-tile
tasks, 14.8 us of gap) computes `x_res += attn . W_o^T`. The head's
contribution to every output is `attn[h] . W_o[n, 128 h .. 128 h + 127]`:
a 128-element dot product per output row n, over a 256-byte contiguous
segment of each of the 2,048 rows of `W_o` (512 KB per head; 256 KB per
task with M4's two tasks per head, 128-byte segments). Each merge task,
after its `attn` values are in LDS, streams its `W_o` slice (16 lanes per
row, four rows per wave-load: full lines; 64 loads per lane at two tasks
per head, two batches) and writes a partial vector `[2048]` FP32 into a
workspace `[32, 2048]`; then the release, the counter, and the last task
sums the 32 partials in fixed order (256 KB: 64 loads per lane, two
batches), adds `x_res` in FP32, rounds and stores `x_res` in place, and
resets the counter. The `attn` tensor is still written (4 KB) so the
boundary compare keeps its row.

- **Worth.** The o_proj operator and its boundary disappear: 14.8 us of
  gap per layer, 27 layers. The merge tasks grow by the `W_o` streaming
  (about 4 us at two batches) and the last task by its reduction (about
  4 us, one CU, on the critical path). Net about 32.9 (18.1 + 14.8) to
  about 20 us per layer after M1 to M4: about 0.35 ms per token, the
  largest item of the page.
- **Numerics.** Today: each `attn[h]` is BF16-rounded (unchanged), then
  the CK MFMA accumulates the 2,048-term dot product in its order, adds
  the residual and rounds. Ours: 32 partial sums of 64 terms each in a
  lane's chain and a wave tree, summed in a fixed order, the residual
  added, rounded. A different FP32 order, deterministic (no atomics on
  the data; the counter only), the same class of change as the GEMV's.
  The boundary compare of `x_res` after o_proj and the ids decide it, as
  for every kernel of the round.
- **Cost, risk.** A new task type with three phases and the counter
  pattern; the plan drops one operator per layer (the counts, the labels
  `L{l}.o_proj` used by `--stop-after`, the compare's boundary list); the
  chain rule holds (the operator reads `partials`, writes `x_res`; the
  router reads `x_res`). With `--gemv-linears` the o_proj it replaces is
  the GEMV's per-tile form; either way it is gone. 64 tasks of `W_o`
  streaming spread over the machine become 32 on 32 CUs: 8 MB at 32 x
  the per-CU rate is within a few microseconds of the per-tile form's
  time, which the ladder says was completion-bound anyway.

### M6. Regular tasks instead of the gang

`--merge-tasks`: 16 (or 32 with M4) regular tasks with whole-tensor
imaps, the head and half from `expert_offset`, a second registration
beside the gang's (as `mla_attend_tile_mi300` beside `mla_attend_mi300`).
On its own within the spread (round 3, H). M4 and M5 could keep the
gang shape (8 slots x 4 tiles), but the counter pattern and the per-task
index are simpler as regular tasks, and the scheduler spreads 32 regular
tasks over the XCDs the same way; so M6 is the shape they build on.

## Not chosen

| Route | Why not |
|---|---|
| `W_uv` absorbed into `W_o` offline | o_proj grows from 8 to 32 MB per layer (0.6 GB per token, 0.11 ms of bandwidth) to save the `W_uv` phase that M2 already hides, and the BF16 rounding point of `attn` moves |
| the merge done by the last attention task | the attention's tasks are per split across all heads; the last one would merge 16 heads on one CU in sequence, 16 x the per-head time |
| the top-k recomputed inside w13's tiles (the router ends after the GEMV) | saves 2 us of the router's tail and adds 1 us to each of 296 w13 tiles; w2, the combine and the log need the same outputs anyway |
| MFMA for the `W_uv` or `W_o` products | 64 K to 256 K MACs per task, 0.5 to 1 us of VALU per task, not the bottleneck; K3 of `01` if it ever shows |
| head-major partials (`[NH][splits][516]`) so a head's rows are contiguous | the rows are 2 KB each and coalesced within; contiguity would help the TLB only |
| FP32 atomics for M5's reduction (no last task) | non-deterministic summation order, so the ids could differ run to run; round 3's rule is bit-stable outputs |

## The stack

On the last worker-timing run's 4,713 us (the finals 4,590):

| Item | Today, us per token | Reachable | Gain | Certainty |
|---|---|---|---|---|
| router: R1 to R4, R6 | 516 (19.8 per layer) | about 340 (13) | 180 | high: the same moves as round 3's, one level deeper |
| router: R5 on top | | about 260 (10) | 80 more | medium: the counter pattern is new to us |
| merge: M1 to M3 | 489 (18.1) | about 300 (11) | 190 | high |
| merge: M4 | | | about 25 | low on its own |
| merge plus o_proj: M5 | 489 + 400 | about 540 (20) | 350 | medium: a new three-phase task; the numerics within the compare's class |
| **total** | | | **370 certain, 800 possible** | |

With the GEMV page's 570 to 1,250 the round's arithmetic is 0.94 ms
certain against a 90 us gap.

## To decide before the split

1. R1 to R4 and R6 as one change to the router kernel (a day); R5 as a
   second step behind a flag.
2. M1 to M3 as one change to the merge kernel (a day); M6 as the flag
   that makes M4 and M5 possible; M5 as the second step, its own task
   type and its own flag.
3. Whether M5 is this round's: it is the largest item of the page and
   the only one that removes an operator, and it is the riskiest. The
   split proposes it as the last laptop item, built only if the first
   ones are done.
