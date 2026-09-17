# 04 - The work split for the router and the merge

Written 2026-09-17 from `03-router-merge-ideas.md` after its double-check.
The same rules as `02-local-gpu-split.md`: no minute of VM time for work
the laptop can do; every item a deliverable, a laptop check, a time box
and the VM row it feeds; every VM row its PASS text and its decision. The
rows here join the GEMV session (`02`, G0 to G9) in one session plan
(`07`), after the GEMV rows: the router and merge changes are independent
of the GEMV ones and are measured on the same 2-layer graph.

## Local part

| Item | Deliverable | Laptop check | Time box | Feeds |
|---|---|---|---|---|
| **N1. The router, one level deeper** (R1, R2, R3, R4, R6) | `moe_router_mi300.cuh`: the first batch's loads issued before `rmsnorm_row`; `ROUTER_BATCH` rows per batch (default 8) under `#pragma unroll 1`; the K9 lane map for the rows and the LDS slice; the gate loads through `StreamSrc`; the writes spread over lanes 0 to 66; the FMA order per row ascending within each lane's four chunks; the math and the outputs unchanged | `check_syntax.sh`; the suite's `moe_router` rows (exact ids, the log, `topk_w` within BF16 tolerance as today); the offline build's `k_moe_router` line (VGPRs at most 180, no scratch) and the `vmcnt` sequence of the batch loop (32 in flight); `ktime` binaries at 4, 8, 16 rows for M5's curve | 4 h | H1 |
| **N2. The router in four tasks** (R5) | `moe_router_mi300.cuh` gains `SPLIT = 4`: the task's 16 experts from `expert_offset`, the logits written to the tensor, the release fence, the counter add, the last task's acquire, softmax, top-k, writes and reset; the registration `moe_router_norm4_mi300` (type 204: the fork holds 198 and 199, `TASK_HOPPER_TASK_END` and `TASK_NVSHMEM_COPY`, and 200 to 203 are its scheduler task types, found at the integration; 204 to 229 are free; the Hopper-range branch of the generated loader is under `MPK_ENABLE_TMA`, off on MI300) with a `counter [1]` int32 input; `graph_plan.py --router-tasks`; the counts | `check_syntax.sh`; a suite row that launches the four blocks and checks the outputs equal the one-task row's bit for bit; the dry run's counts; the offline variant | 4 h | H2 |
| **N3. The merge, one level deeper** (M1, M2, M3) | `mla_merge_uv_mi300.cuh`: the lse words loaded with the partials batch and reduced through one barrier; the `W_uv` rows as whole wave-loads (32 per wave, the first 16 issued before the partials batch, the next 16 after its loads), the eight o values per lane in registers, the rows' sums by the halving butterfly; the partials map at `4 q` and `256 + 4 q`; the `ROWS_IN_FLIGHT` and `PF_W` constants replaced by `MERGE_W_BATCH` (16) | `check_syntax.sh`; the suite's `mla_merge_uv` rows (the o partials bit-exact against `merge_partials`, `attn` within tolerance); the offline `k_mla_merge_uv` line (VGPRs at most 200) and the `vmcnt` sequence; `ktime` | 5 h | H3 |
| **N4. The merge as regular tasks** (M6, M4) | a second registration `mla_merge_uv_tile_mi300` (type 205) with whole-tensor imaps, the head (and the half, `HALVES = 1 or 2`) from `expert_offset`; `graph_plan.py --merge-tasks [--merge-halves 2]`, the `per_tile` pattern of the attention; the counts | the dry run; `task_graph_check.py`; a test on the plan | 3 h | H4 |
| **N5. The merge with o_proj folded in** (M5) | `mla_merge_oproj_mi300.cuh` (type 207): N3's merge, then the task's `W_o` slice streamed (16 or 8 lanes per row), the partial vector to the workspace `[32, 2048]` FP32, the release, the counter, the last task's fixed-order sum with the residual into `x_res` and the reset; `attn` still written; the registration with inputs `partials, W_uv, W_o, x_res, counter` and outputs `x_res, attn, workspace`; `graph_plan.py --merge-oproj` drops `L{l}.o_proj` and moves its label's boundary to the new operator; `numpy_ref` unchanged (the math is o_proj's) | `check_syntax.sh`; a suite row that launches 32 blocks and checks `x_res` against `numpy_ref` o_proj with residual within the linear tolerance and `attn` bit-exact against the merge row; the dry run's counts and the compare's boundary list; the offline variant's registers | 8 h | H5 |
| **N6. The tooling** | the rows below as `queue-g1.txt` (H1 to H4) and `queue-g2.txt` (H5); the session plan `07` gains the section; the rehearsal regenerated | the queue-file tests; the rehearsal | 1 h | every H row |

About 25 hours; N1 and N3 (9 h) are the certain part.

## The VM part

After the GEMV rows of `02` (the same VM, the same session), on the
2-layer graph with every lever that passed there.

| Row | Needs | Command shape | PASS text | RULE or DECIDE |
|---|---|---|---|---|
| **H0. The suites** | N1, N3 | `V kernels` (the router and merge rows), `V ktime` (the router at three depths, the merge) | 100 of 100; the cold times | RULE: the exec counters must move before a clock is read |
| **H1. The router, deeper** | N1 | `--layers 2 --iters 1 ... compare`, `--iters 32 ... --worker-timing table` (the kernel is not a flag: the pushed header) | 0 FAIL, ids equal; the router gap and exec against 19.8 and 16 | DECIDE: on if the exec fell by 4 us or more; the depth constant from H0's curve |
| **H2. The router in four tasks** | N2 | the pair with `--router-tasks` | 0 FAIL, the ids and the route log equal | DECIDE: on if the gap is below H1's by 2 us or more |
| **H3. The merge, deeper** | N3 | the pair (the pushed header) | 0 FAIL, ids equal; the merge gap and exec against 18.1 and 13.7 | DECIDE: on if the exec fell by 4 us or more |
| **H4. The merge as regular tasks, two per head** | N4 | the pair with `--merge-tasks`, then `--merge-halves 2` | 0 FAIL, ids equal; the merge gap | DECIDE: the shape M5 builds on; on if within the spread or better |
| **H5. The merge with o_proj folded in** | N5 | the pair with `--merge-oproj`; the o_proj boundary in the compare is the new operator's | 0 FAIL on the `x_res` boundary, ids equal; the operator's gap against 18.1 + 14.8 | DECIDE: on if below 26 us (the boundary and half the o_proj gone); off and back to `--merge-tasks` otherwise |
| **H6. The finals** | | with `02`'s G7 rows: every lever that passed, 30, 31, 32 iterations, and the `FWD_PASS` row | ids 32 of 32; the number | the round's number |

About 12 graph rows, 25 minutes, $1.25 on top of `02`'s session.

## The dependency graph

    N1 (router) ---------> H1 -> N2 (four tasks) -> H2
    N3 (merge) ----------> H3 -> N4 (regular tasks) -> H4 -> N5 (o_proj folded) -> H5
    N6 (tooling) needs the flags of N2, N4, N5

N1 and N3 are independent of each other and of the GEMV items; N5 needs
N3 and N4 and, for the o_proj it replaces to be the GEMV form, nothing
(it replaces either form).

## What this split does not cover

- The combine (`moe_mul_sum_add`, 3 to 5 us per layer) folded into w2's
  last tile by the same counter pattern: the same shape as M5 on a
  smaller prize (about 0.1 ms per token); a later page.
- MAJ-8: G2 of `02` decides.
