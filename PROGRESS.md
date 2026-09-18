# Progress

Fleet-style batch-1 decode for DeepSeek-Coder-V2-Lite-Base on one AMD MI300X.
Time limit: 5 days. Target: gfx942, BF16, 1024-token prompt, 32 greedy tokens.

Last updated: 2026-09-18 · branch `gpu/round-5` (round 5, the final stage, run on the MI300X: `docs/gpu-experiments/05-final/07-final-numbers.md`, **4,284 to 4,291 us per token**, `FWD_PASS` 4,265, the ids equal and every compare row green; 77 minutes, $3.74, the balance at $0.85; the log in `08-session-log.md`) · round 4 on `gpu/round-4` (run on the MI300X: 4.26 to 4.34 ms per token, below the 4.5 ms target; `docs/gpu-experiments/04-kernels/09-session-log.md`, `10-results.md`; prepared the same day on `local/round-4`, PR #10); round 3 was `gpu/round-3` (2026-09-17: 4.57 to 4.60 ms per token; `docs/gpu-experiments/03-acceleration/`); round 2 was `gpu/round-2` (2026-09-16: M4, the fault's cause, the timings; `docs/gpu-experiments/02-validation/`); round 1 was `local/gpu-bringup` (2026-09-15: hardware record, gate 1, M1 to M3; `docs/gpu-experiments/01-bringup/`)

**Where we are (2026-09-18, the end):** round 5, the final stage
(`docs/gpu-experiments/05-final/`, branch `gpu/round-5`), ran its one
session: the round-4 stack through `run_fleet.py --final` gives **4,284
to 4,291 us per token on the event clock, `FWD_PASS` 4,265**, the 32 ids
equal to the reference's on every final and every compare row green (the
route log at zero disagreements by the tie rule, every captured boundary
PASS), inside round 4's 4,262 to 4,341 band and 4.6 to 4.8% below vLLM's
4.5 ms. The knobs and constants tried at the end of round 4 and in the
last minutes (`POLL_SLEEP` 4, 8 and 16, `GEMV_BATCH` 4, their pair, the
head at 8 and 10 events) each lost on one clock or both and nothing
entered the default; the half-merge fault (MIN-36) was located by halves
(one half faults at 2 layers, two halves run to 14 layers without the
head); the worker-timing hang (MIN-35) stayed in the buffer form, so the
printf reading is out and the problem is open. The VM was deleted at
minute 77 with $0.85 left; no session is planned. Before that, the
laptop preparation (PR #12): the `--final` preset (round 4's stack as one flag), the
compare made green by construction (the route log's tie rule with the
gate's 64 softmax weights in the reference, the iteration-aware
boundaries), the head's event count as a plan argument, the worker-timing
hang fixed in the patch (no device printf on the timing build), the
half-merge fault read and left to a 2-layer locator on the VM, the
merge's standalone 5 us read offline (the weights phase) and recorded;
the session plan (`05-session-plan.md`) and its results page
(`07-final-numbers.md`). Before that, round 4 ran its VM session
(`docs/gpu-experiments/04-kernels/09-session-log.md`, `10-results.md`;
branch `gpu/round-4`, 167 minutes, $8.22): the batch-1 GEMV linear for
qkva, o_proj and the head (`--gemv-linears --linear-grid 48`), the deeper
router (the header) and the merge as regular tasks at two halves per head
(`--merge-tasks --merge-halves 2`) take the token from round 3's 4,571 to
4,600 us to **4,262 to 4,341 us on the event clock** (`FWD_PASS` 4,267 to
4,310), ids equal on every final, 4 to 5% below vLLM's 4.5 ms. The
header's w2 GEMV form, w13 in one round, the four-task router and the
o_proj fold did not make their thresholds (round 3's CK w2 file is the
define's path again); the stream probe puts the megakernel's task-shaped
reads at 2.25 TB/s, the ceiling w13 sits on and the next round's lever
(bytes in flight by direct-to-LDS loads), then the boundary fusions
without a serial tail and the 192 us iteration start. Two hangs
(`--worker-timing` on the round-4 header; the inline CK w2 path) and a
fault (the half merge without the GEMV linears on 27 layers) are recorded
in `OPEN-PROBLEMS.md`. Round 3
(2026-09-17, `docs/gpu-experiments/03-acceleration/`) took the decode
from 9.58 to 4.57 to 4.60 ms per token, ids equal, and found the event
table off by one (`08-results.md`, `09-lessons.md`). Earlier: M0, M1, M2 and M3 reached on the MI300X
on 2026-09-15 (Hot Aisle, one VM, about $12): gate 1 passed after three
fixes, layer 1 validated end to end with all 16 boundaries and exact top-k,
27 layers run, the full model with the head produces the reference's first
two tokens. Round 2 (2026-09-16, one 1x MI300X, 149 minutes): M4 reached,
the 32 ids equal at 27 layers with the head; the fault of round 1 named (the
stock fused gate-up kernel reads 16 rows at batch 1) and fixed in the plan;
9.6 ms per token steady state with E2 and per-tile linears against the 1.15
to 1.35 ms design band, the rest sitting in the megakernel's per-task
overhead (MAJ-7); the image pushed. One page: `docs/gpu-experiments/02-validation/07-summary.md`.
The round was planned in `docs/gpu-experiments/02-validation/` (preparation on the laptop first,
then two sessions on a 1x MI300X for $27).
Earlier state: discovery complete (5 doc sets); the technical design is
written and independently reviewed (`docs/design-doc/`, 14 files, one
counting script); the local harness is complete and independently
reviewed on `local/harness` (prompt, reference run and capture,
comparison, weight packing, NumPy kernel specs, reassociation check,
graph builder, run and measurement scripts, environment scripts, the
gfx942 patch, the four kernels and their glue patch: 54 tests pass; the
review's 1 blocker, 3 major and 5 minor findings are all fixed).
GPU-dependent problems are parked, each with its check. On `local/gpu-ready`
(2026-09-14) the GPU code was compiled for gfx942 offline with ROCm 7.0's
hipcc in Docker (`env/offline_gfx942/`): every patched header parses and
our kernels compile and link, no VGPR spills, the cross-XCD fences lower as
designed; CK's source says its split-KV FMHA cannot take the MLA head
dim, so `mla_attend` is our spec kernel; the transformers conflict
between the checkpoint (4.46) and Fleet (4.57.1) is resolved by two
venvs; `env/preflight.sh` runs every local check; `11-day1-runbook.md`
is the session script. The day-1 question is now whether Fleet's host
library builds and a graph runs on the machine.

---

## Milestone ladder

- [x] **M0** Environment up, model downloaded, reference runs — 2026-09-15, Hot Aisle `enc1-gpuvm005` (`env/check_day1.log`, `harness/ref/`)
- [x] **M1** One validated operator through the Fleet path — 2026-09-15 (`env/hw/20260915/runs/L1_it1_L0.qkva`)
- [x] **M2** Layer 1 (MoE) validated end-to-end ← **required milestone** — 2026-09-15: all 16 boundaries PASS, top-k indices exact, route log PASS (`env/hw/20260915/runs/L2_it1`, B5 in `L2_it1_L1.mla_attend_scores`)
- [x] **M3** N consecutive persistent layers — the 27-layer graph without the head runs 32 iterations (1,822 tasks, 15.6 ms per iteration, `env/hw/20260915/runs/L27_it32`); with the head it produces the reference's first two tokens; layers 0 and 1 validated boundary by boundary, the growth curve of the rest pending
- [x] **M4** End-to-end 32-token decode — reached 2026-09-16 (`env/hw/20260916/runs/L27_head_it32_al65536`, then `runs/L27_head_it32` without any flag): the 27-layer graph with the head runs 32 iterations and the 32 ids equal the reference's. The fault of 2026-09-15 was the stock `gang_linear_silu_kernel` (the dense layer's fused gate-up, `L0.gate_up` by bisection) reading 16 rows of its `[1, 2048]` input at batch 1 (a CK tile GEMM with no active-token mask), 60 KB past a 4 KB buffer; whether that memory was mapped depended on the layout, which is why 8 layers faulted, 16 ran and a uniform shift changed nothing. Fixed in the plan: every single-row activation is backed by 16 rows (`fleet/build_graph.py`, `ROW_SLACK`); `--align-alloc 65536` was the first flag that passed. The story in `docs/gpu-experiments/02-validation/03-session-log.md` and `04-results.md`
- [ ] **M5** FP8 (stretch)

---

## Stage 1 — Discovery, **complete** (10/10)

- [x] Read task description
- [x] Init repo, branch, Fleet submodule, `.gitignore`
- [x] `docs/mi300x/` — architecture, dispatch, memory model, persistent kernel, stack, profiling
- [x] `docs/deepseek-v2-lite/` — config, MLA, MoE, tensor flow, weights, roofline, correctness
- [x] `docs/fleet/` — paper review, task model, runtime, repo map, sync cross-check, task graph, gaps
- [x] `docs/acceleration/` — precision, decode parallelism, kernel craft, ledger
- [x] Achievable bandwidth band + memory-level-parallelism analysis (`docs/mi300x/07`)
- [x] `docs/mla-decode/` — prior art survey + our MLA kernel spec (closes discovery)
- [x] Correctness pass on every doc set

| Doc set | Files | Lines | Covers |
|---|---|---|---|
| `mi300x/` | 9 | 1,388 | hardware, memory model, bandwidth, profiling |
| `deepseek-v2-lite/` | 10 | 1,278 | config, MLA, MoE, tensor flow, roofline, correctness |
| `fleet/` | 9 | 1,128 | paper, task model, runtime, repo map, our task graph |
| `acceleration/` | 6 | 575 | precision, parallelism, kernel craft, ledger |
| `mla-decode/` | 6 | 547 | prior art, kernel anatomy, **our kernel spec** |

**Key numbers**

| | |
|---|---|
| Traffic per token | 4,705.9 MiB (routed experts = 55%) |
| Roofline | 931 µs theoretical floor · **1.15–1.35 ms realistic** (742–871 tok/s) |
| Achievable bandwidth | 3.66–4.3 TB/s (69–81% of 5.3) |
| Layer-1 milestone | 31.6 µs floor / 38.9–45.7 µs realistic |
| FP8 (stretch) | 509 µs floor / 630–740 µs realistic — **1.83×** |
| Task graph | 326 ops, 1,880 tasks per token; **3 kernel dispatches per 32-token generation** |
| Eager baseline | ~800–1,000 launches/token |

**Decisions locked** (interim table; the authoritative list is `docs/design-doc/00-decisions.md`, which withdraws the VALU-for-weight-GEMVs row and restates `P_split`)

| Decision | Rationale |
|---|---|
| Target **gfx942**, SPX + NPS1 | Our hardware; default mode; gfx950 binaries will not run |
| **Extend** Fleet, don't reimplement | MoE, scheduler, sync already exist for MI300; only MLA is missing |
| **Runtime reassociation**, not materialized MLA fusion | Fusion costs +44 MiB/layer to save 8.9 — ~2× worse at S=1024; vLLM does the same |
| Latent KV cache, **split** `c_KV[S,512]` + `k_pe[S,64]` | Matches every implementation's `BLOCK_DMODEL`/`BLOCK_DPE` split |
| **`BLOCK_H`=16, `P_split`=32** | 16 heads share one KV read (MQA); parallelism from sequence splits only |
| **MFMA 16×16** for attention, VALU for weight GEMVs | M=16 fills the tile; M=1 wastes 15/16 |
| **Weight-only** FP8 if reached | Activation is 4 KB at batch 1 — activation scales buy nothing |
| Prefetch depth **4–8** in every inner loop | Required to saturate HBM at 1 wave/SIMD |

---

## Stage 2 — Design doc, **complete, reviewed** — `docs/design-doc/`

Required as the **first deliverable**.

- [x] Plan (`docs/design-doc/PLAN.md`)
- [x] P1 `persistent_kernel.cuh` scheduler/worker loops → `docs/fleet/03-runtime.md`
- [x] P2 `persistent_kernel.py` layer API + MoE demo conventions → `docs/fleet/04-repo-map.md`
- [x] P3 `gang_linear_mi300.cuh` + `ck_tile` idiom → `docs/fleet/04-repo-map.md`
- [x] `00-decisions.md` — 26 decisions with evidence and reversal conditions (supersedes the interim table below)
- [x] `01-execution-flow.md` + `sources/graph_counts.py`
- [x] `02-task-graph.md` — 326 ops / 1,880 tasks, 4 new kernels, 2 variants
- [x] `03-synchronization.md`
- [x] `04-memory-plan.md`
- [x] `05-prefill-interface.md`
- [x] `06-optimization-strategy.md`
- [x] `07-correctness.md`
- [x] `08-milestones.md`
- [x] `09-expected-performance.md`
- [x] `10-local-work.md`
- [x] `README.md`, `99-open-questions.md`
- [x] Independent review pass (17 findings, all confirmed against the source and fixed in `4d36f7a`)
- [x] Merged to `main` (PR #2, merge commit `8e19aa6`)

---

## Parked until GPU access is confirmed

Everything below needs the machine (a build, a disassembly, a counter, or a
timing) and is documented rather than resolved. Each entry names the check
that settles it; nothing here blocks the local work in Stage 3.

- Design-level: `docs/design-doc/99-open-questions.md` DQ1-DQ10 (DQ1 per-boundary latency and DQ3 CK FMHA at 576/512 first)
- Consolidated index, `gpu` or `build` in the When column: MAJ-1, MAJ-2, MAJ-3, MAJ-4, MAJ-5, MAJ-6, MIN-25, MIN-26, MIN-27, MIN-28, MIN-29, MIN-11, MIN-14, MIN-15, MIN-21, MIN-23, MIN-24, MIN-22, MIN-16, MIN-17, MIN-18, MIN-19, MIN-20 (`OPEN-PROBLEMS.md`)
- Day-1 order: `OPEN-PROBLEMS.md`, Triage; day-by-day: `docs/design-doc/08-milestones.md`

Resolvable now, without the GPU: MIN-1, MIN-2, MIN-4, MIN-5, MIN-6 (`OPEN-PROBLEMS.md`, When = local) and Stage 3 below.

---

## Stage 3 — Local work (no GPU), **complete, reviewed** (the GPU run of calibration and routing stays for day 1)

Branch `local/harness`. Every item has a check that runs here; the GPU-only
ones are written to be run on day 1 (`docs/design-doc/10-local-work.md`).
Test suite: `.venv/bin/python -m pytest harness/tests fleet/tests -q` (66 tests).

- [x] Read `gang_attention_merge_mi300.cuh` and `kv_cache_update_mi300.cuh` (both GQA-paged; merge math reusable, append is not)
- [x] Read `gang_linear_mi300.cuh` + `ck_tile` idiom → `docs/fleet/04-repo-map.md` (worker contract, inner GEMV tiers, CK FMHA path, gfx950-only code)
- [x] Read the Python layer API + `demo/qwen3/demo_30B_A3B.py` (the MoE template; the `models/qwen3/` builder has no MoE) → `docs/fleet/04-repo-map.md`
- [x] Read `persistent_kernel.cuh` main loop (launch structure, worker/scheduler loops, event counting, placement rules) → `docs/fleet/03-runtime.md`
- [x] Read the Mirage MPK paper (arXiv 2512.22219) → `docs/fleet/08-mpk-paper.md` (no per-boundary number; DQ1 stays a measurement)
- [x] Read vLLM / AITER / FlashMLA MLA decode kernels → `docs/mla-decode/`
- [x] Split-KV partial-softmax numerics (in `docs/mla-decode/04`)
- [x] **L1** prompt: `harness/prompt_ids.json` (BOS + 1,023 tokens of a pinned Fleet source file), `make_prompt.py` checks it
- [x] **L7** `harness/run_reference.py`: prefill 1,023, 32-step argmax loop from position 1023 with hooks, `generate` cross-check, cache capture; `--smoke` runs it on a tiny random model
- [x] **L9** `harness/compare.py`: metrics, thresholds, calibration override, exact checks, route log, growth curve, report
- [x] **L4** `fleet/pack_weights.py`: `W_qkva`, `W_uk`/`W_uv`, padded + shuffled layer-0 MLP, 66-expert `W13`/`W2`, loader
- [x] **L2** `harness/numpy_ref.py`: the four new kernels with explicit BF16 rounding, tested against the tiny model's own modules (RoPE, `ql_nope` bit-exact; router exact)
- [x] **L3** `harness/reassoc_check.py` → `harness/results/reassoc_check.json` (MIN-1 resolved: within the reference's own ordering noise)
- [x] **L5** `fleet/graph_plan.py` + `fleet/build_graph.py`: the 326-op / 1,880-task graph as data, the `mpk.*` calls, the new `*_layer` methods, `--dry-run` against a recording fake with the wrappers' assertions, `--stop-after <label>` truncation
- [x] **L6** `fleet/tasks/mi300/` (`mla_prep`, `mla_attend`, `mla_merge_uv`, `moe_router`, `copy`; VALU versions of the math of `numpy_ref.py`), the `embedding` and `argmax_reduce` variants and the eight-place glue as `fleet/patches/new_tasks.patch` (applies after `gfx942.patch`); host `clang++ -fsyntax-only` check passes; correctness on the GPU via `kernel_tests.py` (day 2)
- [x] **L12** `env/setup.sh`, `env/check_day1.sh`, `env/probe_ck_fmha_576_512.cpp`
- [x] **L13** `fleet/patches/gfx942.patch` (include guard, 16x16x16 warp GEMM on gfx942, `[FWD_PASS]` every iteration); applies cleanly
- [x] **L10** `harness/run_fleet.py` (build, compile, meta tensors, run, boundary dumps by last writer) · **L11** `harness/measure.py` (`[FWD_PASS]`, event timing, rocprofv3 CSVs, the report table)
- [x] **L8** `harness/calibrate.py` (script; the floor itself needs the GPU) · **L14** `harness/route_analysis.py`
- [x] Calibrate BF16 noise floor and log expert routing on the machine: `run_reference.py`, then `calibrate.py` and `route_analysis.py` (MIN-2, MIN-6) — floors router 3.59e-3, layer 4.33e-3, route overlap 0.21, on both VMs (2026-09-15, 2026-09-16)
- [x] Independent review of the branch against the Fleet source, the HF modeling file and the design docs: 1 blocker (`AMDGPU_TARGETS` unset, so the megakernel compiled for gfx950), 3 major (layer-1 B3 normalized with layer 0's weight; the B5 debug-scores path unreachable; B10 compared element-wise against an unordered `topk`), 5 minor; all fixed in separate commits, MIN-30 and MIN-31 closed by source inspection
- [x] `fleet/tasks/kernel_tests.py` + `kernel_tests_mi300.cu` (07-correctness.md harness table): a standalone HIP launcher that runs each new kernel in isolation on random inputs against `numpy_ref.py`, plus 1 split versus 33 splits; `--dry-run` exercises the plumbing here (12 tests, including a Python-to-C++ contract test parsed from the launcher), the launcher compiles and links for gfx942 offline in both variants; the run itself is day 2

**GPU-readiness pass (`local/gpu-ready`, 2026-09-14):**

- [x] Dependency conflict found and resolved: Fleet's `install_requires` pins transformers 4.57.1, under which the checkpoint's modeling code fails (verified: 17 tests); `env/setup.sh` makes `.venv` (4.46.3) and `.venv-fleet` (Fleet's pins, `env/requirements-fleet.txt`)
- [x] Dependency list corrected: at `51dce4f` no `.gitmodules` entry is a gitlink; CK (`d8ee107a`, the commit Fleet's other branches pin) and nlohmann/json are fetched by commit into `deps/`; PyTorch index fallback `rocm7.0`; ROCm 7.0+ recorded as Fleet's requirement
- [x] Offline gfx942 compile with ROCm 7.0's hipcc in Docker (`env/offline_gfx942/`): three variants compile and device-link, zero errors; worker kernel 182 VGPRs, no VGPR spills; `buffer_wbl2 sc1` / `buffer_inv sc1` present (MAJ-1 compile half, MIN-27, MAJ-3 lowering, MAJ-4 static union)
- [x] DQ3 / MIN-28 answered from CK's source: `static_assert(kSubQKHeaddim <= 256)` in both split-KV pipelines; D12 reversed, `mla_attend` is the spec kernel
- [x] `env/preflight.sh`: tests, prompt check, graph dry run, kernel syntax, script parse, submodule pin, both patches on a clean worktree (8 PASS); `OFFLINE_COMPILE=1` adds the Docker compile
- [x] `docs/design-doc/11-day1-runbook.md`: command, PASS line, time box, action on failure, per session
- [x] Independent review of the branch (see the PR)

**Found on the way:** the checkpoint's remote modeling code needs transformers 4.x (pinned 4.46.3 in `env/requirements.txt`); eager attention asserts on a missing mask at a one-token step; the model returns a legacy tuple cache unless a `DynamicCache` is passed; the reference rounds its attention scores to BF16 before the softmax, which sets the attention-output floor at large score magnitudes (`docs/design-doc/07-correctness.md`, item 7); the shipped `gang_linear_silu` wrapper expects 128-row gate/up groups (`num_groups = 88` for the padded width); `attach_input` asserts row-major, so `W_uk`/`W_uv` are contiguous copies; the online-mode stop test is `step + 2 >= max_seq_length` on the pre-increment step, so a K-iteration run uses `max_seq_length = 1024 + K`.

---

## Stage 4 — GPU bring-up, complete — hardware collection 2026-09-15, round 2 on the 1x MI300X 2026-09-16 (`docs/gpu-experiments/02-validation/`)

The hour of measurements that precedes the build ran on Hot Aisle VM
`enc1-gpuvm005` (ROCm 7.2.4, hipcc 7.2.53211, two MI300X VF devices, device 0
used). Record and readings: `env/hw/20260915/`, summarised in
`docs/gpu-experiments/01-bringup/README.md`.

**Day 1, in order — each gates the next:**

- [x] **Does the repo build for gfx942?** Yes (2026-09-15): `env/setup.sh` builds and `import mirage` works once the isolated build environment's z3 is pinned to the venv's and `z3/lib` is on the loader path; the Qwen3 smoke graph and our graphs run through the persistent kernel (`env/check_day1.log`, gate 1 PASS)
- [x] Capture `rocminfo`, `hipcc --version`, ROCm version, partition mode — ROCm 7.2.4, hipcc 7.2.53211, SPX + NPS1, 304 CUs over 8 XCDs, 64 KiB LDS, 4 MiB L2 per XCD, 192 GB HBM, no L3 row (`docs/mi300x` Q9)
- [x] Assert SPX + NPS1; find `amd-smi` query/set syntax — the query that works is `amd-smi static --partition`; the set syntax was not exercised, the VM was already in the mode we want (`docs/mi300x` Q3)
- [x] Confirm `XCC_ID` returns 0–7; map workgroup → XCD — all values in 0–7, all eight present; the mapping is `xcd == (blockIdx.x + 4) mod 8`, balanced and stable across launches and processes, which is **not** what the runtime assumed: fixed by `fleet/patches/sched_xcd.patch` (`docs/mi300x` Q1, `docs/fleet` Q10, MIN-25)
- [x] Disassemble agent-scope fence: does `buffer_wbl2 sc1` / `buffer_inv sc1` appear? Yes, offline (`env/offline_gfx942/fences.txt`) and again on the VM's own hipcc 7.2 with no `sc0 sc1` anywhere in the four probe kernels, at 115–317 ns per fence; left: whether a system-scope fence sits on the per-task path of the generated kernel (`docs/mi300x` Q4, Q5)
- [x] Verify 38 CUs/XCD detected (repo constants are MI350's 32) — the runtime reports 296 workers and 8 schedulers, 37 workers on every XCD (`[SCHED_XCD]` lines, `env/check_day1.log`)
- [x] Download model (31 GB) — 30 GB in 79 s on the VM
- [x] `rocprofv3 --list-avail` → confirm counter names (`TCC_EA0_*`) — names confirmed, PMC collection works inside the VF, 128 TCC instances (16 channels x 8 XCC) (`docs/mi300x` Q11)
- [x] Validate bytes-from-requests against a known-size copy kernel — exact on a 1 GiB copy, and it showed `docs/mi300x/06-profiling.md`'s read formula undercounts by 2x for want of a 128 B term (`docs/mi300x` Q12, MIN-32)

**Also settled by the collection:** one wave per SIMD at the worker kernel's
LDS footprint with all 304 blocks co-resident (`docs/mi300x` Q6, MAJ-4);
streaming read 3.943 TB/s with the knee at prefetch depth 4 and BabelStream
Triad 4,133,844 MB/s, so the 3.66–4.3 TB/s band holds (Q14, MIN-22);
pointer-chase latency 81 ns in L2, 258 ns at 64 MiB, 342 ns at HBM, so the
memory-side cache tier is real (Q13, MIN-21, MAJ-6); a 703 ns cross-XCD
one-way hop, the first number under `t_b` (DQ1, MAJ-5).

**Fallback if the build fails:** minimal own persistent kernel, M2 only, runtime
documented as blocked. **Decide end of day 1.**

---

## Stage 5 — Implementation, M4 reached on the machine (2026-09-16); M2 since 2026-09-15

- [x] Weight loader: pack experts into W13 `[64, 2816, 2048]`, precision as a parameter — `fleet/pack_weights.py`, 272 tensors, 31.42 GB, checks pass on the VM
- [x] Prefill → latent KV cache conversion (excluded from timed window) — captured by `run_reference.py`, consumed by `run_fleet.py`
- [x] **MLA decode Chiplet-task** — `mla_attend` validated: kernel tests 100/100, B6 attention rel 7e-3 to 8e-3 (threshold 0.040) at layers 0 and 1, B5 scores rel 5.4e-3
- [x] Split-KV merge for MLA — inside `mla_attend` (33 splits), tested against 1 split and the NumPy merge
- [x] Latent KV-cache buffer + append task — `mla_prep`; B3 c_kv and k_pe within threshold at both layers
- [x] DeepSeek-V2 model builder — `fleet/graph_plan.py` and `fleet/build_graph.py` (the graph is built in our harness, not in `python/mirage/mpk/models/`)
- [x] Chiplet-parallel `lm_head` + argmax reduce — runs (a 2-layer graph with the head produces a token); correct only with all 27 layers, see M4
- [x] Wire layer 1 task graph → **M2** — PASS
- [x] Extend to N layers → **M3** — 27 layers run 32 iterations; the growth curve over 27 layers measured (`env/hw/20260916/runs/L27_it1_al65536`): 3.7e-3 to 6.4e-3 at layers 0 to 4, a jump to 2.97e-2 at layer 5 from a step-0 routing tie (expert 49 against 2), then a monotone decay to 7.1e-3; FAIL by the 4x-floor rule at layers 5 to 7, explained (`docs/gpu-experiments/02-validation/04-results.md`)
- [x] End-to-end decode → **M4** — 32 ids equal at 27 layers with the head (2026-09-16)

---

## Stage 6 — Measurement & submission: measured through round 5 (2026-09-18), the counters missing

- [x] Correctness evidence at every completed boundary — layers 0 and 1 at every boundary against the reference; the 32 token ids exact at 27 layers with the head; the expert indices exact at step 0 except one tie, 60 of 832 (step, layer) sets differ by one expert near a tie over 32 steps (`docs/gpu-experiments/02-validation/04-results.md`)
- [x] GPU launches per token — one `mpk()` call runs the 32 iterations (the persistent kernel); the profiler count was not taken (rocprofv3 cannot attach to the torch wheel)
- [x] Median + P95 latency — N = 32 iterations, the runtime's event clock: round 5 (2026-09-18, the final numbers) 4,284 to 4,291 us median over the three finals of the stack, 4,265 by the megakernel's own report (`docs/gpu-experiments/05-final/07-final-numbers.md`); round 4 (2026-09-18) 4,262 to 4,341 (`docs/gpu-experiments/04-kernels/10-results.md`); round 3 (2026-09-17) 4,571 to 4,600 us median over seven final runs, 4,584 to 4,590 by the megakernel's own report with P95 4,634 to 4,649 (`docs/gpu-experiments/03-acceleration/08-results.md`); round 2: 9,575 median / 9,644 P95 with E2 and per-tile linears, 12,268 / 12,338 with the gang linears (`docs/gpu-experiments/02-validation/04-results.md`)
- [ ] Memory traffic, achieved bandwidth, L2 hit rate — not measured: rocprofv3 aborts on the torch wheel's bundled runtime; 0.52 TB/s from the design's byte count and the measured time; the counters need a standalone binary (`docs/gpu-experiments/02-validation/06-lessons.md`, item 7)
- [x] Occupancy + VGPR/LDS per task — from the offline compile: round 4 (2026-09-18) the worker's union with every round-4 task 256 VGPRs, 171 AGPRs, 8 spills, one workgroup per CU; standalone the GEMV linear 248 VGPRs and 32 AGPRs, w13 248, the w2 form 248, the o_proj fold 228, the router 190, the merge 134 (`docs/gpu-experiments/04-kernels/06-checklist.md`, L1c); round 3: the union 253 VGPRs and 32 AGPRs with the MFMA attention (234 in round 2), no spills; the MFMA attention 140 VGPRs and 32 AGPRs standalone, the VALU one 124, `mla_merge_uv` 75 (`docs/gpu-experiments/03-acceleration/04-checklist.md`, O7; `../02-validation/01-preparation.md`, P6)
- [x] TPOT + tokens/s — on the 1x MI300X, steady state on the runtime's event clock: round 5, the final numbers, 4.28 to 4.29 ms per token (233 tokens/s) with the ids equal and every compare row green, 4.6 to 4.8% below the 4.5 ms production baseline; round 4 4.26 to 4.34 ms; round 3 4.57 to 4.60 ms per token (217 to 219 tokens/s) with the per-head prep, the MFMA attention, the batched router, merge and norm loads, the fusions, per-tile linears and E2, against the 4.5 ms production baseline; round 2: 12.3 ms with the gang linears, 9.6 ms with E2 and per-tile linears (104 tokens/s); the design band is 1.15 to 1.35 ms (`docs/gpu-experiments/03-acceleration/08-results.md`, `../02-validation/04-results.md`)
- [x] Fleet-native ops vs remaining fallbacks — the stock gang and per-tile linears, norms, embed, argmax; ours: `mla_prep`, `mla_attend`, `mla_merge_uv`, `moe_router`, `copy`, and behind round 4's flags the GEMV linear (qkva, o_proj, the head), the w2 and w13 GEMV forms, the four-task router, the regular merge and the merge with o_proj folded in, the stream probe (`fleet/tasks/README.md`); layer 0's dense down and the elementwise ops stay stock; no host fallback in the timed window
- [x] Build/run instructions, setup scripts, profiling commands — `env/setup.sh`, `env/session/` (the VM stages, the queue, the laptop driver), `docs/gpu-experiments/01-bringup/06-agent-guide.md`, `docs/gpu-experiments/02-validation/02-session-plan.md`
- [x] Known failures + recommended next steps — `OPEN-PROBLEMS.md`; `docs/gpu-experiments/03-acceleration/09-lessons.md` (every approach with its verdict, the lessons, the next round ranked: the per-XCD completion hierarchy first); `../02-validation/06-lessons.md`; FP8 with arithmetic in `docs/acceleration/`
- [x] Round 3, the acceleration toward the 4.5 ms production baseline (2026-09-17, `docs/gpu-experiments/03-acceleration/`, branch `gpu/round-3`): 9.58 to 4.57 to 4.60 ms per token on the event clock, ids equal (`08-results.md`); the session found the event table's names off by one, the prep task at 4 ms of the 8.9, and four latency-bound kernels (`07-session-log.md`). As prepared: the fused norms and silu (326 to 246 operators), the MFMA attention, the weight prefetch by side operators, the streaming loads, the probe; the per-worker timing, the shader-clock spin, the empty-task ladder, the fence knobs and the clock sampler to attribute the time; all as off-by-default flags, checked on the laptop, the VM session planned (`05-session-plan.md`) and rehearsed; the run and its numbers are the open box
- [x] Round 5, the final stage (prepared 2026-09-18 on `local/round-5`, run the same day on `gpu/round-5`; `docs/gpu-experiments/05-final/`): **4,284 to 4,291 us per token, `FWD_PASS` 4,265, the ids equal and every compare row green (`07-final-numbers.md`)**; the round-4 stack as `run_fleet.py --final`; the compare made green by construction (the route log's tie rule with the gate's softmax weights in the reference, zero disagreements over 30 steps; the iteration-aware boundaries); the knobs and constants (`POLL_SLEEP` 4, 8, 16, `GEMV_BATCH` 4, their pair, the head at 8 and 10 events) each lost on one clock or both, nothing entered the default; MIN-36 located by halves, MIN-35 open in the buffer form; 77 minutes, $3.74, the balance at $0.85 (`08-session-log.md`)
- [x] Round 4, the kernels (prepared 2026-09-18 on `local/round-4`, run the same day on `gpu/round-4`; `docs/gpu-experiments/04-kernels/`): **4,262 to 4,341 us per token, ids equal (`10-results.md`)**; the GEMV linears (831 to 638 us per token over the per-tile linears), the deeper router (516 to 424), the half merge (489 to 406); the w2 GEMV form, w13 in one round, the four-task router and the o_proj fold measured and off; the stream ceiling 2.25 TB/s; the preparation: the ideas for a batch-1 GEMV linear (the CK tile keeps 32 KB per CU in flight, ours 128) and for the router and the merge, double-checked; seven kernels behind flags (`--gemv-linears` with `--linear-grid` and `--head-grid`, `--gemv-w13`, the w2 GEMV form as the fused-silu default with `-DMPK_W2_CK_TILE` as the fallback, `--router-tasks`, `--merge-tasks --merge-halves 2`, `--merge-oproj`, `--graph stream`), each with its suite rows, offline variant and registration; the bit-diff of two runs' boundaries; four compiler conventions found offline (no batch live across a prologue, clamp rather than guard, a heavy body as a call, a call's arguments made wave-uniform); the queue files, the stages, the plan with its thresholds against round 3's numbers and the rehearsal (`05` to `08`); the session log, results and lessons in `09` to `11`

---

## Open questions

| Where | Count | Most urgent |
|---|---|---|
| `docs/mi300x/99-open-questions.md` | 13 open / 1 resolved | Q4 agent-scope fence emits right cache ops |
| `docs/deepseek-v2-lite/99-open-questions.md` | 4 open / 5 resolved | Q6 calibrate the floor (day 1) |
| `docs/fleet/99-open-questions.md` | 10 open / 3 resolved | **Q1 does it build on gfx942**, Q11 CK FMHA at 576/512 |
| `docs/acceleration/99-open-questions.md` | 4 open / 2 resolved | Q2 MFMA vs VALU at M=1 |
| `docs/mla-decode/99-open-questions.md` | 5 open | Q1 is `P_split`=32 right |
| `docs/design-doc/99-open-questions.md` | 10 open (3 restate `docs/fleet` Q3, Q4, Q11) | **DQ1 per-boundary latency** |
| **total** | **46 open / 11 resolved** | |

`OPEN-PROBLEMS.md` holds the consolidated, deduplicated view: **5 major open / 15 minor open / 31 resolved**, plus 9 documentation defects found in AMD and
Fleet sources.

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Repo won't build for gfx942 | Strategy change | Device code compiles offline (2026-09-14); the host build is decided day 1; minimal-runtime fallback |
| MLA task is the whole budget | Miss M3/M4 | M4 reached 2026-09-16; the MLA kernels validated and 34 us standalone |
| Megakernel occupancy = 1 wave/SIMD | Was feared fatal | resolved — `VMCNT`=63 allows enough in-flight loads; becomes a prefetch-depth requirement (MIN-22 to confirm) |
| Attention uses 32 of 296 workers | 13–17% of budget if the model is right | measured: 33 splits run as 40 workgroups, one tile per worker; 61 splits (`--split 17`) change nothing, the cost is per tile (`docs/gpu-experiments/02-validation/06-lessons.md`) |
| Our tasks smaller than anything Fleet measured | Dispatch overhead dominates | measured 2026-09-16: the attention kernel is 34 us standalone and 146 to 215 us in the graph; the megakernel's per-task overhead is the floor (MAJ-7, `docs/gpu-experiments/02-validation/06-lessons.md` item 1) |
| Top-6 experts over 8 XCDs leaves 2 idle | 25% of machine during 99 MB phase | Candidate: fold shared experts in as experts 64-65 (8 active on 8 XCDs); compare vs N-split |
| 5 days, BF16 first | FP8 not reached | Document with arithmetic; precision as a loader parameter |

---

## Not in scope

Prefill optimization · tensor parallelism · continuous batching · speculative
decoding · multi-GPU · cooperative weight tiling (needs batch > 1)
