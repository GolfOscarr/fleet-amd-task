# Progress

Fleet-style batch-1 decode for DeepSeek-Coder-V2-Lite-Base on one AMD MI300X.
Time limit: 5 days. Target: gfx942, BF16, 1024-token prompt, 32 greedy tokens.

Last updated: 2026-09-14 · branch `main` at the merge of PR #3 (Stage 3 complete and reviewed)

**Where we are:** discovery complete (5 doc sets); the technical design is
written and independently reviewed (`docs/design-doc/`, 14 files, one
counting script); the local harness is complete and independently
reviewed on `local/harness` (prompt, reference run and capture,
comparison, weight packing, NumPy kernel specs, reassociation check,
graph builder, run and measurement scripts, environment scripts, the
gfx942 patch, the four kernels and their glue patch: 54 tests pass; the
review's 1 blocker, 3 major and 5 minor findings are all fixed).
GPU-dependent problems are parked, each with its check. Nothing built for
the GPU yet; the day-1 blocker is whether Fleet builds for gfx942.

---

## Milestone ladder

- [ ] **M0** Environment up, model downloaded, reference runs
- [ ] **M1** One validated operator through the Fleet path
- [ ] **M2** Layer 1 (MoE) validated end-to-end ← **required milestone**
- [ ] **M3** N consecutive persistent layers
- [ ] **M4** End-to-end 32-token decode
- [ ] **M5** FP8 (stretch)

---

## Stage 1 — Discovery ✅ **complete** (10/10)

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

## Stage 2 — Design doc ✅ **complete, reviewed** — `docs/design-doc/`

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

## Stage 3 — Local work (no GPU) ✅ **complete, reviewed** (the GPU run of calibration and routing stays for day 1)

Branch `local/harness`. Every item has a check that runs here; the GPU-only
ones are written to be run on day 1 (`docs/design-doc/10-local-work.md`).
Test suite: `.venv/bin/python -m pytest harness/tests fleet/tests -q` (54 tests).

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
- [ ] Calibrate BF16 noise floor and log expert routing on the machine: `run_reference.py`, then `calibrate.py` and `route_analysis.py` (MIN-2, MIN-6)
- [x] Independent review of the branch against the Fleet source, the HF modeling file and the design docs: 1 blocker (`AMDGPU_TARGETS` unset, so the megakernel compiled for gfx950), 3 major (layer-1 B3 normalized with layer 0's weight; the B5 debug-scores path unreachable; B10 compared element-wise against an unordered `topk`), 5 minor; all fixed in separate commits, MIN-30 and MIN-31 closed by source inspection
- [ ] `kernel_tests.py` (07-correctness.md harness table): a standalone HIP launcher that runs each new kernel in isolation on random inputs against `numpy_ref.py`, plus 1 split versus 33 splits. Not in L1-L14; writable locally (syntax check only), runs on day 2. The one local gap left.

**Found on the way:** the checkpoint's remote modeling code needs transformers 4.x (pinned 4.46.3 in `env/requirements.txt`); eager attention asserts on a missing mask at a one-token step; the model returns a legacy tuple cache unless a `DynamicCache` is passed; the reference rounds its attention scores to BF16 before the softmax, which sets the attention-output floor at large score magnitudes (`docs/design-doc/07-correctness.md`, item 7); the shipped `gang_linear_silu` wrapper expects 128-row gate/up groups (`num_groups = 88` for the padded width); `attach_input` asserts row-major, so `W_uk`/`W_uv` are contiguous copies; the online-mode stop test is `step + 2 >= max_seq_length` on the pre-increment step, so a K-iteration run uses `max_seq_length = 1024 + K`.

---

## Stage 4 — GPU bring-up ⬜ blocked on hardware

**Day 1, in order — each gates the next:**

- [ ] **Does the repo build for gfx942?** `AMDGPU_TARGETS=gfx942 pip install -e .` ← **BLOCKING**
- [ ] Capture `rocminfo`, `hipcc --version`, ROCm version, partition mode
- [ ] Assert SPX + NPS1; find `amd-smi` query/set syntax
- [ ] Confirm `XCC_ID` returns 0–7; map workgroup → XCD
- [ ] Disassemble agent-scope fence: does `buffer_wbl2 sc1` / `buffer_inv sc1` appear?
- [ ] Verify 38 CUs/XCD detected (repo constants are MI350's 32)
- [ ] Download model (31 GB)
- [ ] `rocprofv3 --list-avail` → confirm counter names (`TCC_EA0_*`)
- [ ] Validate bytes-from-requests against a known-size copy kernel

**Fallback if the build fails:** minimal own persistent kernel, M2 only, runtime
documented as blocked. **Decide end of day 1.**

---

## Stage 5 — Implementation ⬜ not started

- [ ] Weight loader: pack experts into W13 `[64, 2816, 2048]`, precision as a parameter
- [ ] Prefill → latent KV cache conversion (excluded from timed window)
- [ ] **MLA decode Chiplet-task** ← the core work; **spec ready** in `docs/mla-decode/04-our-kernel-spec.md`
- [ ] Split-KV merge for MLA
- [ ] Latent KV-cache buffer + append task
- [ ] DeepSeek-V2 model builder in `python/mirage/mpk/models/`
- [ ] Chiplet-parallel `lm_head` + argmax reduce
- [ ] Wire layer 1 task graph → **M2**
- [ ] Extend to N layers → **M3**
- [ ] End-to-end decode → **M4**

---

## Stage 6 — Measurement & submission ⬜ not started

- [ ] Correctness evidence at every completed boundary (expert indices exact, 32 token IDs exact)
- [ ] GPU launches per token
- [ ] Median + P95 latency (state N; 32 tokens is too few — loop the decode)
- [ ] Memory traffic, achieved bandwidth, L2 hit rate
- [ ] Occupancy + VGPR/LDS per task
- [ ] TPOT + tokens/s if M4 reached
- [ ] Fleet-native ops vs remaining fallbacks
- [ ] Build/run instructions, setup scripts, profiling commands
- [ ] Known failures + recommended next steps (incl. FP8 with arithmetic)

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

`OPEN-PROBLEMS.md` holds the consolidated, deduplicated view: **6 major /
20 minor open / 20 resolved**, plus 9 documentation defects found in AMD and
Fleet sources.

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Repo won't build for gfx942 | Strategy change | Decide day 1; minimal-runtime fallback |
| MLA task is the whole budget | Miss M3/M4 | ✅ prior art read, spec drafted; M2 is the required bar |
| Megakernel occupancy = 1 wave/SIMD | Was feared fatal | ✅ resolved — `VMCNT`=63 allows enough in-flight loads; becomes a prefetch-depth requirement (MIN-22 to confirm) |
| Attention uses 32 of 296 workers | 13–17% of budget if the model is right | `P_split` is one constant to sweep (MIN-23) |
| Our tasks smaller than anything Fleet measured | Dispatch overhead dominates | Measure task vs dispatch time early |
| Top-6 experts over 8 XCDs leaves 2 idle | 25% of machine during 99 MB phase | Candidate: fold shared experts in as experts 64-65 (8 active on 8 XCDs); compare vs N-split |
| 5 days, BF16 first | FP8 not reached | Document with arithmetic; precision as a loader parameter |

---

## Not in scope

Prefill optimization · tensor parallelism · continuous batching · speculative
decoding · multi-GPU · cooperative weight tiling (needs batch > 1)
