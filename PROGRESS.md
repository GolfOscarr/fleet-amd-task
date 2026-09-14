# Progress

Fleet-style batch-1 decode for DeepSeek-Coder-V2-Lite-Base on one AMD MI300X.
Time limit: 5 days. Target: gfx942, BF16, 1024-token prompt, 32 greedy tokens.

Last updated: 2026-09-14 · branch `research/references-and-planning` · 14 commits

**Where we are:** discovery complete (5 doc sets, 40 files, ~5,300 lines, 9
reproducible scripts). Nothing built yet. Next deliverable is the technical
design document; the day-1 blocker is whether Fleet builds for gfx942.

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
| Task graph | 80 tasks/layer, ~2,135/token, **1 kernel launch** |
| Eager baseline | ~800–1,000 launches/token |

**Decisions locked** (to be consolidated into `docs/decisions.md`)

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

## Stage 2 — Design doc ⬜ not started — **next**

Required as the **first deliverable**. All inputs exist; this is assembly.

- [ ] Model execution flow
- [ ] Fleet task graph (draft in `docs/fleet/06-our-task-graph.md`)
- [ ] Synchronization strategy
- [ ] Memory plan + KV-cache layout
- [ ] Prefill → Fleet-decode interface
- [ ] Correctness methodology (draft in `docs/deepseek-v2-lite/08-correctness.md`)
- [ ] Implementation milestones
- [ ] Expected performance
- [ ] Local-vs-GPU work split
- [ ] `docs/decisions.md` — consolidate SPX+NPS1, runtime reassociation, extend-not-reimplement, gfx942, weight-only FP8

---

## Stage 3 — Local work (no GPU) 🟡 **3/12**

- [x] Read `gang_attention_merge_mi300.cuh` and `kv_cache_update_mi300.cuh` (both GQA-paged; merge math reusable, append is not)
- [ ] Read `gang_linear_mi300.cuh` + `ck_tile` idiom ← the template our MLA task is written against
- [ ] Read `python/mirage/mpk/models/qwen3/` (template for our builder)
- [ ] Read Mirage MPK paper (arXiv:2512.22219) + `persistent_kernel.cuh` main loop
- [x] Read vLLM / AITER / FlashMLA MLA decode kernels → `docs/mla-decode/`
- [x] Split-KV partial-softmax numerics (in `docs/mla-decode/04`)
- [ ] Pick + tokenize the 1,024-token prompt, commit token IDs (recipe in `docs/deepseek-v2-lite` Q8)
- [ ] HF reference: greedy decode (`do_sample=False`), dump per-boundary tensors
- [ ] Calibrate BF16 noise floor → set correctness thresholds
- [ ] Log expert routing across 32 steps (feeds placement decisions)
- [ ] Verify runtime reassociation numerically on CPU
- [ ] Environment setup scripts

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
| `docs/deepseek-v2-lite/99-open-questions.md` | 6 open / 3 resolved | Q1 reassociation within tolerance (no GPU) |
| `docs/fleet/99-open-questions.md` | 7 open / 2 resolved | **Q1 does it build on gfx942** |
| `docs/acceleration/99-open-questions.md` | 4 open / 2 resolved | Q2 MFMA vs VALU at M=1 |
| `docs/mla-decode/99-open-questions.md` | 5 open | Q1 is `P_split`=32 right |
| **total** | **35 open / 8 resolved** | |

`OPEN-PROBLEMS.md` holds the consolidated, deduplicated view: **6 major /
18 minor open / 14 resolved**, plus 9 documentation defects found in AMD and
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
| Top-6 experts over 8 XCDs leaves 2 idle | 25% of machine during 99 MB phase | Compare vs N-split across all 8 |
| 5 days, BF16 first | FP8 not reached | Document with arithmetic; precision as a loader parameter |

---

## Not in scope

Prefill optimization · tensor parallelism · continuous batching · speculative
decoding · multi-GPU · cooperative weight tiling (needs batch > 1)
