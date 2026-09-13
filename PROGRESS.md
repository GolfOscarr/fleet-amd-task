# Progress

Fleet-style batch-1 decode for DeepSeek-Coder-V2-Lite-Base on one AMD MI300X.
Time limit: 5 days. Target: gfx942, BF16, 1024-token prompt, 32 greedy tokens.

Last updated: 2026-09-14 · branch `research/references-and-planning`

---

## Milestone ladder

- [ ] **M0** Environment up, model downloaded, reference runs
- [ ] **M1** One validated operator through the Fleet path
- [ ] **M2** Layer 1 (MoE) validated end-to-end ← **required milestone**
- [ ] **M3** N consecutive persistent layers
- [ ] **M4** End-to-end 32-token decode
- [ ] **M5** FP8 (stretch)

---

## Stage 1 — Discovery ✅ complete

- [x] Read task description
- [x] Init repo, branch, Fleet submodule, `.gitignore`
- [x] `docs/mi300x/` — architecture, dispatch, memory model, persistent kernel, stack, profiling
- [x] `docs/deepseek-v2-lite/` — config, MLA, MoE, tensor flow, weights, roofline, correctness
- [x] `docs/fleet/` — paper review, task model, runtime, repo map, sync cross-check, task graph, gaps
- [x] `docs/acceleration/` — precision, decode parallelism, kernel craft, ledger
- [x] Correctness pass on every doc set

**Key numbers:** roofline 931 µs/token (1,074 tok/s) · 4,705.9 MiB/token ·
routed experts = 55% of traffic · layer 1 roofline 31.6 µs · FP8 → 508 µs

---

## Stage 2 — Design doc ⬜ not started

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

## Stage 3 — Local work (no GPU) ⬜ not started

- [ ] Read `gang_linear_mi300.cuh` + `ck_tile` idiom
- [ ] Read `python/mirage/mpk/models/qwen3/` (template for our builder)
- [ ] Read Mirage MPK paper (arXiv:2512.22219) + `persistent_kernel.cuh` main loop
- [ ] Read vLLM / SGLang / AITER MLA decode kernels ← de-risks the core work
- [ ] Split-KV partial-softmax numerics
- [ ] Pick + tokenize the 1,024-token prompt, commit token IDs
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
- [ ] **MLA decode Chiplet-task** ← the core work, no prior art in the repo
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
| `docs/mi300x/99-open-questions.md` | 13 | Q4 agent-scope fence emits right cache ops |
| `docs/deepseek-v2-lite/99-open-questions.md` | 9 | Q1 reassociation within tolerance (no GPU) |
| `docs/fleet/99-open-questions.md` | 9 | **Q1 does it build on gfx942** |
| `docs/acceleration/99-open-questions.md` | 6 | Q2 MFMA vs VALU at M=1 |

---

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Repo won't build for gfx942 | Strategy change | Decide day 1; minimal-runtime fallback |
| MLA task is the whole budget | Miss M3/M4 | Read prior art first; M2 is the required bar |
| Megakernel occupancy = 1 wave/SIMD | No latency hiding on a memory-bound load | Track VGPRs from commit 1 |
| Our tasks smaller than anything Fleet measured | Dispatch overhead dominates | Measure task vs dispatch time early |
| Top-6 experts over 8 XCDs leaves 2 idle | 25% of machine during 99 MB phase | Compare vs N-split across all 8 |
| 5 days, BF16 first | FP8 not reached | Document with arithmetic; precision as a loader parameter |

---

## Not in scope

Prefill optimization · tensor parallelism · continuous batching · speculative
decoding · multi-GPU · cooperative weight tiling (needs batch > 1)
