# Open Problems

Consolidated index of everything unresolved, so a problem can be located without
re-reading four doc sets. Detail lives in each set's `99-open-questions.md`.

Last updated: 2026-09-14 (offline gfx942 compile) · 6 major (three narrowed) · **18 minor open** · 22 resolved · 9 documentation defects

**When** — `local` = resolvable without a GPU · `gpu` = needs the MI300X ·
`build` = needs a working toolchain

---

## Major — blocks work or changes the design

| ID | Problem | Area | Where | When |
|---|---|---|---|---|
| **MAJ-1** | **Does the Fleet repo build and run on gfx942?** Narrowed offline (2026-09-14): with `gfx942.patch` every megakernel header parses for gfx942 under ROCm 7.0's hipcc (the CK linear templates included, though not instantiated) and our five kernels compile and device-link (`env/offline_gfx942/README.md`). Left for the machine: the cmake and cargo halves of `pip install -e .` and running a graph. Gates extend-vs-reimplement. | Build / runtime | `docs/fleet` Q1, Q12 | build |
| **MAJ-2** | **No MLA kernel exists.** Repo has GQA paged-attention only, and AITER's gfx942 path is hand-written ASM. Spec drafted in `docs/mla-decode/04-our-kernel-spec.md`. **Possible shortcut (MIN-28):** the repo already instantiates CK's split-KV FMHA pipeline with separate QK/V head dims; if the machine's CK has the 576/512 configuration, phase B is an instantiation plus a wrapper. | Kernel | `docs/mla-decode` · `docs/fleet` §07 G1, Q11 | local then gpu |
| **MAJ-3** | **Does the agent-scope fence emit `buffer_wbl2 sc1` / `buffer_inv sc1`?** Yes in the offline gfx942 assembly (2026-09-14): the worker kernel carries 4 `buffer_wbl2 sc1` and 2 `buffer_inv sc1` sites (`env/offline_gfx942/fences.txt`). What remains is a cost question: the same kernel also carries 72 system-scope `buffer_wbl2 sc0 sc1` and 48 `buffer_inv sc0 sc1` sites from the printf and assert hostcall paths and `__threadfence()`; whether any lies on the per-task path is read off the generated `kernel_0.cu` on day 2. | Memory model | `docs/mi300x` Q4 · `docs/fleet` Q6 | build |
| **MAJ-4** | **Register union / prefetch depth.** 1 wave/SIMD is *not* fatal — `VMCNT`=63 lets one wave hold many loads in flight, and only 4-8 are needed. But every inner loop must be unrolled to that depth (~32 VGPRs), additive across all tasks in the megakernel. Offline number (2026-09-14): the worker kernel with the runtime plus our five tasks is 182 VGPRs, no VGPR spills, 2 waves/SIMD; the tasks alone, as the kernel-test launcher's wrappers, are 6-90 VGPRs (`env/offline_gfx942/resources.txt`, `kernel_tests`). The union with the CK linears is measured from the generated kernel on day 2. | Occupancy | `docs/mi300x/07-achievable-bandwidth.md` · `docs/fleet` Q3 | build |
| **MAJ-5** | **Per-boundary latency on a 326-boundary chain.** The runtime's dependency model is a chain and nothing overlaps across an operator boundary; each boundary exposes a release flush, a cross-XCD atomic, a poll wake and an acquire. At 1 us per boundary that is 28% of the bandwidth band, at 5 us it exceeds it. Unmeasured on MI300X; layer 1 alone calibrates it (`docs/design-doc/09-expected-performance.md`, DQ1). | Runtime | `docs/design-doc` DQ1 · `docs/fleet` Q5 | gpu |
| **MAJ-6** | **Re-scoped: does the latent cache survive anywhere?** Every task's acquire executes `buffer_inv sc1`, which invalidates the XCD's non-coherently cached L2 lines, so L2 residency across operators is not expected (`docs/design-doc/03-synchronization.md`, DQ4). The remaining candidate is the memory-side Infinity Cache (`secondary` source), with `USE_NT_WEIGHTS=1` (`sc1 nt`, MALL no-allocate) as the experiment (DQ5). Worth at most the 30 MiB per token of cache reads. | Cache policy | `docs/design-doc` DQ4, DQ5 · `docs/deepseek-v2-lite` Q3 | gpu |

---

## Minor — cost is bounded or the fallback is cheap

### Correctness & numerics

| ID | Problem | Area | Where | When |
|---|---|---|---|---|
| MIN-2 | BF16 noise floor not calibrated — correctness thresholds are reasoned, not measured. | Correctness | `docs/deepseek-v2-lite` Q6 | local |
| MIN-4 | No FP8 accuracy data published for the **Base** variant (only Instruct). | FP8 | `docs/acceleration` Q3 | local |

### Model & data

| ID | Problem | Area | Where | When |
|---|---|---|---|---|
| MIN-6 | Expert routing correlation across 32 steps unknown — decides whether expert→XCD affinity pays. | Scheduling | `docs/deepseek-v2-lite` Q4 | local |

### Runtime & kernels

| ID | Problem | Area | Where | When |
|---|---|---|---|---|
| MIN-25 | **Scheduler block `k` is assumed to sit on XCD `k`.** Worker-to-scheduler queues are same-XCD volatile stores with no fence; the runtime never checks the placement. `[SCHED_XCD] sched_id=k xcd=k` must hold on all 8 lines at every launch. | Runtime | `docs/fleet` Q10 | gpu |
| MIN-26 | **Stock routing is wrong for this model.** `renormalize=true` is hard-coded (`task_register.cc:3689`) but `norm_topk_prob=false`; router logits are BF16 in the demo while the reference router is FP32; the top-k kernel zeroes the logits after reading (B8 needs a copy). One routing variant fixes the first two; a BF16-vs-FP32 selection check over 32 steps decides whether an FP32 router GEMV is also needed. | Correctness | `docs/fleet` Q13 | local then build |
| MIN-29 | **`mla_prep` is one workgroup streaming 2 MiB of `W_uk`**, serial on the chain: estimated 20-40 us per layer if it shows. Local fix: fold the product into `mla_attend` per XCD or make `mla_prep` a 16-tile gang op. | Kernel | `docs/design-doc` DQ2 | gpu |
| MIN-11 | Top-6 experts over 8 XCDs leaves **2 chiplets idle** during the 99 MB phase. **Candidate fix:** fold the two shared experts in as always-selected experts 64 and 65 (exact split of the 2816-wide shared MLP), giving 8 active experts on 8 XCDs and removing the separate shared-expert ops. | Scheduling | `docs/fleet` Q4 | gpu |
| MIN-14 | MFMA vs VALU for the **weight** GEMVs (M=1). Settled for MLA attention: M=`BLOCK_H`=16 fills a 16×16 MFMA tile, and vLLM ships `matrix_instr_nonkdim: 16`. | Kernel | `docs/acceleration` Q2 · `docs/mla-decode` Q2 | build |
| MIN-15 | Split-KV value estimated at ~114 µs from a crude CU-count ratio. | Attention | `docs/acceleration` Q5 | gpu |
| MIN-21 | **HBM load-to-use latency unknown.** No published figure found (ACM 403; Chips and Cheese gives only Infinity Cache ≈218 ns). Does not block the MLP analysis, which holds across 250 ns-2 µs. | Memory | `docs/mi300x/07-achievable-bandwidth.md` | gpu |
| MIN-23 | **`P_split`=32 is chosen from a traffic model, not measured.** Attention may cost 5-7 µs/layer (13-17% of budget) or much less. One constant to sweep. | Attention | `docs/mla-decode` Q1 | gpu |
| MIN-24 | **32 KiB FP32 accumulator vs 64 KiB LDS.** `o_acc` at BLOCK_H=16 × 512 × 4 B is half the LDS budget before staging `ql_nope`. | Kernel | `docs/mla-decode` Q3 | build |
| MIN-22 | **`VMCNT`=63 is necessary, not sufficient.** Per-CU miss-queue (MSHR) and L2 request-queue depths are undocumented and could bind before the wave-level limit. One unroll-depth sweep settles it. | Memory | `docs/mi300x` Q14 | gpu |
| MIN-16 | Cost of `buffer_inv sc1` / `buffer_wbl2 sc1` — sets task-graph granularity. | Memory model | `docs/mi300x` Q5 | gpu |
| MIN-17 | Cooperative launch overhead (known ROCm slowdown issue #3410). | Runtime | `docs/mi300x` Q7 | gpu |
| MIN-18 | Optimal `s_sleep` interval for cross-XCD polling (`S_WAKEUP` can't cross workgroups). | Runtime | `docs/mi300x` Q8 | gpu |

### Environment & measurement

| ID | Problem | Area | Where | When |
|---|---|---|---|---|
| MIN-19 | Counter names unverified (`TCC_EA0_*`); bytes-from-requests formula is ours, not documented. | Profiling | `docs/mi300x` Q11, Q12 | gpu |
| MIN-20 | ROCm version, partition commands, workgroup→XCD mapping, AITER usability. | Setup | `docs/mi300x` Q1, Q3, Q9, Q10 | gpu |

---

## Documentation defects found in sources

Not our bugs — but each one could mislead us, so they are recorded.

| ID | Defect | Source |
|---|---|---|
| DOC-1 | Fleet code comment: "all CUs within an XCD share the same **32MB** L2" — it is 4 MB per XCD, 32 MB aggregate. | `mpk_atoms.cuh` |
| DOC-2 | Fleet code comment: "MI300 (gfx942) has **192KB** shared memory per CU" — not in the ISA or ROCm docs; ISA says 64 kB LDS. | `include/mirage/config.h:61` |
| DOC-3 | Fleet paper §5.1 says the scheduler reads `HW_ID`; the code reads `HW_REG_XCC_ID`. Different registers per the ISA. Code is right. | paper vs `persistent_kernel.cuh:186` |
| DOC-4 | Fleet paper §6.2 and repo README give bs=1 numbers ~4% apart (7.83/6.82/6.73 vs 7.993/7.076/7.012 ms). | `docs/fleet` Q7 |
| DOC-5 | AMD partitioning doc: CPX = 1 XCD per partition, yet partitions "must include an even number of XCDs". Unreconciled. | ROCm partitioning overview |
| DOC-6 | Same doc: "supports two memory partitioning modes", then documents three (NPS1/2/4). | ROCm partitioning overview |
| DOC-7 | Same doc: prose says DPX is NPS1/2/4-compatible; its own summary table says NPS1/2 only. | ROCm partitioning overview |
| DOC-8 | ROCm counter page: "Preliminary validation of all MI300 and MI200 series performance counters is in progress." Asterisked counters need evaluation. | ROCm counters |
| DOC-9 | Infinity Cache (256 MB / ~17 TB/s) appears in **no** primary AMD source we read — ROCm microarch page never mentions MALL. Could change the memory plan if real. | `docs/mi300x` Q13 |

---

## Resolved

| ID | Problem | Resolution |
|---|---|---|
| ✅ | Materialized vs runtime MLA absorption | Materialized fusion is ~2× worse at S=1024 (1,927 vs 979 vs 739 MiB). Runtime reassociation chosen. `docs/deepseek-v2-lite` Q2 |
| ✅ | `XCC_ID` assembler syntax | `s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)` — confirmed in Fleet source. `docs/mi300x` Q2 |
| ✅ | Does Fleet support MoE? | Yes — four MI300 task kernels, Python plumbing, MoE demo. The gap is MLA, not MoE. |
| ✅ | Does an FP8 checkpoint exist for our model? | Yes — `RedHatAI/DeepSeek-Coder-V2-Lite-Base-FP8`, W8A8 static, 16.13 GB. |
| ✅ | Can we run gfx950 code on MI300X? | No. `gfx9-4-generic` spans both but drops FP8/BF8 — so FP8 differs between them. |
| ✅ MIN-3 | MLA + quantization | FP8 scales are **per-tensor scalars**, so they factor out of the reassociated matmul. vLLM's constraint came from materializing weight products; we don't. `docs/acceleration` Q1 |
| ✅ MIN-7 | Weight-shard gating | None. Anonymous range request returns HTTP 206. `docs/deepseek-v2-lite` Q7 |
| ✅ MIN-8 | FP8 loader | `F8_E4M3` + one FP32 scalar per tensor; router, norms and `lm_head` stay BF16. No vLLM needed. `docs/acceleration` Q4 |
| ✅ MIN-9 | `lm_head` argmax | Split for **parallelism**, not traffic — the logit buffer is only 200 KB. Keep a debug path for B15. `docs/deepseek-v2-lite` Q9 |
| ✅ MIN-12 | Split-KV merge kernel | Reuse `merge_splitkv_ck_fmha`; rewrite the GQA-paged wrapper. `docs/fleet` Q8 |
| ✅ MIN-13 | KV-cache append kernel | Paged GQA, not reusable — write our own (~30 lines). Take its 3-phase decomposition and the ~3.8K-cycle cost anchor. `docs/fleet` Q9 |
| ✅ MIN-10 | 32 vs 38 CUs/XCD | Nothing hard-coded: `utils.py:38-60` derives 296 workers + 8 schedulers from the device CU count; per-XCD worker lists are discovered at runtime from `HW_REG_XCC_ID`. `docs/fleet` Q2 |
| ✅ | Can 1 wave/SIMD saturate HBM? | **Yes, subject to MIN-22.** `VMCNT` is 6 bits (63 outstanding loads/wave); only 2-4 are needed. Not a ceiling — a loop-structure requirement. `docs/mi300x/07-achievable-bandwidth.md` |
| ✅ | Is runtime reassociation the right MLA form? | **Yes** — it is what vLLM does (`einsum`/`bmm` against `W_UK`/`W_UV` at runtime, never fused). `docs/mla-decode/02-kernel-anatomy.md` |
| ✅ | Is our split KV-cache layout right? | **Yes** — `BLOCK_DMODEL=512` and `BLOCK_DPE=64` are separate tiles everywhere; AITER flattens paged caches to `page_size=1`. `docs/mla-decode` Q4 |
| ✅ | What bandwidth is actually achievable? | **3.66-4.3 TB/s (69-81%)**. AMD's Dot acceptance threshold and BabelStream peak. Realistic BF16 target 1.15-1.35 ms/token. |
| MIN-1 (resolved) | Runtime MLA reassociation within tolerance? | **Yes, within the reference's own ordering noise.** At the real shapes on CPU, B5 differs by 3.2e-3 and B6 by 2.0e-2 to 2.7e-2 at score magnitude 37, the same as the reference's own arithmetic in another summation order (2.5e-2 to 3.0e-2); score magnitude, not reassociation, sets the attention floor. `harness/results/reassoc_check.json`, `docs/design-doc/07-correctness.md` item 7 |
| MIN-5 (resolved) | Prompt not chosen | `harness/prompt_ids.json`: BOS plus the first 1,023 tokens of `split_linear_tasks.py` from the pinned Fleet submodule; `make_prompt.py` checks it, never regenerates it. `docs/deepseek-v2-lite` Q8 |
| MIN-30 (resolved) | `MAX_OUTPUTS_PER_TASK` 3 -> 5 | **Safe by inspection of the source:** `sizeof(TaskDesc)` grows from 112 to 128 bytes, both static asserts (`% sizeof(int)`, `% 16`) still hold, and the HIP path's fixed `16 * sizeof(TaskDesc)` shared buffer plus task ids goes from 1,920 to 2,176 bytes against the 3,016-byte reserve (`persistent_kernel.cuh:704-713`). Independent review of `local/harness`. |
| MIN-27 (resolved) | gfx950-only code in the gfx942 build | **The three items found by reading are the only ones.** With `fleet/patches/gfx942.patch` the whole header set compiles for gfx942 under ROCm 7.0's hipcc with zero errors, in the variant our graph uses and in the CK FMHA variant the Qwen3 smoke graph uses (`env/offline_gfx942/README.md`). |
| MIN-28 (resolved, negative) | CK split-KV FMHA at MLA head dims | **Not available.** Both `ck_tile` split-KV pipelines `static_assert(kSubQKHeaddim <= 256)` (`block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp:48`, and the `nwarp_sshuffle` variant), at Fleet's pinned CK `d8ee107a` and at `rocm-7.2.4`; no MLA pipeline exists under `ck_tile/ops/fmha`. `mla_attend` is the spec kernel (`fleet/tasks/mi300/mla_attend_mi300.cuh`), D12's reversal condition met. `docs/fleet` Q11, `docs/design-doc` DQ3 |
| MIN-31 (resolved) | `partials` imap with 33 splits over 8 tasks | **Exact by inspection of the source:** the runtime offsets each task's output pointer by `dim[0] / grid_dim.x * bid.x` rows (floor, `runtime.cc:1360-1363`), the registration passes the same floor offset and the kernel subtracts it back, and the event slicing is `gcd(8, 1) = 1` (`runtime.cc:548-552`): one event with 8 triggers, as the design assumes. Independent review of `local/harness`. |

---

## Triage

**Before GPU access** (local, needs PyTorch): MIN-1 and MIN-5 are done;
MIN-2 (the floor) and MIN-6 (routing correlation) have their scripts'
inputs produced by `harness/run_reference.py`, which needs the weights, so
they run on day 1. MIN-4's method is settled, it just needs a run.

**Settled offline on 2026-09-14** (`env/offline_gfx942/`): MIN-27, MIN-28 and the compile half of MAJ-1, the lowering half of MAJ-3, the static half of MAJ-4.

**Day 1 on the machine, in order:** MAJ-1 (the cmake and cargo build, then a graph run) → MAJ-3 (the generated kernel's per-task path) → MIN-25 → MAJ-4 (the union with the CK linears).

**Day 3, from the layer-1 measurement:** MAJ-5 (`t_b`) and MIN-29 (`mla_prep`) are read off one run; they decide whether the graph is restructured before M3 (`docs/design-doc/09-expected-performance.md`).
MAJ-1 gates the strategy; decide extend-vs-reimplement by end of day 1.

**Cheapest high-value experiment:** MAJ-6 (non-temporal weight loads). Hours of
work, and the one place our model's structure beats Fleet's evaluation at batch 1.
