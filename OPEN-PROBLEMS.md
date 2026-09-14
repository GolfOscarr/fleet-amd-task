# Open Problems

Consolidated index of everything unresolved, so a problem can be located without
re-reading four doc sets. Detail lives in each set's `99-open-questions.md`.

Last updated: 2026-09-14 · 6 major · **18 minor open** · 16 resolved · 9 documentation defects

**When** — `local` = resolvable without a GPU · `gpu` = needs the MI300X ·
`build` = needs a working toolchain

---

## Major — blocks work or changes the design

| ID | Problem | Area | Where | When |
|---|---|---|---|---|
| **MAJ-1** | **Does the Fleet repo build and run on gfx942?** Defaults to gfx950; `AMDGPU_TARGETS=gfx942` untested. Gates extend-vs-reimplement. | Build / runtime | `docs/fleet` Q1 | build |
| **MAJ-2** | **No MLA kernel exists.** Repo has GQA paged-attention only, and AITER's gfx942 path is hand-written ASM. The MLA decode Chiplet-task must be written from scratch — **spec now drafted** in `docs/mla-decode/04-our-kernel-spec.md`. | Kernel | `docs/mla-decode` · `docs/fleet` §07 G1 | local then gpu |
| **MAJ-3** | **Does the agent-scope fence emit `buffer_wbl2 sc1` / `buffer_inv sc1`?** If not, cross-XCD reads go stale intermittently and present as a numerics bug. Narrowed by the code read: the runtime hand-writes no cache-control ops; everything rests on two `__builtin_amdgcn_fence(..., "agent")` calls (`mpk_atoms.cuh:300`, `persistent_kernel.cuh:948`). One disassembly settles it. | Memory model | `docs/mi300x` Q4 · `docs/fleet` Q6 | build |
| **MAJ-4** | **Register union / prefetch depth.** 1 wave/SIMD is *not* fatal — `VMCNT`=63 lets one wave hold many loads in flight, and only 4-8 are needed. But every inner loop must be unrolled to that depth (~32 VGPRs), additive across all tasks in the megakernel. | Occupancy | `docs/mi300x/07-achievable-bandwidth.md` · `docs/fleet` Q3 | build |
| **MAJ-5** | **Dispatch overhead at our task sizes.** Our tasks are smaller than anything Fleet measured (158.5 vs 368 MB/layer, hidden 2048 vs 4096). Fleet warns short tasks make scheduling dominate. | Runtime | `docs/fleet` Q5 | gpu |
| **MAJ-6** | **Does the latent KV cache survive in L2?** 144 KiB/XCD/layer vs 158.5 MiB/layer of expert streaming. If non-temporal weight loads protect it, this is our one batch-1 locality win; if not, the idea is worthless. | Cache policy | `docs/deepseek-v2-lite` Q3 · `docs/acceleration` Q6 | gpu |

---

## Minor — cost is bounded or the fallback is cheap

### Correctness & numerics

| ID | Problem | Area | Where | When |
|---|---|---|---|---|
| MIN-1 | Runtime MLA reassociation within tolerance? Reorders BF16 accumulation. | Model | `docs/deepseek-v2-lite` Q1 | local |
| MIN-2 | BF16 noise floor not calibrated — correctness thresholds are reasoned, not measured. | Correctness | `docs/deepseek-v2-lite` Q6 | local |
| MIN-4 | No FP8 accuracy data published for the **Base** variant (only Instruct). | FP8 | `docs/acceleration` Q3 | local |

### Model & data

| ID | Problem | Area | Where | When |
|---|---|---|---|---|
| MIN-5 | Prompt not chosen. Must fix 1,024 token IDs and never change them. | Harness | `docs/deepseek-v2-lite` Q8 | local |
| MIN-6 | Expert routing correlation across 32 steps unknown — decides whether expert→XCD affinity pays. | Scheduling | `docs/deepseek-v2-lite` Q4 | local |

### Runtime & kernels

| ID | Problem | Area | Where | When |
|---|---|---|---|---|
| MIN-25 | **Scheduler block `k` is assumed to sit on XCD `k`.** Worker-to-scheduler queues are same-XCD volatile stores with no fence; the runtime never checks the placement. `[SCHED_XCD] sched_id=k xcd=k` must hold on all 8 lines at every launch. | Runtime | `docs/fleet` Q10 | gpu |
| MIN-11 | Top-6 experts over 8 XCDs leaves **2 chiplets idle** during the 99 MB phase. | Scheduling | `docs/fleet` Q4 | gpu |
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

---

## Triage

**Before GPU access** (local, needs PyTorch): MIN-1, MIN-2, MIN-5, MIN-6 —
and MIN-4's method is settled, it just needs a run. Each now carries a worked
recipe in its `99-open-questions.md` entry.

**Day 1 on the machine, in order:** MAJ-1 → MAJ-3 → MIN-25 → MAJ-4.
MAJ-1 gates the strategy; decide extend-vs-reimplement by end of day 1.

**Cheapest high-value experiment:** MAJ-6 (non-temporal weight loads). Hours of
work, and the one place our model's structure beats Fleet's evaluation at batch 1.
