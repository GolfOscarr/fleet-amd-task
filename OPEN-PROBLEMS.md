# Open Problems

Consolidated index of everything unresolved, so a problem can be located without
re-reading four doc sets. Detail lives in each set's `99-open-questions.md`.

Last updated: 2026-09-17 (round 3 prepared on `local/round-3`: the levers and the instruments against MAJ-7, `docs/gpu-experiments/03-acceleration/`; round 2: M4, the fault's cause, the timings; `docs/gpu-experiments/02-validation/`) · 5 major open (MAJ-7 measured: the megakernel's per-task overhead) · **15 minor open** · 31 resolved · 9 documentation defects

**When** — `local` = resolvable without a GPU · `gpu` = needs the MI300X ·
`build` = needs a working toolchain

---

## Major — blocks work or changes the design

| ID | Problem | Area | Where | When |
|---|---|---|---|---|
| **MAJ-1** | **Resolved 2026-09-15: yes.** `env/setup.sh` builds it on ROCm 7.2.4 once the isolated build environment's z3 is pinned to the venv's (the extension had linked z3 5.1 against a venv holding 4.15) and `z3/lib` is on the loader path; the Qwen3 smoke graph and our 2-, 3- and 27-layer graphs run through the persistent kernel (`env/check_day1.log`, `env/hw/20260915/runs/`). One more gfx942 hunk was needed: the CK split-KV wrapper references the gfx950-only minimal decode kernel. **Does the Fleet repo build and run on gfx942?** Narrowed offline (2026-09-14): with `gfx942.patch` every megakernel header parses for gfx942 under ROCm 7.0's hipcc (the CK linear templates included, though not instantiated) and our five kernels compile and device-link (`env/offline_gfx942/README.md`). Left for the machine: the cmake and cargo halves of `pip install -e .` and running a graph. Gates extend-vs-reimplement. | Build / runtime | `docs/fleet` Q1, Q12 | build |
| **MAJ-2** | **Resolved 2026-09-15: our three MLA kernels are validated on the machine.** Kernel tests 100 of 100 trials for `mla_prep`, `mla_attend` (1 and 33 splits, scores), `mla_merge_uv`; in the graph B3, B4, B5, B6 and B13 at layers 0 and 1 are within threshold (`env/hw/20260915/runs/L2_it1`). What remains is speed (idle-clock timings only so far). **No MLA kernel exists.** Repo has GQA paged-attention only, and AITER's gfx942 path is hand-written ASM. Spec drafted in `docs/mla-decode/04-our-kernel-spec.md`. **Possible shortcut (MIN-28):** the repo already instantiates CK's split-KV FMHA pipeline with separate QK/V head dims; if the machine's CK has the 576/512 configuration, phase B is an instantiation plus a wrapper. | Kernel | `docs/mla-decode` · `docs/fleet` §07 G1, Q11 | local then gpu |
| **MAJ-3** | **Does the agent-scope fence emit `buffer_wbl2 sc1` / `buffer_inv sc1`?** **Answered on the machine (2026-09-15).** On the VM's own hipcc 7.2.53211 the four constructs lower exactly as designed and none takes system scope: release = `buffer_wbl2 sc1`, acquire = `buffer_inv sc1`, `__threadfence()` = both at agent scope, agent-scope atomic load = `sc1` load plus `buffer_inv sc1`, with no `sc0 sc1` in any of the four kernels (`env/hw/probes/fence_probe.cu`, `env/hw/20260915/`). The cost is bounded too: 115 ns on release and 137 ns on acquire uncontended, 317 ns on acquire with eight XCDs (MIN-16). One item remains: the worker kernel also carries 72 system-scope `buffer_wbl2 sc0 sc1` and 48 `buffer_inv sc0 sc1` sites from the printf and assert hostcall paths (`env/offline_gfx942/fences.txt`); whether any lies on the per-task path is read off the generated `kernel_0.cu` on day 2. | Memory model | `docs/mi300x` Q4 · `docs/fleet` Q6 | build |
| **MAJ-4** | **Register union / prefetch depth.** 1 wave/SIMD is *not* fatal — `VMCNT`=63 lets one wave hold many loads in flight, and only 4-8 are needed. But every inner loop must be unrolled to that depth (~32 VGPRs), additive across all tasks in the megakernel. Offline number (2026-09-14): the worker kernel with the runtime plus our five tasks is 182 VGPRs, no VGPR spills, 2 waves/SIMD; the tasks alone, as the kernel-test launcher's wrappers, are 6-90 VGPRs (`env/offline_gfx942/resources.txt`, `kernel_tests`). **Measured on the machine (2026-09-15):** LDS binds before registers. A 184-VGPR 256-thread kernel gets 2 blocks per CU on registers alone but 1 with the worker kernel's 58,368 B dynamic LDS request, and 304 blocks are co-resident on the 304 CUs (`env/hw/probes/occupancy.cu`). So the occupancy is one wave per SIMD as `07-achievable-bandwidth.md` assumes, there is register headroom before that changes, and the cooperative launch does not deadlock. The union with the CK linears is still measured from the generated kernel on day 2. | Occupancy | `docs/mi300x/07-achievable-bandwidth.md` · `docs/fleet` Q3 · `docs/mi300x` Q6 | build |
| **MAJ-5** | **Per-boundary latency on a 326-boundary chain.** The runtime's dependency model is a chain and nothing overlaps across an operator boundary; each boundary exposes a release flush, a cross-XCD atomic, a poll wake and an acquire. At 1 us per boundary that is 28% of the bandwidth band, at 5 us it exceeds it. **First measurement (2026-09-15):** a release store on XCD 0 to an acquire poll on XCD 1 is 703 ns one way over 10,000 rounds, and the fences add 115 to 317 ns (`env/hw/20260915/`), so release plus hop plus acquire is about 0.9 to 1.0 us per boundary before dispatch and wake-up. That puts 326 boundaries at roughly 300 us, 25 to 30% of the 1,148 to 1,349 us band, from a lower bound — so the boundary-count reductions in `09-expected-performance.md` are worth doing rather than contingent — reductions 1 and 2 (the `rmsnorm` and `moe_silu_mul` fusions, 326 boundaries to 245) on this number alone, reduction 3 still behind its own 3 us gate. Not closed: this is a microbenchmark floor, and the layer-1 measurement remains the calibration of `t_b`. | Runtime | `docs/design-doc` DQ1 · `docs/fleet` Q5 | gpu |
| **MAJ-7** | **Opened 2026-09-15: a gang operator runs on eight workgroups, so an operator at batch 1 moves bytes at about 0.2 TB/s.** Event timing of the 2-layer graph over 32 iterations (`env/hw/20260915/runs/L2_it32`): `o_proj` (8 MB of weights) 38 us, `mla_merge_uv` (2 MB) 52 us, `mla_attend` (1.2 MB of cache) 210 us, a norm 14 us, 2.06 to 2.21 ms per iteration for two layers against a 90 us bandwidth time. Not the clocks: the same numbers with a bandwidth probe holding the GPU busy (`measure_gpu1`), and `amd-smi set --perf-level` is refused in the VF. The gang model of the runtime executes one workgroup per XCD per operator with the tiles looped inside, so eight of 304 CUs carry each operator; this is the "dispatch reduction, not bandwidth" limit the design quoted from Fleet's own batch-1 data (`06-optimization-strategy.md`), now measured on our graph. The whole model without the head (`runs/L27_it32`, 32 iterations): 15.6 ms per iteration; by operator class per iteration `mla_attend` 5.7 ms (27 x 211 us), `mla_merge_uv` 1.45 ms, `moe_silu_mul` 1.10 ms (26 x 42 us for an elementwise op), `o_proj` and `down` 1.01 ms, `gang_moe_w13` 0.66 ms, `moe_mul_sum_add` 0.60 ms, norms 0.55 ms, `q`/`kv_a` linears 0.12 ms, router 0.10 ms; about 4 ms is outside any operator (the 326 boundaries at about 1 us each and the serial `mla_prep`, both as predicted). The path forward, in order of the table: prefetch and more splits inside `mla_attend` (ours), the same in `mla_merge_uv` (ours), then per-tile tasks (37 per XCD, the per-task pointer offsets the runtime already supports) or multi-workgroup gangs for the stock linears and the elementwise ops; every per-operator number of the design's band assumes it. **Re-read 2026-09-16 (round 2 preparation):** the table does not indict every gang operator: the plain gang linear `qkva` (15 MB) shows 4.4 us, about 3.4 TB/s, and the silu-fused `gate_up` (92 MB) 4.8 us, which no memory system delivers, so those rows are at the floor or the event gaps mis-attribute; the time sits in the residual variant (`o_proj` 8 MB and `down` 46 MB both about 37 us, a floor not a rate), `moe_silu_mul` (42 us), the norms (14 to 50 us) and our three kernels. Round 2 validates the attribution first (two stop-after runs timed from the kernel trace, B0 of `docs/gpu-experiments/02-validation/02-session-plan.md`), tests the residual variant with the per-tile flag (`--tile-linears`, P5), then works the list in that order. The two attention kernels themselves were latency-bound, not bandwidth-bound: one 16-byte load at a time per thread, 144 per tile; P6 batches four and maps score lanes to heads (`docs/gpu-experiments/02-validation/01-preparation.md`). **Round 2 (2026-09-16):** the attention kernel costs 34 us standalone, warm or cold (the suite binary, `KT_TIME`, `KT_COLD`), and 146 to 215 us inside the megakernel; the number does not move with the dispatch path (a regular task per split, `--attend-tasks`), the prefetch depth (P6) or the rows per tile (`--split 17`); E2 (`--nt-weights`, the runtime's non-temporal weight loads) takes it to 150 us and the iteration from 12.3 to 10.2 ms on the event clock (15.0 to 13.0 as host means); per-tile linears to 9.6 ms with E2. The floor is what the megakernel does around every task, paid 33 times per attention operator; measuring that overhead directly (a graph of empty tasks, the completion fences one at a time) is the next step. `docs/gpu-experiments/02-validation/04-results.md`. **Round 3 (prepared 2026-09-17, `docs/gpu-experiments/03-acceleration/`):** the instruments for that measurement are built (the per-worker timing for every worker and our task classes, the shader-clock spin, the empty-task ladder, the fence knobs as defines, the clock sampler) beside the levers (three boundary fusions taking the graph from 326 to 246 operators, the MFMA attention, the weight prefetch by side operators, the streaming loads), all as off-by-default flags; the session plan runs the gains first and the attribution after (`05-session-plan.md`). | Performance | `docs/design-doc` 06, 09 · `env/hw/20260915/runs/L2_it32` | build |
| **MAJ-6** | **Re-scoped: does the latent cache survive anywhere?** Every task's acquire executes `buffer_inv sc1`, which invalidates the XCD's non-coherently cached L2 lines, so L2 residency across operators is not expected (`docs/design-doc/03-synchronization.md`, DQ4). The remaining candidate is the memory-side cache, and it is **no longer a `secondary` claim: it is measured (2026-09-15)** — pointer-chase latency is 81 ns at 1 MiB, 258 ns at 64 MiB and 342 ns at 1 GiB, so a tier sits between L2 and HBM (`env/hw/probes/chase.cu`, `env/hw/20260915/`), even though `rocminfo` lists no L3 row. Its capacity is unmeasured; 64 MiB hits it and 1 GiB does not, and the 30.4 MB latent cache would fit. `USE_NT_WEIGHTS=1` (`sc1 nt`, MALL no-allocate) is still the experiment (DQ5). Worth at most the 30 MiB per token of cache reads. **Measured 2026-09-16 (E2, `--nt-weights` = `-DMPK_NT_WEIGHT_LOADS`):** with the stock linears' weight loads non-temporal the iteration drops from 12,268 to 10,230 us on the event clock and `mla_attend` from 215 to 150 us (`env/hw/20260916/runs/L27_head_it32_nt_al65536`): the weight streams were evicting what the attention re-reads, so some cache does survive across tasks; the same treatment of our kernels' own streams is the next experiment (`docs/gpu-experiments/02-validation/06-lessons.md`, item 2). | Cache policy | `docs/design-doc` DQ4, DQ5 · `docs/deepseek-v2-lite` Q3 · `docs/mi300x` Q13 | gpu |

---

## Minor — cost is bounded or the fallback is cheap

### Correctness & numerics

| ID | Problem | Area | Where | When |
|---|---|---|---|---|
| MIN-2 (resolved 2026-09-15, again 2026-09-16) | BF16 noise floor calibrated on the machine: router 3.59e-3, scores 2.61e-3, logits 4.1e-2, norm 4.5e-2, layer 4.33e-3, gemv 3.2e-3, rope 5.3e-3, attention 1.0e-2, expert 1.1e-2 (`harness/ref/calibration.json`); the thresholds are 4x the floor. | Correctness | `docs/deepseek-v2-lite` Q6 | local |
| MIN-4 | No FP8 accuracy data published for the **Base** variant (only Instruct). | FP8 | `docs/acceleration` Q3 | local |
| MIN-32 | The route-log rule needs a tie tolerance. Over 32 steps 60 of 832 (step, layer) top-k sets differ from the reference by one expert in most cases, growing after step 22, with every id equal (`env/hw/20260916/runs/L27_head_it32_al65536`); the growth curve's one jump (layer 5, 2.97e-2) is the same step-0 flip at MoE index 4. The router floor is 3.6e-3: a set that differs only by experts whose reference logits are within the floor of the sixth is a tie, and `compare.py` should say so instead of FAIL. | Correctness | `docs/gpu-experiments/02-validation/04-results.md` | local |
| MIN-33 | The stock `gang_linear_silu_kernel` over-reads its input at batch 1: its CK tile GEMM has `MPerBlock = 16` and no active-token mask, so it reads 16 rows, 64 KB, from a `[1, 2048]` activation. The M4 fault of round 1, fixed on our side by backing every single-row activation with 16 rows (`fleet/build_graph.py`, `ROW_SLACK`); the kernel is unchanged, a mask on `num_active_tokens` (the plain and residual variants have one) is the upstream fix, and the same check applies to any stock kernel added later. | Correctness | `docs/gpu-experiments/02-validation/04-results.md`, "The fault" | local |
| MIN-34 | rocprofv3 cannot attach to the torch ROCm wheel: it bundles its own `libamdhip64.so`, `libhsa-runtime64.so`, `librocprofiler-register.so` and `librocprofiler-sdk.so`, and loaded after the system runtime the tool hooked, the second registration is fatal (`api registration failed with error code 16`); linking only the HIP library to the system one does not help. B3's traffic, bandwidth and L2 hit rate were not measured in round 2. The counters come from a standalone binary (the kernel suite under `KT_TIME`; `copy_bytes` proved the counters in round 1) or from a wheel built against the system ROCm. | Measurement | `docs/gpu-experiments/02-validation/06-lessons.md` item 7 | gpu |

### Model & data

| ID | Problem | Area | Where | When |
|---|---|---|---|---|
| MIN-6 | Expert routing correlation across 32 steps unknown — decides whether expert→XCD affinity pays. | Scheduling | `docs/deepseek-v2-lite` Q4 | local |

### Runtime & kernels

| ID | Problem | Area | Where | When |
|---|---|---|---|---|
| MIN-26 | **Resolved 2026-09-15:** our FP32 `moe_router` with the model's `norm_topk_prob=false` gives B8 logits rel 3.7e-3, B9 top-k indices exact ([2, 19, 39, 40, 47, 49]) and B10 weights rel 3.8e-3 at layer 1; route log PASS. **Stock routing is wrong for this model.** `renormalize=true` is hard-coded (`task_register.cc:3689`) but `norm_topk_prob=false`; router logits are BF16 in the demo while the reference router is FP32; the top-k kernel zeroes the logits after reading (B8 needs a copy). One routing variant fixes the first two; a BF16-vs-FP32 selection check over 32 steps decides whether an FP32 router GEMV is also needed. | Correctness | `docs/fleet` Q13 | local then build |
| MIN-29 | **`mla_prep` is one workgroup streaming 2 MiB of `W_uk`**, serial on the chain: estimated 20-40 us per layer if it shows. Local fix: fold the product into `mla_attend` per XCD or make `mla_prep` a 16-tile gang op. | Kernel | `docs/design-doc` DQ2 | gpu |
| MIN-11 | Top-6 experts over 8 XCDs leaves **2 chiplets idle** during the 99 MB phase. **Candidate fix:** fold the two shared experts in as always-selected experts 64 and 65 (exact split of the 2816-wide shared MLP), giving 8 active experts on 8 XCDs and removing the separate shared-expert ops. | Scheduling | `docs/fleet` Q4 | gpu |
| MIN-14 | MFMA vs VALU for the **weight** GEMVs (M=1). Settled for MLA attention: M=`BLOCK_H`=16 fills a 16×16 MFMA tile, and vLLM ships `matrix_instr_nonkdim: 16`. | Kernel | `docs/acceleration` Q2 · `docs/mla-decode` Q2 | build |
| MIN-15 | Split-KV value estimated at ~114 µs from a crude CU-count ratio. | Attention | `docs/acceleration` Q5 | gpu |
| MIN-23 | **`P_split`=32 is chosen from a traffic model, not measured.** Attention may cost 5-7 µs/layer (13-17% of budget) or much less. One constant to sweep. | Attention | `docs/mla-decode` Q1 | gpu |
| MIN-24 | **32 KiB FP32 accumulator vs 64 KiB LDS.** `o_acc` at BLOCK_H=16 × 512 × 4 B is half the LDS budget before staging `ql_nope`. | Kernel | `docs/mla-decode` Q3 | build |
| MIN-17 | Cooperative launch overhead (known ROCm slowdown issue #3410). | Runtime | `docs/mi300x` Q7 | gpu |
| MIN-18 | Optimal `s_sleep` interval for cross-XCD polling (`S_WAKEUP` can't cross workgroups). | Runtime | `docs/mi300x` Q8 | gpu |

### Environment & measurement

| ID | Problem | Area | Where | When |
|---|---|---|---|---|
| MIN-32 | **`docs/mi300x/06-profiling.md`'s read formula undercounts reads by 2x.** It charges a flat 64 B per read request and has no 128 B term; the validated decomposition is in `docs/mi300x` Q12. That file has not been corrected yet, and `harness/measure.py` must use the validated form. | Profiling | `docs/mi300x` Q12 | local |
| MIN-20 | **AITER usability on gfx942** — whether its MLA decode runs on this part and can serve as an oracle. Narrowed 2026-09-15: the ROCm version, the partition commands and the workgroup→XCD mapping are all settled, AITER alone is left. | Setup | `docs/mi300x` Q10 | gpu |

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
| DOC-9 | Infinity Cache (256 MB / ~17 TB/s) appears in **no** primary AMD source we read — ROCm microarch page never mentions MALL, and `rocminfo` on the part lists no L3 row. The defect stands, but the tier is real: pointer chase measures 258 ns at 64 MiB against 81 ns in L2 and 342 ns at HBM (2026-09-15). Size and bandwidth are still undocumented and unmeasured. | `docs/mi300x` Q13 |

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
| ✅ | Can 1 wave/SIMD saturate HBM? | **Yes, and now measured (2026-09-15):** 4,325 GB/s at 304 blocks, one wave per SIMD, unroll 4. `VMCNT` is 6 bits (63 outstanding loads/wave); only 2-4 are needed. Not a ceiling — a loop-structure requirement. A second wave per SIMD does not help (4,033 GB/s). `docs/mi300x/07-achievable-bandwidth.md`, `docs/mi300x` Q14 |
| ✅ | Is runtime reassociation the right MLA form? | **Yes** — it is what vLLM does (`einsum`/`bmm` against `W_UK`/`W_UV` at runtime, never fused). `docs/mla-decode/02-kernel-anatomy.md` |
| ✅ | Is our split KV-cache layout right? | **Yes** — `BLOCK_DMODEL=512` and `BLOCK_DPE=64` are separate tiles everywhere; AITER flattens paged caches to `page_size=1`. `docs/mla-decode` Q4 |
| ✅ | What bandwidth is actually achievable? | **3.66-4.3 TB/s (69-81%), and the band held when measured (2026-09-15).** BabelStream in HIP on the VM: Copy 4,319,424 MB/s, Mul 4,231,808, Add 3,894,077, Triad 4,133,844, Dot 4,038,666, all above AMD's acceptance thresholds; our own streaming read is 3.943 TB/s. Realistic BF16 target 1.15-1.35 ms/token stands. `env/hw/20260915/`, `docs/mi300x` Q14 |
| MIN-1 (resolved) | Runtime MLA reassociation within tolerance? | **Yes, within the reference's own ordering noise.** At the real shapes on CPU, B5 differs by 3.2e-3 and B6 by 2.0e-2 to 2.7e-2 at score magnitude 37, the same as the reference's own arithmetic in another summation order (2.5e-2 to 3.0e-2); score magnitude, not reassociation, sets the attention floor. `harness/results/reassoc_check.json`, `docs/design-doc/07-correctness.md` item 7 |
| MIN-5 (resolved) | Prompt not chosen | `harness/prompt_ids.json`: BOS plus the first 1,023 tokens of `split_linear_tasks.py` from the pinned Fleet submodule; `make_prompt.py` checks it, never regenerates it. `docs/deepseek-v2-lite` Q8 |
| MIN-30 (resolved) | `MAX_OUTPUTS_PER_TASK` 3 -> 5 | **Safe by inspection of the source:** `sizeof(TaskDesc)` grows from 112 to 128 bytes, both static asserts (`% sizeof(int)`, `% 16`) still hold, and the HIP path's fixed `16 * sizeof(TaskDesc)` shared buffer plus task ids goes from 1,920 to 2,176 bytes against the 3,016-byte reserve (`persistent_kernel.cuh:704-713`). Independent review of `local/harness`. |
| MIN-27 (resolved) | gfx950-only code in the gfx942 build | **The three items found by reading are the only ones.** With `fleet/patches/gfx942.patch` the whole header set compiles for gfx942 under ROCm 7.0's hipcc with zero errors, in the variant our graph uses and in the CK FMHA variant the Qwen3 smoke graph uses (`env/offline_gfx942/README.md`). |
| MIN-28 (resolved, negative) | CK split-KV FMHA at MLA head dims | **Not available.** Both `ck_tile` split-KV pipelines `static_assert(kSubQKHeaddim <= 256)` (`block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp:48`, and the `nwarp_sshuffle` variant), at Fleet's pinned CK `d8ee107a` and at `rocm-7.2.4`; no MLA pipeline exists under `ck_tile/ops/fmha`. `mla_attend` is the spec kernel (`fleet/tasks/mi300/mla_attend_mi300.cuh`), D12's reversal condition met. `docs/fleet` Q11, `docs/design-doc` DQ3 |
| MIN-16 (resolved) | Cost of `buffer_inv sc1` / `buffer_wbl2 sc1` | **A few hundred nanoseconds per fence, so per-operator cross-XCD dependencies are affordable.** Agent scope minus workgroup scope on a store-fence-load loop: release +115 ns (525 against 410), acquire +137 ns (894 against 756) uncontended; with all eight XCDs in the loop, release shows no measurable extra (798 against 829) and acquire costs +317 ns (991 against 673). `env/hw/probes/fence_probe.cu`, `env/hw/20260915/`, `docs/mi300x` Q5 |
| MIN-19 (resolved) | Counter names and bytes-from-requests | **Names confirmed and the formula validated to the byte.** `rocprofv3 --list-avail` lists every counter, PMC collection works inside the VF, and there are 128 TCC instances (16 channels x 8 XCC). On a 1 GiB copy the decomposition gives exactly 1.0000 GiB of reads and 1.000 GiB of writes. It also exposed a defect in our own `06-profiling.md`, logged as MIN-32. `docs/mi300x` Q11, Q12 |
| MIN-21 (resolved) | HBM load-to-use latency | **342 ns**, by pointer chase at a 1 GiB working set; 81 ns in L2 and 258 ns at 64 MiB for comparison. Inside the 250 ns to 2 us range the MLP prefetch argument assumed, so that analysis stands unchanged. `env/hw/probes/chase.cu`, `docs/mi300x` Q13 |
| MIN-22 (resolved) | `VMCNT`=63 necessary, not sufficient | **No hidden queue limit binds.** The unroll sweep at one wave per SIMD has its knee at N=4 and a 4.3 TB/s plateau, exactly as predicted: 2,706 / 2,410 / 4,325 / 4,304 / 3,787 / 4,095 GB/s at N = 1, 2, 4, 8, 16, 32. Per-CU miss queues and L2 request queues are not the binding constraint at our occupancy. `env/hw/probes/stream_read.cu`, `docs/mi300x` Q14 |
| MIN-25 (resolved) | Scheduler block `k` assumed to sit on XCD `k` | **The assumption is false and the fix is committed.** Placement is round-robin at workgroup granularity with a constant offset: `xcd == (blockIdx.x + 4) mod 8` on every block of 21 launches, alone and concurrent, perfectly balanced. `fleet/patches/sched_xcd.patch` indexes the scheduler queue by the discovered `HW_REG_XCC_ID` instead of the block id, which is correct for any offset; `env/check_day1.sh` check 3 accepts any constant worker offset. Confirmation is that check's `[SCHED_XCD]` lines on the machine. `env/hw/probes/xcc_map.cu`, `docs/fleet` Q10, `docs/mi300x` Q1 |
| MIN-31 (resolved) | `partials` imap with 33 splits over 8 tasks | **Exact by inspection of the source:** the runtime offsets each task's output pointer by `dim[0] / grid_dim.x * bid.x` rows (floor, `runtime.cc:1360-1363`), the registration passes the same floor offset and the kernel subtracts it back, and the event slicing is `gcd(8, 1) = 1` (`runtime.cc:548-552`): one event with 8 triggers, as the design assumes. Independent review of `local/harness`. |

---

## Triage

**Before GPU access** (local, needs PyTorch): MIN-1 and MIN-5 are done;
MIN-2 (the floor) and MIN-6 (routing correlation) have their scripts'
inputs produced by `harness/run_reference.py`, which needs the weights, so
they run on day 1. MIN-4's method is settled, it just needs a run.

**Settled offline on 2026-09-14** (`env/offline_gfx942/`): MIN-27, MIN-28 and the compile half of MAJ-1, the lowering half of MAJ-3, the static half of MAJ-4.

**Settled on the VM on 2026-09-15** (`env/hw/20260915/`, one hour before the
build): MIN-16, MIN-19, MIN-21, MIN-22 and MIN-25, the fence half of MAJ-3
and the occupancy half of MAJ-4; MAJ-5 and MAJ-6 narrowed by the 703 ns
cross-XCD hop and by the memory-side cache turning out to be real. MIN-32
was opened by the same work. The mapping result is the one that changed
code: `fleet/patches/sched_xcd.patch`.

**Day 1 on the machine, in order:** MAJ-1 (the cmake and cargo build, then a graph run) → MAJ-3 (the generated kernel's per-task path) → the `[SCHED_XCD]` confirmation of the MIN-25 patch → MAJ-4 (the union with the CK linears).

**Day 3, from the layer-1 measurement:** MAJ-5 (`t_b`) and MIN-29 (`mla_prep`) are read off one run; they decide whether the graph is restructured before M3 (`docs/design-doc/09-expected-performance.md`).
MAJ-1 gates the strategy; decide extend-vs-reimplement by end of day 1.

**Cheapest high-value experiment:** MAJ-6 (non-temporal weight loads). Hours of
work, and the one place our model's structure beats Fleet's evaluation at batch 1.
