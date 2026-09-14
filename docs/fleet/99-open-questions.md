# 99 — Open Questions

## Q1 — Does the repo build and run on MI300X / gfx942? `open` — **day 1, blocking**

**Why.** Gates the entire strategy in `07-gap-analysis.md`. If it builds, we
extend and spend our days on MLA. If it does not, we write a minimal runtime and
aim only at the single-layer milestone.

**Check.** `AMDGPU_TARGETS=gfx942 pip install -e . -v`, then run the smallest
`cpp_examples`/`tests` graph, then `demo/qwen3/demo.py`. Requires ROCm 7.0+.

**Decide by end of day 1.** Do not let this drift.

---

## Q2 — Does the runtime correctly detect 38 CUs/XCD on MI300X? `resolved` (2026-09-14)

**Resolved: nothing hard-codes 32 CUs per XCD.** `python/mirage/utils.py:38-60`
derives the counts from `torch.cuda.get_device_properties().multi_processor_count`:
on a device reporting 300 or more CUs, `workers = sm_cnt - 8 = 296` and
`schedulers = 8`. The MI350 case (`:61-68`) is a separate `elif` with its own
constant. On the device side, each scheduler discovers its workers by matching
`worker_xcd_map` against its own `HW_REG_XCC_ID` (`persistent_kernel.cuh:1421-1436`),
so 37 workers per XCD is a runtime outcome, not a constant. `MI300X_NUM_XCDS = 8`
(`:183`) and `NUM_XCDS_INIT = 8` (`:2116`) remain hard-coded, which is correct
for SPX mode. The `[SCHED_XCD] ... workers_on_xcd=37` line printed at startup
is the confirmation. See `03-runtime.md`, "The code, read line by line".

---

## Q3 — Register pressure once an MLA task joins the union `open`

**Why.** §8 reports the dense task set already limits occupancy to **1 wave per
SIMD**. MLA with a 512-wide latent accumulation is a large task. If the union
grows, we get spills on top of no latency hiding — for a memory-bound workload
that needs many outstanding loads, this is the main structural risk.

**Check.** `-Rpass-analysis=kernel-resource-usage` on the megakernel before and
after adding each MLA task; track VGPR/AGPR/LDS as a time series in the repo.

---

## Q4 — Is the 6-of-64 expert → 8-XCD mapping wasteful? `open` — candidate resolution found

**Why.** `ae_idx = xcd_id + expert_local_idx * 8` leaves 2 of 8 chiplets idle
during the 99 MB routed-expert phase — 25% of the machine, on the layer's
largest traffic item.

**Candidate (2026-09-14).** Fold the two shared experts into the routed set
as experts 64 and 65, always selected with weight 1.0. The shared MLP's
intermediate width is 2816 = 2 x 1408, so splitting it into two 1408-wide
experts and summing their `down_proj` outputs is exact up to FP32
accumulation order. The layer becomes top-8-of-66: 8 active experts, 8
XCDs, one each under the round-robin rule, and the separate shared-expert
ops disappear (the chain dependency model in `03-runtime.md` could not have
overlapped them anyway). Needs a routing variant that appends the two forced
entries; that is the same variant that fixes `renormalize` (Q13).

**Check.** Build both graphs; compare per-XCD busy cycles during the expert
phase and the layer time. The fused form should win on both.

---

## Q5 — Is dispatch overhead acceptable at our task sizes? `open`

**Why.** The paper warns about "designs with very short executing tasks". Their
tasks are ~104K cycles (linear) on a 368 MB/layer model; ours are smaller on
every axis. If per-task dispatch cost is a meaningful fraction of a 31.6 µs
layer, the thesis weakens at our model size.

**Check.** Time a single Chiplet-task in isolation with `s_memrealtime`; compare
task execution time against dispatch interval. `profiler.h` already provides the
instrumentation.

---

## Q6 — Does HIP agent-scope fence emit `buffer_wbl2` on our ROCm? `open`

Inherited from `../mi300x/99-open-questions.md` Q4, now narrowed by the code
read (`03-runtime.md`): the runtime issues **no cache-control instruction by
hand**. Every cross-XCD release is `threadfence_gpu()` =
`__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent")` (`mpk_atoms.cuh:300`) and
every acquire is `__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent")`
(`persistent_kernel.cuh:948`); the global atomics are inline asm with
`sc0 sc1` and no fence of their own. So the whole question is what our ROCm's
LLVM emits for those two builtins on gfx942. Check by disassembling
`threadfence_gpu` and the worker's dependency check: expect `buffer_wbl2 sc1`
+ `s_waitcnt` and `s_waitcnt` + `buffer_inv sc1` respectively.

---

## Q7 — Reconcile the paper/README batch-1 discrepancy `open`

Paper §6.2 reports 6.82 / 6.73 ms (M-tile / M-split) at bs=1; the README table
reports 7.076 / 7.012 ms. ~4% apart, unexplained. Minor, but if we cite Fleet's
numbers in our report we should say which and why.

---

## Q8 — Can we reuse `gang_attention_merge_mi300` for split-KV MLA? `resolved` (2026-09-14)

**Resolved: reuse the inner merge math, rewrite the wrapper.**

The wrapper is GQA-paged throughout:

```c
gang_attention_merge_kernel(lse_ptr, o_acc_ptr, qo_indptr, kv_indptr,
                            kv_last_page_len, output_ptr, num_kv_heads,
                            total_work_items, tile_idx)
// tile_idx -> (request_id, kv_head)
```

Template parameters are `NUM_QO_HEADS_PER_KV`, `NUM_KV_HEADS`, `PAGE_SIZE`,
`MAX_TOKENS` — all concepts our path does not have. We have one request, no
paging, and a single shared `k_pe` head.

But the inner `merge_splitkv_ck_fmha` is exactly the right algorithm: reduce
FP32 partial `lse` and `o_acc` across `NUM_KV_CHUNKS` into a BF16 output. That
is the standard running-max / sumexp rescale, and it is what our split-KV merge
needs.

**Plan:** call `merge_splitkv_ck_fmha` (or copy its body) behind our own thin
wrapper that maps `tile_idx -> kv_chunk` with no request or page indirection.

## Q9 — What does `kv_cache_update_mi300` assume about layout? `resolved` (2026-09-14)

**Resolved: paged GQA K/V. Not reusable — write our own (~30 lines).**

```c
kv_cache_update_impl(qkv_ptr, paged_k_cache_ptr, paged_v_cache_ptr,
                     q_workspace_ptr, qo_indptr_buffer_ptr,
                     paged_kv_indptr_buffer_ptr, paged_kv_indices_buffer_ptr,
                     paged_kv_last_page_len_buffer_ptr, request_id,
                     qk_norm, rope, q_norm_weight_ptr, k_norm_weight_ptr, ...)
```

Separate paged K and V caches, per-`(request_id, kv_head)` grid, page tables,
and fused QK-norm. Our latent cache is two contiguous arrays — `c_KV[S][512]`
and `k_pe[S][64]` — with no paging, no per-head K/V split, and one shared `k_pe`
head. Nothing transfers.

**Two things worth taking from it anyway:**

1. **The three-phase decomposition**, stated in its own header comment, which our
   MLA task should copy:
   - Phase A: RoPE on Q and new K, write to cache, stage Q in a workspace
   - Phase B: split-KV attention (batch-independent)
   - Phase C: merge partial results
2. **A cost anchor:** the comment calls Phase A "a lightweight operation (~3.8K
   cycles at decode) since it only does preprocessing and cache writes, no
   attention compute." Matches Table 2 in the paper. Our append should be
   similarly cheap, so if it is not, something is wrong.

---

## Q10 — Is scheduler block `k` guaranteed to run on XCD `k`? `open` — **day 1**

**Why.** `get_rand_sched_id` returns the worker's `xcd_id` as the scheduler
queue index, with the comment "scheduler_kernel block k runs on XCD k"
(`persistent_kernel.cuh:591`). The worker enqueues to that queue with
volatile stores and a compiler barrier only (`:1345-1360`), and the scheduler
reads its own queue with volatile loads. That is sound only if the scheduler
whose `blockIdx.x == k` is physically on XCD `k`. The scheduler discovers its
real XCD from the register and builds its worker list from it, so a
misplacement would not break dispatch, but it would make the worker-to-
scheduler queue a cross-XCD channel with no fence, and events could be lost
or seen late. The runtime never checks the assumption.

**Check.** The startup line `[SCHED_XCD] sched_id=k xcd=m workers_on_xcd=n`
(`:1436`) must show `k == m` for all eight schedulers, every launch. The
`[WORKER_XCD] worker_id=w xcd=x` lines for workers 0-7 (`:751-754`) should
show `x == w mod 8`; that is what puts eight consecutive gang tasks on eight
distinct XCDs under the prelaunch dispatch rule (`03-runtime.md`). If it
ever does not, the fix is to index the scheduler queue by the scheduler's
discovered XCD rather than by block id, which is a small change in
`execute_scheduler` and `get_rand_sched_id`.

---

## Q11 — Can CK's split-KV FMHA be instantiated at MLA head dims (576 / 512)? `open` — **day 1, high value**

**Why.** `paged_attention_ck_fmha_split_kv_mi300.cuh:66-98` instantiates
`BlockFmhaFwdSplitKVPipelineNWarpSShuffleQRKSVS` with tile
`sequence<16, 128, 32, 128, 32, 128>`, whose last entry is the QK head dim
and whose `kN1` is the V head dim. CK upstream keeps these separate and
recent releases add a 576/512 configuration for DeepSeek MLA decode. If the
ROCm on the machine ships it, phase B of `../mla-decode/04-our-kernel-spec.md`
is an instantiation plus a `tile_idx -> split` wrapper over our two
contiguous arrays, and phase C is CK's merge. That would cut the core work
of the project from writing a kernel to wrapping one. `ck_tile` is not
vendored (`deps/` has only `rocblas`), so this cannot be checked locally.

**Check.** On the machine: `grep -rn "576" /opt/rocm/include/ck_tile/ops/fmha`
and the `TileFmhaShape` definition; then compile a one-file instantiation at
`(kM0=16, kQKHeaddim=576, kN1=512)` and read the `static_assert` on LDS
against 57 KiB. If it compiles and fits, take it; if not, the from-scratch
spec stands.

---

## Q12 — Which gfx950-only code is in the gfx942 build? `open` — **day 1, part of Q1**

Found by reading, not building (`04-repo-map.md`, "gfx950-only code"):

1. `paged_attention_decode_minimal_mi300.cuh:27` uses
   `__builtin_amdgcn_mfma_f32_16x16x32_f16` unguarded; included by
   `task_header.cuh:34`, never registered. Expected to fail the gfx942
   compile outright. Fix: drop the include.
2. `linear_ck_mi300.cuh:73`, `:331` select a `16x16x32` BF16 warp GEMM
   "(MI350 2x K)". CK may lower it to two K=16 MFMAs on gfx942 or refuse.
3. `linear_ck_mi300.cuh:394` coherence value 18 (`sc1 nt`) is commented
   "for gfx950"; the encoding is shared with gfx942, expected fine.

**Check.** The first compile with `AMDGPU_TARGETS=gfx942` answers 1 and 2;
the disassembly of a `linear` task answers 3.

---

## Q13 — Router: renormalization and precision `open` — needed for M2

`register_moe_topk_softmax_mi300_task` emits `renormalize = true`
(`task_register.cc:3689`); this model has `norm_topk_prob = false`, so the
stock routing produces wrong expert weights. The demo also feeds the top-k
kernel BF16 logits from the `linear` task, while the reference router is
FP32 end to end (`../deepseek-v2-lite/03-moe.md`); near-tie selections can
flip and break the exact-match requirement on boundary B9. The kernel also
zeroes the logits after reading them (`moe_topk_softmax_mi300.cuh:116-122`),
so boundary B8 needs a copy.

**Resolution path.** One routing variant: `renormalize=false`, the two
forced shared-expert entries from Q4, and (if the router GEMV is kept in
BF16) a check on how often the BF16 and FP32 selections differ over the 32
reference steps, done locally (`OPEN-PROBLEMS.md` MIN-6 already logs the
routing). If they ever differ, add an FP32-output router GEMV; it is 64 dot
products of length 2048.
