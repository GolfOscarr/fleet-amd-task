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

## Q4 — Is the 6-of-64 expert → 8-XCD mapping wasteful? `open`

**Why.** `ae_idx = xcd_id + expert_local_idx * 8` leaves 2 of 8 chiplets idle
during the 99 MB routed-expert phase — 25% of the machine, on the layer's
largest traffic item.

**Check.** Instrument per-XCD busy cycles during tasks 15–16. Compare against an
N-split variant where all 8 XCDs share each expert.

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
(`:1436`) must show `k == m` for all eight schedulers, every launch. If it
ever does not, the fix is to index the scheduler queue by the scheduler's
discovered XCD rather than by block id, which is a small change in
`execute_scheduler` and `get_rand_sched_id`.
