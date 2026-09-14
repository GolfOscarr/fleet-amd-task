# 99 — Open Questions

## Q1 — Does the repo build and run on MI300X / gfx942? `open` — **day 1, blocking**

**Why.** Gates the entire strategy in `07-gap-analysis.md`. If it builds, we
extend and spend our days on MLA. If it does not, we write a minimal runtime and
aim only at the single-layer milestone.

**Check.** `AMDGPU_TARGETS=gfx942 pip install -e . -v`, then run the smallest
`cpp_examples`/`tests` graph, then `demo/qwen3/demo.py`. Requires ROCm 7.0+.

**Decide by end of day 1.** Do not let this drift.

---

## Q2 — Does the runtime correctly detect 38 CUs/XCD on MI300X? `open`

**Why.** The paper says X/W/C are runtime-queried, but every constant in the
code was written against MI350 (32 CUs/XCD). `MI300X_NUM_XCDS = 8` is
hard-coded; workers-per-XCD may be too.

**Check.** Instrument `worker_xcd_map` at startup and print the per-XCD worker
count. Expect 37 workers + 1 scheduler per XCD, 304 CUs total.

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

Inherited from `../mi300x/99-open-questions.md` Q4, now with a stronger prior:
Fleet relies on `__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent")` doing the
right thing. Still verify by disassembly on our ROCm version — a silent
regression here corrupts results intermittently.

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
