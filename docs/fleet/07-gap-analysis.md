# 07 — Gap Analysis

## Correction to the pre-reading assumption

Before reading, I assumed Fleet was dense-only and that **MoE dynamism would be
the research problem**. That was wrong, and it came from conflating the paper's
*evaluation* scope with the codebase's *implementation* scope.

- The paper evaluates exactly one model: Qwen3-8B, dense, on MI350. Its stated
  limitations say so.
- The **code** ships a full MoE path for MI300: `gang_moe_linear_mi300.cuh`,
  `moe_linear_mi300.cuh`, `moe_topk_softmax_mi300.cuh`,
  `moe_mul_sum_add_mi300.cuh`, Python graph-builder methods, and a Qwen3-30B-A3B
  MoE demo.

Data-dependent expert dispatch is handled *inside* a task (routing and mask
pointers are task inputs), so the task graph stays structurally static. No new
task-model capability is required.

**The real gap is MLA.**

## What exists, what does not

| Component | Status | Evidence |
|---|---|---|
| Persistent-kernel runtime | yes exists, MI300 guards throughout | `persistent_kernel.cuh` |
| Chiplet-task scheduling | yes exists, parameterized by X/W/C | paper §8, `gang.py` |
| Hierarchical sync | yes exists, MI300-specific path | `mpk_atoms.cuh` |
| gfx942 build target | yes via `AMDGPU_TARGETS=gfx942` | `CMakeLists.txt:50` |
| RMSNorm, RoPE, SiLU, argmax | yes `*_mi300.cuh` | `tasks/mi300/` |
| Dense linear (N-split, K-split) | yes `gang_linear`, `gang_splitk_linear` | `tasks/mi300/` |
| MoE routing + expert GEMM + combine | yes four kernels + Python | `tasks/mi300/`, `persistent_kernel.py:1203+` |
| KV-cache update | yes but GQA-shaped | `kv_cache_update_mi300.cuh` |
| Attention | yes **paged/GQA only** | `*paged_attention*_mi300.cuh` |
| **MLA attention** | no **absent** | no match for `mla`/`kv_lora`/`latent` |
| **DeepSeek-V2 model builder** | no **absent** | `models/` contains only `qwen3` |
| Latent KV-cache layout | no absent | cache is K/V-shaped |

## What we must build

| # | Item | Why | Est. difficulty |
|---|---|---|---|
| G1 | **MLA decode Chiplet-task** — scores against latent `c_KV` + decoupled `k_pe`, split-KV over the 1,024 context | No MLA kernel exists; GQA paged attention has the wrong shapes | **High** — the core of the work |
| G2 | **Split-KV partial-softmax merge** for MLA | `gang_attention_merge_mi300` exists but is shaped for GQA | Medium |
| G3 | **Latent KV-cache buffer + append task** | Cache is `[heads, S, head_dim]`; we need `c_KV[S,512]` + `k_pe[S,64]` | Medium |
| G4 | **Runtime reassociation of `W_UK`/`W_UV`** inside the MLA task | Per `../deepseek-v2-lite/05-weights.md`, materialized fusion is 2× worse | Medium — folded into G1 |
| G5 | **DeepSeek-V2 model builder** in `python/mirage/mpk/models/` | Only `qwen3` exists | Medium — clone and adapt |
| G6 | **Weight loader**: fuse `gate_proj`+`up_proj` per expert into W13 `[64,2816,2048]` | Fleet's MoE kernel expects fused W13 | Low |
| G7 | **Prefill → latent cache conversion**, excluded from timing | Task requires it; HF caches decompressed K/V | Low–Medium |
| G8 | Config plumbing for `X=8, W=37, C=4 MB` on MI300X | Paper says runtime-queried; verify it reads 38 CUs not 32 | Low |

G1 is the project. Everything else is plumbing around it.

## Adaptation questions Fleet's evaluation never faced

1. **Top-6 experts vs 8 chiplets.** Round-robin expert→XCD leaves 2 of 8
   chiplets idle during the 99 MB routed-expert phase. Qwen3-30B-A3B has top-8,
   which divides evenly. Ours does not. (`06-our-task-graph.md`.)
2. **Very small tasks.** The paper warns: "in designs with very short executing
   tasks, scheduling overhead might become more significant." Our per-layer
   traffic is 158.5 MB against Qwen3-8B's 368 MB, our hidden size is 2048 vs
   4096, and our experts are 1408-wide. Our tasks are **smaller than anything
   Fleet measured**, so dispatch overhead is proportionally worse for us. This
   is the main risk to the whole thesis at our model size.
3. **A KV cache worth caching.** Our latent cache is 144 KB per XCD per layer
   under split-KV, comfortably L2-resident. Qwen3-8B's GQA cache is far larger.
   This is an opportunity Fleet's evaluation had no reason to exploit.
4. **Register pressure with MLA added.** §8 already reports 1 wave/SIMD for the
   dense task set. Adding an MLA task with a 512-wide latent accumulation could
   push the union higher and force spills. Watch from commit one.

## Where Fleet's batch-1 gain actually comes from, and what we should promise

Table 4 is unambiguous: at bs=1, L2 hit moves 16.4%→16.9% and HBM reads are
0.98×. **No bandwidth win.** The gain is task-count collapse (1,407→543,
2.6×) reducing scheduler→worker dispatch traffic.

For our design document, the honest framing is:

- Expected gain = elimination of ~800–1,000 kernel launches/token, plus reduced
  dispatch overhead from Chiplet-task batching. **Not** a bandwidth improvement.
- Our roofline (`../deepseek-v2-lite/07-roofline.md`) is unchanged by Fleet:
  931 µs theoretical floor, **1.15-1.35 ms realistic**. Note the paper's §7
  reference to HazyResearch achieving "78% of H100 memory bandwidth at bs=1"
  does **not** transfer — see `../acceleration/04-technique-ledger.md` for why
  78% of theoretical is near MI300X's hardware ceiling.
- The one place we might beat Fleet's own batch-1 result *in kind* is KV-cache
  L2 residency, which is a property of MLA, not of Fleet.

Promising a bandwidth win from Fleet at batch 1 would contradict the paper's own
measurements. Saying so explicitly is better evidence of understanding than a
optimistic projection would be.

## Recommended strategy

**Extend the existing runtime.** Reuse the scheduler, hierarchical sync, MoE
path, and MI300 task library; add MLA. Reimplementing a persistent-kernel
runtime from scratch in five days would consume the whole budget and reproduce
work that already exists and is already MI300-aware.

Fallback if the build does not come up on gfx942 within day 1: implement a
minimal persistent kernel of our own against `../mi300x/03-memory-model.md`,
targeting the single-layer milestone only, and document the runtime as a
blocked dependency. Decide by end of day 1, not day 3.
