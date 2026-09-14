# 04 — Repository Map

`repos/fleet-chiplet-megakernel` @ `51dce4f` ("Fleet: chiplet-aware megakernel
scheduling for Qwen3-8B on AMD MI350"), 659 tracked files, 15 MB.

**It is the Mirage tree with Fleet added.** Paths are `python/mirage/…`,
`include/mirage/…`, and the install sets `MIRAGE_HOME`. The README points at
`git@github.com:AMD-RAD/fleet.git`, branch `amd_mi350`.

## Layout

| Path | Contents |
|---|---|
| `include/mirage/persistent_kernel/` | **The runtime.** `persistent_kernel.cuh` (scheduler/worker loop), `mpk_atoms.cuh` (cache/atomic primitives), `runtime_header.h`, `profiler.h` |
| `include/mirage/persistent_kernel/tasks/mi300/` | **31 gfx942 task kernels** — see below |
| `include/mirage/persistent_kernel/tasks/cute/hopper/` | NVIDIA Hopper equivalents |
| `python/mirage/mpk/` | Graph builder, task registration, `fleet.py`, `gang.py`, `model_registry.py` |
| `python/mirage/mpk/models/` | Only `qwen3` |
| `demo/` | `qwen3/` (incl. `demo_30B_A3B.py` — an **MoE** model), `qwen2.5/` |
| `cpp_examples/`, `tests/` | Smallest working graphs — our on-ramp |
| `CMakeLists.txt`, `cmake/`, `config.cmake` | Build |
| `docker/`, `conda/`, `INSTALL.md` | Environment |

## MI300 task kernels — 31 files, already written

```
argmax_mi300.cuh                       moe_linear_mi300.cuh
gang_attention_mi300.cuh               moe_mul_sum_add_mi300.cuh
gang_attention_merge_mi300.cuh         moe_topk_softmax_mi300.cuh
gang_ksplit_linear_mi300.cuh           multitoken_paged_attention_*_mi300.cuh (7)
gang_linear_mi300.cuh                  norm_mi300.cuh
gang_moe_linear_mi300.cuh              paged_attention_ck_fmha_split_kv_mi300.cuh
gang_rmsnorm_linear_mi300.cuh          paged_attention_decode_minimal_mi300.cuh
gang_splitk_linear_mi300.cuh           rmsnorm_mi300.cuh
gemm_handtuned_mi300.cuh               rotary_embedding_mi300.cuh
kv_cache_update_mi300.cuh              silu_mul_linear_mi300.cuh
linear_ck_mi300.cuh                    silu_mul_mi300.cuh
linear_mi300.cuh                       task_header.cuh
ck_tile/                               (CK tile library integration)
```

Directly reusable for DeepSeek-V2-Lite: `rmsnorm_mi300`, `rotary_embedding_mi300`,
`gang_linear_mi300` (dense projections), `silu_mul_*`, `argmax_mi300` (our
`lm_head` greedy step), the whole MoE set, and `gang_splitk_linear` /
`gang_ksplit_linear` for reduction-partitioned GEMMs.

**Missing: anything MLA.** Verified by case-insensitive search over `include/`,
`python/`, `demo/`, `src/`:

- `grep -rlwi "mla"` → **no matches** (an unanchored `mla` search returns only
  false positives from `memLayout`).
- `grep -rli "deepseek\|kv_lora"` → **one match**, and it is an attribution
  comment in `demo/qwen3/models/convert.py`: "Based on an implementation from
  DeepSeek-V3 … /inference/convert.py". A borrowed weight-conversion script, not
  MLA support.

The attention kernels are paged-attention variants built for GQA.

## The MoE task, concretely

`gang_moe_linear_mi300.cuh` header comment:

> "Processes one expert at a time per XCD. All ~30 workers on an XCD cooperate to
> tile one expert's GEMM, then move to the next expert. This eliminates L2
> thrashing caused by concurrent expert weight loads on the same XCD.
> Dispatch: 8 gang tasks (1 per XCD), each with flattened `tile_idx` encoding:
> `tile_idx = expert_local_idx * TILES_PER_EXPERT + tile_within_expert`;
> `ae_idx = xcd_id + expert_local_idx * 8` (round-robin XCD assignment)"

Signature: `gang_moe_w13_linear_kernel(input_ptr, weight_ptr, routing_ptr,
mask_ptr, output_ptr, tile_idx)` with template parameters `BATCH_SIZE,
OUTPUT_SIZE, OUTPUT_STRIDE, REDUCTION_SIZE, NUM_EXPERTS, NUM_TOPK,
TILES_PER_EXPERT, N_TILES`. Weight layout expected:
**`[num_experts, N=2*intermediate, K=hidden_size]`** — i.e. gate and up **fused
into one W13 tensor**.

Two direct consequences for us:

1. **Experts are assigned to XCDs round-robin by index** (`ae_idx = xcd_id +
   expert_local_idx * 8`). With top-6 of 64 and 8 XCDs, six of eight chiplets get
   one expert each and **two sit idle** for the routed-expert phase. Worth
   checking whether that is what actually happens, and whether splitting 6
   experts across 8 XCDs would be better at our sizes.
2. **The fused W13 layout matches the packing plan** in
   `../deepseek-v2-lite/05-weights.md`: we must concatenate `gate_proj` and
   `up_proj` per expert at load time into `[64, 2816, 2048]`.

## Build and gfx942

```cmake
# CMakeLists.txt:50-56
if (DEFINED ENV{AMDGPU_TARGETS} AND NOT "$ENV{AMDGPU_TARGETS}" STREQUAL "")
  set(CMAKE_HIP_ARCHITECTURES "$ENV{AMDGPU_TARGETS}" ...)
else()
  set(CMAKE_HIP_ARCHITECTURES "gfx950" ...)
endif()
```

So `AMDGPU_TARGETS=gfx942` selects our target. Python honours the same variable
(`kernel.py:156`, `persistent_kernel.py:350`), and there is explicit MI300
dispatch logic ("MI300 (gfx94x): use 94 as target_cc for MPK task selection").

Stated requirements: ROCm 7.0+, PyTorch 2.4+ (ROCm), Python 3.8+, CMake 3.24+.

Config headers already reason about MI300:
```c
// include/mirage/config.h:61
// AMD MI300 (gfx942) has 192KB shared memory per CU, but we use conservative 64KB
// include/mirage/persistent_kernel/runtime_header.h:38
// AMD MI300 (gfx942) has up to 64KB LDS per workgroup by default
```

The 64 KB figure matches the CDNA3 ISA (`../mi300x/01-architecture.md`). The
"192KB shared memory per CU" claim does **not** appear in the ISA or the ROCm
docs we read and looks wrong for CDNA3; it is inside a comment justifying a
conservative choice, so it is harmless, but it is a reminder that **comments in
this repo are not always accurate**. Another example, in `mpk_atoms.cuh`:
"On MI300X, all CUs within an XCD share the same 32MB L2 cache" — it is 4 MB per
XCD, 32 MB aggregate. Verify claims against the ISA, not the comments.

## Reading order for day 1

1. `README.md`, `INSTALL.md` — build path
2. `include/mirage/persistent_kernel/persistent_kernel.cuh` — the loop
3. `include/mirage/persistent_kernel/mpk_atoms.cuh` — the primitives
4. `include/mirage/persistent_kernel/tasks/mi300/gang_linear_mi300.cuh` — the
   simplest Chiplet-task, the template for our MLA kernels
5. `python/mirage/mpk/models/qwen3/` — how a model becomes a task graph; this is
   what we clone for DeepSeek-V2
6. `demo/qwen3/demo_30B_A3B.py` — the MoE path end to end
