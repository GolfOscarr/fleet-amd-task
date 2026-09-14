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

---

## The Python API, read line by line (2026-09-14)

What `python/mirage/mpk/persistent_kernel.py`, `src/kernel/graph.cc`,
`src/kernel/task_register.cc` and `demo/qwen3/demo_30B_A3B.py` do, recorded
so the design's task graph is written in the API's own terms. Line numbers
refer to the submodule at `51dce4f`.

### How a graph is declared

A model is a straight-line list of `mpk.<op>_layer(...)` calls on a
`PersistentKernel` object. Each call builds one `TBGraph` with
`grid_dim`, lists its tensors with `tb_graph.new_input(tensor, imap,
forloop_dim, True)`, registers the op with `kn_graph.customized(...)`, and
names the task kernel with `kn_graph.register_task(tb_graph, "name",
params)` (`kernel.py:870`). There is no separate output declaration: the
op's `(num_inputs, num_outputs)` tuple in `graph.cc` splits the list.

`imap` is a 3-tuple: for grid axes x, y, z, which tensor dimension that axis
partitions, or -1 for replicated. `gang_linear_layer` (`persistent_kernel.py:1571`)
uses `weight (0,-1,-1)` (rows split 8 ways by bid.x) and `output (1,-1,-1)`
(columns split 8 ways). These maps are what `runtime.cc` uses to slice
events between consecutive ops (`03-runtime.md`, "The dependency model is a
chain"). Every op must read a tensor its predecessor wrote.

Tensors: `attach_input(torch_tensor, name)` wraps an existing PyTorch tensor
(row-major asserted, `:469-480`); `new_tensor(dims, dtype, name,
io_category="cuda_tensor")` allocates one (`:482-503`); `fuse_tensors` and
`shuffle_tensors` (`:505-519`) concatenate along dim 0, the latter
interleaving in groups (used for QKV in the demo). Names are the keys the
generated `kernel_0.cu` binds pointers by.

`generate_task_graph` (`demo_30B_A3B.py:816`) writes `task_graph_0.json`
and `kernel_0.cu`; `compile` (`persistent_kernel.py:2404`) invokes `hipcc`
with `--offload-arch=${AMDGPU_TARGETS:-gfx950}` (`:350-352`), loads the
`.so`, and calls `init_func` with the ten meta tensors in fixed order
(`:2591-2612`). `mpk()` (`:2622`) calls `launch_func` on the current
stream.

### The layer calls our graph will use

| Call (`persistent_kernel.py`) | Tasks | Kernel name registered | Notes |
|---|---|---|---|
| `embed_layer` `:521` | 1 | `embedding` | `input_source=0` reads `tokens[step]` in-kernel; `=1` reads `input_tokens` (`task_register.cc` embedding: `runtime_config.tokens + runtime_config.step[0]`) |
| `rmsnorm_layer` `:544` | `max_num_batched_tokens` = 1 | `rmsnorm` | eps hard-coded `1e-6f` (`task_register.cc:122`); matches `rms_norm_eps` |
| `gang_rmsnorm_layer` `:1553` | 8 | `gang_rmsnorm_mi300` | all 8 compute the same norm; exists to keep the next event XCD-local |
| `gang_linear_layer` `:1571` | 8 | `gang_linear_mi300` | params `[output_stride, tile_n, m_tiles, m_per_tile, total_tiles_per_xcd, n_tiles_per_xcd, wgm]`; `weight.dim(0) % 8 == 0` and `(dim0/8) % tile_n == 0` |
| `gang_linear_with_residual_layer` `:1670` | 8 | `gang_linear_res_mi300` | same, plus residual partitioned like the output |
| `gang_linear_silu_layer` `:1852` | 8 | `gang_linear_silu_mi300` | fused gate/up + SiLU; weight is `shuffle_tensors([gate, up], groups)` |
| `linear_layer` `:2045` | `grid_dim[0]` | `linear` | CU-task; weight rows split `grid_dim[0]` ways, must divide |
| `linear_with_residual_layer` `:2077` | `grid_dim[0]` | `linear_with_residual` | |
| `moe_topk_softmax_routing_layer` `:1203` | 1 | `moe_topk_softmax_mi300` | outputs `(topk_weight f32 [B,k], routing_indices i32 [E,B], mask i32 [E+1])`; **`renormalize=true` hard-coded** (`task_register.cc:3689`) |
| `gang_moe_w13_linear_layer` `:1311` | 8 | `gang_moe_w13_linear_mi300` | W13 `[E, 2I, H]`, `2I % 64 == 0`, `H % 256 == 0`; output `[B, k, 2I]` |
| `moe_silu_mul_layer` `:1262` | `B x k` | `moe_silu_mul` | |
| `gang_moe_w2_linear_layer` `:1361` | 8 | `gang_moe_w2_linear_mi300` | W2 `[E, H, I]`, `I % 128 == 0` |
| `moe_mul_sum_add_layer` `:1411` | `B x H/256` | `moe_mul_sum_add_mi300` | `out = residual + sum_k w_k * in_k`, weights FP32 |
| `argmax_partial_layer` `:2207`, `argmax_reduce_layer` `:2232` | `grid_dim[0]`, 1 | `argmax_partial`, `argmax_reduce` | vocab must divide `grid_dim[0]`; reduce writes `output_tokens` |
| `kv_cache_update_layer` `:937`, `gang_paged_attention_split_kv_layer` `:1064`, `gang_paged_attention_split_kv_merge_layer` `:1141` | 1 / 8 / 8 | paged GQA | **not usable** for MLA; shapes are `[pages, page, kv_heads, head_dim]` |

Missing for us, to be added with the same pattern: an MLA split-KV attend
(gang, 8 tasks), an MLA merge, a latent-cache append, an FP32 router, and a
routing variant with `renormalize=false`.

### Adding a task type: the eight places

Traced for `gang_moe_w13_linear_mi300`:

1. `TaskType` enum value in `runtime_header.h` (`:145`, `= 174`).
2. The kernel in `tasks/mi300/*.cuh`, included from `task_header.cuh`.
3. `register_<name>_task` in `task_register.h:142` / `task_register.cc:1133-1195`:
   reads shapes off the `TBGraph`, emits the C++ call as a string (template
   arguments and `task_desc->input_ptrs[i]`, `output_ptrs[i]`, and
   `tile_idx` for gang tasks), and calls `register_task_variant(TaskType,
   code)`.
4. The name branch in `graph.cc:825-831`: `task_config[op] =
   make_tuple(num_inputs, num_outputs, TaskType, variant_id)`, and for gang
   tasks `gang_task_tiles_per_xcd[op] = params[...]`.
5. `runtime.cc`: the gang list that sets `task_metadata.n_tile_count`
   (`:405-420`), the name table (`:1551`), and the two lists that route the
   type into `_execute_gang_task` rather than `_execute_task`
   (`:1569-1583`, `:1604-1619`).
6. `is_gang_task_type` in `persistent_kernel.cuh:226-236`.
7. The Python `<name>_layer` method.
8. `profiler_persistent.py:58` name map (cosmetic).

The generated `kernel_0.cu` is an if-chain over `(task_type, variant_id)`
(`runtime.cc:1562-1596`); each variant is one line of emitted code. So a
new task is a `.cuh` plus about 60 lines of glue across those files.

### What the MoE demo does (`demo_30B_A3B.py`)

- Runtime: `mode="offline"`, `num_workers, num_schedulers =
  get_configurations_from_gpu()`, one request, `max_num_batched_tokens=1`,
  `page_size=4096`, `max_num_pages=16`, `max_seq_length=512` default
  (`:272-300`, argparse `:42-70`). Ten meta tensors: `step`, `tokens
  [requests, max_seq_length] int64`, `input_tokens [B,1]`, `output_tokens
  [B,1]`, `num_new_tokens`, `prompt_lengths`, and four paged-KV buffers
  that exist even though we will not page.
- Weights: expert `gate_proj` and `up_proj` are concatenated along dim 0
  per expert and stacked to `[E, 2I, H]`; `down_proj` stacked to
  `[E, H, I]` (`:381-392`). `lm_head` is padded to 153,600 rows (`:227`).
  QKV is `shuffle_tensors([q, k, v], groups=kv_heads)`.
- Layer body, in order (`:494-760`): `rmsnorm` -> `gang_linear` (QKV) ->
  attention (three variants by env) -> `gang_linear_with_residual` (o_proj)
  -> `rmsnorm` -> `linear` (router, grid `(1,1,1)`, output **BF16**) ->
  `moe_topk_softmax_routing` -> `gang_moe_w13_linear` -> `moe_silu_mul` ->
  `gang_moe_w2_linear` -> `moe_mul_sum_add` (residual). Head (`:772-801`):
  `rmsnorm` -> `linear` (`lm_head`, grid 296) -> `argmax_partial` (296) ->
  `argmax_reduce` (1) into `output_tokens`.
- Env switches: `USE_GANG=1` selects the gang linears, `USE_GANG_MOE=1`
  the gang MoE (`:241-242`); `USE_NT_WEIGHTS=1` compiles with
  `-DMPK_NT_WEIGHT_LOADS` (`persistent_kernel.py:226`), which is the
  non-temporal weight-load experiment of `OPEN-PROBLEMS.md` MAJ-6 already
  wired as one environment variable.

### Facts that change our plan

**Prefill runs inside the kernel in offline mode.** `prepare_next_batch`
(offline, `persistent_kernel.cuh:380-536`) feeds a request whose `step <
prompt_length` up to `MPK_MAX_TOKENS_PER_REQUEST` = 8 tokens per iteration
(`:118-121`), so the demo prefills its prompt through the same graph, 8
tokens at a time, then decodes. `init_kernel` zeroes `step` at init
(`:275-278`), and the host owns the `step` and `tokens` tensors afterwards.
For our hand-over from an external prefill the kernel must start at the
prompt boundary: set `step`, fill `tokens[0..S)`, and hand the latent cache
over, then launch. Online mode (`:538-568`) reads `step` and
`new_token_nums` from the host and advances `step` itself; it is the closer
fit and avoids the page manager entirely. Decided in the design's
`05-prefill-interface.md`.

**The router is wrong for this model as shipped.** `renormalize` is
hard-coded `true` (`task_register.cc:3689`); DeepSeek-V2-Lite has
`norm_topk_prob=false`. And the demo's router logits are BF16 (the `linear`
task rounds its output; the top-k kernel upcasts on load,
`moe_topk_softmax_mi300.cuh:112`), whereas the reference computes the
router in FP32 end to end. Both change expert selection near ties and break
boundary B9 (`../deepseek-v2-lite/08-correctness.md`). The top-k kernel also
zeroes the logits after reading them (`:116-122`, "for split-k gate linear
compatibility"), so B8 must be captured before routing or from a copy. We
need a routing variant with `renormalize=false` and an FP32-output router
GEMV, both small.

**Shared experts can be two always-selected experts.** The shared-expert
MLP has intermediate width 2816 = 2 x 1408, the same as a routed expert.
Splitting its intermediate dimension in two and summing the two `down_proj`
partials is exact up to FP32 accumulation order (SwiGLU is elementwise per
intermediate unit; `down_proj` sums over it). So the layer can be built as
top-8-of-66 with experts 64 and 65 forced at weight 1.0: one W13 tensor
`[66, 2816, 2048]`, one W2 `[66, 2048, 1408]`, one `gang_moe_w13` and one
`gang_moe_w2` op, and `moe_mul_sum_add` with `k=8`. This removes the
separate shared-expert ops (which the chain model could not overlap
anyway), and it fills all 8 XCDs with one expert each under the
round-robin `ae_idx = xcd_id + expert_local_idx * 8` rule
(`gang_moe_linear_mi300.cuh:24`), which resolves the 6-of-8 idle-chiplet
concern (`99-open-questions.md` Q4, `OPEN-PROBLEMS.md` MIN-11). Cost: the
routing kernel must append the two forced entries; that is the same
variant as the `renormalize=false` change.

**The graph is a chain.** See `03-runtime.md`. The task list in
`06-our-task-graph.md` must be re-expressed as a sequence of ops in which
each op reads its predecessor's output; the "shared experts overlap
routing" edge is replaced by the forced-expert fusion above.

**The head's partition must divide the vocabulary.** 102,400 = 2^12 x 25;
296 does not divide it. `lm_head` as `gang_linear` with `tile_n=64` gives
12,800 rows per XCD = 200 tiles per XCD over 37 workers; or `linear` with
`grid_dim[0]` in {256, 320, 400, 512}. `argmax_partial` then takes the same
count.

---

## The task kernels, read line by line (2026-09-14)

### `gang_linear_mi300.cuh`: the Chiplet-task contract

`gang_linear_kernel<T, BATCH_SIZE=m_per_tile, REDUCTION_SIZE>(input,
weight_chunk, output_cols, num_active_tokens, tile_n, o_stride, m_tiles,
n_tiles, wgm, tile_idx)` (`:44-57`). The worker receives `tile_idx` from
the gang loop in `execute_worker` (`persistent_kernel.cuh:1103`: `t =
xcd_local_rank; t < n_tile_count; t += workers_on_xcd`), decodes it into
`(m_tile, n_tile)` with the HipKittens windowed order (`:63-76`), offsets
the three pointers (`:79-92`), and calls `linear_kernel_ck` on a
`[BATCH_SIZE x tile_n]` output tile (`:97-99`). Weight chunk and output
columns are already the XCD's slice because the Python side partitioned them
by `bid.x`. At batch 1: `m_tiles=1`, `BATCH_SIZE=1`, and a worker's unit
of work is one `tile_n=64`-row slice of the weight, K=2048 deep, 256 KiB of
BF16 read for 64 outputs.

Everything a NEW gang task needs is in this shape: a `tile_idx` decode, a
pointer offset, and an inner kernel. Our MLA attend task decodes `tile_idx
-> split` (4 per XCD at `P_split=32`) instead of `(m_tile, n_tile)`.

### `linear_kernel_ck`: the inner GEMV (`linear_ck_mi300.cuh:293` onward)

Tier selection by `BATCH_SIZE` (`:310-333`): at batch 1 the small tier,
`MPerBlock=16, NPerBlock=64, KPerBlock=256`, one warp in M and four in N,
warp tile `16x16x32`, MFMA `16x16` (`BlockGemmSmallM16Policy`, `:68-84`),
CK pipeline `GemmPipelineAGmemBGmemCRegV2` (`:355`) with a custom LDS policy
(`GemmPipelineSmallTilePolicy`, `:134-289`). The M tile is padded from 1 to
16: 15 of 16 MFMA rows are wasted, which is the "MFMA at M=1" point in
`../acceleration/03-kernel-craft.md`; the kernel is bandwidth-bound so the
waste is in ALU, not time. Pointers are promoted to SGPRs with
`readfirstlane` to avoid waterfall loops (`:49-61`). Weight loads become
non-temporal cache-streaming (`amd_buffer_coherence_enum(18)` = `sc1 nt`)
only under `MPK_NT_WEIGHT_LOADS` (`:393-405`); by default they are plain
loads. Outputs use `__builtin_nontemporal_store` (`:39-45`).

Prefetch depth is CK's pipeline's, not stated in this file; the
`../mi300x/07-achievable-bandwidth.md` requirement (4-8 loads in flight per
wave) has to be checked from the disassembly of this instantiation, not
assumed. Logged under MIN-22.

### `gang_moe_linear_mi300.cuh`: expert dispatch

`gang_moe_w13_linear_kernel` (`:50-65`): 16x64x256 tiles, one expert at a
time per XCD. `ae_idx = xcd_id + expert_local_idx * 8` (`:105`) with
`xcd_id` read from `HW_REG_XCC_ID` inside the task (`:36-44`), so the
expert an XCD serves depends on the hardware XCD, not on `bid.x`.
`expert_id = mask[ae_idx]` where `mask` is the compacted list of active
experts and `mask[NUM_EXPERTS]` their count, both written by the routing
kernel (`moe_topk_softmax_mi300.cuh:219-229`). `routing_indices[expert][row]
= k_idx + 1` (0 = not selected) tells the kernel which top-k slot to write
(`:188`). With 8 active experts, every XCD serves exactly one; with 6, XCDs
6 and 7 return at `:106`.

### `moe_topk_softmax_mi300.cuh`: routing

BF16 logits loaded and upcast (`:112`), softmax in FP32 over `NUM_EXPERTS`
(power of two asserted, `:85-86`), iterative top-k by argmax with the winner
blanked to -1e4 (`:148-204`), optional renormalization (`:206-212`), then a
compaction of active experts by `atomicAdd` on `mask[NUM_EXPERTS]`
(`:219-229`), which makes the order of `mask` nondeterministic across runs.
That order only affects which XCD serves which expert, not the result.

### `ck_tile/` and the CK FMHA path

`ck_tile` is **not vendored**: `deps/` holds only `rocblas`, so the CK
headers come from the ROCm installation and their version is the machine's.
`ck_tile/common.cuh`, `linear.cuh`, `attention.cuh` are thin wrappers
(warp-GEMM type aliases, a hand-written 16x16x16 MFMA GEMM, an attention
tile config with `KV_TILE_SIZE=32`). The real attention path is
`paged_attention_ck_fmha_split_kv_mi300.cuh`, which instantiates CK's
`BlockFmhaFwdSplitKVPipelineNWarpSShuffleQRKSVS` with a decode tile
`sequence<16, 128, 32, 128, 32, 128>` = `(kM0, kN0, kK0, kN1, kK1,
kQKHeaddim)`, one warp in M and four splitting the KV range, merged through
LDS (`:66-98`), and `static_assert`s its LDS need against the 57 KiB budget
(`:118-121`). The QK head dim is the last entry and the V head dim is
`kN1`; both are 128 here. CK upstream carries separate QK and V head
dimensions in this very shape, and newer CK releases add a 576/512
configuration for DeepSeek MLA decode. If the ROCm on the machine has it,
phase B of `../mla-decode/04-our-kernel-spec.md` becomes an instantiation
of this pipeline at `(kQKHeaddim=576, kN1=512)` behind a wrapper that maps
`tile_idx -> split` and reads our two contiguous arrays, and phase C is
CK's existing merge. That is the first thing to try on the machine;
`99-open-questions.md` Q11.

### gfx950-only code in the gfx942 build

- `paged_attention_decode_minimal_mi300.cuh:27` calls
  `__builtin_amdgcn_mfma_f32_16x16x32_f16`, a gfx950 instruction, in an
  unguarded non-template function. The file is included by
  `task_header.cuh:34` and never registered as a task, so it is dead code
  that will still fail to compile for gfx942 (clang rejects target-gated
  builtins in any function it compiles). Fix: drop the include or guard it.
- `linear_ck_mi300.cuh:73` and `:331` pick a `16x16x32` BF16 warp GEMM
  "(MI350 2x K)". Whether CK lowers that to two `16x16x16` MFMAs on gfx942
  or fails to instantiate is a build-time fact.
- `linear_ck_mi300.cuh:394` comments the coherence value 18 "for gfx950";
  the `sc1 nt` bit encoding is the same on gfx942 per
  `../mi300x/03-memory-model.md`, so this one is expected to be fine.

These are the first three known items under `OPEN-PROBLEMS.md` MAJ-1; they
are the reason the build is a question and not a formality.

### RMSNorm and argmax

`rms_norm_impl<T, BATCH_SIZE, HIDDEN_DIM>(in, w, out, eps)` with `eps`
emitted as `1e-6f` (`task_register.cc:122`), reduction in FP32
(`rmsnorm_mi300.cuh:131`). `argmax_partial` and `argmax_reduce` reduce
`(value, int64 index)` pairs through wave shuffles split into 32-bit halves
(`argmax_mi300.cuh:18-38`); the reduce task writes the winning index to
`output_tokens`, which `prepare_next_batch` copies into `tokens[step+1]`.
