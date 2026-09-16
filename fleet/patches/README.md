# Patches to the Fleet submodule

`gfx942.patch` is a unified diff against `repos/fleet-chiplet-megakernel` at
commit `51dce4f` (branch `amd_mi350`). `env/setup.sh` applies it before the
first build (skipped when `git apply --reverse --check` says it is already
in); apply by hand with

```
git -C repos/fleet-chiplet-megakernel apply --check fleet/patches/gfx942.patch
git -C repos/fleet-chiplet-megakernel apply fleet/patches/gfx942.patch
```

Item L13 of `docs/design-doc/10-local-work.md`; the problems are
`OPEN-PROBLEMS.md` MIN-27 and `docs/fleet/99-open-questions.md` Q12.

## Hunks

| File | Change | Why |
|---|---|---|
| `include/mirage/persistent_kernel/tasks/mi300/task_header.cuh` (line 34) | the include of `paged_attention_decode_minimal_mi300.cuh` is wrapped in `#if defined(__gfx950__)` | that file calls `__builtin_amdgcn_mfma_f32_16x16x32_f16` unguarded (`:27`), an MFMA that does not exist on gfx942, so the include fails the gfx942 compile; nothing else in `include/`, `src/` or `python/` references its symbols (`__mfma_qk`, `__mfma_pv`, `paged_attention_minimal_decode`), and no task is registered for it |
| `include/mirage/persistent_kernel/tasks/mi300/linear_ck_mi300.cuh` (`BlockGemmSmallM16Policy::GetWarpGemmMWarpNWarp`, line 73) | `WarpGemmMfmaBf16Bf16F32M16N16K32TransposedCDistribution` is kept under `#if defined(__gfx950__)`; gfx942 gets `WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution` with the same `MWarp = 1, NWarp = 4` | gfx942 has no 16x16x32 BF16 MFMA. This policy is the block GEMM of `GemmPipelineSmallTilePolicy` for every `MPerBlock = 16` tile (`:267`), so it is what all the M = 1 linears (`gang_linear*`, `gang_moe_*`, `linear`) run through |
| same file, `linear_kernel_ck` warp tile (line 331) | `WarpK` is 16 on gfx942 instead of 32 for the small tile | must match the warp GEMM above; the larger tiles already use K = 16 |
| `include/mirage/persistent_kernel/tasks/mi300/paged_attention_ck_fmha_split_kv_mi300.cuh` (`paged_attention_ck_fmha_split_kv_impl`, the `seqlen_q == 1` branch, line 594) | the call to `paged_attention_minimal_decode` is kept under `#if defined(__gfx950__)`; gfx942 sends the one-token step through `paged_attention_ck_fmha_prefill`, which handles `seqlen_q == 1` | the earlier claim that nothing references the hidden kernel's symbols was wrong for this wrapper, which the Qwen3 smoke graph instantiates through `gang_attention_mi300.cuh:131` (found on the VM, 2026-09-15: the JIT failed with "use of undeclared identifier"); the DeepSeek graph never takes this path, it runs `mla_attend` |
| `include/mirage/persistent_kernel/persistent_kernel.cuh` (`:1047` worker variant, `:1535-1536` scheduler variant) | the `[FWD_PASS]` print condition drops `fwd_pass_count < 10 || ... % 50 == 0` and `end_of_graph_count < 10 || ... % 100 == 0` | every iteration is printed so the 32 per-iteration times of a generation are all visible (`docs/design-doc/09-expected-performance.md`, D23); the format strings are unchanged for `measure.py` |
| `python/mirage/mpk/persistent_kernel.py` (`get_compile_command`, after the ROCm `flags` list, line 315) | `MPK_EXTRA_HIPCC_FLAGS` from the environment is appended to the hipcc flags | the generated kernel is built with `-O3` and no debug info, so a fault under rocgdb names the kernel but not the line; `MPK_EXTRA_HIPCC_FLAGS=-gline-tables-only` adds line tables without changing the code (`docs/gpu/02-validation/01-preparation.md`, P1) |

## What was checked and left alone

- `linear_ck_mi300.cuh:394` coherence value 18 (`sc1 nt`) under
  `MPK_NT_WEIGHT_LOADS`: the comment says gfx950 but the encoding is shared
  with gfx942 (`docs/mi300x/03-memory-model.md`); left as is, verified by the
  day-1 disassembly.
- `grep -rn "gfx950\|16x16x32\|mfma_f32_16x16x32"` over
  `include/mirage/persistent_kernel` hits only the two files patched above.
- The `WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution` name exists
  in `ck_tile/ops/gemm/warp/warp_gemm.hpp` at the CK commit Fleet pins
  (`d8ee107a`, line 238) and at `rocm-7.2.4` (line 201), and the patched
  `linear_ck_mi300.cuh` parses for gfx942 with it under ROCm 7.0's hipcc
  (`env/offline_gfx942/README.md`, 2026-09-14); the CK linear pipelines are
  not instantiated there, so their device code is first generated on the
  machine. Both patches were parsed by the real compiler, not only applied.

## `sched_xcd.patch` (MIN-25: the scheduler reads the queue of its own XCD)

One hunk in `include/mirage/persistent_kernel/persistent_kernel.cuh`,
`execute_scheduler`. The stock code sets `sched_id = blockIdx.x + offset` and
reads `sched_queues[sched_id]`, on the comment "scheduler_kernel block k runs
on XCD k". Workers signal `sched_queues[xcd_id]` (`get_rand_sched_id`). On
the first VM the dispatcher placed block k on XCD (k + 4) mod 8, on every
grid and every launch (`env/hw/20260915/summary.md` F2 to F4), so each
scheduler read a queue written from another XCD through a channel that has
no fence. With the patch a local scheduler on AMD, when `worker_xcd_map` is
allocated, takes `sched_id` from `HW_REG_XCC_ID`; the 8 scheduler blocks
cover the 8 XCDs once each (F3), so every queue keeps exactly one reader and
`[SCHED_XCD] sched_id=k xcd=k` holds by construction. `check_day1.sh` check 3
accepts any constant offset for the worker lines. Applied third, after
`new_tasks.patch`; the hunk is independent of both.

## `new_tasks.patch` (L6: the task-registration glue)

Applied after `gfx942.patch` on the same commit:

```
git -C repos/fleet-chiplet-megakernel apply fleet/patches/gfx942.patch
git -C repos/fleet-chiplet-megakernel apply fleet/patches/new_tasks.patch
cp fleet/tasks/mi300/*.cuh repos/fleet-chiplet-megakernel/include/mirage/persistent_kernel/tasks/mi300/
```

The kernels themselves stay in `fleet/tasks/mi300/` (see `fleet/tasks/README.md`);
the copy step puts them where `task_header.cuh` includes them. Both patches
were checked to apply in that order on a clean tree with `git apply --check`.

| File | Change |
|---|---|
| `include/mirage/persistent_kernel/runtime_header.h` | `TaskType` values 185-189 (`TASK_MLA_PREP_MI300`, `TASK_MLA_ATTEND_MI300`, `TASK_MLA_MERGE_UV_MI300`, `TASK_MOE_ROUTER_MI300`, `TASK_COPY_MI300`); `MAX_OUTPUTS_PER_TASK` 3 -> 5 (`mla_prep` has 4 outputs, the router 5) |
| `include/mirage/persistent_kernel/tasks/mi300/task_header.cuh` | includes of the five new `.cuh` files, appended after the merge include (not the CK block that `gfx942.patch` touches) |
| `include/mirage/kernel/task_register.h`, `src/kernel/task_register.cc` | five `register_*_mi300_task` functions emitting the kernel calls (template dims from the tensors, float params from their bit patterns, per-task pointer offsets from the imaps); `register_argmax_reduce_task` accepts a second param and then also passes `runtime_config.tokens + step + 1` |
| `src/kernel/graph.cc` | the name branches (`mla_prep_mi300` 5 in / 4 out, `mla_attend_mi300` 4 in / 1 or 2 out with `gang_task_tiles_per_xcd = params[3]`, `mla_merge_uv_mi300` 2 / 1 with tiles `params[2]`, `moe_router_mi300` 2 / 5, `copy_mi300` 1 / 1) |
| `src/kernel/runtime.cc` | the two gang types added to the `n_tile_count` list, to the `n_tile_start = bid.x * tiles_per_xcd` branch (with `TASK_GANG_ATTN_*`), to the name table, and to the two lists that route gang types into `_execute_gang_task` |
| `include/mirage/persistent_kernel/persistent_kernel.cuh` | `is_gang_task_type` (line 226 block; the `[FWD_PASS]` hunks of `gfx942.patch` are elsewhere) |
| `include/mirage/persistent_kernel/tasks/ampere/embedding.cuh` | agent-scope load of `tokens[step]` on AMD (variant 1) |
| `include/mirage/persistent_kernel/tasks/mi300/argmax_mi300.cuh` | optional `tokens_out` argument (variant 2) |
| `python/mirage/mpk/persistent_kernel.py` | `argmax_reduce_layer(..., output_to_tokens=False)` |
| `python/mirage/mpk/profiler_persistent.py` | the five names in the profiler map |

Not verified locally: the host C++ additions compile only against the
Mirage headers (no local build); the emitted kernel calls were checked
against the kernel signatures by hand and the kernels themselves by
`fleet/tasks/check_syntax.sh`.
