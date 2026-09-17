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

| File | Change |
|---|---|
| `include/mirage/persistent_kernel/runtime_header.h` | `TaskType` values 185 to 194 (`TASK_MLA_PREP_MI300`, `TASK_MLA_ATTEND_MI300`, `TASK_MLA_MERGE_UV_MI300`, `TASK_MOE_ROUTER_MI300`, `TASK_COPY_MI300`, `TASK_MLA_ATTEND_TILE_MI300` 190, `TASK_GANG_MOE_W2_SILU_MI300` 191, `TASK_LINEAR_NORM_MI300` 192, `TASK_PREFETCH_MI300` 193, `TASK_PREFETCH_MOE_MI300` 194); `MAX_OUTPUTS_PER_TASK` 3 -> 6 (`mla_prep` writes 4, the router 5, the fused router 6) |
| `include/mirage/persistent_kernel/tasks/mi300/task_header.cuh` | includes of the `.cuh` files of `fleet/tasks/mi300`, appended after the merge include (not the CK block that `gfx942.patch` touches) |
| `include/mirage/kernel/task_register.h`, `src/kernel/task_register.cc` | the `register_*_mi300_task` functions emitting the kernel calls (template dims from the tensors, float params from their bit patterns, per-task pointer offsets from the imaps): `mla_prep`, `mla_attend`, `mla_attend_tile`, `mla_merge_uv`, `moe_router` (with the `norm` flag of O1), `gang_moe_w2_silu` (O2), `linear_norm` (O3), `prefetch` and `prefetch_moe` (O8), `copy`; `register_argmax_reduce_task` accepts a second param and then also passes `runtime_config.tokens + step + 1` |
| `include/mirage/kernel/graph.h`, `src/kernel/graph.cc` | the name branches with their input and output counts and, for the gang types, `gang_task_tiles_per_xcd`; `Graph::side_ops`, the set the two prefetch registrations join (O8) |
| `src/kernel/runtime.cc` | the gang types (`mla_attend`, `mla_merge_uv`, `gang_moe_w2_silu`) in the `n_tile_count` list, in the `n_tile_start = bid.x * tiles_per_xcd` branch (with `TASK_GANG_ATTN_*`) and in the two lists that route gang types into `_execute_gang_task`; `mla_attend_tile` and `prefetch_moe` in the `expert_offset = bid` list; the name table; the side-operator branch of `register_mugraph` (O8: a side operator's tasks follow its host's, take the host's dependent event through the extended event range, trigger the end-of-graph event with `num_triggers` raised, and leave the chain on the host) |
| `include/mirage/persistent_kernel/persistent_kernel.cuh` | `is_gang_task_type` (line 226 block; the `[FWD_PASS]` hunks of `gfx942.patch` are elsewhere); the worker timing for every worker and our task classes under `MPK_ENABLE_TIMING` (I1); the fence knobs `MPK_NO_COMPLETION_FENCE`, `MPK_NO_ACQUIRE_FENCE`, `MPK_NO_BCAST_CAS`, `MPK_NO_LOCAL_CAS`, `MPK_POLL_SLEEP` (I4), all off unless defined |
| `include/mirage/persistent_kernel/tasks/ampere/embedding.cuh` | agent-scope load of `tokens[step]` on AMD (variant 1) |
| `include/mirage/persistent_kernel/tasks/mi300/argmax_mi300.cuh` | optional `tokens_out` argument (variant 2) |
| `python/mirage/mpk/persistent_kernel.py` | `argmax_reduce_layer(..., output_to_tokens=False)`; the `MPK_DEBUG_SCORES` flag |
| `python/mirage/mpk/profiler_persistent.py` | the names in the profiler map |

The host C++ additions are parsed and type-checked with the ROCm clang by
the host step of `env/offline_gfx942/run.sh` (`graph.cc`, `runtime.cc`,
`task_register.cc` against the fork's headers, since 2026-09-17); the
emitted kernel calls are checked against the kernel signatures by the
offline unit `env/offline_gfx942/mk_tu.cu` and the kernels by
`fleet/tasks/check_syntax.sh`.
