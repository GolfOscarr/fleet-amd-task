# New task kernels (L6)

The four kernels of `docs/design-doc/02-task-graph.md` ("The new tasks"), the
debug copy task, and the two variants of shipped tasks. Math and rounding
points are `harness/numpy_ref.py`; pointer orders and parameter vectors are
`fleet/build_graph.py`; the glue that registers them in the runtime is
`fleet/patches/new_tasks.patch` (applied after `gfx942.patch`).

`env/setup.sh` copies `fleet/tasks/mi300/*.cuh` into
`repos/fleet-chiplet-megakernel/include/mirage/persistent_kernel/tasks/mi300/`
before the build; `task_header.cuh` includes them from there.

`check_syntax.sh` parses every kernel with the host `clang++` against the
stub headers in `stub/` (a syntax and type check only; no HIP, no device
code). Correctness is established on the GPU by `harness/kernel_tests.py`
against `numpy_ref.py`.

## Worker contract

256 threads (four wavefronts of 64), dynamic LDS `extern __shared__ char smem[]`
with 57 KiB available, `void const *` / `void *` arguments, FP32
accumulation, BF16 storage. A gang task receives `tile_idx`; the runtime
sets `task_metadata.n_tile_start = bid.x * tiles_per_xcd` for the two new
gang types (the rule of `TASK_GANG_ATTN_*`, `runtime.cc`), so a tile decodes
its XCD slot as `xcd = tile_idx / tiles_per_xcd`. `step` reaches a task as
`runtime_config.step[0]` in the emitted call (as `embedding` does);
`tokens` as `runtime_config.tokens`; the prompt length as
`runtime_config.prompt_length[0]`.

Per-task pointers: when a tensor's imap partitions a dimension by `bid.x`,
the runtime hands each task a pointer already offset by
`dim / grid_dim.x * bid.x` rows (`runtime.cc`, per-task pointer
computation). The registrations read the imaps and pass the offsets or a
local/global flag to the kernel, so the kernels are correct for either
choice of imap in `build_graph.py`.

## Kernels

| File | Task type (enum) | Grid | Inputs, in order | Outputs, in order | Params (`register_task`) |
|---|---|---|---|---|---|
| `mla_prep_mi300.cuh` | `TASK_MLA_PREP_MI300` (185), CU-task | 1 | `qkva [1,3648]`, `w_kv_norm [512]`, `W_uk [16,128,512]`, `cos [S_max,64]`, `sin [S_max,64]` | `c_kv [S_max,512]` (row `step`), `k_pe [S_max,64]` (row `step`), `ql_nope [16,512]`, `q_pe [16,64]` | `[nh, d_n, d_r, d_c]` |
| `mla_attend_mi300.cuh` | `TASK_MLA_ATTEND_MI300` (186), gang | 8 x `tiles_per_xcd` | `ql_nope`, `q_pe`, `c_kv`, `k_pe` | `partials [n_splits,16,513]` FP32; optional second output: debug scores `[16,S_max]` FP32 | `[softmax_scale_bits, split, n_splits, tiles_per_xcd, nh, d_c, d_r]` |
| `mla_merge_uv_mi300.cuh` | `TASK_MLA_MERGE_UV_MI300` (187), gang | 8 x `heads_per_xcd` | `partials`, `W_uv [16,128,512]` | `attn [1,2048]` | `[split, n_splits, tiles_per_xcd, nh, d_v, d_c]` |
| `moe_router_mi300.cuh` | `TASK_MOE_ROUTER_MI300` (188), CU-task | 1 | `h [1,2048]`, `W_gate [64,2048]` | `topk_w [1,8]` FP32, `routing [66,1]` int32, `mask [67]` int32, `logits [1,64]` FP32, `route_log [32,26,8]` int32 | `[topk, n_experts, n_forced, scaling_bits, layer_index, hidden]` |
| `copy_mi300.cuh` | `TASK_COPY_MI300` (189), CU-task | 1 | `x [1,N]` | `y [1,N]` | `[N]` |

Float parameters travel as IEEE-754 bit patterns (`register_task` takes
ints); the registration turns them back into float literals in the emitted
call.

### `mla_prep`

`c_kv[step] = rmsnorm(c_raw)` with FP32 statistics, the normalized value
rounded to BF16, then the BF16 weight multiply (the reference's
`DeepseekV2RMSNorm` order). RoPE de-interleaves the (even, odd) pairs into
two halves and rounds after every operation, so `k_pe[step]` and `q_pe`
are bit-identical to the reference's `apply_rotary_pos_emb` (verified for
`numpy_ref` in `harness/tests/test_numpy_ref.py`). `ql_nope[h] = q_nope[h] @
W_uk[h]` with FP32 accumulation and a BF16 store; the 2 MiB `W_uk` read is
16 bytes per lane, one full 1 KiB row per wave per step of `n`. LDS about
16.5 KiB.

### `mla_attend`

`split = t * 8 + xcd`; `split >= n_splits` returns; a split beyond `step`
writes `o = 0, lse = -inf`. Scores in FP32 (`ql_nope . c_kv^T + q_pe .
k_pe^T`, times the scale), online softmax in FP32 with the probabilities
rounded to BF16 before the `p . c_kv` accumulation, `o = acc / l` and
`lse = m + ln l` stored per head at column 512: the CK split-KV convention
of the runtime's own merge (`tasks/ampere/merge_splitkv.cuh`). Passes of
32 rows with the running-max rescale; for `split = 32` this is
`numpy_ref.mla_attend` exactly, for larger splits it differs by FP32
reassociation. The accumulator (16 x 512 FP32) lives in registers, 32 per
thread; LDS about 22 KiB (`ql_nope`, `q_pe`, scores, probabilities). VALU
only. `-DMLA_ATTEND_DEBUG_SCORES` and a second output tensor make it write
the scaled scores (boundary B5).

### `mla_merge_uv`

Head `h = xcd * heads_per_xcd + t`; live splits `ceil((step + 1) / split)`;
`M = max lse_j`, `w_j = exp(lse_j - M)`, `o = sum w_j o_j / sum w_j`; `o`
rounded to BF16; `attn[h] = o @ W_uv[h]^T` with FP32 accumulation, two
lanes per output element. LDS about 2.3 KiB.

### `moe_router`

Four waves own 16 experts each, one 16-byte load per lane per 8 columns;
FP32 softmax and top-k on wave 0 with one expert per lane, ties to the
lower index; the forced experts `n_experts .. n_experts + n_forced - 1`
take the last slots at weight 1.0; `routing[e] = slot + 1`, `mask[slot] =
id`, `mask[66] = 8`, unused mask entries `-1` (as the stock kernel; the
consumers read only `mask[count]` and `mask[0..count)`); `route_log[step -
(prompt_len - 1)][layer_index][slot] = id`. LDS 256 B.

### Variants (in `new_tasks.patch`)

- `embedding`, `input_source = 0`: the load of `tokens[step]` is an
  agent-scope atomic load (`__hip_atomic_load(..., __ATOMIC_RELAXED,
  __HIP_MEMORY_SCOPE_AGENT)`) on AMD, since the first task of an iteration
  runs no acquire and the id was written on another XCD
  (`03-synchronization.md`).
- `argmax_reduce`: `argmax_reduce_layer(..., output_to_tokens=True)` adds a
  second parameter; the registration then also passes
  `runtime_config.tokens + runtime_config.step[0] + 1`, and the kernel
  writes the winning id there as well as to `output_tokens`.

## Deliberately left for the GPU

- The MFMA 16x16x16 version of `mla_attend` (phase B of
  `docs/mla-decode/04-our-kernel-spec.md`); the VALU version is the
  correctness baseline and the fallback.
- The register budget of the megakernel after the union with these kernels
  (`-Rpass-analysis=kernel-resource-usage`, `OPEN-PROBLEMS.md` MAJ-4).
- The imap and event verification of day 2: that the runtime creates one
  event with 8 triggers between the gang tasks, and that the per-task
  pointer offsets are as the registrations assume (`02-task-graph.md`,
  "Every operator").
- Whether `MAX_OUTPUTS_PER_TASK = 5` (raised from 3 for `mla_prep` and the
  router) has any effect beyond the task-descriptor size.
- `kernel_tests.py`: 100 random inputs per kernel against `numpy_ref.py`,
  and one split versus 33 for `mla_attend` (`07-correctness.md`).
