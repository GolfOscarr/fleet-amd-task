# New task kernels (L6)

The four kernels of `docs/design-doc/02-task-graph.md` ("The new tasks"), the
debug copy task, and the two variants of shipped tasks. Math and rounding
points are `harness/numpy_ref.py`; pointer orders and parameter vectors are
`fleet/build_graph.py`; the glue that registers them in the runtime is
`fleet/patches/new_tasks.patch` (applied after `gfx942.patch`).

`env/setup.sh` copies `fleet/tasks/mi300/*.cuh` into
`repos/fleet-chiplet-megakernel/include/mirage/persistent_kernel/tasks/mi300/`
before the build; `task_header.cuh` includes them from there.

`check_syntax.sh` parses every kernel and the test launcher with the host
`clang++` against the stub headers in `stub/` (a syntax and type check only;
no HIP, no device code). Correctness is established on the GPU by
`kernel_tests.py` against `numpy_ref.py` (below).

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
| `mla_prep_mi300.cuh` | `TASK_MLA_PREP_MI300` (185), CU-task | 16 (one per head; task 0 writes the cache rows; the head from `expert_offset`) | `qkva [1,3648]`, `w_kv_norm [512]`, `W_uk [16,128,512]`, `cos [S_max,64]`, `sin [S_max,64]` | `c_kv [S_max,512]` (row `step`), `k_pe [S_max,64]` (row `step`), `ql_nope [16,512]`, `q_pe [16,64]` | `[nh, d_n, d_r, d_c]` |
| `mla_attend_mi300.cuh` | `TASK_MLA_ATTEND_MI300` (186), gang | 8 x `tiles_per_xcd` | `ql_nope`, `q_pe`, `c_kv`, `k_pe` | `partials [n_splits,16,513]` FP32; optional second output: debug scores `[16,S_max]` FP32 | `[softmax_scale_bits, split, n_splits, tiles_per_xcd, nh, d_c, d_r]` |
| `mla_attend_mfma_mi300.cuh` (build flag `-DMLA_ATTEND_MFMA`, selected from `mla_attend_mi300.cuh`; `--mfma-attend`, O7 of `docs/gpu-experiments/03-acceleration`) | the same task types as the VALU kernel | the same | the same | the same | the same; the scores and p x V on `v_mfma_f32_16x16x16_bf16`, the tile staged once in LDS |
| `mla_merge_uv_mi300.cuh` | `TASK_MLA_MERGE_UV_MI300` (187), gang | 8 x `heads_per_xcd` | `partials`, `W_uv [16,128,512]` | `attn [1,2048]` | `[split, n_splits, tiles_per_xcd, nh, d_v, d_c]` |
| `moe_router_mi300.cuh` | `TASK_MOE_ROUTER_MI300` (188), CU-task | 1 | `h [1,2048]`, `W_gate [64,2048]` | `topk_w [1,8]` FP32, `routing [66,1]` int32, `mask [67]` int32, `logits [1,64]` FP32, `route_log [32,26,8]` int32 | `[topk, n_experts, n_forced, scaling_bits, layer_index, hidden]` |
| same file, `NORM = true` (registration `moe_router_norm_mi300`; `--fuse-norm2`, O1 of `docs/gpu-experiments/03-acceleration`) | `TASK_MOE_ROUTER_MI300` (188), CU-task | 1 | `x_res [1,2048]`, `w_norm [2048]`, `W_gate [64,2048]` | `h [1,2048]` (the normalised row, for the expert gate-up), then the five above | the six above and `eps_bits` |
| `gang_moe_w2_silu_mi300.cuh` | `TASK_GANG_MOE_W2_SILU_MI300` (191), gang (registration `gang_moe_w2_silu_linear_mi300`; `--fuse-silu`, O2 of `docs/gpu-experiments/03-acceleration`) | 8 x 32 tiles | `mid [1,8,2816]` (gate then up per slot), `W2 [66,2048,1408]`, `routing`, `mask` | `out8 [1,8,2048]`, `w2_scratch [256,1408]` (one activation row per (XCD, tile)) | the stock w2's `[tiles_per_expert, max_experts_per_xcd, total_tiles_per_xcd]`; K from the weight |
| `copy_mi300.cuh` | `TASK_COPY_MI300` (189), CU-task | 1 | `x [1,N]` | `y [1,N]` | `[N]` |
| `prefetch_mi300.cuh` | `TASK_PREFETCH_MI300` (193), regular, a side operator (registration `prefetch_mi300`; `--prefetch`, O8 of `docs/gpu-experiments/03-acceleration`) | `grid_for_linear(N)` stripes | `W [N,K]` (the task's `N / grid` rows) | `dummy [grid,4]` int32 (the task's row: one XOR word per wave, so the loads are not elided) | none |
| same file, `prefetch_moe_mi300_task_impl` | `TASK_PREFETCH_MOE_MI300` (194), regular, a side operator (registration `prefetch_moe_mi300`) | `8 x parts` | `W [E,N,K]` (whole), `mask [E+1]` | `dummy [8 x parts,4]` int32 | `[parts]`; the task's (slot, part) is its `bid.x` through the `expert_offset` metadata |
| `linear_gemv_mi300.cuh` | `TASK_LINEAR_GEMV_MI300` (195), regular (registration `linear_gemv_mi300`; L1 of `docs/gpu-experiments/04-kernels`, whose L2 adds the type and the registration to `new_tasks.patch`) | `grid_for_linear(N)` tasks (96 for `qkva`, 64 for `o_proj`, 400 for `lm_head`) | `x [1,2048]` (whole), `w_norm [2048]` (NORM), `W [N,2048]` (the task's `N / grid` rows), `residual [1,N]` (RESIDUAL, the task's columns) | `out [1,N]` (the task's columns) | `[norm, residual, eps_bits]`; the output size and stride as the stock per-tile `linear` |
| `linear_norm_mi300.cuh` | `TASK_LINEAR_NORM_MI300` (192), regular (registration `linear_norm_mi300`; `--fuse-norm1`, O3 of `docs/gpu-experiments/03-acceleration`) | `grid_for_linear(N)` tasks (96 for `qkva`, 400 for `lm_head`) | `x [1,2048]` (whole), `w_norm [2048]`, `W [N,2048]` (the task's `N / grid` rows) | `out [1,N]` (the task's columns), `scratch [grid,2048]` (the task's normalised row) | `[eps_bits]`; the output size and stride as the stock per-tile `linear` |

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
the scaled scores (boundary B5): `harness/run_fleet.py --debug-scores`
sets `MPK_DEBUG_SCORES=1` (the glue patch turns it into the define at
compile time) and `build_graph.py` passes the `scores [16, S_max]` FP32
tensor as the second output; `run_fleet.py` dumps it as `L{l}.B5.scores`.

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

### Side operators (in `new_tasks.patch`, O8)

An operator registered as `prefetch_mi300` or `prefetch_moe_mi300` is a
*side operator* (`Graph::side_ops`): `register_mugraph` appends its tasks
right behind the tasks of the operator registered before it (its host),
extends the host's last launch event range over them so the end-of-loop
pass gives them the host's dependent event (or makes them first tasks when
the host is the graph's first operator), points their trigger at the
end-of-graph event with `num_triggers` raised by their count, and leaves
`pre_op` on the host, so the next operator still chains to the host. In
the worker queues (task index modulo the worker count, FIFO per worker) the
side tasks land on workers holding no host task and run when the host's
event fires, concurrently with the host; after a gang host they run after
each worker's tile share. `fleet/task_graph_check.py` verifies the wiring
in a build's `task_graph_rank0.json`.

### Variants (in `new_tasks.patch`)

- `embedding`, `input_source = 0`: the load of `tokens[step]` is an
  agent-scope atomic load (`__hip_atomic_load(..., __ATOMIC_RELAXED,
  __HIP_MEMORY_SCOPE_AGENT)`) on AMD, since the first task of an iteration
  runs no acquire and the id was written on another XCD
  (`03-synchronization.md`).
- `copy_mi300`: `params [n, spin, print]` (I2 of `docs/gpu-experiments/03-acceleration`)
  makes thread 0 run `spin` iterations of a dependent integer chain after the
  copy and, with `print`, report `[SPIN] block=.. iters=.. cycles=.. ticks=..`
  (the shader clock against the 100 MHz real-time clock; `measure.py` turns it
  into the SCLK); with `[n]` alone the task is the plain copy. With a grid above
  one the output is partitioned on dim 0 (the empty ladder of I3).
- `argmax_reduce`: `argmax_reduce_layer(..., output_to_tokens=True)` adds a
  second parameter; the registration then also passes
  `runtime_config.tokens + runtime_config.step[0] + 1`, and the kernel
  writes the winning id there as well as to `output_tokens`.

## `kernel_tests`: each kernel in isolation against `numpy_ref.py`

`kernel_tests_mi300.cu` wraps each of the five kernels in a `__global__`
function of 256 threads with the worker's dynamic LDS and calls the
`*_task_impl` as the registration's emitted call does: the same template
dims, `step` and `prompt_length` read from device memory like
`runtime_config.step[0]` and `prompt_length[0]`, float parameters from
their bit patterns, gang tiles as `tile_idx = bid.x * tiles_per_xcd + t`
on a grid of `(8, tiles_per_xcd)`, and the per-XCD pointer offsets the
imaps of `build_graph.py` imply (`partials` by `n_splits / 8` rows for
`mla_attend`; `W_uv` by 2 heads and `attn` by 256 columns for
`mla_merge_uv`). Build, from the repository root
(`FLEET=repos/fleet-chiplet-megakernel`; the defines are the ones
`persistent_kernel.py` passes on its ROCm path):

```
mkdir -p fleet/tasks/build && hipcc --offload-arch=gfx942 -O2 -std=c++17 \
  -D__HIP_PLATFORM_AMD__=1 -DMIRAGE_AMD_MI300 -DMIRAGE_BACKEND_USE_ROCM -DMPK_TARGET_CC=94 -DMODE_ONLINE \
  -I fleet -I $FLEET/include -I $FLEET/include/mirage/persistent_kernel \
  fleet/tasks/kernel_tests_mi300.cu -o fleet/tasks/build/kernel_tests
# the debug-scores variant: the same line with -DMLA_ATTEND_DEBUG_SCORES -o fleet/tasks/build/kernel_tests_debug
# the streaming-loads variant (O6): the same line with -DMLA_NT_STREAMS -o fleet/tasks/build/kernel_tests_nt
```

`kernel_tests.py` is the driver:

```
python fleet/tasks/kernel_tests.py [--n 100] [--seed 0] [--kernel NAME ...] [--dry-run]
```

It generates random inputs at the real shapes (unit-scale BF16
activations, weights at `1 / sqrt(fan_in)`, `step` in 1023..1054), writes
them as raw files, runs the binary once per test over all trial
directories, and compares every output with `numpy_ref.py` using
`compare.py`'s metrics. Outputs are pre-filled with a sentinel so entries
the kernel must leave alone (the other cache rows of `mla_prep`, the other
slots of `route_log`, the columns beyond `step` of the debug scores) are
checked too. Tests: the five kernels, the two prefetch tasks of O8
(`prefetch`, `prefetch_moe`: the dummy output is the XOR of every word of
the streamed slice per wave, so the stripe and the expert (slot, part)
indexing against `mask` are checked exactly, and a slot past the active
count must leave its row untouched), `mla_attend_scores` (the
`-DMLA_ATTEND_DEBUG_SCORES` build's second output, B5), and
`mla_attend_splits` (one split of 1056 rows versus 33 splits of 32,
through both the attend and the merge kernel), and the three forms of the
GEMV linear (L1: `linear_gemv`, `linear_gemv_norm` and `linear_gemv_res`,
one launch per grid of tasks, so the split of the weight's rows and the
output's columns over the tasks is checked with the arithmetic). Tolerances, argued in the
driver's header comment: bit-exact for the RoPE outputs, the router's
selection (derived from the kernel's own FP32 logits, so a near tie cannot
fail it), the copy and the untouched entries; `rel_err <= 1e-4` for FP32
accumulations; one BF16 ulp of each element (plus a noise floor of
`1e-5 max|ref|` for near-zero elements) and `rel_err <= 2e-3` for
BF16-stored accumulations; `5e-4` on the attention partials (exact BF16
products, FP32 summation order only) and `1e-2` for the
one-versus-33-splits comparison (the probabilities are rounded at
different running maxima). Results go to
`fleet/tasks/results/kernel_tests.json`; `--dry-run` runs the whole
pipeline with the references standing in for the binary and writes
`kernel_tests_dryrun.json` instead, which is gitignored
(`fleet/tests/test_kernel_tests.py`, which also checks the tensor and
parameter tables of the two files against each other).

## Deliberately left for the GPU

- The MFMA 16x16x16 version of `mla_attend` (phase B of
  `docs/mla-decode/04-our-kernel-spec.md`) is written (O7, 2026-09-17,
  `-DMLA_ATTEND_MFMA`) and its lane arithmetic tested by emulation
  (`fleet/tests/test_mfma_layout.py`), but the instruction has not run: the
  suite binary `kernel_tests_mfma` (and `_mfma_debug` for the scores) is
  the first VM row; the VALU version stays the correctness baseline and
  the fallback.
- The register budget of the megakernel after the union with these kernels
  (`-Rpass-analysis=kernel-resource-usage`, `OPEN-PROBLEMS.md` MAJ-4).
- The imap and event verification of day 2: that the runtime creates one
  event with 8 triggers between the gang tasks, and that the per-task
  pointer offsets are as the registrations assume (`02-task-graph.md`,
  "Every operator").
- Whether `MAX_OUTPUTS_PER_TASK = 5` (raised from 3 for `mla_prep` and the
  router) has any effect beyond the task-descriptor size.
- The `kernel_tests` run itself: the launcher and driver above are
  syntax-checked and dry-run here, but no kernel has executed
  (`07-correctness.md`).

## Timing the kernels standalone

The suite binary times a kernel's grid on request (round 2, 2026-09-16):

    KT_TIME=50 fleet/tasks/build/kernel_tests mla_attend <trial dir>      # 50 launches under hipEvents, mean us to stderr
    KT_TIME=50 KT_COLD=27 fleet/tasks/build/kernel_tests mla_attend <dir>  # the launches rotate over 27 copies of the cache (cold L2)

    KT_TIME=50 KT_COLD=4 fleet/tasks/build/kernel_tests linear_gemv <dir>   # the 96-task qkva grid over 4 copies of the 15 MB weight

A trial directory comes from `python fleet/tasks/kernel_tests.py --n 1 --kernel mla_attend --work-dir <dir> --keep`.
`KT_SPIN=1000 fleet/tasks/build/kernel_tests copy <dir>` adds a launch whose thread 0 spins 1,000 iterations and prints the
`[SPIN]` line (I2: the SCLK standalone, against the same line from inside the graph).
The other builds are timed the same way (`kernel_tests_nt` for O6, `kernel_tests_mfma` for O7); the suite runs against
one of them with `kernel_tests.py --bin fleet/tasks/build/kernel_tests_mfma --bin-debug fleet/tasks/build/kernel_tests_mfma_debug`.
On the MI300X the attention grid (8 x 5 tiles, step 1032) costs 34 us cold or warm and the merge grid 11.5 us,
against 146 to 215 us and 46 to 61 us inside the megakernel (`docs/gpu-experiments/02-validation/04-results.md`). The `kernels`
stage of `env/session/vm.sh` does not rebuild an existing binary: delete `fleet/tasks/build/kernel_tests*` first.
