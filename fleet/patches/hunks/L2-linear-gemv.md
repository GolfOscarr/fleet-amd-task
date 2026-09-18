# L2: the GEMV linear's task type, registration and dispatcher

The C++ the fork needs for `TASK_LINEAR_GEMV_MI300`, as blocks to add to
`fleet/patches/new_tasks.patch` by hand. Written on the laptop, where the
fork (`repos/fleet-chiplet-megakernel`) is absent, so the patch itself is
untouched: apply these on the patched fork, regenerate the patch with
`git diff` and reset the fork, the flow of `fleet/patches/README.md`.

The kernel is `fleet/tasks/mi300/linear_gemv_mi300.cuh` (L1, merged with
wave 1); its signature is

    linear_gemv_mi300_task_impl<T, K, NORM, RESIDUAL>(x, w_norm, W, residual,
                                                      out, rows, o_stride, eps)

and the argument order below is the one the offline unit already compiles
(`env/offline_gfx942/mk_tu.cu`, the `MK_GEMV` block: `nullptr` for the
absent `w_norm` and `residual`, the input indices shifted by the flags).
The Python half is `fleet/build_graph.py`'s `linear_gemv_layer` and the
plan flag `--gemv-linears` (`fleet/graph_plan.py`); the registration below
is what its `register_task("linear_gemv_mi300", [norm, residual, eps_bits])`
reaches.

Eight blocks, each with its file and the line it goes after (the last one
optional). They are independent; the order here is the patch's file order.

## 1. The task type

**File** `include/mirage/persistent_kernel/runtime_header.h`, in the
`TaskType` enum, after the line

    TASK_PREFETCH_MOE_MI300 = 194,     // regular, side operator: streams part of an active expert's weight (O8)

```cpp
  TASK_LINEAR_GEMV_MI300 = 195,      // regular: the GEMV linear of every dense projection at batch 1 (L1, L2)
```

195 is free (the free values are 195 to 197 below the fork's 198, and 200
to 229 between its 199 and 230). Nothing else in this header changes: the
task descriptor's 7 inputs and 6 outputs already hold this task's 4 and 1.

## 2. The name table (C++)

**File** `src/kernel/runtime.cc`, in `print_task_graph`, after

    task_type_to_name[TASK_PREFETCH_MOE_MI300] = "TASK_PREFETCH_MOE_MI300";

```cpp
  task_type_to_name[TASK_LINEAR_GEMV_MI300] = "TASK_LINEAR_GEMV_MI300";
```

The type is regular, so none of the gang lists, the `_execute_gang_task`
lists or the `expert_offset` list in this file changes.

## 3. The name table (Python profiler)

**File** `python/mirage/mpk/profiler_persistent.py`, in `event_name_list`,
after

    194: "TASK_PREFETCH_MOE_MI300",

```python
    195: "TASK_LINEAR_GEMV_MI300",
```

## 4. The kernel include

**File** `include/mirage/persistent_kernel/tasks/mi300/task_header.cuh`,
after

    #include "tasks/mi300/linear_norm_mi300.cuh"

```cpp
#include "tasks/mi300/linear_gemv_mi300.cuh"
```

(`mla_common_mi300.cuh`, which the kernel also needs, is included above
it already.) With the include in place `env/offline_gfx942/mk_tu.cu` no
longer needs its own `MK_GEMV` include; leave that file as it is all the
same, it is the standalone check.

## 5. The registration's declaration

**File** `include/mirage/kernel/task_register.h`, in the `TaskRegister`
class, after

    int register_linear_norm_mi300_task(threadblock::Graph const &bgraph,
                                        std::vector<int> const &params);   // O3: the per-tile linear with the norm prologue

```cpp
  int register_linear_gemv_mi300_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);   // L2: the GEMV linear of every dense projection
```

## 6. The registration

**File** `src/kernel/task_register.cc`, after the closing brace of
`int TaskRegister::register_linear_norm_mi300_task(...)` (the function
ending `return register_task_variant(TASK_LINEAR_NORM_MI300, code.to_string());`)
and before the comment of `register_prefetch_mi300_task`.

```cpp
// L2 of docs/gpu-experiments/04-kernels (the kernel is L1, linear_gemv_mi300.cuh).
// params: [norm, residual, eps_bits]; inputs x [1, K] (whole), w_norm [K] (norm only),
// W [N, K] (partitioned on dim 0 by the grid: the task's N / grid rows), residual [1, N]
// (residual only, partitioned on dim 1 like the output: the kernel indexes it by the task's
// row, that is by the task's columns); output out [1, N] (partitioned on dim 1). As in
// register_linear_norm_mi300_task the reduction size is x's dim 1 and the per-task row count
// the partitioned output's dim 1 (the stock registration's output_size); the stride is the
// whole output row, which for these [1, N] outputs is the stock output_stride.
int TaskRegister::register_linear_gemv_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 3);
  bool norm = params[0] != 0, residual = params[1] != 0;
  std::vector<tb::TBInputOp *> input_ops, output_ops;
  int num_inputs = 2 + (norm ? 1 : 0) + (residual ? 1 : 0), num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  split_io(bgraph, num_inputs, input_ops, output_ops);
  int in_w = norm ? 2 : 1;                                     // W's input slot
  int in_res = in_w + 1;                                       // the residual's, when there is one
  assert(output_ops[0]->output_tensors[0].num_dims == 2);
  int batch_size = output_ops[0]->output_tensors[0].dim[0];
  int rows = output_ops[0]->output_tensors[0].dim[1];          // this task's columns: its row count
  assert(input_ops[0]->dtensor.num_dims == 2);
  int reduction_size = input_ops[0]->dtensor.dim[1];           // K from x
  assert(batch_size == 1);
  if (norm) {
    assert(input_ops[1]->dtensor.dim[0] == reduction_size);                                   // w_norm [K]
  }
  assert(input_ops[in_w]->dtensor.num_dims == 2);
  assert(input_ops[in_w]->dtensor.dim[1] == reduction_size);                                  // W [N, K]
  assert(input_ops[in_w]->dtensor.dim[0] == output_ops[0]->dtensor.dim[1]);                   // N
  assert(input_ops[in_w]->input_map.x == 0);                                                  // rows by the grid
  if (residual) {
    assert(input_ops[in_res]->dtensor.num_dims == 2);
    assert(input_ops[in_res]->dtensor.dim[0] == batch_size);
    assert(input_ops[in_res]->dtensor.dim[1] == output_ops[0]->dtensor.dim[1]);               // residual [1, N]
    assert(input_ops[in_res]->input_map.x == 1);                                              // the task's columns
  }
  assert(output_ops[0]->input_map.x == 1);                                                    // columns by the grid
  assert(reduction_size % 512 == 0);                           // the kernel's 16-byte loads: K % (8 * WAVE)
  int output_stride = output_ops[0]->dtensor.dim[1];           // the whole row: o_stride
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::linear_gemv_mi300_task_impl<bfloat16, $, $, $>(",
         reduction_size, norm ? "true" : "false", residual ? "true" : "false");
  code.e("    task_desc->input_ptrs[0],");                     // x
  if (norm) {
    code.e("    task_desc->input_ptrs[1],");                   // w_norm
  } else {
    code.e("    nullptr,");
  }
  code.e("    task_desc->input_ptrs[$],", in_w);               // W (this task's rows)
  if (residual) {
    code.e("    task_desc->input_ptrs[$],", in_res);           // residual (this task's columns)
  } else {
    code.e("    nullptr,");
  }
  code.e("    task_desc->output_ptrs[0],");                    // out (this task's columns)
  code.e("    $, $,", rows, output_stride);                    // rows, o_stride
  code.e("    $);", float_literal(params[2]));                 // eps
  return register_task_variant(TASK_LINEAR_GEMV_MI300, code.to_string());
}
```

`split_io` and `float_literal` are the file's own helpers, already in the
anonymous namespace the patch adds above `register_mla_prep_mi300_task`.

The emitted call for the plan's three forms, to compare against the
`MK_GEMV` block of `env/offline_gfx942/mk_tu.cu`:

```cpp
    // qkva (the input norm; 96 tasks of 38 rows)
    kernel::linear_gemv_mi300_task_impl<bfloat16, 2048, true, false>(
        task_desc->input_ptrs[0],
        task_desc->input_ptrs[1],
        task_desc->input_ptrs[2],
        nullptr,
        task_desc->output_ptrs[0],
        38, 3648,
        1.000000000e-06f);
    // o_proj (the residual; 64 tasks of 32 rows; layer 0's down, K 11,264, is not this kernel's: it stays the stock per-tile linear)
    kernel::linear_gemv_mi300_task_impl<bfloat16, 2048, false, true>(
        task_desc->input_ptrs[0],
        nullptr,
        task_desc->input_ptrs[1],
        task_desc->input_ptrs[2],
        task_desc->output_ptrs[0],
        32, 2048,
        0.000000000e+00f);
    // lm_head (the final norm; 400 tasks of 256 rows)
    kernel::linear_gemv_mi300_task_impl<bfloat16, 2048, true, false>(
        task_desc->input_ptrs[0],
        task_desc->input_ptrs[1],
        task_desc->input_ptrs[2],
        nullptr,
        task_desc->output_ptrs[0],
        256, 102400,
        1.000000000e-06f);
```

## 7. The dispatcher branch

**File** `src/kernel/graph.cc`, in `Graph::register_task`, after the
`linear_norm_mi300` branch, that is after

    } else if (name == "linear_norm_mi300") {
      // O3 (docs/gpu-experiments/03-acceleration): the stock per-tile linear with the input norm
      // in its prologue; inputs x, w_norm, W; outputs out, scratch [grid, K]; params [eps_bits]
      assert(params.size() == 1);
      int variant_id = task_register->register_linear_norm_mi300_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(3, 2, TASK_LINEAR_NORM_MI300, variant_id);

```cpp
  } else if (name == "linear_gemv_mi300") {
    // L2 (docs/gpu-experiments/04-kernels): one GEMV task type for every dense linear at batch 1;
    // inputs x, w_norm (norm), W, residual (residual); output out; params [norm, residual, eps_bits]
    assert(params.size() == 3);
    int num_inputs = 2 + (params[0] != 0 ? 1 : 0) + (params[1] != 0 ? 1 : 0);
    int variant_id = task_register->register_linear_gemv_mi300_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(num_inputs, 1, TASK_LINEAR_GEMV_MI300, variant_id);
```

The generated `_execute_task` needs nothing by hand: the code generator
emits one branch per (type, variant) from `task_config`, and a regular
type is on none of the gang lists this file and `runtime.cc` keep.

## 8. Optional: the worker timing class (I1)

**File** `include/mirage/persistent_kernel/persistent_kernel.cuh`, in
`execute_worker`'s `MPK_ENABLE_TIMING` switch, the line

    case TASK_LINEAR_NORM_MI300: ours_cycles[6] += task_time; ours_count[6]++; break;

becomes

```cpp
        case TASK_LINEAR_NORM_MI300:
        case TASK_LINEAR_GEMV_MI300: ours_cycles[6] += task_time; ours_count[6]++; break;
```

so `--worker-timing`'s `lnorm=` column of `[TASK_TIME2]` counts the GEMV
tasks too (the two never appear in one graph: `--gemv-linears` replaces
the fused per-tile linear). Skip this block if the column should stay the
CK linear's alone.

## What to check after applying

- `bash env/offline_gfx942/run.sh`: the host step parses `graph.cc`,
  `runtime.cc` and `task_register.cc` against the fork's headers, and the
  device step compiles `mk_tu.cu` with `MK_GEMV` (the emitted calls
  against the kernel's signature).
- The fork shows zero dirty lines after `git diff > new_tasks.patch` and
  the reset (`fleet/patches/README.md`).
- The dry run whose recorded calls are the inputs and params this
  registration reads: `python fleet/build_graph.py --dry-run --gemv-linears`
  (298 operators and 6,593 tasks at 27 layers with the head, the counts of
  `--fuse-norm1 --tile-linears`), and `fleet/tests/test_graph_plan.py`.

## One deviation from the page

`05-local-preparation.md`'s L2 and `01-gemv-ideas.md`'s I1 both say the
residual is passed whole. It is not: the kernel reads
`residual + r_begin + lane`, a task-local row index like the one it stores
with (`linear_gemv_mi300.cuh`, the wave's residual read before the
prologue; its header says "residual [1, N]
BF16 (RESIDUAL only; the task's columns)", and so does the row in
`fleet/tasks/README.md`), so a whole residual would give every task but
`bid.x == 0` the first task's columns. The registration therefore asserts
`input_map.x == 1` on the residual and `linear_gemv_layer` passes it with
the imap `(1, -1, -1)`, which is what the stock gang residual linear does
too ("residual partitioned like the output", `docs/fleet/04-repo-map.md`).
