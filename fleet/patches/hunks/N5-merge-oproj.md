# N5: the merge with o_proj folded in, its task type, registration and dispatcher

The C++ the fork needs for `TASK_MLA_MERGE_OPROJ_MI300`, as blocks to add to
`fleet/patches/new_tasks.patch` by hand. Written on the laptop, where the
fork (`repos/fleet-chiplet-megakernel`) is absent, so the patch itself is
untouched: apply these on the patched fork, regenerate the patch with
`git diff` and reset the fork, the flow of `fleet/patches/README.md`. The
form is `fleet/patches/hunks/L2-linear-gemv.md`'s.

The kernel is `fleet/tasks/mi300/mla_merge_oproj_mi300.cuh` (N5, M5 of
`03-router-merge-ideas.md`), which calls the merge body of
`mla_merge_uv_mi300.cuh` and then multiplies its `attn` values by its slice
of `W_o`:

    mla_merge_oproj_mi300_task_impl<T, NH, D_V, D_C, HIDDEN, HALVES>(
        partials, w_uv, w_o, x_res, counter, attn, workspace,
        step, split, n_splits, idx)

`idx` is the task index (`h = idx / HALVES`, `half = idx % HALVES`), taken
from `expert_offset` as `mla_merge_uv_tile` and `mla_prep` take theirs;
every tensor is whole. `x_res` is read and written in place, so it is both
input 3 and output 0 of the operator (the same buffer either way, the
precedent being the stock residual linear, which the plan gives `x_res` as
its residual and its output). `counter` is a `[1]` int32 tensor of the
plan, zeroed at allocation, and `workspace` a `[NH * halves, HIDDEN]` FP32
buffer of partial vectors. The Python half is `fleet/build_graph.py`'s
`mla_merge_oproj_layer` and the plan flag `--merge-oproj`
(`fleet/graph_plan.py`); the registration below is what its
`register_task("mla_merge_oproj_mi300", [split, n_splits, halves])`
reaches. The models of the item are N4's registration (the whole-tensor
imaps and the task index) and N2's (the counter as a plain input).

Eight blocks, each with its file and the line it goes after. They are
independent; the order here is the patch's file order.

## 1. The task type

**File** `include/mirage/persistent_kernel/runtime_header.h`, in the
`TaskType` enum, after the line

    TASK_MLA_MERGE_UV_TILE_MI300 = 201,  // regular: one merge task per (head, half), the index from expert_offset (N4)

```cpp
  TASK_MLA_MERGE_OPROJ_MI300 = 207,    // regular: the merge with o_proj folded in, the last task summing the partials (N5)
```

202 is free (200 to 229 lie between the fork's 199 and 230; 200 is N2's
router and 201 N4's merge). Nothing else in this header changes: the
descriptor's 7 inputs and 6 outputs already hold this task's 5 and 3, the
largest of the round.

(The block stands on its own if N4's is not applied: the line it goes after
is then `TASK_LINEAR_GEMV_MI300 = 195,` and the added line is the same.)

## 2. The name table (C++)

**File** `src/kernel/runtime.cc`, in `print_task_graph`, after

    task_type_to_name[TASK_MLA_MERGE_UV_TILE_MI300] = "TASK_MLA_MERGE_UV_TILE_MI300";

```cpp
  task_type_to_name[TASK_MLA_MERGE_OPROJ_MI300] = "TASK_MLA_MERGE_OPROJ_MI300";
```

## 3. The `expert_offset` list

**File** `src/kernel/runtime.cc`, in the per-task loop that packs `bid.x`
into the low 16 bits of `expert_offset` and `bid.y` into the high 16. With
N4's and N2's blocks applied the condition reads

    if (task_type == TASK_PREFETCH_MOE_MI300 ||
        task_type == TASK_MLA_ATTEND_TILE_MI300 ||
        task_type == TASK_MLA_MERGE_UV_TILE_MI300 ||   // (head, half) of the regular merge task (N4)
        task_type == TASK_MOE_ROUTER4_MI300 ||         // the part of the four-task router (N2)
        task_type == TASK_MLA_PREP_MI300) {          // the head of the per-head prep task (round 3)

and becomes

```cpp
    if (task_type == TASK_PREFETCH_MOE_MI300 ||
        task_type == TASK_MLA_ATTEND_TILE_MI300 ||
        task_type == TASK_MLA_MERGE_UV_TILE_MI300 ||   // (head, half) of the regular merge task (N4)
        task_type == TASK_MLA_MERGE_OPROJ_MI300 ||     // (head, half) of the merge with o_proj folded in (N5)
        task_type == TASK_MOE_ROUTER4_MI300 ||         // the part of the four-task router (N2)
        task_type == TASK_MLA_PREP_MI300) {          // the head of the per-head prep task (round 3)
```

(The block stands on its own if the other two are not applied: the added
line is the `TASK_MLA_MERGE_OPROJ_MI300` one.) The type is regular, so none
of the gang lists, the `_execute_gang_task` lists or the MoE gang rule
changes.

## 4. The name table (Python profiler)

**File** `python/mirage/mpk/profiler_persistent.py`, in `event_name_list`,
after

    201: "TASK_MLA_MERGE_UV_TILE_MI300",

```python
    202: "TASK_MLA_MERGE_OPROJ_MI300",
```

## 5. The kernel include

**File** `include/mirage/persistent_kernel/tasks/mi300/task_header.cuh`,
after

    #include "tasks/mi300/mla_merge_uv_mi300.cuh"

```cpp
#include "tasks/mi300/mla_merge_oproj_mi300.cuh"
```

(It includes `mla_merge_uv_mi300.cuh` itself for the shared merge body, so
the order does not matter; `mla_common_mi300.cuh` is included above both
already.) With the include in place `env/offline_gfx942/mk_tu.cu` no longer
needs its own `MK_GEMV` include; leave that file as it is all the same, it
is the standalone check.

## 6. The registration's declaration

**File** `include/mirage/kernel/task_register.h`, in the `TaskRegister`
class, after

    int register_mla_merge_uv_tile_mi300_task(threadblock::Graph const &bgraph,
                                              std::vector<int> const &params);   // N4: one task per (head, half)

```cpp
  int register_mla_merge_oproj_mi300_task(threadblock::Graph const &bgraph,
                                          std::vector<int> const &params);   // N5: the merge with o_proj folded in
```

## 7. The registration

**File** `src/kernel/task_register.cc`, after the closing brace of
`int TaskRegister::register_mla_merge_uv_tile_mi300_task(...)` (the function
ending `return register_task_variant(TASK_MLA_MERGE_UV_TILE_MI300, code.to_string());`)
and before the comment of `register_moe_router_mi300_task`.

```cpp
// N5 of docs/gpu-experiments/04-kernels (M5 of 03-router-merge-ideas.md): the merge with o_proj
// folded in, nh * halves regular tasks. params: [split, n_splits, halves]; inputs partials
// [n_splits, nh, partials_row], W_uv [nh, d_v, d_c], W_o [hidden, hidden], x_res [1, hidden] and
// the counter [1] int32, all whole; outputs x_res (in place: the same tensor as input 3, as the
// stock residual linear takes its residual and its output), attn [1, nh * d_v] and the workspace
// [nh * halves, hidden] FP32, all whole. The task index reaches the kernel through expert_offset
// (bid.x), as it does for mla_merge_uv_tile and mla_prep, and the kernel takes h = idx / halves
// and half = idx % halves from it, so no imap is partitioned and there are no local flags.
int TaskRegister::register_mla_merge_oproj_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 3);
  int split = params[0], n_splits = params[1], halves = params[2];
  assert(n_splits <= 64 && "mla_merge_uv merges one split per lane of one wavefront");
  assert(halves == 1 || halves == 2);
  std::vector<tb::TBInputOp *> input_ops, output_ops;
  int num_inputs = 5, num_outputs = 3;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  split_io(bgraph, num_inputs, input_ops, output_ops);
  assert(input_ops[1]->dtensor.num_dims == 3);
  int nh = input_ops[1]->dtensor.dim[0];
  int d_v = input_ops[1]->dtensor.dim[1];
  int d_c = input_ops[1]->dtensor.dim[2];
  int hidden = nh * d_v;
  assert(input_ops[0]->dtensor.dim[0] == n_splits && input_ops[0]->dtensor.dim[1] == nh &&
         input_ops[0]->dtensor.dim[2] == ((d_c + 1 + 3) / 4) * 4);
  assert(input_ops[2]->dtensor.num_dims == 2 && input_ops[2]->dtensor.dim[0] == hidden &&
         input_ops[2]->dtensor.dim[1] == hidden);          // W_o [N, K], N = K = nh * d_v
  assert(input_ops[3]->dtensor.dim[1] == hidden);          // x_res [1, hidden]
  assert(input_ops[4]->dtensor.num_dims == 1 && input_ops[4]->dtensor.dim[0] == 1);   // counter [1]
  assert(output_ops[0]->dtensor.dim[1] == hidden);         // x_res again, written in place
  assert(output_ops[1]->dtensor.dim[1] == hidden);         // attn [1, nh * d_v]
  assert(output_ops[2]->dtensor.num_dims == 2 && output_ops[2]->dtensor.dim[0] == nh * halves &&
         output_ops[2]->dtensor.dim[1] == hidden);         // workspace [nh * halves, hidden] FP32
  for (int i = 0; i < num_inputs; i++) {
    assert(input_ops[i]->input_map.x == -1);               // every tensor whole
  }
  for (int i = 0; i < num_outputs; i++) {
    assert(output_ops[i]->input_map.x == -1);
  }
  assert((int)bgraph.grid_dim.x == nh * halves);           // one task per (head, half)
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::mla_merge_oproj_mi300_task_impl<bfloat16, $, $, $, $, $>(",
         nh, d_v, d_c, hidden, halves);
  code.e("    task_desc->input_ptrs[0],");   // partials (whole)
  code.e("    task_desc->input_ptrs[1],");   // W_uv (whole)
  code.e("    task_desc->input_ptrs[2],");   // W_o (whole)
  code.e("    task_desc->output_ptrs[0],");  // x_res, in place (input 3 is the same buffer)
  code.e("    task_desc->input_ptrs[4],");   // the arrival counter
  code.e("    task_desc->output_ptrs[1],");  // attn (boundary B6)
  code.e("    task_desc->output_ptrs[2],");  // the partial workspace
  code.e("    runtime_config.step[0],");
  code.e("    $,", split);
  code.e("    $,", n_splits);
  code.e("    (int)(task_desc->task_metadata.expert_offset & 0xFFFF));");   // bid.x: the (head, half) index
  return register_task_variant(TASK_MLA_MERGE_OPROJ_MI300, code.to_string());
}
```

`split_io` is the file's own helper, already in the anonymous namespace the
patch adds above `register_mla_prep_mi300_task`.

The emitted call for the model's dims, to compare against the
`MK_MERGE_OPROJ` lines of `env/offline_gfx942/mk_tu.cu`:

```cpp
    // --merge-oproj (32 tasks of a half head, the last one summing the partials)
    kernel::mla_merge_oproj_mi300_task_impl<bfloat16, 16, 128, 512, 2048, 2>(
        task_desc->input_ptrs[0],
        task_desc->input_ptrs[1],
        task_desc->input_ptrs[2],
        task_desc->output_ptrs[0],
        task_desc->input_ptrs[4],
        task_desc->output_ptrs[1],
        task_desc->output_ptrs[2],
        runtime_config.step[0],
        32,
        33,
        (int)(task_desc->task_metadata.expert_offset & 0xFFFF));
```

## 8. The dispatcher branch

**File** `src/kernel/graph.cc`, in `Graph::register_task`, after the
`mla_merge_uv_tile_mi300` branch, that is after

    } else if (name == "mla_merge_uv_tile_mi300") {
      assert(params.size() == 3);
      int variant_id = task_register->register_mla_merge_uv_tile_mi300_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(2, 1, TASK_MLA_MERGE_UV_TILE_MI300, variant_id);

```cpp
  } else if (name == "mla_merge_oproj_mi300") {
    // N5 (docs/gpu-experiments/04-kernels): the merge with o_proj folded in, nh * halves regular
    // tasks, every tensor whole; inputs partials, W_uv, W_o, x_res, counter; outputs x_res (in
    // place), attn, workspace; params [split, n_splits, halves]
    assert(params.size() == 3);
    int variant_id = task_register->register_mla_merge_oproj_mi300_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(5, 3, TASK_MLA_MERGE_OPROJ_MI300, variant_id);
```

The generated `_execute_task` needs nothing by hand: the code generator
emits one branch per (type, variant) from `task_config`, and a regular type
is on none of the gang lists this file and `runtime.cc` keep.

## What to check after applying

- `bash env/offline_gfx942/run.sh`: the host step parses `graph.cc`,
  `runtime.cc` and `task_register.cc` against the fork's headers, and the
  device step compiles `mk_tu.cu` with `MK_GEMV` (the `MK_TASK_MLA_MERGE_OPROJ_MI300`
  block rides on that define: the emitted call against the kernel's
  signature). The `k_mla_merge_oproj` register line is the one to read off
  the build: its three phases run in sequence, so the peak should be the
  merge's (at most 200), and the worker union's line must not move.
- The fork shows zero dirty lines after `git diff > new_tasks.patch` and the
  reset (`fleet/patches/README.md`).
- The dry runs whose recorded call is the inputs, imaps and params this
  registration reads: `python fleet/build_graph.py --dry-run --merge-oproj`
  (32 tasks per layer, one operator fewer) and
  `--merge-oproj --gemv-linears --gemv-w13 --router-tasks` (40 tasks per
  layer fewer than the same line without it), and
  `fleet/tests/test_graph_plan.py`.
- `bash fleet/tasks/check_syntax.sh`: the new `mla_merge_oproj_mi300` row
  and the launcher, which gains the `mla_merge_oproj` test.

## Notes

- The descriptor's limits are what this task is measured against: 5 inputs
  of 7 and 3 outputs of 6 (`runtime_header.h`, lines 85 and 86, with the
  patch's sixth output). Nothing else on the page comes closer.
- `x_res` appears twice in the operator's tensor list, once as an input and
  once as an output, so the runtime hands the task the same address in
  `input_ptrs[3]` and `output_ptrs[0]`; the kernel takes one pointer and
  writes through it. The stock `linear_with_residual` path does the same
  (`fleet/graph_plan.py` names `x_res` as both the residual and the output
  of `L{l}.o_proj`), which is why the plan's chain rule and the boundary
  dump need no change: `x_res`'s last writer is still an operator whose
  label ends in `.o_proj`.
- One counter and one workspace serve every layer: the chain serialises the
  layers, and the last task of each layer resets the counter to zero, as the
  four-task router's does. The workspace's rows are overwritten by the next
  layer's tasks, which the same acquire and release fences order.
- The operator keeps the label `L{l}.o_proj`, so `--stop-after` and the
  `x_res` boundary keep their key; `attn`'s last writer becomes this
  operator, whose label's layer is the same, and `boundary_dump`'s
  `layer_of` reads the layer off the label, not off the operator's name.
- The merge phases are the regular tile form's at `halves = 2`, so `attn`
  is bit-identical to `--merge-tasks --merge-halves 2`; the suite's
  `mla_merge_oproj` row checks exactly that, and `x_res` against
  `numpy_ref.mla_merge_oproj` within the linear tolerance.
