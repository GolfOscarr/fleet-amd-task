# N4: the merge as regular tasks, its task type, registration and dispatcher

The C++ the fork needs for `TASK_MLA_MERGE_UV_TILE_MI300`, as blocks to add
to `fleet/patches/new_tasks.patch` by hand. Written on the laptop, where the
fork (`repos/fleet-chiplet-megakernel`) is absent, so the patch itself is
untouched: apply these on the patched fork, regenerate the patch with
`git diff` and reset the fork, the flow of `fleet/patches/README.md`. The
form is `fleet/patches/hunks/L2-linear-gemv.md`'s.

The kernel is the second entry point of `fleet/tasks/mi300/mla_merge_uv_mi300.cuh`
(N4, merged with the gang body it shares):

    mla_merge_uv_tile_mi300_task_impl<T, NH, D_V, D_C, HALVES>(partials, w_uv, attn,
                                                               step, split, n_splits, idx)

`idx` is the task index (`h = idx / HALVES`, `half = idx % HALVES`), taken
from `expert_offset` as `mla_prep` takes its head and `mla_attend_tile` its
split; every tensor is whole. The Python half is `fleet/build_graph.py`'s
`mla_merge_uv_tile_layer` and the plan flags `--merge-tasks` and
`--merge-halves N` (`fleet/graph_plan.py`); the registration below is what
its `register_task("mla_merge_uv_tile_mi300", [split, n_splits, halves])`
reaches. The model of the whole item is the attention's per-tile
registration (`register_mla_attend_tile_mi300_task`) beside its gang one.

Seven blocks, each with its file and the line it goes after. They are
independent; the order here is the patch's file order.

## 1. The task type

**File** `include/mirage/persistent_kernel/runtime_header.h`, in the
`TaskType` enum, after the line

    TASK_LINEAR_GEMV_MI300 = 195,      // regular: the GEMV linear of every dense projection at batch 1 (L1, L2)

```cpp
  TASK_MLA_MERGE_UV_TILE_MI300 = 201,  // regular: one merge task per (head, half), the index from expert_offset (N4)
```

201 is free (200 to 229 lie between the fork's 199 and 230; 200 is N2's
router). Nothing else in this header changes: the descriptor's 7 inputs and
6 outputs already hold this task's 2 and 1.

## 2. The name table (C++)

**File** `src/kernel/runtime.cc`, in `print_task_graph`, after

    task_type_to_name[TASK_LINEAR_GEMV_MI300] = "TASK_LINEAR_GEMV_MI300";

```cpp
  task_type_to_name[TASK_MLA_MERGE_UV_TILE_MI300] = "TASK_MLA_MERGE_UV_TILE_MI300";
```

## 3. The `expert_offset` list

**File** `src/kernel/runtime.cc`, in the per-task loop that packs `bid.x`
into the low 16 bits of `expert_offset` and `bid.y` into the high 16. The
condition today reads

    if (task_type == TASK_PREFETCH_MOE_MI300 ||
        task_type == TASK_MLA_ATTEND_TILE_MI300 ||
        task_type == TASK_MLA_PREP_MI300) {          // the head of the per-head prep task (round 3)

and becomes

```cpp
    if (task_type == TASK_PREFETCH_MOE_MI300 ||
        task_type == TASK_MLA_ATTEND_TILE_MI300 ||
        task_type == TASK_MLA_MERGE_UV_TILE_MI300 ||   // (head, half) of the regular merge task (N4)
        task_type == TASK_MLA_PREP_MI300) {          // the head of the per-head prep task (round 3)
```

The type is regular, so none of the gang lists, the `_execute_gang_task`
lists or the MoE gang rule changes.

## 4. The name table (Python profiler)

**File** `python/mirage/mpk/profiler_persistent.py`, in `event_name_list`,
after

    195: "TASK_LINEAR_GEMV_MI300",

```python
    201: "TASK_MLA_MERGE_UV_TILE_MI300",
```

## 5. The registration's declaration

**File** `include/mirage/kernel/task_register.h`, in the `TaskRegister`
class, after

    int register_mla_merge_uv_mi300_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params);

```cpp
  int register_mla_merge_uv_tile_mi300_task(threadblock::Graph const &bgraph,
                                            std::vector<int> const &params);   // N4: one task per (head, half)
```

## 6. The registration

**File** `src/kernel/task_register.cc`, after the closing brace of
`int TaskRegister::register_mla_merge_uv_mi300_task(...)` (the function
ending `return register_task_variant(TASK_MLA_MERGE_UV_MI300, code.to_string());`)
and before the comment of `register_moe_router_mi300_task`.

```cpp
// N4 of docs/gpu-experiments/04-kernels (M6 and M4 of 03-router-merge-ideas.md): the merge as
// NH * halves regular tasks instead of the 8 x heads_per_xcd gang. params: [split, n_splits,
// halves]; inputs partials [n_splits, nh, partials_row] and W_uv [nh, d_v, d_c], both whole;
// output attn [1, nh * d_v], whole. The task index reaches the kernel through expert_offset
// (bid.x), as it does for mla_attend_tile and mla_prep, and the kernel takes h = idx / halves
// and half = idx % halves from it, so no imap is partitioned and there are no local flags.
int TaskRegister::register_mla_merge_uv_tile_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 3);
  int split = params[0], n_splits = params[1], halves = params[2];
  assert(n_splits <= 64 && "mla_merge_uv merges one split per lane of one wavefront");
  assert(halves == 1 || halves == 2);
  std::vector<tb::TBInputOp *> input_ops, output_ops;
  int num_inputs = 2, num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  split_io(bgraph, num_inputs, input_ops, output_ops);
  assert(input_ops[1]->dtensor.num_dims == 3);
  int nh = input_ops[1]->dtensor.dim[0];
  int d_v = input_ops[1]->dtensor.dim[1];
  int d_c = input_ops[1]->dtensor.dim[2];
  assert(input_ops[0]->dtensor.dim[0] == n_splits && input_ops[0]->dtensor.dim[1] == nh &&
         input_ops[0]->dtensor.dim[2] == ((d_c + 1 + 3) / 4) * 4);
  assert(output_ops[0]->dtensor.dim[1] == nh * d_v);
  assert(input_ops[0]->input_map.x == -1 && input_ops[1]->input_map.x == -1 &&
         output_ops[0]->input_map.x == -1);            // every tensor whole
  assert((int)bgraph.grid_dim.x == nh * halves);       // one task per (head, half)
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::mla_merge_uv_tile_mi300_task_impl<bfloat16, $, $, $, $>(", nh, d_v, d_c, halves);
  code.e("    task_desc->input_ptrs[0],");   // partials (whole)
  code.e("    task_desc->input_ptrs[1],");   // W_uv (whole)
  code.e("    task_desc->output_ptrs[0],");  // attn (whole)
  code.e("    runtime_config.step[0],");
  code.e("    $,", split);
  code.e("    $,", n_splits);
  code.e("    (int)(task_desc->task_metadata.expert_offset & 0xFFFF));");   // bid.x: the (head, half) index
  return register_task_variant(TASK_MLA_MERGE_UV_TILE_MI300, code.to_string());
}
```

`split_io` is the file's own helper, already in the anonymous namespace the
patch adds above `register_mla_prep_mi300_task`.

The emitted call for the model's dims, to compare against the
`MK_MERGE_TILE` block of `env/offline_gfx942/mk_tu.cu`:

```cpp
    // --merge-tasks (16 tasks of a whole head)
    kernel::mla_merge_uv_tile_mi300_task_impl<bfloat16, 16, 128, 512, 1>(
        task_desc->input_ptrs[0],
        task_desc->input_ptrs[1],
        task_desc->output_ptrs[0],
        runtime_config.step[0],
        32,
        33,
        (int)(task_desc->task_metadata.expert_offset & 0xFFFF));
    // --merge-tasks --merge-halves 2 (32 tasks of a half head)
    kernel::mla_merge_uv_tile_mi300_task_impl<bfloat16, 16, 128, 512, 2>(
        task_desc->input_ptrs[0],
        task_desc->input_ptrs[1],
        task_desc->output_ptrs[0],
        runtime_config.step[0],
        32,
        33,
        (int)(task_desc->task_metadata.expert_offset & 0xFFFF));
```

## 7. The dispatcher branch

**File** `src/kernel/graph.cc`, in `Graph::register_task`, after the
`mla_merge_uv_mi300` branch, that is after

    } else if (name == "mla_merge_uv_mi300") {
      assert(params.size() == 6);
      int variant_id = task_register->register_mla_merge_uv_mi300_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(2, 1, TASK_MLA_MERGE_UV_MI300, variant_id);

```cpp
  } else if (name == "mla_merge_uv_tile_mi300") {
    // N4 (docs/gpu-experiments/04-kernels): the merge as nh * halves regular tasks, every
    // tensor whole; inputs partials, W_uv; output attn; params [split, n_splits, halves]
    assert(params.size() == 3);
    int variant_id = task_register->register_mla_merge_uv_tile_mi300_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_MLA_MERGE_UV_TILE_MI300, variant_id);
```

The generated `_execute_task` needs nothing by hand: the code generator
emits one branch per (type, variant) from `task_config`, and a regular type
is on none of the gang lists this file and `runtime.cc` keep.

## What to check after applying

- `bash env/offline_gfx942/run.sh`: the host step parses `graph.cc`,
  `runtime.cc` and `task_register.cc` against the fork's headers, and the
  device step compiles `mk_tu.cu` with `MK_GEMV` (the `MK_MERGE_TILE` block
  rides on that define: the emitted calls against the kernel's signature).
- The fork shows zero dirty lines after `git diff > new_tasks.patch` and the
  reset (`fleet/patches/README.md`).
- The dry runs whose recorded calls are the inputs and params this
  registration reads:
  `python fleet/build_graph.py --dry-run --merge-tasks` (16 tasks per layer)
  and `--merge-tasks --merge-halves 2` (32), and `fleet/tests/test_graph_plan.py`.
- `bash fleet/tasks/check_syntax.sh`: the two new instantiations of the
  `mla_merge_uv_mi300` row.

## Notes

- The whole-tensor imaps are the point of the item (M6): a regular merge
  task needs neither `xcd_offset_dim0` nor the gang registration's
  `w_uv_local` and `out_local` flags, and the kernel's shared body takes
  pointers already offset to the head either way.
- The two forms are bit-exact against each other. A row's lane sums are
  reduced by `butterfly_sum<MERGE_W_BATCH>` in both, and the batch constant
  does not change with `halves` (a wave's 32 rows in two batches of 16
  become 16 rows in one), so a row of `attn` is the same float whichever
  form wrote it. The suite's `mla_merge_uv_tile` row checks exactly that.
