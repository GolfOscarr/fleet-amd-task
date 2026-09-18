# N2: the router in four tasks, its task type, registration and dispatcher

The C++ the fork needs for `TASK_MOE_ROUTER4_MI300`, as blocks to add to
`fleet/patches/new_tasks.patch` by hand. Written on the laptop, where the
fork (`repos/fleet-chiplet-megakernel`) is absent, so the patch itself is
untouched: apply these on the patched fork, regenerate the patch with
`git diff` and reset the fork, the flow of `fleet/patches/README.md`. The
form is `fleet/patches/hunks/L2-linear-gemv.md`'s.

The kernel is `fleet/tasks/mi300/moe_router_mi300.cuh` with `SPLIT = 4`
(N2, R5 of `03-router-merge-ideas.md`): the same file, the same template,
two more arguments.

    moe_router_mi300_task_impl<T, HIDDEN, N_EXPERTS, N_FORCED, TOPK, ROUTE_STEPS,
                               ROUTE_LAYERS, NORM, SPLIT>(x, w_norm, W_gate, h_out,
                                                          topk_w, routing, mask, logits,
                                                          route_log, step, prompt_len,
                                                          layer_index, scaling, eps,
                                                          part, counter)

`part` is the task index (`expert_offset`, as `mla_prep` takes its head) and
`counter` a `[1]` int32 tensor of the plan, zeroed at allocation. Both
default (`0`, `nullptr`), so the one-task registrations already in the patch
are untouched. The form registered here is the fused one (`NORM = true`):
the split is what `--router-tasks` issues for the MoE layers, and it carries
the post-attention norm with it. The Python half is
`fleet/build_graph.py`'s `moe_router_norm4_layer` and the plan flag
`--router-tasks` (`fleet/graph_plan.py`); the registration below is what its
`register_task("moe_router_norm4_mi300", [topk, n_experts, n_forced,
scaling_bits, layer_index, hidden, eps_bits])` reaches.

Seven blocks, each with its file and the line it goes after. They are
independent; the order here is the patch's file order.

## 1. The task type

**File** `include/mirage/persistent_kernel/runtime_header.h`, in the
`TaskType` enum, after the line

    TASK_LINEAR_GEMV_MI300 = 195,      // regular: the GEMV linear of every dense projection at batch 1 (L1, L2)

```cpp
  TASK_MOE_ROUTER4_MI300 = 200,      // regular: the fused router's GEMV over four tasks, the last one routes (N2)
```

200 is free (200 to 229 lie between the fork's 199 and 230; 201 is N4's
merge). Nothing else in this header changes: the descriptor's 7 inputs and
6 outputs already hold this task's 4 and 6.

## 2. The name table (C++)

**File** `src/kernel/runtime.cc`, in `print_task_graph`, after

    task_type_to_name[TASK_LINEAR_GEMV_MI300] = "TASK_LINEAR_GEMV_MI300";

```cpp
  task_type_to_name[TASK_MOE_ROUTER4_MI300] = "TASK_MOE_ROUTER4_MI300";
```

## 3. The `expert_offset` list

**File** `src/kernel/runtime.cc`, in the per-task loop that packs `bid.x`
into the low 16 bits of `expert_offset` and `bid.y` into the high 16. With
N4's block applied the condition reads

    if (task_type == TASK_PREFETCH_MOE_MI300 ||
        task_type == TASK_MLA_ATTEND_TILE_MI300 ||
        task_type == TASK_MLA_MERGE_UV_TILE_MI300 ||   // (head, half) of the regular merge task (N4)
        task_type == TASK_MLA_PREP_MI300) {          // the head of the per-head prep task (round 3)

and becomes

```cpp
    if (task_type == TASK_PREFETCH_MOE_MI300 ||
        task_type == TASK_MLA_ATTEND_TILE_MI300 ||
        task_type == TASK_MLA_MERGE_UV_TILE_MI300 ||   // (head, half) of the regular merge task (N4)
        task_type == TASK_MOE_ROUTER4_MI300 ||         // the part of the four-task router (N2)
        task_type == TASK_MLA_PREP_MI300) {          // the head of the per-head prep task (round 3)
```

(The block stands on its own if N4's is not applied: the added line is the
`TASK_MOE_ROUTER4_MI300` one.) The type is regular, so none of the gang
lists, the `_execute_gang_task` lists or the MoE gang rule changes.

## 4. The name table (Python profiler)

**File** `python/mirage/mpk/profiler_persistent.py`, in `event_name_list`,
after

    195: "TASK_LINEAR_GEMV_MI300",

```python
    200: "TASK_MOE_ROUTER4_MI300",
```

## 5. The registration's declaration

**File** `include/mirage/kernel/task_register.h`, in the `TaskRegister`
class, after

    int register_moe_router_mi300_task(threadblock::Graph const &bgraph,
                                       std::vector<int> const &params,
                                       bool norm = false);   // norm: moe_router_norm_mi300 (O1)

```cpp
  int register_moe_router_norm4_mi300_task(threadblock::Graph const &bgraph,
                                           std::vector<int> const &params);   // N2: the fused router in four tasks
```

## 6. The registration

**File** `src/kernel/task_register.cc`, after the closing brace of
`int TaskRegister::register_moe_router_mi300_task(...)` (the function
ending `return register_task_variant(TASK_MOE_ROUTER_MI300, code.to_string());`)
and before the comment of `register_gang_moe_w2_silu_mi300_task`.

```cpp
// N2 of docs/gpu-experiments/04-kernels (R5 of 03-router-merge-ideas.md): the fused router's
// GEMV over four regular tasks of 16 experts, the last to arrive reading the 64 logits back and
// routing. params are the fused router's seven; inputs x_res, w_norm, W_gate and the counter
// [1] int32, all whole; outputs h, topk_w, routing, mask, logits, route_log, all whole. The
// part reaches the kernel through expert_offset (bid.x), as the prep task's head does.
int TaskRegister::register_moe_router_norm4_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 7);
  int topk = params[0], n_experts = params[1], n_forced = params[2];
  int layer_index = params[4], hidden = params[5];
  std::vector<tb::TBInputOp *> input_ops, output_ops;
  int num_inputs = 4, num_outputs = 6;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  split_io(bgraph, num_inputs, input_ops, output_ops);
  assert(input_ops[0]->dtensor.dim[1] == hidden);                       // x_res [1, hidden]
  assert(input_ops[1]->dtensor.dim[0] == hidden);                       // w_norm [hidden]
  assert(input_ops[2]->dtensor.dim[0] == n_experts && input_ops[2]->dtensor.dim[1] == hidden);
  assert(input_ops[3]->dtensor.num_dims == 1 && input_ops[3]->dtensor.dim[0] == 1);   // counter [1]
  assert(output_ops[0]->dtensor.dim[1] == hidden);                      // h [1, hidden]
  assert(output_ops[1]->dtensor.dim[1] == topk + n_forced);             // topk_w
  assert(output_ops[2]->dtensor.dim[0] == n_experts + n_forced);        // routing
  assert(output_ops[3]->dtensor.dim[0] == n_experts + n_forced + 1);    // mask
  assert(output_ops[4]->dtensor.dim[1] == n_experts);                   // logits
  assert(output_ops[5]->dtensor.num_dims == 3 && output_ops[5]->dtensor.dim[2] == topk + n_forced);
  int route_steps = output_ops[5]->dtensor.dim[0];
  int route_layers = output_ops[5]->dtensor.dim[1];
  assert(bgraph.grid_dim.x == 4 && n_experts % (4 * 4) == 0);           // four tasks, four waves
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::moe_router_mi300_task_impl<bfloat16, $, $, $, $, $, $, true, 4>(",
         hidden, n_experts, n_forced, topk, route_steps, route_layers);
  code.e("    task_desc->input_ptrs[0],");   // x_res
  code.e("    task_desc->input_ptrs[1],");   // w_norm
  code.e("    task_desc->input_ptrs[2],");   // W_gate
  code.e("    task_desc->output_ptrs[0],");  // h (written by part 0)
  code.e("    task_desc->output_ptrs[1],");  // topk_w
  code.e("    task_desc->output_ptrs[2],");  // routing
  code.e("    task_desc->output_ptrs[3],");  // mask
  code.e("    task_desc->output_ptrs[4],");  // logits (boundary B8; written by all four)
  code.e("    task_desc->output_ptrs[5],");  // route_log
  code.e("    runtime_config.step[0],");
  code.e("    runtime_config.prompt_length[0],");
  code.e("    $,", layer_index);
  code.e("    $,", float_literal(params[3]));       // routed_scaling_factor
  code.e("    $,", float_literal(params[6]));       // eps
  code.e("    (int)(task_desc->task_metadata.expert_offset & 0xFFFF),");   // bid.x: the part
  code.e("    task_desc->input_ptrs[3]);");  // counter
  return register_task_variant(TASK_MOE_ROUTER4_MI300, code.to_string());
}
```

`split_io` and `float_literal` are the file's own helpers, already in the
anonymous namespace the patch adds above `register_mla_prep_mi300_task`.

The emitted call for the model's dims, to compare against the
`MK_ROUTER4` block of `env/offline_gfx942/mk_tu.cu`:

```cpp
    kernel::moe_router_mi300_task_impl<bfloat16, 2048, 64, 2, 6, 32, 26, true, 4>(
        task_desc->input_ptrs[0],
        task_desc->input_ptrs[1],
        task_desc->input_ptrs[2],
        task_desc->output_ptrs[0],
        task_desc->output_ptrs[1],
        task_desc->output_ptrs[2],
        task_desc->output_ptrs[3],
        task_desc->output_ptrs[4],
        task_desc->output_ptrs[5],
        runtime_config.step[0],
        runtime_config.prompt_length[0],
        1,
        1.000000000e+00f,
        1.000000000e-06f,
        (int)(task_desc->task_metadata.expert_offset & 0xFFFF),
        task_desc->input_ptrs[3]);
```

## 7. The dispatcher branch

**File** `src/kernel/graph.cc`, in `Graph::register_task`, after the
`moe_router_norm_mi300` branch, that is after

    } else if (name == "moe_router_norm_mi300") {
      // O1: the post-attention norm folded into the router (inputs x_res, w_norm, W_gate;
      // outputs h, topk_w, routing, mask, logits, route_log)
      assert(params.size() == 7);
      int variant_id = task_register->register_moe_router_mi300_task(customized->bgraph, params, true);
      task_config[op] = std::make_tuple(3, 6, TASK_MOE_ROUTER_MI300, variant_id);

```cpp
  } else if (name == "moe_router_norm4_mi300") {
    // N2 (docs/gpu-experiments/04-kernels): the fused router's GEMV over four regular tasks,
    // the last one routes; inputs x_res, w_norm, W_gate, counter; the same six outputs
    assert(params.size() == 7);
    int variant_id = task_register->register_moe_router_norm4_mi300_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(4, 6, TASK_MOE_ROUTER4_MI300, variant_id);
```

The generated `_execute_task` needs nothing by hand: the code generator
emits one branch per (type, variant) from `task_config`, and a regular type
is on none of the gang lists this file and `runtime.cc` keep.

## What to check after applying

- `bash env/offline_gfx942/run.sh`: the host step parses `graph.cc`,
  `runtime.cc` and `task_register.cc` against the fork's headers, and the
  device step compiles `mk_tu.cu` with `MK_GEMV` (the `MK_ROUTER4` block
  rides on that define: the emitted call against the kernel's signature).
  The four-task form's register line is the one to read off the build: its
  body is the one-task kernel's with a shorter GEMV.
- The fork shows zero dirty lines after `git diff > new_tasks.patch` and the
  reset (`fleet/patches/README.md`).
- The dry run whose recorded call is the inputs and params this registration
  reads: `python fleet/build_graph.py --dry-run --router-tasks` (four tasks
  per MoE layer, 78 more than the one-task form), and
  `fleet/tests/test_graph_plan.py`.
- `bash fleet/tasks/check_syntax.sh`: the new instantiation of the
  `moe_router_mi300` row.

## Notes

- The counter is a 1-D `[1]` tensor, so `new_workspace` gives it a plain
  zeroed buffer (`build_graph.py`: the `ROW_SLACK` rows are for `[1, D]`
  activations only). One counter serves every layer: the chain serialises
  the routers, and the last task of each resets it to zero.
- The four tasks write the same `routing` zeros and `mask` -1 entries before
  their GEMV, as the one-task kernel does. Each task's release fence orders
  those writes before its increment, and the last task acquires after seeing
  the fourth, so the slot writes that follow cannot be overwritten by a
  straggler's initialisation.
- `h` is written by part 0 only; the expert gate-up reads it after the
  operator's event, so no ordering inside the operator depends on it.
- Every output is bit-identical to the one-task kernel's. A lane's chain of
  32 products per expert is unchanged and the cross-lane reduction is
  `butterfly_sum<ROUTER_BATCH>` in both forms: the wave's four experts are
  one batch of `ROUTER_BATCH` rows whose unused slots hold zero, and the
  butterfly never mixes rows, so the summation order is the batch constant's
  and not the split's. The suite's `moe_router4` row checks every output
  against the one-task row's, bit for bit.
