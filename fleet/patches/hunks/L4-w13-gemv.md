# L4: the w13 GEMV gang task's type, registration and dispatcher

The C++ the fork needs for `TASK_GANG_MOE_W13_GEMV_MI300`, as blocks to add
to `fleet/patches/new_tasks.patch` by hand. Written on the laptop, where the
fork (`repos/fleet-chiplet-megakernel`) is absent, so the patch itself is
untouched: apply these on the patched fork, regenerate the patch with
`git diff` and reset the fork, the flow of `fleet/patches/README.md`.

The kernel is `fleet/tasks/mi300/gang_moe_w13_gemv_mi300.cuh` (L4); its
signature is

    gang_moe_w13_gemv_kernel<T, N, K, NUM_EXPERTS, NUM_TOPK, TILES>(
        h, W13, routing, mask, mid, tile_idx)

and the argument order below is the one `env/offline_gfx942/mk_tu.cu`
compiles (the `MK_GEMV` block's `_execute_gang_task` entry). The Python half
is `fleet/build_graph.py`'s `gang_moe_w13_gemv_layer` and the plan flag
`--gemv-w13` (`fleet/graph_plan.py`); the registration below is what its
`register_task("gang_moe_w13_gemv_mi300", [tiles_per_expert,
max_experts_per_xcd, total_tiles_per_xcd])` reaches.

Nine blocks, each with its file and the line it goes after (the last one
optional). They are independent; the order here is the patch's file order.
Every place the fused w2 (type 191) was added is a place this type is added
too, because both are MoE gang types whose tile count comes from
`graph.gang_task_tiles_per_xcd` and whose `n_tile_start` is 0.

## 1. The task type

**File** `include/mirage/persistent_kernel/runtime_header.h`, in the
`TaskType` enum, after the line L2 adds

    TASK_LINEAR_GEMV_MI300 = 195,      // regular: the GEMV linear of every dense projection at batch 1 (L1, L2)

```cpp
  TASK_GANG_MOE_W13_GEMV_MI300 = 196, // gang: the expert gate-up as the GEMV loop, 37 tiles per XCD (L4)
```

196 is free (the free values are 195 to 197 below the fork's 198, and 200 to
229 between its 199 and 230). Nothing else in this header changes: the task
descriptor's 7 inputs and 6 outputs already hold this task's 4 and 1.

## 2. The name table (C++)

**File** `src/kernel/runtime.cc`, in `print_task_graph`, after

    task_type_to_name[TASK_PREFETCH_MOE_MI300] = "TASK_PREFETCH_MOE_MI300";

```cpp
  task_type_to_name[TASK_GANG_MOE_W13_GEMV_MI300] = "TASK_GANG_MOE_W13_GEMV_MI300";
```

## 3. The name table (Python profiler)

**File** `python/mirage/mpk/profiler_persistent.py`, in `event_name_list`,
after

    194: "TASK_PREFETCH_MOE_MI300",

```python
    196: "TASK_GANG_MOE_W13_GEMV_MI300",
```

(L2 adds 195 on the line between them.)

## 4. The kernel include

**File** `include/mirage/persistent_kernel/tasks/mi300/task_header.cuh`,
after

    #include "tasks/mi300/gang_moe_w2_silu_mi300.cuh"

```cpp
#include "tasks/mi300/gang_moe_w13_gemv_mi300.cuh"
```

(`mla_common_mi300.cuh`, which the kernel also needs, is included above it
already.) With the include in place `env/offline_gfx942/mk_tu.cu` no longer
needs its own `MK_GEMV` include; leave that file as it is all the same, it is
the standalone check.

## 5. The gang-type predicate

**File** `include/mirage/persistent_kernel/persistent_kernel.cuh`, in
`is_gang_task_type`, the line the patch already writes

           t == TASK_GANG_MOE_W2_SILU_MI300;

becomes

```cpp
         t == TASK_GANG_MOE_W2_SILU_MI300 ||
         t == TASK_GANG_MOE_W13_GEMV_MI300;
```

This is what puts the task on the gang dispatch path (`_execute_gang_task`
with the per-worker tile loop `for (t = rank; t < n_tile_count; t +=
workers_on_xcd)`) rather than the regular one.

## 6. The runtime's gang list and the tile rule

**File** `src/kernel/runtime.cc`, in `register_mugraph`, the list that reads
the operator's tile count. The line the patch already writes

              task_type == TASK_GANG_MOE_W2_SILU_MI300 ||

becomes

```cpp
              task_type == TASK_GANG_MOE_W2_SILU_MI300 ||
              task_type == TASK_GANG_MOE_W13_GEMV_MI300 ||
```

so the branch below it finds `graph.gang_task_tiles_per_xcd[op]` (asserting
it is there) and gives the task `n_tile_count = tiles_per_xcd`. The type is
not added to either `n_tile_start` branch of that block (the attention's
`bid.x * tiles_per_xcd` and the K-split's `bid.x`), so it keeps the default
`n_tile_start = 0`: the MoE rule, the one the stock w13 and w2 and the fused
w2 follow, under which `tile_idx` is XCD-local and the XCD comes from the
hardware register the kernel reads.

The count recorded is `max_experts_per_xcd x tiles_per_expert` (9 x 37 = 333
at 66 experts, against today's 9 x 44 = 396): the gang loop hands every
worker 9 tiles and the kernel returns at once for the eight local expert
indices past the XCD's one active expert, exactly as the stock w13 does.

## 7. The dispatcher: the emitted call in `_execute_gang_task`

**File** `src/kernel/runtime.cc`, in `print_task_graph`. The generated
`_execute_task` and `_execute_gang_task` come from two loops over
`task_config`, the first skipping the gang types and the second keeping only
them, so the type goes in both lists.

The first, after

        task.first == TASK_GANG_MOE_W2_SILU_MI300 ||

```cpp
        task.first == TASK_GANG_MOE_W2_SILU_MI300 ||
        task.first == TASK_GANG_MOE_W13_GEMV_MI300 ||
```

and the second, after

          task.first != TASK_GANG_MOE_W2_SILU_MI300 &&

```cpp
          task.first != TASK_GANG_MOE_W2_SILU_MI300 &&
          task.first != TASK_GANG_MOE_W13_GEMV_MI300 &&
```

With both in place the generator emits, into `_execute_gang_task`, the
branch whose body is the registration's code:

```cpp
    // gang_moe_w13_gemv_mi300 (L4): the expert gate-up, 37 tiles per expert per XCD
    kernel::gang_moe_w13_gemv_kernel<bfloat16, 2816, 2048, 66, 8, 37>(
        task_desc->input_ptrs[0],
        task_desc->input_ptrs[1],
        task_desc->input_ptrs[2],
        task_desc->input_ptrs[3],
        task_desc->output_ptrs[0],
        tile_idx);
```

which is the call `env/offline_gfx942/mk_tu.cu` compiles under `MK_GEMV`.

## 8. The registration's declaration

**File** `include/mirage/kernel/task_register.h`, in the `TaskRegister`
class, after

    int register_gang_moe_w2_silu_mi300_task(threadblock::Graph const &bgraph,
                                             std::vector<int> const &params);

```cpp
  int register_gang_moe_w13_gemv_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);   // L4: the expert gate-up as the GEMV loop
```

## 9. The registration

**File** `src/kernel/task_register.cc`, after the closing brace of
`int TaskRegister::register_gang_moe_w2_silu_mi300_task(...)` (the function
ending `return register_task_variant(TASK_GANG_MOE_W2_SILU_MI300, code.to_string());`).

```cpp
// L4 of docs/gpu-experiments/04-kernels (the kernel is gang_moe_w13_gemv_mi300.cuh).
// params: [tiles_per_expert, max_experts_per_xcd, total_tiles_per_xcd], the stock w13's
// three, with tiles_per_expert the XCD's worker count (37, S1) instead of N / 64; inputs
// h [1, K], W13 [E, N, K], routing [E, 1], mask [E + 1]; output mid [1, topk, N]. K comes
// from the weight, as in register_gang_moe_w2_silu_mi300_task, and the slot stride from the
// output tensor's own stride, which the kernel takes to be N (mid is dense).
int TaskRegister::register_gang_moe_w13_gemv_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 3);
  int tiles_per_expert = params[0];
  assert(tiles_per_expert == 37 &&
         "L4: one tile per worker of an XCD (num_workers / 8 = 37); the kernel's row arithmetic is 4 x 77 + 33 x 76");
  std::vector<tb::TBInputOp *> input_ops, output_ops;
  int num_inputs = 4, num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  split_io(bgraph, num_inputs, input_ops, output_ops);
  assert(output_ops[0]->output_tensors[0].num_dims == 3);
  int batch_size = output_ops[0]->output_tensors[0].dim[0];
  int num_experts_per_tok = output_ops[0]->output_tensors[0].dim[1];
  int output_size = output_ops[0]->output_tensors[0].dim[2];        // N: the expert's gate and up rows
  assert(batch_size == 1);                                          // one token per tile
  assert(input_ops[1]->output_tensors[0].num_dims == 3);
  int num_experts = input_ops[1]->output_tensors[0].dim[0];
  assert(input_ops[1]->output_tensors[0].dim[1] == output_size);    // W13 [E, N, K]
  int reduction_size = input_ops[1]->output_tensors[0].dim[2];      // K from the weight
  assert(input_ops[0]->dtensor.num_dims == 2);
  assert(input_ops[0]->dtensor.dim[0] == batch_size);               // h [1, K], the whole row
  assert(input_ops[0]->dtensor.dim[1] == reduction_size);
  assert(reduction_size % 512 == 0);                                // the kernel's 16-byte loads: K % (8 * WAVE)
  assert(output_ops[0]->dtensor.owner_op->op_type == type::KN_INPUT_OP);
  kn::KNInputOp *kn_out = static_cast<kn::KNInputOp *>(output_ops[0]->dtensor.owner_op);
  int output_stride = static_cast<int>(kn_out->input_strides[1]);
  assert(output_stride == output_size);   // mid is dense: the kernel's slot stride is N
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::gang_moe_w13_gemv_kernel<bfloat16, $, $, $, $, $>(",
         output_size, reduction_size, num_experts, num_experts_per_tok, tiles_per_expert);
  code.e("    task_desc->input_ptrs[0],");   // h
  code.e("    task_desc->input_ptrs[1],");   // expert weights
  code.e("    task_desc->input_ptrs[2],");   // routing indices
  code.e("    task_desc->input_ptrs[3],");   // mask
  code.e("    task_desc->output_ptrs[0],");  // mid
  code.e("    tile_idx);");
  return register_task_variant(TASK_GANG_MOE_W13_GEMV_MI300, code.to_string());
}
```

`split_io` is the file's own helper, already in the anonymous namespace the
patch adds above `register_mla_prep_mi300_task`.

## 10. The name branch in `Graph::register_task`

**File** `src/kernel/graph.cc`, in `Graph::register_task`, after the
`gang_moe_w2_silu_linear_mi300` branch, that is after

    } else if (name == "gang_moe_w2_silu_linear_mi300") {
      // O2 (docs/gpu-experiments/03-acceleration): the stock gang w2 with the silu-mul in its
      // prologue; inputs mid, W2, routing, mask; outputs out8, scratch; the stock w2's params
      assert(params.size() == 3);
      int variant_id = task_register->register_gang_moe_w2_silu_mi300_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(4, 2, TASK_GANG_MOE_W2_SILU_MI300, variant_id);
      gang_task_tiles_per_xcd[op] = params[2]; // total_tiles_per_xcd

```cpp
  } else if (name == "gang_moe_w13_gemv_mi300") {
    // L4 (docs/gpu-experiments/04-kernels): the expert gate-up as the GEMV loop in one round
    // per XCD; inputs h, W13, routing, mask; output mid; the stock w13's three params with
    // tiles_per_expert = 37
    assert(params.size() == 3);
    int variant_id = task_register->register_gang_moe_w13_gemv_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(4, 1, TASK_GANG_MOE_W13_GEMV_MI300, variant_id);
    gang_task_tiles_per_xcd[op] = params[2]; // total_tiles_per_xcd
```

This line is where the tile count per XCD is recorded for the runtime; there
is no separate Python-side record. The Python API's gang wrappers in
`python/mirage/mpk/persistent_kernel.py` only call `kn_graph.register_task`
with the stock w13's three params, and `fleet/build_graph.py`'s
`gang_moe_w13_gemv_layer` does the same through `_new_task`, exactly as
`gang_moe_w2_silu_linear_layer` does for type 191. So
`persistent_kernel.py` needs no hunk for this type.

## 11. Optional: the worker timing class (I1)

**File** `include/mirage/persistent_kernel/persistent_kernel.cuh`, in
`execute_worker`'s `MPK_ENABLE_TIMING` switch, the line the patch writes

        case TASK_GANG_MOE_W2_SILU_MI300: ours_cycles[5] += task_time; ours_count[5]++; break;

becomes

```cpp
        case TASK_GANG_MOE_W2_SILU_MI300:
        case TASK_GANG_MOE_W13_GEMV_MI300: ours_cycles[5] += task_time; ours_count[5]++; break;
```

so `--worker-timing`'s fused-MoE column of `[TASK_TIME2]` counts the gate-up
tasks too. Skip this block if the column should stay the fused w2's alone;
the two do appear in one graph (`--gemv-w13` does not replace the w2
operator), so the column then sums both.

## What to check after applying

- `bash env/offline_gfx942/run.sh`: the host step parses `graph.cc`,
  `runtime.cc` and `task_register.cc` against the fork's headers, and the
  device step compiles `mk_tu.cu` with `MK_GEMV` (the emitted call against
  the kernel's signature, in `_execute_gang_task`).
- The fork shows zero dirty lines after `git diff > new_tasks.patch` and the
  reset (`fleet/patches/README.md`).
- The dry run whose recorded calls are the inputs and params this
  registration reads:
  `python fleet/build_graph.py --dry-run --gemv-w13` (326 operators and
  2,285 tasks, unchanged: the flag swaps the kernel of the 26 w13
  operators), and `fleet/tests/test_graph_plan.py`.
- The tile count the runtime reads: 37 per expert, 333 per XCD, against the
  stock 44 and 396.
