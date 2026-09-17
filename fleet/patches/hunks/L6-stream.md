# L6: the stream probe's two task types, registrations and gang lists

The C++ the fork needs for `TASK_STREAM_MI300` and `TASK_STREAM_GANG_MI300`,
as blocks to add to `fleet/patches/new_tasks.patch` by hand. Written on the
laptop, where the fork (`repos/fleet-chiplet-megakernel`) is absent, so the
patch itself is untouched: apply these on the patched fork, regenerate the
patch with `git diff` and reset the fork, the flow of
`fleet/patches/README.md` (the form of `L2-linear-gemv.md`).

The kernel is `fleet/tasks/mi300/stream_mi300.cuh` (L6); its two entry points
are

    stream_mi300_task_impl<T, K>(w, dummy, rows)
    stream_gang_mi300_task_impl<T, K>(w, dummy, rows_per_tile, tiles_per_xcd, tile_idx)

and the argument orders below are the ones the offline unit already compiles
(`env/offline_gfx942/mk_tu.cu`, the `MK_GEMV` block's two stream branches).
The Python half is `fleet/build_graph.py`'s `stream_layer` and
`stream_gang_layer` and the plan mode `--graph stream` (`fleet/graph_plan.py`,
`build_stream_plan`); the registrations below are what their
`register_task("stream_mi300", [])` and
`register_task("stream_gang_mi300", [rows_per_tile, tiles_per_xcd])` reach.

The probe reads an operator's bytes divided by its event gap as a rate, so
both operators are plain chains of reads: no math, no residual, no numerics to
check. The only value written is the XOR word per wave the kernel stores so
the loads are not elided (the prefetch task's device), one `[4]` int32 row per
task or per tile.

Ten blocks, each with its file and the line it goes after (the last one
optional). They are independent; the order here is the patch's file order.

## 1. The task types

**File** `include/mirage/persistent_kernel/runtime_header.h`, in the
`TaskType` enum, after the line L2 adds

    TASK_LINEAR_GEMV_MI300 = 195,      // regular: the GEMV linear of every dense projection at batch 1 (L1, L2)

```cpp
  TASK_STREAM_MI300 = 197,           // regular: the stream probe, one task per grid slot (L6)
  TASK_STREAM_GANG_MI300 = 203,      // gang: the stream probe, 8 slots x tiles_per_xcd tiles (L6)
```

197 is the last free value below the fork's 198 (`TASK_HOPPER_TASK_END`), so
the gang form takes 203 from the free range 200 to 229 between the fork's 199
and 230. Nothing else in this header changes: the task descriptor's 7 inputs
and 6 outputs already hold these tasks' 2 and 1.

## 2. The gang-type list

**File** `include/mirage/persistent_kernel/persistent_kernel.cuh`, in
`is_gang_task_type`, the lines the patch already adds

    t == TASK_MLA_ATTEND_MI300 || t == TASK_MLA_MERGE_UV_MI300 ||
    t == TASK_GANG_MOE_W2_SILU_MI300;

become

```cpp
         t == TASK_MLA_ATTEND_MI300 || t == TASK_MLA_MERGE_UV_MI300 ||
         t == TASK_GANG_MOE_W2_SILU_MI300 || t == TASK_STREAM_GANG_MI300;
```

The regular form is on no list: its task index is its `bid.x` only through the
per-task pointers the imaps already give it, so it needs no `expert_offset`
entry either.

## 3. The name table (C++)

**File** `src/kernel/runtime.cc`, in `print_task_graph`, after

    task_type_to_name[TASK_PREFETCH_MOE_MI300] = "TASK_PREFETCH_MOE_MI300";

(and after L2's `TASK_LINEAR_GEMV_MI300` line, if that block is applied)

```cpp
  task_type_to_name[TASK_STREAM_MI300] = "TASK_STREAM_MI300";
  task_type_to_name[TASK_STREAM_GANG_MI300] = "TASK_STREAM_GANG_MI300";
```

## 4. The runtime's gang rule

**File** `src/kernel/runtime.cc`, in `register_mugraph`. Two lists, both of
which the patch already extends.

The tile count, after

              task_type == TASK_MLA_ATTEND_MI300 ||
              task_type == TASK_MLA_MERGE_UV_MI300) {

becomes

```cpp
              task_type == TASK_MLA_ATTEND_MI300 ||
              task_type == TASK_MLA_MERGE_UV_MI300 ||
              task_type == TASK_STREAM_GANG_MI300) {
```

so `graph.gang_task_tiles_per_xcd[op]` is read for it (block 8 records it).
The tile start, in the branch of the attention and the merge, after

            else if (task_type == TASK_GANG_ATTN_SPLIT_KV_MI300 ||
                task_type == TASK_GANG_ATTN_MERGE_MI300 ||
                task_type == TASK_MLA_ATTEND_MI300 ||
                task_type == TASK_MLA_MERGE_UV_MI300) {

becomes

```cpp
            else if (task_type == TASK_GANG_ATTN_SPLIT_KV_MI300 ||
                task_type == TASK_GANG_ATTN_MERGE_MI300 ||
                task_type == TASK_MLA_ATTEND_MI300 ||
                task_type == TASK_MLA_MERGE_UV_MI300 ||
                task_type == TASK_STREAM_GANG_MI300) {
              task.task_metadata.n_tile_start = (uint16_t)(bid.x * tiles_per_xcd);
```

which is the rule the kernel's decode assumes (`xcd = tile_idx /
tiles_per_xcd`), the merge's exactly.

## 5. The name table (Python profiler)

**File** `python/mirage/mpk/profiler_persistent.py`, in `event_name_list`,
after

    194: "TASK_PREFETCH_MOE_MI300",

```python
    197: "TASK_STREAM_MI300",
    203: "TASK_STREAM_GANG_MI300",
```

## 6. The kernel include

**File** `include/mirage/persistent_kernel/tasks/mi300/task_header.cuh`,
after

    #include "tasks/mi300/prefetch_mi300.cuh"

```cpp
#include "tasks/mi300/stream_mi300.cuh"
```

(`mla_common_mi300.cuh`, which the kernel also needs, is included above it
already.) With the include in place `env/offline_gfx942/mk_tu.cu` no longer
needs its own `MK_GEMV` include; leave that file as it is all the same, it is
the standalone check.

## 7. The registrations' declarations

**File** `include/mirage/kernel/task_register.h`, in the `TaskRegister`
class, after

    int register_prefetch_moe_mi300_task(threadblock::Graph const &bgraph,
                                         std::vector<int> const &params);   // O8: the expert prefetch

```cpp
  int register_stream_mi300_task(threadblock::Graph const &bgraph,
                                 std::vector<int> const &params);        // L6: the stream probe, regular
  int register_stream_gang_mi300_task(threadblock::Graph const &bgraph,
                                      std::vector<int> const &params);   // L6: the stream probe, gang
```

## 8. The registrations

**File** `src/kernel/task_register.cc`, after the closing brace of
`int TaskRegister::register_prefetch_moe_mi300_task(...)` (the function ending
`return register_task_variant(TASK_PREFETCH_MOE_MI300, code.to_string());`).

```cpp
// L6 of docs/gpu-experiments/04-kernels (the kernel is stream_mi300.cuh): the stream probe,
// the regular form. params: []; inputs W [rows_total, K] partitioned on dim 0 by the grid (the
// task's rows_total / grid rows) and the previous operator's dummy [*, 4] int32 whole, which the
// kernel never reads: it is the tensor that makes this operator a consumer of the one before it,
// which the runtime requires of every operator (register_mugraph, num_shared_tensors >= 1), and
// the plan alternates two dummies so every operator has one. Output dummy [grid, 4] int32
// partitioned on dim 0 (one row per task, one XOR word per wave). The per-task row count is the
// partitioned input's dim 0 over the grid, as the prefetch registration reads its stripe.
int TaskRegister::register_stream_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 0);
  std::vector<tb::TBInputOp *> input_ops, output_ops;
  int num_inputs = 2, num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  split_io(bgraph, num_inputs, input_ops, output_ops);
  int grid = (int)bgraph.grid_dim.x;
  assert(input_ops[0]->dtensor.num_dims == 2 && input_ops[0]->input_map.x == 0);
  int rows_total = input_ops[0]->dtensor.dim[0], k = input_ops[0]->dtensor.dim[1];
  assert(rows_total % grid == 0);
  int rows = rows_total / grid;
  assert(k % 512 == 0);                             // the kernel's 16-byte loads: K % (8 * WAVE)
  assert(input_ops[1]->dtensor.num_dims == 2 && input_ops[1]->dtensor.dim[1] == 4);
  assert(input_ops[1]->input_map.x == -1);          // the chain's dummy, read whole and ignored
  assert(output_ops[0]->dtensor.num_dims == 2 && output_ops[0]->input_map.x == 0);
  assert(output_ops[0]->dtensor.dim[0] == grid && output_ops[0]->dtensor.dim[1] == 4);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::stream_mi300_task_impl<bfloat16, $>(", k);
  code.e("    task_desc->input_ptrs[0],");          // the task's rows
  code.e("    task_desc->output_ptrs[0],");         // the task's dummy row
  code.e("    $);", rows);
  return register_task_variant(TASK_STREAM_MI300, code.to_string());
}

// L6: the stream probe, the gang form. params: [rows_per_tile, tiles_per_xcd]; inputs W
// [8 * tiles_per_xcd * rows_per_tile, K] whole and the previous operator's dummy whole (the
// chain's tensor, as above); output dummy [8 * tiles_per_xcd, 4] int32 whole (the tile's row).
// The grid is (8, 1, 1) and the runtime sets n_tile_start = bid.x * tiles_per_xcd for this type
// (the attention's and the merge's branch), so the kernel's decode is theirs.
int TaskRegister::register_stream_gang_mi300_task(
    threadblock::Graph const &bgraph, std::vector<int> const &params) {
  assert(params.size() == 2);
  int rows_per_tile = params[0], tiles_per_xcd = params[1];
  std::vector<tb::TBInputOp *> input_ops, output_ops;
  int num_inputs = 2, num_outputs = 1;
  assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
  split_io(bgraph, num_inputs, input_ops, output_ops);
  assert(bgraph.grid_dim.x == 8);
  int tiles = 8 * tiles_per_xcd;
  assert(input_ops[0]->dtensor.num_dims == 2 && input_ops[0]->input_map.x == -1);
  int k = input_ops[0]->dtensor.dim[1];
  assert(input_ops[0]->dtensor.dim[0] == tiles * rows_per_tile);
  assert(k % 512 == 0);                             // the kernel's 16-byte loads
  assert(input_ops[1]->dtensor.num_dims == 2 && input_ops[1]->dtensor.dim[1] == 4);
  assert(input_ops[1]->input_map.x == -1);          // the chain's dummy, read whole and ignored
  assert(output_ops[0]->dtensor.num_dims == 2 && output_ops[0]->input_map.x == -1);
  assert(output_ops[0]->dtensor.dim[0] == tiles && output_ops[0]->dtensor.dim[1] == 4);
  mirage::transpiler::CodeKeeper code;
  code.inc_indent();
  code.e("kernel::stream_gang_mi300_task_impl<bfloat16, $>(", k);
  code.e("    task_desc->input_ptrs[0],");          // the whole tensor
  code.e("    task_desc->output_ptrs[0],");         // the whole dummy
  code.e("    $,", rows_per_tile);
  code.e("    $,", tiles_per_xcd);
  code.e("    tile_idx);");
  return register_task_variant(TASK_STREAM_GANG_MI300, code.to_string());
}
```

`split_io` is the file's own helper, already in the anonymous namespace the
patch adds above `register_mla_prep_mi300_task`.

The emitted calls for the three rows of G5
(`docs/gpu-experiments/04-kernels/02-local-gpu-split.md`), to compare against
the two stream branches of `env/offline_gfx942/mk_tu.cu`:

```cpp
    // the regular form at 152 KB over 96 tasks (qkva's shape): 3,648 rows of 2,048 by 96
    kernel::stream_mi300_task_impl<bfloat16, 2048>(
        task_desc->input_ptrs[0],
        task_desc->output_ptrs[0],
        38);
    // the regular form at 256 KB over 296 tasks (one per CU): 18,944 rows by 296
    kernel::stream_mi300_task_impl<bfloat16, 2048>(
        task_desc->input_ptrs[0],
        task_desc->output_ptrs[0],
        64);
    // the gang form at 304 KB per tile, 37 tiles per XCD (w13's shape): 22,496 rows
    kernel::stream_gang_mi300_task_impl<bfloat16, 2048>(
        task_desc->input_ptrs[0],
        task_desc->output_ptrs[0],
        76,
        37,
        tile_idx);
```

## 9. The dispatcher branches

**File** `src/kernel/graph.cc`, in `Graph::register_task`, after the
`prefetch_moe_mi300` branch, that is after

    } else if (name == "prefetch_moe_mi300") {
      // O8: a side operator streaming part of an active expert's weight per task; inputs
      // W [E, N, K] whole and mask [E + 1]; output dummy [slots x parts, 4]; params [parts]
      assert(params.size() == 1);
      int variant_id = task_register->register_prefetch_moe_mi300_task(customized->bgraph, params);
      task_config[op] = std::make_tuple(2, 1, TASK_PREFETCH_MOE_MI300, variant_id);
      side_ops.insert(op);

```cpp
  } else if (name == "stream_mi300") {
    // L6 (docs/gpu-experiments/04-kernels): the stream probe, one regular task per grid slot;
    // inputs W (the task's rows) and the chain's dummy (whole, ignored); output dummy [grid, 4]
    assert(params.size() == 0);
    int variant_id = task_register->register_stream_mi300_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_STREAM_MI300, variant_id);
  } else if (name == "stream_gang_mi300") {
    // L6: the stream probe as a gang operator of 8 slots x tiles_per_xcd tiles; inputs W (whole)
    // and the chain's dummy (whole, ignored); output dummy [8 x tiles_per_xcd, 4] (whole);
    // params [rows_per_tile, tiles_per_xcd]
    assert(params.size() == 2);
    int variant_id = task_register->register_stream_gang_mi300_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_STREAM_GANG_MI300, variant_id);
    gang_task_tiles_per_xcd[op] = params[1];   // tiles_per_xcd, as the merge records its heads per XCD
```

Neither operator is a side operator, so `side_ops` is untouched; the generated
`_execute_task` and `_execute_gang_task` need nothing by hand, since the code
generator emits one branch per (type, variant) from `task_config` and the gang
form is on the list of block 2.

## 10. Optional: the worker timing class (I1)

**File** `include/mirage/persistent_kernel/persistent_kernel.cuh`, in
`execute_worker`'s `MPK_ENABLE_TIMING` switch, the lines

        case TASK_PREFETCH_MI300:
        case TASK_PREFETCH_MOE_MI300: ours_cycles[7] += task_time; ours_count[7]++; break;

become

```cpp
        case TASK_PREFETCH_MI300:
        case TASK_PREFETCH_MOE_MI300:
        case TASK_STREAM_MI300:
        case TASK_STREAM_GANG_MI300: ours_cycles[7] += task_time; ours_count[7]++; break;
```

so `--worker-timing`'s prefetch column of `[TASK_TIME2]` counts the probe's
tasks too (the two never appear in one graph: the stream plan has no model).
Skip this block if that column should stay the prefetch tasks' alone.

## What to check after applying

- `bash env/offline_gfx942/run.sh`: the host step parses `graph.cc`,
  `runtime.cc` and `task_register.cc` against the fork's headers, and the
  device step compiles `mk_tu.cu` with `MK_GEMV` (the emitted calls against
  the kernel's two signatures).
- The fork shows zero dirty lines after `git diff > new_tasks.patch` and the
  reset (`fleet/patches/README.md`).
- The dry runs whose recorded calls are the inputs and params these
  registrations read:
  `python fleet/build_graph.py --dry-run --graph stream --ops 10 --tasks 96 --kb 152`
  (10 operators, 960 tasks), `--tasks 296 --kb 256` (10 operators, 2,960
  tasks) and `--ops 10 --tasks 37 --kb 304 --gang` (10 operators, 80 tasks of
  37 tiles each), and `fleet/tests/test_graph_plan.py`.

## Two notes on the shapes

`--kb` is the bytes one task or one tile reads, and a row of the probe's
`[*, 2048]` BF16 tensor is exactly 4 KB, so the plan asserts that `--kb` is a
multiple of 4: a task reads whole rows, and a byte count that did not follow
from the rows would corrupt the one number this probe exists to produce.
`05-local-preparation.md` and `02-local-gpu-split.md` give w13's tile as
305 KB, which is not such a multiple; the G5 row runs at 304 KB (76 rows),
0.3% below it, and 8 x 37 x 304 KB is 90 MB, w13's eight active experts.

The chain's dummy is a real input of both registrations. The runtime rejects a
graph whose operator shares no tensor with the one before it
(`runtime.cc`, `register_mugraph`: `assert(num_shared_tensors >= 1)`), and a
stream operator writes only its dummy, so the plan alternates two dummies and
every operator reads the one the previous operator wrote. The kernel never
touches that pointer; it is there for the registration's shape, exactly as the
empty ladder of I3 alternates its two copy tensors.
