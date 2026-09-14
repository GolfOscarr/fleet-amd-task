# 10 - Work completed locally before GPU access

Everything that does not need the MI300X, with the file each item produces.
The GPU days then start from a repository that already contains the
harness, the packing code, the graph script, the new kernels, and the
reference artifacts' recipes.

## Already done (this design)

| Item | Where |
|---|---|
| Model, hardware, Fleet, acceleration, MLA prior-art analysis | `docs/` discovery sets |
| Fleet runtime, Python API, MoE demo and MI300 kernels read line by line | `docs/fleet/03-runtime.md`, `04-repo-map.md` |
| The design: decisions, execution flow, task graph, synchronization, memory plan, interface, correctness, milestones, performance | `docs/design-doc/` |
| Graph counts and rooflines derived by script | `docs/design-doc/sources/graph_counts.py` |

## To do locally, in order

| # | Item | Output | Depends on | Notes |
|---|---|---|---|---|
| L1 | Choose and tokenize the 1,024-token prompt | `harness/prompt_ids.json` | tokenizer files (small, downloadable without the weights) | a code prompt for a code model; exactly 1,024 ids; committed, never regenerated |
| L2 | NumPy reference of every new kernel's math | `harness/numpy_ref.py` | `01-execution-flow.md` | `mla_prep`, `mla_attend` (with splits and merge), `mla_merge_uv`, `moe_router` as pure NumPy functions on FP32; this is the spec the kernels are tested against on the GPU |
| L3 | CPU check of runtime reassociation | `harness/reassoc_check.py`, its output committed | HF model on CPU, or random weights of the right shapes | for one layer: reference attention (decompressed) versus `q_nope @ W_UK` then `c_kv` scores and `W_UV` after; reports `rel_err` on B5 and B6 with random and with structured inputs; settles `OPEN-PROBLEMS.md` MIN-1 to the extent CPU BF16 can |
| L4 | Weight packing | `fleet/pack_weights.py` | safetensors headers (shapes already known) | builds `W_qkva`, padded layer-0 MLP, `W13`/`W2` for 66 experts, `W_uk`/`W_uv` views; unit-tested on random tensors of the real shapes for shape, order and the exactness of the shared-expert split (`04-memory-plan.md`) |
| L5 | The graph builder script | `fleet/build_graph.py` | Fleet Python API (importable without a GPU for the pure-Python parts) | the call list of `02-task-graph.md` with `--layers N`, `--head`, `--debug`; the new `*_layer` methods; cannot be run end to end locally but its argument arithmetic (tiles, strides, asserts) is |
| L6 | The four new kernels and the glue | `fleet/tasks/mi300/mla_prep_mi300.cuh`, `mla_attend_mi300.cuh`, `mla_merge_uv_mi300.cuh`, `moe_router_mi300.cuh`; the two variants; `task_register.cc`, `graph.cc`, `runtime.cc`, `runtime_header.h`, `persistent_kernel.cuh` edits as a patch | the eight-place recipe in `docs/fleet/04-repo-map.md`; the spec in `docs/mla-decode/04-our-kernel-spec.md` | written against the `gang_linear_mi300.cuh` contract; syntax can be checked with a host `clang++ -fsyntax-only` and stub headers, correctness only on the GPU |
| L7 | Reference-run and capture script | `harness/run_reference.py` | L1 | the code of `05-prefill-interface.md` and the boundary hooks of `07-correctness.md`; runs on the GPU on day 1 |
| L8 | Calibration script | `harness/calibrate.py` | L7 | the two-run floor measurement |
| L9 | Comparison script | `harness/compare.py` | none | metrics, thresholds, exact checks, `correctness_report.md`; unit-tested locally on synthetic pairs |
| L10 | Fleet run script | `harness/run_fleet.py` | L5 | meta-tensor setup (`step = 1022`, `qo_indptr = [0, 1]`, ...), truncated-graph options, boundary dumps, timing capture |
| L11 | Measurement script | `harness/measure.py` | none | parses `[FWD_PASS]` and the event-timing buffer, runs `rocprofv3` with the counter sets of `docs/mi300x/06-profiling.md`, writes `metrics.json` and the report tables of `09-expected-performance.md` |
| L12 | Environment scripts | `env/setup.sh`, `env/check_day1.sh` | none | ROCm and PyTorch-ROCm install, `AMDGPU_TARGETS=gfx942 pip install -e .`, model download, the day-1 checks of `08-milestones.md` (partition mode, `[SCHED_XCD]`, disassembly greps, CK 576/512 probe) |
| L13 | The gfx942 build and instrumentation patch | a patch removing the `paged_attention_decode_minimal_mi300.cuh` include, guarding the `16x16x32` selection, and removing the `[FWD_PASS]` print throttle (`persistent_kernel.cuh:1535-1536`, `:1047`) so every iteration is logged | `docs/fleet/99-open-questions.md` Q12 | applied on day 1 before the first build |
| L14 | Routing correlation analysis | `harness/route_analysis.py` | L7's `ref_route_log.json` | per-layer overlap of selected experts across consecutive steps; decides whether expert affinity is worth a next-step recommendation (`OPEN-PROBLEMS.md` MIN-6) |

L1-L4, L9 and L12-L13 need no Fleet import. L5 and L6 are written against
the code read and are the largest items; they are also the ones whose
correctness can only be established on the machine, so they are written to
be testable in isolation (kernel tests against L2) rather than only through
the full graph.

## What cannot be done locally, and is therefore day 1

- Whether the runtime builds for gfx942 (MAJ-1), and the fence disassembly
  (MAJ-3).
- Whether CK's FMHA instantiates at 576/512 (Q11).
- Any timing, any counter, the BF16 floor on the target GPU, and every
  boundary comparison.

## Order of value

If the local time runs short, the order is L1, L7, L9, L4, L2, L5, L6, L12,
L13, L10, L11, L3, L8, L14: the artifacts that make day 1 productive first
(prompt, reference recipe, comparison), then what the GPU days build on
(packing, kernel specs, graph, kernels), then measurement and analysis.

## Status (2026-09-14)

All of L1-L14 are implemented on branch `local/harness` and independently
reviewed; `PROGRESS.md` Stage 3 lists each item's file, and
`harness/README.md` the run recipe. What the local work could not settle
is exactly the list above ("What cannot be done locally"), plus two items
found on the way: `kernel_tests.py` from `07-correctness.md` is still to
be written (a standalone HIP launcher, day 2), and the CPU reassociation
check showed that the attention threshold must be the calibrated floor,
not the starting value (`07-correctness.md`, item 7).
