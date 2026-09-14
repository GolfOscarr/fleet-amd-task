# Plan for the technical design document

> Working plan, kept as the record of how the set was produced. Section 2
> predates the full code read: F1's launch count is superseded by
> `00-decisions.md` D22 (three dispatches per generation) and F4's two
> layer names by the four kernels and two variants of `02-task-graph.md`.

This file plans the design document set in `docs/design-doc/`. It is the
working plan, not the deliverable; the numbered files are the deliverable. It
records what the design must contain, where each input already exists, what
still has to be established before a section can be written correctly, the
order of writing, and the check that runs after.

Date: 2026-09-14. Branch: `design/technical-spec`. Inputs: the five discovery
sets under `docs/` (41 files) and the Fleet submodule at `51dce4f`.

---

## 1. What the document must contain

The task description (`docs/task-description.pdf`, "First deliverable") names
eight items. The brief for this stage adds a ninth: the optimization strategy
for applying Fleet to this model on this GPU. One file per item, so a reviewer
can find each by name.

| Required item (PDF wording) | File |
|---|---|
| model execution flow | `01-execution-flow.md` |
| proposed Fleet task graph | `02-task-graph.md` |
| synchronization strategy | `03-synchronization.md` |
| memory plan, including KV-cache layout | `04-memory-plan.md` |
| interface between reference prefill and Fleet decode | `05-prefill-interface.md` |
| optimization strategy (this stage's brief) | `06-optimization-strategy.md` |
| correctness methodology | `07-correctness.md` |
| implementation milestones | `08-milestones.md` |
| expected performance | `09-expected-performance.md` |
| work that can be completed locally before using the GPU | `10-local-work.md` |
| (ours) decisions and their rationale, consolidated | `00-decisions.md` |
| (ours) index and one-page summary | `README.md` |
| (ours) design-level unknowns | `99-open-questions.md` |

The PDF also fixes constraints the design must restate and obey: batch 1, BF16
first, 1,024-token prompt, 32 greedy tokens, no tensor parallelism, no
continuous batching, no speculative decoding, prefill out of scope, required
milestone is an MoE layer at index 1 or higher, one-time cache conversion
excluded from timed decode, and the metric list (correctness error,
Fleet-native ops, fallbacks, consecutive layers, GPU launches, median and P95
latency, memory traffic, achieved bandwidth, occupancy and resource usage, TPOT
and tokens/s if end-to-end).

---

## 2. Facts established while planning

These were checked against the Fleet source during planning because each one
changes the shape of a section. File and line references are into
`repos/fleet-chiplet-megakernel` unless stated.

**F1. One launch runs the whole generation, not one token.**
`demo/qwen3/demo.py:1196-1204` loops per token, but that loop is under
`if not args.use_mirage:` (the PyTorch reference path). The Fleet path is the
`else:` at line 1252: a single `mpk()` call. Inside the kernel, on
`EVENT_END_OF_TASK_GRAPH` the scheduler calls `prepare_next_batch`
(`persistent_kernel.cuh:381-536` for `MODE_OFFLINE`, which copies
`output_tokens` into `tokens[step+1]`; `:538-568` for `MODE_ONLINE`, which
only advances `config.step`), stops on EOS or `max_seq_length`, and
otherwise re-issues the graph with
`iteration_num + 1` encoded in the high 32 bits of every task id
(`compute_task_id`, `:263-265`). Both Qwen3 demos run `mode="offline"`
(`demo.py:315`, `demo_30B_A3B.py:273`).

Consequences: (a) "GPU launches" for the Fleet path is 1 per generation, and
the design must say so and also report launches per token as 1/32; (b) the
argmax to next-token embedding is an in-kernel edge across iterations and
belongs in the task graph; (c) per-token median and P95 latency cannot be
taken from host timers around one launch. The design must choose in-kernel
timestamps (`docs/fleet/03-runtime.md`, "In-kernel timing") or a
`max_new_tokens=1` measurement mode, and must state which number is reported.

**F2. HF prefill does not keep the latent cache.**
`docs/deepseek-v2-lite/sources/modeling_deepseek.py:831-867` computes
`compressed_kv` and `k_pe`, then caches the decompressed
`key_states [1,16,S,192]` and `value_states [1,16,S,128]`. Recovering the
512-wide latent from the decompressed cache would need a pseudo-inverse of
`W_UK` and is not exact. The prefill interface is therefore a capture (forward
hooks on the `kv_a_layernorm` output and on post-RoPE `k_pe` during the
reference prefill), not a conversion. It is exact and one-time.

**F3. The MoE graph template is the demo, not the builder class.**
`python/mirage/mpk/models/qwen3/builder.py` contains no `moe_*` calls. The MoE
graph is built inline in `demo/qwen3/demo_30B_A3B.py:694-747`
(`moe_topk_softmax_routing_layer`, `gang_moe_w13_linear_layer`,
`moe_silu_mul_layer`, `gang_moe_w2_linear_layer`, `moe_mul_sum_add_layer`)
with the head at `:795-801` (`argmax_partial_layer`, `argmax_reduce_layer`).
Expert weights are packed at `:381-392` by concatenating `gate_proj` and
`up_proj` per expert and stacking over experts.

**F4. The Python layer API has no MLA entry.**
`python/mirage/mpk/persistent_kernel.py` defines 48 `*_layer` methods
(`attach_input` at line 469 through `verify_layer_dispatcher` at 2390);
`grep -ci "mla\|latent"` returns 0. The design will name the
two additions `mla_decode_split_kv_layer` and `mla_merge_layer` and mark them
NEW wherever they appear.

**F5. The three code reads that gate the graph and sync sections are still
open** (`PROGRESS.md`, Stage 3): `persistent_kernel.cuh` main loop,
`gang_linear_mi300.cuh` plus the `ck_tile/` idiom, and the Qwen3 builder. F1
and F3 are partial results of the first and third; the full reads are in
section 4.

---

## 3. File plan

For each file: what it must establish, the inputs that already exist, and the
gap that has to be closed before it can be written correctly.

### `00-decisions.md`

Consolidates the "Decisions locked" table from `PROGRESS.md` and the
recommended strategy in `docs/fleet/07-gap-analysis.md` into one table:
decision, alternatives considered, rationale, evidence, what would reverse it.
Adds the decisions made by this plan: one launch per generation (F1), capture
not conversion (F2), demo-style inline graph (F3).
Gap: none. Consolidation only.

### `01-execution-flow.md`

One decode step from the previous token id to the next, then the outer loop.
Inputs: `docs/deepseek-v2-lite/04-tensor-flow.md` (per-op shapes for layer 0,
layers 1-26, head), `01-config.md`. Gap: the outer loop per F1 (embedding of
the token written by the previous iteration, `step` and position bookkeeping,
RoPE position for the new token, EOS handling under greedy decode).

### `02-task-graph.md`

The full per-token graph, not only layer 1. Inputs:
`docs/fleet/06-our-task-graph.md` (layer 1, 80 tasks, placement, edge
classes). Gaps:

- Layer 0 (dense MLP, `intermediate_size` 10944) and the head (final norm,
  `lm_head` N-split over 102,400 rows, partial argmax, reduce) with exact task
  counts. The current "~45" and "~10" are estimates; the totals are summed by
  script.
- Cross-layer edges and the cross-iteration edge (F1).
- Every task expressed as a `persistent_kernel.py` `*_layer` call with its
  actual argument convention, so the section is implementable as written. The
  two NEW calls (F4) are marked and specified by their inputs, outputs, tile
  decomposition, and scope.
- Which existing calls are reused unchanged, which need a shape change (the
  merge, the cache append), and which are new. This is the "Fleet-native
  operations vs remaining fallbacks" metric, stated in advance.

### `03-synchronization.md`

Inputs: `docs/mi300x/03-memory-model.md` (fence sequences),
`docs/fleet/03-runtime.md` (four sync levels, cache policy),
`docs/fleet/05-sync-crosscheck.md`. Gap: the description must be the code's,
not the paper's. Read the scheduler and worker loops in
`persistent_kernel.cuh` (event decrement, per-XCD counters, `TASK_TERMINATE`,
`get_rand_sched_id` XCD alignment at `:584`). Then classify every edge in
`02-task-graph.md` as intra-XCD or cross-XCD and state the fence it carries.
The cross-iteration edge from F1 is included.

### `04-memory-plan.md`

Inputs: `docs/deepseek-v2-lite/05-weights.md` (tensor shapes),
`07-roofline.md` (traffic, L2 residency section),
`docs/mla-decode/04-our-kernel-spec.md` (resource budget, 1 MiB partial
buffer per layer). Gap: derive the resident map (31.4 GB of weights; W13
fusion `[64, 2816, 2048]` per F3; shared experts fused to 2816), the KV-cache
buffers `c_KV[S_max][512]` and `k_pe[S_max][64]` with `S_max = 1024 + 32`,
workspace and intermediate tensors (the `moe_*` tensors the demo allocates),
per-task LDS and VGPR budget, and the cache-policy bits assigned to each buffer
class (weights, KV cache, activations, partials).

### `05-prefill-interface.md`

Inputs: F2, `docs/deepseek-v2-lite/02-mla.md`. Gap: specify the hook points
by module name, the captured tensors with dtype and shape per layer, the
position and RoPE state handed over, the reference logits for the first
decode step, how the 1,024-token prefill hidden state enters layer 1 for the
M2 test, and what is timed. State explicitly that the capture is excluded from
measured decode latency, as the PDF requires.

### `06-optimization-strategy.md`

The section this stage exists for. Inputs:
`docs/acceleration/04-technique-ledger.md`, `02-decode-parallelism.md`,
`03-kernel-craft.md`, `docs/fleet/06` placement section,
`docs/mla-decode/04` split-KV sizing, `docs/mi300x/07-achievable-bandwidth.md`
prefetch depth. Gap: rank each technique by expected microseconds saved per
token against the 1.15-1.35 ms band, with the measurement that decides it and
the fallback if the measurement goes the other way. Items, in the order of
value established so far: task-count collapse (the thing Fleet gives),
N-split placement of dense projections, `P_split` for split-KV, the top-6
experts over 8 XCDs alternatives, latent KV L2 residency via non-temporal
weight loads, prefetch depth 4-8 in every inner loop, MFMA for attention and
VALU for weight GEMVs, and FP8 weight-only as the stretch with its arithmetic.
Each item states what Fleet's own measurements support and what they do not
(`docs/fleet/07`, "what we should promise").

### `07-correctness.md`

Inputs: `docs/deepseek-v2-lite/08-correctness.md`, which is complete on
oracle, boundaries B1-B16, thresholds, divergence sources, and per-milestone
protocol. Gap: the harness. Script names, file formats for the committed
prompt ids, reference token ids, and per-boundary `.safetensors`, how a
boundary is captured on the Fleet side (a debug task that copies a tensor
out, or reading the intermediate tensor after the launch), and the
calibration run that sets thresholds.

### `08-milestones.md`

Inputs: `PROGRESS.md` ladder M0-M5, `docs/fleet/06` milestone mapping. Gap:
a day-by-day schedule over five days with entry and exit criteria per
milestone and two decision gates: end of day 1 (does the runtime build for
gfx942; otherwise the minimal-kernel fallback in `docs/fleet/07`), and end of
day 3 (is M2 validated; otherwise stop extending and document).

### `09-expected-performance.md`

Inputs: `docs/deepseek-v2-lite/07-roofline.md`,
`docs/mi300x/07-achievable-bandwidth.md`, `docs/fleet/07`. Gap: add a
dispatch-overhead term on top of the bandwidth band (per-task scheduler
latency times task count; Fleet's paper gives the inputs) so the band does not
understate. Map every metric the PDF requires to the command that measures
it (`docs/mi300x/06-profiling.md`), including the per-token latency choice
forced by F1.

### `10-local-work.md`

Inputs: `PROGRESS.md` Stage 3. Gap: for each item, the file that will exist
in the repository before GPU day 1: prompt token ids, reference output ids,
per-boundary reference tensors, the BF16 calibration record, the routing log
for 32 steps, the CPU check of runtime reassociation, the weight-packing
script, the DeepSeek-V2 graph script written against the demo template (F3),
the MLA task source written against the `gang_linear_mi300.cuh` idiom,
environment setup scripts.

### `README.md` and `99-open-questions.md`

README: index, one-page summary of what is built, what is promised, what is
not, and a checklist mapping the PDF's items to file headings.
`99-open-questions.md`: design-level unknowns only, in the same format as the
other sets; `OPEN-PROBLEMS.md` gets its Where column updated.

---

## 4. Pre-work that gates the writing

Three reads from `PROGRESS.md` Stage 3 must precede the files that depend on
them. Each produces a short note appended to the relevant `docs/fleet/` file
so the design cites verified behavior.

| # | Read | Produces | Gates |
|---|---|---|---|
| P1 | `persistent_kernel.cuh`: scheduler loop, worker loop, event decrement, per-XCD counters, `prepare_next_batch`, termination | note in `docs/fleet/03-runtime.md` | 02, 03 |
| P2 | `persistent_kernel.py` layer API and `demo/qwen3/demo_30B_A3B.py`: argument conventions of the MoE, gang-linear, argmax, rmsnorm, and rotary calls; tensor allocation; expert packing | note in `docs/fleet/04-repo-map.md` | 02, 04, 10 |
| P3 | `gang_linear_mi300.cuh` and `ck_tile/`: how a Chiplet-task receives its tile index, the worker-side contract, LDS and VGPR usage | note in `docs/fleet/04-repo-map.md` | 02 (NEW entries), 04 |

---

## 5. Writing order

```
P1 -> P2 -> P3
00 -> 01 -> 02 -> 03 -> 04 -> 05        each cites the one before
07, 08, 10                              independent of each other
06, 09                                  last; they depend on every number being fixed
README, 99
PROGRESS.md and OPEN-PROBLEMS.md updates
```

One commit per file. Subject line `docs(design): <file> - <one line>`.

---

## 6. Double-check procedure

Runs after the set is written, as a separate pass from the writing.

1. Every number traces to a script under a `sources/` directory or to a line
   in a primary source. Task counts are summed by script, not by hand; the
   earlier 84 to 80 correction came from a hand sum.
2. Every `*_layer` name in `02` is grepped against `persistent_kernel.py`;
   the NEW ones are confirmed absent.
3. Every fence claim in `03` is cross-checked against the sequences in
   `docs/mi300x/03-memory-model.md` and against the code path found in P1.
4. Every constraint and metric in section 1 maps to a heading; the README
   checklist is filled from that mapping, not from memory.
5. Relative links resolve. No emoji. Verification-level tags on anything not
   `primary` or `checkpoint`.
6. Every claim in section 2 of this plan is re-verified against the cited
   lines, since the plan itself contained one error before it was checked
   (section 7).
7. A reviewer pass over the finished set that did not write it.

---

## 7. Corrections made while checking this plan

- The first draft of this plan stated that Fleet launches the persistent
  kernel once per decoded token, citing `demo/qwen3/demo.py:1196`. That loop
  is the PyTorch reference path. The Fleet path is one launch per generation
  with in-kernel iteration (F1). The design's launch count, its graph (a
  cross-iteration edge), and its latency measurement method all follow from
  the corrected fact.

---

## 8. Not settled by this plan

Produced by the pre-work or by the files themselves, not assumed here:

- Exact task counts for layer 0 and the head.
- Whether the Fleet path should be run with `max_new_tokens=1` for per-token
  latency, or read in-kernel timestamps; decided in `09`.
- Whether the DeepSeek-V2 graph lives as a demo script or as a
  `models/deepseek_v2/` builder; the demo template is closer (F3), the
  builder is cleaner. Decided in `00` after P2.
- The per-task scheduler latency figure for the dispatch-overhead term in
  `09`; taken from the Fleet paper if it gives one, otherwise stated as a
  measurement.
