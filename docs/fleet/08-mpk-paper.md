# 08 - The Mirage Persistent Kernel paper, read for our question

Source: arXiv 2512.22219 v2 (revised 10 June 2026), "MPK: A Compiler and
Runtime for Mega-Kernelizing Tensor Programs", 14 pages. Fleet is a fork of
this runtime, so the paper is the origin of the task/event model the code
read in `03-runtime.md` describes. Read on 2026-09-14 from the abstract
page and the HTML rendering; sections 4.2 onward reached us as an
extracted summary rather than verbatim text, so numbers from those
sections are marked (summary) and should be spot-checked against the PDF
before being quoted in the report. The paper is NVIDIA-only (A100, H100,
B200); it says nothing about AMD, chiplets, or MI300X.

## What it settles for us

The question we brought was DQ1, the exposed latency per operator boundary
on a 326-boundary chain. **The paper gives no per-event or per-task latency
figure.** Its only overhead number is aggregate: the in-kernel scheduler
"accounts for only 0.28% of total runtime" (Sec. 6.6, B200, summary). So
DQ1 stays a day-3 measurement; nothing here changes
`09-expected-performance.md`.

What it does give is the mechanism by which that share is kept low, and
Fleet's prelaunch rewrite (`03-runtime.md`) is the same idea:

- **AOT pre-enqueue.** Tasks of operators with data-independent duration
  are pre-enqueued on a round-robin worker at graph time and "only need to
  wait for event e to be activated", one synchronization step; only
  data-dependent operators (attention) go through the scheduler ("JIT"),
  which costs two steps, worker to scheduler to worker (Sec. 5.2, Fig. 8).
  In Fleet's prelaunch rewrite every task waits on a counter in worker
  queue order, which is the AOT path applied to everything.
- **Normalization** inserts empty tasks so every task has at most one
  dependent and one triggering event; for LLM forward passes the graph is
  "almost entirely sequential: no fork/join groups" and the overhead is
  under 1% (Sec. 4.1). Our chain model is the same shape.
- **Linearization** stores each event's successor tasks as a contiguous
  index range, which is what `runtime.cc` does with `first_task_id` and
  `last_task_id`.

## Numbers worth keeping

| Item | Value | Where |
|---|---|---|
| Task descriptor size | 352 bytes; descriptors prefetched into shared memory | Sec. 5.3 |
| Baseline launch cost, B200 | eager 3.8 us per launch; CUDA Graphs 0.8 us; Qwen3-8B has 293 launches per token, 1.1 ms or 0.2 ms per token | Sec. 6.1 (summary) |
| Scheduler share of runtime | 0.28% | Sec. 6.6 (summary) |
| Cross-task pipelining gain | 1.2-1.3x on Qwen3-8B's final linear, B200 (Fig. 12) | Sec. 5.3 |
| Shared-memory pages | 32 KB pages; 5 / 7 / 7 per SM on A100 / H100 / B200 | Sec. 5.3 |
| Worker / scheduler split | A100 104 / 16 SMs; H100 128 / 16; B200 144 / 16 (4 SMs, 4 scheduler warps each) | Table 1 (summary) |
| Batch-1 speedup | 1.0-1.7x over vLLM and SGLang, larger for small models and newer GPUs | Sec. 6.2 |
| Only absolute latency | Qwen3-8B on A100: 14.5 ms per token (vLLM, SGLang) to 12.5 ms (MPK); hardware bound about 10 ms (16 GB at 1.6 TB/s) | Sec. 6.2 (summary) |
| Task counts (B200) | Qwen3-8B: 293 ops, 47.3 tasks per op, 13,867 tasks, 2,366 events; Qwen3-30B-A3B: 533 ops, 32.2 tasks per op, 1,142 events | Table 2 (summary) |
| Event fusion | reduces events 37x to 118x; linearization shrinks the graph 4.4x to 15x | Table 2 (summary) |

For comparison, our graph is 326 operators, 1,880 tasks and 327 events per
token: 5.8 tasks per operator against MPK's 32-47, because at batch 1 on
MI300X the gang tasks put one task per XCD and the tiles inside the task.
That is the Fleet difference, not an MPK one.

## Task granularity, MoE, attention

- Decomposition partitions each operator's output and picks the partition
  that "minimizes data loading from device memory to shared memory"; by
  default "a number of tasks proportional to the number of SMs" (Sec. 4.1).
  No statement of an optimal task size; the 32 KB page is the only budget.
- MoE (Sec. 6.4, summary): routing, dispatch, expert compute, combine as a
  task chain; a "hybrid workload balancer" passes meta-tensors (activated
  experts, tokens per expert) so tasks adapt to skew; gather fused into the
  GEMM load phase (the standalone gather was up to 11% of SGLang's MoE time
  on Qwen3-30B-A3B). Fleet's `routing`/`mask` tensors are those
  meta-tensors.
- Attention is JIT because its duration is data-dependent; no split-KV
  discussion was found.
- Fences and memory model: nothing. The paper mentions only "semaphores in
  device memory", atomicAdd-based queues and intra-SM barriers. MAJ-3 (the
  agent-scope fence on MI300X) is ours alone.

## Limitations the paper states

Register usage is the maximum across task types (our MAJ-4); four SMs are
lost to schedulers; decentralized scheduling is contrasted with a future
"globally coordinated" one; porting "only requires updating the per-task
code generators". The artifact is CUDA 12.8 and NVSHMEM.

## Effect on our documents

- `docs/fleet/99-open-questions.md` Q5 (per-task overhead) and
  `docs/design-doc/99-open-questions.md` DQ1: unchanged, no source number.
- `docs/fleet/01-paper-review.md`: the Fleet paper's dispatch-reduction
  claim now has its baseline; MPK's own per-launch figures (3.8 us eager,
  0.8 us CUDA Graphs) are the numbers behind "800-1,000 launches per token"
  being expensive.
