# 03 - Lessons from the first session, and ideas the measurements suggest

Kept as a running list. The first part is what went wrong or was missed on
2026-09-15 and what changed because of it; the second part is what the
measured numbers suggest for the decode path beyond the design as written.
Each idea names the number it rests on and what it would take to try.

## What was missed or went wrong, and the fix

| What | What happened | Fix, and where it lives now |
|---|---|---|
| Python venvs on the image have no pip | `python3 -m venv` succeeds but `ensurepip` is missing, so `pip` does not exist in the venv; `setup.sh` aborted at its first `pip install` | `sudo apt-get update && sudo apt-get install python3.12-venv` first (the install fails with a 404 without the update); to be added to `setup.sh` step 2, the `hotaisle` user has sudo |
| The model download started from a pip-less venv | the throwaway venv for `huggingface_hub` had no pip either | `python3 -m venv --without-pip` plus `get-pip.py`; the download then took 79 s for 30 GB, so the checkpoint never needs to be saved in an image |
| The laptop dry run of `collect_hw.sh` deleted the real record | the dry run writes to `env/hw/<UTC date>` and the laptop's UTC date was still the 15th; the dry-run cleanup removed the directory the VM results had been copied into | the record is committed immediately after every copy; `collect_hw.sh --out DIR` for dry runs; nothing under `env/hw/2*/` is ever deleted |
| An `rsync` with a relative destination nested the record | the shell's working directory had moved into the record directory between two calls | absolute destination paths for every copy off the VM |
| The TUI confirmation ignores Enter | the provisioning dialog's `yes` button did not take Enter through the scripted terminal; `y` did | `y` on the dialog; the VM appears on the team page about ten seconds later, the IP is on the VM's management page |
| `cmake` is not on the image | BabelStream could not configure | `pip install cmake` into a throwaway venv; `collect_hw.sh` does it when `cmake` is absent |
| `amd-smi metric --throttle` is not a flag on amd-smi 25.x | D4 lost the throttle column | temperature is read; D4 is INFO with the reason |
| `rocm-bandwidth-test` is not installed | E6 (host-to-device bandwidth) is UNAVAILABLE | a small HIP `hipMemcpy` timing probe would replace it; the 31.4 GB weight upload of session 2 is the number it sizes |
| Only the 2x MI300X VM was available, at $5.98 per hour | the second GPU is idle | see idea 7 below |
| `TCC_EA0_RDREQ` counts 128-byte requests too | the read formula of `06-profiling.md` and `measure.py` undercounted reads by 2x | `TCC_BUBBLE` term added (validated exact on the 1 GiB copy); `measure.py` updated |
| The profiler sums include every dispatch | fill, warm-up and blit kernels tripled the write count | the summarizer keeps `copy_kernel` rows only; `measure.py` will need the same discipline for the megakernel (one dispatch per generation, so it is simpler there) |
| Workgroup k lands on XCD (k + c) mod 8 with c not 0 | the runtime's scheduler read the queue of another XCD | `fleet/patches/sched_xcd.patch`; the offset is per launch, not per boot: the probe saw c = 4, the worker kernel of the smoke graph c = 5 and its scheduler kernel c = 6 in the same process, so nothing may assume a value, the register is the only source; the day-1 check accepts any constant per kernel |
| `calibrate.py` had never run on a GPU | it compared a captured GPU tensor with a reference loaded to the CPU and crashed on the device mismatch | `compare.metrics` moves both operands to the CPU; `capture()` returns CPU tensors; the smoke tests ran on the CPU only and could not see it |
| pip's isolated build environment chose its own z3 | Fleet's build requirement is unpinned, so the extension linked against z3 5.1 while the venv held the pinned 4.15; `import mirage` failed after a successful build | `PIP_CONSTRAINT` pins the build environment to the venv's z3; the venv's `z3/lib` goes on the loader path through `activate` |
| The CK split-KV wrapper references the gfx950-only decode kernel | the day-1 smoke graph's JIT failed with an undeclared identifier; the offline compile had parsed the file without instantiating that wrapper | a `#if defined(__gfx950__)` hunk in `gfx942.patch` routes the one-token step through the CK prefill pipeline |
| The calibrated floors for `norm` and `logits` are 4.5% and 4.1% | far above the other classes (0.3 to 1.1%); the two-row padded batch changes the accumulation order of the residual stream more than expected | recorded as measured; the 4x threshold rule makes them 18% and 16%, which is loose; to be re-examined with a third ordering before the layer-1 comparison relies on them |
| Per-operator times at 8 iterations are 10 to 100x the bandwidth time | `mla_attend` 209 us, a norm 50 us, `o_proj` 36 us per layer in a 27 ms run: the engine clock idles at 138 MHz and a run this short never ramps it | time only after a warm-up that ramps the clocks (idea 9), and measure with 32 iterations; the 8-iteration numbers are not performance numbers |
| `--iters 64` is impossible | the RoPE tables and `max_seq_length` are sized for 1,024 + 32 positions (`S_MAX` 1056); the graph asserted on the `cos` table shape | 32 iterations is the ceiling of a run without re-sizing the tables; the run-book's timing runs use 32 |
| A head compare on a truncated graph is meaningless | with 2 of 27 layers the final norm, logits and token are compared against the 27-layer reference and fail by construction | `compare.py` should mark head boundaries SKIP unless the graph has all 27 layers; not yet done |
| The route-log SKIP rule misses a graph stopped before the router | the placeholder entries of a truncated run count as routes | the rule should read the plan for a router op before the stop; not yet done |
| The route log of the 27-layer run differs at one slot | step 0, MoE layer 4: reference top-6 [4, 19, 30, 34, 46, 49], Fleet [2, 4, 19, 30, 34, 46]; a near-tie at the sixth slot after four layers of BF16 drift, while layer 1 is exact | the near-tie report the correctness method reserved for B9 (`07-correctness.md`); the head still produced the exact token at 27 layers, so the tie did not change the argmax; `compare.py` should grade a one-slot swap as PARTIAL with the logit gap |
| The engine clock reads 131 MHz while the megakernel runs | `amd-smi metric --clock` sampled during a 27-layer run; the memory clock 900 of 1,300 MHz; the persistent kernel's polling load does not raise the DPM state in the VF | try `amd-smi set --perf-level high` (needs sudo, result pending) and warm up with a bandwidth-heavy kernel before the timed generation; until the clocks are confirmed high, no per-operator time is a performance number |
| The wall-clock attribute is 1.8% off | `hipDeviceAttributeWallClockRate` says 100 MHz; 10,000,000 `s_memrealtime` ticks took 101.87 ms on the host, so the counter runs at 98.2 MHz | every `s_memrealtime`-derived number in the record (group H, including the 703 ns hop) is 1.8% low; the wall-clock column of the chase probe agrees with the corrected value; B11 is graded MISMATCH so the correction is not forgotten when `t_b` is read from the runtime's event timing |
| E3's "half the knee" expectation is unresolved | both the one- and two-wave sweeps knee at N=4, and the E3 verdict turns on a 0.2% margin in a noisy curve | rerun E2 and E3 with `--passes` next session before filing anything against `07-achievable-bandwidth.md` |
| Two graph runs on the two GPUs clobbered each other | the runtime JIT-compiles every graph into one fixed `permanent_output_dir` under the Fleet tree, so concurrent `run_fleet.py` processes overwrite each other's generated source and shared object; one run failed at `hipcc`, the others may have executed a mismatched megakernel | one graph run at a time on a VM, whatever the GPU; the reference, calibration, kernel tests and probes are safe on the other GPU because they do not use the JIT directory; results collected concurrently on 2026-09-15 (`timing_l27_32`, `q_l8h4`, `q_l4h8`) are discarded and re-run sequentially |
| The scripted TUI cannot read the personal-settings page | the SSH keys registered on the account were never listed | the default key worked for both the admin host and the VM, which is what mattered |

## Ideas the measurements suggest

1. **Boundary fusion is worth about 100 us per token.** The measured
   per-boundary cost is release 115 ns plus a 703 ns cross-XCD hop plus
   acquire 137 ns, about 1 us. At 326 boundaries that is 25 to 30% of the
   bandwidth band; the three fusions of `09-expected-performance.md`
   (326 to 218 boundaries) save about 108 us. They were "if large"; they
   are large. Cost: the fused kernels named there. Confirm on the layer-1
   timing first (`t_b` is the direct measurement).

2. **Keep producer and consumer on the same XCD where the chain allows.**
   The hop dominates the boundary cost (703 of about 950 ns). The
   scheduler already dispatches to workers on its own XCD, and a task
   whose consumer is dispatched by the same scheduler skips the hop only
   if the event is signalled to that scheduler. Whether the runtime can
   prefer the local queue for a chain-only graph is a scheduling question
   worth one experiment: signal the local scheduler first for non-gang
   successors. Needs `t_b` per boundary from the layer-1 run to see which
   boundaries are cross-XCD today.

3. **Prefetch depth 4 is enough, and one wave per SIMD is the better
   regime.** At one block per CU the read plateau is 4.3 TB/s, above the
   3.94 TB/s of full residency and above BabelStream Copy at 4.32. The
   megakernel's occupancy is not a bandwidth problem; kernels can stop at
   N = 4 and spend the registers elsewhere (MAJ-4's union shrinks).

4. **A cache tier above L2 exists and answers in 258 ns.** The 64 MiB
   working set hits it, 1 GiB does not. The 31.3 MiB latent cache plus the
   per-layer activations fit in it many times over. The experiment MAJ-6
   names is now concrete: weights loaded with `sc1 nt` (no MALL allocate)
   so the 4.6 GiB per token of weights does not evict the cache lines the
   next layer's attention reads. Worth at most 30 MiB per token of reads
   at HBM cost versus MALL cost, about 7 us per token; small but free.

5. **HBM latency 342 ns fixes the bytes-in-flight budget.** At 4.3 TB/s
   the chip needs about 1.5 MB in flight, 5 KB per CU, which at 256
   threads is 20 B per thread: one `uint4` per thread would do it in the
   ideal case and four cover the queueing. This is why the knee is at
   N = 4 and why deeper unrolls gain nothing. Use it to size every new
   kernel's prefetch, and stop arguing from `VMCNT`.

6. **Fences are cheap, so finer dependencies are affordable.** 115 and
   137 ns per fence, no measurable extra release cost with eight XCDs.
   The design batches dependency resolution per operator; a finer split
   inside `mla_attend` (per split-KV partial) would cost fences only, not
   hops, if the merge task is on the same XCD. Low priority, but it
   removes the "fences are expensive" assumption from the design.

7. **Use the second GPU.** The 2x VM was the only shape available. The
   reference run, calibration and route analysis need a GPU for about 40
   minutes; they can run on GPU 1 while the Fleet build and the first
   graph runs use GPU 0. `HIP_VISIBLE_DEVICES=1` for `run_reference.py`
   and `calibrate.py`; nothing else changes. Saves about 40 minutes of a
   $5.98 hour per session.

8. **Placement offset is a per-boot fact; read it, never assume it.**
   Everything that maps work to XCDs must go through `HW_REG_XCC_ID`
   (the runtime does for workers and, with the patch, for schedulers).
   The expert-to-XCD affinity idea of MIN-11 should key on the physical id
   too, and be re-derived at launch.

9. **The clocks idle at 138 MHz engine and 901 MHz memory.** The first
   iteration of any timed run is slow for that reason alone. The 32-token
   generation should be preceded by a warm-up generation, and the
   measurement matrix should report iteration 0 separately.

10. **Counters work inside the VF.** The full measurement matrix of
    `06-profiling.md` is possible on this machine; the doubt about PMC
    access in a guest is settled. The `TCC_BUBBLE` term must be in every
    traffic figure.

11. **Save the built environment as an image once gate 1 passes.** The
    download is 79 s and the probes compile in seconds; only the Fleet
    build (torch wheels of 6.2 GB each plus cmake and cargo) costs the
    session. A Docker image with both venvs and the built tree, pushed to
    a registry at the end of the first successful build, turns every
    later session's setup into a pull. Measure the build time first (the
    watcher on `setup.sh` gives it).

13. **The gang model is the batch-1 bottleneck, not the memory system.**
    Measured per-operator times are 20 to 100 times the bandwidth time and
    do not move when the clocks are held up (MAJ-7). One workgroup per
    XCD per operator cannot stream more than about 25 GB/s each; the
    design's band assumed the whole machine streams every operator.
    The whole model (`env/hw/20260915/runs/L27_it32`): 15.6 ms per
    iteration, of which `mla_attend` 5.7 ms, `mla_merge_uv` 1.45 ms,
    `moe_silu_mul` 1.1 ms, `o_proj` plus `down` 1.0 ms, and about 4 ms
    outside any operator (boundaries plus the serial `mla_prep`). Two
    ways out, both in the graph builder and the task glue rather than in
    the runtime: give `mla_attend` and `mla_merge_uv` a prefetch loop and
    more splits (they are ours and they are half the time), and issue the
    stock linears and elementwise ops as per-tile tasks (37 per XCD, the
    runtime's per-task pointer offsets). Expect an order of magnitude;
    the 4 ms of boundaries (idea 1) then becomes the next term.

14. **The head faults when the run is configured for many iterations,
    before the first iteration completes.** Facts from the bisection
    (2026-09-15): with the head, 27 layers at 1 and 2 iterations are
    exact ([25], [25, 16228]) and 2 layers at 4 iterations run; 27 layers
    at 32 and 8 layers at 8, 16 and 32 iterations fault with an illegal
    address before any iteration reports, with and without event timing,
    and with the runtime's queues raised from 1,024 to 16,384 entries (so
    the queues are not the cause); 2 layers with the head run all 32
    iterations. Without the head, 27 layers at 4 and 2 layers at 32
    iterations run. So the fault needs both a deeper graph and a larger
    configured iteration count, which is the signature of a race or of
    a size that scales with both, not of a single index. The only build-time quantity that
    grows with the iteration count is `max_seq_length = 1,024 + K` and
    the buffers sized from it, and only the head path breaks: the first
    suspects are the `lm_head` gang tiles, the `argmax_partial` slices
    and `argmax_reduce`'s `tokens + step + 1` against those sizes. Next
    session: `--layers 2 --head` at 8, 16 and 32 iterations, then the
    runtime's verbose mode to name the faulting task.

12. **The E4 working-set sweep needs constant loads per thread.** The
    first version changed code path with size (the reviewer's finding);
    the `--passes` flag fixes it. Re-run E4 alone next session (seconds)
    to get the MALL size from the plateau, which idea 4 would like to know.
