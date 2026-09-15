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
| Workgroup k lands on XCD (k + 4) mod 8 | the runtime's scheduler read the queue of another XCD | `fleet/patches/sched_xcd.patch`; the offset is presumably boot- or VF-dependent, so the day-1 check accepts any constant |
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

12. **The E4 working-set sweep needs constant loads per thread.** The
    first version changed code path with size (the reviewer's finding);
    the `--passes` flag fixes it. Re-run E4 alone next session (seconds)
    to get the MALL size from the plateau, which idea 4 would like to know.
