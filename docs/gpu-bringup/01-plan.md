# 01 - Hardware collection plan

What is recorded in the first hour on the Hot Aisle VM, before
`env/setup.sh` starts, and why. The design was written from documentation
alone; this session turns each machine assumption it leans on into a
measured fact, and files the ones that disagree.

## Why before the build

- The VM is deleted after every session (see "The rented machine"), so
  anything not written into the repo is lost. The record is committed and
  pushed before the VM goes away.
- The model download (31 GB) starts first and runs in the background; it
  uses the network, the collection uses the GPU. The Fleet build does not
  overlap: `pip install -e .` saturates every core for 45 to 90 minutes and
  would disturb the three timing groups (E, H, I). It starts after the
  collection.
- Three of the numbers (achievable bandwidth, the XCD placement rule and
  the fence lowering) decide whether the expected-performance band and
  the synchronization argument of the design hold on this machine. If one
  is off, the day-1 gate decision needs to know it before the graph runs.
  They are settled independently of whether Fleet builds.

## Before the session, on the laptop

The script and the probes are written and compiled for gfx942 before a VM
exists, so no paid minute is spent typing or fixing a compile error.
`env/offline_gfx942/run.sh` already compiles gfx942 device code with ROCm
7.0's hipcc in Docker without a GPU; the probes go through the same path.

```
env/collect_hw.sh                 one command; runs everything below
env/hw/probes/stream_read.cu      group E
env/hw/probes/xcc_map.cu          group F
env/hw/probes/fence_probe.cu      group G
env/hw/probes/chase.cu            group H (latency, fence cost, cross-XCD round trip)
env/hw/probes/copy_bytes.cu       group I
env/hw/probes/occupancy.cu        group B (residency and wall-clock rate)
```

Each probe has a fixed command line, listed in `02-checklist.md`, so two
people produce the same measurement. `OFFLINE_COMPILE=1 bash env/preflight.sh`
compiles all six in Docker (`env/hw/probes/compile_offline.sh`) and must
pass before the session. On 2026-09-15 the whole collection ran in about
one minute on the VM, plus four for BabelStream.

## What is produced on the VM

```
env/hw/<YYYYMMDD>/raw/            every command's stdout, one file per command
env/hw/<YYYYMMDD>/summary.md      the checklist of 02-checklist.md filled in
```

The summary is the deliverable: one line per check with the measured
value, the expected value, and PASS, MISMATCH, UNAVAILABLE or INFO.
Mismatches are copied into `OPEN-PROBLEMS.md` and the owning
`99-open-questions.md` with the date and the command, which is the
convention every doc set already uses.

## The collection, by group

Each group names the design assumption it verifies and the file that
holds that assumption. The expected values and their sources are in
`02-checklist.md`. Times are command time; the session box is in
"Procedure".

### A. Identity and software (2 min, no GPU time)

`rocminfo`, `amd-smi version`, `amd-smi list`, `amd-smi static` (asic,
bus, driver, vbios), `hipcc --version`, `/opt/rocm/.info/version`,
`dpkg -l | grep -i rocm`, `ls /opt/rocm/include` (whether CK or ck_tile
headers ship with the install), `uname -a`, `/etc/os-release`,
`/sys/module/amdgpu/version`, `docker --version`, `python3 --version`,
whether `torch` is importable in the system Python.

Verifies: gfx942 target; ROCm 7.0 or newer (Fleet's README requirement;
the offline compile used hipcc 7.0.51831, `env/offline_gfx942/README.md`);
whether the CK version on the machine differs from the one the offline
compile used (d8ee107a). The number of GPUs the VM exposes and their bus
ids decide `HIP_VISIBLE_DEVICES=0` for every later run.

### B. Topology and residency (2 min)

`rocminfo` agent block for the GPU; `/sys/class/kfd/kfd/topology/nodes/*/properties`
(`simd_count`, `simd_per_cu`, `wave_front_size`, `max_waves_per_simd`,
`array_count`, `simd_arrays_per_engine`, `cu_per_simd_array`,
`lds_size_in_kb`, `num_xcc`, `max_engine_clk_fcompute`; there is no
`cu_count` property, the CU count is `simd_count / simd_per_cu`);
`amd-smi static --asic`. Then probe `occupancy.cu`: a 256-thread kernel
whose register footprint is forced to about 182 VGPRs, queried with
`hipOccupancyMaxActiveBlocksPerMultiprocessor` twice, once with no
dynamic LDS and once with the 58,368 bytes the runtime launches the
worker kernel with (`runtime_header.h`, `MAX_DYNAMIC_SHARED_MEMORY_SIZE`
= 60 KiB minus the 3 KiB static reserve), and the wall-clock rate from
`hipDeviceGetAttribute(hipDeviceAttributeWallClockRate)`.

Verifies: 304 CUs and 8 XCDs (`docs/mi300x/01-architecture.md`); the
runtime's worker count of 296 plus 8 schedulers derives from the CU count
(`docs/fleet/99-open-questions.md` Q2); wavefront 64; 4 SIMDs per CU;
the L2 per XCD, LDS per CU and any reported last-level cache size
(`01-architecture.md`; the Infinity Cache row is from a secondary source
only, Q13). The occupancy query is the check `docs/mi300x` Q6 names: the
worker kernel is 182 VGPRs and 2 waves/SIMD by the compiler's report
(`env/offline_gfx942/resources.txt`), but that figure ignores the
launch-time LDS request, which with 2,256 B of static LDS leaves room
for one block per CU; the expected answer is therefore one block per CU,
one wave per SIMD, and 304 blocks exactly co-resident on 304 CUs. A grid
that is not co-resident deadlocks the megakernel. The wall-clock rate is the constant that turns
`s_memrealtime` ticks into seconds for every in-kernel time number in the
project, including the per-boundary latency of MAJ-5 (group H) and the
runtime's event timing (`docs/mi300x/04-persistent-kernel.md`,
"S_MEMREALTIME ... does not vary with engine clock").

### C. Partition mode (1 min)

The four candidates `env/setup.sh` already tries, in that order:
`amd-smi static --partition`, `amd-smi partition`,
`amd-smi static -g 0 --partition`,
`rocm-smi --showcomputepartition --showmemorypartition`. Then
`amd-smi set --help`, read only, to learn whether a set is offered to a
guest at all; no set is executed.

Verifies: SPX and NPS1, the one-agent-eight-L2 model that
`docs/mi300x/03-memory-model.md` and `03-synchronization.md` rest on;
settles Q3 (the query syntax). On a VM the partition is usually fixed by
the host: if it is already SPX+NPS1 and cannot be changed, the check is
PASS with the note that the remediation line of `check_day1.sh` check 1
(`amd-smi set --compute-partition ...`) is unusable in a guest.

### D. Clocks, power and memory (1 min)

`amd-smi metric --clock --power --mem-usage --temperature`
(`--throttle` if the version has it), `amd-smi static --limit --vram`,
`amd-smi process`, `rocm-smi -a`, plus the "Max Clock Freq. (MHz)" line of
`rocminfo` and kfd `max_engine_clk_fcompute` for the maximum engine clock.

Verifies: the 5.3 TB/s HBM peak the roofline divides by
(`docs/deepseek-v2-lite/07-roofline.md`) assumes the memory clock the
part is rated for; 192 GB of HBM, of which the design needs about 63 GB
at peak: under 32 GB resident for the Fleet path
(`docs/design-doc/04-memory-plan.md`) alongside the 31 GB HF reference
model during the cache capture. The engine clock converts `s_sleep`
counts (`04-persistent-kernel.md`). The power cap and throttle flags are
recorded because a throttled VM measures below the band without any
error message. `amd-smi process` names the holder if memory is in use
before we start.

### E. Achievable bandwidth (6 min)

Probe `stream_read.cu`: a read-only reduction over a working set W with
unroll depth N loads issued before the first wait. Two occupancies:

- `full`: `hipOccupancyMaxActiveBlocksPerMultiprocessor` blocks per CU
  times 304 CUs, 256 threads each, the kernel's maximum residency.
- `one`: 304 workgroups of 256 threads, one per CU, forced with
  `__launch_bounds__(256, 1)` and a dynamic LDS request above 32 KiB so
  that no CU can hold two; the launch prints the occupancy query as
  evidence. This is one wave per SIMD, the worker kernel's occupancy on
  the machine (group B). `--grid 608` forces two per CU the same way,
  with a 32 KiB request, and prints its query too: the register-limited
  ceiling the compiler reports, an upper bracket.

Runs:

- `full`, W = 1 GiB, N = 8: the achievable read bandwidth, compared with
  the Dot line of AMD's acceptance thresholds (3.66 TB/s at 69.1%,
  `docs/mi300x/07-achievable-bandwidth.md`). Three times, for the spread.
- `one` and 608, W = 1 GiB, N in {1, 2, 4, 8, 16, 32}: the prefetch-depth
  curve. Expect a knee at N about 4 and a plateau within 10% of the full
  number (Q14, MIN-22). A plateau well below it means a queue limit binds
  before `VMCNT` does, and the prefetch-depth requirement in every kernel
  (`docs/design-doc/06-optimization-strategy.md`) is not enough.
- `full`, N = 8, W in {4, 16, 32, 64, 128, 256, 512, 1024} MiB: the cache
  plateau. A step above 32 MiB (the sum of the L2s) and another above
  256 MiB is the Infinity Cache (Q13); the design counts on nothing
  above HBM, so a large plateau is an opportunity to note, not a
  correction.

Vendor cross-checks: HIP BabelStream at AMD's own configuration
(`-n 50 -s 268435456`, one GPU), Dot against 3,660,781 MB/s and Copy
against 4,177,285 MB/s (`07-achievable-bandwidth.md`, the acceptance
thresholds), built from source on the VM if the compile is quick, else
skipped; and `rocm-bandwidth-test -a` for host-to-device bandwidth, which
sizes the 31.4 GB weight upload of session 2 and is not an HBM number.

### F. Workgroup-to-XCD placement (2 min)

Probe `xcc_map.cu`: every workgroup writes its `HW_REG_XCC_ID` (read with
the same `s_getreg_b32` spelling Fleet uses, `docs/mi300x/02-chiplet-dispatch.md`)
and its block id. Grids of 296 and 8 (the worker and scheduler grids the
runtime launches on two streams in the split mode the design uses,
`persistent_kernel.cuh` around line 2287), 304 (the single-kernel path),
608, 1000 and 37; each launched three times, the 8-block grid also
concurrently with a running 296-block grid on another stream.

Verifies: the round-robin rule `xcd == blockIdx.x mod 8` that the design
uses to put eight consecutive gang tasks on eight distinct XCDs and that
the runtime's scheduler-queue indexing assumes (Q1 in `docs/mi300x`,
Q10 in `docs/fleet`, MIN-25). The 8-block grid is the one MIN-25 hangs
on: scheduler block k must land on XCD k. The probe settles the dispatch
rule for a plain launch; the `[SCHED_XCD]` lines of `check_day1.sh`
check 3 later settle it for the runtime's own launch. Run-to-run
stability is recorded because a rule that holds on average and not on
every launch is the worst case for the runtime.

### G. Fence lowering on the machine's compiler (1 min, no GPU time)

Probe `fence_probe.cu`: four kernels, each with a `volatile` global store
before and a load after the fence so the backend cannot drop it, matching
the sequences of `docs/mi300x/03-memory-model.md`:
`__builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent")`,
`__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent")`, `__threadfence()`,
and an agent-scope `__hip_atomic_load`. Compiled with
`hipcc --offload-arch=gfx942 -S --offload-device-only`.

Verifies: `buffer_wbl2 sc1` on release and `buffer_inv sc1` on acquire
(`03-memory-model.md`, `docs/mi300x` Q4, MAJ-3), on this hipcc rather than
the offline 7.0.51831. The `__threadfence()` kernel was compiled offline
first (2026-09-15): on hipcc 7.0.51831 it lowers to the agent-scope pair,
which settled a contradiction the repo had held (`03-memory-model.md`
item 5 said agent scope, the offline census in MAJ-3 had attributed the
72 system-scope sites of the worker kernel partly to it; the census text
is corrected). On the machine the row confirms the same on its hipcc.
`check_day1.sh` check 4 repeats the grep on the megakernel's disassembly
after the build; the per-task-path half of MAJ-3 is the day-2 read of the
generated `kernel_0.cu`.

### H. Latency, fence cost and the cross-XCD round trip (6 min)

Probe `chase.cu`, three modes:

- `--size`: one thread follows a random cyclic permutation of 8-byte
  pointers for 2^20 dependent loads over working sets of 1 MiB (L2),
  64 MiB (Infinity Cache if it exists), 1 GiB (HBM); reports nanoseconds
  per load from the wall clock and from `s_memrealtime` at the rate
  measured in group B.
- `--fence`: 10^5 iterations of store, fence, load; agent-scope release,
  agent-scope acquire, and the workgroup-scope pair as the baseline
  (`docs/mi300x` Q5 asks for the difference); once from a single
  workgroup and once with eight workgroups on eight XCDs running the same
  loop.
- `--pingpong`: two workgroups, placed on two different XCDs by reading
  `HW_REG_XCC_ID` and retiring workgroups on the wrong one, alternate a
  release-store and an acquire-poll on one flag 10^4 times; reports the
  one-way latency.

Verifies: the prefetch-depth argument uses an assumed HBM latency between
250 ns and 2 us because no published figure was found (MIN-21,
`docs/mi300x/07-achievable-bandwidth.md`); the cost of one
`buffer_wbl2 sc1` and one `buffer_inv sc1` (MIN-16, Q5); and the
round trip itself, which is the four-part boundary cost of
`docs/design-doc/09-expected-performance.md` (release flush, cross-XCD
atomic, poll wake, acquire) measured directly, the `t_b` that MAJ-5 and
DQ1 turn on. The chase numbers are INFO; the round trip is the first
estimate of the number that decides whether the graph is restructured
before M3.

### I. Profiler (3 min)

`rocprofv3 --list-avail` (with `-L` and `--list-metrics` as fallbacks,
as `check_day1.sh` check 6 does) grepped for the counters `measure.py`
sums and the two sub-counters the documented decomposition needs; then
one real collection on probe `copy_bytes.cu`, a copy of exactly 1 GiB,
with `TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_32B_sum TCC_EA0_WRREQ_sum
TCC_EA0_WRREQ_64B_sum TCC_HIT_sum TCC_MISS_sum`, and one
`--kernel-trace` run.

Verifies: the counter names (Q11); whether counters can be read inside
the VM at all; and the bytes-from-requests arithmetic (Q12). Two forms
are evaluated against the known 1 GiB read and 1 GiB written: the
decomposition of `docs/mi300x/06-profiling.md` (reads
`32 x RDREQ_32B + 64 x (RDREQ - RDREQ_32B)`, writes
`64 x WRREQ_64B + 32 x (WRREQ - WRREQ_64B)`) and the flat 64 bytes per
request that `harness/measure.py` currently uses. Whichever reproduces
the copy within 5% is the one `measure.py` keeps. If PMC access is
blocked in the VM, the fallback is recorded now: traffic is reported from
the byte count of the graph (`docs/design-doc/sources/graph_counts.py`)
and bandwidth from time over those bytes, with the profiler column marked
unavailable.

### J. Host, disk and network (2 min)

`lscpu`, `nproc`, `free -g`, `df -h`, `ulimit -a`, the download rate from
the first minutes of the Hugging Face log, `~/.cache` location and free
space, and one container start with the GPU passed through.

Verifies: the build-time estimate (the cmake and cargo halves of the
Fleet build use every core), that 31 GB of checkpoint plus a few GiB of
build products fit, and how long a fresh VM needs before the reference
run can start. The download rate decides whether the model is fetched
per session or should be included in the saved Docker image; the
container start is the saved-image plan itself.

## Procedure on the VM

1. `ssh hotaisle@<ip>`; the repo goes over by rsync from the laptop (the
   VM has no GitHub credentials; the command is in `06-agent-guide.md`)
   and the record comes back the same way, to an absolute path.
2. Start the model download in the background without depending on the
   venvs, which do not exist yet: `mkdir -p env/logs`, then
   `python3 -m venv --without-pip /tmp/hfdl && curl -sS https://bootstrap.pypa.io/get-pip.py | /tmp/hfdl/bin/python3 - -q && /tmp/hfdl/bin/pip install huggingface_hub`
   (a throwaway venv made without `ensurepip`, which the image lacks;
   `pip install --user` is refused on an externally managed system
   Python), then
   `nohup /tmp/hfdl/bin/python3 -c "from huggingface_hub import snapshot_download; snapshot_download('deepseek-ai/DeepSeek-Coder-V2-Lite-Base')" > env/logs/download.log 2>&1 &`.
3. `bash env/collect_hw.sh`. It writes `env/hw/<date>/raw/*`, compiles
   the six probes with the machine's hipcc, and prints the summary table.
4. Read the summary against `02-checklist.md`. Every MISMATCH gets a line
   in `OPEN-PROBLEMS.md` and the owning `99-open-questions.md`.
5. rsync `env/hw/<date>` back to the laptop, `git add` and commit it there
   before anything else happens on the VM (a laptop dry run of the script
   on the same UTC date once deleted the record: `collect_hw.sh --out`).
6. Only then `SKIP_DOWNLOAD=1 bash env/setup.sh` (the download is
   already running) and the rest of `docs/design-doc/11-day1-runbook.md`.

Time: 26 minutes of command time across the groups; box the whole of
steps 1 to 5 at 45 minutes, which covers the clone, the probe compiles,
reading the summary and the commit. The VM is billed for one hour at
minimum either way.

## What a VM can hide

- Partition set, power cap set and clock pinning are usually denied to a
  guest. Each is recorded as UNAVAILABLE with the read value rather than
  FAIL, because the design needs the value, not the ability to change it.
- Performance counters need the host to expose them. If `rocprofv3` lists
  counters but the collection returns zeros or an error, group I records
  the fallback and `measure.py` reports bandwidth from time and computed
  bytes only.
- A 2x VM exposes two GPUs, a 1x (round 2) one. Every probe and every
  later run pins `HIP_VISIBLE_DEVICES=0`; group A records which bus id
  that is, and on a 2x the second GPU is idle and noted as such.
- Clocks under a hypervisor can sit at a lower state until load arrives.
  The bandwidth probes warm up for one second before timing.
- `rocminfo` may list only L1 and L2 cache rows. A missing L3 row is not
  evidence against the Infinity Cache; only the plateau of group E is.

## The rented machine

Facts about Hot Aisle that the plan depends on, with their sources
(read 2026-09-15):

- Billing is per minute after a one-hour minimum reservation; only
  deleting the VM ends billing, a stopped VM keeps billing, and deletion
  removes all data; there are no persistent volumes
  (`hotaisle-cli`, `cmd/cli/command_virtual_machine.go`: "Delete ... Ends
  billing", "stop ... Continues billing"; https://admin.hotaisle.app/:
  "When you're finished, it is deleted forever").
- Shapes (https://hotaisle.xyz/pricing, and the provisioning list in the
  admin TUI on 2026-09-15): 1x MI300X with 8 or 13 cores and 224 GB RAM;
  2x MI300X with 26 cores, 448 GB RAM and 13 TB disk at $5.98 per hour.
  Only the 2x was listed on 2026-09-15.
- The image ships "a recent ROCm setup, and Docker or Podman"; no
  PyTorch; ssh as `hotaisle@<ip>` on port 22 only
  (https://hotaisle.xyz/quick-start/, the `cloud-init-templates` README).

## After the session

The summary and the raw files are the primary source for the numbers the
final report quotes about the machine. Updated from it, each with the
date and the command: `docs/mi300x/99-open-questions.md` Q1, Q3, Q4, Q5,
Q6, Q11, Q12, Q13, Q14; `docs/fleet/99-open-questions.md` Q6 and Q10;
`OPEN-PROBLEMS.md` MAJ-3, MAJ-5 (first `t_b` estimate), MIN-16, MIN-21,
MIN-22, MIN-25; `docs/mi300x/03-memory-model.md` item 5 or
`env/offline_gfx942/fences.txt`, whichever the `__threadfence()` probe
contradicts; `harness/measure.py` if the flat 64-byte form loses;
`docs/design-doc/09-expected-performance.md` gets the measured band in
place of the documentation-derived 3.66 to 4.3 TB/s if they differ.
