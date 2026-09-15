# 02 - Checklist

One row per check. `Expected` is what the design assumes and where that
assumption is written, or INFO when the repo holds no expectation and the
value is recorded for the report; `Settles` is the open question or
problem the measured value closes or narrows. `env/collect_hw.sh` fills
`Measured` and `Result` into `env/hw/<date>/summary.md`; a copy of this
table with those columns is the record.

Result values: PASS (matches expected), MISMATCH (differs; filed in
`OPEN-PROBLEMS.md`), UNAVAILABLE (the VM does not expose it; the read
value, if any, is kept), INFO (no expectation). Where an expected value
comes from general knowledge rather than a repo source it says "gk"; a
MISMATCH there corrects the checklist, not the design.

Probe command lines are fixed here; `full` and `one` occupancy are defined
in `01-plan.md` group E.

## A. Identity and software

| # | Check | Command or file | Expected | Settles |
|---|---|---|---|---|
| A1 | GPU name and target | `rocminfo` (Name, Marketing Name) | `gfx942`, MI300X | target of every build flag (`00-decisions.md` D1) |
| A2 | Number of GPUs and their bus ids | `rocminfo`, `amd-smi list`, `amd-smi static --bus` | 1 or 2; device 0's BDF recorded | `HIP_VISIBLE_DEVICES=0` for every run |
| A3 | ROCm version | `/opt/rocm/.info/version`, `amd-smi version` | 7.0 or newer | Fleet README requirement; `env/setup.sh` warns below 7 |
| A4 | hipcc version | `hipcc --version` | same major as the offline 7.0.51831 | whether the offline compile results transfer (`env/offline_gfx942/README.md`) |
| A5 | Driver version | `/sys/module/amdgpu/version`, `amd-smi static --driver` | INFO | report |
| A6 | CK headers shipped with ROCm | `ls /opt/rocm/include/ck_tile /opt/rocm/include/ck` | INFO: absent or a version | whether a machine CK could replace deps/composable_kernel d8ee107a (not used; Fleet reads its own copy) |
| A7 | OS, kernel, Docker, Python | `/etc/os-release`, `uname -r`, `docker --version`, `python3 --version` | INFO; Ubuntu with Docker is what the vendor page says (`01-plan.md`, "The rented machine") | `env/setup.sh` venvs; the saved-image plan |
| A8 | System torch | `python3 -c "import torch"` | INFO; expected absent (vendor page: ROCm and Docker only) | the two venvs install their own |

## B. Topology and residency

| # | Check | Command or file | Expected | Settles |
|---|---|---|---|---|
| B1 | Compute units | `rocminfo` Compute Unit; kfd `simd_count / simd_per_cu` | 304 (kfd 1216 / 4) | `01-architecture.md`; runtime derives 296 workers + 8 schedulers (`docs/fleet` Q2) |
| B2 | XCD count | kfd `num_xcc` if present; else the distinct ids of F1 | 8 | `MI300X_NUM_XCDS = 8` hard-coded in the runtime |
| B3 | SIMDs per CU, wavefront size | `rocminfo` SIMDs per CU, Wavefront Size; kfd `simd_per_cu`, `wave_front_size` | 4, 64 | `01-architecture.md` |
| B4 | Max waves per CU | `rocminfo` Max Waves Per CU; kfd `max_waves_per_simd` | 32 per CU, 8 per SIMD (gk; `01-architecture.md` does not state it, Q6 says it is not derivable from the sources read) | half of `docs/mi300x` Q6 |
| B5 | LDS per workgroup | `rocminfo` Pool Info GROUP segment size; kfd `lds_size_in_kb` | 64 KiB | `01-architecture.md`; MIN-24 (the 32 KiB accumulator of `mla_attend`) |
| B6 | Workgroup max size | `rocminfo` Workgroup Max Size | 1024 work-items (gk) | kept as its own row so the work-item count is not mistaken for the LDS figure of B5 |
| B7 | L2 size reported | `rocminfo` Cache Info L2 | 4 MB per XCD (`01-architecture.md`) | the acquire-invalidates-L2 argument (`03-synchronization.md`) |
| B8 | Last-level cache reported | `rocminfo` Cache Info L3, if listed | INFO; 256 MB is the secondary figure (`01-architecture.md`); a missing row is not evidence either way | Q13, together with E4 |
| B9 | HBM size | `amd-smi static --vram`, `rocminfo` GLOBAL segment | 192 GB (`01-architecture.md`; about 196,000 MiB reported) | reference model 31 GB + Fleet path under 32 GB fit at once (`04-memory-plan.md`) |
| B10 | Residency of a 182-VGPR, 256-thread kernel with the worker kernel's LDS request | `occupancy --vgprs 182 --dynamic-lds 58368` and `occupancy --vgprs 182` (`hipOccupancyMaxActiveBlocksPerMultiprocessor`) | 1 block per CU, LDS-limited by the 58,368 B dynamic request the runtime launches with (`runtime_header.h`, `MAX_DYNAMIC_SHARED_MEMORY_SIZE`), so one wave per SIMD and 304 blocks exactly co-resident on 304 CUs; 2 blocks per CU register-only (`env/offline_gfx942/resources.txt`) | `docs/mi300x` Q6; MAJ-4; a non-co-resident grid deadlocks the megakernel |
| B11 | Wall-clock rate and `s_memrealtime` calibration | `occupancy --clock` (`hipDeviceAttributeWallClockRate`; ticks over a 100 ms host interval) | INFO; the two agree within 1% | every in-kernel time number: group H, the runtime's event timing, `t_b` of MAJ-5 (`04-persistent-kernel.md`) |

## C. Partition

| # | Check | Command or file | Expected | Settles |
|---|---|---|---|---|
| C1 | Compute partition | the four candidates of `env/setup.sh` step 1, first that works | SPX | one agent, eight L2s (`03-memory-model.md`); `check_day1.sh` check 1 |
| C2 | Memory partition | same | NPS1 | interleaved 192 GB; same |
| C3 | Query syntax that works on this ROCm | which candidate printed | INFO | `docs/mi300x` Q3 |
| C4 | Whether a guest may set the partition | `amd-smi set --help`; no set executed | UNAVAILABLE is acceptable if C1 and C2 pass | what to do if a later session finds CPX |

## D. Clocks, power, memory

| # | Check | Command or file | Expected | Settles |
|---|---|---|---|---|
| D1 | Memory clock, current and max | `amd-smi metric --clock` | INFO; the max is the rated HBM3 clock behind the 5.3 TB/s peak (`07-roofline.md`) | a lower max means the peak, the band and every floor scale down |
| D2 | Engine clock, current and max | `amd-smi metric --clock`, `rocminfo` "Max Clock Freq. (MHz)", kfd `max_engine_clk_fcompute` | INFO; about 2,100 MHz max (gk) | `s_sleep` count conversion (`04-persistent-kernel.md`) |
| D3 | Power cap and current draw | `amd-smi metric --power`, `amd-smi static --limit` | INFO; cap at the part's rating, 750 W class (gk) | a throttled VM measures below the band |
| D4 | Temperature and throttle status | `amd-smi metric --temperature`, `amd-smi metric --throttle` if present | no throttle flags | same |
| D5 | Memory in use and processes before we start | `amd-smi metric --mem-usage`, `amd-smi process` | near 0, no other process | another tenant or leftover process would OOM the reference run |

## E. Achievable bandwidth (`stream_read.cu`, BabelStream, `rocm-bandwidth-test`)

| # | Check | Command or file | Expected | Settles |
|---|---|---|---|---|
| E1 | Read bandwidth, `full`, 1 GiB, N=8, three runs | `stream_read --occupancy full --size 1G --unroll 8 --repeat 3` | 3.66 to 4.3 TB/s (69 to 81% of 5.3); spread under 3% | the band of `09-expected-performance.md`; `07-achievable-bandwidth.md` |
| E2 | Read bandwidth at one wave/SIMD versus N | `stream_read --occupancy one --size 1G --unroll N`, N in 1,2,4,8,16,32 | knee at N about 4, plateau within 10% of E1 | Q14, MIN-22 (prefetch depth suffices; no hidden queue limit) |
| E3 | Same at two waves/SIMD | `stream_read --grid 608 --size 1G --unroll N` (two blocks per CU forced by a 32 KiB LDS request; the occupancy query is printed) | plateau reached at about half the N of E2 (`07-achievable-bandwidth.md`: "at 2 waves/SIMD every figure halves") | the register-limited ceiling the compiler reports (`resources.txt`), an upper bracket; the runtime's occupancy is one wave per SIMD (B10) |
| E4 | Read bandwidth versus working set | `stream_read --occupancy full --unroll 8 --size W`, W in 4,16,32,64,128,256,512,1024 MiB | step above 32 MiB (L2) and above 256 MiB (Infinity Cache, if it exists) | Q13; whether anything above HBM serves the 31.3 MiB latent cache, read at about 30 MiB per token (MAJ-6) |
| E5 | Vendor streaming numbers | HIP BabelStream `-n 50 -s 268435456`, one GPU | Dot at or above 3,660,781 MB/s, Copy at or above 4,177,285 MB/s (AMD acceptance thresholds, `07-achievable-bandwidth.md`) | cross-check of E1; skipped if the source build is not quick |
| E6 | Host-to-device bandwidth | `rocm-bandwidth-test -a` | INFO | time to upload the 31.4 GB of packed weights in session 2 |

## F. Workgroup placement (`xcc_map.cu`)

| # | Check | Command or file | Expected | Settles |
|---|---|---|---|---|
| F1 | XCC_ID readable and in range | `xcc_map --grid 304` | ids in 0..7, all eight present | `02-chiplet-dispatch.md`; the assembler spelling (Q2 resolved) |
| F2 | Placement rule at the worker grid | `xcc_map --grid 296` | `xcd == blockIdx.x mod 8` for every block, 37 per XCD | `docs/mi300x` Q1; the gang-task placement (`03-runtime.md`); `docs/fleet` Q10 |
| F3 | Placement of the scheduler grid | `xcc_map --grid 8`, alone and while a 296-block grid runs on another stream | block k on XCD k, both ways | MIN-25: the scheduler-queue index of `persistent_kernel.cuh` line 591 |
| F4 | Rule at other grids | `xcc_map --grid 608`, `1000`, `37` (F1's 304 map is reused) | same rule; at 37 blocks, 5 on XCD 0..4 and 4 on 5..7 | whether the rule depends on grid size |
| F5 | Stability across launches | each grid three times | identical maps | a rule that holds on average only would lose scheduler events |
| F6 | Blocks per XCD at 304 | count per id | 38 each | 38 CUs per XCD (B1) |

## G. Fence lowering (`fence_probe.cu`)

| # | Check | Command or file | Expected | Settles |
|---|---|---|---|---|
| G1 | Agent-scope release | `hipcc --offload-arch=gfx942 -S --offload-device-only fence_probe.cu`, grep per kernel | `buffer_wbl2 sc1`, no `sc0 sc1` in that kernel | MAJ-3 and `docs/mi300x` Q4 on this compiler; `03-memory-model.md` |
| G2 | Agent-scope acquire | same | `buffer_inv sc1` | same |
| G3 | `__threadfence()` | same | agent scope: `buffer_wbl2 sc1` and `buffer_inv sc1`, no `sc0 sc1` (the offline compile of the probe on hipcc 7.0.51831, 2026-09-15, settled what the repo had held two ways; `03-memory-model.md` item 5 was right, the census attribution in MAJ-3 is corrected); the machine's hipcc must agree | the attribution of the 72 system-scope sites in the worker kernel: printf and assert paths only |
| G4 | Agent-scope atomic load | same | `sc1` on the load, `buffer_inv sc1` after | the counter poll of the runtime |

## H. Latency, fence cost, round trip (`chase.cu`)

| # | Check | Command or file | Expected | Settles |
|---|---|---|---|---|
| H1 | L2 hit latency | `chase --size 1M` | INFO | MIN-21 lower bound |
| H2 | Infinity Cache latency | `chase --size 64M` | INFO; a secondary source gives about 218 ns absolute (`07-achievable-bandwidth.md`, "Unresolved: HBM latency") | Q13 |
| H3 | HBM latency | `chase --size 1G` | INFO; the prefetch analysis holds anywhere in 250 ns to 2 us (`07-achievable-bandwidth.md`); flag only above 2 us | MIN-21; the bytes-in-flight table |
| H4 | Release fence cost, one workgroup | `chase --fence release` (agent minus workgroup scope) | INFO; under 1 us | MIN-16, `docs/mi300x` Q5; the per-boundary floor of MAJ-5 |
| H5 | Acquire fence cost, one workgroup | `chase --fence acquire` | INFO; under 1 us | same |
| H6 | Both with eight XCDs participating | `chase --fence release --xcds 8`, `--fence acquire --xcds 8` | INFO; Q5 asks for this case | same |
| H7 | Cross-XCD one-way latency | `chase --pingpong` | INFO; the secondary cross-workgroup atomic figure is 116 to 202.5 ns (`07-achievable-bandwidth.md`); at 1 us per boundary the chain costs 28% of the band, at 5 us more than the band (MAJ-5) | first estimate of `t_b` (MAJ-5, DQ1, `09-expected-performance.md`) |

## I. Profiler (`copy_bytes.cu`)

| # | Check | Command or file | Expected | Settles |
|---|---|---|---|---|
| I1 | Counter names present | `rocprofv3 --list-avail` (fallbacks `-L`, `--list-metrics`) grep | `TCC_EA0_RDREQ_sum`, `TCC_EA0_RDREQ_32B_sum`, `TCC_EA0_WRREQ_sum`, `TCC_EA0_WRREQ_64B_sum`, `TCC_HIT_sum`, `TCC_MISS_sum` | `docs/mi300x` Q11; `measure.py parse_pmc` keys |
| I2 | Counters readable in the VM | `rocprofv3 --pmc <the six> -- copy_bytes` | non-zero values, no error | whether the traffic column of the report is measured or computed |
| I3 | Bytes-from-requests arithmetic | both forms against the 1 GiB read and 1 GiB written: the `06-profiling.md` decomposition and `measure.py`'s flat 64 B per request | the decomposition within 5%; the flat form recorded either way, and `measure.py` changed if it is off | `docs/mi300x` Q12 |
| I4 | Kernel trace works | `rocprofv3 --kernel-trace -- copy_bytes` | one dispatch row with a duration | the launches-per-token metric (`measure.py parse_kernel_trace`) |
| I5 | TCC instance count | number of `TCC_EA0_RDREQ[n]` instances listed | INFO; 16 channels per XCD times 8 XCDs = 128 in SPX (gk); `06-profiling.md`'s "reportedly [0..31]" is the MI200 figure | Q11 |

## J. Host, disk, network

| # | Check | Command or file | Expected | Settles |
|---|---|---|---|---|
| J1 | CPU model and cores | `lscpu`, `nproc` | INFO; vendor page lists 8 or 13 cores (1x) and 26 (2x) | build time of the cmake and cargo halves |
| J2 | RAM | `free -g` | INFO; vendor page lists 224 GB (1x), 448 GB (2x) | the reference run loads 31 GB through host memory |
| J3 | Disk free | `df -h /`, `df -h ~` | over 100 GiB free | 31 GB checkpoint, build products, saved image |
| J4 | Download rate | first minutes of `env/logs/download.log` | INFO; minutes for 31 GB | whether the checkpoint goes into the saved Docker image |
| J5 | Container with GPU passthrough | `docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add render --security-opt seccomp=unconfined rocm/dev-ubuntu-<A7>:<A3> rocminfo` (image tag chosen from A3 and A7; pulled on the laptop before the session to confirm it exists) | the GPU visible inside the container | the saved-image plan for later sessions |
| J6 | Time from VM creation to first ssh | wall clock | INFO | session planning |

## Sign-off

- [ ] `env/hw/<date>/summary.md` has every row with a Result.
- [ ] every MISMATCH has a line in `OPEN-PROBLEMS.md` and the owning `99-open-questions.md`, dated, with the command.
- [ ] E1 within the band, or `09-expected-performance.md` updated with the measured band.
- [ ] F2, F3 and F5 PASS, or `docs/fleet` Q10 raised to major before the graph runs.
- [ ] G1 and G2 PASS, or MAJ-3 reopened before the graph runs; G3's answer written into `03-memory-model.md` or `fences.txt`.
- [ ] B10 shows at least 1 block per CU with the 58,368 B request, or the worker count is reconsidered before the graph runs.
- [ ] H7 recorded next to MAJ-5.
- [ ] `git commit` and push of `env/hw/<date>/` (or `scp` to the laptop) before the VM is deleted.
