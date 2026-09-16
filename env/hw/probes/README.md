# Hardware probes

Six HIP programs that measure the machine assumptions the design rests on,
plus the helper that reads the fence assembly. They are written and compiled
for gfx942 before the VM exists, so no paid minute is spent on a compile
error. `docs/gpu/01-bringup/01-plan.md` says why each group is collected and
`docs/gpu/01-bringup/02-checklist.md` gives the row each command line fills.

Every probe behaves the same way at the edges. The GPU is device 0 as HIP
sees it, so the caller selects it with `HIP_VISIBLE_DEVICES`. Every HIP call
is checked; a failure prints `<probe> error: <what> at <file>:<line>` to
stderr and exits 2, and a hard failure that is not a HIP call (an unopenable
dump file, a workgroup that never claimed its XCD) uses the same form and the
same exit code. A bad command line prints `<probe> usage` and the option list
to stderr and exits 1. Results go to stdout as one line of space-separated
`key=value` pairs whose first token is the probe name; `env/hw/summarize.py`
parses those keys, so they are a contract.

## Building

On the machine, one line per probe:

```
hipcc --offload-arch=gfx942 -O2 -std=c++17 env/hw/probes/<name>.cu -o env/hw/build/<name>
```

with `<name>` in `occupancy stream_read xcc_map fence_probe chase copy_bytes`.
`env/hw/probes/probe_common.h` sits next to the sources and is picked up
without an include path.

On a laptop with no GPU, the same compile runs in Docker against the real
ROCm 7.0 hipcc, the way `env/offline_gfx942/run.sh` compiles the megakernel:

```
bash env/hw/probes/compile_offline.sh
```

It pulls `rocm/dev-ubuntu-22.04:7.0` for `linux/amd64`, mounts the repo at
`/w`, compiles all six into `env/hw/probes/work/out/`, compiles each a second
time with `-Rpass-analysis=kernel-resource-usage` to record the register
footprint, generates `fence_probe.s` and runs `fence_grep.py` over it, prints
a summary row per probe and exits non-zero if any compile failed. Under
emulation on Apple silicon each hipcc start costs about 30 seconds, so the
whole run takes a few minutes. `env/hw/probes/work/` and `env/hw/build/` are
gitignored.

## occupancy

Measures how many 256-thread blocks of a given register footprint are
resident per CU, and separately the rate of the constant-rate counter
`s_memrealtime` reads. The residency is the assumption the megakernel
deadlocks on if it is wrong: 296 worker blocks plus 8 scheduler blocks must
be co-resident on 304 CUs. The footprint is set by a template array length
whose elements are claimed inside a runtime loop by an empty asm statement
with a `+v` constraint; that asm is what makes the count predictable, because
it leaves the scheduler nothing to pipeline and the allocator takes exactly
the array plus four registers instead of filling whatever budget it was
given. Without it the allocator jumps from 166 to 254 registers between
adjacent lengths and 182 cannot be hit at all. The ladder runs from 64 to 256
registers in steps of 8, the allocation granularity, so any request lands
within four of what was asked for. The probe reads the true count of each
variant with `hipFuncGetAttributes(numRegs)`, picks the closest to `--vgprs`,
and queries `hipOccupancyMaxActiveBlocksPerMultiprocessor` with 256 threads
and the requested dynamic LDS, after opting into a request above the default
per-block limit with `hipFuncSetAttribute`. Neither a refused attribute nor a
failed query is treated as a probe failure when an LDS request is what caused
it: the line is still printed with `blocks_per_cu=0`, because a block that
does not fit is a measurement and a grid that is not co-resident deadlocks the
megakernel. With no LDS request a failed query is a real fault and exits 2.

`--dynamic-lds` matters because the runtime launches the worker kernel with
58,368 bytes of dynamic LDS (`runtime_header.h:41`,
`MAX_DYNAMIC_SHARED_MEMORY_SIZE` is 60 KiB minus the 3 KiB static
reservation), which puts the block plus its static reservation at 60 KiB of
the CU's 64 KiB. Residency is then LDS-limited to one block per CU rather
than register-limited to two, so the two runs bracket the answer.

The clock mode launches a one-thread kernel
that spins on `s_memrealtime` for about 100 ms as computed from
`hipDeviceAttributeWallClockRate`, and compares the ticks it counted with
what the attribute predicts over the host interval.

Feeds B10 and B11.

```
occupancy --vgprs 182 --dynamic-lds 58368
occupancy --vgprs 182
occupancy --clock
occupancy                       --vgprs 182 at 0 and at 58368 bytes, then --clock
```

Keys: `occupancy requested_vgprs actual_vgprs dynamic_lds blocks_per_cu
waves_per_simd cus coresident_blocks`, and for the clock mode
`wallclock rate_khz ticks host_ns ratio event_ms`. A `ratio` near 1 means the
attribute is right. `waves_per_simd` equals `blocks_per_cu` because 256
threads is four wavefronts spread over the CU's four SIMDs.

## stream_read

Measures achievable read bandwidth as a function of prefetch depth and
occupancy. The kernel is a read-only reduction over a `uint4` buffer with a
grid stride; `--unroll N` issues N independent loads into a register array
before consuming any of them, so N loads are in flight per thread behind a
single wait. The accumulator is xor-folded across the wavefront with shuffles
and one lane per wave `atomicXor`s it into a global the host reads back as
`checksum`: without a consumer the entire load stream is dead code and the
compiler deletes it. The fold uses shuffles rather than LDS because the `one`
occupancy mode claims the whole 64 KiB of LDS at launch, which is the lever
on residency. A block asking for 64 KiB leaves none for a second block on the
same CU, so `one` is one block per CU and one wave per SIMD; `--grid G` asks
for `65536/K` bytes where `K = ceil(G/cus)`, which leaves room for exactly K.
If the runtime refuses 65536 the probe falls back to 40960 and reports it in
`lds_bytes`. `full` is the kernel's own maximum residency with no dynamic
LDS.

`--passes P` repeats the whole grid-stride sweep P times inside one launch and
counts P times the buffer in the bandwidth. It exists for the working-set
sweep: the unrolled loop runs only while UNROLL further strides still fit, so
at full residency a buffer smaller than about a thread's worth of strides
falls entirely into the scalar tail and runs at one load in flight. Without
passes the sweep would confound prefetch depth with size, so the collection
holds `passes` times size constant at 1 GiB. The accumulator is rotated
between passes, because re-reading the same buffer an even number of times
would otherwise xor itself back to zero and leave a checksum that proves
nothing.

Timing warms up for about one second of launches first, because clocks under
a hypervisor sit in a low state until load arrives. The warm-up is bounded by
elapsed time rather than by a launch count, so a small working set is not
silently under-warmed, which is the case where a low clock state matters
most. Each repeat is then timed with its own event pair.

Feeds E1 to E4.

```
stream_read --occupancy full --size 1G --unroll 8 --repeat 3
stream_read --occupancy one  --size 1G --unroll N      N in 1 2 4 8 16 32
stream_read --grid 608 --size 1G --unroll N
stream_read --occupancy full --unroll 8 --size W --passes P
```

For the working-set sweep W runs over 4, 16, 32, 64, 128, 256, 512 and 1024
MiB with `P = 1024/W` in MiB, so every point reads the same 1 GiB.

Keys, one line per repeat: `stream_read occupancy grid block lds_bytes
blocks_per_cu_query size_mib unroll ms gbps tbps checksum vgprs passes`. `gbps` is
decimal GB/s over the buffer read once. `vgprs` is the register footprint of
the kernel that ran: N `uint4` in flight needs 4N registers for the data
alone, so a count well below that means the compiler split the batch and the
requested prefetch depth was not reached. ROCm 7.0.51831 allocates 14, 20,
32, 56, 72 and 78 registers for unroll 1, 2, 4, 8, 16 and 32, so depths above
16 are not delivered and a flat curve there is a compiler limit rather than a
hardware queue limit.

## xcc_map

Measures which XCD each workgroup of a plain launch lands on. Thread 0 of
every block reads `HW_REG_XCC_ID` with the same `s_getreg_b32` spelling the
Fleet runtime uses and stores it next to its own block id, so a block that
never ran shows as a sentinel rather than a plausible zero. The rule under
test is `xcd == blockIdx.x mod 8`: the design puts eight consecutive gang
tasks on eight distinct XCDs and the runtime indexes its scheduler queues the
same way, so a rule that holds on average but not on every launch would lose
scheduler events. `--concurrent` runs a 296-block busy kernel that spins for
about 50 ms on another stream and launches the probe grid while it is
running, which is the case MIN-25 hangs on. `--dump` writes `block,xcd` per
line for the last run.

Feeds F1 to F6.

```
xcc_map --grid 304
xcc_map --grid 296
xcc_map --grid 8
xcc_map --grid 8 --concurrent
xcc_map --grid 608
xcc_map --grid 1000
xcc_map --grid 37
```

Keys, one line per run: `xcc_map grid run concurrent distinct ids_in_range
rule_mod8_violations per_xcd stable_vs_run0`. `per_xcd` is eight
comma-separated counts, `rule_mod8_violations` counts blocks where the rule
failed, `stable_vs_run0` is 1 when the map is identical to the first run.

## fence_probe

Measures what the machine's hipcc lowers each fence to. Four kernels, each
with a volatile global store before the fence and a volatile global load
after it so the backend has something to order and cannot drop the fence:
agent-scope release, agent-scope acquire, `__threadfence()`, and an
agent-scope `__hip_atomic_load`. The measurement is the assembly, not the
run, and it settles a contradiction the repo still holds: the memory model
doc calls `__threadfence()` agent scope while the offline census attributes
system-scope sites to it. The file also links, so the same source is an
executable that proves the four kernels launch and prints `fence_probe
ran=4`.

`fence_grep.py` reads the assembly, splits it at the four kernel symbols and
counts the cache-maintenance instructions and the scoped loads per kernel. On
gfx942 the assembler spells the cache-policy bits as separate operands in the
fixed order `sc0 sc1`, so `buffer_wbl2 sc1` (agent scope, what the design
relies on) and `buffer_wbl2 sc0 sc1` (system scope) are distinct
instructions; the counting matches whole operand tokens, never substrings, so
`sc0` is not a prefix match on `sc1` and a line carrying both bits never
lands in the agent-scope column. `env/offline_gfx942/fences.txt` is the same census over the
megakernel.

Feeds G1 to G4.

```
hipcc --offload-arch=gfx942 -S --offload-device-only -O2 -std=c++17 \
    env/hw/probes/fence_probe.cu -o env/hw/<date>/raw/fence_probe.s
python3 env/hw/probes/fence_grep.py env/hw/<date>/raw/fence_probe.s
```

Keys, one line per kernel: `fence kernel wbl2_sc1 inv_sc1 wbl2_sc0_sc1
inv_sc0_sc1 load_sc1 load_sc0_sc1`. The two load counters are separate for the
same reason the fence counters are: a `volatile` load lowers to `sc0 sc1` and
is system scope, so counting it as `load_sc1` would make every kernel look as
though it carried an agent-scope load. On ROCm 7.0.51831 only `k_atomic_load`
has a true `sc1` load; the volatile loads of the other three are `sc0 sc1`.

## chase

Measures three latencies. `--size` follows a random cyclic permutation of
8-byte slots for 2^20 dependent loads after a 2^16 warm-up; each load's
address is the previous load's result, so the hardware cannot overlap them
and the loop time is the memory latency. The permutation is built with
Sattolo's algorithm rather than Fisher-Yates because drawing the swap index
from `[0, i)` instead of `[0, i]` is exactly what makes the result a single
cycle: a permutation with more than one cycle would only be walked inside the
cycle the chase started in, so the working set would be smaller than
requested and the latency would read too low. The seed is fixed and the
generator is written out rather than taken from the standard library, so the
permutation is identical on every machine. `--fence` runs 100,000 iterations
of store, fence, load at agent scope and the same loop at workgroup scope;
the difference is the cost of widening the fence, which is what buys the
cross-XCD visibility. One thread per participating workgroup and one
workgroup per XCD: the first block to take its XCD's slot with an `atomicCAS`
runs the loop and every later block on that XCD leaves, so `--xcds 1` (a
one-block grid) and `--xcds 8` (64 blocks, eight winners) share one code
path, and the slowest participant is reported. `--pingpong` has the first
block on XCD 0 and the first block on XCD 1 alternate ownership of one 32-bit
flag 10,000 times; the owner publishes with an agent-scope release store and
the other spins on agent-scope acquire loads with no sleep, so a handover is
one release flush plus one cross-XCD read, which is the boundary cost the
expected-performance argument turns on. The two players cannot be the same
workgroup because each claims its own XCD slot; a bounded spin, checked once
every 1,024 polls so the guard does not become part of the measurement,
covers the case where no block landed on one of the two XCDs. The flag is
plain `hipMalloc` device memory, coarse-grained, which is what the runtime's
own counters live in; `--fine` allocates it with
`hipExtMallocWithFlags(hipDeviceMallocFinegrained)` instead.

Feeds H1 to H7.

```
chase --size 1M
chase --size 64M
chase --size 1G
chase --fence release
chase --fence acquire
chase --fence release --xcds 8
chase --fence acquire --xcds 8
chase --pingpong
chase                    all of the above
```

Keys: `chase mode=size size_mib loads ns_per_load_wall
ns_per_load_memrealtime rate_khz`; `chase mode=fence kind xcds
ns_per_iter_agent ns_per_iter_workgroup ns_fence_cost`; `chase mode=pingpong
xcd_a xcd_b rounds ns_one_way fine`. When the pingpong does not complete the
line is still printed and a warning naming the unfinished flag goes to
stderr, so one placement failure does not abort the collection.

## copy_bytes

Copies exactly 1 GiB device to device with a plain 256-thread `uint4` copy
kernel, so the traffic is known in advance: 1 GiB read and 1 GiB written.
The profiler wraps the binary and the counter arithmetic is checked against
those two numbers. There is exactly one `copy_kernel` dispatch and it is the
1 GiB copy; the warm-up runs a separately named `warmup_kernel` on its own
small buffers so it can never be mistaken for the measured dispatch, and the
buffer fill is a third kernel, `fill_kernel`. The output is verified from
eight contiguous windows pulled back in one transfer each, rather than
element by element, so the kernel trace stays free of blit work.

Feeds I1 to I5.

```
rocprofv3 --pmc TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_32B_sum TCC_EA0_WRREQ_sum \
    TCC_EA0_WRREQ_64B_sum TCC_HIT_sum TCC_MISS_sum -- env/hw/build/copy_bytes
rocprofv3 --kernel-trace -- env/hw/build/copy_bytes
env/hw/build/copy_bytes
```

Keys: `copy_bytes bytes ms gbps`. `gbps` counts read plus write, so it is
twice the buffer over the time. `--bytes N[M|G]` changes the size; the
checklist rows use the default 1 GiB.
