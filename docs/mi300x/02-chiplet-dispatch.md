# 02 — Workgroup Dispatch, XCD Identity, and Partitioning

This is the file that decides how a Fleet Chiplet-task is pinned to a chiplet.

## Discovering which XCD a wave is on

CDNA3 exposes the chiplet identity to the wave through a dedicated hardware
register, **separate from `HW_ID`**.

### `XCC_ID` — hardware register code 20 (CDNA3 ISA Table 7)

| Field | Bits | Description |
|---|---|---|
| `XCC_ID` | 3:0 | ID of this XCC |

The ISA's hardware-register table lists code `20` as `XCC_ID`, "ID of the XCC
this wave is running on". Four bits covers 0–15, comfortably more than the 8
XCDs on MI300X.

Read it with `s_getreg_b32` via inline asm, e.g.:

```c
__device__ inline unsigned xcc_id() {
    unsigned id;
    // hwreg(HW_REG_XCC_ID, offset=0, size=4)
    asm volatile("s_getreg_b32 %0, hwreg(20, 0, 4)" : "=s"(id));
    return id;
}
```

The `hwreg(id, offset, size)` form matches the ISA's encoding of `S_GETREG_B32`
(CDNA3 ISA §; `S_GETREG_B32` is SOPK, "Read a hardware register into the LSBs of
D"):

> "`SIMM16 = {size[4:0], offset[4:0], hwRegId[5:0]}`; offset is 0..31, size is
> 1..32."

So register id 20, offset 0, size 4 is exactly the field we want. The register
number, the field width, and this encoding are all primary; only the *symbolic
name* the assembler accepts (e.g. whether `HW_REG_XCC_ID` is recognised) needs
checking against the compiler in use — see `99-open-questions.md` Q2.

### `HW_ID` — hardware register code 4 (CDNA3 ISA Table 6)

| Field | Bits | Description |
|---|---|---|
| `WAVE_ID` | 3:0 | Wave buffer slot number |
| `SIMD_ID` | 5:4 | SIMD which the wave is assigned to within the CU |
| `PIPE_ID` | 7:6 | Pipeline from which the wave was dispatched |
| `CU_ID` | 11:8 | Compute Unit the wave is assigned to |
| `SH_ID` | 12 | Shader Array (within an SE). **"Is set to zero."** |
| `SE_ID` | 15:13 | Shader Engine the wave is assigned to |
| `TG_ID` | 19:16 | Thread-group ID |
| `VM_ID` | 23:20 | Virtual Memory ID |
| `QUEUE_ID` | 26:24 | Queue from which this wave was dispatched |
| `STATE_ID` | 29:27 | State ID (UNUSED) |
| `ME_ID` | 31:30 | Micro-engine ID |

The ISA marks `HW_ID` as "Read only. Debug only." — usable for a diagnostic
microbenchmark that maps workgroup ID to physical placement, but we should not
build scheduling logic on it. `XCC_ID` carries no such caveat.

**Design consequence:** a persistent kernel can determine its own chiplet at
runtime and self-organize (elect a scheduler per XCD, claim the right task
queue) rather than depending on an assumed launch-order→XCD mapping. This is
strictly more robust than computing the chiplet from `blockIdx`, and it
survives changes in partition mode and dispatch policy.

## Dispatch policy

The AMD partitioning documentation states, per mode (verbatim):

- SPX: "Workgroups are automatically distributed across all XCDs (round-robin)."
- DPX: "Workgroups distributed within each partition's 4 XCDs."
- CPX: "Workgroups are explicitly launched to a specific XCD (i.e., scheduling
  can be controlled at the application level)."

That is the **entire** documented statement of dispatch behaviour. The docs do
not specify the round-robin granularity (per workgroup? per ACE?), the starting
XCD, or how the policy interacts with launch order or with a grid that exactly
fills the device. Community work on "swizzling" assumes plain per-workgroup
round-robin — i.e. workgroup `i` lands on XCD `i % 8` — and reports large L2
hit-rate gains from remapping tile indices on that assumption, but this is an
inference from measurement, not documentation.

**Do not hard-code `blockIdx.x % 8`.** Verify with `XCC_ID` first; see
`99-open-questions.md`.

## Compute partitioning modes

| Mode | Meaning | XCDs/partition | Logical devices | CUs/device | Memory/device |
|---|---|---|---|---|---|
| **SPX** | Single Partition X-celerator | 8 | 1 | 304 | 192 GB |
| DPX | Dual Partition X-celerator | 4 | 2 | 152 | 96 GB |
| CPX | Core Partitioned X-celerator | 1 | 8 | 38 | 24 GB |

- **SPX is the default mode for MI300X**, and the GPU "will always revert back
  to this default SPX mode when the system is rebooted or when the amdgpu
  driver is unloaded and reloaded."
- Under SPX, "Implicit synchronization across XCDs is handled by the hardware."
  Note this refers to the driver/hardware presenting one logical device — it
  does **not** mean the L2 caches are coherent. See `03-memory-model.md`.
- Mode changes are runtime, via `amd-smi`; no reboot, no hypervisor needed.
- The docs state partitions "must include an even number of XCDs (e.g., 2, 4,
  6, 8)" while also documenting CPX as 1 XCD per partition. The docs do not
  reconcile this; flagged as a documentation inconsistency, not a fact.

## Memory partitioning (NPS)

| Mode | Description | Compute mode compatibility |
|---|---|---|
| NPS1 | Unified memory pool (192 GB), interleaved across all 8 stacks | SPX, DPX, CPX |
| NPS2 | 2 partitions of 96 GB | DPX |
| NPS4 | 4 partitions of 48 GB (2 HBM stacks each) | CPX only |

Constraint: "the number of memory partitions must be less than or equal to the
number of compute partitions." Mixing NPS modes in a single node is "not
recommended".

## Our configuration

The task specifies one MI300X, batch 1, no tensor parallelism. **SPX + NPS1**
is the right choice and is also the default:

- Fleet's premise is one persistent kernel spanning the whole device with
  per-chiplet scheduling *inside* it. CPX would hand us 8 separate logical
  GPUs, which would require multi-device programming to span — the opposite of
  what we want, and arguably a form of the parallelism the task excludes.
- The model must see all 192 GB as one pool.

**Action:** record the partition mode in every benchmark run, and assert it at
startup, because a reboot silently restores SPX and a prior experiment may have
left the machine in CPX. The partitioning overview page gives no command lines;
the query/set syntax (`amd-smi`, and whether `rocm-smi --showcomputepartition`
still works) must be confirmed on the machine.

## Open items tracked elsewhere

- Actual workgroup→XCD mapping → `99-open-questions.md` Q1
- `s_getreg_b32 hwreg(20,...)` assembler syntax → `99-open-questions.md` Q2
- Partition query/set command lines → `99-open-questions.md` Q3
