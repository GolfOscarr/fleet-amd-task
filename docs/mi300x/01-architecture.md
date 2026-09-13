# 01 — MI300X Architecture

## Die and compute hierarchy

| Fact | Value | Source | Verified |
|---|---|---|---|
| XCDs per package (MI300X) | 8 | ROCm MI300 microarch | primary |
| XCDs per package (MI300A) | 6 XCDs + 3 CCDs | ROCm MI300 microarch | primary |
| Physical CUs per XCD | 40 | ROCm MI300 microarch | primary |
| Active CUs per XCD | 38 (2 disabled for yield) | ROCm MI300 microarch | primary |
| Total active CUs | 304 | ROCm MI300 microarch + partitioning doc | primary |
| ACEs per XCD (workgroup dispatch) | 4 | ROCm MI300 microarch | primary |
| I/O dies (IOD/AID) | 4 | ROCm microarch ("4 I/O dies"), partitioning doc ("4 IODs") | primary |
| XCD↔IOD stacking | each pair of XCDs 3D-stacked on one IOD | Partitioning doc | primary |
| Wavefront size | 64 | CDNA3 ISA | primary |
| SIMDs per CU | 4 | derived: `HW_ID.SIMD_ID` is 2 bits (5:4) | **derived** |
| LLVM target | `gfx942`, `EF_AMDGPU_MACH_AMDGCN_GFX942 = 0x04c` | LLVM AMDGPUUsage | primary |

The ROCm microarchitecture page never prints the string "304" next to
"MI300X" specifically; it says "up to 304 CUs" and separately that MI300X has
8 XCDs of 38 active CUs. The partitioning doc states "304 total CUs" for
MI300X directly, so the figure is primary — but note 8 × 38 = 304 is also the
consistent arithmetic.

## Memory hierarchy

| Level | Size | Scope | Source | Verified |
|---|---|---|---|---|
| Vector L1 (per CU) | 32 KB | 1 CU | ROCm microarch (XCD diagram caption) | primary |
| LDS (per CU) | 64 kB | 1 CU (workgroup) | CDNA3 ISA §3.6 | primary |
| Scalar L1 | shared by a group of CUs; size not stated | group of CUs | LLVM AMDGPUUsage | primary |
| L2 (per XCD) | 4 MB | 1 XCD | ROCm microarch | primary |
| L2 aggregate | 32 MB (8 × 4 MB) | package | derived | derived |
| Infinity Cache / MALL | 256 MB, ~17 TB/s peak | package, on the AIDs | search summary only | **secondary** |
| HBM3 | 192 GB across 8 stacks | package | Partitioning doc | primary |
| HBM3 aggregate bandwidth | 5.3 TB/s theoretical peak | package | ROCm microarch | primary |

The ROCm microarchitecture page does **not** mention Infinity Cache or MALL at
all — it describes the I/O dies only as "containing system infrastructure".
The 256 MB / 17 TB/s figures come from secondary reporting. The partitioning
doc does use the term MALL ("Memory Attached Last Level Cache") in passing
without giving a size. Treat the size and bandwidth as unconfirmed; see
`99-open-questions.md`.

### LDS detail (CDNA3 ISA §3.6, §11)

> "Each compute unit has a 64kB memory space that enables low-latency
> communication between work-items within a work-group, or the work-items
> within a wavefront; this is the local data share (LDS)."

- 64 kB per CU, segmented into **32 banks of 512 Dwords**, each bank 32 bits.
- A single work-group can request **up to 64 kB**.
- Allocated "in contiguous blocks of 512 bytes on 512-byte alignment."
- Reads across a wavefront are dispatched over four cycles.

### Register file (CDNA3 ISA §3.6.4)

> "A wave may have up to 512 total VGPRs, 256 of each type. When a wave has
> fewer than 512 total VGPRs, the number of each type is flexible — it is not
> required to be equal numbers of both types."

- Two pools: **architectural VGPRs** and **accumulation VGPRs (AccVGPRs)**.
  AccVGPRs are used by matrix (MFMA) instructions and can be loaded directly
  from memory.
- Allocated in groups of **eight Dwords**.
- 64-bit operands require even-aligned VGPRs.
- MFMA instructions select the pool via the `ACC` bit (0 = VGPR, 1 = AccVGPR);
  `ACC_CD` selects the pool for the C and D matrices.

This flexible 512-register budget is the main lever on persistent-kernel
occupancy — see `04-persistent-kernel.md`.

## Peak compute (MI300X, as published by ROCm)

| Computation and data type | FLOPS/CLOCK/CU | Peak TFLOPS |
|---|---|---|
| Matrix FP64 | 256 | 163.4 |
| Vector FP64 | 128 | 81.7 |
| Matrix FP32 | 256 | 163.4 |
| Vector FP32 | 256 | 163.4 |
| Vector TF32 | 1024 | 653.7 |
| Matrix FP16 | 2048 | 1307.4 |
| **Matrix BF16** | **2048** | **1307.4** |
| Matrix FP8 | 4096 | 2614.9 |
| Matrix INT8 | 4096 | 2614.9 |

Source: ROCm MI300 microarchitecture page (primary). The page gives no clock
speed and no TDP, so a clock could only be back-derived from these numbers —
it is not stated and is not recorded here as fact.

### BF16 MFMA instructions (CDNA3 ISA)

Relevant opcodes present in the CDNA3 ISA for our BF16 path:

- `V_MFMA_F32_32X32X8_BF16` (opcode 96)
- `V_MFMA_F32_16X16X16_BF16` (opcode 97)

Both accumulate in FP32 from BF16 inputs. Full MFMA opcode table is in the ISA
guide; these two are the shapes most relevant to batch-1 decode, where the M
dimension is 1 and the instruction is heavily under-utilized in M.

## What this means for batch-1 decode

Batch-1 decode is **memory-bound, not compute-bound**. With one token in
flight, every weight matrix is read once and used for a single GEMV. The
1307 TFLOPS of BF16 matrix throughput is almost entirely unreachable; the
5.3 TB/s of HBM bandwidth and the 32 MB of aggregate L2 are the resources that
actually determine latency.

That is precisely the regime Fleet targets, and why its Chiplet-task
abstraction (bind work to the XCD whose 4 MB L2 already holds the data) is the
relevant lever rather than better MFMA tiling.
