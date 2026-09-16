# Hardware collection 20260915

Collected on 20260915 on enc1-gpuvm005, ROCm 7.2.4, hipcc 7.2.53211. 62 checklist rows: 32 PASS, 5 MISMATCH, 24 INFO, 1 UNAVAILABLE. Raw captures are in raw/, one file per command, and collect.log holds the run. Every MISMATCH belongs in OPEN-PROBLEMS.md and the owning 99-open-questions.md with its date and command (docs/gpu-experiments/01-bringup/02-checklist.md, Sign-off).

| # | Check | Measured | Expected | Result |
|---|---|---|---|---|
| A1 | GPU name and target | gfx942, AMD Instinct MI300X VF | gfx942, MI300X | PASS |
| A2 | GPUs and bus ids | 2 GPU agent(s), device 0 BDF 0000:fd:00.0 | 1 or 2; BDF of device 0 recorded | PASS |
| A3 | ROCm version | 7.2.4 | 7.0 or newer | PASS |
| A4 | hipcc version | 7.2.53211 | same major as the offline 7.0.51831 | PASS |
| A5 | Driver version | 6.16.13 (amd-smi 6.16.13) | INFO | INFO |
| A6 | CK headers in ROCm | ck_tile present, ck present | INFO: absent or a version | INFO |
| A7 | OS, kernel, Docker, Python | Ubuntu 24.04.4 LTS, 6.8.0-124-generic, docker 29.5.3, python 3.12.3 | INFO; Ubuntu with Docker | INFO |
| A8 | System torch | absent | INFO; expected absent | INFO |
| B1 | Compute units | rocminfo 304, kfd 1216/4 = 304 | 304 (kfd 1216 / 4) | PASS |
| B2 | XCD count | 8 (kfd num_xcc) | 8 | PASS |
| B3 | SIMDs per CU, wavefront | 4 SIMDs per CU, wavefront 64, kfd 4/64 | 4 SIMDs per CU, wavefront 64 | PASS |
| B4 | Max waves per CU | 32 per CU, kfd 8 per SIMD | 32 per CU, 8 per SIMD (gk) | PASS |
| B5 | LDS per workgroup | GROUP segment 64 KB, kfd 64 KB | 64 KiB | PASS |
| B6 | Workgroup max size | 1024 work-items | 1024 work-items (gk) | PASS |
| B7 | L2 size | 4096(0x1000) KB | 4 MB per XCD | PASS |
| B8 | Last-level cache | 262144(0x40000) KB | INFO; 256 MB is the secondary figure | INFO |
| B9 | HBM size | rocminfo GLOBAL 191.7 GiB, amd-smi 191.7 GiB | 192 GB (about 196,000 MiB) | PASS |
| B10 | Residency of the worker kernel | register-only 2 block(s) per CU (184 VGPRs), with the 58,368 B request 1 block(s) per CU, 304 co-resident | 1 block per CU with the 58,368 B dynamic LDS request; 2 register-limited | PASS |
| B11 | Wall-clock rate | rate 100000 kHz, ratio 0.981665 | the attribute and the host clock agree within 1% | MISMATCH |
| C1 | Compute partition | SPX | SPX | PASS |
| C2 | Memory partition | NPS1 | NPS1 | PASS |
| C3 | Query syntax that works | amd-smi static --partition | INFO | INFO |
| C4 | Guest may set the partition | amd-smi set does not offer --compute-partition | UNAVAILABLE acceptable if C1 and C2 pass | INFO |
| D1 | Memory clock | current 901 MHz, max 1,300 MHz | INFO; the rated HBM3 clock behind the 5.3 TB/s peak | INFO |
| D2 | Engine clock | current 138 MHz, max 2,100 MHz, rocminfo 0 MHz, kfd 2100 kHz | INFO; about 2,100 MHz max (gk) | INFO |
| D3 | Power cap and draw | draw 144 W | INFO; cap at the 750 W class (gk) | INFO |
| D4 | Temperature and throttle | 46 °C, throttle status not readable: --throttle is not a flag in this amd-smi | no throttle flags | INFO |
| D5 | Memory in use, processes | 0.28 GiB used, 0 process(es) | near 0, no other process | PASS |
| E1 | Read bandwidth, full, 1 GiB, N=8 | 3 runs, mean 3.943 TB/s, spread 0.32% | 3.66 to 4.3 TB/s; spread under 3% | PASS |
| E2 | Bandwidth at one wave/SIMD vs N | knee at N=4, plateau 4325 GB/s (N1:2706, N2:2410, N4:4325, N8:4304, N16:3787, N32:4095) | knee at N about 4, plateau within 10% of E1 | PASS |
| E3 | Same at two waves/SIMD | knee at N=4, plateau 4033 GB/s (E2 knee 4, so at or below N=2) | plateau at about half the N of E2 | MISMATCH |
| E4 | Bandwidth vs working set | 4MiB:44, 16MiB:175, 32MiB:350, 64MiB:627, 128MiB:1152, 256MiB:2202, 512MiB:3249, 1024MiB:3529; step above 32 MiB: yes; above 256 MiB: yes; sizes under 256 MiB are launch-bound in this run (no --passes), so only the 512 and 1024 MiB points are bandwidth | step above 32 MiB (L2) and above 256 MiB (Infinity Cache) | INFO |
| E5 | BabelStream | Copy 4,319,424 MB/s, Mul 4,231,808 MB/s, Add 3,894,077 MB/s, Triad 4,133,844 MB/s, Dot 4,038,666 MB/s | Dot >= 3,660,781 MB/s, Copy >= 4,177,285 MB/s | PASS |
| E6 | Host-to-device bandwidth | - | INFO | UNAVAILABLE |
| F1 | XCC_ID readable and in range | 8 distinct ids, in range: yes | ids in 0..7, all eight present | PASS |
| F2 | Rule at the worker grid (296) | per XCD [37, 37, 37, 37, 37, 37, 37, 37], round robin with offset 4 (xcd == (block + 4) mod 8, from the 296-block dump), 888 strict-rule violation(s) over 3 run(s) | xcd == block mod 8 for every block, 37 per XCD | MISMATCH |
| F3 | Scheduler grid (8) | alone 24 violation(s) over 3 run(s), concurrent 24 over 3, blocks 0..7 on XCD 4,5,6,7,0,1,2,3 (derived from the offset) | block k on XCD k, alone and concurrent | MISMATCH |
| F4 | Rule at other grids | grid 608: 1824 violation(s), grid 1000: 3000 violation(s), grid 37: 111 violation(s), 37 per XCD [5, 4, 4, 4, 5, 5, 5, 5], counts match offset 4 | same rule at 608, 1000, 37; at 37: 5,5,5,5,5,4,4,4 | MISMATCH |
| F5 | Stability across launches | 21 of 21 runs identical to run 0 | identical maps across three launches | PASS |
| F6 | Blocks per XCD at 304 | per XCD [38, 38, 38, 38, 38, 38, 38, 38] | 38 per XCD at 304 | PASS |
| G1 | Agent-scope release | buffer_wbl2 sc1 x1, sc0 sc1 x0 | buffer_wbl2 sc1, no sc0 sc1 in that kernel | PASS |
| G2 | Agent-scope acquire | buffer_inv sc1 x1, sc0 sc1 x0 | buffer_inv sc1, no sc0 sc1 in that kernel | PASS |
| G3 | __threadfence() | wbl2 sc1 x1, inv sc1 x1, wbl2 sc0 sc1 x0, inv sc0 sc1 x0: agent | agent scope: buffer_wbl2 sc1 and buffer_inv sc1, no sc0 sc1 | PASS |
| G4 | Agent-scope atomic load | load sc1 x1, buffer_inv sc1 x1 | sc1 on the load, buffer_inv sc1 after | PASS |
| H1 | L2 hit latency | 81.3 ns wall, 81.1 ns s_memrealtime | INFO | INFO |
| H2 | Infinity Cache latency | 258.2 ns wall, 257.6 ns s_memrealtime | INFO; about 218 ns from a secondary source | INFO |
| H3 | HBM latency | 341.6 ns wall, 340.8 ns s_memrealtime | INFO; 250 ns to 2 us; flag only above 2 us | INFO |
| H4 | Release fence cost | 115.1 ns (agent 525.2 ns, workgroup 410.1 ns) | INFO; under 1 us | INFO |
| H5 | Acquire fence cost | 137.4 ns (agent 893.6 ns, workgroup 756.2 ns) | INFO; under 1 us | INFO |
| H6 | Fence cost, eight XCDs | release -30.1 ns (agent 798.4 ns, workgroup 828.5 ns); acquire 317.3 ns (agent 990.5 ns, workgroup 673.2 ns) | INFO | INFO |
| H7 | Cross-XCD one-way latency | 703.4 ns one way (XCD 0 to 1, 10000 rounds) | INFO; 116 to 202.5 ns from a secondary source | INFO |
| I1 | Counter names present | all 7 present | the six counter names plus TCC_BUBBLE_sum | PASS |
| I2 | Counters readable | TCC_BUBBLE_sum 8,388,608, TCC_EA0_RDREQ_32B_sum 0, TCC_EA0_RDREQ_sum 8,388,760, TCC_EA0_WRREQ_64B_sum 16,777,216, TCC_EA0_WRREQ_sum 16,777,216, TCC_HIT_sum 8,391,568, TCC_MISS_sum 16,777,296 | non-zero values, no error | PASS |
| I3 | Bytes-from-requests arithmetic | reads 1.0000 GiB (+0.0%), writes 1.0000 GiB (+0.0%), flat 64 B reads 0.5000 GiB (-50.0%), flat 64 B writes 1.0000 GiB (+0.0%) | 128 B per TCC_BUBBLE read and 64 B per write request, both within 5% of 1 GiB | PASS |
| I4 | Kernel trace | 1 copy_kernel dispatch(es), first 551,532 ns | one dispatch row with a duration | PASS |
| I5 | TCC instance count | 16 per XCC x 8 XCC = 128 | INFO; 128 in SPX (gk) | INFO |
| J1 | CPU model and cores | Intel(R) Xeon(R) Platinum 8470, 26 cores | INFO; 8, 13 or 26 cores | INFO |
| J2 | RAM | 440 GB total | INFO; 224 GB (1x), 448 GB (2x) | INFO |
| J3 | Disk free | / 12,288 GiB free, home 12,288 GiB free | over 100 GiB free | PASS |
| J4 | Download rate | captured, no rate line | INFO; minutes for 31 GB | INFO |
| J5 | Container with GPU passthrough | gfx942 visible in the container | the GPU visible inside the container | PASS |
| J6 | VM creation to first ssh | recorded by hand | INFO; recorded by hand | INFO |
