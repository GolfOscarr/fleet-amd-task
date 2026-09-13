# 05 — Synchronization Cross-Check

Fleet's paper, Fleet's code, and the LLVM/CDNA3 documentation we read
independently in `../mi300x/03-memory-model.md` are three separate sources. They
agree on the substance. This file records where, and where they differ.

## Agreement on the core fact

| Source | Statement |
|---|---|
| LLVM AMDGPUUsage, Memory Model GFX942 | gfx942 may be "fewer (possibly one) larger agents with groups of CUs on each agent each sharing separate L2 caches"; `buffer_wbl2` writes back, `buffer_inv sc1` invalidates |
| Fleet paper §2.1 | "the producer must issue `buffer_wbl2` to write back dirty L2 lines, and the consumer must invalidate stale L2 entries" |
| Fleet code, `mpk_atoms.cuh:298` | "MI300X: L2 NOT coherent across XCDs — `buffer_wbl2` required. Agent scope (`sc1`) sufficient — no need for system scope (`sc0 sc1`)." |

All three say the same thing, and the code's comment is almost a restatement of
our own conclusion. This is strong corroboration that the memory-model notes are
right and that the risk is real rather than theoretical.

## Scope-bit encoding

The paper gives a compact table we did not have:

> "SC1 and SC0 encode the coherence scope: `0_0`=wave, `0_1`=group, `1_0`=device,
> `1_1`=system."

Mapping to the LLVM terminology in our notes:

| SC1 SC0 | Fleet's name | LLVM's name | Our usage |
|---|---|---|---|
| 0 0 | wave | wavefront | default loads/stores |
| 0 1 | group | workgroup | intra-workgroup |
| **1 0** | **device** | **agent** | **task-graph flags** |
| 1 1 | system | system | not needed intra-GPU |

`sc1` alone = device/agent scope. **Consistent with LLVM and with the CDNA3 ISA
truth tables** in `../mi300x/03-memory-model.md`. No contradiction.

## Where Fleet's practice differs from the textbook sequence

LLVM's canonical agent-scope release is `buffer_wbl2 sc1` → `s_waitcnt` → store.
Fleet's code does several things differently, each deliberate:

1. **Fences via builtin, not hand-written asm.**
   ```c
   __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
   ```
   with the comment that agent scope suffices. This is exactly what our
   `../mi300x/99-open-questions.md` Q4 proposed to verify by disassembly — and
   Fleet is betting on the same thing. **Their bet does not remove our need to
   check it on our ROCm version**; it raises our prior that it works.

2. **Atomics with explicit `sc0 sc1` in inline asm**, bypassing the compiler:
   ```c
   // Inline asm: no compiler-generated buffer_wbl2/buffer_inv around atomic.
   // sc0 sc1 required on GFX942 for cross-CU atomic visibility.
   "flat_atomic_add %0, %1, %2 sc0 sc1\n"
   ```
   Note `sc0` on an atomic RMW is the **return-original-value** bit, not a
   coherence bit (CDNA3 ISA; also LLVM). So this reads as "returning atomic at
   system scope". They use system scope for the *global* event counter while
   using agent scope for the fence. Slightly conservative, and consistent.

3. **Write-through stores** (`sc0=1 sc1=1`) that bypass L2 entirely: "No
   `buffer_wbl2` needed after WT stores — data is already in memory." A cheaper
   alternative to writeback for small, write-once payloads such as flags. We did
   not have this option in our notes; it is a genuinely useful addition.

4. **Intra-XCD communication uses plain volatile loads/stores** with no fence at
   all, justified by all participating CUs sharing one L2 partition. Matches
   LLVM: within an agent sharing a single L2, `buffer_wbl2` "does nothing".

5. **NT does not imply ordering.** The code is emphatic:
   > "NT provides cache bypass but NOT memory ordering. ... Producer:
   > `st_nt(data)` → fence → `st_nt(flag)`; Consumer: `ld_nt(flag)` → fence →
   > `ld_nt(data)`."

   Our notes did not state this. It is an easy and dangerous mistake — using
   non-temporal accesses for flags and assuming visibility implies ordering.
   **Added to our memory-model understanding.**

## The one discrepancy

**Paper §5.1 says the scheduler reads `HW_ID`; the code reads `HW_REG_XCC_ID`.**

The CDNA3 ISA is clear that these are different registers: `HW_ID` is code 4,
marked "Read only. Debug only.", with fields `WAVE_ID/SIMD_ID/CU_ID/SE_ID/…` and
**no XCC field**; `XCC_ID` is code 20, "ID of the XCC this wave is running on".

The code is right and the paper's prose is loose. Use `HW_REG_XCC_ID`. This
also **resolves `../mi300x/99-open-questions.md` Q2** — the symbolic name works
in ROCm inline asm.

## Net effect on our open questions

| Question | Status after this cross-check |
|---|---|
| `mi300x` Q2 — `s_getreg` syntax for XCC_ID | **Resolved** — `hwreg(HW_REG_XCC_ID, 0, 16)` |
| `mi300x` Q4 — does HIP agent-scope emit the right ops | **Still open**, but Fleet relies on it; verify on our ROCm version |
| `mi300x` Q5 — cost of `buffer_wbl2`/`buffer_inv` | **Partly answered** — high enough that Fleet built two-level counting to cut writebacks from ~24K to ~300 per step |
| `mi300x` Q6 — waves per SIMD | **Answered by the paper**: expect **1 wave/SIMD** for a megakernel |
| `mi300x` Q1 — workgroup→XCD mapping | Not needed for correctness — Fleet reads `XCC_ID` instead of assuming |
