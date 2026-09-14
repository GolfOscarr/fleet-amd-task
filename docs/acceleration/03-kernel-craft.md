# 03 — Kernel Craft on gfx942

Lever-B detail: how to make the bytes arrive as fast as the hardware allows.

## MFMA is the wrong tool at M=1

Our GEMVs are `[1,2048] × [2048,N]`. The smallest BF16 matrix instruction is
`V_MFMA_F32_16X16X16_BF16`, which computes a 16×16 output tile. With M=1,
**15 of 16 output rows are wasted** — we would be issuing matrix instructions at
1/16 utilization.

That does not matter for throughput (we are memory-bound, not compute-bound),
but it does matter for **register pressure**, which per `../fleet/03-runtime.md`
is already the binding constraint on a megakernel: the combined footprint limits
occupancy to 1 wave/SIMD. Accumulators for a 16×16 tile we do not need are
registers not spent on outstanding loads.

The alternative is a **VALU dot-product**: each lane accumulates a partial
product over its slice of K, then a cross-lane reduction sums the 64 lanes.
Fewer registers, no wasted tile, and at M=1 no throughput loss.

**This should be settled by microbenchmark, not opinion** — see Q2. Note that
Fleet's own MI300 kernels use `ck_tile` GEMM templates (MFMA-based) even at
bs=1, which is evidence that the simple argument above may be missing something
about how CK schedules loads.

## Cross-lane reduction primitives

For the dot-product reduction, in rough order of preference:

| Primitive | Notes |
|---|---|
| DPP (`v_add_f32 ... row_shr:`) | Register-to-register, no LDS, lowest latency |
| `ds_swizzle_b32` | LDS-hardware permute, no LDS storage |
| `ds_bpermute_b32` | Arbitrary lane gather |
| LDS scratch + tree | Simplest, costs LDS and `lgkmcnt` waits |

A wave-wide reduction of 64 lanes takes 6 DPP steps.

## Cache policy — adopt Fleet's, with one inversion

Fleet's three-tier scheme (`../fleet/01-paper-review.md` §4.1):

| Data | Policy | Purpose |
|---|---|---|
| Weight loads | cache-streaming, `sc1=1 nt=1` | read once, mark for early eviction |
| Activation stores | non-temporal, `NT=1` | don't evict weights |
| Cross-XCD scheduler polling | non-temporal loads | bypass stale L2 |
| Intra-XCD scheduler traffic | volatile loads via shared L2 | cheap |

**Our inversion:** Fleet marks weights streaming to protect *other weights*
(cooperative M-tile reuse across workers). At batch 1 there is no such reuse —
their own Table 4 shows it. We would mark weights streaming to protect the
**latent KV cache slice** (144 KiB per XCD per layer), which is the one thing we
have that is small enough to stay resident and gets re-read every token.

Same mechanism, different beneficiary, and ours works at batch 1 where theirs
does not. This is Q3 in `../mi300x/99-open-questions.md` and is the cheapest
high-value experiment in the project.

### The trap

From Fleet's `mpk_atoms.cuh`:

> "NT provides cache bypass but NOT memory ordering. ... Producer: `st_nt(data)`
> → fence → `st_nt(flag)`; Consumer: `ld_nt(flag)` → fence → `ld_nt(data)`."

Non-temporal is a cache hint, not a synchronization primitive. Flags still need
the agent-scope fences from `../mi300x/03-memory-model.md`.

## Wide loads and prefetch depth — a hard requirement

Use `global_load_dwordx4` (128 bits per lane) wherever alignment allows. A
64-lane wave then pulls 1 KiB per instruction.

**Every GEMV and MLA inner loop must issue 4-8 independent loads before the
first `s_waitcnt`.** Per `../mi300x/07-achievable-bandwidth.md`, at 1 wave/SIMD
the device needs ~2-4 outstanding loads per wave to saturate HBM, and `VMCNT`
allows 63 — so the hardware is not the constraint, the loop structure is. A
naive `load -> waitcnt -> use` loop will run at roughly `1/N` of peak and will
look like a bandwidth problem when it is a scheduling problem.

Budget ~32 VGPRs for a depth of 8. That is additive with every other task in the
megakernel's register union.

For BF16 weights that is 8 elements per lane; for FP8, 16.

## LDS

64 KiB per CU, 32 banks × 512 dwords, allocated in 512-byte granules
(`../mi300x/01-architecture.md`). At batch 1 there is little to stage in LDS for
a GEMV — the activation vector is 4 KiB and fits in registers. LDS is mainly
useful for the cross-lane reduction and for staging the latent KV slice.

Keeping LDS usage low is also an occupancy lever, though per Fleet the binding
constraint is registers, not LDS.

## Weight-only FP8 dequantization in-kernel

Per `01-precision.md`, weight-only FP8 with FP32 accumulation is the right shape
for us. The inner loop becomes: `global_load_dwordx4` of FP8 bytes →
`V_CVT_PK_F32_FP8` to unpack → scale → FMA accumulate. The conversion is VALU
work in a memory-bound loop, so it should be free.
