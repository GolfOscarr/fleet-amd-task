# Patches to the Fleet submodule

`gfx942.patch` is a unified diff against `repos/fleet-chiplet-megakernel` at
commit `51dce4f` (branch `amd_mi350`). `env/setup.sh` applies it before the
first build (skipped when `git apply --reverse --check` says it is already
in); apply by hand with

```
git -C repos/fleet-chiplet-megakernel apply --check fleet/patches/gfx942.patch
git -C repos/fleet-chiplet-megakernel apply fleet/patches/gfx942.patch
```

Item L13 of `docs/design-doc/10-local-work.md`; the problems are
`OPEN-PROBLEMS.md` MIN-27 and `docs/fleet/99-open-questions.md` Q12.

## Hunks

| File | Change | Why |
|---|---|---|
| `include/mirage/persistent_kernel/tasks/mi300/task_header.cuh` (line 34) | the include of `paged_attention_decode_minimal_mi300.cuh` is wrapped in `#if defined(__gfx950__)` | that file calls `__builtin_amdgcn_mfma_f32_16x16x32_f16` unguarded (`:27`), an MFMA that does not exist on gfx942, so the include fails the gfx942 compile; nothing else in `include/`, `src/` or `python/` references its symbols (`__mfma_qk`, `__mfma_pv`, `paged_attention_minimal_decode`), and no task is registered for it |
| `include/mirage/persistent_kernel/tasks/mi300/linear_ck_mi300.cuh` (`BlockGemmSmallM16Policy::GetWarpGemmMWarpNWarp`, line 73) | `WarpGemmMfmaBf16Bf16F32M16N16K32TransposedCDistribution` is kept under `#if defined(__gfx950__)`; gfx942 gets `WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution` with the same `MWarp = 1, NWarp = 4` | gfx942 has no 16x16x32 BF16 MFMA. This policy is the block GEMM of `GemmPipelineSmallTilePolicy` for every `MPerBlock = 16` tile (`:267`), so it is what all the M = 1 linears (`gang_linear*`, `gang_moe_*`, `linear`) run through |
| same file, `linear_kernel_ck` warp tile (line 331) | `WarpK` is 16 on gfx942 instead of 32 for the small tile | must match the warp GEMM above; the larger tiles already use K = 16 |
| `include/mirage/persistent_kernel/persistent_kernel.cuh` (`:1047` worker variant, `:1535-1536` scheduler variant) | the `[FWD_PASS]` print condition drops `fwd_pass_count < 10 || ... % 50 == 0` and `end_of_graph_count < 10 || ... % 100 == 0` | every iteration is printed so the 32 per-iteration times of a generation are all visible (`docs/design-doc/09-expected-performance.md`, D23); the format strings are unchanged for `measure.py` |

## What was checked and left alone

- `linear_ck_mi300.cuh:394` coherence value 18 (`sc1 nt`) under
  `MPK_NT_WEIGHT_LOADS`: the comment says gfx950 but the encoding is shared
  with gfx942 (`docs/mi300x/03-memory-model.md`); left as is, verified by the
  day-1 disassembly.
- `grep -rn "gfx950\|16x16x32\|mfma_f32_16x16x32"` over
  `include/mirage/persistent_kernel` hits only the two files patched above.
- The `WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution` name is what
  `ck_tile` uses for the gfx942 16x16x16 BF16 warp GEMM in the CK versions
  read for the design; it could not be compiled here (no ROCm, and the CK
  submodule is not checked out locally). If the first gfx942 build reports it
  missing, `ck_tile/ops/gemm/warp/warp_gemm.hpp` in the checked-out CK lists
  the available names.
