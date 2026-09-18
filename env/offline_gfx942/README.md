# Offline gfx942 compile

Generating gfx942 code objects needs the ROCm compiler, not the GPU. So the
gate-1 question of `docs/design-doc/08-milestones.md` ("does the patched
Fleet megakernel compile for gfx942?") was answered before the first GPU
session, on a laptop, with the real hipcc of ROCm 7.0 in the
`rocm/dev-ubuntu-22.04:7.0` image (amd64, under emulation on Apple silicon;
about 30 s per variant).

    bash env/offline_gfx942/run.sh          # or OFFLINE_COMPILE=1 bash env/preflight.sh

## What is compiled

`mk_tu.cu` is the translation unit the runtime's code generator would
emit for our graph, written by hand: the standard-header preamble
(`runtime.cc:696-713`), `persistent_kernel.cuh`, and the two dispatchers
`_execute_task` / `_execute_gang_task` with the exact call code that the
patched `task_register.cc` emits for our task types at the model's
dimensions (round 3's eleven, and under `MK_GEMV` the round-4 GEMV linear,
the w13 gang, the stream pair, the merge tile, the router split and the
o_proj fold). The compile flags are those of `persistent_kernel.py` for
`mode = online`, batch 1, `max_seq_length = 1056`, `USE_GANG = 1`.
The first three variants: `ours` (what our graph needs), `ckfmha`
(`MPK_USE_CK_FMHA`, what the day-1 Qwen3 smoke graph needs; exercises the
patched CK block of `task_header.cuh`), `debugscores`
(`MLA_ATTEND_DEBUG_SCORES`, the B5 build); `run.sh` compiles seventeen
today (the list in the script, the round-4 ones below).

Sources: the submodule at `51dce4f` plus `fleet/patches/gfx942.patch` and
`new_tasks.patch`, our kernels copied in, `composable_kernel` at
`d8ee107a` and nlohmann/json at `8c391e04` (the commits Fleet's other
branches pin; `51dce4f` itself has no gitlink for either).

## Result (2026-09-14, hipcc 7.0.51831, clang roc-7.0.0)

All three variants compile and device-link with zero errors. Per-kernel
resource usage is in `resources.txt`, the fence census in `fences.txt`.

| Kernel | VGPRs | VGPR spill | SGPR spill | LDS (static) | Occupancy |
|---|---|---|---|---|---|
| `worker_kernel` (runtime + our five tasks) | 182 | 0 | 66 | 2,256 B | 2 waves/SIMD |
| `scheduler_kernel` | 59 | 0 | 25 | 0 | 7 |
| `kernel_tests` launcher: `k_mla_prep` | 63 | 0 | 0 | dynamic | 8 |
| `k_mla_attend` | 90 | 0 | 0 | dynamic | 5 |
| `k_mla_merge_uv` | 32 | 0 | 0 | dynamic | 8 |
| `k_moe_router` | 75 | 0 | 0 | dynamic | 6 |
| `k_copy` | 6 | 0 | 0 | dynamic | 8 |

The standalone launcher `fleet/tasks/kernel_tests_mi300.cu` also compiles
and links to an executable in both variants with the build line of
`fleet/tasks/README.md`, so day 2 starts from a binary that is known to
build.

What this settles and what it does not:

- **MAJ-1, compile half.** Every header the megakernel includes parses for
  gfx942, with the CK linears on the `16x16x16` warp GEMM substituted by
  `gfx942.patch`; the CK linear templates are parsed, not instantiated (no
  dispatcher calls them), so their device code is first generated on day 1. The `WarpGemmMfmaBf16Bf16F32M16N16K16TransposedCDistribution`
  name exists in CK at `d8ee107a` (`warp_gemm.hpp:238`) and at `rocm-7.2.4`.
  The cmake and cargo parts of `pip install -e .`, and running anything,
  stay for the machine.
- **MIN-27.** The three gfx950-only items are the only ones: with the patch,
  no error remains in either variant.
- **MAJ-4, static part.** The worker kernel's register union with our five
  tasks is 182 VGPRs with no VGPR spills; the CK linears are not instantiated
  by this dispatcher, so the real union is measured on day 2
  from the generated `kernel_0.cu` with the same `-Rpass-analysis` flag.
  The 66 SGPR spills are the runtime's, present without our tasks too.
- **MAJ-3.** In the gfx942 assembly (`fences.txt`) the worker kernel carries
  `buffer_wbl2 sc1` (4 sites) and `buffer_inv sc1` (2 sites): the agent-scope
  release and acquire the design relies on, as the LLVM memory model says
  `__builtin_amdgcn_fence(..., "agent")` lowers on gfx942. It also carries
  system-scope `buffer_wbl2 sc0 sc1` / `buffer_inv sc0 sc1` (72 / 48 static
  sites), which come from the printf and assert hostcall paths; a plain
  `__threadfence()` is not among them, it lowers to the agent-scope pair
  `buffer_wbl2 sc1` / `buffer_inv sc1` on this hipcc
  (`env/hw/probes/fence_probe.cu`, 2026-09-15). Whether any system-scope
  site sits on the per-task path is a day-2 look at the generated kernel,
  since a system-scope fence also writes back L2.
  `s_getreg_b32 hwreg(HW_REG_XCC_ID, 0, 16)` appears once in the worker and
  scheduler kernels and four times in `persistent_kernel`.
- **DQ3 / MIN-28, by source.** Both CK split-KV pipelines
  (`block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp:48` and the `nwarp_sshuffle`
  variant) `static_assert(kSubQKHeaddim <= 256)`, at `d8ee107a` and at
  `rocm-7.2.4`, and no MLA pipeline exists under `ck_tile/ops/fmha`. So
  `mla_attend` is the spec kernel in `fleet/tasks/mi300/`, not a CK
  wrapper (`00-decisions.md` D12, reversal condition met).

## Files

| File | What |
|---|---|
| `run.sh` | builds the patched copy under `work/` (gitignored), fetches CK and json, pulls the image, compiles the three variants, writes `resources.txt` and `fences.txt` |
| `mk_tu.cu` | the hand-written generated-code stand-in |
| `resources.txt` | `-Rpass-analysis=kernel-resource-usage` per kernel per variant |
| `fences.txt` | static counts of cache-control, fence, `XCC_ID` and `s_sleep` instructions per kernel |

## The GEMV probe (round 4, 2026-09-17)

`gemv_probe/run.sh` compiles a 64-row batch-1 GEMV tile for gfx942 at several
batch depths and prints the registers and the `s_waitcnt vmcnt` sequence (the
loads in flight per lane), and checks `v_dot2_f32_bf16` (gfx950 only). The
results of 2026-09-17 are in `gemv_probe/results.txt`; the reading is in
`docs/gpu-experiments/04-kernels/01-gemv-ideas.md`.

## Round 4 variants (2026-09-18)

`run.sh` also compiles `gemv` (`MK_GEMV`: every round-4 task in the worker,
the GEMV linear's three forms, the w13 gang, the stream pair, the merge
tile, the router split and the o_proj fold, keyed on the types the patch
adds), `gemvnt` (the same under `MLA_NT_STREAMS`, the build the graph rows
make), `w2ck` (the fused w2's CK path under `MPK_W2_CK_TILE`) and `union`
(`MK_GEMV`, `MK_CK_GANG`, `MLA_ATTEND_MFMA` and `MLA_NT_STREAMS` together:
the worker G1's row builds, the round-3 levers stacked on the round-4
tasks) and `unionT` (`union` with `MPK_ENABLE_TIMING`: the timing build
over the round-4 union, F4 of `docs/gpu-experiments/05-final`), disassembles
`gemv`, `gemvnt`, `ckgang`, `union` and `unionT` (`kt_r3.sh` besides
disassembles round 3's launcher tree at 8946804 into `dev_kt_r3.s`, F6 of
`docs/gpu-experiments/05-final`: round 3's `k_mla_merge_uv` beside round 4's), and writes
`dev_kt.s`, the standalone launcher's device code, where `k_linear_gemv`,
`k_moe_router` and `k_mla_merge_uv` are named kernels whose `s_waitcnt vmcnt`
sequences show the loads in flight per batch. The probes under `gemv_probe/`
(`gemv_probe.cu`, `bfly_probe.cu`, `order_probe.cu`) are single-kernel
compiles of 30 s each; their findings are in `gemv_probe/results.txt` and in
`docs/gpu-experiments/04-kernels/05-local-preparation.md` (the conventions).
