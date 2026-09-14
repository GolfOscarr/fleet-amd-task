# MLA Decode — Prior Art and Our Kernel Spec

Closes discovery gap 1. The MLA decode Chiplet-task (MAJ-2) is the one piece of
this project with no equivalent in the Fleet repo, so this reads how working
implementations shape it before we write ours.

## Files

| File | Contents |
|---|---|
| `01-implementations.md` | Survey: which project, which file, which arch, gfx942 status |
| `02-kernel-anatomy.md` | The common structure, and vLLM's own MLA derivation |
| `03-design-choices.md` | The ten questions, answered from source |
| `04-our-kernel-spec.md` | **The deliverable** — shapes, tiling, split sizing, registers |
| `99-open-questions.md` | |
| `sources/` | Fetched implementation files |

## What the prior art confirmed

1. **Runtime reassociation is what everyone does.** vLLM's "data-movement
   friendly" path computes `ql_nope = einsum("snh,lnh->snl", q_nope, W_UK)` and
   `o = einsum("snl,lnv->snv", sdpa_o, W_UV)` **at runtime**, keeping `W_UK` and
   `W_UV` as separate `bmm` operands. Nobody materializes the fused weights.
   Independent confirmation of `../deepseek-v2-lite/02-mla.md`.
2. **MLA decode is MQA, not MHA.** Both vLLM and FlashMLA frame it the same way:
   one KV "head" of width `Lkv + R = 576` for QK and `Lkv = 512` for V, with the
   16 query heads as the **M dimension**. This is a better mental model than
   "16 independent heads" and it changes the MFMA answer.
3. **`kv_b_proj` is kept in BF16.** vLLM: "we currently do not have quantized
   bmm's which are needed for `W_UV` and `W_UK_T`, we just store fp16/bf16
   copies ... the extra memory overhead of this is fairly low." That is exactly
   our MIN-3 fallback, and they price it the same way we did.
4. **`nope` and `pe` stay separate.** vLLM's grouped kernel uses
   `BLOCK_DMODEL = 512` and `BLOCK_DPE = 64` as distinct tiles — they do *not*
   concatenate into one 576-wide operand.

## What it changed

**Our split-KV sizing was wrong in both directions.**

- vLLM's own heuristic would give **2 splits** at S=1024
  (`next_pow2(1024 // 512)`), and with `BLOCK_H=16` the grid collapses to
  `(1, 1, 2)` — **2 workgroups, 0.7% of 296 worker CUs**. That is tuned for
  large batches, where `batch × heads` supplies the parallelism. At batch 1 it
  is pathological.
- Our own `../fleet/06-our-task-graph.md` proposed **8** splits, one per XCD.
  Also too few.

Working the trade-off out (`04-our-kernel-spec.md`) gives **32–64 splits**, and
— more importantly — shows that **`BLOCK_H` must stay 16**. Because the latent
is shared across all query heads (the MQA property), one block covering all 16
heads reads the cache **once**; splitting heads across blocks re-reads it per
head-block. Parallelism should come from splitting the sequence, never from
splitting heads.

**MFMA is right for attention, wrong for the weight GEMVs.** vLLM's ROCm tuning
passes `matrix_instr_nonkdim: 16` — MFMA 16×16 — with `waves_per_eu: 1`. With
`BLOCK_H=16` the M dimension is 16, which fills a 16×16 tile exactly. My earlier
claim in `../acceleration/03-kernel-craft.md` that "MFMA is the wrong tool at
M=1" holds for `q_proj` and the expert GEMVs, but **not** for MLA attention,
where M is the head count. Corrected there.

## Status

| Question | Answer |
|---|---|
| Is our KV layout right? | Yes — split `c_KV[S,512]` + `k_pe[S,64]`, confirmed by `BLOCK_DMODEL`/`BLOCK_DPE` |
| Is runtime reassociation right? | Yes — it is what vLLM does |
| Is 8-way split-KV right? | **No** — use 32–64 |
| Is MFMA right? | Yes for attention (M=16), no for weight GEMVs (M=1) |
| Can we reuse an existing gfx942 kernel? | **No** — AITER's gfx942 MLA decode is hand-written ASM |
