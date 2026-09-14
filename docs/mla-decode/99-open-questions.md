# 99 — Open Questions

## Q1 — Is `P_split = 32` right, or is attention off the critical path anyway? `open`

**Why.** `04-our-kernel-spec.md` picks 32 from a traffic model that assumes
achievable bandwidth scales with the fraction of CUs issuing loads. With 32 of
296 workers busy, attention could cost ~5–7 µs/layer (~150–190 µs/token, 13–17%
of the 1,148 µs realistic budget) — or much less if the model is pessimistic.

**Check.** Time phase B alone at `P_split` = 8, 16, 32, 64, 128 on the machine.
Expect a knee around 32–64. **Needs GPU**; the constant is trivial to sweep.

---

## Q2 — MFMA or VALU for the phase-A `q_nope @ W_UK` product? `open`

**Why.** Phase B is settled — M=16 fills a 16×16 MFMA tile, and vLLM ships
`matrix_instr_nonkdim: 16`. Phase A is different: 16 separate `[1,128]×[128,512]`
products with M=1, where MFMA wastes 15/16 of the tile.

**Check.** Microbenchmark both against the 16-head batched form; compare VGPR
count and time. Related to `../acceleration` Q2 but a distinct shape.

---

## Q3 — Does the 32 KiB FP32 accumulator fit alongside everything else in LDS? `open`

**Why.** `o_acc` at `BLOCK_H=16 × Lkv=512 × 4 B` is 32 KiB of the 64 KiB LDS
budget, before `ql_nope` staging. If it does not fit, either `BLOCK_H` drops
(multiplying cache reads) or the accumulator spills.

**Check.** Compile with `-Rpass-analysis=kernel-resource-usage` once the kernel
exists. Consider keeping `ql_nope` in registers and re-reading `c_KV` instead.

---

## Q4 — Is AITER's gfx942 ASM MLA decode usable as an oracle? `open`

Inherited from `../mi300x/99-open-questions.md` Q10, now sharper: the gfx942
path is hand-written assembly, and ROCm/aiter#4363 reports it faulting at
`page_size=1` — which is the mode vLLM always calls it in. Test before relying
on it; the Triton MLA path is the safer oracle.

---

## Q5 — Does our head-major `ql_nope` layout match what the score matmul wants? `open`

vLLM writes the `bmm` result "straight into a token-major (B, N, L) buffer so the
MQA query is already contiguous". At batch 1, B=1, so token-major and head-major
coincide — but confirm the MFMA fragment layout wants `[16, 512]` row-major
before committing phase A's output layout.
