# 03 — The Ten Questions, Answered

| # | Question | vLLM | FlashMLA | AITER gfx942 | **Our choice** |
|---|---|---|---|---|---|
| 1 | Where does `W_UK` reassociation happen? | **Runtime** `bmm`, per head | runtime (MQA mode) | runtime | **Runtime, in the task prologue** |
| 2 | Tiling of heads × latent × sequence | `BLOCK_H=16`, `BLOCK_DMODEL=512`, `BLOCK=32` | tile-scheduler metadata | ASM, opaque | `BLOCK_H=16`, KV tile 32 |
| 3 | Is `k_pe` separate or concatenated? | **Separate** (`BLOCK_DPE=64`) | separate | — | **Separate** |
| 4 | KV layout | paged, flattened to `page_size=1` for AITER | paged, token-major rows | wants `page_size=1` | **Contiguous split arrays** |
| 5 | Split-KV chunking + merge | `lse` + `o_acc` FP32, `next_pow2(S/512)` splits | `get_mla_metadata` precomputes splits | — | **32–64 splits** (see `04`) |
| 6 | MFMA or VALU at the score matmul? | **MFMA 16×16** (`matrix_instr_nonkdim: 16`) | MFMA/WGMMA | — | **MFMA 16×16** |
| 7 | Softmax precision, where `softmax_scale` applies | FP32 accumulator, scale on `qk` | FP32 | — | **FP32**, scale = 0.1147213867929261 |
| 8 | Solving low head parallelism | rely on `batch`; splits secondary | tile scheduler | persistent scheduling | **Sequence splits only** |
| 9 | Register / LDS footprint | `waves_per_eu: 1` | — | — | budget for 1 wave/SIMD |
| 10 | Real gfx942 path? | Triton yes, AITER ASM yes | no (Hopper) | ASM only | **Write our own** |

## Notes on the answers that matter

### Q1 — runtime reassociation, and how it is shaped

```python
# (N, B, P) x (N, P, L) -> (N, B, L), written into a token-major (B, N, L) buffer
torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope.transpose(0, 1))
```

with the comment:

> "Write the (N, B, L) bmm result straight into a token-major (B, N, L) buffer
> so the MQA query is already contiguous"

So: **16 small `[1,128] × [128,512]` products**, results laid out head-major so
the subsequent MQA query `[N, L+R]` is contiguous. That is exactly the shape our
`../deepseek-v2-lite/02-mla.md` derived, and the layout hint is free advice.

### Q4 — our layout is the one the kernels want

AITER flattens paged caches to `page_size=1` before calling. FlashMLA uses
token-major rows. Nobody wants pages inside the kernel; paging exists for the
serving system's allocator, which we do not have. **Contiguous
`c_KV[S,512]` + `k_pe[S,64]` is right**, and splitting the two arrays matches
`BLOCK_DMODEL`/`BLOCK_DPE` being separate tiles.

### Q6 — MFMA is correct here, and this corrects an earlier claim

`../acceleration/03-kernel-craft.md` argued MFMA wastes 15/16 of a tile at M=1.
That is true for **weight GEMVs** (`q_proj`, experts), where M = batch = 1. It is
**false for MLA attention**, where M = `BLOCK_H` = 16 query heads sharing one KV
read. A `V_MFMA_F32_16X16X16_BF16` tile is filled exactly.

vLLM ships `matrix_instr_nonkdim: 16` for ROCm, which is direct evidence.

### Q8 — nobody has solved batch-1 head parallelism, because nobody needed to

Every implementation surveyed relies on `batch × heads` for parallelism.
vLLM's split heuristic tops out at `next_pow2(S / 512)`; FlashMLA precomputes a
tile schedule sized to the batch. At batch 1 with 16 heads in one block, all of
them degenerate to a handful of workgroups.

**This is genuinely our problem to solve**, not something we can copy. It is
also the one place where the Fleet task model helps: a Chiplet-task can spread
one split across 37 workers on an XCD, which a monolithic kernel launch cannot
express as cleanly.

### Q3 — why `nope` and `pe` stay separate

Concatenating into a 576-wide operand would force a single tile of width 576,
which is not a power of two and pads to 1024 — wasting 44% of the tile. Keeping
`512 + 64` lets the latent use a clean 512-wide tile and the RoPE slice a 64-wide
one. Their score contributions are summed before the softmax.

`next_pow2(576) = 1024` appears explicitly in the non-grouped Triton path
(`BLOCK_DMODEL = next_power_of_2(Lk)`), which is why the grouped path
special-cases `Lk == 576`.
