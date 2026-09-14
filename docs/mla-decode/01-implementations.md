# 01 — Implementations Surveyed

| Project | File | Backend | Arch | Readable? | Useful for |
|---|---|---|---|---|---|
| vLLM | `model_executor/layers/attention/mla_attention.py` (129 KB) | base for all MLA | any | **yes** | The MLA derivation itself, absorption, weight prep |
| vLLM | `v1/attention/ops/triton_decode_attention.py` (23 KB) | Triton | portable + ROCm tuning | **yes** | Tiling, split-KV, MFMA hints |
| vLLM | `v1/attention/backends/mla/triton_mla.py` (13 KB) | Triton | portable | **yes** | Split-count heuristic |
| vLLM | `v1/attention/backends/mla/rocm_aiter_mla.py` (82 KB) | AITER | gfx942 / gfx950 | orchestration only | Head-count constraints, gfx942 path selection |
| AITER | `mla_decode` ASM | hand-written | **gfx942** | **no** | — |
| FlashMLA | `README.md` | CUDA | Hopper (sm90) | partial | MQA framing, tile-scheduler metadata |

Files fetched into `sources/`. vLLM has 24 files in its MLA backend directory;
we read the four that matter for dense batch-1 decode and ignored the sparse,
CPU, XPU, CUTLASS, and FlashInfer variants.

## The gfx942 situation

`rocm_aiter_mla.py` documents the backend selection explicitly:

> "`auto` (default): let the arch decide — divisor head counts keep the Gluon
> decode where a build exists (gfx950), everything else (non-divisor counts and
> all counts on gfx942) uses the padded persistent-scheduling ASM decode. ...
> On gfx942 (no Gluon build) the ASM path is always used."

So on **our** hardware, AITER's MLA decode is a **hand-written assembly kernel**.
Consequences:

- We cannot read its tiling from source. Our design has to come from the Triton
  and vLLM-base implementations, which are readable and portable.
- It is still usable as a **numerical oracle** and as a **fallback** — this
  partially answers `../mi300x/99-open-questions.md` Q10, though whether it runs
  correctly for our shapes still needs the machine.
- The newest MLA work targeting gfx950 (Gluon) does not apply to us, confirming
  the caution in `../mi300x/05-software-stack.md`.

## Head-count constraint — we pass, but only just

```python
"ROCM AITER MLA requires a positive multiple of 16 heads, or an unaligned head
 count up to 128 (padded to the next multiple of 16), but got {num_heads}."
```

`_AITER_MIN_MLA_HEADS = 16`. DeepSeek-Coder-V2-Lite has **exactly 16** query
heads, so `16 % 16 == 0` and no padding is applied. Worth knowing that this was
close: a model with 12 heads would be padded to 16, wasting a quarter of the
attention work.

Note also: DeepSeek-V3 has 128 heads and is typically run at TP=8, giving **16
heads per rank**. So the shapes AITER and vLLM are tuned for — `W_K [16, 512,
128]`, `W_V [16, 128, 512]`, as the comments in `mla_attention.py` show — are
**exactly our shapes**. We are, by coincidence, in the well-trodden
configuration.

## Paging

> "The aiter MLA decode kernel always operates with `page_size=1` internally
> (the wrapper flattens `kv_buffer` via `.view(-1, 1, 1, H)`)."

AITER's kernel does not want pages; vLLM flattens its paged cache to per-token
rows before calling it. Our contiguous `c_KV[S, 512]` / `k_pe[S, 64]` layout is
therefore *closer* to what the kernel wants than a paged layout would be —
a point in favour of the choice in `../deepseek-v2-lite/02-mla.md`.

There is an open AITER issue (ROCm/aiter#4363) about a gfx942 persistent MLA
decode kernel faulting at `page_size=1`; worth knowing before depending on it.
