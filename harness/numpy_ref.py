"""NumPy specification of the four new kernels' math.

This is what the GPU kernels are tested against (docs/design-doc/10-local-work.md
L2, 07-correctness.md kernel_tests.py). Everything is float32 with explicit
BF16 rounding at the points where the design stores BF16 or feeds an MFMA:

  mla_prep      kv_a_layernorm, RoPE, cache append, ql_nope = q_nope @ W_uk
  mla_attend    split-KV online softmax over the latent cache -> partials
  mla_merge_uv  merge the partials, then attn[h] = o[h] @ W_uv[h]^T
  moe_router    FP32 GEMV, softmax, top-k, forced experts -> topk_w, routing, mask
  moe_w13       the expert gate-up of every routed slot -> mid (L4)
  moe_w2        silu(gate) * up, then the expert down projection -> out8 (O2, L3)

Rounding follows the reference model where the design says "match the
reference" (docs/design-doc/01-execution-flow.md, constants table):
RMSNorm rounds the normalized value to BF16 before the BF16 weight multiply
(modeling_deepseek.py:103-108); RoPE runs in BF16 arithmetic on BF16 tables
(:339-372); softmax probabilities are cast to BF16 before the value product
(:889-891). Matmul accumulation is FP32 and the summation order is not
specified, so kernels are compared with a tolerance, not bit for bit.

Partials follow the CK split-KV convention the runtime's merge already
implements (tasks/ampere/merge_splitkv.cuh): per split and head, the
normalized output o_j [512] and lse_j = m_j + ln(l_j) in natural log, packed
as [n_splits, heads, 513]; an empty split stores o = 0 and lse = -inf.
"""
import numpy as np

F32 = np.float32


# ----------------------------------------------------------------------------
# BF16 emulation


def bf16(x):
    """Round a float32 array to BF16 precision (round to nearest even), as float32."""
    x = np.ascontiguousarray(np.asarray(x, dtype=F32))
    u = x.view(np.uint32)
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) & 0xFFFF0000
    return u.view(F32)


def from_torch_bf16(t):
    """torch bf16/float tensor -> float32 ndarray."""
    return t.detach().float().cpu().numpy().astype(F32)


# ----------------------------------------------------------------------------
# pieces shared by the kernels


def rmsnorm(x, w, eps=1e-6):
    """DeepseekV2RMSNorm on BF16 inputs: FP32 statistics, BF16 normalized value,
    BF16 multiply by the weight."""
    x = np.asarray(x, F32)
    var = np.mean(x * x, axis=-1, keepdims=True, dtype=F32)
    xn = bf16(x * (1.0 / np.sqrt(var + F32(eps))).astype(F32))
    return bf16(np.asarray(w, F32) * xn)


def rope(x, cos_row, sin_row):
    """apply_rotary_pos_emb on the last dim (64): de-interleave (even, odd) pairs
    into two halves, then x * cos + rotate_half(x) * sin, BF16 at every op."""
    x = np.asarray(x, F32)
    d = x.shape[-1]
    x = x.reshape(*x.shape[:-1], d // 2, 2).swapaxes(-1, -2).reshape(*x.shape[:-1], d)
    h = d // 2
    rot = np.concatenate([-x[..., h:], x[..., :h]], axis=-1)
    return bf16(bf16(x * cos_row) + bf16(rot * sin_row))


def softmax_f32(x):
    x = np.asarray(x, F32)
    m = x.max(axis=-1, keepdims=True)
    e = np.exp(x - m, dtype=F32)
    return e / e.sum(axis=-1, keepdims=True, dtype=F32)


# ----------------------------------------------------------------------------
# mla_prep


def mla_prep(qkva, w_kv_norm, W_uk, cos_row, sin_row, *, nh=16, d_n=128, d_r=64, d_c=512,
             eps=1e-6):
    """One decode token.

    qkva     [nh*(d_n+d_r) + d_c + d_r] BF16 (the fused q_proj | kv_a_proj output)
    w_kv_norm [d_c]; W_uk [nh, d_n, d_c] BF16; cos_row, sin_row [d_r] BF16 (row `step`)
    Returns c_row [d_c], k_pe_row [d_r], ql_nope [nh, d_c], q_pe [nh, d_r], all BF16 values.
    """
    q_out = nh * (d_n + d_r)
    q = np.asarray(qkva[:q_out], F32).reshape(nh, d_n + d_r)
    c_raw = np.asarray(qkva[q_out:q_out + d_c], F32)
    k_pe_raw = np.asarray(qkva[q_out + d_c:q_out + d_c + d_r], F32)
    c_row = rmsnorm(c_raw, w_kv_norm, eps)
    k_pe_row = rope(k_pe_raw, cos_row, sin_row)
    q_nope, q_pe = q[:, :d_n], q[:, d_n:]
    q_pe = rope(q_pe, cos_row, sin_row)
    # 16 products [1, d_n] x [d_n, d_c], FP32 accumulate, stored BF16
    ql_nope = bf16(np.einsum("hn,hnc->hc", q_nope, np.asarray(W_uk, F32), dtype=F32))
    return c_row, k_pe_row, ql_nope, q_pe


# ----------------------------------------------------------------------------
# mla_attend (split-KV) and the merge


def mla_attend(ql_nope, q_pe, c_kv, k_pe, step, softmax_scale, *, split=32, n_splits=None,
               debug_scores=False):
    """Split-KV attention over cache rows [0, step].

    ql_nope [nh, d_c], q_pe [nh, d_r] BF16; c_kv [S_max, d_c], k_pe [S_max, d_r] BF16.
    Returns partials [n_splits, nh, d_c + 1] FP32 (o_j normalized, lse_j at [.., d_c]).
    With debug_scores=True also returns the scaled scores [nh, step + 1] FP32.
    """
    ql_nope = np.asarray(ql_nope, F32)
    q_pe = np.asarray(q_pe, F32)
    c_kv = np.asarray(c_kv, F32)
    k_pe = np.asarray(k_pe, F32)
    nh, d_c = ql_nope.shape
    s_max = c_kv.shape[0]
    n_splits = n_splits or -(-s_max // split)
    partials = np.zeros((n_splits, nh, d_c + 1), F32)
    scores_all = np.zeros((nh, step + 1), F32) if debug_scores else None
    for j in range(n_splits):
        lo, hi = j * split, min(j * split + split, step + 1)
        if lo > step:
            partials[j, :, d_c] = -np.inf          # empty split: o = 0, lse = -inf
            continue
        m = np.full(nh, -np.inf, F32)
        l = np.zeros(nh, F32)
        acc = np.zeros((nh, d_c), F32)
        for t0 in range(lo, hi, split):            # one tile per split at split == tile
            t1 = min(t0 + split, hi)
            s = ql_nope @ c_kv[t0:t1].T + q_pe @ k_pe[t0:t1].T     # FP32 accumulate
            s = (s * F32(softmax_scale)).astype(F32)
            if debug_scores:
                scores_all[:, t0:t1] = s
            m_new = np.maximum(m, s.max(axis=1))
            alpha = np.exp(m - m_new, dtype=F32)
            p = np.exp(s - m_new[:, None], dtype=F32)
            l = l * alpha + p.sum(axis=1, dtype=F32)
            acc = acc * alpha[:, None] + bf16(p) @ c_kv[t0:t1]     # BF16 probabilities to the MFMA
            m = m_new
        partials[j, :, :d_c] = acc / l[:, None]
        partials[j, :, d_c] = m + np.log(l, dtype=F32)
    return (partials, scores_all) if debug_scores else partials


def merge_partials(partials, step, *, split=32, d_c=None):
    """CK-style merge of the live splits: o = sum_j o_j exp(lse_j - M) / sum_j exp(lse_j - M).

    d_c: the logical o-width; the lse is column d_c. Defaults to partials.shape[2] - 1, but the
    device buffer pads the row past d_c + 1 (P2), so pass d_c explicitly for a padded input."""
    nh = partials.shape[1]
    if d_c is None:
        d_c = partials.shape[2] - 1
    live = -(-(step + 1) // split)
    lse = partials[:live, :, d_c]                     # [live, nh]
    M = lse.max(axis=0)                               # [nh]
    w = np.exp(lse - M[None, :], dtype=F32)           # [live, nh]; exp(-inf) = 0 for empty splits
    o = np.einsum("jh,jhc->hc", w, partials[:live, :, :d_c], dtype=F32) / w.sum(axis=0)[:, None]
    return o.astype(F32)                              # [nh, d_c]


def mla_merge_uv(partials, W_uv, step, *, split=32, d_c=None):
    """attn [nh * d_v] BF16: per head, merge then o[h] @ W_uv[h]^T with FP32 accumulate.

    o is rounded to BF16 before the product (it is an MFMA operand). d_c: see merge_partials."""
    o = bf16(merge_partials(partials, step, split=split, d_c=d_c))
    attn = np.einsum("hc,hvc->hv", o, np.asarray(W_uv, F32), dtype=F32)
    return bf16(attn.reshape(-1))


def attention_reference_decompressed(q_nope, q_pe, c_kv, k_pe, W_uk, W_uv, step, softmax_scale):
    """The reference's arithmetic (decompressed keys and values) for the same
    token, used to bound the reassociation error: k_nope = W_uk @ c, v = W_uv @ c
    per position, BF16 like the reference's kv_b_proj output."""
    c = np.asarray(c_kv[:step + 1], F32)
    k_nope = bf16(np.einsum("hnc,sc->hsn", np.asarray(W_uk, F32), c, dtype=F32))   # [nh, S, d_n]
    v = bf16(np.einsum("hvc,sc->hsv", np.asarray(W_uv, F32), c, dtype=F32))        # [nh, S, d_v]
    s = np.einsum("hn,hsn->hs", np.asarray(q_nope, F32), k_nope, dtype=F32)
    s = s + np.asarray(q_pe, F32) @ np.asarray(k_pe[:step + 1], F32).T
    s = bf16(s * F32(softmax_scale))                  # the reference's BF16 attn_weights
    p = bf16(softmax_f32(s))
    return bf16(np.einsum("hs,hsv->hv", p, v, dtype=F32).reshape(-1)), s


# ----------------------------------------------------------------------------
# moe_router


def moe_router(h, W_gate, *, topk=6, n_experts=64, forced=(64, 65), scaling=1.0):
    """FP32 router with forced shared experts (D6, D7).

    h [H] BF16, W_gate [E, H] BF16. Returns
      logits  [E] FP32
      topk_w  [topk + len(forced)] FP32: softmax probabilities x scaling, then 1.0 per forced
      routing [E + len(forced)] int32: slot + 1 for a selected expert, 0 otherwise
      mask    [E + len(forced) + 1] int32: the active expert ids in slot order, -1 in the
              unused entries (as the stock and the new kernel write them), then the count
    Ties in the top-k go to the lower expert index (a stated choice, 02-task-graph.md).
    """
    logits = (np.asarray(W_gate, F32) @ np.asarray(h, F32)).astype(F32)
    p = softmax_f32(logits)
    idx = []
    remaining = p.copy()
    for _ in range(topk):
        e = int(np.argmax(remaining))                 # argmax returns the first (lowest) index on ties
        idx.append(e)
        remaining[e] = -np.inf
    ids = idx + list(forced)
    n_slots = topk + len(forced)
    n_total = n_experts + len(forced)
    topk_w = np.array([p[e] * scaling for e in idx] + [1.0] * len(forced), F32)
    routing = np.zeros(n_total, np.int32)
    for k, e in enumerate(ids):
        routing[e] = k + 1
    mask = np.full(n_total + 1, -1, np.int32)
    mask[:n_slots] = ids
    mask[n_total] = n_slots
    return logits, topk_w, routing, mask


def moe_router_norm(x_res, w_norm, W_gate, *, eps=1e-6, topk=6, n_experts=64, forced=(64, 65),
                    scaling=1.0):
    """The router with the post-attention norm folded in (docs/gpu-experiments/03-acceleration,
    O1; the kernel's NORM = true path): h = rmsnorm(x_res, w_norm), then moe_router on h.
    Returns (h, logits, topk_w, routing, mask); h is the row the expert gate-up reads."""
    h = rmsnorm(x_res, w_norm, eps)
    logits, topk_w, routing, mask = moe_router(h, W_gate, topk=topk, n_experts=n_experts, forced=forced,
                                               scaling=scaling)
    return h, logits, topk_w, routing, mask


def silu(x):
    """SiLU in FP32, the arithmetic of the kernels' fast_silu (silu_mul_mi300.cuh):
    x / (1 + exp(-x)), no BF16 rounding of its own."""
    x = np.asarray(x, F32)
    return (x / (F32(1.0) + np.exp(-x, dtype=F32))).astype(F32)


def moe_w13(h, W13, mask, n_slots=8):
    """The expert gate-up at batch 1 (the gang_moe_w13 task, L4 of
    docs/gpu-experiments/04-kernels): slot s takes expert mask[s] and computes
    mid[s] = bf16(W13[e] @ h), the gate rows then the up rows of the expert as the packing
    lays them out (pack_weights.pack_moe: rows [0, I) gate, [I, 2I) up).

    h [K] BF16, W13 [E, N, K] BF16, mask [E + 1] int32 (the active expert ids in slot order,
    then the count, as the router writes them). Returns mid [n_slots, N] float32 of BF16
    values; a slot past the active count is NaN, the sentinel an untouched row keeps.
    """
    h = np.asarray(h, F32)
    W13 = np.asarray(W13, F32)
    n_active = int(mask[-1])
    mid = np.full((n_slots, W13.shape[1]), np.nan, F32)
    for s in range(n_active):
        mid[s] = bf16(W13[int(mask[s])] @ h)
    return mid


def moe_w2(mid, W2, mask, n_slots=8):
    """The expert down projection with the silu-mul in its prologue (the
    gang_moe_w2_silu task, O2 and L3): per slot, act = bf16(silu(gate) * up) element by
    element as silu_mul_task_impl rounds it, then out[s] = bf16(W2[e] @ act).

    mid [n_slots, 2K] BF16 (gate in columns [0, K), up in [K, 2K)), W2 [E, N, K] BF16,
    mask as in moe_w13. Returns out8 [n_slots, N] float32 of BF16 values, NaN past the
    active count.
    """
    mid = np.asarray(mid, F32)
    W2 = np.asarray(W2, F32)
    k = W2.shape[2]
    n_active = int(mask[-1])
    out = np.full((n_slots, W2.shape[1]), np.nan, F32)
    for s in range(n_active):
        act = bf16(silu(mid[s, :k]) * mid[s, k:2 * k])
        out[s] = bf16(W2[int(mask[s])] @ act)
    return out


def linear_norm(x, w_norm, W, eps=1e-6):
    """The per-tile linear with the input norm in its prologue (docs/gpu-experiments/03-acceleration,
    O3; the linear_norm_mi300 task): rmsnorm(x, w_norm) rounded to BF16, then W @ h with FP32
    accumulation and one BF16 rounding of the result, as the CK linear does. Returns (h, out)."""
    h = rmsnorm(x, w_norm, eps)
    out = bf16(np.asarray(W, F32) @ h.astype(F32))
    return h, out


def linear(x, W):
    """The plain per-tile linear at batch 1 (the linear_gemv_mi300 task without a prologue):
    FP32 accumulation of exact BF16 products, one BF16 rounding of the result."""
    return bf16(np.asarray(W, F32) @ np.asarray(x, F32))


def linear_residual(x, W, res):
    """The same linear with the residual added in FP32 before the rounding (the task's
    RESIDUAL flag; o_proj and layer 0's down projection)."""
    return bf16(np.asarray(W, F32) @ np.asarray(x, F32) + np.asarray(res, F32))


def moe_combine(out8, topk_w, x_res):
    """moe_mul_sum_add: x + sum_k w_k * out_k, FP32 accumulate, BF16 result."""
    acc = np.asarray(x_res, F32).copy()
    for k in range(len(topk_w)):
        acc = acc + np.asarray(out8[k], F32) * F32(topk_w[k])
    return bf16(acc)
