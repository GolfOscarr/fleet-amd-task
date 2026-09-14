#!/usr/bin/env python3
"""Enumerate the Fleet task graph for one decode iteration of
DeepSeek-Coder-V2-Lite-Base and derive task counts, event counts, bytes,
and rooflines. Every number in 01-execution-flow.md, 02-task-graph.md and
09-expected-performance.md that is not a config value comes from here.

Shapes from docs/deepseek-v2-lite/01-config.md (checkpoint-verified).
Task-count rules from docs/fleet/04-repo-map.md (code-verified):
  gang ops        -> 8 tasks (one per XCD), tiles per XCD listed separately
  rmsnorm         -> max_num_batched_tokens = 1 task
  moe_silu_mul    -> batch x topk tasks
  moe_mul_sum_add -> batch x hidden/256 tasks
"""

MiB = 1024 * 1024
BF16 = 2

H = 2048          # hidden_size
V = 102400        # vocab_size
L = 27            # num_hidden_layers
NH = 16           # num_attention_heads
D_C = 512         # kv_lora_rank
D_R = 64          # qk_rope_head_dim
D_N = 128         # qk_nope_head_dim
D_V = 128         # v_head_dim
Q_OUT = NH * (D_N + D_R)      # 3072
KVA_OUT = D_C + D_R           # 576
KVB_OUT = NH * (D_N + D_V)    # 4096
I_DENSE = 10944               # intermediate_size (layer 0)
I_DENSE_PAD = 11264           # D8: padded to 8 x 1408 = 44 x 256
I_MOE = 1408                  # moe_intermediate_size
E_ROUTED = 64
E_TOTAL = 66                  # D6: + 2 shared halves
TOPK = 6
TOPK_TOTAL = 8                # D6: + 2 forced
S0 = 1024                     # positions attended in iteration 0
S_MAX = 1056                  # prompt + 32 outputs
SPLIT = 32                    # positions per attention split
N_SPLITS = -(-S_MAX // SPLIT) # 33
SPLITS_PER_XCD = -(-N_SPLITS // 8)  # 5
ARGMAX_SLICES = 50

XCDS = 8
WORKERS = 296


def lin_bytes(n, k):
    return n * k * BF16


# (op name, kernel, tasks, tiles per task (gang) or '-', weight bytes, cache bytes, new?)
def attention_ops(s_eff):
    cache = s_eff * (D_C + D_R) * BF16
    return [
        ("input_layernorm", "rmsnorm", 1, "-", H * BF16, 0, False),
        ("qkv_a_proj (q_proj | kv_a_proj_with_mqa)", "gang_linear_mi300", 8,
         (Q_OUT + KVA_OUT) // 8 // 24, lin_bytes(Q_OUT + KVA_OUT, H), 0, False),
        ("mla_prep (kv_a_layernorm, RoPE, append, q_nope @ W_UK)", "mla_prep_mi300 (NEW)", 1, "-",
         lin_bytes(NH * D_N, D_C) + D_C * BF16, 0, True),
        ("mla_attend (split-KV, 16 heads)", "mla_attend_mi300 (NEW)", 8, SPLITS_PER_XCD, 0, cache, True),
        ("mla_merge_uv (merge + W_UV per head)", "mla_merge_uv_mi300 (NEW)", 8, NH // 8,
         lin_bytes(NH * D_V, D_C), 0, True),
        ("o_proj + residual", "gang_linear_res_mi300", 8, H // 8 // 32, lin_bytes(H, H), 0, False),
    ]


def dense_mlp_ops():
    return [
        ("post_attention_layernorm", "rmsnorm", 1, "-", H * BF16, 0, False),
        ("gate_up (shuffled) + SiLU*mul, padded to 11264", "gang_linear_silu_mi300", 8,
         (2 * I_DENSE_PAD) // 8 // 64 // 2, lin_bytes(2 * I_DENSE_PAD, H), 0, False),
        ("down_proj + residual, K padded to 11264", "gang_linear_res_mi300", 8, H // 8 // 32,
         lin_bytes(H, I_DENSE_PAD), 0, False),
    ]


def moe_mlp_ops():
    w13_active = TOPK_TOTAL * lin_bytes(2 * I_MOE, H)
    w2_active = TOPK_TOTAL * lin_bytes(H, I_MOE)
    return [
        ("post_attention_layernorm", "rmsnorm", 1, "-", H * BF16, 0, False),
        ("router (FP32 GEMV, softmax, top-6, + forced 64,65)", "moe_router_mi300 (NEW)", 1, "-",
         lin_bytes(E_ROUTED, H), 0, True),
        ("W13 for 8 active of 66 experts", "gang_moe_w13_linear_mi300", 8, (2 * I_MOE) // 64,
         w13_active, 0, False),
        ("silu * mul per (token, slot)", "moe_silu_mul", TOPK_TOTAL, "-", 0, 0, False),
        ("W2 for 8 active experts", "gang_moe_w2_linear_mi300", 8, H // 64, w2_active, 0, False),
        ("weighted sum of 8 + residual", "moe_mul_sum_add_mi300", H // 256, "-", 0, 0, False),
    ]


def head_ops():
    return [
        ("model.norm", "rmsnorm", 1, "-", H * BF16, 0, False),
        ("lm_head", "gang_linear_mi300", 8, V // 8 // 64, lin_bytes(V, H), 0, False),
        ("argmax_partial", "argmax_partial", ARGMAX_SLICES, "-", 0, 0, False),
        ("argmax_reduce -> output_tokens", "argmax_reduce", 1, "-", 0, 0, False),
    ]


def embed_ops():
    return [("embed_tokens[tokens[step]]", "embedding (input_source=0)", 1, "-", H * BF16, 0, False)]


def summarize(ops):
    tasks = sum(o[2] for o in ops)
    wb = sum(o[4] for o in ops)
    cb = sum(o[5] for o in ops)
    return tasks, len(ops), wb, cb


def table(title, ops):
    print(f"\n### {title}\n")
    print("| # | Op | Kernel | Tasks | Tiles/task | Weight MiB | Cache MiB |")
    print("|---|---|---|---|---|---|---|")
    for i, (name, kern, tasks, tiles, wb, cb, new) in enumerate(ops, 1):
        print(f"| {i} | {name} | `{kern}` | {tasks} | {tiles} | {wb / MiB:.3f} | {cb / MiB:.3f} |")
    t, n, wb, cb = summarize(ops)
    print(f"| | **total** | | **{t}** | | **{wb / MiB:.3f}** | **{cb / MiB:.3f}** |")
    return t, n, wb, cb


def main():
    s_eff = S0
    emb = embed_ops()
    l0 = attention_ops(s_eff) + dense_mlp_ops()
    lm = attention_ops(s_eff) + moe_mlp_ops()
    hd = head_ops()

    te, ne, we, ce = table("Prologue (per iteration)", emb)
    t0, n0, w0, c0 = table("Layer 0 (dense)", l0)
    tm, nm, wm, cm = table("MoE layer (each of layers 1-26)", lm)
    th, nh, wh, ch = table("Head", hd)

    tasks = te + t0 + 26 * tm + th
    ops = ne + n0 + 26 * nm + nh
    wbytes = we + w0 + 26 * wm + wh
    cbytes = ce + c0 + 26 * cm + ch
    total = wbytes + cbytes
    print("\n### Per iteration (S = 1024)\n")
    print(f"ops (events): {ops}")
    print(f"tasks in graph: {tasks}  (+ TASK_BEGIN_TASK_GRAPH and TASK_TERMINATE = {tasks + 2} entries)")
    print(f"tasks per worker: {tasks / WORKERS:.2f}")
    print(f"weight bytes: {wbytes / MiB:.1f} MiB   cache bytes: {cbytes / MiB:.2f} MiB   total: {total / MiB:.1f} MiB")
    for bw in (5.3e12, 4.3e12, 3.66e12):
        print(f"  @ {bw / 1e12:.2f} TB/s: {total / bw * 1e6:8.1f} us/token  {1e6 / (total / bw * 1e6):7.0f} tok/s")
    lt = wm + cm
    print(f"\nlayer 1 bytes: {lt / MiB:.2f} MiB")
    for bw in (5.3e12, 4.3e12, 3.66e12):
        print(f"  @ {bw / 1e12:.2f} TB/s: {lt / bw * 1e6:6.1f} us")
    print("\nnew kernels:", sorted({o[1] for o in l0 + lm if o[6]}))
    print(f"splits: {N_SPLITS} of {SPLIT} positions over S_max={S_MAX}; {SPLITS_PER_XCD} tiles per XCD")
    print(f"partial buffer per layer: {N_SPLITS * NH * (D_C + 1) * 4 / 1024:.0f} KiB (FP32 [splits][16][513])")
    print(f"dense padding cost per token: {(lin_bytes(2*I_DENSE_PAD,H)+lin_bytes(H,I_DENSE_PAD)-lin_bytes(2*I_DENSE,H)-lin_bytes(H,I_DENSE))/MiB:.2f} MiB")
    # tiling checks (assertions mirror persistent_kernel.py and the CK pipeline)
    assert (Q_OUT + KVA_OUT) % 8 == 0 and ((Q_OUT + KVA_OUT) // 8) % 24 == 0
    assert (2 * I_DENSE_PAD) % 8 == 0 and ((2 * I_DENSE_PAD) // 8) % 64 == 0 and (((2 * I_DENSE_PAD) // 8) // 64) % 2 == 0
    assert I_DENSE_PAD % 256 == 0 and H % 256 == 0 and D_C % 256 == 0
    assert (2 * I_MOE) % 64 == 0 and I_MOE % 128 == 0
    assert V % 8 == 0 and (V // 8) % 64 == 0 and V % ARGMAX_SLICES == 0
    print("tiling assertions: ok")


if __name__ == "__main__":
    main()
