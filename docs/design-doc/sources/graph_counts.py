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

This is the shipped graph, the one the plan builds with no flag: 326 operators
and 2,285 tasks. The plan's optional forms move both counts (the fusions of
O1 to O3, the per-tile linears, and --gemv-linears with --linear-grid and
--head-grid, L2 and L5 of docs/gpu-experiments/04-kernels); their counts are
not enumerated here but are asserted, flag by flag, in
fleet/tests/test_graph_plan.py against fleet/graph_plan.py.
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


# (op name, kernel, tasks, tiles per task (gang) or '-', weight bytes, cache bytes, status)
# status: "reuse" = Fleet kernel and Python call as shipped; "variant" = one-line
# change to an existing task; "new" = kernel written for this design.
def attention_ops(s_eff):
    cache = s_eff * (D_C + D_R) * BF16
    return [
        ("input_layernorm", "rmsnorm", 1, "-", H * BF16, 0, "reuse"),
        ("qkv_a_proj (q_proj | kv_a_proj_with_mqa)", "gang_linear_mi300", 8,
         (Q_OUT + KVA_OUT) // 8 // 24, lin_bytes(Q_OUT + KVA_OUT, H), 0, "reuse"),
        ("mla_prep (kv_a_layernorm, RoPE, append, q_nope @ W_UK)", "mla_prep_mi300 (NEW)", 16, "-",
         lin_bytes(NH * D_N, D_C) + D_C * BF16, 0, "new"),
        ("mla_attend (split-KV, 16 heads)", "mla_attend_mi300 (NEW)", 8, SPLITS_PER_XCD, 0, cache, "new"),
        ("mla_merge_uv (merge + W_UV per head)", "mla_merge_uv_mi300 (NEW)", 8, NH // 8,
         lin_bytes(NH * D_V, D_C), 0, "new"),
        ("o_proj + residual", "gang_linear_res_mi300", 8, H // 8 // 32, lin_bytes(H, H), 0, "reuse"),
    ]


def dense_mlp_ops():
    return [
        ("post_attention_layernorm", "rmsnorm", 1, "-", H * BF16, 0, "reuse"),
        ("gate_up (shuffled) + SiLU*mul, padded to 11264", "gang_linear_silu_mi300", 8,
         (2 * I_DENSE_PAD) // 8 // 64 // 2, lin_bytes(2 * I_DENSE_PAD, H), 0, "reuse"),
        ("down_proj + residual, K padded to 11264", "gang_linear_res_mi300", 8, H // 8 // 32,
         lin_bytes(H, I_DENSE_PAD), 0, "reuse"),
    ]


def moe_mlp_ops():
    w13_active = TOPK_TOTAL * lin_bytes(2 * I_MOE, H)
    w2_active = TOPK_TOTAL * lin_bytes(H, I_MOE)
    return [
        ("post_attention_layernorm", "rmsnorm", 1, "-", H * BF16, 0, "reuse"),
        ("router (FP32 GEMV, softmax, top-6, + forced 64,65)", "moe_router_mi300 (NEW)", 1, "-",
         lin_bytes(E_ROUTED, H), 0, "new"),
        ("W13 for 8 active of 66 experts", "gang_moe_w13_linear_mi300", 8, (2 * I_MOE) // 64,
         w13_active, 0, "reuse"),
        ("silu * mul per (token, slot)", "moe_silu_mul", TOPK_TOTAL, "-", 0, 0, "reuse"),
        ("W2 for 8 active experts", "gang_moe_w2_linear_mi300", 8, H // 64, w2_active, 0, "reuse"),
        ("weighted sum of 8 + residual", "moe_mul_sum_add_mi300", H // 256, "-", 0, 0, "reuse"),
    ]


def head_ops():
    return [
        ("model.norm", "rmsnorm", 1, "-", H * BF16, 0, "reuse"),
        ("lm_head", "gang_linear_mi300", 8, V // 8 // 64, lin_bytes(V, H), 0, "reuse"),
        ("argmax_partial", "argmax_partial", ARGMAX_SLICES, "-", 0, 0, "reuse"),
        ("argmax_reduce -> tokens[step+1]", "argmax_reduce (variant)", 1, "-", 0, 0, "variant"),
    ]


def embed_ops():
    return [("embed_tokens[tokens[step]]", "embedding (variant: agent-scope load)", 1, "-", H * BF16, 0, "variant")]


def summarize(ops):
    tasks = sum(o[2] for o in ops)
    wb = sum(o[4] for o in ops)
    cb = sum(o[5] for o in ops)
    return tasks, len(ops), wb, cb


def table(title, ops):
    print(f"\n### {title}\n")
    print("| # | Op | Kernel | Tasks | Tiles/task | Weight MiB | Cache MiB |")
    print("|---|---|---|---|---|---|---|")
    for i, (name, kern, tasks, tiles, wb, cb, st) in enumerate(ops, 1):
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
    print("\nnew kernels:", sorted({o[1] for o in l0 + lm if o[6] == "new"}))
    # Fleet-native vs new, per iteration (the "Fleet-native operations / fallbacks" metric)
    allops = emb + l0 + 26 * lm + hd
    print("\n### By status (per iteration)\n")
    print("| Status | Ops | Tasks | Weight+cache MiB | Share of bytes |")
    print("|---|---|---|---|---|")
    for st in ("reuse", "variant", "new"):
        sel = [o for o in allops if o[6] == st]
        b = sum(o[4] + o[5] for o in sel)
        print(f"| {st} | {len(sel)} | {sum(o[2] for o in sel)} | {b / MiB:.1f} | {100 * b / total:.1f}% |")
    # synchronization counts: one release flush per XCD that ran producer tasks
    # (tasks land on XCDs round-robin by graph position), one acquire per task
    # with a dependency (every task but the first op's)
    flushes = sum(min(o[2], XCDS) for o in allops)
    acquires = tasks - emb[0][2]
    print(f"\nrelease flushes (buffer_wbl2 sc1) per iteration: {flushes}")
    print(f"acquires (buffer_inv sc1) per iteration: {acquires}")
    print(f"events per iteration: {ops} op boundaries (+1 where the partition gcd exceeds 1: lm_head -> argmax_partial gives gcd(8, {ARGMAX_SLICES}) = {__import__('math').gcd(8, ARGMAX_SLICES)})")
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
