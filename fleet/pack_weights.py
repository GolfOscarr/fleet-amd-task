#!/usr/bin/env python3
"""Pack the DeepSeek-Coder-V2-Lite-Base checkpoint into the tensors the Fleet
graph attaches (docs/design-doc/04-memory-plan.md).

Every packed tensor is BF16, row-major, `[out, in]` as nn.Linear stores it.
The pure functions below take tensors and dims only, so they run on random
tensors in the tests; the loader at the bottom reads the safetensors shards
by name and is the only part that needs the checkpoint.

    python fleet/pack_weights.py --model-dir <snapshot dir> --dry-run   # shapes from the index only
"""
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

XCDS = 8
TILE_N_QKVA = 24
TILE_N_LM = 64
SILU_GROUP_ROWS = 128       # gang_linear_silu: 128 gate rows then 128 up rows per group
CHECKPOINT_BYTES = 31_412_968_448
DENSE_PAD_BYTES = 3 * 1024 * 1024 + 768 * 1024      # 3.75 MiB (D8)
EXPECTED_PACKED_BYTES = CHECKPOINT_BYTES + DENSE_PAD_BYTES


@dataclass(frozen=True)
class Dims:
    H: int          # hidden_size
    NH: int         # num_attention_heads
    D_N: int        # qk_nope_head_dim
    D_R: int        # qk_rope_head_dim
    D_V: int        # v_head_dim
    D_C: int        # kv_lora_rank
    V: int          # vocab_size
    L: int          # num_hidden_layers
    I_DENSE: int    # intermediate_size (layer 0)
    I_MOE: int      # moe_intermediate_size
    E: int          # n_routed_experts
    TOPK: int       # num_experts_per_tok
    N_SHARED: int   # n_shared_experts

    @classmethod
    def from_config(cls, cfg):
        get = (lambda k: getattr(cfg, k)) if not isinstance(cfg, dict) else (lambda k: cfg[k])
        return cls(
            H=get("hidden_size"), NH=get("num_attention_heads"),
            D_N=get("qk_nope_head_dim"), D_R=get("qk_rope_head_dim"),
            D_V=get("v_head_dim"), D_C=get("kv_lora_rank"), V=get("vocab_size"),
            L=get("num_hidden_layers"), I_DENSE=get("intermediate_size"),
            I_MOE=get("moe_intermediate_size"), E=get("n_routed_experts"),
            TOPK=get("num_experts_per_tok"), N_SHARED=get("n_shared_experts"),
        )

    @property
    def Q_OUT(self):
        return self.NH * (self.D_N + self.D_R)

    @property
    def KVA_OUT(self):
        return self.D_C + self.D_R

    @property
    def KVB_OUT(self):
        return self.NH * (self.D_N + self.D_V)

    @property
    def I_DENSE_PAD(self):
        # per-XCD chunk of the shuffled gate/up must be a multiple of 128 rows
        # and K of W_down_pad a multiple of 256; 1024 = 8 XCDs x 128 covers both
        return math.ceil(self.I_DENSE / 1024) * 1024

    @property
    def E_TOTAL(self):
        return self.E + self.N_SHARED

    @property
    def I_SHARED(self):
        return self.I_MOE * self.N_SHARED

    @property
    def SILU_GROUPS(self):
        return self.I_DENSE_PAD // SILU_GROUP_ROWS


REAL_DIMS = Dims(H=2048, NH=16, D_N=128, D_R=64, D_V=128, D_C=512, V=102400, L=27,
                 I_DENSE=10944, I_MOE=1408, E=64, TOPK=6, N_SHARED=2)


def check_tiling(d: Dims):
    """The constraints of the reused kernels (04-memory-plan.md, graph_counts.py)."""
    assert (d.Q_OUT + d.KVA_OUT) % XCDS == 0 and ((d.Q_OUT + d.KVA_OUT) // XCDS) % TILE_N_QKVA == 0
    assert d.H % 256 == 0                     # K of every CK linear
    assert (2 * d.I_MOE) % 64 == 0            # W13 N per expert
    assert d.I_MOE % 128 == 0                 # W2 K
    assert d.I_DENSE_PAD % 256 == 0           # W_down_pad K
    assert (d.I_DENSE_PAD // XCDS) % SILU_GROUP_ROWS == 0
    assert d.V % XCDS == 0 and (d.V // XCDS) % TILE_N_LM == 0


# ----------------------------------------------------------------------------
# pure packing functions


def pad_rows(w, n_rows):
    assert w.shape[0] <= n_rows
    return F.pad(w, (0, 0, 0, n_rows - w.shape[0]))


def pad_cols(w, n_cols):
    assert w.shape[1] <= n_cols
    return F.pad(w, (0, n_cols - w.shape[1]))


def shuffle_rows(tensors, num_groups):
    """Host replica of the runtime's ShuffledTorchTensor copy (runtime.cc:904-938).

    Output = for g in 0..G-1: [t0.chunk(G)[g]; t1.chunk(G)[g]; ...], concatenated.
    """
    for t in tensors:
        assert t.shape[0] % num_groups == 0, (t.shape, num_groups)
        assert t.shape[1:] == tensors[0].shape[1:]
    chunks = [t.chunk(num_groups, dim=0) for t in tensors]
    return torch.cat([c[g] for g in range(num_groups) for c in chunks], dim=0)


def unshuffle_rows(shuffled, row_counts, num_groups):
    """Inverse of shuffle_rows: returns the list of original tensors."""
    per_group = [n // num_groups for n in row_counts]
    group_rows = sum(per_group)
    outs = [[] for _ in row_counts]
    for g in range(num_groups):
        base = g * group_rows
        off = 0
        for i, n in enumerate(per_group):
            outs[i].append(shuffled[base + off: base + off + n])
            off += n
    return [torch.cat(o, dim=0) for o in outs]


def pack_qkva(w_q, w_kva):
    """[Q_OUT + KVA_OUT, H]: q rows (head-major, nope then pe), then c, then k_pe_raw."""
    assert w_q.shape[1] == w_kva.shape[1]
    return torch.cat([w_q, w_kva], dim=0)


def split_uk_uv(d: Dims, w_kvb, contiguous=False):
    """kv_b_proj [NH*(D_N+D_V), D_C] -> W_uk [NH, D_N, D_C], W_uv [NH, D_V, D_C].

    Views of w_kvb unless contiguous=True. Per head h: k_nope[h] = W_uk[h] @ c,
    v[h] = W_uv[h] @ c.
    """
    assert w_kvb.shape == (d.KVB_OUT, d.D_C), w_kvb.shape
    v = w_kvb.view(d.NH, d.D_N + d.D_V, d.D_C)
    w_uk, w_uv = v[:, :d.D_N], v[:, d.D_N:]
    if contiguous:
        w_uk, w_uv = w_uk.contiguous(), w_uv.contiguous()
    return w_uk, w_uv


def pack_dense_mlp(d: Dims, w_gate, w_up, w_down):
    """Layer-0 MLP padded from I_DENSE to I_DENSE_PAD and shuffled for gang_linear_silu.

    With tile_n = 64 the kernel reads, inside each XCD's chunk, output tile t as
    gate weight tile grp*4 + sub and up weight tile grp*4 + 2 + sub with
    grp = t // 2, sub = t % 2: 128 gate rows then 128 up rows per group, so the
    shuffle groups are I_DENSE_PAD / 128 rows of each tensor.
    """
    assert w_gate.shape == (d.I_DENSE, d.H) and w_up.shape == (d.I_DENSE, d.H)
    assert w_down.shape == (d.H, d.I_DENSE)
    I_PAD = d.I_DENSE_PAD
    gate_pad = pad_rows(w_gate, I_PAD)
    up_pad = pad_rows(w_up, I_PAD)
    num_groups = d.SILU_GROUPS
    return {
        "gate_pad": gate_pad,
        "up_pad": up_pad,
        "W_gu_shuffled": shuffle_rows([gate_pad, up_pad], num_groups),
        "W_down_pad": pad_cols(w_down, I_PAD),
        "num_groups": num_groups,
    }


def pack_moe(d: Dims, w_gate_router, experts, shared):
    """W13 [E_TOTAL, 2*I_MOE, H] and W2 [E_TOTAL, H, I_MOE] with the shared MLP as
    N_SHARED extra experts (D6): expert E + s holds rows [s*I_MOE, (s+1)*I_MOE) of
    the shared gate/up and the same column range of the shared down.
    """
    assert w_gate_router.shape == (d.E, d.H)
    assert len(experts) == d.E
    sg, su, sd = shared
    assert sg.shape == (d.I_SHARED, d.H) and su.shape == (d.I_SHARED, d.H)
    assert sd.shape == (d.H, d.I_SHARED)
    w13 = torch.empty(d.E_TOTAL, 2 * d.I_MOE, d.H, dtype=sg.dtype, device=sg.device)
    w2 = torch.empty(d.E_TOTAL, d.H, d.I_MOE, dtype=sg.dtype, device=sg.device)
    for e, (g, u, dn) in enumerate(experts):
        assert g.shape == (d.I_MOE, d.H) and u.shape == (d.I_MOE, d.H) and dn.shape == (d.H, d.I_MOE)
        w13[e, :d.I_MOE] = g
        w13[e, d.I_MOE:] = u
        w2[e] = dn
    for s in range(d.N_SHARED):
        e = d.E + s
        rows = slice(s * d.I_MOE, (s + 1) * d.I_MOE)
        w13[e, :d.I_MOE] = sg[rows]
        w13[e, d.I_MOE:] = su[rows]
        w2[e] = sd[:, rows]
    return {"W_gate": w_gate_router, "W13": w13, "W2": w2}


def pack_attention(d: Dims, lw):
    """Attention and norm tensors of one layer from a dict keyed by HF suffixes.

    W_uk and W_uv are contiguous copies: attach_input asserts row-major
    tensors (persistent_kernel.py:473-475), and since kv_b_proj itself is
    never attached the bytes are the same as keeping the views.
    """
    w_uk, w_uv = split_uk_uv(d, lw["self_attn.kv_b_proj.weight"], contiguous=True)
    out = {
        "w_norm1": lw["input_layernorm.weight"],
        "w_norm2": lw["post_attention_layernorm.weight"],
        "W_qkva": pack_qkva(lw["self_attn.q_proj.weight"], lw["self_attn.kv_a_proj_with_mqa.weight"]),
        "w_kv_norm": lw["self_attn.kv_a_layernorm.weight"],
        "W_uk": w_uk,
        "W_uv": w_uv,
        "W_o": lw["self_attn.o_proj.weight"],
    }
    assert out["W_qkva"].shape == (d.Q_OUT + d.KVA_OUT, d.H)
    assert out["W_o"].shape == (d.H, d.H)
    assert out["w_kv_norm"].shape == (d.D_C,)
    return out


# ----------------------------------------------------------------------------
# byte accounting (from dims only)


def packed_bytes(d: Dims, bf16=2):
    per_layer = (
        2 * d.H                                 # two norms
        + (d.Q_OUT + d.KVA_OUT) * d.H           # W_qkva
        + d.D_C                                 # kv_a_layernorm
        + d.KVB_OUT * d.D_C                     # W_uk + W_uv
        + d.H * d.H                             # W_o
    )
    dense = 2 * d.I_DENSE_PAD * d.H + d.H * d.I_DENSE_PAD
    moe = d.E * d.H + d.E_TOTAL * 2 * d.I_MOE * d.H + d.E_TOTAL * d.H * d.I_MOE
    total = 2 * d.V * d.H + d.H + d.L * per_layer + dense + (d.L - 1) * moe
    return total * bf16


# ----------------------------------------------------------------------------
# loader (needs the checkpoint; not runnable locally)


def layer_names(d: Dims, l):
    p = f"model.layers.{l}."
    names = [p + s for s in (
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.kv_a_layernorm.weight", "self_attn.kv_b_proj.weight",
        "self_attn.o_proj.weight")]
    if l == 0:
        names += [p + f"mlp.{s}.weight" for s in ("gate_proj", "up_proj", "down_proj")]
    else:
        names.append(p + "mlp.gate.weight")
        names += [p + f"mlp.shared_experts.{s}.weight" for s in ("gate_proj", "up_proj", "down_proj")]
        for e in range(d.E):
            names += [p + f"mlp.experts.{e}.{s}.weight" for s in ("gate_proj", "up_proj", "down_proj")]
    return names


class Shards:
    def __init__(self, model_dir, device):
        from safetensors import safe_open

        self.dir = Path(model_dir)
        self.map = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.device = device
        self.open = {}
        self._safe_open = safe_open

    def get(self, name):
        shard = self.map[name]
        if shard not in self.open:
            self.open[shard] = self._safe_open(str(self.dir / shard), framework="pt", device=self.device)
        t = self.open[shard].get_tensor(name)
        assert t.dtype == torch.bfloat16, (name, t.dtype)
        return t


def load_checkpoint_layer(model_dir, l, d: Dims, device="cpu", shards=None):
    """The raw tensors of layer l, keyed by their suffix after 'model.layers.l.'."""
    shards = shards or Shards(model_dir, device)
    p = f"model.layers.{l}."
    return {n[len(p):]: shards.get(n) for n in layer_names(d, l)}


def pack_layer(d: Dims, l, lw):
    out = pack_attention(d, lw)
    if l == 0:
        out.update(pack_dense_mlp(d, lw["mlp.gate_proj.weight"], lw["mlp.up_proj.weight"],
                                  lw["mlp.down_proj.weight"]))
        del out["gate_pad"], out["up_pad"], out["num_groups"]
    else:
        experts = [(lw[f"mlp.experts.{e}.gate_proj.weight"], lw[f"mlp.experts.{e}.up_proj.weight"],
                    lw[f"mlp.experts.{e}.down_proj.weight"]) for e in range(d.E)]
        shared = (lw["mlp.shared_experts.gate_proj.weight"], lw["mlp.shared_experts.up_proj.weight"],
                  lw["mlp.shared_experts.down_proj.weight"])
        out.update(pack_moe(d, lw["mlp.gate.weight"], experts, shared))
    return out


def pack_all(model_dir, device="cpu", dims: Dims = None, layers: int = None, head: bool = True):
    """Every tensor of 04-memory-plan.md, keyed by name with the layer suffix `_{l}`.

    layers/head restrict the load to a truncated graph's needs; the byte
    assertion applies to the full load only."""
    model_dir = Path(model_dir)
    if dims is None:
        dims = Dims.from_config(json.loads((model_dir / "config.json").read_text()))
    check_tiling(dims)
    shards = Shards(model_dir, device)
    packed = {"W_embed": shards.get("model.embed_tokens.weight")}
    if head:
        packed["W_lm"] = shards.get("lm_head.weight")
        packed["w_final_norm"] = shards.get("model.norm.weight")
        assert packed["W_lm"].shape == (dims.V, dims.H)
    assert packed["W_embed"].shape == (dims.V, dims.H)
    for l in range(layers if layers is not None else dims.L):
        lw = load_checkpoint_layer(model_dir, l, dims, device, shards)
        for k, v in pack_layer(dims, l, lw).items():
            packed[f"{k}_{l}"] = v
    total = sum(v.numel() * v.element_size() for v in packed.values())
    if dims == REAL_DIMS and head and (layers is None or layers == dims.L):
        assert total == EXPECTED_PACKED_BYTES, (total, EXPECTED_PACKED_BYTES)
    return packed


def dry_run(model_dir):
    """Shapes and dtypes from the index and shard headers only."""
    from safetensors import safe_open

    model_dir = Path(model_dir)
    dims = Dims.from_config(json.loads((model_dir / "config.json").read_text()))
    check_tiling(dims)
    wmap = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    names = ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]
    for l in range(dims.L):
        names += layer_names(dims, l)
    missing = [n for n in names if n not in wmap]
    assert not missing, missing[:5]
    total = 0
    for shard in sorted(set(wmap.values())):
        with safe_open(str(model_dir / shard), framework="pt") as f:
            for n in f.keys():
                sl = f.get_slice(n)
                total += math.prod(sl.get_shape()) * 2
    print(f"{len(names)} tensors named, checkpoint bytes {total}, "
          f"packed bytes expected {packed_bytes(dims)}")
    return dims


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.dry_run:
        dry_run(args.model_dir)
        return
    packed = pack_all(args.model_dir, args.device)
    total = sum(v.numel() * v.element_size() for v in packed.values())
    print(f"packed {len(packed)} tensors, {total} bytes ({total / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
