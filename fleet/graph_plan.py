"""The Fleet task graph of docs/design-doc/02-task-graph.md as data.

A plan is a list of operator calls (method name, tensor names, parameters)
plus the tensors to allocate or attach, derived from the model dims and the
truncation options (--layers N, --head, --debug). It contains no Fleet
import, so its arithmetic (tiles, strides, the asserts of the reused
kernels' Python wrappers and the silent constraints the design lists) is
checked here; build_graph.py turns it into mpk.* calls on the machine.

Counts are cross-checked in the tests against docs/design-doc/sources/
graph_counts.py: 326 operators and 1,880 tasks for the full graph.
"""
import struct
from dataclasses import dataclass, field

from fleet.pack_weights import Dims, REAL_DIMS, XCDS, TILE_N_QKVA, TILE_N_LM

TILE_N_O = 32          # o_proj and down_proj (tile_n 32: 8 tiles per XCD, 02-task-graph.md)
TILE_N_SILU = 64
SPLIT = 32             # positions per attention split (D11)
ARGMAX_SLICES = 50     # D13
N_FORCED = 2           # D6: experts 64, 65 at weight 1.0
TOPK_TOTAL_SLOTS = 8
SOFTMAX_SCALE = 0.1147213867929261
ROUTED_SCALING = 1.0


def float_bits(x: float) -> int:
    """A float parameter for register_task, which takes ints: its IEEE-754 bit pattern."""
    return struct.unpack("<i", struct.pack("<f", x))[0]


@dataclass
class Tensor:
    name: str
    shape: tuple
    dtype: str = "bf16"           # bf16 | f32 | i32 | i64
    kind: str = "new"             # new (mpk.new_tensor) | input (attach_input, from the loader)
    source: str = ""              # for inputs: the packed-weights key or the capture name

    @property
    def num_dims(self):
        return len(self.shape)

    def dim(self, i):
        return self.shape[i]


@dataclass
class Call:
    method: str                   # the mpk.* method (or a new one in build_graph.py)
    args: dict                    # tensor names and scalars, in the method's keyword names
    tasks: int                    # tasks this operator creates (for the counts)
    tiles: int = 0                # tiles per gang task (0 for non-gang)
    status: str = "reuse"         # reuse | variant | new
    note: str = ""
    label: str = ""               # "L{l}.<op>", "head.<op>", "prologue.embed": for --stop-after and dumps


# Which argument names of each plan method are outputs (the rest are inputs). The
# runtime accepts a graph only if every operator reads at least one tensor the
# previous operator wrote (src/kernel/runtime.cc, register_mugraph: the consumer's
# inputs against the producer's outputs, assert(num_shared_tensors >= 1)).
OUTPUT_ARGS = {
    "embed_layer": ["output"], "rmsnorm_layer": ["output"], "gang_linear_layer": ["output"],
    "gang_linear_with_residual_layer": ["output"], "gang_linear_silu_layer": ["output"],
    "linear_layer": ["output"], "linear_with_residual_layer": ["output"],
    "mla_prep_layer": ["c_kv", "k_pe", "ql_nope", "q_pe"], "mla_attend_layer": ["partials", "scores"],
    "mla_merge_uv_layer": ["output"], "moe_router_layer": ["topk_w", "routing", "mask", "logits", "route_log"],
    "gang_moe_w13_linear_layer": ["output"], "moe_silu_mul_layer": ["output"],
    "gang_moe_w2_linear_layer": ["output"], "moe_mul_sum_add_layer": ["output"],
    "argmax_partial_layer": ["output"], "argmax_reduce_layer": ["output"], "copy_layer": ["output"],
}


def _tensor_args(call, names):
    out = set()
    for k in names:
        v = call.args.get(k)
        if v is None:
            continue
        out.update(v if isinstance(v, (tuple, list)) else [v])
    return out


@dataclass
class Plan:
    dims: Dims
    s_max: int
    layers: int
    head: bool
    debug: bool
    tensors: dict = field(default_factory=dict)
    calls: list = field(default_factory=list)

    def t(self, name, shape, dtype="bf16", kind="new", source=""):
        assert name not in self.tensors, name
        self.tensors[name] = Tensor(name, tuple(shape), dtype, kind, source)
        return name

    def op(self, method, tasks, tiles=0, status="reuse", note="", label="", **args):
        self.calls.append(Call(method, args, tasks, tiles, status, note, label))

    def index_of(self, label):
        for i, c in enumerate(self.calls):
            if c.label == label:
                return i
        raise KeyError(label)

    def truncate(self, stop_label):
        """Keep the calls up to and including the labelled operator (07-correctness.md, M1 protocol)."""
        self.calls = self.calls[: self.index_of(stop_label) + 1]
        return self

    def chain_violations(self):
        """Consecutive operators that share no tensor from producer outputs to consumer inputs:
        the runtime rejects such a graph at registration. Empty for a valid plan."""
        bad = []
        for prev, cur in zip(self.calls, self.calls[1:]):
            outs = _tensor_args(prev, OUTPUT_ARGS[prev.method])
            ins = _tensor_args(cur, [k for k in cur.args if k not in OUTPUT_ARGS[cur.method]])
            if not (outs & ins):
                bad.append((prev.label, cur.label))
        return bad

    @property
    def n_ops(self):
        return len(self.calls)

    @property
    def n_tasks(self):
        return sum(c.tasks for c in self.calls)

    @property
    def n_splits(self):
        return -(-self.s_max // SPLIT)


def partials_row(d_c):
    """Innermost width of the split-KV partials buffer: D_C values of o plus one lse at
    column D_C, padded up to a multiple of 4 floats so each [split, head] row is 16-byte
    aligned when the buffer base is (docs/round-2 P2). 513 -> 516 at D_C = 512."""
    return ((d_c + 1 + 3) // 4) * 4


def gang_tiles(n_out, tile_n):
    """The reused gang linears' Python asserts (persistent_kernel.py:1591-1595)."""
    assert n_out % XCDS == 0, f"N {n_out} must be divisible by 8"
    chunk = n_out // XCDS
    assert chunk % tile_n == 0, f"chunk {chunk} must be divisible by tile_n {tile_n}"
    return chunk // tile_n


def grid_for_linear(size):
    """Tasks for the stock non-gang linear (persistent_kernel.py linear_layer), which splits
    the output columns across grid_dim[0] tasks over all 296 workers instead of the gang path's
    8 (one per XCD). The heuristic is the demo's tested one (demo/qwen3/demo_30B_A3B.py,
    grid_for_rmsnorm_linear_layer), including its size//256 workaround for a large lm_head."""
    if size / 96 > 400:
        assert size % 256 == 0, f"per-tile linear size {size} must be a multiple of 256"
        return size // 256
    if size % 96 == 0:
        return 96
    assert size % 64 == 0, f"per-tile linear size {size} must be a multiple of 96 or 64"
    return 64


def build_plan(dims: Dims = REAL_DIMS, s_max: int = 1056, layers: int = 27, head: bool = True,
               debug: bool = False, debug_scores: bool = False, tile_linears: bool = False) -> Plan:
    """debug_scores: the mla_attend kernel also writes the scaled pre-softmax scores
    [NH, s_max] FP32 (boundary B5); needs the MLA_ATTEND_DEBUG_SCORES build (MPK_DEBUG_SCORES=1).
    tile_linears: issue the four dense linears (qkva, o_proj, down, lm_head) as per-tile
    linear_layer tasks over all workers instead of 8-task gangs (MAJ-7; docs/round-2 P5). The
    silu-fused gate_up and the MoE linears stay gang (no drop-in non-gang equivalent)."""
    d = dims
    assert 1 <= layers <= d.L
    assert d.H % 256 == 0                   # K of every CK linear (silent truncation otherwise)
    assert d.D_C % 256 == 0                 # K of mla_merge_uv's W_uv product
    assert d.I_DENSE_PAD % 256 == 0
    assert d.I_MOE % 128 == 0 and (2 * d.I_MOE) % 64 == 0
    assert d.V % ARGMAX_SLICES == 0         # argmax_partial: input.dim(1) // num_tasks, no assert in the API
    p = Plan(d, s_max, layers, head, debug)
    n_splits = p.n_splits
    splits_per_xcd = -(-n_splits // XCDS)

    # ---- meta and shared tensors ------------------------------------------------
    p.t("tok_dummy", (1, 1), "i64", "input", "meta:input_tokens")      # satisfies embed's input arg
    p.t("W_embed", (d.V, d.H), kind="input", source="W_embed")
    p.t("x_res", (1, d.H))
    p.t("h", (1, d.H))
    p.t("qkva", (1, d.Q_OUT + d.KVA_OUT))
    p.t("ql_nope", (d.NH, d.D_C))
    p.t("q_pe", (d.NH, d.D_R))
    p.t("partials", (n_splits, d.NH, partials_row(d.D_C)), "f32")   # padded row, P2
    p.t("attn", (1, d.NH * d.D_V))
    p.t("cos", (s_max, d.D_R), kind="input", source="capture:cos")
    p.t("sin", (s_max, d.D_R), kind="input", source="capture:sin")
    if layers >= 1:
        p.t("act", (1, d.I_DENSE_PAD))
    if layers >= 2:
        p.t("topk_w", (1, TOPK_TOTAL_SLOTS), "f32")
        p.t("routing", (d.E_TOTAL, 1), "i32")
        p.t("mask", (d.E_TOTAL + 1,), "i32")
        p.t("logits_router", (1, d.E), "f32")
        p.t("mid", (1, TOPK_TOTAL_SLOTS, 2 * d.I_MOE))
        p.t("act8", (1, TOPK_TOTAL_SLOTS, d.I_MOE))
        p.t("out8", (1, TOPK_TOTAL_SLOTS, d.H))
        p.t("route_log", (32, d.L - 1, TOPK_TOTAL_SLOTS), "i32")
    if debug:
        for l in range(layers):
            p.t(f"dbg_x_res_{l}", (1, d.H))
    if debug_scores:
        p.t("scores", (d.NH, s_max), "f32")

    # ---- prologue ----------------------------------------------------------------
    p.op("embed_layer", 1, status="variant", note="sc1 load of tokens[step]", label="prologue.embed",
         input="tok_dummy", weight="W_embed", output="x_res", grid_dim=(1, 1, 1),
         block_dim=(256, 1, 1), input_source=0)

    # ---- layers ------------------------------------------------------------------
    for l in range(layers):
        p.t(f"w_norm1_{l}", (d.H,), kind="input", source=f"w_norm1_{l}")
        p.t(f"w_norm2_{l}", (d.H,), kind="input", source=f"w_norm2_{l}")
        p.t(f"W_qkva_{l}", (d.Q_OUT + d.KVA_OUT, d.H), kind="input", source=f"W_qkva_{l}")
        p.t(f"w_kv_norm_{l}", (d.D_C,), kind="input", source=f"w_kv_norm_{l}")
        p.t(f"W_uk_{l}", (d.NH, d.D_N, d.D_C), kind="input", source=f"W_uk_{l}")
        p.t(f"W_uv_{l}", (d.NH, d.D_V, d.D_C), kind="input", source=f"W_uv_{l}")
        p.t(f"W_o_{l}", (d.H, d.H), kind="input", source=f"W_o_{l}")
        p.t(f"c_kv_{l}", (s_max, d.D_C), kind="input", source=f"capture:c_kv_{l}")
        p.t(f"k_pe_{l}", (s_max, d.D_R), kind="input", source=f"capture:k_pe_{l}")

        # with the debug snapshots, the operator after a snapshot must read the copy (same values):
        # the runtime's chain rule, see OUTPUT_ARGS; the residual adds still read x_res
        x_in = f"dbg_x_res_{l - 1}" if debug and l > 0 else "x_res"
        p.op("rmsnorm_layer", 1, label=f"L{l}.norm1", input=x_in, weight=f"w_norm1_{l}", output="h",
             grid_dim=(1, 1, 1), block_dim=(256, 1, 1))
        if tile_linears:
            g = grid_for_linear(d.Q_OUT + d.KVA_OUT)
            p.op("linear_layer", g, label=f"L{l}.qkva", input="h", weight=f"W_qkva_{l}",
                 output="qkva", grid_dim=(g, 1, 1), block_dim=(256, 1, 1))
        else:
            p.op("gang_linear_layer", XCDS, gang_tiles(d.Q_OUT + d.KVA_OUT, TILE_N_QKVA), label=f"L{l}.qkva",
                 input="h", weight=f"W_qkva_{l}", output="qkva", tile_n=TILE_N_QKVA,
                 output_stride=d.Q_OUT + d.KVA_OUT)
        p.op("mla_prep_layer", 1, status="new", label=f"L{l}.mla_prep",
             qkva="qkva", w_kv_norm=f"w_kv_norm_{l}", w_uk=f"W_uk_{l}", cos="cos", sin="sin",
             c_kv=f"c_kv_{l}", k_pe=f"k_pe_{l}", ql_nope="ql_nope", q_pe="q_pe",
             block_dim=(256, 1, 1))
        p.op("mla_attend_layer", XCDS, splits_per_xcd, status="new", label=f"L{l}.mla_attend",
             ql_nope="ql_nope", q_pe="q_pe", c_kv=f"c_kv_{l}", k_pe=f"k_pe_{l}",
             partials="partials", softmax_scale=SOFTMAX_SCALE, split=SPLIT, n_splits=n_splits,
             **({"scores": "scores"} if debug_scores else {}))
        p.op("mla_merge_uv_layer", XCDS, d.NH // XCDS, status="new", label=f"L{l}.mla_merge_uv",
             partials="partials", w_uv=f"W_uv_{l}", output="attn", split=SPLIT, n_splits=n_splits)
        if tile_linears:
            g = grid_for_linear(d.H)
            p.op("linear_with_residual_layer", g, label=f"L{l}.o_proj", input="attn", weight=f"W_o_{l}",
                 residual="x_res", output="x_res", grid_dim=(g, 1, 1), block_dim=(256, 1, 1))
        else:
            p.op("gang_linear_with_residual_layer", XCDS, gang_tiles(d.H, TILE_N_O), label=f"L{l}.o_proj",
                 input="attn", weight=f"W_o_{l}", residual="x_res", output="x_res",
                 tile_n=TILE_N_O, output_stride=d.H)
        p.op("rmsnorm_layer", 1, label=f"L{l}.norm2", input="x_res", weight=f"w_norm2_{l}", output="h",
             grid_dim=(1, 1, 1), block_dim=(256, 1, 1))
        if l == 0:
            p.t("W_gu_shuffled", (2 * d.I_DENSE_PAD, d.H), kind="input", source="W_gu_shuffled_0")
            p.t("W_down_pad", (d.H, d.I_DENSE_PAD), kind="input", source="W_down_pad_0")
            n_weight_tiles = (2 * d.I_DENSE_PAD) // XCDS // TILE_N_SILU
            assert n_weight_tiles % 2 == 0, "gang_linear_silu: weight tiles per XCD must be even (silent // 2)"
            p.op("gang_linear_silu_layer", XCDS, n_weight_tiles // 2, label=f"L{l}.gate_up",
                 input="h", weight="W_gu_shuffled", output="act", tile_n=TILE_N_SILU,
                 output_stride=d.I_DENSE_PAD)
            if tile_linears:
                g = grid_for_linear(d.H)
                p.op("linear_with_residual_layer", g, label=f"L{l}.down", input="act", weight="W_down_pad",
                     residual="x_res", output="x_res", grid_dim=(g, 1, 1), block_dim=(256, 1, 1))
            else:
                p.op("gang_linear_with_residual_layer", XCDS, gang_tiles(d.H, TILE_N_O), label=f"L{l}.down",
                     input="act", weight="W_down_pad", residual="x_res", output="x_res",
                     tile_n=TILE_N_O, output_stride=d.H)
        else:
            p.t(f"W_gate_{l}", (d.E, d.H), kind="input", source=f"W_gate_{l}")
            p.t(f"W13_{l}", (d.E_TOTAL, 2 * d.I_MOE, d.H), kind="input", source=f"W13_{l}")
            p.t(f"W2_{l}", (d.E_TOTAL, d.H, d.I_MOE), kind="input", source=f"W2_{l}")
            p.op("moe_router_layer", 1, status="new", label=f"L{l}.router",
                 input="h", w_gate=f"W_gate_{l}", topk_w="topk_w", routing="routing", mask="mask",
                 logits="logits_router", route_log="route_log", layer_index=l - 1,
                 topk=d.TOPK, n_experts=d.E, n_forced=N_FORCED, scaling=ROUTED_SCALING,
                 block_dim=(256, 1, 1))
            p.op("gang_moe_w13_linear_layer", XCDS, (2 * d.I_MOE) // 64, label=f"L{l}.w13",
                 input="h", weight=f"W13_{l}", moe_routing_indices="routing", moe_mask="mask",
                 output="mid")
            p.op("moe_silu_mul_layer", TOPK_TOTAL_SLOTS, label=f"L{l}.silu", input="mid", output="act8",
                 grid_dim=(1, TOPK_TOTAL_SLOTS, 1), block_dim=(256, 1, 1))
            p.op("gang_moe_w2_linear_layer", XCDS, d.H // 64, label=f"L{l}.w2",
                 input="act8", weight=f"W2_{l}", moe_routing_indices="routing", moe_mask="mask",
                 output="out8")
            p.op("moe_mul_sum_add_layer", d.H // 256, label=f"L{l}.combine", input="out8", weight="topk_w",
                 residual="x_res", output="x_res", grid_dim=(1, d.H // 256, 1), block_dim=(256, 1, 1))
        if debug:
            p.op("copy_layer", 1, status="new", note="debug only: per-layer residual snapshot", label=f"L{l}.snapshot",
                 input="x_res", output=f"dbg_x_res_{l}", grid_dim=(1, 1, 1), block_dim=(256, 1, 1))

    # ---- head ----------------------------------------------------------------------
    if head:
        p.t("w_final_norm", (d.H,), kind="input", source="w_final_norm")
        p.t("W_lm", (d.V, d.H), kind="input", source="W_lm")
        p.t("logits", (1, d.V))
        p.t("amax_v", (1, ARGMAX_SLICES))
        p.t("amax_i", (1, ARGMAX_SLICES), "i64")
        p.t("tok_out", (1, 1), "i64", "input", "meta:output_tokens")
        p.op("rmsnorm_layer", 1, label="head.norm", input=f"dbg_x_res_{layers - 1}" if debug else "x_res",
             weight="w_final_norm", output="h", grid_dim=(1, 1, 1), block_dim=(256, 1, 1))
        if tile_linears:
            g = grid_for_linear(d.V)
            p.op("linear_layer", g, label="head.lm_head", input="h", weight="W_lm",
                 output="logits", grid_dim=(g, 1, 1), block_dim=(256, 1, 1))
        else:
            p.op("gang_linear_layer", XCDS, gang_tiles(d.V, TILE_N_LM), label="head.lm_head",
                 input="h", weight="W_lm", output="logits", tile_n=TILE_N_LM, output_stride=d.V)
        p.op("argmax_partial_layer", ARGMAX_SLICES, label="head.argmax_partial", input="logits", output=("amax_v", "amax_i"),
             grid_dim=(ARGMAX_SLICES, 1, 1), block_dim=(256, 1, 1))
        p.op("argmax_reduce_layer", 1, status="variant", note="writes tokens[step + 1]", label="head.argmax_reduce",
             input=("amax_v", "amax_i"), output="tok_out", grid_dim=(1, 1, 1),
             block_dim=(256, 1, 1), output_to_tokens=True)
    return p


def summary(p: Plan) -> dict:
    by_status = {}
    for c in p.calls:
        by_status[c.status] = by_status.get(c.status, 0) + 1
    input_bytes = sum(
        t.shape_bytes if hasattr(t, "shape_bytes") else _bytes(t) for t in p.tensors.values() if t.kind == "input")
    return {"ops": p.n_ops, "tasks": p.n_tasks, "by_status": by_status,
            "tensors_new": sum(t.kind == "new" for t in p.tensors.values()),
            "tensors_input": sum(t.kind == "input" for t in p.tensors.values()),
            "input_bytes": input_bytes, "n_splits": p.n_splits}


def _bytes(t: Tensor) -> int:
    size = {"bf16": 2, "f32": 4, "i32": 4, "i64": 8}[t.dtype]
    n = 1
    for s in t.shape:
        n *= s
    return n * size
