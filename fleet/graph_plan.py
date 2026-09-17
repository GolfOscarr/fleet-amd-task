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
RMS_EPS = 1e-6                 # rms_norm_eps of config.json; the fused router's norm (O1)
PREFETCH_PARTS = 32            # O8: tasks per active expert of the W2 prefetch (2048 rows / 32 = 64 rows, 180 KB each)
NUM_WORKERS = 296              # build_graph.build's default; it is not passed to build_plan, so the
                               # tile count of L4 is checked against the constant here
W13_GEMV_TILES = NUM_WORKERS // XCDS   # 37: one w13 tile per worker of an XCD (S1 of 01-gemv-ideas.md)
OPROJ_HALVES = 2               # N5: the merge with o_proj folded in is always one task per half head
                               # (32 tasks, 128 bytes of W_o per row per task); --merge-halves does not apply


def w13_tile_rows(tile: int, n: int = 2 * REAL_DIMS.I_MOE, tiles: int = W13_GEMV_TILES):
    """(first row, row count) of one tile of the w13 GEMV gang kernel (L4, S1): the 37 tiles
    partition the expert's n rows as 4 of ceil(n / tiles) then the rest of floor(n / tiles),
    which at n = 2,816 is 4 x 77 + 33 x 76. The kernel computes the same two numbers from
    tile_idx by arithmetic (gang_moe_w13_gemv_mi300.cuh), and the suite's reference reuses
    this function so both sides read one formula."""
    assert 0 <= tile < tiles, (tile, tiles)
    small, big_tiles = n // tiles, n % tiles
    big = small + 1
    if tile < big_tiles:
        return big * tile, big
    return big_tiles * big + small * (tile - big_tiles), small


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
    side: bool = False            # a side operator (O8): registered right after the operator it accompanies;
                                  # the runtime patch gives its tasks that operator's dependent event and keeps
                                  # the chain on that operator, so it is skipped by the chain rule


# Which argument names of each plan method are outputs (the rest are inputs). The
# runtime accepts a graph only if every operator reads at least one tensor the
# previous operator wrote (src/kernel/runtime.cc, register_mugraph: the consumer's
# inputs against the producer's outputs, assert(num_shared_tensors >= 1)).
OUTPUT_ARGS = {
    "embed_layer": ["output"], "rmsnorm_layer": ["output"], "gang_linear_layer": ["output"],
    "gang_linear_with_residual_layer": ["output"], "gang_linear_silu_layer": ["output"],
    "linear_layer": ["output"], "linear_with_residual_layer": ["output"],
    "linear_norm_layer": ["output", "scratch"],
    "linear_gemv_layer": ["output"],
    "prefetch_layer": ["dummy"], "prefetch_moe_layer": ["dummy"],
    "mla_prep_layer": ["c_kv", "k_pe", "ql_nope", "q_pe"], "mla_attend_layer": ["partials", "scores"],
    "mla_merge_uv_layer": ["output"], "mla_merge_uv_tile_layer": ["output"],
    "mla_merge_oproj_layer": ["output", "attn", "workspace"],
    "moe_router_layer": ["h", "topk_w", "routing", "mask", "logits", "route_log"],
    "moe_router_norm4_layer": ["h", "topk_w", "routing", "mask", "logits", "route_log"],
    "gang_moe_w13_linear_layer": ["output"], "gang_moe_w13_gemv_layer": ["output"],
    "moe_silu_mul_layer": ["output"],
    "gang_moe_w2_linear_layer": ["output"], "gang_moe_w2_silu_linear_layer": ["output", "scratch"],
    "moe_mul_sum_add_layer": ["output"],
    "argmax_partial_layer": ["output"], "argmax_reduce_layer": ["output"], "copy_layer": ["output"],
    "stream_layer": ["dummy"], "stream_gang_layer": ["dummy"],
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

    def op(self, method, tasks, tiles=0, status="reuse", note="", label="", side=False, **args):
        self.calls.append(Call(method, args, tasks, tiles, status, note, label, side))

    def chain(self):
        """The operators the runtime chains: every call but the side operators (O8)."""
        return [c for c in self.calls if not c.side]

    def index_of(self, label):
        for i, c in enumerate(self.calls):
            if c.label == label:
                return i
        raise KeyError(label)

    def truncate(self, stop_label):
        """Keep the calls up to and including the labelled operator (07-correctness.md, M1 protocol)."""
        self.calls = self.calls[: self.index_of(stop_label) + 1]
        return self

    def insert_probe(self, label):
        """O5 (docs/gpu-experiments/03-acceleration/03-local-preparation.md): a one-task copy
        operator in front of the labelled operator, so that operator's event gap is measured
        with a one-task predecessor. The copy takes the [1, N] BF16 tensor the operator shares
        with its predecessor (the chain's tensor) into a twin `<name>_probe`, and the operator
        reads the twin; every other argument is unchanged. The label is `<prefix>.probe_<op>`.
        Only a single-row BF16 tensor can be probed (the copy task's contract)."""
        i = self.index_of(label)
        assert i > 0, f"{label} has no predecessor to probe"
        cur = self.calls[i]
        assert not cur.side, f"{label} is a side operator"
        j = i - 1
        while j > 0 and self.calls[j].side:          # the predecessor in the chain, past its side operators
            j -= 1
        prev = self.calls[j]
        outs = _tensor_args(prev, OUTPUT_ARGS[prev.method])
        ins = _tensor_args(cur, [k for k in cur.args if k not in OUTPUT_ARGS[cur.method]])
        shared = [n for n in ins & outs
                  if len(self.tensors[n].shape) == 2 and self.tensors[n].shape[0] == 1
                  and self.tensors[n].dtype == "bf16"]
        assert shared, f"{label}: no [1, N] BF16 tensor shared with {prev.label} to probe ({sorted(ins & outs)})"
        name = sorted(shared)[0]
        twin = self.t(name + "_probe", self.tensors[name].shape)
        prefix, op = label.split(".", 1)
        probe = Call("copy_layer", dict(input=name, output=twin, grid_dim=(1, 1, 1), block_dim=(256, 1, 1)),
                     1, 0, "new", f"probe before {label} (O5)", f"{prefix}.probe_{op}")
        def rewire(k, v):
            if k in OUTPUT_ARGS[cur.method]:
                return v
            if isinstance(v, (tuple, list)):          # a pair of inputs (argmax_reduce's values and indices)
                return type(v)(twin if x == name else x for x in v)
            return twin if v == name else v
        args = {k: rewire(k, v) for k, v in cur.args.items()}
        self.calls[i] = Call(cur.method, args, cur.tasks, cur.tiles, cur.status, cur.note, cur.label)
        self.calls.insert(i, probe)
        return self

    def chain_violations(self):
        """Consecutive operators that share no tensor from producer outputs to consumer inputs:
        the runtime rejects such a graph at registration. Empty for a valid plan."""
        bad = []
        chain = self.chain()
        for prev, cur in zip(chain, chain[1:]):
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
    aligned when the buffer base is (docs/gpu-experiments/02-validation P2). 513 -> 516 at D_C = 512."""
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


def linear_grid_for(size, override=None, strict=True):
    """The task count of a dense linear: the override of --linear-grid or --head-grid when one is
    given, else the heuristic of grid_for_linear. The runtime hands each task size / grid rows of
    the weight and the same columns of the output, so an override must divide the row count: the
    page's values are 3,648 by 96, 48 or 32, 2,048 by 64 or 32, and 102,400 by 400 or 320 (L2 and
    L5 of docs/gpu-experiments/04-kernels/05-local-preparation.md). strict is the single-operator
    case (--head-grid): the override must divide. --linear-grid names two operators of different
    row counts, so a value that divides one and not the other (48: 3,648 but not 2,048) leaves
    the other at the heuristic instead of failing the plan."""
    if override is None:
        return grid_for_linear(size)
    assert override >= 1, override
    if size % override:
        assert not strict, (size, override)
        return grid_for_linear(size)
    return override


def build_plan(dims: Dims = REAL_DIMS, s_max: int = 1056, layers: int = 27, head: bool = True,
               debug: bool = False, debug_scores: bool = False, tile_linears: bool = False,
               attend_tasks: bool = False, fuse_norm2: bool = False, fuse_silu: bool = False,
               fuse_norm1: bool = False, prefetch: bool = False, gemv_linears: bool = False,
               linear_grid: int = None, head_grid: int = None, gemv_w13: bool = False, merge_tasks: bool = False,
               merge_halves: int = 1, router_tasks: bool = False, merge_oproj: bool = False) -> Plan:
    """debug_scores: the mla_attend kernel also writes the scaled pre-softmax scores
    [NH, s_max] FP32 (boundary B5); needs the MLA_ATTEND_DEBUG_SCORES build (MPK_DEBUG_SCORES=1).
    tile_linears: issue the four dense linears (qkva, o_proj, down, lm_head) as per-tile
    linear_layer tasks over all workers instead of 8-task gangs (MAJ-7; docs/gpu-experiments/02-validation P5). The
    silu-fused gate_up and the MoE linears stay gang (no drop-in non-gang equivalent).
    fuse_norm2: in the MoE layers the post-attention norm is folded into the router
    (docs/gpu-experiments/03-acceleration, O1): no L{l}.norm2 operator, the router reads x_res
    and the norm weight and writes h for the expert gate-up (kernel NORM = true). Layer 0
    keeps its norm (its consumer is the stock silu gang kernel).
    fuse_silu: the silu-mul folded into the expert down projection's prologue (O2): no
    L{l}.silu operator, w2 reads mid and computes its slot's activation row into a per-tile
    scratch (w2_scratch [8 x tiles per XCD, I_MOE]) before the unchanged CK GEMM.
    fuse_norm1: the input norm folded into the per-tile Q/KV projection and the final norm
    into lm_head (O3): no L{l}.norm1 and no head.norm; the two linears are our linear_norm_mi300,
    the stock per-tile linear with a prologue that normalises the row into a per-task scratch
    row (qkva_scratch [96, H], lm_scratch [400, H]), so they are per-tile whether or not
    tile_linears is set. Off under debug (the stock norms stay, and the snapshot wiring).
    gemv_linears: the four dense linears at batch 1 (qkva, o_proj, layer 0's down, lm_head) as our
    GEMV kernel instead of the CK tile (L2 of docs/gpu-experiments/04-kernels, kernel L1): one
    linear_gemv_mi300 task type for all four, with the input norm (qkva, lm_head) and the residual
    add (o_proj, down) as template flags, so there is no L{l}.norm1 and no head.norm, as with
    fuse_norm1, and no scratch tensor at all (the normalised row stays in LDS). The grids are
    fuse_norm1 + tile_linears', so the operator and task counts are theirs. Off under debug (the
    snapshot wiring needs the stock norms, as fuse_norm1 does). Layer 0's down is the one call
    site whose K is not H: 11,264, so a row is 22 sixteen-byte loads per lane instead of 4 and a
    batch of 8 rows is 704 VGPRs of raw words; -DGEMV_BATCH=2 is the knob if the VM's A/B finds
    it spilling (the kernel's body is a call, so the worker union does not carry those registers).
    router_tasks: the MoE layers' router as four regular tasks of 16 experts each, the last to
    arrive reading the 64 logits back and routing (N2 of docs/gpu-experiments/04-kernels, R5). It
    is the fused router's split form, so for those layers it carries fuse_norm2's wiring (no
    L{l}.norm2 operator; the router reads x_res and the norm weight and writes h) and it adds one
    tensor, router_counter [1] int32, shared by every layer and zeroed at allocation. Three more
    tasks per MoE layer; off under debug, as fuse_norm1 and gemv_linears are.
    merge_oproj: the merge with o_proj folded into it (N5 of docs/gpu-experiments/04-kernels, M5):
    for every layer the merge and the o_proj operators become one operator of NH * OPROJ_HALVES
    (32) regular tasks labelled L{l}.o_proj, so --stop-after and the compare's x_res boundary keep
    their key and the attn boundary's last writer is the same operator. Each task merges its head,
    stores its 64 attn values and multiplies them by its 128-byte slice of every row of W_o into a
    partial vector; the last task to arrive sums the 32 partials, adds the residual and stores
    x_res in place. It adds two tensors, oproj_ws [32, H] FP32 and oproj_counter [1] int32, both
    shared by every layer and zeroed at allocation. One operator fewer per layer; the tasks drop by
    64 + 8 - 32 where o_proj is the per-tile or GEMV form (--tile-linears, --gemv-linears) and rise
    by 32 - 8 - 8 where it is the 8-task gang. It overrides --merge-tasks (its merge is the tile
    form at two halves), and it needs no other flag.
    merge_tasks, merge_halves: the merge as NH * merge_halves regular tasks with whole-tensor
    imaps instead of the 8-task gang (N4 of docs/gpu-experiments/04-kernels, M6 and M4): the
    head and the half come from the task index, and with merge_halves 2 each task multiplies
    half of its head's W_uv rows after merging the whole head. The operator keeps its label and
    its tensors, so only the task count moves (16 or 32 per layer against 8).
    linear_grid, head_grid: the task count of qkva and o_proj, and of lm_head, under gemv_linears
    (--linear-grid, --head-grid; L2 and L5): grid_for_linear's heuristic otherwise, and also where
    linear_grid does not divide the operator's row count (linear_grid_for). Layer 0's down keeps
    the heuristic (it is one operator, and the page's override names qkva and o_proj).
    gemv_w13: every MoE layer's expert gate-up as our GEMV gang task instead of the stock CK
    one (L4 of docs/gpu-experiments/04-kernels): the same operator with the same label, the same
    tensors and the same 8 tasks, but 37 tiles per expert (one per worker of an XCD, S1) instead
    of 2,816 / 64 = 44, so the operator ends in one round per XCD. Independent of gemv_linears
    and of --debug: it changes no norm and allocates no tensor.
    prefetch: the side operators of O8 (idea D1): each registered right after the operator it
    accompanies and given, by the runtime patch, that operator's dependent event, so its tasks
    run on the workers that hold none of that operator's tasks. Three per layer: after qkva
    the layer's W_o (64 tasks), after o_proj the next layer's W_qkva (96 tasks), after w13 the
    active experts' W2 (8 slots x PREFETCH_PARTS tasks, the ids from mask). Each task streams
    its slice with ordinary loads into a dummy [grid, 4] int32 output."""
    d = dims
    assert 1 <= layers <= d.L
    assert d.H % 256 == 0                   # K of every CK linear (silent truncation otherwise)
    assert d.D_C % 256 == 0                 # K of mla_merge_uv's W_uv product
    assert d.I_DENSE_PAD % 256 == 0
    assert d.I_MOE % 128 == 0 and (2 * d.I_MOE) % 64 == 0
    assert d.V % ARGMAX_SLICES == 0         # argmax_partial: input.dim(1) // num_tasks, no assert in the API
    p = Plan(d, s_max, layers, head, debug)
    n_splits = p.n_splits
    gemv = gemv_linears and not debug
    fuse1 = fuse_norm1 and not debug               # the GEMV form folds the norm in too, below
    if gemv_w13:
        # the gang loop hands tile t to the worker of rank t mod (workers per XCD), so one round
        # per XCD needs exactly as many tiles as that XCD has workers. num_workers is an argument
        # of build_graph.build and never reaches the plan, so the constant is what is checked.
        assert NUM_WORKERS // XCDS == W13_GEMV_TILES == 37, (NUM_WORKERS, W13_GEMV_TILES)
        assert sum(w13_tile_rows(t, 2 * d.I_MOE)[1] for t in range(W13_GEMV_TILES)) == 2 * d.I_MOE
    router4 = router_tasks and not debug
    assert merge_halves in (1, 2), merge_halves
    assert merge_halves == 1 or merge_tasks, "--merge-halves applies to --merge-tasks"
    assert linear_grid is None or gemv_linears, "--linear-grid applies to --gemv-linears"
    assert head_grid is None or gemv_linears, "--head-grid applies to --gemv-linears"
    assert linear_grid is None or (d.Q_OUT + d.KVA_OUT) % linear_grid == 0 or d.H % linear_grid == 0, \
        f"--linear-grid {linear_grid} divides neither {d.Q_OUT + d.KVA_OUT} nor {d.H}"
    if prefetch:
        p.t("pf_dummy_o", (grid_for_linear(d.H), 4), "i32")
        p.t("pf_dummy_qkva", (grid_for_linear(d.Q_OUT + d.KVA_OUT), 4), "i32")
        if layers >= 2:
            p.t("pf_dummy_w2", (TOPK_TOTAL_SLOTS * PREFETCH_PARTS, 4), "i32")
    splits_per_xcd = -(-n_splits // XCDS)

    # ---- meta and shared tensors ------------------------------------------------
    p.t("tok_dummy", (1, 1), "i64", "input", "meta:input_tokens")      # satisfies embed's input arg
    p.t("W_embed", (d.V, d.H), kind="input", source="W_embed")
    p.t("x_res", (1, d.H))
    p.t("h", (1, d.H))
    p.t("qkva", (1, d.Q_OUT + d.KVA_OUT))
    if fuse1 and not gemv:
        # one normalised row per task of the fused per-tile linear (the runtime offsets the
        # scratch pointer by rows / grid x bid.x, so the row count is the grid); the GEMV
        # folds the same norm in but keeps the row in LDS, so it allocates none
        p.t("qkva_scratch", (grid_for_linear(d.Q_OUT + d.KVA_OUT), d.H))
    p.t("ql_nope", (d.NH, d.D_C))
    p.t("q_pe", (d.NH, d.D_R))
    p.t("partials", (n_splits, d.NH, partials_row(d.D_C)), "f32")   # padded row, P2
    p.t("attn", (1, d.NH * d.D_V))
    if merge_oproj:
        # N5: the 32 tasks' partial vectors and their arrival counter, one pair for all the layers
        # (the chain serialises them, and the last task of each resets the counter to zero)
        p.t("oproj_ws", (d.NH * OPROJ_HALVES, d.H), "f32")
        p.t("oproj_counter", (1,), "i32")
    p.t("cos", (s_max, d.D_R), kind="input", source="capture:cos")
    p.t("sin", (s_max, d.D_R), kind="input", source="capture:sin")
    if layers >= 1:
        p.t("act", (1, d.I_DENSE_PAD))
    if layers >= 2:
        p.t("topk_w", (1, TOPK_TOTAL_SLOTS), "f32")
        p.t("routing", (d.E_TOTAL, 1), "i32")
        p.t("mask", (d.E_TOTAL + 1,), "i32")
        p.t("logits_router", (1, d.E), "f32")
        if router4:
            # N2: the arrival counter of the four-task router, one for all the MoE layers (the
            # chain serialises them, and the last task of each resets it to zero)
            p.t("router_counter", (1,), "i32")
        p.t("mid", (1, TOPK_TOTAL_SLOTS, 2 * d.I_MOE))
        if fuse_silu:
            # one row per (XCD, tile) of the w2 gang: max_experts_per_xcd x n_tiles tiles per XCD
            p.t("w2_scratch", (XCDS * (-(-d.E_TOTAL // XCDS)) * (d.H // 64), d.I_MOE))
        else:
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
        if f"W_qkva_{l}" not in p.tensors:     # the previous layer's prefetch may have declared it (O8)
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
        if gemv:
            g = linear_grid_for(d.Q_OUT + d.KVA_OUT, linear_grid, strict=False)
            p.op("linear_gemv_layer", g, status="new", label=f"L{l}.qkva", input=x_in, w_norm=f"w_norm1_{l}",
                 weight=f"W_qkva_{l}", residual=None, output="qkva", grid_dim=(g, 1, 1),
                 block_dim=(256, 1, 1), norm=True, residual_add=False, eps=RMS_EPS)
        elif fuse1:
            g = grid_for_linear(d.Q_OUT + d.KVA_OUT)
            p.op("linear_norm_layer", g, status="new", label=f"L{l}.qkva", input=x_in, w_norm=f"w_norm1_{l}",
                 weight=f"W_qkva_{l}", output="qkva", scratch="qkva_scratch", grid_dim=(g, 1, 1),
                 block_dim=(256, 1, 1), eps=RMS_EPS)
        else:
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
        if prefetch:
            g = grid_for_linear(d.H)
            p.op("prefetch_layer", g, status="new", side=True, label=f"L{l}.prefetch_W_o",
                 weight=f"W_o_{l}", dummy="pf_dummy_o", grid_dim=(g, 1, 1), block_dim=(256, 1, 1))
        p.op("mla_prep_layer", dims.NH, status="new", label=f"L{l}.mla_prep",
             qkva="qkva", w_kv_norm=f"w_kv_norm_{l}", w_uk=f"W_uk_{l}", cos="cos", sin="sin",
             c_kv=f"c_kv_{l}", k_pe=f"k_pe_{l}", ql_nope="ql_nope", q_pe="q_pe",
             block_dim=(256, 1, 1))
        p.op("mla_attend_layer", n_splits if attend_tasks else XCDS, 1 if attend_tasks else splits_per_xcd,
             status="new", label=f"L{l}.mla_attend", per_tile=attend_tasks,
             ql_nope="ql_nope", q_pe="q_pe", c_kv=f"c_kv_{l}", k_pe=f"k_pe_{l}",
             partials="partials", softmax_scale=SOFTMAX_SCALE, split=SPLIT, n_splits=n_splits,
             **({"scores": "scores"} if debug_scores else {}))
        if merge_oproj:
            # N5: one operator for the merge and o_proj together, with o_proj's label, so
            # --stop-after and the x_res boundary keep their key and attn's last writer is this
            # operator, whose layer the dump's layer_of reads off the same label
            p.op("mla_merge_oproj_layer", d.NH * OPROJ_HALVES, status="new", label=f"L{l}.o_proj",
                 partials="partials", w_uv=f"W_uv_{l}", w_o=f"W_o_{l}", residual="x_res",
                 counter="oproj_counter", output="x_res", attn="attn", workspace="oproj_ws",
                 split=SPLIT, n_splits=n_splits, halves=OPROJ_HALVES)
        elif merge_tasks:
            p.op("mla_merge_uv_tile_layer", d.NH * merge_halves, status="new",
                 label=f"L{l}.mla_merge_uv", partials="partials", w_uv=f"W_uv_{l}", output="attn",
                 split=SPLIT, n_splits=n_splits, halves=merge_halves)
        else:
            p.op("mla_merge_uv_layer", XCDS, d.NH // XCDS, status="new", label=f"L{l}.mla_merge_uv",
                 partials="partials", w_uv=f"W_uv_{l}", output="attn", split=SPLIT, n_splits=n_splits)
        if not merge_oproj:                            # N5 issued the layer's o_proj with its merge
            if gemv:
                g = linear_grid_for(d.H, linear_grid, strict=False)
                p.op("linear_gemv_layer", g, status="new", label=f"L{l}.o_proj", input="attn", w_norm=None,
                     weight=f"W_o_{l}", residual="x_res", output="x_res", grid_dim=(g, 1, 1),
                     block_dim=(256, 1, 1), norm=False, residual_add=True, eps=0.0)
            elif tile_linears:
                g = grid_for_linear(d.H)
                p.op("linear_with_residual_layer", g, label=f"L{l}.o_proj", input="attn", weight=f"W_o_{l}",
                     residual="x_res", output="x_res", grid_dim=(g, 1, 1), block_dim=(256, 1, 1))
            else:
                p.op("gang_linear_with_residual_layer", XCDS, gang_tiles(d.H, TILE_N_O), label=f"L{l}.o_proj",
                     input="attn", weight=f"W_o_{l}", residual="x_res", output="x_res",
                     tile_n=TILE_N_O, output_stride=d.H)
        if prefetch and l + 1 < layers:
            g = grid_for_linear(d.Q_OUT + d.KVA_OUT)
            p.t(f"W_qkva_{l + 1}", (d.Q_OUT + d.KVA_OUT, d.H), kind="input", source=f"W_qkva_{l + 1}")
            p.op("prefetch_layer", g, status="new", side=True, label=f"L{l}.prefetch_W_qkva_next",
                 weight=f"W_qkva_{l + 1}", dummy="pf_dummy_qkva", grid_dim=(g, 1, 1), block_dim=(256, 1, 1))
        if not ((fuse_norm2 or router4) and l > 0):
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
            # layer 0's dense down projection stays on the stock per-tile linear under
            # --gemv-linears: its K is I_DENSE_PAD (11,264), and the GEMV keeps a lane's K slice
            # in registers (K / 64 values: 176 here, against 32 at K 2,048), which does not fit
            # (docs/gpu-experiments/04-kernels/05-local-preparation.md, S4: layer 0 is a later pass)
            if tile_linears or gemv:
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
            router_io = (dict(input="x_res", w_norm=f"w_norm2_{l}", h="h", eps=RMS_EPS)
                         if (fuse_norm2 or router4) else dict(input="h"))
            if router4:
                router_io["counter"] = "router_counter"
            p.op("moe_router_norm4_layer" if router4 else "moe_router_layer",
                 4 if router4 else 1, status="new", label=f"L{l}.router",
                 w_gate=f"W_gate_{l}", topk_w="topk_w", routing="routing", mask="mask",
                 logits="logits_router", route_log="route_log", layer_index=l - 1,
                 topk=d.TOPK, n_experts=d.E, n_forced=N_FORCED, scaling=ROUTED_SCALING,
                 block_dim=(256, 1, 1), **router_io)
            if gemv_w13:
                p.op("gang_moe_w13_gemv_layer", XCDS, W13_GEMV_TILES, status="new", label=f"L{l}.w13",
                     input="h", weight=f"W13_{l}", moe_routing_indices="routing", moe_mask="mask",
                     output="mid", tiles_per_expert=W13_GEMV_TILES)
            else:
                p.op("gang_moe_w13_linear_layer", XCDS, (2 * d.I_MOE) // 64, label=f"L{l}.w13",
                     input="h", weight=f"W13_{l}", moe_routing_indices="routing", moe_mask="mask",
                     output="mid")
            if prefetch:
                p.op("prefetch_moe_layer", TOPK_TOTAL_SLOTS * PREFETCH_PARTS, status="new", side=True,
                     label=f"L{l}.prefetch_W2", weight=f"W2_{l}", moe_mask="mask", dummy="pf_dummy_w2",
                     parts=PREFETCH_PARTS, block_dim=(256, 1, 1))
            if fuse_silu:
                p.op("gang_moe_w2_silu_linear_layer", XCDS, d.H // 64, status="new", label=f"L{l}.w2",
                     input="mid", weight=f"W2_{l}", moe_routing_indices="routing", moe_mask="mask",
                     output="out8", scratch="w2_scratch")
            else:
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
        if gemv:
            g = linear_grid_for(d.V, head_grid)
            p.op("linear_gemv_layer", g, status="new", label="head.lm_head", input="x_res", w_norm="w_final_norm",
                 weight="W_lm", residual=None, output="logits", grid_dim=(g, 1, 1),
                 block_dim=(256, 1, 1), norm=True, residual_add=False, eps=RMS_EPS)
        elif fuse1:
            g = grid_for_linear(d.V)
            p.t("lm_scratch", (g, d.H))
            p.op("linear_norm_layer", g, status="new", label="head.lm_head", input="x_res", w_norm="w_final_norm",
                 weight="W_lm", output="logits", scratch="lm_scratch", grid_dim=(g, 1, 1),
                 block_dim=(256, 1, 1), eps=RMS_EPS)
        else:
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


EMPTY_WIDTH = 256    # elements per row of the empty ladder's tensors (one 512-byte copy per task)


def build_empty_plan(ops: int, tasks: int, spin: int = 0, dims: Dims = REAL_DIMS, s_max: int = 1056) -> Plan:
    """I3 (docs/gpu-experiments/03-acceleration): the empty-task ladder, M operators of N copy
    tasks alternating two [N, EMPTY_WIDTH] BF16 tensors. Every operator reads its input whole and
    writes its own row of the output (partitioned on dim 0), so each boundary is one event with N
    triggers: the per-operator cost of the runtime as a function of the task count, with the
    per-task cost from the worker timing (I1). spin > 0: every task also runs the shader-clock
    spin of I2; the first operator's tasks print their [SPIN] lines (one per task). No model, no
    weights, no head: the plan has only the two workspaces."""
    assert ops >= 1 and tasks >= 1
    p = Plan(dims, s_max, 0, False, False)
    p.t("empty_a", (tasks, EMPTY_WIDTH))
    p.t("empty_b", (tasks, EMPTY_WIDTH))
    for k in range(ops):
        src, dst = ("empty_a", "empty_b") if k % 2 == 0 else ("empty_b", "empty_a")
        p.op("copy_layer", tasks, status="new", label=f"E{k}.copy", input=src, output=dst,
             grid_dim=(tasks, 1, 1), block_dim=(256, 1, 1), spin=spin, spin_print=1 if (spin and k == 0) else 0)
    return p


STREAM_K = 2048                              # the probe's row width: one BF16 row is exactly 4 KB
STREAM_ROW_BYTES = STREAM_K * 2


def stream_rows(kb: int) -> int:
    """Rows of the probe's [*, STREAM_K] BF16 tensor a task or a tile reads for `kb` kilobytes.

    A task reads whole rows (the GEMV's batch of eight rows is the load loop this probe
    measures), so `kb` must be a multiple of the 4 KB row: a byte count that did not follow
    from the rows would corrupt the one number the probe exists to produce. w13's tile, given
    as 305 KB in docs/gpu-experiments/04-kernels, is run at 304 KB (76 rows), 0.3% below it."""
    assert kb > 0, kb
    assert (kb * 1024) % STREAM_ROW_BYTES == 0, (
        f"--kb {kb}: a task reads whole {STREAM_ROW_BYTES // 1024} KB rows of the [*, {STREAM_K}] "
        f"BF16 tensor, so --kb must be a multiple of {STREAM_ROW_BYTES // 1024} "
        f"({kb // 4 * 4} or {(kb + 3) // 4 * 4} here)")
    return kb * 1024 // STREAM_ROW_BYTES


def build_stream_plan(ops: int, tasks: int, kb: int, gang: bool = False,
                      dims: Dims = REAL_DIMS, s_max: int = 1056) -> Plan:
    """L6 (M7 of docs/gpu-experiments/04-kernels/01-gemv-ideas.md): the stream probe, M operators
    of N tasks (or of 8 tiles x N tiles_per_xcd, --gang) that read `kb` kilobytes each with the
    GEMV's load loop and no multiply, on the empty ladder's machinery (I3).

    Two weight tensors alternate as the empty ladder's two copy tensors do, so an operator never
    reads the tensor the operator before it has just read. The weight is an input and never an
    output, so it cannot be what makes an operator a consumer of its predecessor, which the
    runtime requires of every operator (runtime.cc, register_mugraph: assert(num_shared_tensors
    >= 1)): two dummies alternate as well, each operator writing one and reading the other, and
    that second dummy is an input of the task the kernel never touches (fleet/patches/hunks/
    L6-stream.md, the last note). Every tensor is `new` (a zeroed buffer): the probe measures the
    time the bytes take to arrive, not their values."""
    assert ops >= 1 and tasks >= 1
    rows = stream_rows(kb)
    p = Plan(dims, s_max, 0, False, False)
    tiles = XCDS * tasks if gang else tasks          # the tasks, or the tiles over the 8 XCD slots
    for s in ("a", "b"):
        p.t(f"stream_w_{s}", (tiles * rows, STREAM_K))
        p.t(f"stream_dummy_{s}", (tiles, 4), "i32")
    for k in range(ops):
        cur, prev = ("a", "b") if k % 2 == 0 else ("b", "a")
        args = dict(weight=f"stream_w_{cur}", prev=f"stream_dummy_{prev}", dummy=f"stream_dummy_{cur}",
                    block_dim=(256, 1, 1))
        if gang:
            p.op("stream_gang_layer", XCDS, tasks, status="new", label=f"S{k}.stream",
                 rows_per_tile=rows, tiles_per_xcd=tasks, **args)
        else:
            p.op("stream_layer", tasks, status="new", label=f"S{k}.stream",
                 grid_dim=(tasks, 1, 1), **args)
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
