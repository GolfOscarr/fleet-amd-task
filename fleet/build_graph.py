#!/usr/bin/env python3
"""Build the Fleet persistent kernel for DeepSeek-Coder-V2-Lite decode.

Turns the plan of graph_plan.py into mpk.* calls: allocates or attaches
every tensor, issues the operators in chain order, adds the four new
layer methods (the Python half of the eight-place recipe in
docs/fleet/04-repo-map.md), and compiles. Runs on the machine; locally,
--dry-run drives the same code against a recording fake that enforces
the reused wrappers' assertions, and writes the call list as JSON.

    python fleet/build_graph.py --dry-run [--layers N] [--no-head] [--debug] [--out calls.json]

On the machine (from run_fleet.py):
    mpk, tensors = build(packed, capture, meta, layers=N, head=True)
    mpk.compile(output_dir=...); set meta tensors; mpk()
"""
import argparse
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fleet import graph_plan as G  # noqa: E402
from fleet.pack_weights import Dims, REAL_DIMS, XCDS  # noqa: E402

TORCH_DTYPE = {"bf16": "torch.bfloat16", "f32": "torch.float32", "i32": "torch.int32", "i64": "torch.int64"}


# ----------------------------------------------------------------------------
# the new layer methods (Python half of the recipe)


def _new_task(mpk, grid_dim, block_dim, inputs, task_type, params):
    """inputs: list of (DTensor, imap, forloop_dim). Mirrors the shipped *_layer methods."""
    if hasattr(mpk, "new_task_fake"):                 # the dry-run recorder
        return mpk.new_task_fake(grid_dim, block_dim, inputs, task_type, params)
    from mirage.mpk.persistent_kernel import TBGraph, CyTBGraph

    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    for t, imap, fdim in inputs:
        tb_graph.new_input(t, imap, fdim, True)
    mpk.kn_graph.customized([t for t, _, _ in inputs], tb_graph)
    mpk.kn_graph.register_task(tb_graph, task_type, params)


def mla_prep_layer(mpk, qkva, w_kv_norm, w_uk, cos, sin, c_kv, k_pe, ql_nope, q_pe,
                   block_dim=(256, 1, 1)):
    """One task per layer: kv_a_layernorm, RoPE, cache append at row step, ql_nope = q_nope @ W_uk."""
    assert qkva.num_dims == 2 and w_uk.num_dims == 3 and c_kv.num_dims == 2 and k_pe.num_dims == 2
    assert ql_nope.num_dims == 2 and q_pe.num_dims == 2
    nh, d_n, d_c = w_uk.dim(0), w_uk.dim(1), w_uk.dim(2)
    d_r = k_pe.dim(1)
    assert qkva.dim(1) == nh * (d_n + d_r) + d_c + d_r, qkva.dim(1)
    assert ql_nope.dim(0) == nh and ql_nope.dim(1) == d_c and q_pe.dim(1) == d_r
    _new_task(mpk, (1, 1, 1), block_dim,
              [(qkva, (-1, -1, -1), -1), (w_kv_norm, (-1, -1, -1), -1), (w_uk, (-1, -1, -1), -1),
               (cos, (-1, -1, -1), -1), (sin, (-1, -1, -1), -1),
               (c_kv, (-1, -1, -1), -1), (k_pe, (-1, -1, -1), -1),
               (ql_nope, (-1, -1, -1), -1), (q_pe, (-1, -1, -1), -1)],
              "mla_prep_mi300", [nh, d_n, d_r, d_c])


def mla_attend_layer(mpk, ql_nope, q_pe, c_kv, k_pe, partials, softmax_scale, split, n_splits,
                     scores=None, block_dim=(256, 1, 1), per_tile=False):
    """Gang task, 8 x ceil(n_splits / 8) tiles: split = tile * 8 + bid.x; partials [n_splits, nh, partials_row(d_c)].

    per_tile: one regular task per split instead (grid n_splits, task type mla_attend_tile_mi300):
    the runtime hands each task its row of partials and its index; no gang dispatch (session B,
    2026-09-16: the gang path cost 100 to 175 us per attention against 38 us for the grid standalone).

    scores: optional second output [nh, s_max] FP32 for boundary B5; only written by the
    MLA_ATTEND_DEBUG_SCORES build (MPK_DEBUG_SCORES=1 at compile time)."""
    assert partials.num_dims == 3 and partials.dim(0) == n_splits
    assert partials.dim(1) == ql_nope.dim(0) and partials.dim(2) == G.partials_row(ql_nope.dim(1))
    assert c_kv.dim(0) == k_pe.dim(0) and -(-c_kv.dim(0) // split) == n_splits
    tiles_per_xcd = 1 if per_tile else -(-n_splits // XCDS)
    grid = (n_splits, 1, 1) if per_tile else (XCDS, 1, 1)
    tensors = [(ql_nope, (-1, -1, -1), -1), (q_pe, (-1, -1, -1), -1),
               (c_kv, (-1, -1, -1), -1), (k_pe, (-1, -1, -1), -1),
               (partials, (0, -1, -1), -1)]                        # partition by split slot
    if scores is not None:
        assert scores.num_dims == 2 and scores.dim(0) == ql_nope.dim(0) and scores.dim(1) == c_kv.dim(0)
        tensors.append((scores, (-1, -1, -1), -1))
    _new_task(mpk, grid, block_dim, tensors,
              "mla_attend_tile_mi300" if per_tile else "mla_attend_mi300",
              [G.float_bits(softmax_scale), split, n_splits, tiles_per_xcd, ql_nope.dim(0),
               ql_nope.dim(1), q_pe.dim(1)])


def mla_merge_uv_layer(mpk, partials, w_uv, output, split, n_splits, block_dim=(256, 1, 1)):
    """Gang task, 8 x (nh / 8) tiles: head h = 2 bid.x + t; output columns [128 h, 128 h + 128)."""
    nh, d_v, d_c = w_uv.dim(0), w_uv.dim(1), w_uv.dim(2)
    assert nh % XCDS == 0 and output.dim(1) == nh * d_v and partials.dim(2) == G.partials_row(d_c)
    assert d_c % 256 == 0, "K of the W_uv product must be a multiple of 256"
    assert n_splits <= 64, "mla_merge_uv merges one split per lane of one wavefront"
    _new_task(mpk, (XCDS, 1, 1), block_dim,
              [(partials, (-1, -1, -1), -1), (w_uv, (0, -1, -1), -1), (output, (1, -1, -1), -1)],
              "mla_merge_uv_mi300", [split, n_splits, nh // XCDS, nh, d_v, d_c])


def moe_router_layer(mpk, input, w_gate, topk_w, routing, mask, logits, route_log, layer_index,
                     topk, n_experts, n_forced, scaling, block_dim=(256, 1, 1),
                     w_norm=None, h=None, eps=None):
    """One task per MoE layer: FP32 GEMV, softmax, top-k, forced experts; also logs the routing.
    With w_norm, h and eps (O1, docs/gpu-experiments/03-acceleration): the post-attention norm is
    folded in; `input` is then x_res, and h [1, H] is written for the expert gate-up
    (registration moe_router_norm_mi300: inputs x_res, w_norm, W_gate; outputs h, ...)."""
    assert w_gate.dim(0) == n_experts and routing.dim(0) == n_experts + n_forced
    assert mask.dim(0) == n_experts + n_forced + 1 and topk_w.dim(1) == topk + n_forced
    assert logits.dim(1) == n_experts
    fused = w_norm is not None
    assert fused == (h is not None) == (eps is not None), "w_norm, h and eps come together"
    ins = [(input, (-1, -1, -1), -1)]
    outs = []
    params = [topk, n_experts, n_forced, G.float_bits(scaling), layer_index, input.dim(1)]
    if fused:
        assert w_norm.dim(0) == input.dim(1) and h.dim(1) == input.dim(1)
        ins.append((w_norm, (-1, -1, -1), -1))
        outs.append((h, (-1, -1, -1), -1))
        params.append(G.float_bits(eps))
    ins.append((w_gate, (-1, -1, -1), -1))
    outs += [(topk_w, (-1, -1, -1), -1), (routing, (-1, -1, -1), -1), (mask, (-1, -1, -1), -1),
             (logits, (-1, -1, -1), -1), (route_log, (-1, -1, -1), -1)]
    _new_task(mpk, (1, 1, 1), block_dim, ins + outs,
              "moe_router_norm_mi300" if fused else "moe_router_mi300", params)


def gang_moe_w2_silu_linear_layer(mpk, input, weight, moe_routing_indices, moe_mask, output, scratch,
                                  block_dim=(256, 1, 1)):
    """O2 (docs/gpu-experiments/03-acceleration): the stock gang w2 with the silu-mul in its
    prologue. input is mid [1, topk, 2K] (gate | up per slot); each tile writes its slot's
    activation row into scratch [8 x tiles per XCD, K] and runs the CK GEMM on it. The imaps
    and the three params are the stock gang_moe_w2_linear_layer's; K comes from the weight."""
    assert input.num_dims == 3 and weight.num_dims == 3 and output.num_dims == 3 and scratch.num_dims == 2
    k = weight.dim(2)
    assert input.dim(2) == 2 * k and output.dim(2) == weight.dim(1) and scratch.dim(1) == k
    assert weight.dim(1) % 64 == 0 and k % 128 == 0, weight.shape
    assert moe_routing_indices.dim(0) == weight.dim(0) and moe_mask.dim(0) == weight.dim(0) + 1
    n_tiles = weight.dim(1) // 64
    max_e = (weight.dim(0) + 7) // 8
    total = max_e * n_tiles
    assert scratch.dim(0) == XCDS * total, (scratch.shape, XCDS * total)
    _new_task(mpk, (XCDS, 1, 1), block_dim,
              [(input, (-1, -1, -1), 2), (weight, (-1, 1, -1), 2),
               (moe_routing_indices, (-1, -1, -1), -1), (moe_mask, (-1, -1, -1), -1),
               (output, (-1, 2, -1), -1), (scratch, (-1, -1, -1), -1)],
              "gang_moe_w2_silu_linear_mi300", [n_tiles, max_e, total])


def copy_layer(mpk, input, output, grid_dim=(1, 1, 1), block_dim=(256, 1, 1)):
    """Debug builds only: snapshot of the residual after a layer (an identity task)."""
    assert input.num_dims == 2 and output.num_dims == 2 and input.dim(1) == output.dim(1)
    _new_task(mpk, grid_dim, block_dim, [(input, (-1, -1, -1), -1), (output, (-1, -1, -1), -1)],
              "copy_mi300", [input.dim(1)])


NEW_LAYERS = {
    "mla_prep_layer": mla_prep_layer,
    "mla_attend_layer": mla_attend_layer,
    "mla_merge_uv_layer": mla_merge_uv_layer,
    "gang_moe_w2_silu_linear_layer": gang_moe_w2_silu_linear_layer,
    "moe_router_layer": moe_router_layer,
    "copy_layer": copy_layer,
}


# ----------------------------------------------------------------------------
# driving the plan


TORCH_DTYPES = {"bf16": "bfloat16", "f32": "float32", "i32": "int32", "i64": "int64"}


def aligned_copy(torch, t, align):
    """A copy of t whose base address is a multiple of align bytes (the M4 fault tooling,
    docs/gpu-experiments/02-validation P1: the caching allocator aligns to 512 only). The backing buffer stays alive
    through the returned view's storage."""
    assert align > 0 and align & (align - 1) == 0, "align must be a power of two"
    n = t.numel() * t.element_size()
    raw = torch.empty(n + align, dtype=torch.uint8, device=t.device)
    off = (-raw.data_ptr()) % align
    out = raw[off:off + n].view(t.dtype).view(t.shape)
    out.copy_(t)
    assert out.data_ptr() % align == 0
    return out


ROW_SLACK = 16   # rows of backing storage behind every single-row activation (the M4 fault)


def new_workspace(torch, t, align=0, device="cuda"):
    """A zeroed workspace for plan tensor t, aligned to align bytes when align is set.

    A [1, D] activation is backed by ROW_SLACK rows and the first row is returned: the stock
    gang_linear_silu_kernel tiles M by 16 with no active-token mask, so at batch 1 it reads
    16 rows from its input (docs/gpu-experiments/02-validation/04-results.md, the fault of M4). The over-read stays
    inside this allocation whatever the allocator puts after it; the view's storage keeps the
    rows alive."""
    dtype = getattr(torch, TORCH_DTYPES[t.dtype])
    if len(t.shape) == 2 and t.shape[0] == 1:
        raw = torch.zeros((ROW_SLACK, t.shape[1]), dtype=dtype, device=device)
        raw = aligned_copy(torch, raw, align) if align else raw
        return raw[:1]
    buf = torch.zeros(t.shape, dtype=dtype, device=device)
    return aligned_copy(torch, buf, align) if align else buf


def allocate_workspaces(torch, plan, align=0, device="cuda"):
    """Every non-input tensor of the plan, allocated now (--workspaces-first: before the weights
    are packed, so the workspaces sit below the weights instead of above them)."""
    return {t.name: new_workspace(torch, t, align, device) for t in plan.tensors.values() if t.kind != "input"}


def make_tensors(mpk, plan, packed, capture, meta, torch, align=0, workspaces=None):
    """Attach inputs and allocate workspaces; returns name -> DTensor and name -> torch tensor.
    workspaces: pre-allocated buffers by name (allocate_workspaces); align: alignment of any
    workspace allocated here (the inputs are aligned by the caller, see run_fleet.py)."""
    dt = {}
    host = {}
    workspaces = workspaces or {}
    for t in plan.tensors.values():
        if t.kind == "input":
            if t.source.startswith("meta:"):
                src = meta[t.source[5:]]
            elif t.source.startswith("capture:"):
                src = capture[t.source[8:]]
            else:
                src = packed[t.source]
            assert tuple(src.shape) == t.shape, (t.name, tuple(src.shape), t.shape)
            host[t.name] = src
            dt[t.name] = mpk.attach_input(torch_tensor=src, name=t.name)
        else:
            buf = workspaces.get(t.name)
            if buf is None:
                buf = new_workspace(torch, t, align)
            assert tuple(buf.shape) == t.shape, (t.name, tuple(buf.shape), t.shape)
            host[t.name] = buf
            dt[t.name] = mpk.attach_input(torch_tensor=buf, name=t.name)
    return dt, host


def issue_calls(mpk, plan, dt):
    """Issue every operator of the plan, in order."""
    for c in plan.calls:
        kwargs = {}
        for k, v in c.args.items():
            if isinstance(v, str) and v in dt:
                kwargs[k] = dt[v]
            elif isinstance(v, tuple) and v and all(isinstance(x, str) and x in dt for x in v):
                kwargs[k] = tuple(dt[x] for x in v)
            else:
                kwargs[k] = v
        if c.method in NEW_LAYERS:
            NEW_LAYERS[c.method](mpk, **kwargs)
        elif c.method == "argmax_reduce_layer":
            try:
                mpk.argmax_reduce_layer(**kwargs)
            except TypeError:
                # unpatched API: no output_to_tokens kwarg; the run will not advance tokens
                kwargs.pop("output_to_tokens")
                print("WARNING: argmax_reduce_layer without output_to_tokens (glue patch not applied)")
                mpk.argmax_reduce_layer(**kwargs)
        else:
            getattr(mpk, c.method)(**kwargs)


def make_meta(torch, s_max, prompt_ids, n_prompt):
    """The ten meta tensors of 04-memory-plan.md, with the values D14 prescribes."""
    tokens = torch.zeros(1, s_max, dtype=torch.int64, device="cuda")
    tokens[0, :n_prompt] = torch.tensor(prompt_ids, dtype=torch.int64)
    meta = {
        "step": torch.full((1,), n_prompt - 2, dtype=torch.int32, device="cuda"),   # 1022: seeded prepare_next_batch makes it 1023
        "tokens": tokens,
        "input_tokens": torch.zeros(1, 1, dtype=torch.int64, device="cuda"),
        "output_tokens": torch.zeros(1, 1, dtype=torch.int64, device="cuda"),
        "num_new_tokens": torch.ones(1, dtype=torch.int32, device="cuda"),
        "prompt_lengths": torch.full((1,), n_prompt, dtype=torch.int32, device="cuda"),
        "qo_indptr_buffer": torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        "paged_kv_indptr_buffer": torch.zeros(2, dtype=torch.int32, device="cuda"),
        "paged_kv_indices_buffer": torch.zeros(1, dtype=torch.int32, device="cuda"),
        "paged_kv_last_page_len_buffer": torch.zeros(1, dtype=torch.int32, device="cuda"),
    }
    return meta


def plan_json(plan):
    """The plan as JSON-able data (written next to every run for measure.py and the dumps)."""
    return {"layers": plan.layers, "head": plan.head, "debug": plan.debug, "s_max": plan.s_max,
            "debug_scores": "scores" in plan.tensors,
            "tensors": {n: {"shape": list(t.shape), "dtype": t.dtype, "kind": t.kind, "source": t.source}
                        for n, t in plan.tensors.items()},
            "calls": [{"method": c.method, "label": c.label, "status": c.status, "tasks": c.tasks,
                       "tiles": c.tiles, "args": {k: (list(v) if isinstance(v, tuple) else v)
                                                  for k, v in c.args.items()}} for c in plan.calls]}


def build(packed, capture, meta, dims=REAL_DIMS, s_max=1056, layers=27, head=True, debug=False,
          stop_after=None, debug_scores=False, tile_linears=False, attend_tasks=False, num_workers=296, num_schedulers=8,
          profiler_tensor=None, align=0, workspaces=None, fuse_norm2=False, fuse_silu=False):
    """On the machine: construct the PersistentKernel, attach, issue, return (mpk, host tensors, plan)."""
    import torch
    import mirage as mi

    plan = G.build_plan(dims, s_max, layers, head, debug, debug_scores, tile_linears, attend_tasks, fuse_norm2,
                        fuse_silu)
    if stop_after:
        plan.truncate(stop_after)
    assert not plan.chain_violations(), f"the runtime would reject this graph: {plan.chain_violations()}"
    mpk = mi.PersistentKernel(
        mode="online", world_size=1, mpi_rank=0, num_workers=num_workers,
        num_local_schedulers=num_schedulers, num_remote_schedulers=0,
        max_seq_length=s_max, max_num_batched_requests=1, max_num_batched_tokens=1,
        max_num_pages=1, page_size=s_max, meta_tensors=meta, profiler_tensor=profiler_tensor,
        trace_name="", spec_decode_config=None, use_cutlass_kernel=False, eos_token_id=-1,
    )
    meta_for_inputs = {"input_tokens": meta["input_tokens"], "output_tokens": meta["output_tokens"]}
    dt, host = make_tensors(mpk, plan, packed, capture, meta_for_inputs, torch, align, workspaces)
    issue_calls(mpk, plan, dt)
    return mpk, host, plan


# ----------------------------------------------------------------------------
# dry run: the same code against a recording fake


class FakeDTensor:
    def __init__(self, name, shape):
        self.name, self.shape = name, tuple(shape)

    @property
    def num_dims(self):
        return len(self.shape)

    def dim(self, i):
        return self.shape[i]


class FakeMPK:
    """Records every call; re-implements the reused wrappers' assertions
    (persistent_kernel.py) so the plan's arithmetic is checked without Fleet."""

    def __init__(self, max_num_batched_tokens=1):
        self.calls = []
        self.max_num_batched_tokens = max_num_batched_tokens
        self.kn_graph = self
        self.argmax_partial_output_size = None

    # the recording fake for kn_graph
    def customized(self, tensors, tb_graph):
        pass

    def new_task_fake(self, grid_dim, block_dim, inputs, task_type, params):
        self._rec(task_type + "@new", grid_dim=grid_dim, inputs=[t.name for t, _, _ in inputs],
                  imaps=[list(m) for _, m, _ in inputs])
        self.register_task(None, task_type, params)

    def register_task(self, tb_graph, task_type, params=()):
        self.calls[-1]["task_type"] = task_type
        self.calls[-1]["params"] = list(params)

    def _rec(self, method, **kw):
        rec = {"method": method}
        for k, v in kw.items():
            if isinstance(v, FakeDTensor):
                rec[k] = v.name
            elif isinstance(v, tuple) and v and isinstance(v[0], FakeDTensor):
                rec[k] = [x.name for x in v]
            else:
                rec[k] = v
        self.calls.append(rec)
        return rec

    def attach_input(self, torch_tensor, name):
        return FakeDTensor(name, torch_tensor.shape)

    def embed_layer(self, input, weight, output, grid_dim, block_dim, input_source=0):
        self._rec("embed_layer", input=input, weight=weight, output=output, input_source=input_source)
        self.register_task(None, "embedding", [input_source])

    def rmsnorm_layer(self, input, weight, output, grid_dim, block_dim):
        assert input.num_dims == 2 and output.num_dims == 2
        self._rec("rmsnorm_layer", input=input, weight=weight, output=output)
        self.register_task(None, "rmsnorm", [])

    def _gang(self, method, input, weight, output, tile_n, output_stride, **extra):
        assert input.num_dims == 2 and weight.num_dims == 2 and output.num_dims == 2
        n = weight.dim(0)
        assert n % 8 == 0, n
        chunk = n // 8
        assert chunk % tile_n == 0, (chunk, tile_n)
        assert output.dim(1) == output_stride, (output.dim(1), output_stride)
        n_tiles = chunk // tile_n
        self._rec(method, input=input, weight=weight, output=output, tile_n=tile_n,
                  output_stride=output_stride, **extra)
        return n_tiles

    def gang_linear_layer(self, input, weight, output, tile_n, output_stride, m_tiles=1, wgm=0):
        n_tiles = self._gang("gang_linear_layer", input, weight, output, tile_n, output_stride)
        self.register_task(None, "gang_linear_mi300", [output_stride, tile_n, 1, 1, n_tiles, n_tiles, wgm])

    def gang_linear_with_residual_layer(self, input, weight, residual, output, tile_n, output_stride,
                                        m_tiles=1, wgm=0):
        assert residual.num_dims == 2
        n_tiles = self._gang("gang_linear_with_residual_layer", input, weight, output, tile_n,
                             output_stride, residual=residual)
        self.register_task(None, "gang_linear_res_mi300", [output_stride, tile_n, 1, 1, n_tiles, n_tiles, wgm])

    def gang_linear_silu_layer(self, input, weight, output, tile_n, output_stride, m_tiles=1, wgm=0):
        assert input.num_dims == 2 and weight.num_dims == 2 and output.num_dims == 2
        gate_up = weight.dim(0)
        assert gate_up % 8 == 0
        n_weight_tiles = (gate_up // 8) // tile_n
        n_tiles = n_weight_tiles // 2
        assert n_tiles * 2 == n_weight_tiles, "silent truncation in the shipped wrapper"
        assert output.dim(1) == output_stride == gate_up // 2
        self._rec("gang_linear_silu_layer", input=input, weight=weight, output=output, tile_n=tile_n,
                  output_stride=output_stride)
        self.register_task(None, "gang_linear_silu_mi300", [output_stride, tile_n, 1, 1, n_tiles, n_tiles, wgm])

    def linear_layer(self, input, weight, output, grid_dim, block_dim):
        # the stock non-gang linear (persistent_kernel.py:2045): grid_dim[0] tasks split
        # the output columns; the runtime reads the output stride from the tensor
        assert input.num_dims == 2 and weight.num_dims == 2 and output.num_dims == 2
        assert weight.dim(1) == input.dim(1), (weight.dim(1), input.dim(1))    # reduction
        assert weight.dim(0) == output.dim(1), (weight.dim(0), output.dim(1))  # output size
        assert output.dim(1) % grid_dim[0] == 0, (output.dim(1), grid_dim[0])
        self._rec("linear_layer", input=input, weight=weight, output=output, grid_dim=list(grid_dim))
        self.register_task(None, "linear", [])

    def linear_with_residual_layer(self, input, weight, residual, output, grid_dim, block_dim):
        assert input.num_dims == 2 and weight.num_dims == 2 and output.num_dims == 2 and residual.num_dims == 2
        assert weight.dim(1) == input.dim(1), (weight.dim(1), input.dim(1))
        assert weight.dim(0) == output.dim(1) == residual.dim(1), (weight.dim(0), output.dim(1), residual.dim(1))
        assert output.dim(1) % grid_dim[0] == 0, (output.dim(1), grid_dim[0])
        self._rec("linear_with_residual_layer", input=input, weight=weight, residual=residual,
                  output=output, grid_dim=list(grid_dim))
        self.register_task(None, "linear_with_residual", [])

    def _gang_moe(self, method, input, weight, moe_routing_indices, moe_mask, output, k_mult):
        assert weight.num_dims == 3 and moe_routing_indices.num_dims == 2 and moe_mask.num_dims == 1
        assert output.num_dims == 3
        assert weight.dim(1) % 64 == 0 and weight.dim(2) % k_mult == 0, weight.shape
        assert moe_routing_indices.dim(0) == weight.dim(0) and moe_mask.dim(0) == weight.dim(0) + 1
        self._rec(method, input=input, weight=weight, moe_routing_indices=moe_routing_indices,
                  moe_mask=moe_mask, output=output)
        n_tiles = weight.dim(1) // 64
        max_e = (weight.dim(0) + 7) // 8
        return [n_tiles, max_e, max_e * n_tiles]

    def gang_moe_w13_linear_layer(self, input, weight, moe_routing_indices, moe_mask, output):
        assert input.num_dims == 2 and output.dim(2) == weight.dim(1) and input.dim(1) == weight.dim(2)
        p = self._gang_moe("gang_moe_w13_linear_layer", input, weight, moe_routing_indices, moe_mask, output, 256)
        self.register_task(None, "gang_moe_w13_linear_mi300", p)

    def gang_moe_w2_linear_layer(self, input, weight, moe_routing_indices, moe_mask, output):
        assert input.num_dims == 3 and input.dim(2) == weight.dim(2) and output.dim(2) == weight.dim(1)
        p = self._gang_moe("gang_moe_w2_linear_layer", input, weight, moe_routing_indices, moe_mask, output, 128)
        self.register_task(None, "gang_moe_w2_linear_mi300", p)

    def moe_silu_mul_layer(self, input, output, grid_dim, block_dim):
        assert input.num_dims == 3 and output.num_dims == 3 and input.dim(2) == 2 * output.dim(2)
        assert grid_dim[1] == input.dim(1)
        self._rec("moe_silu_mul_layer", input=input, output=output, grid_dim=grid_dim)
        self.register_task(None, "moe_silu_mul", [])

    def moe_mul_sum_add_layer(self, input, weight, residual, output, grid_dim, block_dim):
        assert input.num_dims == 3 and weight.num_dims == 2 and residual.num_dims == 2 and output.num_dims == 2
        assert weight.dim(1) == input.dim(1) and grid_dim[1] * 256 == input.dim(2)
        self._rec("moe_mul_sum_add_layer", input=input, weight=weight, residual=residual, output=output,
                  grid_dim=grid_dim)
        self.register_task(None, "moe_mul_sum_add_mi300", [])

    def argmax_partial_layer(self, input, output, grid_dim, block_dim):
        assert input.num_dims == 2 and len(output) == 2
        assert input.dim(1) % grid_dim[0] == 0, "argmax_partial: vocab must divide the slice count"
        self.argmax_partial_output_size = input.dim(1) // grid_dim[0]
        self._rec("argmax_partial_layer", input=input, output=output, grid_dim=grid_dim)
        self.register_task(None, "argmax_partial", [grid_dim[0]])

    def argmax_reduce_layer(self, input, output, grid_dim, block_dim, output_to_tokens=False):
        assert len(input) == 2 and output.num_dims == 2
        self._rec("argmax_reduce_layer", input=input, output=output, output_to_tokens=output_to_tokens)
        self.register_task(None, "argmax_reduce", [self.argmax_partial_output_size, int(output_to_tokens)])


def dry_run(dims=REAL_DIMS, s_max=1056, layers=27, head=True, debug=False, stop_after=None,
            debug_scores=False, tile_linears=False, attend_tasks=False, fuse_norm2=False, fuse_silu=False):
    plan = G.build_plan(dims, s_max, layers, head, debug, debug_scores, tile_linears, attend_tasks, fuse_norm2,
                        fuse_silu)
    if stop_after:
        plan.truncate(stop_after)
    assert not plan.chain_violations(), f"the runtime would reject this graph: {plan.chain_violations()}"
    mpk = FakeMPK()
    dt = {t.name: FakeDTensor(t.name, t.shape) for t in plan.tensors.values()}
    issue_calls(mpk, plan, dt)
    return plan, mpk.calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--layers", type=int, default=27)
    ap.add_argument("--no-head", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--s-max", type=int, default=1056)
    ap.add_argument("--stop-after", default=None, help="operator label, e.g. L1.o_proj")
    ap.add_argument("--debug-scores", action="store_true")
    ap.add_argument("--tile-linears", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.dry_run:
        sys.exit("the real build is driven from harness/run_fleet.py on the machine; use --dry-run here")
    plan, calls = dry_run(REAL_DIMS, args.s_max, args.layers, not args.no_head, args.debug, args.stop_after,
                          args.debug_scores, args.tile_linears)
    s = G.summary(plan)
    print(json.dumps({k: v for k, v in s.items()}, indent=None))
    print(f"{len(calls)} calls recorded; task types: {sorted(set(c['task_type'] for c in calls))}")
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": s, "calls": calls}, indent=1) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
