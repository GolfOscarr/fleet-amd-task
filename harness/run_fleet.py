#!/usr/bin/env python3
"""Build and run the Fleet graph, then dump every boundary it left behind (L10).

    python harness/run_fleet.py --layers N [--head] [--iters K] [--debug] [--stop-after L1.o_proj]
                                [--model-dir <snapshot>] [--ref harness/ref] [--out harness/fleet_out/<name>]
                                [--event-timing] [--nt-weights] [--pad-alloc GB] [--align-alloc BYTES]
                                [--workspaces-first] [--tile-linears] [--attend-tasks] [--split N]

--align-alloc BYTES re-bases every weight, capture and workspace on an aligned address and
--workspaces-first allocates the workspaces before the weights (the candidate M4 fault fixes,
docs/gpu-experiments/02-validation/02-session-plan.md, row A4); both are recorded in fleet_run_meta.json.
--pad-alloc GB holds a dummy device allocation of that size for the whole run,
made before the weights are packed, so every later buffer moves to a different
address without any change to the graph; fleet_run_meta.json records the pad
and the device address of every tensor (the M4 fault test of docs/gpu-experiments/02-validation).

Positions follow D14/D15: step is set to 1022 after compile() (the seeded
prepare_next_batch makes it 1023), num_new_tokens 1, qo_indptr [0, 1],
tokens[0:1024] the prompt; the run terminates after K iterations because
max_seq_length = 1024 + K (the stop test is step + 2 >= max_seq_length on
the step before the increment, persistent_kernel.cuh:560).

What is dumped (compare.py's inputs): fleet_boundaries.safetensors with
the keys of harness/common.py that are recoverable at the point the graph
stopped (every workspace holds the value of the last operator that wrote
it, so a run stopped after L1.o_proj yields B1, B2, B3, B4, B6, B7 of
layer 1 and a full-layer run yields B2, B3, B4, B6, B8-B13),
fleet_output_ids.json, fleet_route_log.json, fleet_hidden_per_layer.safetensors
(--debug), fwd_pass.log, event_timing.json (--event-timing), wall.json,
plan.json, fleet_run_meta.json. Boundary tensors are meaningful for
--iters 1 (decode step 0, what the reference captured); with more
iterations they hold the last iteration's values and are dumped with a note.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

def _outputs():
    # which argument names are outputs, per plan method (the rest are inputs); lazy: fleet.* imports torch
    from fleet.graph_plan import OUTPUT_ARGS
    return OUTPUT_ARGS


def last_writers(calls):
    """tensor name -> the label of the last operator that wrote it, in plan order."""
    w = {}
    outputs = _outputs()
    for c in calls:
        for k in outputs[c["method"]]:
            if k not in c["args"]:            # optional outputs (scores)
                continue
            v = c["args"][k]
            for name in (v if isinstance(v, list) else [v]):
                w[name] = c["label"]
    return w


def boundary_dump(plan, host, tokens, n_prompt, dims, iters=1):
    """Map the workspaces to boundary keys given who wrote them last.

    plan: plan_json(); host: name -> tensor (torch, on any device); tokens: the meta tokens.
    Returns (boundaries dict of CPU tensors, notes list)."""
    import torch

    calls = plan["calls"]
    w = last_writers(calls)
    b, notes = {}, []
    if iters != 1:
        notes.append(f"boundaries are from iteration {iters - 1}, not decode step 0")
    h = host

    def layer_of(label):
        return int(label.split(".")[0][1:]) if label.startswith("L") else None

    def put(key, t):
        b[key] = t.detach().cpu().clone()

    lab = w.get("h")
    if lab and lab.endswith(".norm1"):
        put(f"L{layer_of(lab)}.B1.norm1", h["h"][0])
    elif lab and (lab.endswith(".norm2") or lab.endswith(".router")):   # the router writes h under --fuse-norm2 (O1)
        put(f"L{layer_of(lab)}.norm2", h["h"][0])
    elif lab == "head.norm":
        put("head.B14.norm", h["h"][0])
    lab = w.get("qkva")
    if lab:
        put(f"L{layer_of(lab)}.B2.q", h["qkva"][0, : dims["Q_OUT"]])
        put(f"L{layer_of(lab)}.kva", h["qkva"][0, dims["Q_OUT"]:])
    lab = w.get("q_pe")
    if lab:
        put(f"L{layer_of(lab)}.B4.q_pe", h["q_pe"])
        put(f"L{layer_of(lab)}.ql_nope", h["ql_nope"])
    for name, lab in w.items():
        if name.startswith("c_kv_"):
            l = int(name.split("_")[-1])
            put(f"L{l}.B3.c_kv", h[name][n_prompt - 1])
            put(f"L{l}.B3.k_pe", h[f"k_pe_{l}"][n_prompt - 1])
    lab = w.get("scores")
    if lab:
        put(f"L{layer_of(lab)}.B5.scores", h["scores"][:, :n_prompt])     # positions 0..1023 at step 0
    lab = w.get("attn")
    if lab:
        put(f"L{layer_of(lab)}.B6.attn", h["attn"][0])
    lab = w.get("x_res")
    if lab and lab.endswith(".o_proj"):
        put(f"L{layer_of(lab)}.B7.x_res_attn", h["x_res"][0])
    elif lab and (lab.endswith(".combine") or lab.endswith(".down")):
        put(f"L{layer_of(lab)}.B13.layer_out", h["x_res"][0])
    elif lab == "prologue.embed":
        put("prologue.embed", h["x_res"][0])
    lab = w.get("mask")
    if lab:
        l = layer_of(lab)
        k = dims["TOPK"]
        mask = h["mask"].cpu()
        put(f"L{l}.B8.router_logits", h["logits_router"][0])
        put(f"L{l}.B9.topk_idx", mask[:k])
        put(f"L{l}.B10.topk_w", h["topk_w"][0, :k])
        if w.get("out8") and layer_of(w["out8"]) == l:
            out8 = h["out8"][0].float()
            for slot in range(k):
                put(f"L{l}.B11.expert_{int(mask[slot])}", h["out8"][0, slot])
            put(f"L{l}.B12.shared", out8[k] + out8[k + 1])
    if w.get("logits"):
        put("head.B15.logits", h["logits"][0])
    if w.get("tok_out"):
        put("head.B16.token", tokens[0, n_prompt: n_prompt + 1].cpu())
    hidden = {}
    for name, lab in w.items():
        if name.startswith("dbg_x_res_"):
            hidden[int(name.split("_")[-1])] = h[name][0].detach().cpu()
    if hidden:
        b["_hidden"] = torch.stack([hidden[l] for l in sorted(hidden)])
    return b, notes


class StdoutToFile:
    """Redirect fd 1 (the GPU printf sink) to a file for the duration of a block."""

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.stdout.flush()
        self.saved = os.dup(1)
        self.f = open(self.path, "ab")
        os.dup2(self.f.fileno(), 1)
        return self

    def __exit__(self, *a):
        sys.stdout.flush()
        os.dup2(self.saved, 1)
        os.close(self.saved)
        self.f.close()


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--head", action="store_true")
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--stop-after", default=None, help="operator label, e.g. L1.o_proj (truncated graph)")
    ap.add_argument("--model-dir", required=True, help="the checkpoint snapshot directory")
    ap.add_argument("--ref", default=str(common.REF_DIR))
    ap.add_argument("--out", default=None)
    ap.add_argument("--event-timing", action="store_true", help="compile with MPK_EVENT_TIMING=1")
    ap.add_argument("--nt-weights", action="store_true", help="USE_NT_WEIGHTS=1 (E2)")
    ap.add_argument("--attend-tasks", action="store_true",
                    help="mla_attend as one regular task per split instead of a gang task (session B, 2026-09-16)")
    ap.add_argument("--fuse-norm2", action="store_true",
                    help="the post-attention norm folded into the router in the MoE layers (O1, docs/gpu-experiments/03-acceleration)")
    ap.add_argument("--fuse-silu", action="store_true",
                    help="the silu-mul folded into the expert down projection's prologue (O2, docs/gpu-experiments/03-acceleration)")
    ap.add_argument("--split", type=int, default=0,
                    help="positions per attention split (graph_plan.SPLIT, 32); more, smaller splits give more tiles, 64 at most")
    ap.add_argument("--debug-scores", action="store_true",
                    help="MPK_DEBUG_SCORES=1 build; mla_attend also writes the scores (boundary B5)")
    ap.add_argument("--prompt", default=str(common.PROMPT_IDS))
    ap.add_argument("--pad-alloc", type=float, default=0.0, metavar="GB",
                    help="hold a dummy device allocation of GB gibibytes before packing (address shift)")
    ap.add_argument("--tile-linears", action="store_true",
                    help="issue qkva, o_proj, down and lm_head as per-tile linear_layer tasks (MAJ-7, docs/gpu-experiments/02-validation P5)")
    ap.add_argument("--align-alloc", type=int, default=0, metavar="BYTES",
                    help="re-base every weight, capture and workspace on a BYTES-aligned address (power of two; "
                         "the M4 fault candidates, docs/gpu-experiments/02-validation)")
    ap.add_argument("--workspaces-first", action="store_true",
                    help="allocate the workspaces from the plan before the weights are packed (address order)")
    return ap


def run_name(args):
    """The run directory name under harness/fleet_out, from the arguments."""
    pad = f"_pad{args.pad_alloc:g}" if args.pad_alloc else ""
    tile = "_tile" if args.tile_linears else ""
    al = f"_al{args.align_alloc}" if args.align_alloc else ""
    ws = "_wsfirst" if args.workspaces_first else ""
    nt = "_nt" if args.nt_weights else ""   # session B, 2026-09-16: the E2 runs overwrote their baselines
    sp = f"_s{args.split}" if args.split else ""
    at = "_at" if args.attend_tasks else ""
    fn2 = "_fn2" if args.fuse_norm2 else ""
    fs = "_fs" if args.fuse_silu else ""
    return (f"L{args.layers}{'_head' if args.head else ''}_it{args.iters}"
            + (f"_{args.stop_after}" if args.stop_after else "") + ("_scores" if args.debug_scores else "")
            + tile + at + fn2 + fs + nt + sp + al + ws + pad)


def tensor_addresses(host):
    """name -> device address of every attached tensor, for the address-dependence test."""
    return {name: int(t.data_ptr()) for name, t in host.items()}


def main():
    args = build_parser().parse_args()

    import torch
    from safetensors.torch import load_file, save_file
    from fleet import build_graph as B
    from fleet.pack_weights import pack_all, Dims

    name = run_name(args)
    out = Path(args.out) if args.out else common.ROOT / "harness/fleet_out" / name
    out.mkdir(parents=True, exist_ok=True)
    if args.event_timing:
        os.environ["MPK_EVENT_TIMING"] = "1"
    if args.nt_weights:
        os.environ["USE_NT_WEIGHTS"] = "1"
    if args.split:
        import fleet.graph_plan as _G
        assert 0 < args.split and -(-1056 // args.split) <= 64, "mla_merge_uv merges at most 64 splits"
        _G.SPLIT = args.split
    if args.debug_scores:
        os.environ["MPK_DEBUG_SCORES"] = "1"
    os.environ.setdefault("USE_GANG", "1")
    # persistent_kernel.py:350 defaults --offload-arch to gfx950; the MI300X is gfx942,
    # and the gfx942 patch's #if defined(__gfx950__) guards depend on this
    os.environ.setdefault("AMDGPU_TARGETS", "gfx942")

    prompt = json.loads(Path(args.prompt).read_text())
    n_prompt = len(prompt)
    s_max = n_prompt + args.iters
    ref_dir = Path(args.ref)
    pad = None
    if args.pad_alloc:
        pad = torch.empty(int(args.pad_alloc * 2 ** 30), dtype=torch.uint8, device="cuda")
        print(f"pad-alloc {args.pad_alloc:g} GiB at 0x{pad.data_ptr():x}")
    t0 = time.time()
    dims = Dims.from_config(json.loads((Path(args.model_dir) / "config.json").read_text()))
    if args.align_alloc:
        assert args.align_alloc >= 512 and args.align_alloc & (args.align_alloc - 1) == 0, "--align-alloc: power of two, >= 512"
    workspaces = None
    if args.workspaces_first:
        # the plan's tensors do not depend on --stop-after (it only cuts calls)
        from fleet import graph_plan as G
        pre_plan = G.build_plan(dims, s_max, args.layers, args.head, args.debug, args.debug_scores, args.tile_linears,
                                args.attend_tasks, args.fuse_norm2, args.fuse_silu)
        workspaces = B.allocate_workspaces(torch, pre_plan, args.align_alloc)
        print(f"workspaces-first: {len(workspaces)} buffers allocated before the weights")
    packed = pack_all(args.model_dir, "cuda", dims, layers=args.layers, head=args.head or None)
    if args.align_alloc:
        packed = {k: B.aligned_copy(torch, v, args.align_alloc) for k, v in packed.items()}
        print(f"align-alloc {args.align_alloc}: {len(packed)} weight tensors re-based")
    t_pack = time.time() - t0

    cache = load_file(str(ref_dir / "ref_cache.safetensors"))
    capture = {"cos": cache["cos"][:s_max].contiguous().cuda(), "sin": cache["sin"][:s_max].contiguous().cuda()}
    for l in range(args.layers):
        capture[f"c_kv_{l}"] = cache["c_kv"][l, :s_max].contiguous().cuda()
        capture[f"k_pe_{l}"] = cache["k_pe"][l, :s_max].contiguous().cuda()
    if args.align_alloc:
        capture = {k: B.aligned_copy(torch, v, args.align_alloc) for k, v in capture.items()}
    meta = B.make_meta(torch, s_max, prompt, n_prompt)

    t1 = time.time()
    mpk, host, plan = B.build(packed, capture, meta, dims=dims, s_max=s_max, layers=args.layers,
                              head=args.head, debug=args.debug, stop_after=args.stop_after,
                              debug_scores=args.debug_scores, tile_linears=args.tile_linears,
                              attend_tasks=args.attend_tasks, fuse_norm2=args.fuse_norm2, fuse_silu=args.fuse_silu,
                              align=args.align_alloc, workspaces=workspaces)
    pj = B.plan_json(plan)
    (out / "plan.json").write_text(json.dumps(pj) + "\n")
    mpk.compile(output_dir=str(out / "build"))
    t_build = time.time() - t1
    # meta tensors after compile(): init_kernel zeroes step and qo_indptr (04-memory-plan.md)
    meta["step"].fill_(n_prompt - 2)
    meta["num_new_tokens"].fill_(1)
    meta["qo_indptr_buffer"].copy_(torch.tensor([0, 1], dtype=torch.int32, device="cuda"))
    meta["tokens"][0, :n_prompt] = torch.tensor(prompt, dtype=torch.int64, device="cuda")
    meta["tokens"][0, n_prompt:] = 0
    torch.cuda.synchronize()

    # the static part of the record, with the addresses, before the run: a faulting run keeps it
    # (docs/gpu-experiments/02-validation/02-session-plan.md, the fault decision tree reads the addresses of the failing run)
    meta_out = {
        "layers": args.layers, "head": args.head, "iters": args.iters, "debug": args.debug,
        "stop_after": args.stop_after, "s_max": s_max, "n_prompt": n_prompt, "tile_linears": args.tile_linears,
        "attend_tasks": args.attend_tasks, "fuse_norm2": args.fuse_norm2, "fuse_silu": args.fuse_silu,
        "ops": len(pj["calls"]), "tasks": sum(c["tasks"] for c in pj["calls"]),
        "env": {k: os.environ.get(k) for k in ("MPK_EVENT_TIMING", "USE_NT_WEIGHTS", "USE_GANG", "AMDGPU_TARGETS",
                                                "MPK_DEBUG_SCORES")},
        "pad_alloc_gb": args.pad_alloc, "pad_addr": int(pad.data_ptr()) if pad is not None else None,
        "align_alloc": args.align_alloc, "workspaces_first": args.workspaces_first,
        "addresses": tensor_addresses(host), "completed": False,
    }
    (out / "fleet_run_meta.json").write_text(json.dumps(meta_out, indent=2) + "\n")

    fwd_log = out / "fwd_pass.log"
    fwd_log.write_text("")
    t2 = time.time()
    with StdoutToFile(fwd_log):
        mpk()
        torch.cuda.synchronize()
    t_mpk = time.time() - t2
    print(fwd_log.read_text()[-2000:])
    if Path("event_timing.json").exists():
        Path("event_timing.json").replace(out / "event_timing.json")

    ids = meta["tokens"][0, n_prompt: n_prompt + args.iters].cpu().tolist()
    (out / "fleet_output_ids.json").write_text(json.dumps(ids) + "\n")
    d = {"Q_OUT": dims.Q_OUT, "TOPK": dims.TOPK}
    b, notes = boundary_dump(pj, host, meta["tokens"], n_prompt, d, args.iters)
    hidden = b.pop("_hidden", None)
    save_file({k: v.contiguous() for k, v in b.items()}, str(out / "fleet_boundaries.safetensors"))
    if hidden is not None:
        save_file({"hidden": hidden.contiguous()}, str(out / "fleet_hidden_per_layer.safetensors"))
    if "route_log" in host:
        rl = host["route_log"].cpu()
        n_moe = max(0, args.layers - 1)
        log = [[{"idx": rl[s, j].tolist(), "w": []} for j in range(n_moe)] for s in range(args.iters)]
        (out / "fleet_route_log.json").write_text(json.dumps(log) + "\n")
    wall = {"mpk_wall_s": t_mpk, "iters": args.iters, "pack_s": t_pack, "build_s": t_build}
    (out / "wall.json").write_text(json.dumps(wall) + "\n")
    meta_out.update({"boundary_keys": sorted(b.keys()), "output_ids": ids, "notes": notes, "timings_s": wall,
                     "completed": True})
    (out / "fleet_run_meta.json").write_text(json.dumps(meta_out, indent=2) + "\n")
    print(f"ids {ids}; {len(b)} boundary tensors; mpk() {t_mpk * 1e3:.1f} ms for {args.iters} iterations -> {out}")


if __name__ == "__main__":
    main()
