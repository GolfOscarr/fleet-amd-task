#!/usr/bin/env python3
"""Build and run the Fleet graph, then dump every boundary it left behind (L10).

    python harness/run_fleet.py --layers N [--head] [--iters K] [--debug] [--stop-after L1.o_proj]
                                [--model-dir <snapshot>] [--ref harness/ref] [--out harness/fleet_out/<name>]
                                [--event-timing] [--nt-weights]

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

# which argument names are outputs, per plan method (the rest are inputs)
OUTPUTS = {
    "embed_layer": ["output"], "rmsnorm_layer": ["output"], "gang_linear_layer": ["output"],
    "gang_linear_with_residual_layer": ["output"], "gang_linear_silu_layer": ["output"],
    "mla_prep_layer": ["c_kv", "k_pe", "ql_nope", "q_pe"], "mla_attend_layer": ["partials"],
    "mla_merge_uv_layer": ["output"], "moe_router_layer": ["topk_w", "routing", "mask", "logits", "route_log"],
    "gang_moe_w13_linear_layer": ["output"], "moe_silu_mul_layer": ["output"],
    "gang_moe_w2_linear_layer": ["output"], "moe_mul_sum_add_layer": ["output"],
    "argmax_partial_layer": ["output"], "argmax_reduce_layer": ["output"], "copy_layer": ["output"],
}


def last_writers(calls):
    """tensor name -> the label of the last operator that wrote it, in plan order."""
    w = {}
    for c in calls:
        for k in OUTPUTS[c["method"]]:
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
    elif lab and lab.endswith(".norm2"):
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


def main():
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
    ap.add_argument("--prompt", default=str(common.PROMPT_IDS))
    args = ap.parse_args()

    import torch
    from safetensors.torch import load_file, save_file
    from fleet import build_graph as B
    from fleet.pack_weights import pack_all, Dims

    name = f"L{args.layers}{'_head' if args.head else ''}_it{args.iters}" + (f"_{args.stop_after}" if args.stop_after else "")
    out = Path(args.out) if args.out else common.ROOT / "harness/fleet_out" / name
    out.mkdir(parents=True, exist_ok=True)
    if args.event_timing:
        os.environ["MPK_EVENT_TIMING"] = "1"
    if args.nt_weights:
        os.environ["USE_NT_WEIGHTS"] = "1"
    os.environ.setdefault("USE_GANG", "1")

    prompt = json.loads(Path(args.prompt).read_text())
    n_prompt = len(prompt)
    s_max = n_prompt + args.iters
    ref_dir = Path(args.ref)
    t0 = time.time()
    dims = Dims.from_config(json.loads((Path(args.model_dir) / "config.json").read_text()))
    packed = pack_all(args.model_dir, "cuda", dims, layers=args.layers, head=args.head or None)
    t_pack = time.time() - t0

    cache = load_file(str(ref_dir / "ref_cache.safetensors"))
    capture = {"cos": cache["cos"][:s_max].contiguous().cuda(), "sin": cache["sin"][:s_max].contiguous().cuda()}
    for l in range(args.layers):
        capture[f"c_kv_{l}"] = cache["c_kv"][l, :s_max].contiguous().cuda()
        capture[f"k_pe_{l}"] = cache["k_pe"][l, :s_max].contiguous().cuda()
    meta = B.make_meta(torch, s_max, prompt, n_prompt)

    t1 = time.time()
    mpk, host, plan = B.build(packed, capture, meta, dims=dims, s_max=s_max, layers=args.layers,
                              head=args.head, debug=args.debug, stop_after=args.stop_after)
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
    meta_out = {
        "layers": args.layers, "head": args.head, "iters": args.iters, "debug": args.debug,
        "stop_after": args.stop_after, "s_max": s_max, "n_prompt": n_prompt,
        "ops": len(pj["calls"]), "tasks": sum(c["tasks"] for c in pj["calls"]),
        "env": {k: os.environ.get(k) for k in ("MPK_EVENT_TIMING", "USE_NT_WEIGHTS", "USE_GANG", "AMDGPU_TARGETS")},
        "boundary_keys": sorted(b.keys()), "output_ids": ids, "notes": notes, "timings_s": wall,
    }
    (out / "fleet_run_meta.json").write_text(json.dumps(meta_out, indent=2) + "\n")
    print(f"ids {ids}; {len(b)} boundary tensors; mpk() {t_mpk * 1e3:.1f} ms for {args.iters} iterations -> {out}")


if __name__ == "__main__":
    main()
