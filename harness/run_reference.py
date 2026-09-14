#!/usr/bin/env python3
"""Reference run: HF DeepseekV2ForCausalLM, BF16, greedy, with capture.

Produces every reference artifact of docs/design-doc/07-correctness.md and
the latent-cache capture of 05-prefill-interface.md, in --out:

  ref_output_ids.json                 32 greedy ids (manual argmax loop)
  ref_output_ids_generate.json        the same from model.generate (cross-check)
  ref_boundaries_step0.safetensors    B1-B16 for layers 0, 1 and the head at step 0
  ref_hidden_per_layer_step0.safetensors   x_res after every layer at step 0
  ref_cache_row1023.safetensors       c_kv, k_pe at the hand-over position, all layers
  ref_cache.safetensors               c_kv [L, S_MAX, 512], k_pe [L, S_MAX, 64], cos, sin
  ref_route_log.json                  top-6 ids and weights, [step][moe layer]
  ref_run_meta.json                   versions, device, agreement, timings

Positions (05-prefill-interface.md). The prompt has N = 1024 ids. The
reference prefills ids[0:1023] with the cache on, then runs position 1023
as a one-token step: that is Fleet iteration 0 (step = 1023), so its
boundary tensors are captured at M = 1 exactly as the Fleet path computes
them, and its kv_a_proj output is the row that mla_prep writes in
iteration 0 (ref_cache_row1023). The loop continues for 32 steps; the
argmax of each step is the next input. model.generate on the full 1024-id
prompt is run afterwards as the canonical HF greedy result; agreement
between the two is recorded, not assumed.

    python harness/run_reference.py --device cuda            # the real run (GPU, 31 GB)
    python harness/run_reference.py --smoke                  # tiny random model, CPU, seconds
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


# ----------------------------------------------------------------------------
# model loading


def smoke_config(config):
    """Shrink the real config to a model that runs on a laptop CPU in seconds."""
    config.update(
        dict(
            hidden_size=64,
            num_attention_heads=2,
            num_key_value_heads=2,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=16,
            kv_lora_rank=32,
            intermediate_size=96,
            moe_intermediate_size=32,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=2,
            num_hidden_layers=3,
            vocab_size=256,
            max_position_embeddings=4096,
        )
    )
    return config


def load_model(args):
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if args.smoke:
        config = smoke_config(config)
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        # RMSNorm weights initialise to ones, which would hide any per-layer
        # mix-up of norm weights; give every norm its own random weight
        with torch.no_grad():
            for name, mod in model.named_modules():
                if type(mod).__name__ == "DeepseekV2RMSNorm":
                    mod.weight.copy_(1.0 + 0.5 * torch.randn_like(mod.weight.float()).to(mod.weight.dtype))
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
    model = model.to(args.device).eval()
    return model


# ----------------------------------------------------------------------------
# capture helpers


class Capture:
    """Forward hooks that stash module inputs/outputs by key."""

    def __init__(self):
        self.store = {}
        self.handles = []

    def out(self, module, key, index=None):
        def hook(m, inp, output):
            o = output[index] if index is not None else output
            self.store[key] = o.detach()
        self.handles.append(module.register_forward_hook(hook))

    def inp(self, module, key):
        def hook(m, inp):
            self.store[key] = inp[0].detach()
        self.handles.append(module.register_forward_pre_hook(hook))

    def out_tuple(self, module, key):
        def hook(m, inp, output):
            self.store[key] = tuple(o.detach() if torch.is_tensor(o) else o for o in output)
        self.handles.append(module.register_forward_hook(hook))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def moe_layers(model):
    return [l for l, layer in enumerate(model.model.layers) if hasattr(layer.mlp, "gate")]


def register_route_hooks(model, cap):
    for l in moe_layers(model):
        cap.out_tuple(model.model.layers[l].mlp.gate, f"route.L{l}")


def register_kva_hooks(model, cap):
    for l, layer in enumerate(model.model.layers):
        cap.out(layer.self_attn.kv_a_proj_with_mqa, f"kva.L{l}")


def register_boundary_hooks(model, cap, layers):
    for l in layers:
        layer = model.model.layers[l]
        at = layer.self_attn
        cap.inp(layer, f"L{l}.layer_in")
        cap.out(layer.input_layernorm, f"L{l}.B1.norm1")
        cap.out(at.q_proj, f"L{l}.B2.q")
        cap.out(at.kv_a_layernorm, f"L{l}.B3.c_kv")
        cap.inp(at.o_proj, f"L{l}.B6.attn")
        cap.out(at.o_proj, f"L{l}.o_proj_out")
        cap.inp(layer.post_attention_layernorm, f"L{l}.B7.x_res_attn")
        cap.out(layer.post_attention_layernorm, f"L{l}.norm2")
        if hasattr(layer.mlp, "gate"):
            cap.inp(layer.mlp.gate, f"L{l}.gate_in")
            for e, expert in enumerate(layer.mlp.experts):
                cap.out(expert, f"L{l}.B11.expert_{e}")
            cap.out(layer.mlp.shared_experts, f"L{l}.B12.shared")
        cap.out(layer, f"L{l}.B13.layer_out", index=0)
    for l, layer in enumerate(model.model.layers):
        cap.out(layer, f"hidden.L{l}", index=0)
    cap.out(model.model.norm, "head.B14.norm")


# ----------------------------------------------------------------------------
# the run


def rope_tables(model, seq_len):
    """The model's own cos/sin tables, BF16, as its attention consumes them."""
    at = model.model.layers[0].self_attn
    probe = torch.zeros(1, dtype=torch.bfloat16, device=next(model.parameters()).device)
    cos, sin = at.rotary_emb(probe, seq_len=seq_len)
    return cos[:seq_len].contiguous(), sin[:seq_len].contiguous()


def latent_rows(model, kva, positions, cos, sin, layers=None):
    """kv_a_proj output [1, s, 576] at `positions` -> (c_kv [s, 512], k_pe [s, 64]).

    Applies the model's own kv_a_layernorm and apply_rotary_pos_emb, in the
    model's dtype, exactly as the attention forward does (05-prefill-interface.md).
    kva[i] belongs to layer layers[i] (default: layer i); each layer has its
    own kv_a_layernorm weight.
    """
    from importlib import import_module

    mod = import_module(type(model).__module__)
    apply_rotary_pos_emb = mod.apply_rotary_pos_emb
    layer0 = model.model.layers[0].self_attn
    d_c, d_r = layer0.kv_lora_rank, layer0.qk_rope_head_dim
    outs = []
    layers = list(range(len(kva))) if layers is None else layers
    assert len(layers) == len(kva)
    for l, x in zip(layers, kva):
        at = model.model.layers[l].self_attn
        c, kpe = x.split([d_c, d_r], dim=-1)
        c = at.kv_a_layernorm(c)                         # [1, s, d_c]
        kpe = kpe.view(1, -1, 1, d_r).transpose(1, 2)    # [1, 1, s, d_r]
        _, kpe = apply_rotary_pos_emb(kpe, kpe, cos, sin, positions)
        outs.append((c[0].contiguous(), kpe[0, 0].contiguous()))
    return outs


def step0_boundaries(model, cap, pkv, l, pos, cos, sin, row=0):
    """Derived boundaries for layer l at the one-token step at position `pos`."""
    from importlib import import_module

    mod = import_module(type(model).__module__)
    apply_rotary_pos_emb = mod.apply_rotary_pos_emb
    at = model.model.layers[l].self_attn
    nh, d_n, d_r = at.num_heads, at.qk_nope_head_dim, at.qk_rope_head_dim
    out = {}
    q = cap.store[f"L{l}.B2.q"].reshape(-1)                           # [nh*(d_n+d_r)]
    q = q.view(1, 1, nh, d_n + d_r).transpose(1, 2)                   # [1, nh, 1, d]
    q_nope, q_pe = q.split([d_n, d_r], dim=-1)
    pos_ids = torch.tensor([[pos]], device=q.device)
    q_pe_rot, _ = apply_rotary_pos_emb(q_pe, q_pe, cos, sin, pos_ids)
    out["B4.q_pe"] = q_pe_rot[0, :, 0].contiguous()                   # [nh, d_r]
    q_states = torch.cat([q_nope, q_pe_rot], dim=-1)                  # [1, nh, 1, d]
    keys = pkv.key_cache[l][row:row + 1]                              # [1, nh, S, d], RoPE applied
    scores = torch.matmul(q_states, keys.transpose(2, 3)) * at.softmax_scale
    out["B5.scores"] = scores[0, :, 0].float().contiguous()           # [nh, S]
    # self-check: the reference's own attention output must follow from these
    # scores (FP32 softmax, BF16 probabilities, BF16 values), which proves the
    # recomputed B5 is what the model used
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    attn = torch.matmul(probs, pkv.value_cache[l][row:row + 1])       # [1, nh, 1, d_v]
    attn = attn[0, :, 0].reshape(-1)
    err = (attn.float() - cap.store[f"L{l}.B6.attn"].reshape(-1).float()).abs().max().item()
    out["_selfcheck_b5_to_b6_max_abs_err"] = err
    # B3: the kv_a_proj row this step produced, after the model's norm and RoPE
    kva = cap.store[f"kva.L{l}"].reshape(1, 1, -1)                    # [1, 1, d_c + d_r]
    c, kpe = latent_rows(model, [kva], pos_ids, cos, sin, layers=[l])[0]   # [1, d_c], [1, d_r]
    out["B3.c_kv"] = c[0]
    out["B3.k_pe"] = kpe[0]
    return out


def first_row(v, n_rows):
    """Row 0 of a captured tensor when the forward ran n_rows identical rows."""
    if n_rows > 1 and torch.is_tensor(v) and v.dim() >= 2 and v.shape[0] == n_rows:
        return v[0:1]
    return v


def ordered_topk(idx, w):
    """torch.topk(sorted=False) leaves the order unspecified; order by descending
    weight, lower expert id on ties, which is the order the Fleet router emits."""
    pairs = sorted(zip(idx.tolist(), w.float().tolist()), key=lambda p: (-p[1], p[0]))
    return [p[0] for p in pairs], [p[1] for p in pairs]


def route_entry(model, cap, n_rows=1):
    out = []
    for l in moe_layers(model):
        idx, w = ordered_topk(first_row(cap.store[f"route.L{l}"][0], n_rows)[0],
                              first_row(cap.store[f"route.L{l}"][1], n_rows)[0])
        out.append({"idx": idx, "w": w})
    return out


def capture_step0(model, cap, ids, P, layers, cos, sin, pkv, n_rows=1):
    """Run position P as a one-token step (n_rows identical rows) with the
    boundary hooks registered and collect every step-0 artifact.

    Returns a dict: boundaries, selfcheck (B5 -> B6 max abs err per layer),
    hidden [L, H], row_handover [(c, k_pe)] per layer, logits [V], token,
    route (this step's routing entry), pkv (the cache after the step).
    """
    cfg = model.config
    device = next(model.parameters()).device
    L = cfg.num_hidden_layers
    cur = ids[:, P:P + 1].repeat(n_rows, 1)
    mask = torch.ones(n_rows, P + 1, device=device, dtype=torch.long)
    with torch.inference_mode():
        out = model(cur, attention_mask=mask, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        if n_rows > 1:
            cap.store = {k: (tuple(first_row(x, n_rows) for x in v) if isinstance(v, tuple)
                             else first_row(v, n_rows)) for k, v in cap.store.items()}
        logits = out.logits[0, -1].detach().clone()
        tok = int(torch.argmax(logits.float()).item())
        boundaries, selfcheck = {}, {}
        for l in layers:
            for k, v in cap.store.items():
                if k.startswith(f"L{l}."):
                    boundaries[k] = v.reshape(-1).clone() if v.numel() == v.shape[-1] else v.clone()
            derived = step0_boundaries(model, cap, pkv, l, P, cos, sin, row=0)
            selfcheck[f"L{l}"] = derived.pop("_selfcheck_b5_to_b6_max_abs_err")
            for k, v in derived.items():
                boundaries[f"L{l}.{k}"] = v.clone()
            if hasattr(model.model.layers[l].mlp, "gate"):
                h_in = cap.store[f"L{l}.gate_in"].reshape(-1)
                w_gate = model.model.layers[l].mlp.gate.weight
                boundaries[f"L{l}.B8.router_logits"] = F.linear(h_in.float(), w_gate.float()).clone()
                idx, w, _ = cap.store[f"route.L{l}"]
                ids, ws = ordered_topk(idx[0], w[0])
                boundaries[f"L{l}.B9.topk_idx"] = torch.tensor(ids, dtype=torch.int64)
                boundaries[f"L{l}.B10.topk_w"] = torch.tensor(ws, dtype=torch.float32)
                for e in range(cfg.n_routed_experts):
                    if e not in idx[0].tolist():
                        boundaries.pop(f"L{l}.B11.expert_{e}", None)
        boundaries["head.B14.norm"] = cap.store["head.B14.norm"].reshape(-1).clone()
        boundaries["head.B15.logits"] = logits.clone()
        boundaries["head.B16.token"] = torch.tensor([tok], dtype=torch.int64)
        hidden = torch.stack([cap.store[f"hidden.L{l}"].reshape(-1) for l in range(L)])
        kva_step0 = [cap.store[f"kva.L{l}"].reshape(1, 1, -1) for l in range(L)]
        pos_ids = torch.tensor([[P]], device=device)
        row_handover = latent_rows(model, kva_step0, pos_ids, cos, sin)
        route = route_entry(model, cap)
        # keep only the routing entries so later steps do not accumulate stale keys
        cap.store = {k: v for k, v in cap.store.items() if k.startswith("route.")}
    return dict(boundaries=boundaries, selfcheck=selfcheck, hidden=hidden, row_handover=row_handover,
                logits=logits, token=tok, route=route, pkv=pkv)


def run(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = load_model(args)
    device = next(model.parameters()).device
    cfg = model.config
    t_load = time.time() - t0

    if args.smoke:
        torch.manual_seed(1)
        n_prompt = 16
        ids = torch.randint(0, cfg.vocab_size, (1, n_prompt))
    else:
        ids = torch.tensor(json.loads(Path(args.prompt).read_text()))[None]
        n_prompt = ids.shape[1]
        assert n_prompt == common.N_PROMPT, n_prompt
    ids = ids.to(device)
    n_steps = args.steps
    P = n_prompt - 1                     # prefill length; position P is Fleet iteration 0
    s_max = n_prompt + n_steps
    L = cfg.num_hidden_layers
    layers = [l for l in common.BOUNDARY_LAYERS if l < L]

    cap = Capture()
    register_kva_hooks(model, cap)
    register_route_hooks(model, cap)

    # ---- prefill positions 0..P-1 -------------------------------------------
    from transformers import DynamicCache

    t1 = time.time()
    with torch.inference_mode():
        # an explicit Cache object: with None the model returns a legacy tuple
        out = model(ids[:, :P], attention_mask=torch.ones(1, P, device=device, dtype=torch.long),
                    past_key_values=DynamicCache(), use_cache=True)
        pkv = out.past_key_values
    t_prefill = time.time() - t1
    kva_prefill = [cap.store[f"kva.L{l}"] for l in range(L)]        # [1, P, 576] each
    cos, sin = rope_tables(model, s_max)

    # ---- the greedy loop: position P (iteration 0) .. P + n_steps - 1 --------
    register_boundary_hooks(model, cap, layers)
    t2 = time.time()
    st = capture_step0(model, cap, ids, P, layers, cos, sin, pkv)
    boundaries, selfcheck, hidden_per_layer = st["boundaries"], st["selfcheck"], st["hidden"]
    row_handover, logits_step0, pkv = st["row_handover"], st["logits"], st["pkv"]
    out_ids = [st["token"]]
    route_log = [st["route"]]
    cur = torch.tensor([[st["token"]]], device=device)
    with torch.inference_mode():
        for step in range(1, n_steps):
            pos = P + step
            mask = torch.ones(1, pos + 1, device=device, dtype=torch.long)
            out = model(cur, attention_mask=mask, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            tok = int(torch.argmax(out.logits[0, -1].float()).item())
            out_ids.append(tok)
            route_log.append(route_entry(model, cap))
            cur = torch.tensor([[tok]], device=device)
    t_loop = time.time() - t2
    cap.remove()

    # ---- canonical HF greedy on the full prompt (cross-check) -----------------
    t3 = time.time()
    with torch.inference_mode():
        gen = model.generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=n_steps, min_new_tokens=0,
            do_sample=False, num_beams=1, eos_token_id=None, pad_token_id=cfg.eos_token_id,
        )
    t_gen = time.time() - t3
    gen_ids = gen[0, n_prompt:].tolist()
    first_diff = next((i for i in range(min(len(gen_ids), n_steps)) if gen_ids[i] != out_ids[i]), None)
    agree = first_diff is None and len(gen_ids) == n_steps

    # ---- latent cache for the Fleet path ---------------------------------------
    d_c = cfg.kv_lora_rank
    d_r = cfg.qk_rope_head_dim
    c_kv = torch.zeros(L, s_max, d_c, dtype=torch.bfloat16)
    k_pe = torch.zeros(L, s_max, d_r, dtype=torch.bfloat16)
    pos_prefill = torch.arange(P, device=device)[None]
    rows = latent_rows(model, kva_prefill, pos_prefill, cos, sin)
    for l, (c, kpe) in enumerate(rows):
        c_kv[l, :P] = c.cpu()
        k_pe[l, :P] = kpe.cpu()
    row_c = torch.stack([c[0] for c, _ in row_handover]).cpu()
    row_k = torch.stack([k[0] for _, k in row_handover]).cpu()

    # ---- write ------------------------------------------------------------------
    from safetensors.torch import save_file

    def cpu(d):
        return {k: v.detach().cpu().contiguous() for k, v in d.items()}

    (out_dir / "ref_output_ids.json").write_text(json.dumps(out_ids) + "\n")
    (out_dir / "ref_output_ids_generate.json").write_text(json.dumps(gen_ids) + "\n")
    (out_dir / "ref_route_log.json").write_text(json.dumps(route_log) + "\n")
    save_file(cpu(boundaries), str(out_dir / "ref_boundaries_step0.safetensors"))
    save_file({"hidden": hidden_per_layer.cpu().contiguous()},
              str(out_dir / "ref_hidden_per_layer_step0.safetensors"))
    save_file({"c_kv": row_c, "k_pe": row_k}, str(out_dir / "ref_cache_row1023.safetensors"))
    save_file({"c_kv": c_kv, "k_pe": k_pe, "cos": cos.cpu().contiguous(), "sin": sin.cpu().contiguous()},
              str(out_dir / "ref_cache.safetensors"))
    import transformers

    meta = {
        "model": args.model if not args.smoke else "smoke (random weights, shrunken config)",
        "smoke": args.smoke,
        "n_prompt": n_prompt, "n_steps": n_steps, "prefill_len": P, "s_max": s_max,
        "handover_position": P,
        "device": str(device), "dtype": "bfloat16", "attn_implementation": "eager",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "platform": platform.platform(),
        "generate_agrees_with_loop": agree, "first_disagreement_index": first_diff,
        "generate_len": len(gen_ids),
        "softmax_scale": float(model.model.layers[0].self_attn.softmax_scale),
        "b5_selfcheck_max_abs_err": selfcheck,
        "timings_s": {"load": round(t_load, 2), "prefill": round(t_prefill, 3),
                      "loop_32_steps": round(t_loop, 3), "generate": round(t_gen, 3)},
        "boundary_keys": sorted(boundaries.keys()),
    }
    (out_dir / "ref_run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {len(boundaries)} boundary tensors, {len(out_ids)} ids -> {out_dir}")
    print(f"generate agrees with loop: {agree}" + ("" if agree else f" (first diff at {first_diff})"))
    print(f"B5 -> B6 self-check, max abs err per layer: {selfcheck}")
    if not args.smoke:
        assert abs(meta["softmax_scale"] - common.SOFTMAX_SCALE) < 1e-12, meta["softmax_scale"]
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=common.MODEL)
    ap.add_argument("--prompt", default=str(common.PROMPT_IDS))
    ap.add_argument("--out", default=str(common.REF_DIR))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=common.N_STEPS)
    ap.add_argument("--smoke", action="store_true", help="tiny random model, exercises every path")
    args = ap.parse_args()
    if args.smoke and args.out == str(common.REF_DIR):
        args.out = str(common.ROOT / "harness/ref_smoke")
    run(args)


if __name__ == "__main__":
    main()
