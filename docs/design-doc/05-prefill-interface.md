# 05 - Prefill to Fleet-decode interface

The task requires the design to define the interface between the reference
prefill and the Fleet decode path, including the KV-cache layout, and to
exclude any one-time conversion from measured decode latency. This file is
that definition.

## Summary

| | |
|---|---|
| Prefill | HF `DeepseekV2ForCausalLM` from `modeling_deepseek.py`, BF16, `trust_remote_code=True`, on the same GPU |
| Prompt | 1,024 token ids, committed to the repository as `prompt_ids.json` |
| What the reference produces that we use | per layer, the output of `kv_a_proj_with_mqa` for all 1,024 prompt positions |
| What Fleet needs | per layer, `c_kv [1056, 512]` and `k_pe [1056, 64]` with rows 0..1022 filled |
| Conversion | none: the captured values are the exact quantities, after the model's own `kv_a_layernorm` and RoPE are applied to them |
| Hand-over point | position 1023: the Fleet path consumes prompt token 1023 as its first decode input (`00-decisions.md` D15) |
| Timed window | from `mpk()` entry to return; capture and upload are before it |

## Why capture rather than convert

The reference caches decompressed keys and values (`key_states [1, 16, S, 192]`,
`value_states [1, 16, S, 128]`, `modeling_deepseek.py:862-867`) and discards
the latent it computed on the way (`:831-856`). Recovering the latent from
the decompressed cache means inverting `W_UK` `[2048, 512]`, a least-squares
problem, not an exact inverse. The latent itself is a module output, so a
forward hook returns it exactly (`docs/deepseek-v2-lite/02-mla.md`).

## The capture, step by step

```python
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             trust_remote_code=True).cuda().eval()
ids = torch.tensor(json.load(open("prompt_ids.json")))[None].cuda()   # [1, 1024]
assert ids.shape[1] == 1024

kva = {}                       # layer -> [1, 1024, 576] BF16, output of kv_a_proj_with_mqa
for l, layer in enumerate(model.model.layers):
    layer.self_attn.kv_a_proj_with_mqa.register_forward_hook(
        lambda m, i, o, l=l: kva.__setitem__(l, o.detach()))

with torch.inference_mode():
    out = model(ids, use_cache=True)                        # the prefill; also gives logits[:, -1]
    # reference continuation for the oracle (07-correctness.md), same call chain as HF generate:
    ref = model.generate(ids, max_new_tokens=32, do_sample=False, eos_token_id=None,
                         pad_token_id=model.config.eos_token_id)

S_MAX = 1056
c_kv = torch.zeros(27, S_MAX, 512, dtype=torch.bfloat16, device="cuda")
k_pe = torch.zeros(27, S_MAX,  64, dtype=torch.bfloat16, device="cuda")
pos = torch.arange(1024, device="cuda")[None]
cos, sin = model.model.layers[0].self_attn.rotary_emb(kva[0], seq_len=S_MAX)   # [S_MAX, 64] each, BF16
for l, layer in enumerate(model.model.layers):
    c, kpe = kva[l].split([512, 64], dim=-1)               # [1, 1024, 512], [1, 1024, 64]
    c = layer.self_attn.kv_a_layernorm(c)                  # the model's own module, eps 1e-6
    kpe = kpe.view(1, 1024, 1, 64).transpose(1, 2)         # [1, 1, 1024, 64], the reference's shape
    _, kpe = apply_rotary_pos_emb(kpe, kpe, cos, sin, pos) # the model's own function
    c_kv[l, :1023] = c[0, :1023]
    k_pe[l, :1023] = kpe[0, 0, :1023]
```

Three properties of this capture:

1. **Exactness.** `kv_a_layernorm` and `apply_rotary_pos_emb` are the model's
   own module and function, applied to the model's own intermediate, in the
   model's own dtype. The stored rows are bit-identical to what the reference
   computed for those positions during its prefill.
2. **Row 1023 is deliberately left for Fleet.** The reference computed it;
   we keep it aside as `ref_row_1023[l] = (c[0, 1023], kpe[0, 0, 1023])` and
   compare it with what `mla_prep` writes in iteration 0. That is boundary
   B3/B4 at every layer for free (`07-correctness.md`).
3. **The RoPE tables** are the model's `rotary_emb` output, which for this
   checkpoint carries a table multiplier of exactly 1.0 (`mscale ==
   mscale_all_dim`, `modeling_deepseek.py:316-326`), and the softmax scale
   `0.1147213867929261` is the model's own `self.softmax_scale`
   (`:745-751`); `mla_prep` and `mla_attend` take both as inputs rather than
   recomputing them.

`apply_rotary_pos_emb` in this model interleaves the 64 RoPE dimensions
(`x.view(b, h, s, d // 2, 2).transpose(4, 3)` before the rotation,
`modeling_deepseek.py`, `apply_rotary_pos_emb`); `mla_prep` reproduces the
same permutation on `q_pe` and on the new `k_pe` row so that decode-time
positions rotate identically to the captured ones. This is the one place
where "use the model's own function" cannot be followed inside a kernel, and
it is checked by the row-1023 comparison above.

## What is handed to the Fleet path

| Tensor | Shape, dtype | Source | Attached as |
|---|---|---|---|
| `tokens` | `[1, 1056]` int64 | prompt ids in `[0, 1024)`, zeros after | meta tensor |
| `c_kv[l]`, `k_pe[l]` for `l` in 0..26 | `[1056, 512]`, `[1056, 64]` BF16 | capture above, rows 0..1022 | `attach_input(..., name=f"c_kv_{l}")`, `f"k_pe_{l}"` |
| `cos`, `sin` | `[1056, 64]` BF16 | `rotary_emb` | `attach_input` |
| weights | `04-memory-plan.md` | loader | `attach_input` |
| `step = 1022`, `new_token_nums = 1`, `qo_indptr = [0, 1]` | | host | meta tensors, written after `compile()` |

Positions: iteration `i` runs at `step = 1023 + i`, appends cache row
`step`, rotates with `cos[step]`, `sin[step]`, attends over rows
`[0, step]`, and writes `tokens[step + 1]`. Iteration 0 therefore attends
over 1,024 positions (1,023 captured plus its own), which is the "1,024-token
input context" of the task at the first decode step.

## What the reference is asked to produce for the oracle

Same process, same model object, before the hooks are removed:

- `ref_tokens = ref[0, 1024:1056]`, 32 ids, committed as `ref_output_ids.json`.
- Per-boundary tensors for layer 0 and layer 1 at decode step 0, captured by
  hooks on the same modules during a single `model(ids[:, :1024])` forward
  followed by `model(ids_1024, past_key_values=...)` for one step; saved as
  `.safetensors` (`07-correctness.md` lists the boundaries).
- The BF16 noise-floor calibration, from two orderings of the same
  computation (`07-correctness.md`).

`generate` with `do_sample=False` and `eos_token_id=None` is greedy and never
stops early, which matches the Fleet path's `eos = -1`. The shipped
`generation_config.json` defaults to sampling at temperature 0.3
(`docs/deepseek-v2-lite/01-config.md`); the explicit arguments override it.

## What is timed, and what is not

| Phase | In the timed window? | How measured |
|---|---|---|
| model load, weight packing, upload | no | wall clock, reported separately |
| reference prefill and `generate` | no | not part of the Fleet path |
| cache capture, `kv_a_layernorm`, RoPE, copy into `c_kv`/`k_pe` | **no**: this is the one-time conversion the task says to exclude | wall clock, reported for completeness |
| `mpk.compile()`, `init_kernel` | no | once per build |
| `mpk()`: `prepare_kernel` + 32 iterations | **yes** | `[FWD_PASS]` per iteration; event timing buffer; host wall clock around `mpk()` as the outer check (`09-expected-performance.md`) |
| reading `tokens[1024:1056]` back | no | |

The capture is not free (27 x 1,024 x 576 elements through a norm and a
rotation, about 16 M elements, milliseconds on the GPU), and it is not
decode work; it is the price of the reference storing a different
representation, exactly the case the task anticipates.

## Prefill validity conditions

- The reference must be run with the **same** prompt ids the Fleet path is
  given; the ids file is the single source, the text is never re-tokenized.
- The reference must be run in BF16 on the GPU; the capture's rounding is
  then the reference's rounding.
- Positions 0..1022 must not be recomputed by the Fleet path (they are not:
  `mla_prep` writes only row `step`).
- The prompt must be 1,024 tokens exactly; `S_MAX = 1056` and the split
  count 33 are derived from it and are compile-time constants of the graph.

## Alternatives considered

| Alternative | Why not |
|---|---|
| Fleet prefills the prompt itself (offline mode, 8 tokens per iteration through the decode graph) | 128 iterations of a graph tuned for one token; prefill is out of scope and would be slow and unvalidated; offline mode also ignores a preset `step` (`docs/fleet/04-repo-map.md`) |
| Recompute the latent from `output_hidden_states` | equivalent, but reruns 27 projections the hooks already observed |
| Invert the decompressed cache | not exact |
| Interleaved `[S][576]` cache | the two halves feed different matmuls; split arrays keep both reads coalesced (`00-decisions.md` D5) |
| Hand over at position 1024 (Fleet produces 31 tokens; the first comes from the prefill logits) | then one of the 32 reported tokens is not the Fleet path's, and iteration 0 would attend over 1,025 positions; D15 keeps all 32 on the Fleet path |
