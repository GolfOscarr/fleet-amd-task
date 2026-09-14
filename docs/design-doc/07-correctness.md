# 07 - Correctness methodology

The oracle, the boundaries, the thresholds and how they are calibrated, the
per-milestone protocol, and the harness that produces the evidence. The
methodology was set in `docs/deepseek-v2-lite/08-correctness.md`; this file
maps it onto the graph of `02-task-graph.md` and adds the harness.

## Oracle

HF `DeepseekV2ForCausalLM` (`modeling_deepseek.py` from the checkpoint),
BF16, on the MI300X, `trust_remote_code=True`, `generate(do_sample=False,
max_new_tokens=32, eos_token_id=None)`. The prompt is a committed file of
1,024 token ids; the text is never re-tokenized (`05-prefill-interface.md`).

Committed reference artifacts (produced by `run_reference.py`, `10-local-work.md`):

| File | Content |
|---|---|
| `prompt_ids.json` | 1,024 ids |
| `ref_output_ids.json` | the 32 greedy output ids |
| `ref_boundaries_step0.safetensors` | every boundary below for layers 0, 1, and the head, at decode step 0 |
| `ref_cache_row1023.safetensors` | `c_kv`, `k_pe` at position 1023 for all 27 layers |
| `ref_route_log.json` | top-6 expert ids and weights, 26 layers x 32 steps |
| `ref_hidden_per_layer_step0.safetensors` | `x_res` after every layer at step 0 (for the error-growth curve) |
| `calibration.json` | the BF16 noise floor per boundary class (below) |

## Boundaries, mapped to our tensors

Decode step 0 unless stated; `l` is the layer.

| # | Boundary | Reference tensor | Our tensor | Shape | Class |
|---|---|---|---|---|---|
| B1 | `input_layernorm` out | hook on `input_layernorm` | `h` after op A1 | `[2048]` | norm |
| B2 | `q_proj` out | hook on `q_proj` | `qkva[:3072]` | `[3072]` | GEMV |
| B3 | latent and rope key at the new position | `kv_a_layernorm` out; `k_pe` after RoPE (captured as `ref_cache_row1023`) | `c_kv[l][1023]`, `k_pe[l][1023]` written by `mla_prep` | `[512]`, `[64]` | norm / RoPE |
| B4 | post-RoPE `q_pe` | recomputed from the hook on `q_proj` with the model's `apply_rotary_pos_emb` | `q_pe` | `[16, 64]` | RoPE |
| B5 | attention scores, pre-softmax | recomputed from hooks: `softmax_scale x (q_nope . k_nope^T + q_pe . k_pe^T)` over the 1,024 positions | `mla_attend` in **debug mode** (one split, scores written to a debug buffer) | `[16, 1024]` | scores |
| B6 | attention output, pre-`o_proj` | hook on `o_proj` input | `attn` | `[2048]` | attention |
| B7 | `o_proj` out + residual | hook on `o_proj` plus the residual | `x_res` after A6 | `[2048]` | GEMV |
| B8 | router logits (FP32) | forward **pre**-hook on `mlp.gate` captures its input; logits recomputed as `F.linear(h.float(), W_gate.float())` (`MoEGate.forward` keeps `logits` local and returns only the top-k, `modeling_deepseek.py:424-426`, `:497`) | `logits_router` | `[64]` | router |
| B9 | **top-6 expert indices** | `MoEGate.forward` `topk_idx` | `mask[0:6]` | `[6]` as a set | **exact** |
| B10 | top-6 weights | `topk_weight` | `topk_w[0:6]` matched by index | `[6]` | router |
| B11 | each expert output | hook on `experts[e]` for the six selected | `out8[k]` for `k` in 0..5 | `[2048]` each | expert |
| B12 | shared-expert output | hook on `shared_experts` | `out8[6] + out8[7]` in FP32 | `[2048]` | expert |
| B13 | MoE block out + residual | decoder layer output | `x_res` after M5 | `[2048]` | layer |
| B14 | final norm out | hook on `model.norm` | `h` after H1 | `[2048]` | norm |
| B15 | logits | model output `logits[:, -1]` | `logits` | `[102400]` | logits |
| B16 | argmax token id | `ref_output_ids[0]` | `tokens[1024]` | scalar | **exact** |

B3 at every layer is free: `05-prefill-interface.md` keeps the reference's
row 1023 aside and the Fleet path recomputes it in iteration 0. It checks
`kv_a_layernorm`, the interleaved RoPE, and the append index in one
comparison per layer, and it is the first thing to look at when anything
downstream disagrees.

B5 needs the kernel to materialize scores it normally keeps in registers.
`mla_attend` gets a compile-time debug flag that, with one split covering
all positions, writes the `[16, S]` FP32 scores to a debug buffer before the
softmax. It is a validation build, not the measured one.

## Metrics and thresholds

Per boundary, on FP32 upcasts of both sides:

```
max_abs_err = max |a - b|
rel_err     = ||a - b||_2 / ||b||_2
cos_sim     = <a, b> / (||a|| ||b||)
```

| Class | Starting threshold on `rel_err` | Set by calibration to |
|---|---|---|
| norm, RoPE (B1, B3, B4, B14) | 1e-2 | 4 x the measured floor |
| GEMV (B2, B7) | 2e-2 | 4 x floor |
| scores (B5) | 3e-2 | 4 x floor; reassociation is the extra term here |
| attention (B6) | 3e-2 | 4 x floor |
| router logits and weights (B8, B10) | 2e-2 | 4 x floor |
| expert (B11, B12) | 3e-2 | 4 x floor |
| layer (B13), per layer | 5e-2 | 4 x floor, and the growth curve must be monotone and sub-linear in `l` |
| logits (B15) | 5e-2 | 4 x floor |
| **B9 expert indices** | **exact set equality** | - |
| **B16 output ids, all 32** | **exact** | - |

The "4 x" multiplier is the decision; the floor is a measurement. The
thresholds above are placeholders until `calibration.json` exists, and the
report states both the floor and the threshold next to every result.

### Calibration of the BF16 noise floor

Two runs of the reference that differ only in accumulation order, compared
at every boundary:

1. The same model on GPU versus CPU (`torch_dtype=bfloat16` both), same ids.
2. On GPU, the same forward with `torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction`
   toggled, or with the batch padded to 2 rows (different GEMM tiling), same ids.

The larger of the two per-boundary `rel_err` values is the floor for that
class. If a class's floor is above its starting threshold, the threshold
moves, not the floor. Recorded once, committed, never edited after a Fleet
result has been seen.

## Known legitimate sources of divergence

Documented, not eliminated (`docs/deepseek-v2-lite/08-correctness.md`):

1. Runtime reassociation (`q_nope @ W_UK`, then against `c_kv`) reorders
   the matmuls relative to the reference's `kv_b_proj` decompression.
   Accumulated in FP32; affects B5, B6 and everything after.
2. Expert accumulation order: the reference's `topk(sorted=False)` order is
   unspecified; our sum is in slot order. FP32 accumulation; affects B13.
3. The shared MLP as two experts: two FP32 sums of 1408 instead of one of
   2816; affects B12, B13.
4. Softmax in FP32 then BF16 cast: matched exactly (a choice, not noise).
5. Router in FP32: matched by construction (D7); B9 must not diverge, and
   if it does the run is wrong, not noisy.
6. Chiplet-parallel GEMVs: each output column is one worker's dot product
   over the full K in the CK pipeline; the summation order differs from
   PyTorch's, within the BF16 floor.
7. Score rounding versus score magnitude. The reference computes its
   scores with a BF16 matmul and rounds them to BF16 before the FP32
   softmax; at a score magnitude of 40 one BF16 ulp is 0.25, so any
   difference in summation order moves individual probabilities by
   percent. `harness/reassoc_check.py` (real attention shapes, random
   weights, 1,024 positions) measures the reassociated path at 2.0e-2 to
   2.7e-2 on B6 with scores up to 37, and the reference's own arithmetic
   in another summation order at 2.5e-2 to 3.0e-2 on the same inputs;
   B5 is 3.2e-3 in every case. Reassociation is therefore within the
   reference's own ordering noise, and the attention threshold is whatever
   the calibration says, not the starting 3e-2
   (`harness/results/reassoc_check.json`).

## Protocol per milestone

| Milestone | Graph run | Required evidence |
|---|---|---|
| M1, one operator | truncated graph: embed, A1, then the operator under test; one iteration (`max_seq_length = 1025`) | that operator's boundary within threshold on the reference's real input; plus 100 random inputs against a NumPy model of the kernel (`10-local-work.md`) for the four new kernels |
| **M2, layer 1** | truncated graph: embed, layer 0, layer 1; one iteration; layer 0's output is compared first | **B1-B13 for layer 1 within threshold, B9 exact**; B3 exact-class at layers 0 and 1; B13 for layer 0 within threshold (layer 0 is on the path, so it is validated too) |
| M3, N consecutive layers | truncated graph with layers 0..N-1; one iteration | B13 at every layer boundary; the error-growth curve against `ref_hidden_per_layer_step0`; B3 at every layer |
| M4, end to end | the full graph; 32 iterations | **all 32 ids identical to `ref_output_ids.json`**; B15 within threshold at step 0; `route_log` versus `ref_route_log` exact at all 26 x 32 entries |

The end-to-end check is binary. If the 32 ids match, every routing decision
in 832 (layer, step) pairs matched, which is stronger evidence than any
tolerance.

## Harness

| Script | Runs on | Does |
|---|---|---|
| `make_prompt.py` | anywhere | recomputes the 1,024 ids and compares them with the committed `prompt_ids.json` (`--write` only once) |
| `run_reference.py` | GPU (once); `--smoke` anywhere | loads HF, prefills 1,023 ids, runs the 32-step argmax loop with hooks (position 1023 first), cross-checks `generate`, writes every artifact in the table above and the cache capture of `05-prefill-interface.md` |
| `numpy_ref.py` | anywhere | the four new kernels' math in NumPy with explicit BF16 rounding; tested against the tiny model's own modules |
| `reassoc_check.py` | anywhere | the reassociation error at the real attention shapes on CPU (item 7 above) |
| `calibrate.py` | GPU | the two-run floor measurement; writes `calibration.json` |
| `run_fleet.py --layers N [--head] [--iters K] [--stop-after <op>] [--debug] [--debug-scores]` | GPU | builds the truncated or full graph, sets meta tensors, runs `mpk()`, dumps every boundary tensor to `fleet_boundaries.safetensors` and `tokens` to `fleet_output_ids.json` |
| `compare.py` | anywhere | pairs reference and Fleet tensors by boundary name, prints the three metrics, the floor, the threshold, and PASS/FAIL per boundary; exact-match checks for B9, B16 and the route log; writes `correctness_report.md` |
| `kernel_tests.py` | GPU | for each new kernel, 100 random inputs against `numpy_ref.py`; also `mla_attend` with 1 split versus 33 splits (isolates the merge) |

How Fleet-side boundaries are read: after `mpk()` returns, every
`new_tensor` holds the value from the **last** operator that wrote it in the
last iteration. A truncated one-iteration graph therefore leaves each
boundary tensor of its last layer in place; for layers before the last, the
graph is built with per-layer debug copies (`dbg_x_res[l]`, written by an
extra `identity`-style task after each layer, present only in `--debug`
builds) so one run yields the whole growth curve. The route log is always
on: the router writes its 8 ids and weights to `route_log[step - 1023][l]`
(`[32, 26, 8]`, 26 KiB), which costs nothing measurable and doubles as the
routing-distribution data for the placement question.

## Reporting

Every completed boundary appears in the final report as a row: boundary,
reference tensor, `max_abs_err`, `rel_err`, `cos_sim`, floor, threshold,
result. Exact-match rows show the ids. The report is generated by
`compare.py`, not written by hand, so a re-run on the provided machine
reproduces it.
