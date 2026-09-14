# Harness

The correctness and measurement scripts of `docs/design-doc/07-correctness.md`
and `10-local-work.md`. Everything here runs without the GPU except the
reference run on the real checkpoint and the calibration.

## Setup

```
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r env/requirements.txt
```

transformers is pinned to 4.46.3: the checkpoint's `trust_remote_code`
modeling file does not import under 5.x.

## Files

| File | What | Runs |
|---|---|---|
| `prompt_ids.json`, `prompt_meta.json` | the fixed 1,024-token prompt (BOS + `split_linear_tasks.py` from the pinned Fleet submodule) and its provenance | committed, never regenerated |
| `make_prompt.py` | recomputes the ids and compares them with the committed file | `python harness/make_prompt.py` |
| `common.py` | model name, positions (`HANDOVER = 1023`, `S_MAX = 1056`), boundary keys, class thresholds | imported |
| `run_reference.py` | the HF reference: prefill `ids[0:1023]`, 32-step argmax loop from position 1023 with hooks, `generate` cross-check, all `ref_*` artifacts and the latent-cache capture | `python harness/run_reference.py --device cuda` (GPU, 31 GB); `--smoke` for a tiny random model anywhere |
| `compare.py` | pairs `fleet_*` and `ref_*` by boundary key, metrics, thresholds (calibrated when `calibration.json` exists), exact checks, `correctness_report.md` | `python harness/compare.py --fleet <dir>` |
| `numpy_ref.py` | NumPy math of `mla_prep`, `mla_attend`, `mla_merge_uv`, `moe_router` with explicit BF16 rounding; the spec the GPU kernels are tested against | imported |
| `reassoc_check.py` | reassociation error at the real attention shapes on CPU; result in `results/reassoc_check.json` | `python harness/reassoc_check.py` |
| `tests/` | 23 checks: the smoke reference run's artifacts, `compare.py` on synthetic pairs, `numpy_ref` against the tiny model's own modules | `.venv/bin/python -m pytest harness/tests fleet/tests -q` |

## Boundary keys

`L{l}.B{n}.{name}` for layers 0 and 1, `head.B14.norm`, `head.B15.logits`,
`head.B16.token`; experts as `L{l}.B11.expert_{e}`. Both sides write the
same keys (`common.py`); auxiliary tensors (`L{l}.layer_in`, `o_proj_out`,
`norm2`, `gate_in`) are stored for calibration and ignored by `compare.py`.

## Positions

The reference prefills positions 0..1022 and runs position 1023 as a
one-token step: that step is Fleet iteration 0 (`step = 1023`), so its
boundary tensors are computed at M = 1 like the Fleet path's, and its
`kv_a_proj` output is `ref_cache_row1023`. Output token `i` is written to
`tokens[1024 + i]`.
