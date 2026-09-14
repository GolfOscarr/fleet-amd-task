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
| `calibrate.py` | the BF16 floor: the reference against itself under other accumulation orders (batch padded to 2 rows; `--cpu`), per boundary and per class; writes `calibration.json` once | `python harness/calibrate.py --device cuda` after `run_reference.py`; `--smoke` anywhere |
| `route_analysis.py` | consecutive-step expert overlap, distinct experts, usage entropy per MoE layer from `ref_route_log.json` | `python harness/route_analysis.py` |
| `run_fleet.py` | packs the weights, builds the (optionally truncated) graph, compiles, sets the meta tensors, runs `mpk()`, dumps every recoverable boundary, ids, route log, timing | `python harness/run_fleet.py --layers 2 --model-dir <snapshot> [--stop-after L1.o_proj] [--head] [--iters K]` (GPU) |
| `measure.py` | per-iteration latency (`[FWD_PASS]`, event timing, wall clock), per-operator time, rocprofv3 traffic and launches; `metrics.json` and the predicted-versus-measured table | `python harness/measure.py --run harness/fleet_out/<name> [--pmc p.csv] [--kernel-trace k.csv]` |
| `tests/` | the smoke reference run's artifacts, `compare.py` on synthetic pairs, `numpy_ref` against the tiny model's own modules, calibration and route analysis on the smoke artifacts, the boundary mapping and the measurement parsers | `.venv/bin/python -m pytest harness/tests fleet/tests -q` |

The Fleet side lives in `fleet/`: `pack_weights.py` (the packed tensors of
`04-memory-plan.md`), `graph_plan.py` and `build_graph.py` (the operator
list of `02-task-graph.md` and the `mpk.*` calls, with a local `--dry-run`),
`tasks/mi300/` (the new kernels), `patches/` (the gfx942 build patch and the
task-registration glue).

## Milestone runs (the protocol of `07-correctness.md`)

```
# M1: one reused operator
python harness/run_fleet.py --layers 1 --model-dir $SNAP --stop-after L0.qkva
python harness/compare.py --fleet harness/fleet_out/L1_it1_L0.qkva
# M2: layer 1; two runs cover B1-B13 (x_res and h are overwritten in place)
python harness/run_fleet.py --layers 2 --model-dir $SNAP --stop-after L1.o_proj
python harness/run_fleet.py --layers 2 --model-dir $SNAP
python harness/compare.py --fleet harness/fleet_out/L2_it1_L1.o_proj
python harness/compare.py --fleet harness/fleet_out/L2_it1
# M4: the full graph, 32 iterations
python harness/run_fleet.py --layers 27 --head --iters 32 --model-dir $SNAP --event-timing
python harness/compare.py --fleet harness/fleet_out/L27_head_it32
python harness/measure.py --run harness/fleet_out/L27_head_it32
```

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
