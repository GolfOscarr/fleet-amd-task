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
modeling file does not import under 5.x, and under 4.57.1 (Fleet's own
pin) its cache calls fail (`DynamicCache.get_usable_length`). On the
MI300X box `env/setup.sh` therefore makes two venvs: `.venv` for the
reference side (`run_reference.py`, `calibrate.py`, `reassoc_check.py`,
`make_prompt.py`) and `.venv-fleet`, where the Fleet submodule is
installed with its own pins, for `run_fleet.py`, `kernel_tests.py`,
`compare.py` and `measure.py` (`env/requirements-fleet.txt`).

## Files

| File | What | Runs |
|---|---|---|
| `prompt_ids.json`, `prompt_meta.json` | the fixed 1,024-token prompt (BOS + `split_linear_tasks.py` from the pinned Fleet submodule) and its provenance | committed, never regenerated |
| `make_prompt.py` | recomputes the ids and compares them with the committed file | `python harness/make_prompt.py` |
| `common.py` | model name, positions (`HANDOVER = 1023`, `S_MAX = 1056`), boundary keys, class thresholds | imported |
| `run_reference.py` | the HF reference: prefill `ids[0:1023]`, 32-step argmax loop from position 1023 with hooks, `generate` cross-check, all `ref_*` artifacts and the latent-cache capture | `python harness/run_reference.py --device cuda` (GPU, 31 GB); `--smoke` for a tiny random model anywhere |
| `compare.py` | pairs `fleet_*` and `ref_*` by boundary key, metrics, thresholds (calibrated when `calibration.json` exists), exact checks, `correctness_report.md` | `python harness/compare.py --fleet <dir>` |
| `bitdiff.py` | pairs two runs' `fleet_boundaries.safetensors` by key, per key the element count, the count of differing elements, the max ULP distance and the max absolute difference, as a markdown table | `python harness/bitdiff.py <dir-a> <dir-b> [--out file.md]` |
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
# M2: layer 1; two runs cover B1-B4, B6-B13 (x_res and h are overwritten in place)
python harness/run_fleet.py --layers 2 --model-dir $SNAP --stop-after L1.o_proj
python harness/run_fleet.py --layers 2 --model-dir $SNAP
python harness/compare.py --fleet harness/fleet_out/L2_it1_L1.o_proj
python harness/compare.py --fleet harness/fleet_out/L2_it1
# B5 (scores): a debug build in which mla_attend also writes the scaled scores
python harness/run_fleet.py --layers 2 --model-dir $SNAP --stop-after L1.mla_attend --debug-scores
python harness/compare.py --fleet harness/fleet_out/L2_it1_L1.mla_attend_scores
# M4: the full graph, 32 iterations
python harness/run_fleet.py --layers 27 --head --iters 32 --model-dir $SNAP --event-timing
python harness/run_fleet.py --layers 27 --head --iters 32 --model-dir $SNAP --final   # round 4's finals' stack (F3 of docs/gpu-experiments/05-final): the thirteen flags and -DMPK_W2_CK_TILE for every flag not named; --no-event-timing, --no-nt-streams, --no-gemv-linears turn one off; the run name gains _final
python harness/compare.py --fleet harness/fleet_out/L27_head_it32
python harness/measure.py --run harness/fleet_out/L27_head_it32
```

## Boundary keys

`L{l}.B{n}.{name}` for layers 0 and 1, `head.B14.norm`, `head.B15.logits`,
`head.B16.token`; experts as `L{l}.B11.expert_{e}`. Both sides write the
same keys (`common.py`); auxiliary tensors (`L{l}.layer_in`, `o_proj_out`,
`norm2`, `gate_in`) are stored for calibration and ignored by `compare.py`.

## Bit-diff

`bitdiff.py <dir-a> <dir-b> [--out file.md]` loads `fleet_boundaries.safetensors`
from two run directories and reports, for every key present in both, the
element count, the count of differing elements, the max ULP distance and
the max absolute difference, as a markdown table sorted by key; a key
present on only one side gets a line below the table instead of a row.
BF16 and FP32 tensors compare their ULP distance as the raw bit patterns'
distance, read as 16-bit or 32-bit integers; integer tensors carry the
max absolute difference again, as their ULP column. It runs on the CPU
and always exits 0: it is the finer-grained story beside `compare.py`'s
tolerance check (L7 of `docs/gpu-experiments/04-kernels/05-local-preparation.md`,
I5 of `01-gemv-ideas.md`), not a check of its own. On the VM,
`env/session/queue.sh bitdiff <run-a> <run-b>` runs it on two
`harness/fleet_out/` directories under the fleet venv and writes
`$RECORD/bitdiff_<run-a>_<run-b>.md`.

## Positions

The reference prefills positions 0..1022 and runs position 1023 as a
one-token step: that step is Fleet iteration 0 (`step = 1023`), so its
boundary tensors are computed at M = 1 like the Fleet path's, and its
`kv_a_proj` output is `ref_cache_row1023`. Output token `i` is written to
`tokens[1024 + i]`.
