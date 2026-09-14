"""Constants and naming shared by every harness script.

Boundary keys are the join between the reference side (run_reference.py)
and the Fleet side (run_fleet.py); compare.py pairs tensors by these keys.
Shapes are the checkpoint's (docs/deepseek-v2-lite/01-config.md); the
scripts read the live config where a value can be derived so that the
tiny smoke model runs through the same code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"
PROMPT_IDS = ROOT / "harness/prompt_ids.json"
REF_DIR = ROOT / "harness/ref"

N_PROMPT = 1024
N_STEPS = 32
S_MAX = N_PROMPT + N_STEPS          # 1056
HANDOVER = N_PROMPT - 1             # 1023: Fleet iteration 0 runs at step 1023 (D15)

SOFTMAX_SCALE = 0.1147213867929261  # 192^-0.5 * mscale^2 (01-execution-flow.md)

# Layers whose per-boundary tensors are captured at decode step 0.
BOUNDARY_LAYERS = (0, 1)

# Boundary classes and starting thresholds on rel_err (07-correctness.md).
# calibration.json, when present, replaces a class threshold by 4 x floor.
CLASS_THRESHOLD = {
    "norm": 1e-2,
    "rope": 1e-2,
    "gemv": 2e-2,
    "scores": 3e-2,
    "attention": 3e-2,
    "router": 2e-2,
    "expert": 3e-2,
    "layer": 5e-2,
    "logits": 5e-2,
    "exact": 0.0,
}
THRESHOLD_MULTIPLIER = 4.0

# (key suffix, boundary id, class). Per-layer keys are "L{l}.{suffix}";
# expert keys carry the expert id: "L{l}.B11.expert_{e}".
LAYER_BOUNDARIES = [
    ("B1.norm1", "B1", "norm"),
    ("B2.q", "B2", "gemv"),
    ("B3.c_kv", "B3", "norm"),
    ("B3.k_pe", "B3", "rope"),
    ("B4.q_pe", "B4", "rope"),
    ("B5.scores", "B5", "scores"),
    ("B6.attn", "B6", "attention"),
    ("B7.x_res_attn", "B7", "gemv"),
    ("B8.router_logits", "B8", "router"),
    ("B9.topk_idx", "B9", "exact"),
    ("B10.topk_w", "B10", "router"),
    ("B11.expert_*", "B11", "expert"),
    ("B12.shared", "B12", "expert"),
    ("B13.layer_out", "B13", "layer"),
]
HEAD_BOUNDARIES = [
    ("head.B14.norm", "B14", "norm"),
    ("head.B15.logits", "B15", "logits"),
    ("head.B16.token", "B16", "exact"),
]

MOE_ONLY = {"B8", "B9", "B10", "B11", "B12"}


def boundary_class(key: str) -> str:
    """Class of a boundary key, or None if the key is not a boundary."""
    if key.startswith("head."):
        for k, _, cls in HEAD_BOUNDARIES:
            if key == k:
                return cls
        return None
    if not key.startswith("L"):
        return None
    _, suffix = key.split(".", 1)
    for k, _, cls in LAYER_BOUNDARIES:
        if k.endswith("*"):
            if suffix.startswith(k[:-1]):
                return cls
        elif suffix == k:
            return cls
    return None


def boundary_id(key: str) -> str:
    if key.startswith("head."):
        return key.split(".")[1]
    return key.split(".")[1]
