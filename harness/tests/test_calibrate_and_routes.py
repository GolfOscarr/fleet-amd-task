"""calibrate.py --smoke and route_analysis.py on the smoke reference artifacts."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARNESS))
import common  # noqa: E402
import compare  # noqa: E402
import route_analysis  # noqa: E402


@pytest.fixture(scope="module")
def smoke_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("ref_smoke")
    subprocess.run([sys.executable, str(HARNESS / "run_reference.py"), "--smoke", "--steps", "4",
                    "--out", str(out)], check=True, capture_output=True, text=True)
    return out


def test_calibrate_smoke_writes_floors(smoke_dir):
    r = subprocess.run([sys.executable, str(HARNESS / "calibrate.py"), "--smoke", "--ref", str(smoke_dir),
                        "--out", str(smoke_dir / "calibration.json")],
                       check=True, capture_output=True, text=True)
    assert "exact checks hold in every run" in r.stdout
    cal = json.loads((smoke_dir / "calibration.json").read_text())
    assert set(cal["floor"]) == {k for k in common.CLASS_THRESHOLD if k != "exact"}
    assert all(v >= 0.0 for v in cal["floor"].values())
    assert cal["runs"] == ["B_batch2_row0", "C_cpu_1thread"]
    assert "L1.B7.x_res_attn" in cal["per_boundary"] and "L1.B9.topk_idx" in cal["exact_checks_hold"]
    # compare.py consumes it: thresholds become 4 x floor
    th, source = compare.thresholds(cal)
    assert "calibrated" in source
    for cls, v in cal["floor"].items():
        assert th[cls] == pytest.approx(4 * v)


def test_route_analysis_on_smoke_log(smoke_dir):
    log = json.loads((smoke_dir / "ref_route_log.json").read_text())
    s = route_analysis.analyze(log, n_experts=8)
    assert s["steps"] == 4 and s["moe_layers"] == 2 and s["topk"] == 2
    for pl in s["per_layer"]:
        assert 0.0 <= pl["mean_consecutive_overlap"] <= 1.0
        assert 1 <= pl["distinct_experts"] <= 8
        assert 0.0 <= pl["usage_entropy_bits"] <= pl["max_entropy_bits"]
    assert s["reading"]


def test_route_analysis_synthetic():
    same = [[{"idx": [1, 2, 3], "w": [0.3, 0.2, 0.1]}] for _ in range(5)]
    s = route_analysis.analyze(same, n_experts=64)
    assert s["mean_consecutive_overlap_all_layers"] == 1.0 and s["per_layer"][0]["distinct_experts"] == 3
    disjoint = [[{"idx": [3 * t, 3 * t + 1, 3 * t + 2], "w": [0.3, 0.2, 0.1]}] for t in range(5)]
    s = route_analysis.analyze(disjoint, n_experts=64)
    assert s["mean_consecutive_overlap_all_layers"] == 0.0 and s["per_layer"][0]["distinct_experts"] == 15
    assert "rarely pay" in s["reading"]
