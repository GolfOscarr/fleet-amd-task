"""bitdiff.py on two synthetic fleet_boundaries.safetensors reports (L7)."""
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

HARNESS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARNESS))
import bitdiff  # noqa: E402


def bump_bits(t: torch.Tensor, int_dtype, index: int, delta: int) -> torch.Tensor:
    """t with element `index`'s raw bit pattern moved by `delta` (an exact ULP count)."""
    bits = t.contiguous().view(int_dtype).clone()
    bits[index] += delta
    return bits.view(t.dtype)


def make_pair(tmp_path):
    a_dir, b_dir = tmp_path / "a", tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()

    bf_a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    bf_b = bump_bits(bf_a, torch.int16, 0, 1)          # element 0: exactly one ULP

    fp_a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    fp_b = bump_bits(fp_a, torch.int32, 1, 1_000_000)  # element 1: many ULPs

    int_a = torch.tensor([5, 6, 7], dtype=torch.int32)  # present in a only

    save_file({"bf_key": bf_a, "fp_key": fp_a, "int_key": int_a}, str(a_dir / "fleet_boundaries.safetensors"))
    save_file({"bf_key": bf_b, "fp_key": fp_b}, str(b_dir / "fleet_boundaries.safetensors"))
    return a_dir, b_dir, bf_a, bf_b, fp_a, fp_b, int_a


def row_cols(text, key):
    """Dtype, N, Differing, Max ULP, Max abs diff of a key's row (the Key column dropped)."""
    for line in text.splitlines():
        if line.startswith(f"| `{key}`"):
            return [c.strip() for c in line.split("|")[1:-1]][1:]
    raise AssertionError(f"no row for {key} in:\n{text}")


def test_bf16_key_differs_by_one_ulp(tmp_path):
    a_dir, b_dir, bf_a, bf_b, *_ = make_pair(tmp_path)
    text = bitdiff.run(a_dir, b_dir)
    dtype, n, differing, ulp, max_abs = row_cols(text, "bf_key")
    assert dtype == "bfloat16" and n == "3" and differing == "1" and ulp == "1"
    expected = (bf_b[0].double() - bf_a[0].double()).abs().item()
    assert abs(float(max_abs) - expected) < 1e-6


def test_fp32_key_differs_by_many_ulps(tmp_path):
    a_dir, b_dir, _, _, fp_a, fp_b, _ = make_pair(tmp_path)
    text = bitdiff.run(a_dir, b_dir)
    dtype, n, differing, ulp, max_abs = row_cols(text, "fp_key")
    assert dtype == "float32" and n == "3" and differing == "1" and ulp == "1000000"
    expected = (fp_b[1].double() - fp_a[1].double()).abs().item()
    assert abs(float(max_abs) - expected) / expected < 1e-2   # the table prints to 3 decimals


def test_key_present_on_only_one_side_is_a_line_not_a_row(tmp_path):
    a_dir, b_dir, *_ = make_pair(tmp_path)
    text = bitdiff.run(a_dir, b_dir)
    assert "`int_key`: only in a." in text
    assert not any(line.startswith("| `int_key`") for line in text.splitlines())


def test_identical_tensors_report_zero(tmp_path):
    a_dir, b_dir = tmp_path / "a2", tmp_path / "b2"
    a_dir.mkdir()
    b_dir.mkdir()
    t = {"k": torch.tensor([1.0, -2.0, 0.5], dtype=torch.bfloat16)}
    save_file(t, str(a_dir / "fleet_boundaries.safetensors"))
    save_file(t, str(b_dir / "fleet_boundaries.safetensors"))
    text = bitdiff.run(a_dir, b_dir)
    dtype, n, differing, ulp, max_abs = row_cols(text, "k")
    assert differing == "0" and ulp == "0" and float(max_abs) == 0.0


def test_int_tensor_ulp_column_is_the_max_absolute_difference(tmp_path):
    a_dir, b_dir = tmp_path / "a3", tmp_path / "b3"
    a_dir.mkdir()
    b_dir.mkdir()
    save_file({"ids": torch.tensor([1, 2, 3], dtype=torch.int64)}, str(a_dir / "fleet_boundaries.safetensors"))
    save_file({"ids": torch.tensor([1, 2, 9], dtype=torch.int64)}, str(b_dir / "fleet_boundaries.safetensors"))
    text = bitdiff.run(a_dir, b_dir)
    dtype, n, differing, ulp, max_abs = row_cols(text, "ids")
    assert differing == "1" and ulp == "6" and max_abs == "6.000"


def test_cli_out_flag_writes_the_file_and_exits_zero(tmp_path):
    a_dir, b_dir, *_ = make_pair(tmp_path)
    out = tmp_path / "report.md"
    r = subprocess.run([sys.executable, str(HARNESS / "bitdiff.py"), str(a_dir), str(b_dir), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert str(out) in r.stdout
    assert out.exists() and "# Bit-diff report" in out.read_text()
    assert "`int_key`: only in a." in out.read_text()
