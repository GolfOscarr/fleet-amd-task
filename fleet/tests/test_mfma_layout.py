"""The operand layout of v_mfma_f32_16x16x16_bf16 on gfx942, emulated (O7,
docs/gpu-experiments/03-acceleration/03-local-preparation.md).

One instruction multiplies a 16 x 16 A (M x K) by a 16 x 16 B (K x N) into a
16 x 16 FP32 D held across the 64 lanes, four values per lane:

    lane l holds A[l % 16][4 (l // 16) + i], B[4 (l // 16) + i][l % 16] and
    D[4 (l // 16) + i][l % 16] for i in 0..3

(CK's WarpGemmAttributeMfmaImplBf16Bf16F32M16N16K16: kAMLane = kBNLane = 16,
kABKLane = 4, kABKPerLane = 4; kCMLane = 4, kCNLane = 16, kCM1PerLane = 4;
the runtime's own tasks/mi300/ck_tile/linear.cuh compute_gemm_mfma uses the
same arithmetic). The functions below are the index formulas of
fleet/tasks/mi300/mla_attend_mfma_mi300.cuh, so a slip in the lane
arithmetic fails here before the machine sees it; the suite on the VM
verifies the instruction itself.
"""
import numpy as np
import pytest

LANES = 64


def mfma_16x16x16(a_lanes, b_lanes, c_lanes):
    """One instruction: per-lane fragments [64, 4] in, the accumulator [64, 4] out."""
    A = np.zeros((16, 16), np.float64)
    B = np.zeros((16, 16), np.float64)
    for l in range(LANES):
        for i in range(4):
            A[l % 16, 4 * (l // 16) + i] = a_lanes[l, i]
            B[4 * (l // 16) + i, l % 16] = b_lanes[l, i]
    D = A @ B
    out = np.array(c_lanes, np.float64)
    for l in range(LANES):
        for i in range(4):
            out[l, i] += D[4 * (l // 16) + i, l % 16]
    return out


def d_to_matrix(d_lanes):
    D = np.zeros((16, 16), np.float64)
    for l in range(LANES):
        for i in range(4):
            D[4 * (l // 16) + i, l % 16] = d_lanes[l, i]
    return D


# --- the kernel's fragments -------------------------------------------------

def score_fragments(q_rows, tile_rows, step):
    """Score block, K step `step` of 16: A[h][k] = q[h][16 step + k], B[k][p] = tile[p][16 step + k].
    Both fragments are four consecutive elements of a row (one ds_read_b64 each)."""
    a = np.zeros((LANES, 4)); b = np.zeros((LANES, 4))
    for l in range(LANES):
        k0 = 16 * step + 4 * (l // 16)
        a[l] = q_rows[l % 16, k0:k0 + 4]
        b[l] = tile_rows[l % 16, k0:k0 + 4]
    return a, b


def pv_fragments(p_rows, tile_rows, cb):
    """p x V block `cb` of 16 columns: A[h][p] = P[h][p], B[p][c] = V[p][16 cb + c].
    A is four consecutive probabilities of a head; B is four rows at one column."""
    a = np.zeros((LANES, 4)); b = np.zeros((LANES, 4))
    for l in range(LANES):
        a[l] = p_rows[l % 16, 4 * (l // 16):4 * (l // 16) + 4]
        for i in range(4):
            b[l, i] = tile_rows[4 * (l // 16) + i, 16 * cb + l % 16]
    return a, b


def test_single_instruction_matches_the_plain_product():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((16, 16)); B = rng.standard_normal((16, 16))
    a = np.zeros((LANES, 4)); b = np.zeros((LANES, 4))
    for l in range(LANES):
        for i in range(4):
            a[l, i] = A[l % 16, 4 * (l // 16) + i]
            b[l, i] = B[4 * (l // 16) + i, l % 16]
    d = mfma_16x16x16(a, b, np.zeros((LANES, 4)))
    assert np.allclose(d_to_matrix(d), A @ B)


def test_score_block_over_36_steps_split_over_4_waves():
    """S[h][p] = sum_k Q[h][k] Kt[p][k] over K = 576 (512 of c_kv and 64 of k_pe stored
    contiguously per row): 36 steps of 16, wave w taking steps 9 w .. 9 w + 9, the four
    partial accumulators summed per (h, p) as the kernel's LDS reduction does."""
    rng = np.random.default_rng(1)
    Q = rng.standard_normal((16, 576)); Kt = rng.standard_normal((16, 576))
    partial = np.zeros((4, LANES, 4))
    for w in range(4):
        acc = np.zeros((LANES, 4))
        for step in range(9 * w, 9 * w + 9):
            a, b = score_fragments(Q, Kt, step)
            acc = mfma_16x16x16(a, b, acc)
        partial[w] = acc
    # the reduction: thread (h, p) sums the four waves' D[h][p]
    S = np.zeros((16, 16))
    for h in range(16):
        for p in range(16):
            lane, i = p + 16 * (h // 4), h % 4
            S[h, p] = partial[:, lane, i].sum()
    assert np.allclose(S, Q @ Kt.T)


def test_pv_block_columns_and_accumulator_placement():
    """O[h][c] = sum_p P[h][p] V[p][c] for 512 columns as 32 blocks, wave w owning blocks
    8 w .. 8 w + 8; lane l's accumulator [cb][i] is O[4 (l // 16) + i][16 cb + l % 16],
    the placement the epilogue writes to partials."""
    rng = np.random.default_rng(2)
    P = rng.standard_normal((16, 16)); V = rng.standard_normal((16, 576))
    O = np.zeros((16, 512))
    for w in range(4):
        for j in range(8):
            cb = 8 * w + j
            a, b = pv_fragments(P, V, cb)
            d = mfma_16x16x16(a, b, np.zeros((LANES, 4)))
            for l in range(LANES):
                for i in range(4):
                    O[4 * (l // 16) + i, 16 * cb + l % 16] = d[l, i]
    assert np.allclose(O, P @ V[:, :512])


def test_masked_rows_contribute_nothing_when_zero_filled():
    """A tail pass with fewer than 16 rows: the kernel zero-fills the missing tile rows and
    masks their probabilities to 0, so the block equals the product over the real rows."""
    rng = np.random.default_rng(3)
    rows = 5
    P = rng.standard_normal((16, 16)); P[:, rows:] = 0.0
    V = rng.standard_normal((16, 576)); V[rows:] = 0.0
    a, b = pv_fragments(P, V, 3)
    d = d_to_matrix(mfma_16x16x16(a, b, np.zeros((LANES, 4))))
    assert np.allclose(d, P[:, :rows] @ V[:rows, 48:64])


@pytest.mark.parametrize("stride", [584])
def test_lds_row_stride_is_16_byte_aligned_and_holds_the_row(stride):
    """The staged rows are [c_kv | k_pe] = 576 BF16 with a 16-byte-aligned stride so the
    cooperative ds_write_b128 and the ds_read_b64 fragments are aligned."""
    assert stride >= 576 and (stride * 2) % 16 == 0
    # the fragment offsets 16 step + 4 (l // 16) are multiples of 4 elements: 8-byte aligned
    for step in range(36):
        for l in range(LANES):
            assert ((16 * step + 4 * (l // 16)) * 2) % 8 == 0
