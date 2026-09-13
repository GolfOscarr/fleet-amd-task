# Acceleration Techniques

What can actually make batch-1 decode faster on MI300X, filtered by the task's
constraints.

## Files

| File | Contents |
|---|---|
| `01-precision.md` | FP8 and weight-only quantization — the only lever that moves the roofline |
| `02-decode-parallelism.md` | Split-KV attention; why batch-1 decode otherwise uses 5% of the GPU |
| `03-kernel-craft.md` | Cache policy, MFMA-vs-VALU at M=1, wide loads, cross-lane reduction |
| `04-technique-ledger.md` | Every technique: lever, estimated gain, cost, in/out of scope |
| `99-open-questions.md` | |
| `sources/` | FP8 checkpoint config and index |

## The frame

From `../deepseek-v2-lite/07-roofline.md`: 4,705.9 MiB/token, **931 µs floor**,
arithmetic intensity ~0.5 FLOP/byte. Batch-1 decode is memory-bound with no
weight reuse. That leaves exactly two levers:

| Lever | What it does | Ceiling |
|---|---|---|
| **A. Read fewer bytes** | moves the roofline itself | FP8 halves it → **~479 µs** |
| **B. Waste less time not reading** | closes the gap to the roofline | bounded by the gap |

Fleet is entirely a lever-B technique, and — per `../fleet/README.md` — at
batch 1 its own measurements show **no** bandwidth reduction (HBM reads 0.98×).
So the largest single win available to this project is not Fleet. It is FP8.

## Out of scope — excluded by the task, listed so the exclusion is deliberate

| Technique | Why excluded |
|---|---|
| Continuous batching, paged attention for throughput | batch-1 only |
| Speculative / Medusa / EAGLE / n-gram decoding | explicitly excluded |
| Tensor, pipeline, or expert parallelism | one MI300X, no TP |
| Prefill optimization, chunked prefill | prefill out of scope |
| Cooperative weight tiling across batch (Fleet's bs≥32 win) | requires `m_tiles ≥ 2`, i.e. batch > 1 |

The last row matters: it is Fleet's headline result and it is unavailable to us
by construction.

## Headline

| Configuration | Traffic/token | Roofline TPOT | tok/s |
|---|---|---|---|
| BF16 (task baseline) | 4,705.9 MiB | 931 µs | 1,074 |
| FP8 weights, BF16 `lm_head` | 2,568.1 MiB | 508 µs | 1,968 |
| FP8 weights incl. `lm_head` | 2,422.1 MiB | 479 µs | 2,087 |

**FP8 is worth ~1.9× — more than everything else in this document combined.**
And an FP8 checkpoint for our exact model already exists.
