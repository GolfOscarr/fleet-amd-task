# 03 — Mixture of Experts

Layers 1–26 (26 of 27) are MoE. Layer 1 is our required milestone.

## Structure per MoE layer

| Component | Shape | Count | Bytes (BF16) |
|---|---|---|---|
| `mlp.gate.weight` (router) | `[64, 2048]` | 1 | 256 KB |
| `mlp.shared_experts.{gate,up}_proj` | `[2816, 2048]` | 2 | 22 MB |
| `mlp.shared_experts.down_proj` | `[2048, 2816]` | 1 | 11 MB |
| `mlp.experts.K.{gate,up}_proj` | `[1408, 2048]` | 2 × 64 | 11 MB/expert |
| `mlp.experts.K.down_proj` | `[2048, 1408]` | 1 × 64 | 5.5 MB/expert |

- **One routed expert = 16.5 MB.** Six of them = **99 MB per token per layer**.
- **Shared experts = 33 MB**, read every token unconditionally.
- All 64 routed experts = 1,056 MB per layer; the whole MoE layer including
  attention is 1,115.5 MB resident, of which only **158.5 MB is touched** per
  token.

The shared experts are stored **fused**: `n_shared_experts = 2` with
`moe_intermediate_size = 1408` becomes a single MLP of width 2816, not two of
1408. Confirmed by the `[2816, 2048]` shape. So there is one shared-expert
GEMM, not two.

## Routing (`MoEGate.forward`)

```python
logits = F.linear(hidden_states.float(), self.weight.float(), None)   # FP32
scores = logits.softmax(dim=-1, dtype=torch.float32)                  # FP32, 64-wide
topk_weight, topk_idx = torch.topk(scores, k=6, dim=-1, sorted=False)

if self.top_k > 1 and self.norm_topk_prob:      # norm_topk_prob is FALSE
    ...
else:
    topk_weight = topk_weight * self.routed_scaling_factor            # × 1.0
```

Four consequences, each a potential silent bug:

1. **The router runs in FP32**, both the matmul and the softmax, even though
   everything around it is BF16. The reference explicitly casts. A BF16 router
   can flip the top-6 selection near ties, and one different expert changes the
   output materially. **Match the FP32 router.**
2. **`norm_topk_prob: false`** → the six weights are raw softmax probabilities
   over 64 experts and **do not sum to 1**. Normalizing them "for cleanliness"
   is a correctness bug.
3. **`routed_scaling_factor: 1.0`** → the multiply is a no-op here, but keep it
   parameterized rather than dropping it.
4. **`topk_method: "greedy"` with `n_group: 1`, `topk_group: 1`** → plain top-6
   over all 64. The `group_limited_greedy` branch in the reference is dead code
   for this checkpoint. Do not implement it.

Also note `sorted=False` on the topk: expert order is unspecified. Since the
combination is a weighted sum, order does not affect the mathematical result —
but it **does** affect BF16 accumulation order, so our output can differ from
the reference in the last bits depending on the order we accumulate in. Worth
knowing when a tolerance is exceeded by a hair.

## Combination

```python
y = moe_infer(hidden_states, topk_idx, topk_weight)     # weighted sum of 6 experts
y = y + self.shared_experts(identity)                   # shared added AFTER
```

The shared-expert output is added to the routed sum, and the shared path reads
the **pre-MoE residual** (`identity`), the same input the router saw.

Each expert is SwiGLU: `down_proj(silu(gate_proj(x)) * up_proj(x))`.

## Batch-1 decode behaviour — why this is the hard part

With one token:

- The router picks **6 of 64** experts. Which six is **data-dependent** and not
  known until the router runs — so the expert weights cannot be prefetched
  before the gate completes. This is a genuine serialization point inside the
  layer, and it is the hardest dependency in the Fleet task graph.
- Each expert GEMM is `[1,2048] × [2048,1408]` — a **GEMV**. Arithmetic
  intensity is ~1 FLOP per 2 bytes. The MFMA units are irrelevant; this is a
  pure bandwidth streaming problem.
- 99 MB must be read from **scattered locations** in a 1,056 MB pool, chosen at
  the last moment.

### Implications for chiplet scheduling

This is where Fleet's Chiplet-task abstraction has to earn its place, and where
the design has real choices:

- **Expert-parallel across XCDs**: distribute the 6 chosen experts over the 8
  chiplets, one expert per XCD. Simple, balanced, but each expert's 16.5 MB
  exceeds one XCD's 4 MB L2, so it streams from HBM regardless — the placement
  buys little.
- **Split each expert across XCDs**: shard the 1408-wide intermediate dimension
  so all 8 chiplets participate in each expert. Better bandwidth utilization,
  more cross-XCD reduction traffic (and therefore more of the agent-scope
  writeback/invalidate from `../mi300x/03-memory-model.md`).
- **Static expert→XCD affinity**: pin experts to chiplets by index so a
  re-selected expert may still be warm in that XCD's L2. Only pays off if
  routing is temporally correlated across the 32 decode steps — measure before
  building. See Q4.

None of these is obviously right; the choice should follow measurement, and the
roofline in `07-roofline.md` says the ceiling either way is set by HBM.

### The serialization chain per MoE layer

```
RMSNorm → router GEMV (FP32) → softmax → top-6 → gather 6 expert addresses
        → 6 × [gate,up GEMV → SiLU×mul → down GEMV] → weighted sum
        → + shared-expert MLP (independent, can run concurrently with routing)
```

**The shared-expert path does not depend on the router.** It reads the same
input and can be issued immediately, overlapping the entire routing latency.
That is a free and obvious parallel edge in the task graph, and it should be in
the first version of the design.
