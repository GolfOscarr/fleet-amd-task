"""Regenerates ../07-achievable-bandwidth.md.

Gap 3: can 1 wave/SIMD saturate HBM?   Gap 2: what bandwidth is achievable?
"""
import json, pathlib

MB = 1024 ** 2
CU_TOTAL, CU_SCHED, SIMD_PER_CU, LANES = 304, 8, 4, 64
BYTES_PER_LANE = 16          # global_load_dwordx4
VMCNT_BITS = 6               # CDNA3 ISA: VMCNT is 6 bits
VMCNT_MAX = 2 ** VMCNT_BITS - 1

BW = {"theoretical (spec)": 5.3e12,
      "measured (BabelStream peak)": 4.3e12,
      "conservative (AMD Dot threshold)": 3.66e12}

AMD_THRESHOLDS_MBPS = {"Copy": 4177285, "Mul": 4067069, "Add": 3920853,
                       "Triad": 3885301, "Dot": 3660781}

# ---------------------------------------------------------------- Part 1
print("=== AMD acceptance thresholds vs 5.3 TB/s peak ===")
for k, v in AMD_THRESHOLDS_MBPS.items():
    print(f"  {k:6s} {v:>10,} MB/s = {v*1e6/1e12:5.2f} TB/s = {v*1e6/5.3e12*100:5.1f}%")

cfg = pathlib.Path(__file__).parents[2] / "deepseek-v2-lite/sources/config.json"
c = json.load(open(cfg))
H, L, nh = c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"]
dc, dn, dr, dv = (c["kv_lora_rank"], c["qk_nope_head_dim"],
                  c["qk_rope_head_dim"], c["v_head_dim"])
E, K, SH = c["n_routed_experts"], c["num_experts_per_tok"], c["n_shared_experts"]
mi, inter, V = c["moe_intermediate_size"], c["intermediate_size"], c["vocab_size"]
nd = c["first_k_dense_replace"]; nmoe = L - nd; S = 1024

attn = (dn+dr)*nh*H + (dc+dr)*H + nh*(dn+dv)*dc + H*nh*dv
moe = attn + E*H + 3*(mi*SH)*H + K*3*mi*H
bf16 = (nd*(attn + 3*inter*H) + nmoe*moe + V*H) * 2 / MB + (dc+dr)*2*S*L / MB

print("\n=== roofline band ===")
for name, mib in (("BF16 full decode", bf16),
                  ("layer 1 (MoE)", (moe*2 + (dc+dr)*2*S)/MB),
                  ("27 layers, no head", bf16 - V*H*2/MB)):
    cells = "".join(f" {mib*MB/bw*1e6:9.1f}" for bw in BW.values())
    print(f"  {name:22s} {mib:8.1f} MiB |{cells}  us")
print(f"  {'BF16 tokens/s':22s} {'':8s}     " +
      "".join(f" {1/(bf16*MB/bw):9.0f}" for bw in BW.values()))

# ---------------------------------------------------------------- Part 2
waves = (CU_TOTAL - CU_SCHED) * SIMD_PER_CU
per_load = waves * LANES * BYTES_PER_LANE
print(f"\n=== memory-level parallelism, 1 wave/SIMD ===")
print(f"  worker waves {waves}, {per_load/MB:.2f} MiB in flight per outstanding load")
print(f"  VMCNT is {VMCNT_BITS} bits -> max {VMCNT_MAX} outstanding loads per wave")
print(f"  {'bandwidth':34s}" + "".join(f"  L={l*1e9:>6.0f}ns" for l in (250e-9, 500e-9, 1e-6, 2e-6)))
for name, bw in BW.items():
    print(f"  {name:34s}" + "".join(f"  N={bw*l/per_load:8.1f}" for l in (250e-9, 500e-9, 1e-6, 2e-6)))
print(f"  max tolerable latency at VMCNT={VMCNT_MAX}, 4.3 TB/s: "
      f"{VMCNT_MAX*per_load/4.3e12*1e6:.1f} us")
print("\n  prefetch depth -> VGPR cost (dwordx4 = 4 VGPRs each, of 512):")
for n in (2, 4, 8, 16):
    print(f"    N={n:2d} -> {n*4:3d} VGPRs ({n*4/512*100:.1f}%)")
