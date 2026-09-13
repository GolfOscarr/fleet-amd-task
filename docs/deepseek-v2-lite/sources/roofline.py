"""Reproduces every number in ../07-roofline.md from config.json alone."""
import json, math, pathlib
c = json.load(open(pathlib.Path(__file__).with_name("config.json")))
H=c["hidden_size"]; L=c["num_hidden_layers"]; nh=c["num_attention_heads"]
dc=c["kv_lora_rank"]; dn=c["qk_nope_head_dim"]; dr=c["qk_rope_head_dim"]; dv=c["v_head_dim"]
E=c["n_routed_experts"]; K=c["num_experts_per_tok"]; SH=c["n_shared_experts"]
mi=c["moe_intermediate_size"]; inter=c["intermediate_size"]; V=c["vocab_size"]
dense_layers=c["first_k_dense_replace"]; B=2; S=1024
MB=lambda x: x/1024**2

attn = (dn+dr)*nh*H + (dc+dr)*H + nh*(dn+dv)*dc + H*nh*dv
dense = 3*inter*H
gate, shared, one_exp = E*H, 3*(mi*SH)*H, 3*mi*H
moe_active = attn + gate + shared + K*one_exp
n_moe = L - dense_layers
weights = dense_layers*(attn+dense) + n_moe*moe_active + V*H
cache   = (dc+dr)*B*S*L
total   = weights*B + cache

print(f"attention/layer      {MB(attn*B):9.2f} MB")
print(f"dense MLP            {MB(dense*B):9.2f} MB")
print(f"one routed expert    {MB(one_exp*B):9.2f} MB")
print(f"MoE layer active     {MB(moe_active*B):9.2f} MB")
print(f"lm_head              {MB(V*H*B):9.2f} MB")
print(f"weights/token        {MB(weights*B):9.2f} MB")
print(f"MLA cache @{S}      {MB(cache):9.2f} MB")
print(f"TOTAL/token          {MB(total):9.2f} MB")
for bw,name in ((5.3e12,"peak"),):
    print(f"roofline TPOT ({name}) {total/bw*1e6:7.1f} us -> {bw/total:7.1f} tok/s")
ms = 0.1*c["rope_scaling"]["mscale_all_dim"]*math.log(c["rope_scaling"]["factor"])+1.0
print(f"softmax_scale        {(dn+dr)**-0.5*ms*ms!r}")
