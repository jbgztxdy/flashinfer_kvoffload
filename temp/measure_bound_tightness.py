# Purpose: Decide whether a strictly lossless "skip provably negligible tokens" scheme can work.
#          Needs two things to hold on REAL data, not a Zipf assumption:
#            (1) the attention weight distribution within the kept top-k must be skewed enough
#            (2) the Cauchy-Schwarz bound s_j <= |q||k_j| must be TIGHT enough to prove it
#          Uses real MLA queries and latents from DeepSeek-V2-Lite.
# Usage: PYTHONPATH=. python temp/measure_bound_tightness.py
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/mnt/data/mhy/model/deepseekv2-16b"
CTX, TOPK = 16384, 2048
BF16_EPS = 2 ** -8

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
ids = tok(Path("temp/mtp_overlap/docs/novel_victorian.txt").read_text(errors="ignore"),
          return_tensors="pt").input_ids[:, :CTX]

cap = {}
def hook(i):
    def fn(m, inp, out):
        cap.setdefault(i, {})["latent"] = out.detach()[0].float()
    return fn
def qhook(i):
    def fn(m, inp, out):
        cap.setdefault(i, {})["q"] = out.detach()[0].float()
    return fn

inner = model.model
LAYERS = [5, 13, 21]
for i in LAYERS:
    inner.layers[i].self_attn.kv_a_layernorm.register_forward_hook(hook(i))
    inner.layers[i].self_attn.q_proj.register_forward_hook(qhook(i))
with torch.no_grad():
    h = inner.embed_tokens(ids)
    pos = torch.arange(CTX)[None]
    emb = inner.rotary_emb(h, position_ids=pos)
    for layer in inner.layers:
        h = layer(h, attention_mask=None, position_embeddings=emb)

cfg = model.config
qh, nope, rope = cfg.num_attention_heads, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim
print(f"model: {qh} heads, qk_nope {nope}, qk_rope {rope}, kv_lora {cfg.kv_lora_rank}\n")

stats = {"skew": [], "tight": [], "provable": [], "oracle": []}
for i in LAYERS:
    lat = cap[i]["latent"]                       # [ctx, 512]
    q = cap[i]["q"].view(CTX, qh, nope + rope)   # [ctx, heads, 192]
    # Absorbed form: fold W_UK into q so scores are q_eff . latent. W_UK is kv_b_proj's nope part.
    W = inner.layers[i].self_attn.kv_b_proj.weight.detach().float()   # [heads*(nope+v), 512]
    W = W.view(qh, -1, cfg.kv_lora_rank)[:, :nope, :]                # [heads, nope, 512]
    qn = q[:, :, :nope]                                              # [ctx, heads, nope]
    q_eff = torch.einsum("thn,hnc->thc", qn, W)                      # [ctx, heads, 512]

    for t in (CTX - 1, CTX - 513):                                   # a couple of query rows
        s = torch.einsum("hc,tc->ht", q_eff[t], lat[:t + 1]) / (nope + rope) ** 0.5
        top = s.topk(min(TOPK, t + 1), dim=-1)
        w = top.values.softmax(-1)                                   # [heads, topk]
        # (1) skew: fraction of kept tokens whose true weight is below the bf16 floor
        stats["oracle"].append((w < BF16_EPS * w.cumsum(-1)).float().mean().item())
        # (2) tightness of Cauchy-Schwarz: bound / actual, per selected token
        kn = lat[:t + 1].norm(dim=-1)                                # |k_j|
        for hh in range(0, qh, 16):
            idx = top.indices[hh]
            bound = q_eff[t, hh].norm() * kn[idx] / (nope + rope) ** 0.5
            actual = s[hh, idx]
            stats["tight"].append((bound / actual.clamp(min=1e-6)).median().item())
            # (3) provable: using the bound, how many can we prove negligible?
            m = actual.max()
            wb = (bound - m).exp()                                   # upper bound on exp(s-m)
            d_lo = (actual - m).exp().sum()                          # true denominator
            stats["provable"].append((wb / d_lo < BF16_EPS).float().mean().item())

print("=== (1) 真实权重分布的偏斜度（oracle，假装已知所有分数）===")
print(f"  top-2048 内权重低于 BF16 舍入下限的 token: {sum(stats['oracle'])/len(stats['oracle']):.1%}")
print("=== (2) Cauchy-Schwarz 上界的松紧度 ===")
print(f"  bound / actual 的中位数: {sum(stats['tight'])/len(stats['tight']):.1f}x")
print("=== (3) 用该上界实际能证明可跳过的比例 ===")
print(f"  {sum(stats['provable'])/len(stats['provable']):.1%}")
