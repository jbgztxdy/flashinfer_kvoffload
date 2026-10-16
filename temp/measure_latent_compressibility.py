# Purpose: Test the premises behind lossless PCIe-reduction proposals on real MLA latents:
#          (1) cross-layer latent correlation (residual coding), (2) raw lossless compressibility.
# Usage: PYTHONPATH=. python temp/measure_latent_compressibility.py
import lzma
import sys
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL = "/mnt/data/mhy/model/deepseekv2-16b"
TOKENS = 2048

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
tok = AutoTokenizer.from_pretrained(MODEL)
text = Path("temp/mtp_overlap/docs/novel_victorian.txt").read_text(errors="ignore")
ids = tok(text, return_tensors="pt").input_ids[:, :TOKENS]

# The cached latent is kv_a_layernorm(kv_nope): capture it per layer.
latents = {}


def hook(idx):
    def fn(module, inp, out):
        latents[idx] = out.detach()[0].float()  # [tokens, 512]
    return fn


inner = model.model
for i, layer in enumerate(inner.layers):
    layer.self_attn.kv_a_layernorm.register_forward_hook(hook(i))
with torch.no_grad():
    hidden = inner.embed_tokens(ids)
    pos = torch.arange(ids.shape[1])[None]
    emb = inner.rotary_emb(hidden, position_ids=pos)
    for layer in inner.layers:
        hidden = layer(hidden, attention_mask=None, position_embeddings=emb)

L = len(latents)
print(f"captured {L} layers x {latents[0].shape} latents\n")

print("=== 1. 跨层 latent 相关性（残差编码的前提）===")
print(f"{'层对':>8} {'余弦相似':>9} {'|残差|/|原始|':>13} {'结论':>10}")
for gap in (1, 4):
    sims, ratios = [], []
    for i in range(L - gap):
        a, b = latents[i], latents[i + gap]
        sims.append(torch.cosine_similarity(a, b, dim=-1).mean().item())
        ratios.append((b - a).norm(dim=-1).mean().item() / b.norm(dim=-1).mean().item())
    s, r = sum(sims) / len(sims), sum(ratios) / len(ratios)
    verdict = "可压缩" if r < 0.7 else "无相关"
    print(f"  gap={gap:<3} {s:9.3f} {r:13.3f} {verdict:>10}")

# Same-layer adjacent-token correlation, for contrast.
sims = [torch.cosine_similarity(latents[i][:-1], latents[i][1:], dim=-1).mean().item() for i in range(L)]
print(f"  同层相邻 token 余弦相似: {sum(sims)/len(sims):.3f}")

print("\n=== 2. BF16 latent 的无损可压缩性 ===")
sample = torch.cat([latents[i] for i in range(0, L, 4)]).bfloat16().contiguous()
raw = sample.view(torch.uint8).numpy().tobytes()
print(f"  样本 {len(raw)/1e6:.1f} MB")
for name, fn in (("zlib-6", lambda b: zlib.compress(b, 6)),
                 ("zlib-9", lambda b: zlib.compress(b, 9)),
                 ("lzma", lambda b: lzma.compress(b, preset=1))):
    c = fn(raw)
    print(f"    {name:8} {len(raw)/len(c):5.2f}x")

u16 = sample.view(torch.int16).numpy()
import numpy as np  # noqa: E402
for name, arr in (("符号+指数(高9位)", (u16 >> 7) & 0x1FF), ("尾数(低7位)", u16 & 0x7F)):
    _, counts = np.unique(arr, return_counts=True)
    p = counts / counts.sum()
    ent = -(p * np.log2(p)).sum()
    bits = 9 if "指数" in name else 7
    print(f"  {name:16} 熵 {ent:5.2f} / {bits} bit  -> 最优压缩 {bits/ent:.2f}x")
