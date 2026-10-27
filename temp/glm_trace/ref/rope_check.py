# Check our rope() against transformers' apply_rotary_pos_emb_interleave / apply_rotary_pos_emb.
import importlib.util, sys, torch
import tf_rope as m
ns = vars(m)
sys.path.insert(0, ".."); sys.argv = ["x"]
T, h, d = 5, 3, 64
pos = torch.arange(T).float()
inv = 1.0 / (8e6 ** (torch.arange(0, d, 2).float() / d))
f = pos[:, None] * inv[None]
cos, sin = f.cos(), f.sin()
emb_cos, emb_sin = torch.cat([cos, cos], -1), torch.cat([sin, sin], -1)
def rope(x, cos, sin, interleave):
    c, s = cos[:, None], sin[:, None]
    if interleave: x1, x2 = x[..., 0::2], x[..., 1::2]
    else: x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)
q = torch.randn(T, h, d); k = torch.randn(T, 1, d)
# transformers attention path: q [B,h,T,d], cos [B,T,d] unsqueeze(1)
qi, ki = ns["apply_rotary_pos_emb_interleave"](q.transpose(0, 1)[None], k.transpose(0, 1)[None], emb_cos[None], emb_sin[None])
print("interleave q", torch.allclose(qi[0].transpose(0, 1), rope(q, cos, sin, True), atol=1e-6), "k", torch.allclose(ki[0].transpose(0, 1), rope(k, cos, sin, True), atol=1e-6))
qn, kn = ns["apply_rotary_pos_emb"](q[None], k[None], emb_cos[None], emb_sin[None], unsqueeze_dim=2)
print("half-split q", torch.allclose(qn[0], rope(q, cos, sin, False), atol=1e-6))
