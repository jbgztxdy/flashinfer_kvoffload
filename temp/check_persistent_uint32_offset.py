# Purpose: Check that BatchAttention (persistent.cuh) reads pages beyond 2^32 KV elements correctly.
# Usage: PYTHONPATH=. python temp/check_persistent_uint32_offset.py
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_glm52_decode  # noqa: E402,F401  (points JIT at the source tree)
from flashinfer.attention import BatchAttention  # noqa: E402

torch.manual_seed(0)
device = torch.device("cuda", 3)
torch.cuda.set_device(device)
heads, dim = 8, 128
per_page = heads * dim  # page_size 1, NHD
limit = 2**32 // per_page
pages = limit + 600_000
cache = torch.empty(pages, 1, heads, dim, dtype=torch.bfloat16, device=device)
flat = cache.view(-1)
for i in range(0, flat.numel(), 2**28):
    flat[i : i + 2**28].normal_()
kv_len, qo_heads = 2048, 32
q = torch.randn(1, qo_heads, dim, dtype=torch.bfloat16, device=device)


def attend(rows):  # rows: (kv_len, heads, dim); K and V alias the same cache
    k = rows.float().repeat_interleave(qo_heads // heads, dim=1)
    s = torch.einsum("hd,lhd->hl", q[0].float(), k) / dim**0.5
    return torch.einsum("hl,lhd->hd", s.softmax(-1), k)


worst = 0.0
for name, idx in {
    "all below limit": torch.randint(0, limit, (kv_len,)),
    "all above limit": torch.randint(limit, pages, (kv_len,)),
    "half above": torch.cat([torch.randint(0, limit, (kv_len // 2,)), torch.randint(limit, pages, (kv_len // 2,))]),
}.items():
    idx = idx.int().to(device)
    attention = BatchAttention(kv_layout="NHD", device=str(device))
    attention.plan(
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([0, kv_len], dtype=torch.int32, device=device),
        idx, torch.tensor([kv_len], dtype=torch.int32, device=device),
        qo_heads, heads, dim, dim, 1, causal=False,
        q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
    )
    out, _ = attention.run(q, (cache, cache))
    intended = attend(cache[idx.long(), 0])
    wrapped_page_offset = (idx.long() * per_page) % 2**32
    wrapped = attend(flat[wrapped_page_offset[:, None] + torch.arange(per_page, device=device)].view(kv_len, heads, dim))
    error = (out[0].float() - intended).abs().max().item()
    worst = max(worst, error)
    print(f"{name:16s} max|out-intended|={error:.4f} max|out-wrapped|={(out[0].float() - wrapped).abs().max():.4f}", flush=True)
print(f"pages={pages} old_limit={limit} worst={worst:.4f}")
