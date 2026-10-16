# Purpose: Show that native MLA load_kv wraps page offsets at 2^32 elements.
# Usage: PYTHONPATH=. python temp/check_mla_uint32_offset.py
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_glm52_decode import BatchMLAPagedAttentionWrapper  # noqa: E402  (points JIT at the source tree)

torch.manual_seed(0)
device = torch.device("cuda", 3)
torch.cuda.set_device(device)
pages = 8_000_000  # > 2**32 / 576 = 7,456,540
limit = 2**32 // 576
flat = torch.empty(pages * 576, dtype=torch.bfloat16, device=device)
for i in range(0, flat.numel(), 2**28):
    flat[i : i + 2**28].normal_()
cache = flat.view(pages, 1, 576)
topk = 2048
cases = {
    "all below limit": torch.randint(0, limit, (topk,)),
    "all above limit": torch.randint(limit, pages, (topk,)),
    "half above": torch.cat([torch.randint(0, limit, (topk // 2,)), torch.randint(limit, pages, (topk // 2,))]),
}
q = torch.randn(1, 64, 512, dtype=torch.bfloat16, device=device)
qpe = torch.randn(1, 64, 64, dtype=torch.bfloat16, device=device)
workspace = torch.empty(128 * 2**20, dtype=torch.uint8, device=device)


def attend(rows):
    s = (q[0].float() @ rows[:, :512].T + qpe[0].float() @ rows[:, 512:].T) / 16
    return s.softmax(-1) @ rows[:, :512]


for name, idx in cases.items():
    idx = idx.int().to(device)
    w = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    w.plan(torch.tensor([0, 1], dtype=torch.int32), torch.tensor([0, topk], dtype=torch.int32), idx,
           torch.tensor([topk], dtype=torch.int32), 64, 512, 64, 1, False, 1 / 16,
           q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    out = w.run(q, qpe, cache[..., :512], cache[..., 512:]).float()[0]
    intended = attend(cache[idx.long(), 0].float())
    wrapped_off = (idx.long() * 576) % 2**32  # uint32 product of indices[q] * ckv_stride_page
    wrapped = attend(flat[wrapped_off[:, None] + torch.arange(576, device=device)].float())
    print(f"{name:16s} max|out-intended|={(out - intended).abs().max():.4f} "
          f"max|out-wrapped|={(out - wrapped).abs().max():.4f}")
