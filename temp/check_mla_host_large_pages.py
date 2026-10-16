# Purpose: Check native and fused Host MLA paths with page ids beyond the old 32-bit offset limit.
# Usage: PYTHONPATH=. python temp/check_mla_host_large_pages.py --context 33554432
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_glm52_host_comparison import make_wrapper  # noqa: E402
from bench_glm52_sparse_decode_hbm_host_direct import MixedHBMHost  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=3)
parser.add_argument("--context", type=int, default=16 * 2**20)
args = parser.parse_args()
torch.manual_seed(0)
device = torch.device("cuda", args.device)
torch.cuda.set_device(device)
tokens = args.context
half = tokens // 2
allocation = MixedHBMHost(tokens, args.device)
cache = allocation.tensor(args.device)
for begin in range(0, tokens, 2**20):
    cache[begin : begin + 2**20].normal_()
limit = 2**32 // 576
q = torch.randn(1, 64, 512, dtype=torch.bfloat16, device=device)
qpe = torch.randn(1, 64, 64, dtype=torch.bfloat16, device=device)
out = torch.empty_like(q)
indices = torch.empty(2048, dtype=torch.int32, device=device)
wrapper = make_wrapper(indices, torch.empty(32 * 2**20, dtype=torch.uint8, device=device), 2048)
worst = 0.0
for fraction in (0.0, 0.1, 0.25, 0.3, 1.0):
    host_rows = round(fraction * 2048)
    # Local rows straddle the old limit when the HBM half is large enough.
    local = torch.randperm(half, device=device)[: 2048 - host_rows]
    local[: len(local) // 2] = torch.randint(min(limit, half - 1), half, (len(local) // 2,), device=device)
    local = local.unique()
    while len(local) < 2048 - host_rows:
        local = torch.cat([local, torch.randint(0, half, (1,), device=device)]).unique()
    host = half + torch.randperm(half, device=device)[:host_rows]
    selected = torch.cat([local, host])[torch.randperm(2048, device=device)].int()
    indices.copy_(selected)
    rows = cache[selected.long(), 0].float()
    scores = (q[0].float() @ rows[:, :512].T + qpe[0].float() @ rows[:, 512:].T) / 16
    expected = scores.softmax(-1) @ rows[:, :512]
    for mode, boundary in (("native", None), ("fused_host", half)):
        wrapper.run(q, qpe, cache[..., :512], cache[..., 512:], out=out, host_page_start=boundary)
        error = (out[0].float() - expected).abs().max().item()
        worst = max(worst, error)
        above = int((selected.long() >= limit).sum())
        print(f"host={fraction:.0%} {mode:10s} rows_above_old_limit={above} max_abs_error={error:.4f}", flush=True)
print(f"pages={tokens} old_limit={limit} worst={worst:.4f}")
assert worst < 1e-2
del cache, wrapper
torch.cuda.synchronize()
allocation.close()
