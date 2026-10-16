# Purpose: Check the batched Host kernel's staging list against Python dedup after a launch.
# Usage: PYTHONPATH=. python temp/hb_regs/check_unique.py --requests 8
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_glm52_host_batch import make_wrapper  # noqa: E402
from bench_glm52_host_batch_kernel import load_captures, make_mtp_indices  # noqa: E402
from bench_glm52_sparse_decode_hbm_host_direct import MixedHBMHost  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=3)
parser.add_argument("--requests", type=int, default=8)
parser.add_argument("--fraction", type=float, default=0.3)
args = parser.parse_args()
device = torch.device("cuda", args.device)
torch.cuda.set_device(device)
context = 65536
tokens = 2 * args.requests * context
allocation = MixedHBMHost(tokens, args.device)
cache = allocation.tensor(args.device)
captures = load_captures("temp/mtp_overlap/overlap_ctx65536_*.pt")
sel, _ = make_mtp_indices(args.requests, 8, context, 2048, [args.fraction], 42, captures, device, tokens // 2)
rows = args.requests * 8
q = torch.randn(rows, 64, 512, dtype=torch.bfloat16, device=device)
qpe = torch.randn(rows, 64, 64, dtype=torch.bfloat16, device=device)
out = torch.empty_like(q)
idx = sel[args.fraction].clone()
w = make_wrapper(rows, idx, torch.empty(64 * 2**20, dtype=torch.uint8, device=device), 2048)
host_pages = idx[idx >= tokens // 2].long()
expected = host_pages.unique().numel()
for launch in range(3):
    w.run(q, qpe, cache[..., :512], cache[..., 512:], out=out, host_page_start=tokens // 2)
    torch.cuda.synchronize()
    _, staging, host_slot, remap, unique_pages, state, *_ = w._host_batch_buffers
    staged = remap[remap < 0]
    slots = (-staged - 1).long()
    n = int(slots.max()) + 1 if slots.numel() else 0
    listed = unique_pages[:n].long()
    print(f"launch {launch}: expected unique {expected}, max staged slot+1 {n}, "
          f"distinct listed {listed.unique().numel()}, host_slot nonzero after {int((host_slot != 0).sum())}")
del w, cache
torch.cuda.synchronize()
allocation.close()
