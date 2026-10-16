# Purpose: Per-CTA phase timeline of the batched Host kernel (build with -DHB_TRACE edits applied).
# Usage: PYTHONPATH=. python temp/trace_host_batch.py --requests 1 --fraction 0.1
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_glm52_host_batch_kernel import load_captures, make_mtp_indices  # noqa: E402
from bench_glm52_host_batch import make_wrapper  # noqa: E402
from bench_glm52_sparse_decode_hbm_host_direct import MixedHBMHost  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=3)
parser.add_argument("--requests", type=int, default=1)
parser.add_argument("--fraction", type=float, default=0.1)
parser.add_argument("--fetch-ctas", type=int, default=16)
args = parser.parse_args()
device = torch.device("cuda", args.device)
torch.cuda.set_device(device)
context = 65536
tokens = 2 * args.requests * context
allocation = MixedHBMHost(tokens, args.device)
cache = allocation.tensor(args.device)
for b in range(0, tokens, 2**20):
    cache[b : b + 2**20].normal_()
captures = load_captures("temp/mtp_overlap/overlap_ctx65536_*.pt")
sel, _ = make_mtp_indices(args.requests, 8, context, 2048, [args.fraction], 42, captures, device, tokens // 2)
rows = args.requests * 8
q = torch.randn(rows, 64, 512, dtype=torch.bfloat16, device=device)
qpe = torch.randn(rows, 64, 64, dtype=torch.bfloat16, device=device)
out = torch.empty_like(q)
idx = sel[args.fraction].clone()
w = make_wrapper(rows, idx, torch.empty(64 * 2**20, dtype=torch.uint8, device=device), 2048)
w.host_fetch_ctas = args.fetch_ctas
for _ in range(20):
    w.run(q, qpe, cache[..., :512], cache[..., 512:], out=out, host_page_start=tokens // 2)
torch.cuda.synchronize()
t = w._host_batch_buffers[7].view(torch.int64)[: 108 * 8].view(108, 8).cpu()
base = t[:, 0].min()
rel = (t[:, :5] - base).double() / 1000
fetch = t[:, 6] == 1
unique = int(w._host_batch_buffers[5][0]) if False else None
print(f"rows={rows} host={args.fraction:.0%} fetch_ctas={args.fetch_ctas}")
for name, mask in (("workers", ~fetch), ("fetchers", fetch)):
    r = rel[mask]
    print(f"{name:8s} start {r[:,0].max():6.1f}  after_prep {r[:,1].min():6.1f}-{r[:,1].max():6.1f}  "
          f"items_start {r[:,2].min():6.1f}-{r[:,2].max():6.1f}  items_end {r[:,3].min():6.1f}-{r[:,3].max():6.1f}  "
          f"end {r[:,4].max():6.1f}  wait_us median {t[mask,5].double().median()/1000:6.1f} max {t[mask,5].double().max()/1000:6.1f}")
p = w._host_batch_buffers[7].view(torch.int64)[108 * 8 : 2 * 108 * 8].view(108, 8)[: rows, :6].cpu()
if p[:, 0].min() > 0:
    d = (p - p[:, :1]).double() / 1000
    print("prepare per row (us from row start, slowest thread): count+table, loads+CAS, claim, ranks-sync, place")
    print("  ", [f"{v:.1f}" for v in d.median(0).values.tolist()[1:6]])
del w, cache
torch.cuda.synchronize()
allocation.close()
