# Purpose: Time the batched Host path with two prebuilt modules on the same inputs (A/B).
# Usage: PYTHONPATH=. python temp/hb_regs/ab_modules.py a.so b.so --requests 128 --fraction 0.3
import argparse
import sys
from pathlib import Path

import torch
import tvm_ffi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_glm52_host_batch import make_wrapper  # noqa: E402
from bench_glm52_host_batch_kernel import load_captures, make_mtp_indices  # noqa: E402
from bench_glm52_sparse_decode_hbm_host_direct import MixedHBMHost  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("modules", nargs="+", type=Path)
parser.add_argument("--device", type=int, default=3)
parser.add_argument("--requests", type=int, default=128)
parser.add_argument("--fraction", type=float, default=0.3)
parser.add_argument("--fetch-ctas", type=int, default=16)
parser.add_argument("--rounds", type=int, default=5)
parser.add_argument("--profile", action="store_true")
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
modules = [tvm_ffi.load_module(str(m.resolve())) for m in args.modules]
# Separate buffers per module: builds may lay out the state array differently.
wrappers = []
for module in modules:
    w = make_wrapper(rows, idx, torch.empty(64 * 2**20, dtype=torch.uint8, device=device), 2048)
    w.host_fetch_ctas = args.fetch_ctas
    w.run(q, qpe, cache[..., :512], cache[..., 512:], out=out, host_page_start=tokens // 2)
    torch.cuda.synchronize()
    for buffer in (w._host_batch_buffers[2], w._host_batch_buffers[5]):
        buffer.zero_()
    w._host_prefetch_module = module
    wrappers.append(w)
times = {m.name: [] for m in args.modules}
for _ in range(args.rounds if not args.profile else 1):
    for path, w in zip(args.modules, wrappers):
        for _ in range(0 if args.profile else 2):
            w.run(q, qpe, cache[..., :512], cache[..., 512:], out=out, host_page_start=tokens // 2)
        if args.profile:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            w.run(q, qpe, cache[..., :512], cache[..., 512:], out=out, host_page_start=tokens // 2)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            continue
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(3):
            w.run(q, qpe, cache[..., :512], cache[..., 512:], out=out, host_page_start=tokens // 2)
        e.record(); e.synchronize()
        times[path.name].append(s.elapsed_time(e) * 1000 / 3)
for name, t in times.items():
    if not t:
        continue
    print(f"{name}: median {sorted(t)[len(t)//2]:.1f} us  all {[round(x) for x in t]}")
del w, wrappers, cache
torch.cuda.synchronize()
allocation.close()
