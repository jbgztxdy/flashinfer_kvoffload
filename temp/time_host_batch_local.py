# Purpose: Time the batched Host kernel with no Host pages against native MLA, isolating overheads.
# Usage: PYTHONPATH=. python temp/time_host_batch_local.py --rows 1024
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_glm52_decode  # noqa: E402,F401  (points JIT at the source tree)
from flashinfer.mla import BatchMLAPagedAttentionWrapper  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=3)
parser.add_argument("--rows", type=int, nargs="+", default=[8, 64, 1024])
parser.add_argument("--fetch", type=int, nargs="+", default=[1, 4, 8, 16])
parser.add_argument("--rounds", type=int, default=10)
args = parser.parse_args()
device = torch.device("cuda", args.device)
torch.cuda.set_device(device)
torch.manual_seed(0)
pages = 8 * 2**20
cache = torch.randn(pages, 1, 576, dtype=torch.bfloat16, device=device)
for rows in args.rows:
    q = torch.randn(rows, 64, 512, dtype=torch.bfloat16, device=device)
    qpe = torch.randn(rows, 64, 64, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(q)
    indices = torch.randint(0, pages, (rows * 2048,), dtype=torch.int32, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(torch.empty(512 * 2**20, dtype=torch.uint8, device=device), backend="fa2")
    wrapper.plan(torch.arange(rows + 1, dtype=torch.int32), torch.arange(rows + 1, dtype=torch.int32) * 2048, indices,
                 torch.full((rows,), 2048, dtype=torch.int32), 64, 512, 64, 1, False, 1 / 16,
                 q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
    reps = max(1, 20000 // rows)
    variants = [("native", None, None)] + [(f"batch_f{f}", pages, f) for f in args.fetch]
    graphs = {}
    for name, boundary, fetch in variants:
        if fetch:
            wrapper.host_fetch_ctas = fetch
        wrapper.run(q, qpe, cache[..., :512], cache[..., 512:], out=out, host_page_start=boundary)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                wrapper.run(q, qpe, cache[..., :512], cache[..., 512:], out=out, host_page_start=boundary)
        graphs[name] = g
    times = {name: [] for name, _, _ in variants}
    for _ in range(args.rounds):
        for name, _, _ in variants:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); graphs[name].replay(); e.record(); e.synchronize()
            times[name].append(s.elapsed_time(e) * 1000 / reps)
    print(f"rows={rows}: " + "  ".join(f"{n}={sorted(t)[len(t)//2]:.1f}" for n, t in times.items()), flush=True)
