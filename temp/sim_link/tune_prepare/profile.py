"""Debug-only phase trace. Apply profile.patch, and restore the header after use."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, "temp/h800")
sys.path.insert(0, "temp")
import torch
import tiers
from bench_glm52_host_batch import make_wrapper
from bench_sweep import setup_emu

parser = argparse.ArgumentParser()
parser.add_argument("--captures", default="temp/mtp_overlap/rounds_ctx65536_*.pt")
parser.add_argument("--requests", type=int, required=True)
parser.add_argument("--mtp", type=int, required=True)
parser.add_argument("--steps", type=int, default=32)
parser.add_argument("--emu", default=None)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
assert not args.output.exists()
dev = torch.device("cuda", 0)
torch.cuda.set_device(dev)
torch.set_num_threads(8)
torch.manual_seed(42)
caps = tiers.load_captures(args.captures)
ctx, topk, requests, mtp = 65536, 2048, args.requests, args.mtp
kv = tiers.TieredKV(requests * ctx, requests * ctx, requests * mtp * topk, 0, "peer", 0)
main = kv.main()
for b in range(0, kv.near + kv.far, 2**20):
    main[b:b + 2**20].normal_()
setup_emu(args.emu, kv)
reqs = tiers.Requests(caps, requests, ctx, kv.near, requests, 1.0, 42 + requests, "cpu")
rows = requests * mtp
indices = torch.empty(rows * topk, dtype=torch.int32, device=dev)
w = make_wrapper(rows, indices, torch.empty(256 << 20, dtype=torch.uint8, device=dev), topk)
w.host_stage = True
w.host_fetch_ctas = 16
w.set_host_cache(torch.arange(requests + 1, dtype=torch.int32) * mtp, 4096)
q = torch.randn(rows, 64, 512, dtype=torch.bfloat16, device=dev)
qp = torch.randn(rows, 64, 64, dtype=torch.bfloat16, device=dev)
out = torch.empty_like(q)
names = ["start", "prepare", "barrier1", "resolve", "before_barrier2", "before_fetch",
         "fetch", "items", "barrier3", "merge", "lru_start", "lru_end"]
data = []
steps = min(args.steps, reqs.tokens // mtp)
for t in range(steps):
    pages, _, _ = reqs.rows(t * mtp, mtp)
    indices.copy_(pages.flatten().to(dev))
    if w._host_batch_buffers is not None:
        w._host_batch_buffers[6].flatten()[-4096 * 4:].zero_()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    w.run(q, qp, main[..., :512], main[..., 512:], out=out, host_page_start=kv.near)
    end.record()
    torch.cuda.synchronize()
    if t < steps - 2:
        continue
    trace = w._host_batch_buffers[6].flatten()[-4096 * 4:].view(torch.int64).view(-1, 16).cpu()
    trace = trace[trace[:, 0] > 0]
    zero = int(trace[:, 0].min())
    summary = {}
    for k, name in enumerate(names):
        values = trace[:, k][trace[:, k] > 0]
        if values.numel():
            summary[name] = dict(min_us=(int(values.min()) - zero) / 1000,
                                 max_us=(int(values.max()) - zero) / 1000)
    record = dict(step=t, event_us=start.elapsed_time(end) * 1000,
                  unique=int(w._host_batch_buffers[5][1]), phases=summary,
                  timestamps_ns=trace.tolist())
    data.append(record)
    print(json.dumps({k: v for k, v in record.items() if k != "timestamps_ns"}), flush=True)
args.output.write_text(json.dumps(dict(configuration=vars(args) | {"output": str(args.output)},
    header_sha256=hashlib.sha256(Path("include/flashinfer/attention/mla_host_batch.cuh").read_bytes()).hexdigest(),
    records=data), indent=2) + "\n")
