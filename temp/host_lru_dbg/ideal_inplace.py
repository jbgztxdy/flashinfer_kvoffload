# Ideal in-place cost of a cached step: the native kernel on the same entries with cache hits
# read from local HBM and misses read in place from the far tier, no cache management at all.
import sys
sys.path.insert(0, "temp/h800"); sys.path.insert(0, "temp")
import torch, tiers
from bench_glm52_host_batch import make_wrapper
dev = torch.device("cuda", 0); torch.cuda.set_device(dev)
tier, R, mtp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
caps = tiers.load_captures("temp/mtp_overlap/rounds_ctx65536_*.pt")
ctx, topk = 65536, 2048
kv = tiers.TieredKV(min(8, R) * ctx, R * ctx, R * mtp * topk, 0, tier, 1)
main = kv.main()
for b in range(0, kv.near + kv.far, 2**20): main[b:b + 2**20].normal_()
reqs = tiers.Requests(caps, R, ctx, kv.near, min(8, R), 1.0, 42 + R, "cpu")
rows = R * mtp
idx = torch.empty(rows * topk, dtype=torch.int32, device=dev)
w = make_wrapper(rows, idx, torch.empty(256 << 20, dtype=torch.uint8, device=dev), topk)
w.host_stage = False
w.set_host_cache(torch.arange(R + 1, dtype=torch.int32) * mtp, 4096)
idn = torch.empty_like(idx)
wn = make_wrapper(rows, idn, torch.empty(256 << 20, dtype=torch.uint8, device=dev), topk)
q = torch.randn(rows, 64, 512, dtype=torch.bfloat16, device=dev); qp = torch.randn(rows, 64, 64, dtype=torch.bfloat16, device=dev)
out = torch.empty_like(q)
steps = 32 // mtp
ev = lambda: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
res = []
for t in range(steps):
    pages, _, _ = reqs.rows(t * mtp, mtp)
    idx.copy_(pages.view(-1).to(dev))
    # Which entries hit: resident before this launch.
    resident = w._host_cache_views()[0] if w._host_cache["store"] is not None else None
    flat = pages.view(-1).to(dev).long()
    if resident is not None:
        rel = (flat - kv.near).clamp(min=0)
        hit = (flat >= kv.near) & (resident[:-1][rel.clamp(max=kv.far - 1)] >= 0)
    else:
        hit = torch.zeros_like(flat, dtype=torch.bool)
    ideal = torch.where(hit, flat % kv.near, flat).int()
    idn.copy_(ideal)
    s, e = ev(); s.record(); w.run(q, qp, main[..., :512], main[..., 512:], out=out, host_page_start=kv.near); e.record(); torch.cuda.synchronize()
    ours = s.elapsed_time(e) * 1000
    ts = []
    for _ in range(5):
        s, e = ev(); s.record(); wn.run(q, qp, main[..., :512], main[..., 512:], out=out); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1000)
    far_refs = int(((flat >= kv.near) & ~hit).sum())
    res.append((ours, sorted(ts)[2], far_refs))
tail = res[len(res) // 2 :]
o = sum(x[0] for x in tail) / len(tail); n = sum(x[1] for x in tail) / len(tail); f = sum(x[2] for x in tail) / len(tail)
print(f"{tier} R={R} mtp={mtp}: ours in place {o:.0f} us | native, hits near + misses in place {n:.0f} us | far refs/step {f:.0f} ({f * 1152 / 1e6:.0f} MB)")
