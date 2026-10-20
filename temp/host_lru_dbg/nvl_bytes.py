# NVLink Rx bytes on GPU 0 during the steady steps of the in-place cached kernel (peer tier),
# versus unique misses x 1152 and missed references x 1152.
import sys
sys.path.insert(0, "temp/h800"); sys.path.insert(0, "temp")
import torch, tiers
from bench_glm52_host_batch import make_wrapper
from bench_glm52_sparse_decode_p2p import nvlink_counters
dev = torch.device("cuda", 0); torch.cuda.set_device(dev)
R, mtp = int(sys.argv[1]), int(sys.argv[2])
caps = tiers.load_captures("temp/mtp_overlap/rounds_ctx65536_*.pt")
ctx, topk = 65536, 2048
kv = tiers.TieredKV(min(8, R) * ctx, R * ctx, R * mtp * topk, 0, "peer", 1)
main = kv.main()
for b in range(0, kv.near + kv.far, 2**20): main[b:b + 2**20].normal_()
reqs = tiers.Requests(caps, R, ctx, kv.near, min(8, R), 1.0, 42 + R, "cpu")
rows = R * mtp
idx = torch.empty(rows * topk, dtype=torch.int32, device=dev)
w = make_wrapper(rows, idx, torch.empty(256 << 20, dtype=torch.uint8, device=dev), topk)
w.host_stage = False
w.set_host_cache(torch.arange(R + 1, dtype=torch.int32) * mtp, 4096)
q = torch.randn(rows, 64, 512, dtype=torch.bfloat16, device=dev); qp = torch.randn(rows, 64, 64, dtype=torch.bfloat16, device=dev)
out = torch.empty_like(q)
steps = 32 // mtp
for t in range(steps):
    pages, _, _ = reqs.rows(t * mtp, mtp)
    idx.copy_(pages.view(-1).to(dev))
    resident = w._host_cache_views()[0] if w._host_cache["store"] is not None else None
    flat = pages.view(-1).to(dev).long(); far = flat >= kv.near
    hit = far & (resident[:-1][(flat - kv.near).clamp(0, kv.far - 1)] >= 0) if resident is not None else torch.zeros_like(far)
    miss_refs = int((far & ~hit).sum()); miss_unique = int(torch.unique(flat[far & ~hit]).numel())
    torch.cuda.synchronize(); before = nvlink_counters(0)
    w.run(q, qp, main[..., :512], main[..., 512:], out=out, host_page_start=kv.near)
    torch.cuda.synchronize(); after = nvlink_counters(0)
    rx = after["rx_bytes"] - before["rx_bytes"]
    print(f"step {t}: NVLink rx {rx / 1e6:.0f} MB | unique misses {miss_unique * 1152 / 1e6:.0f} MB | missed refs {miss_refs * 1152 / 1e6:.0f} MB")
