# Same step repeated: launch 0 is cold, later launches hit everything (no far reads); compare with
# the native kernel on the remapped all-near pages (floor) and on the far pages in place (direct).
import sys
sys.path.insert(0, "temp/h800"); sys.path.insert(0, "temp")
import torch, tiers
from bench_glm52_host_batch import make_wrapper
dev = torch.device("cuda", 0); torch.cuda.set_device(dev)
tier, R, mtp = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
mode = sys.argv[4] if len(sys.argv) > 4 else "inplace"
caps = tiers.load_captures("temp/mtp_overlap/rounds_ctx65536_*.pt")
ctx, topk = 65536, 2048
kv = tiers.TieredKV(min(8, R) * ctx, R * ctx, R * mtp * topk, 0, tier, 1)
main, alias = kv.main(), kv.alias()
for b in range(0, kv.near + kv.far, 2**20): main[b:b + 2**20].normal_()
reqs = tiers.Requests(caps, R, ctx, kv.near, min(8, R), 1.0, 42 + R, "cpu")
rows = R * mtp
pages, _, _ = reqs.rows(0, mtp)
flat = pages.view(-1); mask = flat >= kv.near
uniq, inv = torch.unique(flat[mask], return_inverse=True); remap = flat.clone(); remap[mask] = (kv.near + inv).int()
q = torch.randn(rows, 64, 512, dtype=torch.bfloat16, device=dev); qp = torch.randn(rows, 64, 64, dtype=torch.bfloat16, device=dev)
out = torch.empty_like(q)
def timed(fn, n=10):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(n):
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) * 1000)
    return sorted(ts)[len(ts) // 2], ts[0]
idx = torch.empty(rows * topk, dtype=torch.int32, device=dev)
w = make_wrapper(rows, idx, torch.empty(256 << 20, dtype=torch.uint8, device=dev), topk)
w.host_stage = mode != "inplace"
w.set_host_cache(torch.arange(R + 1, dtype=torch.int32) * mtp, 8192)
idx.copy_(flat.to(dev))
ours_warm, ours_cold = timed(lambda: w.run(q, qp, main[..., :512], main[..., 512:], out=out, host_page_start=kv.near))
idx2 = torch.empty_like(idx); w2 = make_wrapper(rows, idx2, torch.empty(256 << 20, dtype=torch.uint8, device=dev), topk)
idx2.copy_(remap.to(dev))
floor, _ = timed(lambda: w2.run(q, qp, alias[..., :512], alias[..., 512:], out=out))
idx3 = torch.empty_like(idx); w3 = make_wrapper(rows, idx3, torch.empty(256 << 20, dtype=torch.uint8, device=dev), topk)
idx3.copy_(flat.to(dev))
direct, _ = timed(lambda: w3.run(q, qp, main[..., :512], main[..., 512:], out=out))
print(f"{tier} R={R} mtp={mtp} {mode}: cold {ours_cold:.0f} warm(all hits) {ours_warm:.0f} | native all-near {floor:.0f} | native direct far {direct:.0f} | unique {len(uniq)}")
