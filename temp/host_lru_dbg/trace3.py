import sys
sys.path.insert(0, "temp/h800"); sys.path.insert(0, "temp")
import torch, tiers
from bench_glm52_host_batch import make_wrapper
dev = torch.device("cuda", 0); torch.cuda.set_device(dev)
tier, R, mode, mtp, fetch = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
caps = tiers.load_captures("temp/mtp_overlap/rounds_ctx65536_*.pt")
ctx, topk = 65536, 2048
kv = tiers.TieredKV(min(8, R) * ctx, R * ctx, R * mtp * topk, 0, tier, 1)
main = kv.main()
for b in range(0, kv.near + kv.far, 2**20): main[b:b + 2**20].normal_()
reqs = tiers.Requests(caps, R, ctx, kv.near, min(8, R), 1.0, 42 + R, "cpu")
rows = R * mtp
idx = torch.empty(rows * topk, dtype=torch.int32, device=dev)
w = make_wrapper(rows, idx, torch.empty(256 << 20, dtype=torch.uint8, device=dev), topk)
w.host_stage = mode != "inplace"; w.host_fetch_ctas = fetch
w.set_host_cache(torch.arange(R + 1, dtype=torch.int32) * mtp, 4096)
q = torch.randn(rows, 64, 512, dtype=torch.bfloat16, device=dev); qp = torch.randn(rows, 64, 64, dtype=torch.bfloat16, device=dev)
out = torch.empty_like(q)
steps = 32 // mtp
for t in range(steps):
    pages, _, _ = reqs.rows(t * mtp, mtp)
    idx.copy_(pages.view(-1).to(dev))
    if w._host_batch_buffers is not None: w._host_batch_buffers[6].view(-1)[-4096 * 4:].zero_()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); w.run(q, qp, main[..., :512], main[..., 512:], out=out, host_page_start=kv.near); e.record(); torch.cuda.synchronize()
    if t == steps - 1:
        tr = w._host_batch_buffers[6].view(-1)[-4096 * 4:].view(torch.int64).view(-1, 16)[:, :14].clone().cpu()
        g = int((tr[:, 0] > 0).sum()); tr = tr[:g]; t0 = tr[:, 0].min()
        def q_(k, f): col = tr[:, k]; col = col[col > 0]; return f"{(f(col) - t0).item() / 1000:.0f}"
        fe = tr[-fetch:, 6]; fe = fe[fe > 0]
        names = ["start", "prep", "bar1", "resolve", "lru", "items0", "fetch", "items", "bar3", "end"]
        names2 = {3: "resolve", 10: "lru_keys", 11: "lru_radix", 12: "lru_list", 13: "lru_bind", 4: "lru_end"}
        r0 = tr[0]
        print("  CTA0:", " ".join(f"{n}={(r0[k] - t0).item() / 1000:.1f}" for k, n in sorted(names2.items()) if r0[k] > 0))
        print("  max:", " ".join(f"{n}={q_(k, torch.max)}" for k, n in enumerate(names) if (tr[:, k] > 0).any()), "| items min", q_(7, torch.min))
        print(f"{mode} f{fetch} R={R}: total={s.elapsed_time(e)*1000:.0f}us unique={int(w._host_batch_buffers[5][1])} items_start={q_(5, torch.max)} fetch_end(max)={q_(6, torch.max)} worker_items_end min/max={q_(7, torch.min)}/{q_(7, torch.max)}")
