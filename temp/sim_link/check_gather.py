# Purpose: validate the emulated link with the emulated gather: achieved bandwidth vs target, and
#          the latency of one dependent 16-row access vs the latency setting.
import sys; sys.path.insert(0, "temp/sim_link"); sys.path.insert(0, "temp/h800"); sys.path.insert(0, "temp")
import torch, emu, tiers
dev = torch.device("cuda", 0); torch.cuda.set_device(dev)
rows = 1 << 20
kv = tiers.TieredKV(rows, 2 * rows, rows, 0, "peer", 1); m = kv.main(); a = kv.alias()
m[kv.near:kv.near + rows].normal_()
far_range = (m[kv.near].data_ptr(), m[kv.near].data_ptr() + kv.far * 1152)
ids = torch.randperm(kv.far, device=dev)[:rows] + kv.near
dst = a[kv.near:kv.near + rows]
def timed(fn, n=7):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn(); ts = []
    for _ in range(n):
        s.record(); fn(); e.record(); e.synchronize(); ts.append(s.elapsed_time(e) * 1000)
    return sorted(ts)[n // 2]
for g in (0, 200, 100, 50, 25):
    emu.kernel_env(g, 0, *far_range)
    print(f"bandwidth target {g:4} GB/s -> {rows * 1152 / timed(lambda: emu.gather(m, ids, dst)) / 1e3:7.1f} GB/s")
for lat in (0, 2000, 5000, 10000, 20000):
    emu.kernel_env(0, lat, *far_range)
    print(f"latency setting {lat / 1000:4.0f} us -> dependent far read {emu.latency_ns(m, ids[:4096]) / 1000:6.2f} us")
emu.kernel_env(100, 0, *far_range); emu.gather(m, ids, dst)
print("correct", torch.equal(torch.index_select(m, 0, ids), dst))
