# Purpose: before a run on the emulated link, check that it delivers the setting: gather
#          bandwidth within 5% of the target and dependent-read latency within 10% (+ base).
# Usage: python temp/sim_link/validate.py <GB/s> <latency ns> <out.json>
import json, sys
sys.path.insert(0, "temp/sim_link"); sys.path.insert(0, "temp/h800"); sys.path.insert(0, "temp")
import torch, emu, tiers
gbps, lat, out = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
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
emu.kernel_env(0, 0, *far_range); base_lat = min(emu.latency_ns(m, ids[:4096]) for _ in range(3))
emu.kernel_env(gbps, lat, *far_range)
bw = rows * 1152 / timed(lambda: emu.gather(m, ids, dst)) / 1e3
got_lat = min(emu.latency_ns(m, ids[:4096]) for _ in range(3))
ok = (gbps == 0 or abs(bw - gbps) <= 0.05 * gbps) and abs(got_lat - max(lat, base_lat)) <= 0.1 * max(lat, base_lat) + 600
r = dict(target_GBps=gbps, target_latency_ns=lat, measured_GBps=bw, base_latency_ns=base_lat, measured_latency_ns=got_lat, ok=ok)
json.dump(r, open(out, "w"), indent=1); print(r)
sys.exit(0 if ok else 1)
