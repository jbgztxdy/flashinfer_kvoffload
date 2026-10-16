# Purpose: Measure CPU local DRAM bandwidth (NUMA-local), to separate it from the PCIe number.
# Usage: numactl --cpunodebind=1 --membind=1 python temp/bench_cpu_dram_bw.py
import time

import torch

torch.set_num_threads(32)  # one socket
N = 1 << 30  # 4 GiB of float32, far beyond the ~160 MB L3
a = torch.empty(N, dtype=torch.float32)
b = torch.empty(N, dtype=torch.float32)
a.uniform_()
b.uniform_()
gib = N * 4 / 2**30


def timed(fn, iters=3):
    fn()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters


dt = timed(lambda: a.sum())
print(f"  read  (sum)  {dt*1e3:7.1f} ms over {gib:.0f} GiB -> {gib*2**30/dt/1e9:6.0f} GB/s")
dt = timed(lambda: b.copy_(a))
print(f"  copy (r+w)   {dt*1e3:7.1f} ms over {gib:.0f} GiB -> {2*gib*2**30/dt/1e9:6.0f} GB/s (读+写计入)")
