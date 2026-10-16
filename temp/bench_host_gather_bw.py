# Purpose: Is 27 GB/s really the PCIe ceiling, or does scattered 1152 B gather lose bandwidth?
#          Holds total bytes and concurrency fixed; varies only the access pattern.
# Usage: PYTHONPATH=. python temp/bench_host_gather_bw.py
import ctypes
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_glm52_sparse_decode_hbm_host_direct import MixedHBMHost  # noqa: E402

DEV = 3
ROW_B = 1152
VECS = ROW_B // 16
POOL = 4 << 20            # pages in the Host half
FETCH = 256 * 1024        # rows fetched per measurement: 288 MiB, constant across patterns

torch.cuda.set_device(DEV)
alloc = MixedHBMHost(2 * POOL, DEV)
cache = alloc.tensor(DEV)
for b in range(0, 2 * POOL, 2**20):
    cache[b : b + 2**20].normal_()
torch.cuda.synchronize()

drv = ctypes.CDLL("libcuda.so.1")
mod = ctypes.c_void_p()
drv.cuModuleLoad(ctypes.byref(mod), b"temp/bench_host_gather_bw.cubin")
fn = ctypes.c_void_p()
drv.cuModuleGetFunction(ctypes.byref(fn), mod, b"gather")

host_base = cache.data_ptr() + POOL * ROW_B
sink = torch.zeros(1 << 20, dtype=torch.int32, device="cuda")


def run(idx, blocks, iters=3):
    idx = idx.to("cuda", torch.int32)
    args = (ctypes.c_void_p * 5)(
        *[ctypes.cast(ctypes.pointer(x), ctypes.c_void_p) for x in
          (ctypes.c_void_p(host_base), ctypes.c_void_p(idx.data_ptr()),
           ctypes.c_void_p(sink.data_ptr()), ctypes.c_int(len(idx)), ctypes.c_int(VECS))])
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    drv.cuLaunchKernel(fn, blocks, 1, 1, 256, 1, 1, 0, None, args, None)
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        drv.cuLaunchKernel(fn, blocks, 1, 1, 256, 1, 1, 0, None, args, None)
    e.record(); e.synchronize()
    return len(idx) * ROW_B / (s.elapsed_time(e) / iters / 1e3) / 1e9


g = torch.Generator().manual_seed(0)
print(f"固定搬运 {FETCH*ROW_B/2**20:.0f} MiB，只改访问模式（blocks=432）\n")
print(f"{'模式':34} {'GB/s':>7} {'vs 顺序':>8}")
seq = torch.arange(FETCH)
base = run(seq, 432)
print(f"{'顺序（连续 288 MiB）':34} {base:7.1f} {1.0:7.2f}")
for name, idx in [
    ("随机，池 288 MiB（=取数量）", torch.randperm(FETCH, generator=g)),
    ("随机，池 1 GiB", torch.randperm(1 << 20, generator=g)[:FETCH]),
    ("随机，池 4.5 GiB（全池）", torch.randperm(POOL, generator=g)[:FETCH]),
]:
    bw = run(idx, 432)
    print(f"{name:34} {bw:7.1f} {bw/base:7.2f}")
for stride in (2, 4, 16):
    idx = (torch.arange(FETCH) * stride) % POOL
    bw = run(idx, 432)
    print(f"{'跨步 '+str(stride)+' 行（每 '+str(stride*ROW_B//1024)+' KiB 取 1152B）':34} {bw:7.1f} {bw/base:7.2f}")
print("\n=== 排序能否恢复带宽（页表局部性 vs 请求合并）===")
rnd = torch.randperm(POOL, generator=g)[:FETCH]
print(f"{'随机（全池）':34} {run(rnd,432):7.1f}")
print(f"{'同样的页，但排序后访问':34} {run(rnd.sort().values,432):7.1f}")
print("\n=== 并发度 ===")
for blocks in (108, 216, 432, 864):
    print(f"  blocks={blocks:4d}  顺序 {run(seq,blocks):5.1f}  随机(全池) {run(rnd,blocks):5.1f} GB/s")
del cache
torch.cuda.synchronize()
alloc.close()
