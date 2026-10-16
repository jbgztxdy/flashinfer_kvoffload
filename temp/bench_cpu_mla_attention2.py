# Purpose: Tighter CPU bound for host-side MLA attention: L3-resident peak probe and a
#          per-request GEMM (all 8 MTP rows against the union of their tokens).
# Usage: OMP_NUM_THREADS=64 taskset -c 0-63 python temp/bench_cpu_mla_attention2.py
import time

import torch

HEADS, CKV, KPE = 64, 512, 64
PCIE = 27e9
torch.set_num_threads(64)


def timed(fn, warmup=2, iters=3):
    for _ in range(warmup):
        fn()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters


print("=== 1. 真实算力上限（L3 驻留）===")
for n in (1024, 2048, 3072, 8192):
    a = torch.randn(n, n, dtype=torch.bfloat16)
    b = torch.randn(n, n, dtype=torch.bfloat16)
    dt = timed(lambda: a @ b)
    print(f"  {n}^3: {dt*1e3:7.2f} ms -> {2*n**3/dt/1e12:6.1f} TFLOPS  (操作数 {3*n*n*2/2**20:.0f} MiB)")

print("\n=== 2. 按请求组批：8 个 MTP 行 vs 它们 token 的并集 ===")
REQ, MTP, UNION = 128, 8, 1200   # 1200 ~= 8 rows x 614 host tokens x 0.41 去重
M = MTP * HEADS
q = torch.randn(REQ, M, CKV + KPE, dtype=torch.bfloat16)
kv = torch.randn(REQ, CKV + KPE, UNION, dtype=torch.bfloat16)
v = torch.randn(REQ, UNION, CKV, dtype=torch.bfloat16)


def run():
    s = torch.bmm(q, kv)
    return torch.bmm(s.float().softmax(-1).bfloat16(), v)


dt = timed(run, warmup=1, iters=2)
flop = REQ * M * UNION * 2 * (CKV + KPE + CKV)
rows = REQ * MTP
kv_bytes = REQ * UNION * (CKV + KPE) * 2
out_bytes = rows * HEADS * CKV * 2
print(f"  [{M},576]x[576,{UNION}] x {REQ} 请求: {dt*1e3:7.1f} ms  ({flop/dt/1e12:.1f} TFLOPS, {flop/1e9:.0f} GFLOP)")
print(f"  (含 ~2x 掩码浪费；有效 FLOP {flop/2/1e9:.0f} GFLOP)")
print(f"  对比  搬 KV 到 GPU {kv_bytes/PCIE*1e3:.1f} ms ({kv_bytes/1e6:.0f} MB)")
print(f"        搬 partial 回 {out_bytes/PCIE*1e3:.1f} ms ({out_bytes/1e6:.0f} MB)")
