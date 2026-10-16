# Purpose: Can this CPU compute MLA host-side attention faster than PCIe can move the KV?
# Usage: OMP_NUM_THREADS=64 python temp/bench_cpu_mla_attention.py
import time

import torch

HEADS, CKV, KPE = 64, 512, 64
PCIE = 27e9

torch.set_num_threads(64)
print(f"threads={torch.get_num_threads()}  bf16 amx via oneDNN")

# Peak probe: one big square BF16 GEMM.
a = torch.randn(8192, 8192, dtype=torch.bfloat16)
b = torch.randn(8192, 8192, dtype=torch.bfloat16)
for _ in range(2):
    a @ b
t = time.perf_counter()
for _ in range(3):
    a @ b
dt = (time.perf_counter() - t) / 3
print(f"8192^3 GEMM: {dt*1e3:7.1f} ms -> {2*8192**3/dt/1e12:6.1f} TFLOPS (achievable peak)")

# The real shape: per query row, [64 heads, 576] x [576, H host tokens] then [64, H] x [H, 512].
for rows, host in ((1024, 614), (1024, 205), (64, 614)):
    q = torch.randn(rows, HEADS, CKV + KPE, dtype=torch.bfloat16)
    kv = torch.randn(rows, CKV + KPE, host, dtype=torch.bfloat16)   # gathered per row
    v = torch.randn(rows, host, CKV, dtype=torch.bfloat16)
    for _ in range(1):
        s = torch.bmm(q, kv); torch.bmm(s.softmax(-1).bfloat16(), v)
    t = time.perf_counter()
    s = torch.bmm(q, kv)
    o = torch.bmm(s.float().softmax(-1).bfloat16(), v)
    dt = time.perf_counter() - t
    flop = rows * host * 2 * HEADS * (CKV + KPE + CKV)
    kv_bytes = rows * host * 0.408 * (CKV + KPE) * 2      # deduped KV over PCIe
    out_bytes = rows * HEADS * CKV * 2                     # BF16 partial output back
    print(f"\nrows={rows} host_tokens/row={host}:")
    print(f"  CPU attention {dt*1e3:8.1f} ms  ({flop/dt/1e12:.1f} TFLOPS, {flop/1e9:.0f} GFLOP)")
    print(f"  搬 KV 到 GPU  {kv_bytes/PCIE*1e3:8.1f} ms  ({kv_bytes/1e6:.0f} MB)")
    print(f"  搬 partial 回 {out_bytes/PCIE*1e3:8.1f} ms  ({out_bytes/1e6:.0f} MB)")
