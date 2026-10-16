# Purpose: Amplification a page-granular Host fetch would pay on real DSA-like top-k selections.
# Usage: python temp/measure_page_granularity.py
import glob

import torch

RECENT = 4096  # positions this close to the query stay in HBM
MTP = 8

rows_per_page = (1, 8, 16, 32, 64)
per_row = {p: [] for p in rows_per_page}
after_dedup = {p: [] for p in rows_per_page}
for path in sorted(glob.glob("temp/mtp_overlap/overlap_ctx65536_*.pt")):
    data = torch.load(path)
    selections, starts = data["selections"], data["group_starts"]
    layers, groups = selections.shape[:2]
    for layer in range(0, layers, 4):
        for group in range(groups):
            sets = selections[layer, group, :MTP].long()
            far = sets[sets < int(starts[group]) - RECENT]
            if far.numel() == 0:
                continue
            tokens = far.unique().numel()  # what we fetch today, after dedup
            for p in rows_per_page:
                # Naive page fetch: every distinct page touched, whole page moved.
                pages = (far // p).unique().numel()
                after_dedup[p].append(pages * p / tokens)
                # Same, but without cross-row dedup (one row at a time).
                row = sets[0][sets[0] < int(starts[group]) - RECENT]
                per_row[p].append((row // p).unique().numel() * p / row.numel())

print(f"真实 DSA-like top-k（远端 >4K 的选择），MTP {MTP} token/请求")
print(f"{'page size':>10} {'每 token 字节':>13} {'按页取 / 按 token 取':>22} {'单行内':>10}")
for p in rows_per_page:
    a = sum(after_dedup[p]) / len(after_dedup[p])
    r = sum(per_row[p]) / len(per_row[p])
    print(f"{p:10d} {p * 1152:13,} {a:21.1f}x {r:9.1f}x")
