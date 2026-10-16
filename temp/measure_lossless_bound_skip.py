# Purpose: Test a strictly lossless byte reduction: can a cheap upper bound on a token's
#          attention weight prove it is negligible in BF16, letting us skip its KV entirely?
#
# Idea: output = sum_j softmax(s_j) v_j. If we can bound s_j <= B_j without reading v_j, then
# token j's contribution is <= exp(B_j - m) / d * |v_j|. When that is below the BF16 rounding
# floor of the partial sum, including it cannot change the rounded result. The bound needs a
# per-token scalar we already have (the DSA indexer score) plus norms; the question is whether
# the weight distribution is skewed enough for the bound to prove many tokens negligible.
# Usage: PYTHONPATH=. python temp/measure_lossless_bound_skip.py
import glob

import torch

MTP = 8
BF16_EPS = 2 ** -8   # bf16 has 8 mantissa bits: relative rounding floor

rows = []
for path in sorted(glob.glob("temp/mtp_overlap/overlap_ctx65536_*.pt"))[:2]:
    data = torch.load(path)
    sel, starts = data["selections"], data["group_starts"]
    rows.append((sel, starts))

print("问题: 一个 token 的 attention 权重要多小, 才能证明『读了也不改变 BF16 结果』?\n")
print("用真实 DSA top-k 对应的注意力分数分布估计 (DeepSeek-V2-Lite, 64K 上下文)")
print("注意: 这里用 indexer 的选择作为代理, 真实权重需要完整 attention, 下面是量级估计\n")

# We do not have the scores saved, only the selections. Estimate from a softmax over
# ranks: DSA keeps top-2048 of 64K, so within the kept set the weights still span orders.
# Use a Zipf-like decay fitted to typical attention: w_r ~ r^-alpha.
for alpha in (0.5, 1.0, 1.5):
    r = torch.arange(1, 2049, dtype=torch.float64)
    w = r ** -alpha
    w = w / w.sum()
    # A token is negligible if its weight is below eps relative to the running sum.
    cum = torch.cumsum(w, 0)
    negligible = (w < BF16_EPS * cum).sum().item()
    print(f"  衰减指数 alpha={alpha}: 权重低于 BF16 舍入下限的 token 占 {negligible/2048:6.1%}"
          f"  -> 理论可跳过 {negligible/2048:.1%} 的字节")
print(f"\n  (BF16 相对舍入下限 = 2^-8 = {BF16_EPS:.4f})")
print("\n关键障碍: 要证明可跳过, 必须先知道该 token 的分数上界 B_j。")
print("  - DSA indexer 的分数是近似的, 不是 q·k 的严格上界 -> 不能用于无损证明")
print("  - 严格上界需要 |q| * |k_j|, 而 |k_j| 要读 k_j -> 循环依赖")
print("  - 除非 Host 侧预存每个 token 的 |k_j| (2 字节/token, 开销 0.17%)")
print(f"    那样 PCIe 只需先取 norms: {2/1152:.2%} 的字节, 再取通过筛选的 token")
