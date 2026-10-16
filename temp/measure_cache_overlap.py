# Purpose: Do the "recent 4K window" and the "cross-round LRU cache" catch the same bytes?
#          If the recent window is already a subset of what the LRU cache holds, the two
#          factors must not be multiplied.
# Usage: python temp/measure_cache_overlap.py
import glob

import torch

MTP = 8
RECENT = 4096

stats = {k: [] for k in ("recent", "far_new", "recent_in_cache", "far_in_cache",
                         "marginal_of_window", "lru_only", "window_only")}
for path in sorted(glob.glob("temp/mtp_overlap/rounds_ctx65536_*.pt")):
    data = torch.load(path)
    sel, starts = data["selections"], data["group_starts"]
    layers, groups, span = sel.shape[:3]
    rounds = span // MTP
    for layer in range(0, layers, 3):
        for group in range(groups):
            base = int(starts[group])
            # Warm the cross-round cache on rounds 0..r-1, then look at round r.
            seen = set()
            for r in range(rounds):
                rows = sel[layer, group, r * MTP : (r + 1) * MTP].long()
                cutoff = base + r * MTP - RECENT
                uniq = set(rows.unique().tolist())
                recent = {t for t in uniq if t >= cutoff}
                far = uniq - recent
                if r > 0 and uniq:
                    stats["recent"].append(len(recent) / len(uniq))
                    # Of what a far-only LRU cache would still have to fetch:
                    stats["far_new"].append(len(far - seen) / max(1, len(far)))
                    # How much of each class the cross-round cache already holds:
                    if recent:
                        stats["recent_in_cache"].append(len(recent & seen) / len(recent))
                    if far:
                        stats["far_in_cache"].append(len(far & seen) / len(far))
                    # Marginal value of pinning the window, ON TOP of an LRU cache that
                    # already holds everything seen so far (including recent tokens).
                    miss = uniq - seen
                    stats["marginal_of_window"].append(
                        len({t for t in miss if t >= cutoff}) / max(1, len(miss)))
                    stats["lru_only"].append(len(miss) / len(uniq))
                    stats["window_only"].append(len(far) / len(uniq))
                seen |= uniq

m = {k: sum(v) / len(v) for k, v in stats.items() if v}
print("真实 DSA-like top-k，MTP 8 行/轮，4 个连续 verify 轮\n")
print("=== 一轮所需 token 的构成 ===")
print(f"  最近 {RECENT} 窗口内: {m['recent']:6.1%}      窗口外(远端): {1-m['recent']:6.1%}")
print("\n=== 跨轮缓存（保留所有见过的 token）已持有多少 ===")
print(f"  窗口内 token 已在缓存: {m['recent_in_cache']:6.1%}")
print(f"  远端   token 已在缓存: {m['far_in_cache']:6.1%}")
print("\n=== 两种策略单独的搬运量（相对无任何缓存）===")
print(f"  只常驻窗口，不缓存远端:  {m['window_only']:6.1%}  -> {1/m['window_only']:.2f}x")
print(f"  只做跨轮缓存，不常驻窗口: {m['lru_only']:6.1%}  -> {1/m['lru_only']:.2f}x")
print("\n=== 在跨轮缓存之上，再常驻窗口的『边际』收益 ===")
print(f"  缓存未命中的部分里，属于窗口内的: {m['marginal_of_window']:6.1%}")
print(f"  叠加后搬运量: {m['lru_only']*(1-m['marginal_of_window']):6.1%}"
      f"  -> 总计 {1/(m['lru_only']*(1-m['marginal_of_window'])):.2f}x")
print(f"  (若两者独立相乘应为 {1/m['window_only']*1/m['lru_only']:.2f}x —— 差距即重复计数)")
