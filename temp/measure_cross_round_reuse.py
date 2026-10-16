# Purpose: Of the Host tokens a verify round needs, how many did earlier rounds already fetch?
#          Decides whether a persistent HBM staging cache is worth building.
# Usage: python temp/measure_cross_round_reuse.py
import glob

import torch

RECENT = 4096   # positions this close to the query stay in HBM
MTP = 8         # rows per verify round

first, cumulative, marginal = [], [], []
for path in sorted(glob.glob("temp/mtp_overlap/rounds_ctx65536_*.pt")):
    data = torch.load(path)
    selections, starts = data["selections"], data["group_starts"]
    layers, groups, span = selections.shape[:3]
    rounds = span // MTP
    for layer in range(0, layers, 3):
        for group in range(groups):
            base = int(starts[group])
            sets = []
            for r in range(rounds):
                rows = selections[layer, group, r * MTP : (r + 1) * MTP].long()
                cutoff = base + r * MTP - RECENT
                sets.append(set(rows[rows < cutoff].unique().tolist()))
            if not all(sets):
                continue
            seen = set(sets[0])
            for r in range(1, rounds):
                hit_prev = len(sets[r] & sets[r - 1]) / len(sets[r])
                hit_all = len(sets[r] & seen) / len(sets[r])
                (first if r == 1 else cumulative).append(hit_prev)
                marginal.append((r, hit_all, len(sets[r] - seen) / len(sets[0])))
                seen |= sets[r]

print(f"真实 DSA-like top-k，MTP {MTP} 行/轮，远端 (>{RECENT}) token")
print(f"  轮 t 与轮 t-1 的重合率:        {sum(first+cumulative)/len(first+cumulative):.1%}")
for r in sorted({x[0] for x in marginal}):
    h = [x[1] for x in marginal if x[0] == r]
    m = [x[2] for x in marginal if x[0] == r]
    print(f"  轮 {r} 已在缓存中(对比前面所有轮): {sum(h)/len(h):6.1%}   "
          f"新增搬运量(相对轮0): {sum(m)/len(m):5.1%}")
