# Purpose: per step of the per-request LRU (kernel semantics), unique missed pages vs references
#          to missed pages over all MTP rows: what in-place reads (refs) vs staged reads (unique)
#          cost on the far link.
import sys
sys.path.insert(0, "temp/lru_sim")
from sim import load, request_streams
import random, torch

caps = load("temp/mtp_overlap/rounds_ctx65536_*.pt")
for mtp, R, cap in ((8, 128, 4096), (8, 16, 4096), (1, 128, 4096)):
    rng = random.Random(42)
    combos = [(c, l, g) for c in range(len(caps)) for l in range(caps[c][0].shape[0]) for g in range(caps[c][0].shape[1])]
    rng.shuffle(combos)
    tokens = caps[0][0].shape[2]
    stamps = [dict() for _ in range(R)]
    out = []
    for t in range(tokens // mtp):
        uniq = refs = allrefs = 0
        for r in range(R):
            c, l, g = combos[r % len(combos)]
            sel, starts = caps[c]
            start = int(starts[g])
            rows = [set(sel[l, g, t * mtp + j].tolist()) - {start + t * mtp + j} for j in range(mtp)]
            need = set().union(*rows)
            st = stamps[r]
            miss = [p for p in need if p not in st]
            ms = set(miss)
            refs += sum(len(row & ms) for row in rows)
            allrefs += sum(len(row) for row in rows)
            uniq += len(miss)
            for p in need:
                if p in st: st[p] = t
            free = cap - len(st)
            ev = sorted((v, p) for p, v in st.items() if v != t)
            for i, p in enumerate(miss[: free + len(ev)]):
                if i >= free: del st[ev[i - free][1]]
                st[p] = t
        out.append((uniq, refs, allrefs))
    tail = out[len(out) // 2 :]
    u = sum(x[0] for x in tail) / len(tail); m = sum(x[1] for x in tail) / len(tail); a = sum(x[2] for x in tail) / len(tail)
    print(f"mtp={mtp} R={R}: steady per step unique misses {u:.0f}, miss refs {m:.0f} ({m / u:.2f}x unique), all refs {a:.0f}; "
          f"far bytes in place {m * 1152 / 1e6:.0f} MB vs staged {u * 1152 / 1e6:.0f} MB")
