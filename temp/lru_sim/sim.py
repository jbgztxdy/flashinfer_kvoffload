# Purpose: per-request LRU pools (the kernel today) vs one global LRU pool of the same total size,
#          on measured consecutive-step DSA-proxy top-k (temp/mtp_overlap/rounds_ctx65536_*.pt).
#          Exact step-granular LRU as in the kernel: pages used this step are stamped and never
#          evicted in that step; misses take the least recently stamped slots; misses beyond the
#          free/evictable slots stay uncached. All KV but the newest token is far (HiSparse setup).
#          No kernel change; CPU only.
# Usage: python temp/lru_sim/sim.py --mtp 8 --requests 128 --capacity 2048 4096 8192
import argparse
import glob
import json
import random

import torch


def load(pattern):
    caps = []
    for p in sorted(glob.glob(pattern)):
        d = torch.load(p)
        caps.append((d["selections"], d["group_starts"]))
    return caps


def request_streams(caps, requests, mtp, seed, mixed):
    """Per request, per step: the set of far positions its MTP rows select."""
    rng = random.Random(seed)
    combos = [(c, l, g) for c in range(len(caps)) for l in range(caps[c][0].shape[0]) for g in range(caps[c][0].shape[1])]
    rng.shuffle(combos)
    tokens = caps[0][0].shape[2]
    streams = []
    for r in range(requests):
        c, l, g = combos[r % len(combos)]
        sel, starts = caps[c]
        start = int(starts[g])
        # mixed: requests are at different points of their decode, so they need not all be
        # in their first steps at once; offsets rotate the captured window.
        off = rng.randrange(tokens // mtp) * mtp if mixed else 0
        steps = []
        for t in range(tokens // mtp):
            tt = [(off + t * mtp + j) % tokens for j in range(mtp)]
            s = set()
            for j, tok in enumerate(tt):
                row = sel[l, g, tok].tolist()
                newest = start + tok
                s.update(p for p in row if p != newest)
            steps.append(s)
        streams.append(steps)
    return streams


def simulate(streams, capacity, global_pool):
    """Returns per-step (misses, refs_unique) summed over requests."""
    requests = len(streams)
    steps = len(streams[0])
    out = []
    if global_pool:
        stamp = {}  # (r, page) -> last step used
        total = capacity * requests
        for t in range(steps):
            need = [(r, p) for r in range(requests) for p in streams[r][t]]
            hits = [k for k in need if k in stamp]
            misses = [k for k in need if k not in stamp]
            for k in hits:
                stamp[k] = t
            free = total - len(stamp)
            evictable = sorted((v, k) for k, v in stamp.items() if v != t)
            take = misses[: free + len(evictable)]
            for i, k in enumerate(take):
                if i >= free:
                    del stamp[evictable[i - free][1]]
                stamp[k] = t
            out.append((len(misses), len(need)))
    else:
        stamps = [dict() for _ in range(requests)]
        for t in range(steps):
            m = n = 0
            for r in range(requests):
                st = stamps[r]
                need = streams[r][t]
                miss = [p for p in need if p not in st]
                for p in need:
                    if p in st:
                        st[p] = t
                free = capacity - len(st)
                evictable = sorted((v, p) for p, v in st.items() if v != t)
                for i, p in enumerate(miss[: free + len(evictable)]):
                    if i >= free:
                        del st[evictable[i - free][1]]
                    st[p] = t
                m += len(miss)
                n += len(need)
            out.append((m, n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default="temp/mtp_overlap/rounds_ctx65536_*.pt")
    ap.add_argument("--mtp", type=int, default=8)
    ap.add_argument("--requests", type=int, default=128)
    ap.add_argument("--capacity", nargs="+", type=int, default=[2048, 4096, 8192])
    ap.add_argument("--mixed", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    caps = load(a.captures)
    streams = request_streams(caps, a.requests, a.mtp, a.seed, a.mixed)
    steps = len(streams[0])
    res = dict(config=vars(a), steps=steps, results=[])
    for cap in a.capacity:
        row = dict(capacity_per_request=cap)
        for name, g in (("per_request", False), ("global", True)):
            per = simulate(streams, cap, g)
            tail = per[steps // 2 :]
            row[name] = dict(per_step=per, steady_miss_share=sum(m for m, _ in tail) / sum(n for _, n in tail),
                             steady_misses=sum(m for m, _ in tail) / len(tail))
        row["global_vs_per_request_misses"] = row["global"]["steady_misses"] / row["per_request"]["steady_misses"]
        res["results"].append(row)
        print(f"mtp={a.mtp} R={a.requests} cap={cap}/req mixed={a.mixed}: steady miss share per-request "
              f"{row['per_request']['steady_miss_share']:.3f}, global {row['global']['steady_miss_share']:.3f}, "
              f"global/per-request misses {row['global_vs_per_request_misses']:.3f}", flush=True)
    json.dump(res, open(a.output, "w"), indent=1)


if __name__ == "__main__":
    main()
