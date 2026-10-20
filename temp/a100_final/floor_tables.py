# Purpose: report tables for the A100 run: time vs theoretical floor for the single-step sweeps and
#          the LRU rounds. Floor = max(all-near attention, unique far bytes / measured gather BW).
# Usage: python3 temp/a100_final/floor_tables.py
import json
O = "temp/a100_final/"


def load(n):
    return json.load(open(O + n))


def sweep_table(name, fractions=(0.05, 0.1, 0.2, 0.3)):
    d = load(name)
    bw = d["bandwidth"]["sm_gather_random_GBps"]
    modes = [m for m in d["configuration"]["modes"] if m.startswith("batch")]
    print(f"\n#### {name}: best of ours ({', '.join(modes)}) — µs and ratio to floor; Memcpy / Direct for reference\n")
    print("| req | rows | " + " | ".join(f"{f:.0%}: ours (×floor) | Memcpy | Direct" for f in fractions) + " |")
    print("|---:|---:|" + "---:|---:|---:|" * len(fractions))
    for r in d["results"]:
        cs = {c["far_fraction"]: c for c in r["cases"]}
        cells = []
        for f in fractions:
            c = cs[f]
            ours = min(c[m]["median_us"] for m in modes)
            ub = sorted(c["far_unique_bytes"])[len(c["far_unique_bytes"]) // 2]
            floor = max(c["attn"]["median_us"], ub / bw / 1e3)
            cells.append(f"{ours:.0f} ({ours / floor:.2f}) | {c['copy']['median_us']:.0f} | {c['direct']['median_us']:.0f}")
        print(f"| {r['requests']} | {r['query_rows']} | " + " | ".join(cells) + " |")


def rounds_table(names, label):
    data = [load(n) for n in names]
    bw = data[0]["bandwidth"]["sm_gather_random_GBps"]
    print(f"\n#### {label}: steady state (mean of last-half steps), µs\n")
    print("| req | rows | best cached (mode) | no cache | compute floor | transfer floor | best / max(floors) | cache speedup |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    base = data[0]
    for i, r in enumerate(base["results"]):
        tail = range(r["steps"] // 2, r["steps"])
        mean = lambda d, m: sum(d["results"][i]["per_step"][m][t]["median_us"] for t in tail) / len(tail)
        best, best_mode, misses = 1e30, "", 0
        for d, tag in zip(data, [n.split("_")[-1].replace(".json", "") for n in names]):
            for m in ("ours_stage", "ours_inplace"):
                if m in d["results"][i]["per_step"]:
                    v = mean(d, m)
                    if v < best:
                        best, best_mode = v, f"{m.replace('ours_', '')}{'' if tag.startswith('mtp') else ' ' + tag}"
                        misses = sum(d["results"][i]["cached_misses"][m][t] for t in tail) / len(tail)
        nocache = mean(base, "ours_nocache")
        floor_c = mean(base, "floor_fa2")
        floor_t = misses * 1152 / bw / 1e3
        print(f"| {r['requests']} | {r['query_rows']} | {best:.0f} ({best_mode}) | {nocache:.0f} | {floor_c:.0f} | {floor_t:.0f} | "
              f"{best / max(floor_c, floor_t):.2f} | {nocache / best:.2f} |")


sweep_table("sweep_host_mtp8.json")
sweep_table("sweep_peer_mtp8.json")
sweep_table("sweep_host_mtp1.json")
sweep_table("sweep_peer_mtp1.json")
rounds_table(["rounds_host_mtp8.json"], "PCIe, MTP8, capacity 4096/request, 8 fetch CTAs")
rounds_table(["rounds_peer_mtp8.json", "rounds_peer_mtp8_f48.json"], "NVLink, MTP8, capacity 4096/request, 8 or 48 fetch CTAs")
rounds_table(["rounds_host_mtp1.json"], "PCIe, MTP1, capacity 4096/request")
rounds_table(["rounds_peer_mtp1.json", "rounds_peer_mtp1_f48.json"], "NVLink, MTP1, capacity 4096/request, 8 or 48 fetch CTAs")
