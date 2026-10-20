# Purpose: Markdown tables from the H800 sweep (bench_sweep.py) and rounds (bench_rounds.py) JSONs.
# Usage: python3 temp/h800/summarize.py temp/h800/results/*/sweep.json temp/h800/results/*/rounds*.json
import json
from pathlib import Path
import sys


def fmt(us):
    return "-" if us is None else (f"{us / 1000:.2f} ms" if us >= 1000 else f"{us:.1f}")


def sweep(path, data):
    cfg = data["configuration"]
    modes = [m for m in cfg["modes"]]
    fractions = cfg["fractions"]
    bw = data.get("bandwidth", {})
    print(f"## sweep: tier={cfg['tier']} mtp={cfg['mtp']} ({data['device_name']}, {Path(path).parent.name})\n")
    print(f"far-tier bandwidth: DMA {bw.get('dma_contiguous_GBps', 0):.1f} GB/s, SM gather "
          f"{bw.get('sm_gather_random_GBps', 0):.1f} GB/s; commit {data['git_commit'][:8]} dirty={data['git_dirty']}\n")
    gbps = bw.get("sm_gather_random_GBps") or bw.get("dma_contiguous_GBps")
    for mode in modes:
        print(f"### {mode} (µs)\n")
        print("| req | rows | " + " | ".join(f"{f:.0%}" for f in fractions) + " |")
        print("|---:|---:|" + "---:|" * len(fractions))
        for r in data["results"]:
            cells = []
            for c in r["cases"]:
                cells.append(fmt(c.get(mode, {}).get("median_us")))
            print(f"| {r['requests']} | {r['query_rows']} | " + " | ".join(cells) + " |")
        print()
    # Ours vs the best baseline and vs max(compute floor, transfer floor).
    ours = [m for m in modes if m.startswith("batch")]
    base = [m for m in modes if m in ("copy", "direct")]
    print("### best of ours / best baseline, and best of ours / max(attn, unique bytes ÷ bandwidth)\n")
    print("| req | rows | " + " | ".join(f"{f:.0%}" for f in fractions) + " |")
    print("|---:|---:|" + "---:|" * len(fractions))
    for r in data["results"]:
        cells = []
        for c in r["cases"]:
            o = min(c[m]["median_us"] for m in ours if m in c)
            b = min(c[m]["median_us"] for m in base if m in c)
            floor = max(c["attn"]["median_us"], sorted(c["far_unique_bytes"])[len(c["far_unique_bytes"]) // 2] / gbps / 1e3)
            cells.append(f"{o / b:.2f} / {o / floor:.2f}")
        print(f"| {r['requests']} | {r['query_rows']} | " + " | ".join(cells) + " |")
    print()


def rounds(path, data):
    cfg = data["configuration"]
    modes = cfg["modes"]
    print(f"## rounds: tier={cfg['tier']} mtp={cfg['mtp']} capacity={cfg['capacity']} fraction={cfg['fraction']} "
          f"steps={cfg['steps']} ({Path(path).parent.name})\n")
    print("Steady state = mean of the per-step medians over the last half of the steps (µs).\n")
    print("| req | rows | " + " | ".join(modes) + " | misses/step (ours) | uncached unique/step |")
    print("|---:|---:|" + "---:|" * len(modes) + "---:|---:|")
    for r in data["results"]:
        steps = r["steps"]
        tail = range(steps // 2, steps)
        cells = []
        for m in modes:
            per = r["per_step"].get(m)
            cells.append(fmt(sum(per[t]["median_us"] for t in tail) / len(tail)) if per else "-")
        miss = r["cached_misses"].get("ours_stage") or r["cached_misses"].get("ours_inplace") or []
        miss_s = f"{sum(miss[t] for t in tail) / len(tail):.0f}" if miss else "-"
        uniq = sum(r["far_unique_uncached"][t] for t in tail) / len(tail)
        print(f"| {r['requests']} | {r['query_rows']} | " + " | ".join(cells) + f" | {miss_s} | {uniq:.0f} |")
    print()
    if "hisparse" in modes:
        ours = [m for m in modes if (m.startswith("ours") and m != "ours_nocache") or m.endswith("_lru")]
        print("cached methods / HiSparse, steady state (memcpy_lru/direct_lru keep the LRU on the CPU, untimed):\n")
        print("| req | " + " | ".join(ours) + " |\n|---:|" + "---:|" * len(ours))
        for r in data["results"]:
            tail = range(r["steps"] // 2, r["steps"])
            h = sum(r["per_step"]["hisparse"][t]["median_us"] for t in tail)
            print(f"| {r['requests']} | " + " | ".join(
                f"{sum(r['per_step'][m][t]['median_us'] for t in tail) / h:.2f}" for m in ours) + " |")
        print()


for path in sys.argv[1:]:
    data = json.loads(Path(path).read_text())
    (rounds if "steps" in data["configuration"] else sweep)(path, data)
