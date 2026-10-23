# Purpose: Markdown tables for docs/mla_host_lru_report.md (same definitions as temp/report_lru/plot.py).
# Usage: python3 temp/report_lru/tables.py <name>   (rounds_host_mtp8 | ... | sweep_host_mtp8 | capacity | hisparse)
import json
import sys
from pathlib import Path

A100 = Path("temp/a100_full")
H800 = Path("temp/h800/results/fi-validate-697235")
ROW = 1152


def steady(r, m):
    t = range(r["steps"] // 2, r["steps"])
    return sum(r["per_step"][m][x]["median_us"] for x in t) / len(t)


def rounds(name):
    d = json.loads((A100 / f"{name}.json").read_text())
    ours = "ours_stage" if "host" in name else "ours_inplace"
    bw = d["bandwidth"]["sm_gather_random_GBps"]
    print("| 请求 | 行 | 本文 kernel + LRU | Memcpy + LRU | Direct + LRU | 本文不开缓存 | 计算下限 | 传输下限 | **本文 ÷ 理论下限** | **本文 ÷ 朴素较快者** | LRU 加速 |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in d["results"]:
        t = range(r["steps"] // 2, r["steps"])
        miss = sum(r["cached_misses"][ours][x] for x in t) / len(t)
        o, m, di, nc, fc = (steady(r, k) for k in (ours, "memcpy_lru", "direct_lru", "ours_nocache", "floor_fa2"))
        ft = miss * ROW / bw / 1e3
        nv = min(m, di)
        bold = "**" if o < nv else ""
        print(f"| {r['requests']} | {r['query_rows']} | {o:.0f} | {m:.0f} | {di:.0f} | {nc:.0f} | {fc:.0f} | {ft:.0f} | "
              f"{o / max(fc, ft):.2f} | {bold}{o / nv:.2f}{bold} | {nc / o:.2f}× |")


def sweep(name):
    d = json.loads((A100 / f"{name}.json").read_text())
    bw = d["bandwidth"]["sm_gather_random_GBps"]
    fr = [c["far_fraction"] for c in d["results"][0]["cases"]]
    print("| 请求 | 行 | " + " | ".join(f"{f:.0%}" for f in fr) + " |")
    print("|---:|---:|" + "---:|" * len(fr))
    for r in d["results"]:
        cells = []
        for c in r["cases"]:
            o = c["batch"]["median_us"]
            nv = min(c["copy"]["median_us"], c["direct"]["median_us"])
            ub = sorted(c["far_unique_bytes"])[len(c["far_unique_bytes"]) // 2]
            fl = max(c["attn"]["median_us"], ub / bw / 1e3)
            b = "**" if o < nv and c["far_fraction"] > 0 else ""
            cells.append(f"{o:.0f} ({o / fl:.2f} / {b}{o / nv:.2f}{b})")
        print(f"| {r['requests']} | {r['query_rows']} | " + " | ".join(cells) + " |")


def capacity():
    print("| 远端 | 请求 | 2048 槽：本文 / 朴素较快者 | 4096 槽：本文 / 朴素较快者 | 8192 槽：本文 / 朴素较快者 |")
    print("|---|---:|---:|---:|---:|")
    for tier, ours, label in (("host", "ours_stage", "PCIe"), ("peer", "ours_inplace", "NVLink")):
        ds = {c: {r["requests"]: r for r in json.loads((A100 / n).read_text())["results"]}
              for c, n in ((2048, f"rounds_{tier}_mtp8_c2048.json"), (4096, f"rounds_{tier}_mtp8.json"),
                           (8192, f"rounds_{tier}_mtp8_c8192.json"))}
        for q in (1, 4, 16, 64, 128):
            cells = []
            for c in (2048, 4096, 8192):
                r = ds[c][q]
                o = steady(r, ours); nv = min(steady(r, "memcpy_lru"), steady(r, "direct_lru"))
                cells.append(f"{o:.0f} / {nv:.0f}（{o / nv:.2f}）")
            print(f"| {label} | {q} | " + " | ".join(cells) + " |")


def hisparse():
    print("| 场景 | **本文 staged + LRU** | 本文原地读 + LRU | 本文不开缓存 | **HiSparse** | FA2 计算下限 | **HiSparse ÷ 本文 staged** |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, req, label in (("rounds_host_tiny.json", 1, "MTP1，1 请求"), ("rounds_host_tiny.json", 4, "MTP1，4 请求"),
                             ("rounds_host_mtp8_tiny.json", 2, "MTP8，2 请求（16 行）")):
        r = next(x for x in json.loads((H800 / name).read_text())["results"] if x["requests"] == req)
        s, i, n, h, f = (steady(r, k) for k in ("ours_stage", "ours_inplace", "ours_nocache", "hisparse", "floor_fa2"))
        print(f"| {label} | {s:.1f} | {i:.1f} | {n:.1f} | {h:.1f} | {f:.1f} | **{h / s:.2f}×** |")


arg = sys.argv[1]
{"capacity": capacity, "hisparse": hisparse}.get(arg, (lambda: (rounds if arg.startswith("rounds") else sweep)(arg)))()
