# Purpose: every measured value of temp/a100_full as Markdown tables, one section per result file:
#          single-step sweeps (each method, p10/median/p90) and LRU rounds (each method per step).
# Usage: python3 temp/a100_full/full_tables.py > temp/a100_full/full_tables.md
import json
from pathlib import Path

O = Path("temp/a100_full")
SWEEPS = ["sweep_host_mtp8.json", "sweep_peer_mtp8.json", "sweep_host_mtp1.json", "sweep_peer_mtp1.json"]
ROUNDS = ["rounds_host_mtp8.json", "rounds_peer_mtp8.json", "rounds_host_mtp1.json", "rounds_peer_mtp1.json",
          "rounds_host_mtp8_c2048.json", "rounds_host_mtp8_c8192.json",
          "rounds_peer_mtp8_c2048.json", "rounds_peer_mtp8_c8192.json"]
NAMES = dict(attn="计算下限（全在本地）", copy="Memcpy", direct="Direct", batch="ours",
             ours_stage="ours（staged）", ours_inplace="ours（原地读）", ours_nocache="ours 不开缓存",
             memcpy_lru="Memcpy+LRU", direct_lru="Direct+LRU", floor_fa2="计算下限（全在本地）")


def anchor(name):
    return name.replace(".json", "").replace("_", "-")


print("# A100 全量结果（每个测量值）\n")
print("由 `temp/a100_full/full_tables.py` 从同目录的 JSON 生成；说明见 [`docs/mla_host_a100_full.md`](../../docs/mla_host_a100_full.md)。"
      "比值和下限对比见 [`tables.md`](tables.md)。单位 µs。\n")
print("## 目录\n")
for n in SWEEPS + ROUNDS:
    print(f"- [{n}](#{anchor(n)})：[原始 JSON]({n})")
print()

for n in SWEEPS:
    d = json.loads((O / n).read_text())
    cfg = d["configuration"]; bw = d["bandwidth"]
    modes = [m for m in cfg["modes"] if m in d["results"][0]["cases"][0]]
    print(f"<a id=\"{anchor(n)}\"></a>\n## {n}\n")
    print(f"单步，tier={cfg['tier']}，MTP{cfg['mtp']}，{d['device_name']}，commit `{d['git_commit'][:8]}`；"
          f"远端带宽 DMA {bw['dma_contiguous_GBps']:.1f} / gather {bw['sm_gather_random_GBps']:.1f} GB/s。"
          f"每格：中位数（p10～p90）。[原始 JSON]({n})\n")
    for r in d["results"]:
        print(f"**{r['requests']} 请求（{r['query_rows']} 行）**\n")
        print("| 远端占比 | 实际占比 | 去重率 | 去重远端 MB | " + " | ".join(NAMES.get(m, m) for m in modes) + " |")
        print("|---:|---:|---:|---:|" + "---:|" * len(modes))
        for c in r["cases"]:
            ub = sorted(c["far_unique_bytes"])[len(c["far_unique_bytes"]) // 2] / 1e6
            cells = [f"{c[m]['median_us']:.1f} ({c[m]['p10_us']:.0f}～{c[m]['p90_us']:.0f})" for m in modes]
            print(f"| {c['far_fraction']:.0%} | {c['far_fraction_actual']:.3f} | {c['unique_ratio']:.3f} | {ub:.1f} | " + " | ".join(cells) + " |")
        print()

for n in ROUNDS:
    d = json.loads((O / n).read_text())
    cfg = d["configuration"]; bw = d["bandwidth"]
    modes = cfg["modes"]
    print(f"<a id=\"{anchor(n)}\"></a>\n## {n}\n")
    print(f"跨 decode，tier={cfg['tier']}，MTP{cfg['mtp']}，每请求 {cfg['capacity']} 槽，{cfg['steps']} 步，{cfg['passes']} 次冷启动 pass，"
          f"commit `{d['git_commit'][:8]}`；远端 gather {bw['sm_gather_random_GBps']:.1f} GB/s。每格：该步中位数。[原始 JSON]({n})\n")
    for r in d["results"]:
        ours = next((m for m in ("ours_stage", "ours_inplace") if m in r["cached_misses"]), None)
        print(f"**{r['requests']} 请求（{r['query_rows']} 行）**\n")
        hdr = ["步"] + [NAMES.get(m, m) for m in modes] + ["ours 未命中页", "不开缓存的去重远端页"]
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---:|" * len(hdr))
        for t in range(r["steps"]):
            cells = [f"{r['per_step'][m][t]['median_us']:.1f}" for m in modes]
            miss = r["cached_misses"][ours][t] if ours else "-"
            print(f"| {t} | " + " | ".join(cells) + f" | {miss} | {r['far_unique_uncached'][t]} |")
        print()
