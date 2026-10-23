# Purpose: report tables for temp/a100_full (A100, current kernel).
import json
O = "temp/a100_full/"
L = lambda n: json.load(open(O + n))



def compute_floor(result):
    """Fastest all-local attention measured for this batch: the native kernel with every row
    remapped to HBM, or either kernel at 0% far (native's fixed KV split is not optimal at every
    shape, so taking only it would put the floor above what was measured)."""
    zero = next(c for c in result["cases"] if c["far_fraction"] == 0)
    return min(min(c["attn"]["median_us"] for c in result["cases"]), zero["batch"]["median_us"], zero["direct"]["median_us"])

def sweep(name):
    d = L(name); bw = max(d["bandwidth"]["sm_gather_random_GBps"], d["bandwidth"]["dma_contiguous_GBps"])
    fr = [c["far_fraction"] for c in d["results"][0]["cases"]]
    print(f"\n#### {name}（远端带宽 {bw:.0f} GB/s）：ours µs（÷下限 / ÷朴素较快者）\n")
    print("| 请求 | 行 | " + " | ".join(f"{f:.0%}" for f in fr) + " |")
    print("|---:|---:|" + "---:|" * len(fr))
    lose = []
    for r in d["results"]:
        cells = []
        for c in r["cases"]:
            o = c["batch"]["median_us"]; nv = min(c["copy"]["median_us"], c["direct"]["median_us"])
            ub = sorted(c["far_unique_bytes"])[len(c["far_unique_bytes"]) // 2]
            fl = max(compute_floor(r), ub / bw / 1e3)
            cells.append(f"{o:.0f} ({o / fl:.2f}/{o / nv:.2f})")
            if c["far_fraction"] > 0 and o >= nv:
                lose.append(f"{r['requests']}请求/{c['far_fraction']:.0%}")
        print(f"| {r['requests']} | {r['query_rows']} | " + " | ".join(cells) + " |")
    print(f"\n远端占比 > 0 时输给朴素较快者的格子：{', '.join(lose) if lose else '无'}")


def rounds(name, ours):
    d = L(name); bw = max(d["bandwidth"]["sm_gather_random_GBps"], d["bandwidth"]["dma_contiguous_GBps"])
    print(f"\n#### {name}：稳态 µs（后一半步的平均）\n")
    print("| 请求 | 行 | ours | 不开缓存 | memcpy_lru | direct_lru | 计算下限 | 传输下限 | ours÷max(下限) | ours÷朴素较快者 | 缓存加速 |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in d["results"]:
        t = range(r["steps"] // 2, r["steps"])
        st = lambda m: sum(r["per_step"][m][x]["median_us"] for x in t) / len(t)
        mi = sum(r["cached_misses"][ours][x] for x in t) / len(t)
        ft = mi * 1152 / bw / 1e3; fc = st("floor_fa2"); o = st(ours)
        nv = min(st("memcpy_lru"), st("direct_lru")); nc = st("ours_nocache")
        print(f"| {r['requests']} | {r['query_rows']} | {o:.0f} | {nc:.0f} | {st('memcpy_lru'):.0f} | {st('direct_lru'):.0f} | {fc:.0f} | {ft:.0f} | "
              f"{o / max(fc, ft):.2f} | {o / nv:.2f} | {nc / o:.2f} |")


def capacity(tier, ours):
    print(f"\n#### LRU 容量，{tier}，MTP8：ours µs（每步未命中页）/ 朴素较快者 µs\n")
    names = {2048: f"rounds_{tier}_mtp8_c2048.json", 4096: f"rounds_{tier}_mtp8.json", 8192: f"rounds_{tier}_mtp8_c8192.json"}
    ds = {c: L(n) for c, n in names.items()}
    reqs = [r["requests"] for r in ds[2048]["results"]]
    print("| 请求 | " + " | ".join(f"{c} 槽" for c in ds) + " |")
    print("|---:|" + "---:|" * len(ds))
    for req in reqs:
        cells = []
        for c, d in ds.items():
            r = next(x for x in d["results"] if x["requests"] == req)
            t = range(r["steps"] // 2, r["steps"])
            st = lambda m: sum(r["per_step"][m][x]["median_us"] for x in t) / len(t)
            mi = sum(r["cached_misses"][ours][x] for x in t) / len(t)
            cells.append(f"{st(ours):.0f} ({mi:.0f}) / {min(st('memcpy_lru'), st('direct_lru')):.0f}")
        print(f"| {req} | " + " | ".join(cells) + " |")


print("# A100 比值表\n")
print("每格：ours 中位数 µs（÷理论下限 / ÷朴素较快者）。由 `temp/a100_full/tables.py` 生成；说明见 "
      "[`docs/mla_host_a100_full.md`](../../docs/mla_host_a100_full.md)，每种方法的全部测量值见 [`full_tables.md`](full_tables.md)，原始数据为同目录的 JSON。")
print("\n理论下限 = max(计算下限, 传输下限)：计算下限取全本地时测得的最快 attention（原生全本地，或 0% 远端时两种 kernel），传输下限用链路实测峰值 max(DMA, SM gather)。"
      "唯一低于 1 的是 NVLink、MTP1、1 请求、25%～30% 远端两格（0.95、0.93）：单行时 staged 模式把 Host 条目单独拆成 item，"
      "KV 拆分比原生 kernel 更细，计算本身更快；这是单步 NVLink 的非推荐配置，不出现在报告中。")
for n in ("sweep_host_mtp8.json", "sweep_peer_mtp8.json", "sweep_host_mtp1.json", "sweep_peer_mtp1.json"):
    sweep(n)
rounds("rounds_host_mtp8.json", "ours_stage"); rounds("rounds_peer_mtp8.json", "ours_inplace")
rounds("rounds_host_mtp1.json", "ours_stage"); rounds("rounds_peer_mtp1.json", "ours_inplace")
capacity("host", "ours_stage"); capacity("peer", "ours_inplace")
