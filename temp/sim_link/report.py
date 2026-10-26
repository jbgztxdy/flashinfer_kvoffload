# Purpose: tables for the emulated far-link runs (temp/sim_link/{bw,lat}): every method with both
#          floors — (a) max(all-HBM compute, deduplicated missed bytes / link bandwidth) and
#          (b) all-HBM compute alone (the native kernel with every selected row in local HBM).
# Usage: python3 temp/sim_link/report.py > temp/sim_link/results.md
import glob
import json
from pathlib import Path

ROW = 1152
D = Path("temp/sim_link")


def steady(r, m):
    t = range(r["steps"] // 2, r["steps"])
    return sum(r["per_step"][m][x]["median_us"] for x in t) / len(t)


def misses(r):
    m = r["cached_misses"]
    key = "ours_stage" if "ours_stage" in m else "ours_inplace"
    t = range(r["steps"] // 2, r["steps"])
    return sum(m[key][x] for x in t) / len(t)


def by_req(path):
    return {r["requests"]: r for r in json.loads(Path(path).read_text())["results"]} if Path(path).exists() else {}


def table(title, runs, floors, link_gbps, ours_modes, note=""):
    print(f"\n### {title}\n")
    if note:
        print(note + "\n")
    cols = [n for n, _ in ours_modes] + ["Memcpy+LRU", "Direct+LRU"]
    print("| 请求 | " + " | ".join(cols) + " | **本文最优** | 下限 a：max(计算, 传输) | 下限 b：纯本地计算 | 本文 ÷ a | 本文 ÷ b | 本文 ÷ 朴素较快者 |")
    print("|---:|" + "---:|" * (len(cols) + 6))
    for q in sorted(floors):
        cells, ours = [], []
        for name, (path, mode) in ours_modes:
            r = runs[path].get(q)
            v = steady(r, mode) if r else None
            cells.append(f"{v:.0f}" if v else "-")
            if v:
                ours.append(v)
        base = runs[ours_modes[0][1][0]][q]
        mc, dl = steady(base, "memcpy_lru"), steady(base, "direct_lru")
        best = min(ours)
        compute = floors[q]
        transfer = misses(base) * ROW / link_gbps / 1e3
        fa = max(compute, transfer)
        print(f"| {q} | " + " | ".join(cells) + f" | {mc:.0f} | {dl:.0f} | **{best:.0f}** | {fa:.0f} | {compute:.0f} | {best / fa:.2f} | {best / compute:.2f} | {best / min(mc, dl):.2f} |")


def main():
    print("# 模拟远端链路：带宽与延迟（A100，NVLink 远端 = GPU1 HBM，跨 decode 稳态，开 LRU，每请求 4096 槽，µs）\n")
    print("由 `temp/sim_link/report.py` 从同目录的 JSON 生成。两种理论下限：")
    print("- **下限 a** = max(纯本地计算, 本步去重后的未命中字节 ÷ 模拟链路带宽)：本文 kernel 在这条链路上能达到的极限。")
    print("- **下限 b** = 纯本地计算：所有选中行都在本地 HBM、零传输时原生 kernel 的耗时（正常构建、无模拟钩子），即 KV 全放 HBM 时的性能。")
    print("- 「不限速」行的传输下限按 NVLink 实测峰值 262 GB/s 计。所有本文与朴素方法都在同一个模拟构建下测（含钩子开销），下限 b 在正常构建下测，对本文偏保守。\n")
    for mtp, reqs in ((8, "MTP8"), (1, "MTP1")):
        floors = {q: steady(r, "floor_fa2") for q, r in by_req(D / f"bw/floor_mtp{mtp}.json").items()}
        print(f"\n## 带宽：{reqs}")
        for g, label in ((0, "不限速（262 GB/s）"), (100, "100 GB/s"), (50, "50 GB/s")):
            files = {p: by_req(p) for p in (D / f"bw/g{g}_mtp{mtp}.json", D / f"bw/g{g}_mtp{mtp}_f16.json")}
            val = json.loads((D / f"bw/validate_{g}.json").read_text())
            note = f"运行前链路实测：带宽 {val['measured_GBps']:.1f} GB/s，依赖读延迟 {val['measured_latency_ns'] / 1000:.2f} µs。"
            table(f"{reqs}，{label}", files, floors, g or 262,
                  [("原地读", (D / f"bw/g{g}_mtp{mtp}.json", "ours_inplace")),
                   ("staged 8 CTA", (D / f"bw/g{g}_mtp{mtp}.json", "ours_stage")),
                   ("staged 16 CTA", (D / f"bw/g{g}_mtp{mtp}_f16.json", "ours_stage"))], note)
    lat = sorted(glob.glob(str(D / "lat/g*_l*_mtp8.json")))
    if lat:
        for mtp, reqs in ((8, "MTP8"), (1, "MTP1")):
            floors = {q: steady(r, "floor_fa2") for q, r in by_req(D / f"bw/floor_mtp{mtp}.json").items()}
            print(f"\n## 延迟：{reqs}")
            for L in (2000, 5000, 10000, 20000):
                for g, label in ((0, "不限速"), (100, "100 GB/s"), (50, "50 GB/s")):
                    p = D / f"lat/g{g}_l{L}_mtp{mtp}.json"
                    if not p.exists():
                        continue
                    vp = D / f"lat/validate_g{g}_l{L}.json"
                    val = json.loads(vp.read_text()) if vp.exists() else None
                    note = (f"运行前链路实测：带宽 {val['measured_GBps']:.1f} GB/s，依赖读延迟 {val['measured_latency_ns'] / 1000:.2f} µs。"
                            if val else "")
                    modes = ([("原地读", (p, "ours_inplace")), ("staged 16 CTA", (D / f"lat/g{g}_l{L}_mtp8_f16.json", "ours_stage"))]
                             if mtp == 8 else [("原地读", (p, "ours_inplace")), ("staged 16 CTA", (p, "ours_stage"))])
                    files = {m[1][0]: by_req(m[1][0]) for m in modes}
                    table(f"{reqs}，附加延迟 {L / 1000:.0f} µs，{label}", files, floors, g or 262, modes, note)


if __name__ == "__main__":
    main()
