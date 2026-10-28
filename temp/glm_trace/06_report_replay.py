"""Report real top-k replay timings, unique-page hit rates and both lower bounds.

Usage: python 06_report_replay.py results/a100_*.json --out results/replay.md
"""
import argparse
import json
from pathlib import Path


def report(paths, probes):
    yield "# GLM top-k 多步回放\n"
    yield ("命中率按远端去重页计：1 − 本步实测未命中页数 ÷ 不开缓存时本步远端去重页数。"
           "它不是按 top-k 引用数计的命中率；首步冷启动，摘要取后一半步。\n")
    yield ("下限 b 为同一批次纯本地 attention 的实测最快样本（含显式指定的全本地补测）；下限 a = max(b, "
           "本步未命中页数 × 1152 B ÷ 实测链路带宽)。链路带宽取 DMA/SM gather 较快值。"
           "比值直接计算，不截断到 1。\n")
    yield ("朴素方法仅用于计时：CPU LRU 记账不计时，PCIe Memcpy 的 CPU gather 也不计时；"
           "其 attention 不验证数值。本文 staged/原地读会逐步验证采样行。"
           "查询和 KV 为随机 BF16，top-k 来自 GLM；HiSparse 需 H800，本机 A100 不测。\n")
    if probes:
        yield ("全本地补测文件：" + "、".join(f"`{p}`" for p in probes) +
               "。补测用 CUDA graph 连续执行，0% 远端，候选含原生 attention 和本文无缓存路径；"
               "与逐步回放的计时方式不同，用于取最好实测计算下限。\n")
    for path in paths:
        d = json.loads(path.read_text())
        cfg = d["configuration"]
        bw = max(d["bandwidth"].values())
        yield f"## {path.name}\n"
        yield (f"{d['device_name']}，{cfg['tier']}，context {cfg['context']}，MTP{cfg['mtp']}，"
               f"{cfg['steps']} 步，{cfg['passes']} 次冷启动，LRU {cfg['capacity']} 槽/请求，"
               f"带宽 {bw:.2f} GB/s。capture：`{cfg['captures']}`。\n")
        if cfg["context"] < 65536 or cfg["steps"] * cfg["mtp"] < 1024:
            yield "**仅用于接入检查；上下文长度或连续位置数未达到正式 64K/1024 要求。**\n"
        yield "| 请求 | 本文 | 稳态 µs | 去重页命中率 | 下限 a µs | 下限 b µs | 本文÷a | 本文÷b | Memcpy+LRU µs | Direct+LRU µs |"
        yield "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        details = []
        for r in d["results"]:
            steps = r["steps"]
            tail = range(steps // 2, steps)
            b = min(x for s in r["per_step"]["floor_fa2"] for x in s["samples_us"])
            for probe in probes:
                p = json.loads(probe.read_text())
                if any(p["configuration"][k] != cfg[k] for k in ("tier", "mtp", "context", "topk")):
                    continue
                assert p["device_name"] == d["device_name"]
                for other in p["results"]:
                    if other["requests"] != r["requests"]:
                        continue
                    for case in other["cases"]:
                        if case["far_fraction"] == 0:
                            for mode in ("attn", "direct", "batch"):
                                if mode in case:
                                    b = min(b, min(case[mode]["samples_us"]))
            avg = lambda values: sum(values[t] for t in tail) / len(tail)
            naive = {m: avg([s["median_us"] for s in r["per_step"][m]])
                     for m in ("memcpy_lru", "direct_lru")}
            for mode, misses in r["cached_misses"].items():
                times = [s["median_us"] for s in r["per_step"][mode]]
                a = [max(b, m * 1152 / bw / 1000) for m in misses]
                unique = r["far_unique_uncached"]
                hit = 1 - sum(misses[t] for t in tail) / sum(unique[t] for t in tail)
                duration, floor_a = avg(times), avg(a)
                yield (f"| {r['requests']} | {mode} | {duration:.2f} | {hit:.2%} | {floor_a:.2f} | "
                       f"{b:.2f} | {duration/floor_a:.2f} | {duration/b:.2f} | "
                       f"{naive['memcpy_lru']:.2f} | {naive['direct_lru']:.2f} |")
                details.append((r, mode, b, times, misses, a))
        yield "\n逐步数据（包括冷启动，JSON 保存全部 pass 样本）：\n"
        for r, mode, b, times, misses, a in details:
            yield f"### {r['requests']} 请求，{mode}\n"
            yield f"采样行最大绝对误差：{r['correctness_max_abs_error'][mode]:.6f}。\n"
            yield "| 步 | 远端去重页 | 未命中页 | 去重页命中率 | 本文 µs | 下限 a µs | 下限 b µs | 本文÷a | 本文÷b |"
            yield "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
            for t, (us, m, lower) in enumerate(zip(times, misses, a)):
                u = r["far_unique_uncached"][t]
                hit = 1 - m / u if u else 1.0
                yield (f"| {t} | {u} | {m} | {hit:.2%} | {us:.2f} | {lower:.2f} | {b:.2f} | "
                       f"{us/lower:.2f} | {us/b:.2f} |")
            yield ""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--hbm-probes", nargs="*", type=Path, default=[])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(report(args.paths, args.hbm_probes)).rstrip() + "\n")
