"""Report the serial old/new A100 runs, including both lower bounds and slower cases."""
import json
from pathlib import Path

D = Path("temp/sim_link/tune_100")


def rows(name):
    return {r["requests"]: r for r in json.loads((D / f"{name}.json").read_text())["results"]}


def steady(row, mode):
    return sum(s["median_us"] for s in row["per_step"][mode][row["steps"] // 2:]) / (row["steps"] - row["steps"] // 2)


def emit():
    yield "# A100：100 GB/s 高延迟与小 batch 调优（2026-10-28）\n"
    yield ("旧版基线取 c7921958 的 kernel，16 个 fetch CTA、host_item_entries=64。"
           "新版仅改变 fetch CTA 多于 8 个时的 16/64 行搬运块选择，另扫描 fetch 数和 attention item 大小。"
           "表中最优值从列明的新配置中取，包含调参收益；同参数对照单独列出。\n")
    yield ("f 为 fetch CTA 数，h 为 host_item_entries（远端 attention 任务的目标 KV 条目数）。"
           "比较基线固定为 f16/h64，不代表旧版所有参数中的最优值；例如无附加延迟的 MTP8、4 请求，"
           "此前 f8 的结果已接近本次 f16/h256。\n")
    yield ("GPU0/1 为 A100-SXM4-40GB，NV12；模拟带宽 100 GB/s，延迟为附加值。"
           "BF16、64 头、top-k 2048、64K 上下文、4096 LRU 槽/请求；"
           "使用此前 DeepSeek-V2-Lite 代理 top-k，MTP8 为 4 步、MTP1 为 32 步，"
           "每种配置 7 次冷启动，取各步中位数后对后一半步平均。GLM 正式 64K trace 仍排队。\n")
    yield ("b = 正常构建、纯 HBM FA2 attention 的最好实测样本；"
           "a = 逐步 max(b, 本文去重后未命中页 × 1152 B / 模拟带宽) 的稳态平均。"
           "a 是未计依赖延迟的乐观下限。比值直接计算，未截断。朴素方法 CPU LRU 不计时；"
           "本次远端为 peer GPU，其 gather 计时。\n")
    violations = []
    for mtp, latency in ((8, 0), (8, 20000), (1, 20000)):
        base = rows(f"verify_base_mtp{mtp}_l{latency}")
        same = rows(f"verify_final_mtp{mtp}_l{latency}")
        candidates = [("f16/h64", same)]
        if mtp == 8 and latency == 20000:
            candidates.append(("f24/h64", rows("verify_final_mtp8_l20000_f24")))
        if mtp == 8:
            candidates.append(("f16/h256", rows(f"verify_final_mtp8_l{latency}_h256")))
        if mtp == 1:
            candidates.append(("f32/h64", rows("verify_final_mtp1_l20000_f32")))
        floors = rows(f"floor_mtp{mtp}")
        yield f"## MTP{mtp}，100 GB/s，附加 {latency / 1000:g} µs\n"
        yield "| 请求 | query 行 | 旧版基线 µs | 新版同参数 µs | 新版最优 µs | 最优配置 | 对基线耗时下降 | Memcpy+LRU µs | Direct+LRU µs | a µs | b µs | 新版÷a | 新版÷b |"
        yield "|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"
        for q, old in base.items():
            label, row = min(((label, rs[q]) for label, rs in candidates if q in rs), key=lambda x: steady(x[1], "ours_stage"))
            t = steady(row, "ours_stage")
            previous = steady(old, "ours_stage")
            b = min(x for step in floors[q]["per_step"]["floor_fa2"] for x in step["samples_us"])
            tail = row["cached_misses"]["ours_stage"][row["steps"] // 2:]
            a = sum(max(b, x * 1152 / 100 / 1000) for x in tail) / len(tail)
            if t < a or t < b:
                violations.append(dict(mtp=mtp, latency=latency, requests=q, time=t, a=a, b=b))
            yield (f"| {q} | {q*mtp} | {previous:.2f} | {steady(same[q], 'ours_stage'):.2f} | {t:.2f} | {label} | "
                   f"{1-t/previous:.1%} | {steady(old,'memcpy_lru'):.2f} | {steady(old,'direct_lru'):.2f} | "
                   f"{a:.2f} | {b:.2f} | {t/a:.2f} | {t/b:.2f} |")
        yield ""
    yield "## 原始结果与验证\n"
    names = sorted(p for p in D.glob("verify_*.json")) + sorted(D.glob("floor_*.json"))
    count = 0
    error = 0
    for path in names:
        data = json.loads(path.read_text())
        assert len(data["results"]) == len(data["configuration"]["requests"]), path
        for r in data["results"]:
            for value in r["correctness_max_abs_error"].values():
                assert value < .01, (path, value)
                error = max(error, value)
            for mode, misses in r["cached_misses"].items():
                assert len(misses) == r["steps"]
                assert all(0 <= m <= u for m, u in zip(misses, r["far_unique_uncached"]))
            count += 1
    yield f"最终核查 {len(names)} 份计时 JSON、{count} 个批次配置；本文每步采样行正确性通过，最大绝对误差 {error:.6f}。"
    yield "\n`verification_runs.json` 保存最终命令和源码哈希；同目录保留初始扫描及未采用方案的数据。"
    yield "\n未采用：四级 fetch 流水在一些小 batch 有收益，但固定 f16 时大 batch 回退；单 warp 前缀扫描收益不足；缩小 cooperative grid 损失 attention 并行度。候选补丁和计时数据均保留，未进入最终 kernel。"
    yield "\n50 GB/s＋20 µs 的新版检查保存为 `verify_final_mtp8_g50_l20000.json`，未据其声明达到纯 device 性能。"
    assert not violations, violations


if __name__ == "__main__":
    content = "\n".join(emit()).rstrip() + "\n"
    (D / "results.md").write_text(content)
    print(content)
