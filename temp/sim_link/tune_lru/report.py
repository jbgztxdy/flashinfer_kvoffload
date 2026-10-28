"""Summarize an LRU candidate against this round's fixed-parameter baseline."""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("label")
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent)
args = ap.parse_args()
dest = args.data_dir


def steady(r, mode):
    return sum(s["median_us"] for s in r["per_step"][mode][r["steps"] // 2:]) / (r["steps"] - r["steps"] // 2)


def read(name):
    return json.loads((dest / name).read_text())


def report():
    yield "# LRU 与小 batch 分块：本机 A100 前后对照\n"
    yield ("基线为 d92e7e83；同一 trace、每请求 4096 槽，前后各格使用完全相同的 planner 参数。"
           "本轮不混入上一轮调参收益。稳态为后一半步的逐步中位数平均，单位 µs。\n")
    yield ("代理 trace 为 64K 上下文、32 连续位置；GLM8K 为真实 top-k 的 256 连续位置接入。"
           "每份数据保留选中页的逐步参考校验、未命中页数和计时 pass 样本。\n")
    yield ("下限 b 为本轮正常构建纯 HBM attention 的最快样本；下限 a 为逐步 "
           "max(b, 去重后未命中字节/链路带宽) 的稳态平均。模拟链路采用校准目标带宽；"
           "Host 取当次 DMA/SM gather 实测较快值。比值不截断。\n")
    summary = []
    for path in sorted(dest.glob(args.label + "_*.json")):
        d = json.loads(path.read_text())
        if "configuration" not in d:
            continue
        case = path.name.removeprefix(args.label + "_")
        prefix = {"candidate_repeat": "base_repeat_", "tuned_candidate": "tuned_base_"}.get(args.label, "base_")
        baseline = read(prefix + case)
        old_rows = {r["requests"]: r for r in baseline["results"]}
        assert len(d["results"]) == len(d["configuration"]["requests"])
        assert d["configuration"]["passes"] == baseline["configuration"]["passes"]
        cfg = d["configuration"]
        floors = {r["requests"]: r for r in read(f"floor_ctx{cfg['context']}_mtp{cfg['mtp']}.json")["results"]}
        bw = d["emulated_link"]["gbps"] if d["emulated_link"] else max(d["bandwidth"].values())
        yield f"## {case.removesuffix('.json')}\n"
        yield f"fetch CTA={cfg['fetch_ctas']}，host_item_entries={d['planner_overrides']['host_item_entries'] or 64}。\n"
        yield "| 请求 | query 行 | 模式 | 旧版 | 新版 | 耗时下降 | 去重页命中率（旧→新） | a | b | 新版÷a | 新版÷b | max abs |"
        yield "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        for r in d["results"]:
            q = r["requests"]
            old = old_rows[q]
            for mode in ("ours_stage", "ours_inplace"):
                if mode not in r["per_step"]:
                    continue
                a, b = steady(old, mode), steady(r, mode)
                tail = range(r["steps"] // 2, r["steps"])
                hit = lambda row: 1 - sum(row["cached_misses"][mode][t] for t in tail) / sum(row["far_unique_uncached"][t] for t in tail)
                error = r["correctness_max_abs_error"][mode]
                assert error < .01
                floor_b = min(x for s in floors[q]["per_step"]["floor_fa2"] for x in s["samples_us"])
                floor_a = sum(max(floor_b, r["cached_misses"][mode][t] * 1152 / bw / 1000) for t in tail) / len(tail)
                assert b >= floor_a and b >= floor_b, (case, q, mode, b, floor_a, floor_b)
                yield (f"| {q} | {r['query_rows']} | {mode} | {a:.2f} | {b:.2f} | {1-b/a:.1%} | "
                       f"{hit(old):.2%}→{hit(r):.2%} | {floor_a:.2f} | {floor_b:.2f} | {b/floor_a:.2f} | {b/floor_b:.2f} | {error:.6f} |")
                summary.append(dict(case=case, requests=q, mode=mode, old_us=a, new_us=b,
                                    reduction=1-b/a, hit_old=hit(old), hit_new=hit(r),
                                    floor_a_us=floor_a, floor_b_us=floor_b,
                                    ratio_a=b/floor_a, ratio_b=b/floor_b))
        yield ""
    (dest / (args.label + "_summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
    yield "正值为耗时下降；负值为回退。原始 JSON 保存逐步数据与源码哈希。"


args.out.write_text("\n".join(report()).rstrip() + "\n")
print(args.out)
