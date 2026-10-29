"""Final single-query comparisons, including regressions and both measured floors."""
import json
from pathlib import Path

dest = Path(__file__).resolve().parent
root = dest.parents[2]
summary = []


def steady(row, mode):
    tail = row["per_step"][mode][row["steps"] // 2:]
    return sum(s["median_us"] for s in tail) / len(tail)


def report():
    yield "# A100 单 query 合并：同参数对照\n"
    yield ("基线为 aeca4403 的 kernel（与 614655f1 相同）。每格固定 f16/h64、4096 槽/请求，"
           "7 pass，稳态取后一半步的逐步中位数平均。正值为耗时下降，负值为回退。"
           "本表采用候选后重跑的基线；另有基线先跑的数据和未采用候选，均保留。\n")
    yield ("代理 trace 为 64K/32 连续位置；glm64k 为真实 GLM 的 64K/1024 混合角色位置，"
           "不是完整 assistant 窗口。MTP8 的 128 请求为 1024 行，属于大 batch。\n")
    yield ("下限 b 采用此前相同维度正常构建全 HBM attention 的最快样本；a = 逐步 "
           "max(b, 去重未命中字节/带宽) 的稳态平均。模拟链路用设置带宽，真实链路用当次 DMA/SM "
           "gather 较快值。固定延迟未计入 a；不截断比值。\n")
    for path in sorted(dest.glob("guarded_candidate_*.json")):
        case = path.name.removeprefix("guarded_candidate_")
        new = json.loads(path.read_text())
        baseline = json.loads((dest / ("guarded_base_" + case)).read_text())
        cfg = new["configuration"]
        assert cfg | {"output": None} == baseline["configuration"] | {"output": None}
        if case.startswith("glm64k_"):
            floor_path = dest / "guarded_floor_glm64k_peer_mtp1.json"
        else:
            floor_path = root / f"temp/sim_link/tune_lru/floor_ctx65536_mtp{cfg['mtp']}.json"
        floor = {r["requests"]: r for r in json.loads(floor_path.read_text())["results"]}
        bw = new["emulated_link"]["gbps"] if new["emulated_link"] else max(new["bandwidth"].values())
        yield f"## {case.removesuffix('.json')}\n"
        yield f"计算下限来源：`{floor_path.relative_to(root)}`。\n"
        yield "| 请求 | 行 | 模式 | 旧版 µs | 新版 µs | 耗时下降 | a µs | b µs | 新版÷a | 新版÷b | max abs |"
        yield "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        for row, old in zip(new["results"], baseline["results"]):
            assert row["requests"] == old["requests"]
            for result in (row, old):
                for mode in cfg["modes"]:
                    for pass_index in range(cfg["passes"]):
                        guards = [g for g in result["enqueue_guards"]
                                  if g["mode"] == mode and g["pass_index"] == pass_index]
                        accepted = [g for g in guards if g["attempt"] == max(v["attempt"] for v in guards)]
                        assert all(g["valid"] for g in accepted)
                        assert sum(g["last_step_exclusive"]-g["first_step"] for g in accepted) == result["steps"]
            b = min(sample for s in floor[row["requests"]]["per_step"]["floor_fa2"]
                    for sample in s["samples_us"])
            for mode in ("ours_stage", "ours_inplace"):
                before, after = steady(old, mode), steady(row, mode)
                tail = range(row["steps"] // 2, row["steps"])
                a = sum(max(b, row["cached_misses"][mode][t] * 1152 / bw / 1000) for t in tail) / len(tail)
                error = row["correctness_max_abs_error"][mode]
                assert error < .01 and after >= a >= b
                yield (f"| {row['requests']} | {row['query_rows']} | {mode} | {before:.2f} | {after:.2f} | "
                       f"{1-after/before:.1%} | {a:.2f} | {b:.2f} | {after/a:.2f} | {after/b:.2f} | {error:.6f} |")
                summary.append(dict(case=case, requests=row["requests"], mode=mode,
                                    old_us=before, new_us=after, reduction=1-after/before,
                                    floor_a_us=a, floor_b_us=b, ratio_a=after/a, ratio_b=after/b,
                                    max_abs=error))
        yield ""


(dest / "results.md").write_text("\n".join(report()).rstrip() + "\n")
(dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
