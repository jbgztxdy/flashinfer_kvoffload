"""Same-parameter latency comparisons, retaining every slower cell and both floors."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("label")
parser.add_argument("--base", default="base")
args = parser.parse_args()
dest = Path(__file__).resolve().parent
root = dest.parents[2]
summary = []


def read(path):
    return json.loads(path.read_text())


def avg(row, mode):
    tail = row["per_step"][mode][row["steps"] // 2:]
    return sum(s["median_us"] for s in tail) / len(tail)


def floor(cfg, requests):
    if cfg["captures"].startswith("temp/glm_trace/assistant"):
        paths = [root / f"temp/glm_trace/results/a100_64k_assistant_20261029/peer_mtp{cfg['mtp']}.json"]
        paths += list(dest.glob(f"floor_assistant_mtp{cfg['mtp']}*.json"))
    else:
        paths = [root / f"temp/sim_link/tune_lru/floor_ctx65536_mtp{cfg['mtp']}.json"]
    values = [v for path in paths for row in read(path)["results"] if row["requests"] == requests
              for step in row["per_step"]["floor_fa2"] for v in step["samples_us"]]
    return min(values)


def report():
    yield "# A100 高延迟：同参数旧新对照\n"
    yield ("基线109b1c76，4096槽/请求，默认f16/h64，tuned段落使用列出的f32；每格7次冷启动。代理为64K/32连续位置；"
           "assistant为真实64K/1024完整assistant窗口。稳态为后一半步的逐步中位数平均。\n")
    yield ("b为相同trace/形状正常构建纯HBM attention的最快样本；a为逐步max(b, "
           "去重未命中字节/带宽)的稳态平均。模拟场景用设置的50/100GB/s；固定延迟未计入a，"
           "所以它是乐观下限。比值不截断。原始JSON保存全部逐步pass、未命中和入队检查。\n")
    for path in sorted(dest.glob(args.label + "_*.json")):
        new = read(path)
        if "configuration" not in new:
            continue
        case = path.name.removeprefix(args.label + "_")
        old = read(dest / (args.base + "_" + case))
        cfg = new["configuration"]
        assert cfg | {"output": None} == old["configuration"] | {"output": None}
        assert len(new["results"]) == len(old["results"]) == len(cfg["requests"])
        bw = new["emulated_link"]["gbps"] if new["emulated_link"] else max(new["bandwidth"].values())
        yield f"## {case.removesuffix('.json')}\n"
        yield f"fetch CTA={cfg['fetch_ctas']}，host_item_entries={new['planner_overrides']['host_item_entries'] or 64}。\n"
        yield "| 请求 | 行 | 模式 | 旧版µs | 新版µs | 耗时下降 | aµs | bµs | 新版÷a | 新版÷b | max abs |"
        yield "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        for row, before in zip(new["results"], old["results"]):
            assert row["requests"] == before["requests"]
            for result in (row, before):
                guards = result["enqueue_guards"]
                for mode in cfg["modes"]:
                    for pass_index in range(cfg["passes"]):
                        attempts = [g for g in guards if g["mode"] == mode and g["pass_index"] == pass_index]
                        accepted = [g for g in attempts if g["attempt"] == max(a["attempt"] for a in attempts)]
                        assert all(g["valid"] for g in accepted)
                        assert sum(g["last_step_exclusive"] - g["first_step"] for g in accepted) == row["steps"]
            b = floor(cfg, row["requests"])
            for mode in row["per_step"]:
                a = sum(max(b, row["cached_misses"][mode][t] * 1152 / bw / 1000)
                        for t in range(row["steps"] // 2, row["steps"])) / (row["steps"] - row["steps"] // 2)
                old_us, new_us = avg(before, mode), avg(row, mode)
                assert new_us >= a >= b and row["correctness_max_abs_error"][mode] < .01
                yield (f"| {row['requests']} | {row['query_rows']} | {mode} | {old_us:.2f} | {new_us:.2f} | "
                       f"{1-new_us/old_us:.1%} | {a:.2f} | {b:.2f} | {new_us/a:.2f} | {new_us/b:.2f} | "
                       f"{row['correctness_max_abs_error'][mode]:.6f} |")
                summary.append(dict(case=case, requests=row["requests"], mode=mode, old_us=old_us,
                                    new_us=new_us, reduction=1-new_us/old_us, floor_a_us=a, floor_b_us=b,
                                    ratio_a=new_us/a, ratio_b=new_us/b))
        yield ""


(dest / (args.label + "_results.md")).write_text("\n".join(report()).rstrip() + "\n")
(dest / (args.label + "_summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
