"""Compare fixed/old full matrices and report the reproduced regression separately."""
import json
from pathlib import Path

dest = Path(__file__).resolve().parent
root = dest.parents[2]


def read(path):
    return json.loads(path.read_text())


def steady(row, mode):
    tail = row["per_step"][mode][row["steps"] // 2:]
    return sum(s["median_us"] for s in tail) / len(tail)


def validate(data):
    cfg = data["configuration"]
    for row in data["results"]:
        for mode in cfg["modes"]:
            for index in range(cfg["passes"]):
                trials = [g for g in row["enqueue_guards"] if g["mode"] == mode and g["pass_index"] == index]
                accepted = [g for g in trials if g["attempt"] == max(v["attempt"] for v in trials)]
                assert all(g["valid"] for g in accepted)
                assert sum(g["last_step_exclusive"] - g["first_step"] for g in accepted) == row["steps"]
        assert all(error < .01 for error in row["correctness_max_abs_error"].values())


def floor(cfg, requests):
    if cfg["captures"].startswith("temp/glm_trace/assistant"):
        paths = [root / f"temp/glm_trace/results/a100_64k_assistant_20261029/peer_mtp{cfg['mtp']}.json"]
        if cfg["mtp"] == 1:
            paths += [root / "temp/sim_link/tune_latency/floor_assistant_mtp1.json"]
    else:
        paths = [root / f"temp/sim_link/tune_lru/floor_ctx65536_mtp{cfg['mtp']}.json"]
    return min(v for path in paths for row in read(path)["results"] if row["requests"] == requests
               for step in row["per_step"]["floor_fa2"] for v in step["samples_us"])


summary = []


def report():
    yield "# 大 batch 回退修复：同参数对照\n"
    yield ("old为109b1c76，regressed为5c44fd70，fixed为编译期分派的修复。"
           "固定fetch_ctas=16/host_item_entries=64、BF16/64头、4096槽/请求、7次冷启动，稳态取后一半步逐步中位数平均。"
           "代理trace为64K/32连续位置；assistant为64K/1024完整assistant窗口。"
           "MTP8 128请求是1024行。\n")
    yield ("下表old_full在fixed之后复测。a为逐步max(b,去重未命中字节/带宽)的稳态平均；"
           "b为同trace/形状正常构建纯HBM最好实测样本。固定延迟不计入a，比值不截断。\n")
    for path in sorted(dest.glob("fixed_*.json")):
        case = path.name.removeprefix("fixed_")
        if case.startswith("repeat_") or "configuration" not in read(path):
            continue
        old_path = dest / ("old_full_" + case)
        if not old_path.exists():
            continue
        new, old = read(path), read(old_path)
        validate(new)
        validate(old)
        cfg = new["configuration"]
        assert cfg | {"output": None} == old["configuration"] | {"output": None}
        regress_path = dest / ("regressed_" + case)
        regressed = read(regress_path) if regress_path.exists() else None
        if regressed:
            validate(regressed)
        bw = new["emulated_link"]["gbps"] if new["emulated_link"] else max(new["bandwidth"].values())
        yield f"## {case.removesuffix('.json')}\n"
        yield "| 请求 | 行 | 模式 | 优化前µs | 回退版µs | 修复µs | 修复÷优化前 | aµs | bµs | 修复÷a | 修复÷b |"
        yield "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        for row, before in zip(new["results"], old["results"]):
            assert row["requests"] == before["requests"]
            b = floor(cfg, row["requests"])
            for mode in cfg["modes"]:
                old_us, new_us = steady(before, mode), steady(row, mode)
                regress_us = next((steady(r, mode) for r in regressed["results"]
                                   if r["requests"] == row["requests"]), None) if regressed else None
                a = sum(max(b, m * 1152 / bw / 1000)
                        for m in row["cached_misses"][mode][row["steps"] // 2:]) / (row["steps"] - row["steps"] // 2)
                assert new_us >= a >= b
                rg = f"{regress_us:.2f}" if regress_us else "—"
                yield (f"| {row['requests']} | {row['query_rows']} | {mode} | {old_us:.2f} | {rg} | {new_us:.2f} | "
                       f"{new_us/old_us:.3f} | {a:.2f} | {b:.2f} | {new_us/a:.2f} | {new_us/b:.2f} |")
                summary.append(dict(case=case, requests=row["requests"], query_rows=row["query_rows"], mode=mode,
                                    old_us=old_us, regressed_us=regress_us, fixed_us=new_us,
                                    fixed_to_old=new_us/old_us, floor_a_us=a, floor_b_us=b,
                                    ratio_a=new_us/a, ratio_b=new_us/b))
        yield ""
    yield "## 第二次关键场景复测\n"
    yield "下表为完整反向矩阵之后再次测fixed、再测regressed；其余参数相同。\n"
    yield "| MTP | 请求 | 模式 | 再测修复µs | 再测回退µs |"
    yield "|---:|---:|---|---:|---:|"
    for mtp in (1, 8):
        new = read(dest / f"fixed_repeat_proxy_g100_mtp{mtp}.json")
        bad = read(dest / f"regressed_repeat_proxy_g100_mtp{mtp}.json")
        validate(new)
        validate(bad)
        assert new["configuration"] | {"output": None} == bad["configuration"] | {"output": None}
        for row, before in zip(new["results"], bad["results"]):
            assert row["requests"] == before["requests"]
            for mode in new["configuration"]["modes"]:
                yield f"| {mtp} | {row['requests']} | {mode} | {steady(row, mode):.2f} | {steady(before, mode):.2f} |"


(dest / "results.md").write_text("\n".join(report()).rstrip() + "\n")
(dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
