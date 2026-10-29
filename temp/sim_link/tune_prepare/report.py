"""Retain paired samples, both floors, and miss counts for the final same-GPU comparison."""
import argparse
import json
from pathlib import Path

dest = Path(__file__).resolve().parent
label = "enabled"


def read(path):
    return json.loads(path.read_text())


def steady(row, mode):
    tail = row["per_step"][mode][row["steps"] // 2:]
    return sum(s["median_us"] for s in tail) / len(tail)


def validate(data):
    cfg = data["configuration"]
    assert cfg["device"] == cfg["peer"] == 0
    for row in data["results"]:
        assert all(error < .01 for error in row["correctness_max_abs_error"].values())
        for mode in cfg["modes"]:
            for step in row["per_step"][mode]:
                assert len(step["samples_us"]) == cfg["passes"]
            for index in range(cfg["passes"]):
                trials = [g for g in row["enqueue_guards"] if g["mode"] == mode and g["pass_index"] == index]
                accepted = sorted((g for g in trials if g["attempt"] == max(v["attempt"] for v in trials)),
                                  key=lambda g: g["first_step"])
                end = 0
                for guard in accepted:
                    assert guard["valid"] and guard["first_step"] == end
                    end = guard["last_step_exclusive"]
                assert end == row["steps"]


summary = []


def floor(cfg, requests):
    if "assistant" in cfg["captures"]:
        paths = [dest / f"floor_assistant_mtp{cfg['mtp']}.json"]
    else:
        paths = [dest / f"{version}_proxy_host_mtp{cfg['mtp']}.json" for version in ("default", "enabled", "retained", "accepted", "final", "base_full")
                 if (dest / f"{version}_proxy_host_mtp{cfg['mtp']}.json").exists()]
    return min(v for path in paths for row in read(path)["results"] if row["requests"] == requests
               for step in row["per_step"]["floor_fa2"] for v in step["samples_us"])


def report():
    yield "# A100 PCIe：准备与取数流水优化\n"
    yield ("本表比较显式开启SMALL_PREPARE与默认关闭，两者源码相同、编译选项不同；默认保留原实现。\n" if label == "enabled" else
           "本表为隔离函数后的中间候选；物理Host较慢格再次出现，未作为默认。\n" if label == "retained" else
           "本表为隔离通用辅助函数前的prepare/scan候选。\n" if label == "accepted" else
           "本表为早期prepare/scan＋32行四缓冲候选；因低延迟回退未设为默认。\n")
    yield ("GPU3 A100-PCIE-40GB，BF16/64头，4096槽/请求，7pass。模拟链路以同卡独立HBM分配为far，"
           "限速器及20us延迟逻辑保持；本组不是物理NVLink实测，也不与先前SXM结果直接作速度比较。"
           "Host组是同卡物理PCIe正常构建。逐步中位数取后半步平均。\n")
    yield ("a为逐步max(b,去重未命中字节/设置带宽)，Host使用实测带宽；b是正常构建同形状纯HBM最好样本，"
           "不计固定延迟入a、不截断比值。精确LRU在同一launch内的等时戳项可任意选择，"
           "因此同时报告未命中量，不能假定每格实际传输字节逐字相同。\n")
    for path in sorted(dest.glob(f"{label}_*.json")):
        case = path.name.removeprefix(label + "_")
        before_path = dest / (("default_" if label == "enabled" else "base_full_") + case)
        if not before_path.exists():
            continue
        new, old = read(path), read(before_path)
        validate(new)
        validate(old)
        cfg = new["configuration"]
        assert cfg | {"output": None} == old["configuration"] | {"output": None}
        if label == "enabled":
            assert "-DFLASHINFER_HOST_BATCH_SMALL_PREPARE" in new["extra_cuda_flags"]
            assert "-DFLASHINFER_HOST_BATCH_SMALL_PREPARE" not in old["extra_cuda_flags"]
        bw = new["emulated_link"]["gbps"] if new["emulated_link"] else max(new["bandwidth"].values())
        yield f"## {case.removesuffix('.json')}\n"
        yield "| 请求 | 行 | 模式 | 旧µs | 新µs | 新÷旧 | 旧miss/步 | 新miss/步 | aµs | bµs | 新÷a | 新÷b |"
        yield "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        for row, before in zip(new["results"], old["results"]):
            assert row["requests"] == before["requests"]
            start = row["steps"] // 2
            b = floor(cfg, row["requests"])
            for mode in ("ours_stage", "ours_inplace"):
                old_us, new_us = steady(before, mode), steady(row, mode)
                old_miss = sum(before["cached_misses"][mode][start:]) / (row["steps"] - start)
                new_miss = sum(row["cached_misses"][mode][start:]) / (row["steps"] - start)
                a = sum(max(b, m * 1152 / bw / 1000) for m in row["cached_misses"][mode][start:]) / (row["steps"] - start)
                assert new_us >= a >= b
                yield (f"| {row['requests']} | {row['query_rows']} | {mode} | {old_us:.2f} | {new_us:.2f} | "
                       f"{new_us/old_us:.3f} | {old_miss:.1f} | {new_miss:.1f} | {a:.2f} | {b:.2f} | "
                       f"{new_us/a:.2f} | {new_us/b:.2f} |")
                summary.append(dict(case=case, requests=row["requests"], query_rows=row["query_rows"], mode=mode,
                                    old_us=old_us, final_us=new_us, final_to_old=new_us/old_us,
                                    old_misses=old_miss, final_misses=new_miss, floor_a_us=a, floor_b_us=b,
                                    ratio_a=new_us/a, ratio_b=new_us/b))
        yield ""
    yield "## 候选关键场景再次复测\n"
    yield "在完整候选→基线矩阵之后重测候选；同一参数下保存所有模式。\n"
    yield "| MTP | 请求 | 模式 | 首测候选µs | 再测候选µs |"
    yield "|---:|---:|---|---:|---:|"
    for mtp in (1, 8):
        first = read(dest / f"{label}_proxy_g100_mtp{mtp}.json")
        repeat = read(dest / f"{label}_repeat_proxy_g100_mtp{mtp}.json")
        validate(repeat)
        for row, before in zip(repeat["results"], first["results"]):
            assert row["requests"] == before["requests"]
            for mode in ("ours_stage", "ours_inplace"):
                yield f"| {mtp} | {row['requests']} | {mode} | {steady(before, mode):.2f} | {steady(row, mode):.2f} |"
    if label in ("retained", "accepted") and (dest / "base_host_recheck_proxy_host_mtp1.json").exists():
        yield "\n## 物理 Host 的反向复测\n"
        yield "完整矩阵中一格原地读较慢，因此再按默认新版→旧版重测，参数及请求列表完全一致。\n"
        yield "| 请求 | 模式 | 复测旧µs | 复测新µs | 新÷旧 |"
        yield "|---:|---|---:|---:|---:|"
        new = read(dest / "accepted_host_recheck_proxy_host_mtp1.json")
        old = read(dest / "base_host_recheck_proxy_host_mtp1.json")
        validate(new)
        validate(old)
        assert new["configuration"] | {"output": None} == old["configuration"] | {"output": None}
        for row, before in zip(new["results"], old["results"]):
            for mode in ("ours_stage", "ours_inplace"):
                a, b = steady(before, mode), steady(row, mode)
                yield f"| {row['requests']} | {mode} | {a:.2f} | {b:.2f} | {b/a:.3f} |"
        if label == "retained":
            isolated = read(dest / "isolated_host_proxy_host_mtp1.json")
            yield "\n隔离原始通用辅助函数后的复测（同原始旧版反向结果对照）：\n"
            yield "| 请求 | 模式 | 旧µs | 隔离版µs | 新÷旧 |"
            yield "|---:|---|---:|---:|---:|"
            for row, before in zip(isolated["results"], old["results"]):
                for mode in ("ours_stage", "ours_inplace"):
                    a, b = steady(before, mode), steady(row, mode)
                    yield f"| {row['requests']} | {mode} | {a:.2f} | {b:.2f} | {b/a:.3f} |"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=("enabled", "retained", "accepted", "final"), default="enabled")
    label = parser.parse_args().label
    prefix = "pipeline_" if label == "final" else "pre_isolation_" if label == "accepted" else "retained_" if label == "retained" else ""
    (dest / (prefix + "results.md")).write_text("\n".join(report()).rstrip() + "\n")
    (dest / (prefix + "summary.json")).write_text(json.dumps(summary, indent=2) + "\n")
