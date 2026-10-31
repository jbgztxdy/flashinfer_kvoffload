"""Audit the fixed-trace scheduling sweep; keep the original matrix unchanged."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[2]
ORIGINAL = ROOT / "temp/glm_trace/kernel_report_20261029"
load = lambda p: json.loads(p.read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
local_case = load(ORIGINAL / "raw/floor_floor_mtp8_b128.json")["cases"][0]
compute = [statistics.mean(v) for v in local_case["timings"]["local"]["per_step_samples_us"]]
rows = []
files = []
for path in sorted((DEST / "raw").glob("*.json")):
    d = load(path)
    assert d["complete"]
    cfg = d["configuration"]
    c = d["cases"][0]
    n, passes = cfg["steps"], cfg["passes"]
    confirmation = path.name.startswith("confirm_")
    assert n == 128 and passes == (5 if confirmation else 3)
    assert cfg["kind"] == "lru" and cfg["mtp"] == 8 and cfg["requests"] == 128
    assert cfg["capacity"] == 4096 and cfg["ours_dispatch"] == "batched_staged_all_shapes"
    assert cfg["gpu_only"] and not cfg["baseline_cpu_lru_planning_timed"]
    assert cfg["setting"]["name"] in ("g100_l0", "g100_l20000")
    assert cfg["setting"]["bandwidth_GBps"] == 100
    # End snapshots include this benchmark's own allocated memory/context.
    assert d["resources_start"]["available"]
    assert d["resources_start"]["foreign"] == d["resources_end"]["foreign"] == []
    assert c["step_indices"] == list(range(n))
    assert c["numerical_checks"]["checked_steps"] == [0, 1, 64, 127]
    assert max(c["numerical_checks"]["max_abs_error"].values()) < .01
    base = load(ORIGINAL / "raw" / f"lru_{cfg['setting']['name']}_mtp8_b128.json")
    assert base["bindings"] == d["bindings"]
    for field in ("far_refs", "far_unique"):
        assert base["cases"][0][field] == c[field]
    for name in ("include/flashinfer/attention/mla_host_batch.cuh", "include/flashinfer/attention/mla.cuh",
                 "include/flashinfer/attention/far_link_emu.cuh", "flashinfer/mla/_core.py", "csrc/batch_mla_run.cu"):
        # The executed files are checked against retained snapshots below; historical
        # production headers may change in later tasks.
        assert name in base["source_sha256"] and name in d["source_sha256"]
    for name, sha in d["source_sha256"].items():
        snap = DEST / "source_snapshots" / (sha + Path(name).suffix)
        assert snap.exists() and digest(snap) == sha
    for field in ("ours_unique_miss_samples", "ours_reference_miss_samples"):
        assert len(c[field]) == n and all(len(v) == passes for v in c[field])
    assert all(abs(statistics.mean(v) - m) < 1e-8
               for v, m in zip(c["ours_unique_miss_samples"], c["ours_unique_misses"]))
    assert all(v == c["far_unique"][0] for v in c["ours_unique_miss_samples"][0])
    assert all(v == c["far_refs"][0] for v in c["ours_reference_miss_samples"][0])
    assert all(0 <= v <= far for samples, far in zip(c["ours_unique_miss_samples"], c["far_unique"]) for v in samples)
    assert all(0 <= v <= far for samples, far in zip(c["ours_reference_miss_samples"], c["far_refs"]) for v in samples)
    hit = 1 - sum(map(sum, c["ours_reference_miss_samples"])) / passes / sum(c["far_refs"])
    assert abs(hit - c["ours_reference_cache_hit_rate"]) < 1e-12
    for mode, t in c["timings"].items():
        samples = t["per_step_samples_us"]
        assert len(samples) == n and all(len(v) == passes for v in samples)
        assert all(v > 0 and math.isfinite(v) for step in samples for v in step)
        assert abs(statistics.mean(v for step in samples for v in step) - t["mean_us"]) < 1e-8
        for p in range(passes):
            assert abs(statistics.mean(step[p] for step in samples) - t["pass_mean_us"][p]) < 1e-8
            end = 0
            for g in (g for g in c["guards"] if g["mode"] == mode and g["pass_index"] == p):
                assert g["valid"] and g["first"] == end
                if g["timing_queue"] != "cuda_graph":
                    assert g["guard_ms"] >= g["enqueue_ms"] + max(1, g["enqueue_ms"] * .05)
                end = g["last"]
            assert end == n
    transfer = [m * 1152 / 100000 for m in c["ours_unique_misses"]]
    reference = statistics.mean(max(a, b) for a, b in zip(compute, transfer))
    ours = c["timings"]["ours"]
    copy = c["timings"]["memcpy"]
    rows.append(dict(confirmation=confirmation, setting=cfg["setting"]["name"],
        fetch_ctas=cfg["fetch_ctas"], host_item_entries=cfg["host_item_entries"], host_split=cfg["host_split"],
        passes=passes, mean_us=ours["mean_us"], pass_means_us=ours["pass_mean_us"],
        pass_range_over_mean=(max(ours["pass_mean_us"]) - min(ours["pass_mean_us"])) / ours["mean_us"],
        memcpy_mean_us=copy["mean_us"], local_mean_us=statistics.mean(compute),
        reference_mean_stepwise_max_us=reference, ours_to_reference=ours["mean_us"] / reference,
        reference_cache_hit_rate=hit, unique_miss_MB=statistics.mean(c["ours_unique_misses"]) * 1152 / 1e6,
        max_abs_error=c["numerical_checks"]["max_abs_error"], raw=str(path.relative_to(ROOT))))
    files.append(dict(path=str(path.relative_to(ROOT)), sha256=digest(path)))

expected = {(f, 64, s, setting) for f in (8, 16, 32, 64) for s in (True, False)
            for setting in ("g100_l0", "g100_l20000")}
expected |= {(f, 256, True, setting) for f in (16, 32) for setting in ("g100_l0", "g100_l20000")}
sweep = [r for r in rows if not r["confirmation"]]
assert {(r["fetch_ctas"], r["host_item_entries"], r["host_split"], r["setting"]) for r in sweep} == expected
assert len(sweep) == 20
confirmation = [r for r in rows if r["confirmation"]]
assert len(confirmation) in (0, 4)
if confirmation:
    assert {(r["fetch_ctas"], r["host_item_entries"], r["host_split"], r["setting"]) for r in confirmation} == {
        (f, 64, split, setting) for f, split in ((32, False), (16, True))
        for setting in ("g100_l0", "g100_l20000")}
comparisons = []
for confirmed in (False, True):
    for setting in ("g100_l0", "g100_l20000"):
        selected = [r for r in rows if r["confirmation"] == confirmed and r["setting"] == setting]
        if not selected:
            continue
        default = next(r for r in selected if r["fetch_ctas"] == 16 and r["host_item_entries"] == 64 and r["host_split"])
        best = min(selected, key=lambda r: r["mean_us"])
        comparisons.append(dict(confirmation=confirmed, setting=setting, default=default, best=best,
            reduction_vs_same_round_default=1 - best["mean_us"] / default["mean_us"]))
summary = dict(complete=len(rows) == 24, sweep_complete=True, sweep_cells=20,
    confirmation_cells=len(confirmation), conditions="A100 SXM4, complete GLM53 trace 1024 positions, MTP8, 128 requests, staged, LRU4096/request",
    original_matrix_sha256=digest(ORIGINAL / "matrix.json"), rows=rows, comparisons=comparisons, files=files)
(DEST / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
with (DEST / "parameters.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

lines = ["# 100 GB/s报告核查：现有调度参数扫描", "",
    "所有数据均为A100-SXM4 GPU0/1、相同GLM-5.3完整1024位置访问trace、128请求×MTP8、staged、每请求4096槽LRU。只调整现有fetch CTA数、Host item大小及host_split；未改生产kernel、模拟钩子和原5304格矩阵。每pass清空缓存，含冷启动首步，128次launch取算术平均。", "",
    "主扫描10套参数×100 GB/s的0/20 µs两链路，共20格、每格3pass。随后将20 µs候选最优配置和默认配置倒序复测，每格5pass，共4格。数学参考采样、完整步骤/均值、guard、冷启动miss及相同trace/源码SHA审计均通过。CPU LRU规划与Host整理仍在计时外，Memcpy保留真实GPU搬运、必要缓存scatter及attention。", "",
    "Host split控制尾部attention工作项是否按远端/本地分开，不是staged开关；off仍保留staged去重取数和LRU。更多fetch CTA可能减少attention计算资源，参数收益不保证单调。下表全部参数保留。", "",
    "| 阶段 | 附加延迟 µs | fetch CTA | item | host split | 本文 µs | Memcpy µs | 纯device µs | 重叠参考 µs | 高于参考 | LRU引用命中率 | unique miss MB/步 | pass均值 µs |",
    "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"]
for r in sorted(rows, key=lambda r: (r["confirmation"], r["fetch_ctas"], r["host_item_entries"], not r["host_split"], r["setting"])):
    lines.append(f"| {'复测5pass' if r['confirmation'] else '扫描3pass'} | {0 if r['setting']=='g100_l0' else 20} | {r['fetch_ctas']} | {r['host_item_entries']} | {'on' if r['host_split'] else 'off'} | {r['mean_us']:.2f} | {r['memcpy_mean_us']:.2f} | {r['local_mean_us']:.2f} | {r['reference_mean_stepwise_max_us']:.2f} | {(r['ours_to_reference']-1)*100:.1f}% | {r['reference_cache_hit_rate']*100:.3f}% | {r['unique_miss_MB']:.3f} | {', '.join(f'{v:.2f}' for v in r['pass_means_us'])} |")
lines += ["", "参考使用同一步全本地attention时间和本次实际unique miss字节：mean_t(max(C_t, bytes_t/100 GB/s))，不含附加延迟，是乐观重叠参考。纯device来自原矩阵完整窗口同输入参考，本轮未重测全部本地格。命中率因同stamp淘汰并列顺序有微小变化，逐配置分别记录。", "",
    "## 结果解释", ""]
for p in comparisons:
    d, b = p["default"], p["best"]
    lines.append(f"- {'倒序确认' if p['confirmation'] else '主扫描'}，100 GB/s＋{0 if p['setting']=='g100_l0' else 20} µs：默认{d['mean_us']:.2f} µs；本轮最低{b['mean_us']:.2f} µs（fetch{b['fetch_ctas']}/item{b['host_item_entries']}/split{'on' if b['host_split'] else 'off'}），降低{p['reduction_vs_same_round_default']*100:.2f}%，仍高于其重叠参考{(b['ours_to_reference']-1)*100:.1f}%。")
lines += ["", "100 GB/s＋20 µs原主矩阵4450.15 µs没有被参数搜索结果替换。主扫描同轮默认4400.24 µs，候选4230.80 µs，说明现有参数仍有改善空间，但只消除小部分差距。本轮没有做kernel内部阶段计时，不能将剩余差距精确归因为LRU、staging、同步或模拟器中的某一项，也不能宣称搜索已穷尽。", "",
    "100 GB/s相比50 GB/s的绝对耗时仍更低；较差的是相对乐观参考的效率。max模型假设搬运和attention同时各自达到独占资源时的速度，忽略共享SM/HBM、工作项粒度及数据依赖。50 GB/s的较长传输可掩盖一些成本，100 GB/s下计算/取数竞争及调度尾部更明显；这是结合代码和结果的解释，尚非逐项消融证明。", "",
    "后续应先测prepare/resolve、fetch完成及attention尾部的时间，再针对取数资源分配、先到数据的消费顺序和staging写回优化。扫描表显示fetch8明显不足、fetch64也回退，继续单纯增加取数CTA不足以解决问题。", "",
    "复现：使用run.py执行20格扫描，confirm.py执行倒序5pass确认，report.py审计并生成本文件及CSV。原始样本在raw/；源码快照在source_snapshots/；运行退出与信号分别见signal.json、confirmation_signal.json。"]
(DEST / "README.md").write_text("\n".join(lines) + "\n")
manifest = dict(complete=summary["complete"], summary_sha256=digest(DEST / "summary.json"),
    csv_sha256=digest(DEST / "parameters.csv"), readme_sha256=digest(DEST / "README.md"),
    source_sha256={p.name: digest(p) for p in DEST.glob("*.py")},
    source_matrix_sha256=summary["original_matrix_sha256"], files=files,
    sampled_max_abs_error=max(v for r in rows for v in r["max_abs_error"].values()),
    max_pass_range_over_mean=max(r["pass_range_over_mean"] for r in rows))
(DEST / "audit.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(dict(complete=summary["complete"], sweep_cells=20, confirmation_cells=len(confirmation)), ensure_ascii=False))
