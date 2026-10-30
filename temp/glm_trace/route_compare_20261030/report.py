"""Audit paired full-trace route experiments and generate comparison tables."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[3]
DEST=Path(__file__).resolve().parent
ORIGINAL=ROOT/"temp/glm_trace/kernel_report_20261029"
DOC=ROOT/"docs/mla_host_glm53_route_comparison.md"
load=lambda p:json.loads(p.read_text())
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[];smokes=[];files=[]
for path in sorted((DEST/"raw").glob("*.json")):
    d=load(path)
    if not d.get("complete"):continue
    cfg=d["configuration"];c=d["cases"][0];n=cfg["steps"]
    assert cfg["kind"]=="lru" and cfg["mtp"]==8 and cfg["capacity"]==4096 and cfg["passes"]==3
    assert cfg["route"] in ("staged","inplace")
    assert cfg["ours_dispatch"]=="batched_"+cfg["route"]+"_all_shapes"
    assert c["step_indices"]==list(range(n))
    assert d["resources_start"]["foreign"]==d["resources_end"]["foreign"]==[]
    assert len(c["ours_unique_miss_samples"])==len(c["ours_reference_miss_samples"])==n
    assert all(len(v)==3 for v in c["ours_unique_miss_samples"]+c["ours_reference_miss_samples"])
    assert all(abs(statistics.mean(v)-m)<1e-8 for v,m in zip(c["ours_unique_miss_samples"],c["ours_unique_misses"]))
    assert c["ours_unique_misses"][0]==c["far_unique"][0]
    assert all(x==c["far_refs"][0] for x in c["ours_reference_miss_samples"][0])
    assert all(0<=m<=u for m,u in zip(c["ours_unique_misses"],c["far_unique"]))
    assert all(math.isfinite(error) and error>=0 for error in c["numerical_checks"]["max_abs_error"].values())
    assert all(0<=v<=far for samples,far in zip(c["ours_reference_miss_samples"],c["far_refs"]) for v in samples)
    assert abs(c["ours_reference_cache_hit_rate"]-
        (1-sum(map(sum,c["ours_reference_miss_samples"]))/3/sum(c["far_refs"])))<1e-12
    for mode,t in c["timings"].items():
        assert len(t["per_step_samples_us"])==n and all(len(v)==3 for v in t["per_step_samples_us"])
        assert abs(statistics.mean(v for step in t["per_step_samples_us"] for v in step)-t["mean_us"])<1e-8
        assert all(v>0 and math.isfinite(v) for step in t["per_step_samples_us"] for v in step)
        for pass_index in range(3):
            end=0
            for g in (g for g in c["guards"] if g["mode"]==mode and g["pass_index"]==pass_index):
                assert g["valid"] and g["first"]==end
                if g["timing_queue"]!="cuda_graph":
                    assert g["guard_ms"]>=g["enqueue_ms"]+max(1,g["enqueue_ms"]*.05)
                end=g["last"]
            assert end==n
    for name,sha in d["source_sha256"].items():
        snap=DEST/"source_snapshots"/(sha+Path(name).suffix)
        assert snap.exists() and hashfile(snap)==sha
    if path.name.startswith("smoke_"):
        assert n==2 and c["numerical_checks"]["checked_steps"]==[0,1]
        smokes.append(str(path.relative_to(ROOT)));continue
    assert n==128
    base=load(ORIGINAL/"raw"/f"lru_{cfg['setting']['name']}_mtp8_b{cfg['requests']}.json")
    assert base["bindings"]==d["bindings"]
    assert base["cases"][0]["far_refs"]==c["far_refs"] and base["cases"][0]["far_unique"]==c["far_unique"]
    for name in ("include/flashinfer/attention/mla_host_batch.cuh","include/flashinfer/attention/mla.cuh",
                 "include/flashinfer/attention/far_link_emu.cuh","flashinfer/mla/_core.py","csrc/batch_mla_run.cu"):
        assert base["source_sha256"][name]==d["source_sha256"][name]
    unique_bytes=statistics.mean(c["ours_unique_misses"])*1152
    ref_bytes=statistics.mean(statistics.mean(v) for v in c["ours_reference_miss_samples"])*1152
    # Remap reference counts describe logical reads. This estimate is not a
    # hardware byte counter and does not include protocol/cache-line effects.
    bytes_read=unique_bytes if cfg["route"]=="staged" else ref_bytes
    local=load(ORIGINAL/"raw"/f"floor_floor_mtp8_b{cfg['requests']}.json")["cases"][0]["timings"]["local"]["mean_us"]
    bw=cfg["setting"].get("bandwidth_GBps")
    t=c["timings"]["ours"];copy=c["timings"]["memcpy"]
    row=dict(setting=cfg["setting"]["name"],route=cfg["route"],mtp=8,requests=cfg["requests"],
        effective_kernel_staging_rows=cfg["query_rows"]*2048 if cfg["route"]=="staged" else 0,
        mean_us=t["mean_us"],pass_means_us=t["pass_mean_us"],
        pass_range_over_mean=(max(t["pass_mean_us"])-min(t["pass_mean_us"]))/t["mean_us"],
        memcpy_mean_us=copy["mean_us"],local_mean_us=local,ours_to_local=t["mean_us"]/local,
        reference_cache_hit_rate=c["ours_reference_cache_hit_rate"],
        unique_miss_bytes_per_step=unique_bytes,miss_reference_bytes_per_step=ref_bytes,
        estimated_route_read_bytes_per_step=bytes_read,
        duplicate_reference_factor=ref_bytes/unique_bytes,
        route_transfer_budget_us=bytes_read/(bw*1000) if bw else None,
        raw=str(path.relative_to(ROOT)),max_abs_error=c["numerical_checks"]["max_abs_error"])
    rows.append(row);files.append(dict(path=str(path.relative_to(ROOT)),sha256=hashfile(path)))

pairs=[]
for batch in (16,64,128):
    for setting in ("g100_l0","g100_l20000","g50_l20000","nvlink"):
        values={r["route"]:r for r in rows if r["requests"]==batch and r["setting"]==setting}
        if len(values)<2:continue
        staged,inplace=values["staged"],values["inplace"]
        assert abs(staged["memcpy_mean_us"]/inplace["memcpy_mean_us"]-1)<.10
        pairs.append(dict(setting=setting,requests=batch,staged=staged,inplace=inplace,
            inplace_to_staged=inplace["mean_us"]/staged["mean_us"],
            faster_route=min(values,key=lambda k:values[k]["mean_us"])))
complete=len(rows)==24 and len(pairs)==12 and len(smokes)==2
summary=dict(complete=complete,completed_route_cases=len(rows),expected_route_cases=24,
    pairs=pairs,smoke_files=smokes,files=files)
(DEST/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
if rows:
    with (DEST/"routes.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
labels={"g100_l0":"100 GB/s＋0 µs","g100_l20000":"100 GB/s＋20 µs",
        "g50_l20000":"50 GB/s＋20 µs","nvlink":"物理NVLink"}
lines=["# GLM-5.3 trace：staged 与原地读路径对照","",
    ("状态：全部24个路径测试格完成并通过审计。" if complete else f"状态：已完成{len(rows)}/24个路径测试格。"),"",
    "同一对A100-SXM4-40GB，GPU0计算、GPU1远端；MTP8，16/64/128请求；每请求4096槽LRU，完整1024位置、3次冷启动pass、包括首步的算术平均。相同seed、query/KV、trace绑定及生产kernel SHA；只切换现有host_stage选项，fetch参数固定16，item64。", "",
    "计时延续原矩阵：16步CUDA Graph、逐步CUDA event、事件外miss诊断。两模式均校验数学参考、LRU状态和冷启动。远端未命中引用的诊断对原地读改为统计正远端页号，并去掉owner标记位。", "",
    "两模式都保留相同的外部VMM/基线缓冲配额。原地读host_stage=False时，wrapper内部staging容量为0，attention直接读取未命中远端页并在读取后写缓存；raw配置的staging_rows沿用公共缓冲计划，不代表原地读启用了staging。派生routes.csv明确记录effective_kernel_staging_rows。", "",
    "| 链路 | 请求 | staged µs | 原地读 µs | 原地读/staged | Memcpy µs（同轮staged对照） | 纯device µs | 较快路径 |",
    "|---|---:|---:|---:|---:|---:|---:|---|"]
for p in pairs:
    s,i=p["staged"],p["inplace"]
    lines.append(f"| {labels[p['setting']]} | {p['requests']} | {s['mean_us']:.2f} | {i['mean_us']:.2f} | {p['inplace_to_staged']:.3f} | {s['memcpy_mean_us']:.2f} | {s['local_mean_us']:.2f} | {p['faster_route']} |")
lines+=["","传输预算按逻辑读引用估算，不是硬件计数器：staged按unique未命中页，原地读按未命中引用。两者均有LRU，重复只发生在同一次launch未命中引用之间。","",
    "| 链路 | 请求 | 路径 | 平均未命中unique MB/步 | 平均未命中引用 MB/步 | 预计路径读取 MB/步 | 仅传输预算 µs | LRU引用命中率 |",
    "|---|---:|---|---:|---:|---:|---:|---:|"]
for p in pairs:
    for route in ("staged","inplace"):
        r=p[route];budget=r["route_transfer_budget_us"]
        lines.append(f"| {labels[p['setting']]} | {p['requests']} | {route} | {r['unique_miss_bytes_per_step']/1e6:.3f} | {r['miss_reference_bytes_per_step']/1e6:.3f} | {r['estimated_route_read_bytes_per_step']/1e6:.3f} | {budget:.2f} | {r['reference_cache_hit_rate']*100:.2f}% |" if budget is not None else
            f"| {labels[p['setting']]} | {p['requests']} | {route} | {r['unique_miss_bytes_per_step']/1e6:.3f} | {r['miss_reference_bytes_per_step']/1e6:.3f} | {r['estimated_route_read_bytes_per_step']/1e6:.3f} | — | {r['reference_cache_hit_rate']*100:.2f}% |")
lines+=["","原地读的GPU访问仍经过同一带宽/延迟模拟钩子。不同路由的LRU淘汰同stamp并列顺序可不同，因此分别记录实际命中与流量。完整输出、每步样本、guard和源码快照在temp/glm_trace/route_compare_20261030/。原5304格矩阵保持原样，本对照单独呈现，不把切换路径的收益当作固定staged配置的收益。"]
if complete:
    assert all(p["faster_route"]==("inplace" if p["setting"]=="nvlink" else "staged") for p in pairs)
    lines+=["","## 结论","",
        "在本轮所测16/64/128请求、MTP8和4096槽LRU条件下：物理NVLink三个形状均是原地读更快；100 GB/s的0/20 µs及50 GB/s＋20 µs九个对照均是staged更快。此结果支持按链路条件选择路径，不支持将NVLink的原地读优势直接外推到100 GB/s。",
        "",
        "128请求时，原地读每步约509 MB未命中引用，staged实际去重后约260 MB，重复因子约1.96。100 GB/s时，原地读的逻辑读取字节预算本身约5.09 ms，已高于staged的总耗时。物理NVLink带宽更高，这笔重复读取代价更容易被计算隐藏，省去staging中转可带来收益。",
        "",
        "100 GB/s＋20 µs下原地读比零附加延迟慢得更多，符合直接attention tile需要等待多轮远端块的路径差异；本轮未独立消融每一项成本，不能精确分摊额外开销。两种模式共用同一链路模拟钩子，输出数学参考和计时队列审计均通过。",
        "",
        "物理NVLink的16请求原地读约为纯device的1.11倍；128请求仍约1.33倍。既要按实际链路选路径，也应按batch评估距本地参考的差距，不把单个接近下限的例子概括为所有形状。"]
DOC.write_text("\n".join(lines)+"\n")
(DEST/"report_manifest.json").write_text(json.dumps(dict(complete=complete,
    document=str(DOC.relative_to(ROOT)),document_sha256=hashfile(DOC),
    summary_sha256=hashfile(DEST/"summary.json"),csv_sha256=hashfile(DEST/"routes.csv"),
    original_matrix_sha256=hashfile(ORIGINAL/"matrix.json"),
    sampled_max_abs_error=max((v for r in rows for v in r["max_abs_error"].values()),default=0.),
    max_ours_pass_range_over_mean=max((r["pass_range_over_mean"] for r in rows),default=0.),
    raw_files=len(files),source_sha256={name:hashfile(DEST/name) for name in
        ("bench_routes.py","report.py","run.py","resources.py","plans.cpp","experiment.json")}),indent=2)+"\n")
print(json.dumps(dict(complete=complete,cases=len(rows),pairs=len(pairs),report=str(DOC))))
