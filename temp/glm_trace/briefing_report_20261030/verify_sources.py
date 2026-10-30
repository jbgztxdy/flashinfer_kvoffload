"""Retain independent repeats and historical reference definitions without altering matrix."""
import hashlib
import json
from pathlib import Path
import statistics
import time
from references import for_row, ROOT, SOURCE, DEST

load=lambda p:json.loads(p.read_text())
rows=load(SOURCE/"matrix.json")
lookup={(r["kind"],r["setting"],r["mtp"],r["requests"],r["host_fraction_target"]):r for r in rows}
verification=dict(timestamp=time.strftime("%FT%T%z"),
    matrix_unchanged_sha256=hashlib.sha256((SOURCE/"matrix.json").read_bytes()).hexdigest(),
    original_near_fraction=[],key_repeats=[],recomputed_references=[],historical_comparisons=[])
for batch in (1,4,16,128):
    r=lookup["lru","g100_l20000",8,batch,None]
    near=1-r["host_fraction_actual"];cache=r["host_fraction_actual"]-r["ours_remote_reference_fraction"]
    verification["original_near_fraction"].append(dict(requests=batch,original_local_fraction=near,
        lru_local_fraction_of_all_refs=cache,total_local_fraction=near+cache,raw=r["raw"]))
for name in ("lru_g100_l20000_mtp8_b128.json","lru_g50_l20000_mtp8_b128.json",
             "floor_floor_mtp8_b128.json"):
    old=load(SOURCE/"raw"/name);new=load(DEST/"recheck/raw"/name)
    assert new["complete"] and old["bindings"]==new["bindings"]
    source_names=("include/flashinfer/attention/mla_host_batch.cuh","include/flashinfer/attention/mla.cuh",
        "include/flashinfer/attention/far_link_emu.cuh","flashinfer/mla/_core.py","csrc/batch_mla_run.cu")
    assert all(old["source_sha256"][n]==new["source_sha256"][n] for n in source_names)
    a,b=old["cases"][0],new["cases"][0]
    assert a["far_refs"]==b["far_refs"] and a["far_unique"]==b["far_unique"]
    record=dict(file=name,mode_results={},max_abs_error=b["numerical_checks"]["max_abs_error"])
    for mode,t in b["timings"].items():
        assert len(t["per_step_samples_us"])==128
        assert all(len(step)==3 for step in t["per_step_samples_us"])
        average=statistics.mean(v for step in t["per_step_samples_us"] for v in step)
        assert abs(average-t["mean_us"])<1e-8
        previous=a["timings"][mode]["mean_us"]
        record["mode_results"][mode]=dict(original_us=previous,repeated_us=average,change_fraction=average/previous-1)
        assert abs(average/previous-1)<.03
        for pass_index in range(3):
            end=0
            for g in (g for g in b["guards"] if g["mode"]==mode and g["pass_index"]==pass_index):
                assert g["valid"] and g["first"]==end
                if g.get("timing_queue")!="cuda_graph":
                    assert g["guard_ms"]>=g["enqueue_ms"]+max(1,g["enqueue_ms"]*.05)
                end=g["last"]
            assert end==128
    verification["key_repeats"].append(record)
for path in (DEST/"recheck/source_snapshots").iterdir():
    assert hashlib.sha256(path.read_bytes()).hexdigest()==path.stem
for kind in ("nocache","lru"):
    for setting in ("pcie","g50_l20000","g100_l20000"):
        verification["recomputed_references"].append(for_row(lookup[kind,setting,8,128,.3 if kind=="nocache" else None]))
old=load(ROOT/"temp/a100_full/sweep_host_mtp8.json")
r=next(r for r in old["results"] if r["requests"]==112)
c=next(c for c in r["cases"] if c["far_fraction"]==.25)
bw=max(old["bandwidth"].values());upper=max(c["far_unique_bytes"])/(bw*1000)
average=statistics.mean(c["far_unique_bytes"])/(bw*1000);ours=c["batch"]["median_us"]
verification["historical_comparisons"].append(dict(source="temp/a100_full/sweep_host_mtp8.json",
    requests=112,mtp=8,target_far_fraction=.25,ours_us=ours,upper_middle_transfer_us=upper,published_ratio=ours/upper,
    mean_byte_transfer_us=average,ratio_using_mean_byte_budget=ours/average,
    note="Old tables.py takes the upper of two index-set budgets; new report uses all-step means."))
for bw in (50,100):
    path=ROOT/f"temp/sim_link/lat/g{bw}_l20000_mtp8_f16.json";d=load(path)
    r=next(r for r in d["results"] if r["requests"]==128)
    ticks=list(range(r["steps"]//2,r["steps"]))
    miss=statistics.mean(r["cached_misses"]["ours_stage"][t] for t in ticks)
    ours=statistics.mean(r["per_step"]["ours_stage"][t]["median_us"] for t in ticks)
    transfer=miss*1152/(bw*1000)
    verification["historical_comparisons"].append(dict(source=str(path.relative_to(ROOT)),requests=128,mtp=8,
        steps=r["steps"],averaged_steps=ticks,bandwidth_GBps=bw,ours_us=ours,transfer_bound_us=transfer,
        ours_to_transfer=ours/transfer,mean_unique_misses=miss))
(DEST/"verification.json").write_text(json.dumps(verification,ensure_ascii=False,indent=2)+"\n")
lines=["# 汇报图表与下限口径复核（2026-10-30）","","## 1. 本次修改","",
    "- 所有GPU耗时仍来自原5304格矩阵；不替换原始样本。",
    "- 三组性能柱状图均增加逐步 max(计算, 搬运) 的平均参考线。",
    "- 参考线不含附加访问延迟、LRU及同步成本，是乐观重叠参考。",
    "- 物理带宽补测：PCIe 27.039965 GB/s、NVLink 281.123759 GB/s，取9次测量中位数的max(DMA,gather)。",
    "","## 2. 图3原始本地并非零","",
    "| MTP8请求 | 原始本地占全部引用 | LRU命中本地占全部引用 | 两者合计 |","|---:|---:|---:|---:|"]
for r in verification["original_near_fraction"]:
    lines.append(f"| {r['requests']} | {100*r['original_local_fraction']:.5f}% | {100*r['lru_local_fraction_of_all_refs']:.3f}% | {100*r['total_local_fraction']:.3f}% |")
lines.extend(["",
    "本轮设置为历史KV远端、最新token本地。原始本地约1/2048，4096指LRU槽，不是原始本地窗口。",
    "","## 3. max与平均的顺序","",
    "旧衍生矩阵为max(mean(C),mean(T))；本版为mean(max(C_t,T_t))。原矩阵不改写。",
    "100GB/s20us、MTP8/128请求：2595.95655→2691.44634 µs，kernel4450.15466 µs，高于参考71.426%→65.344%。",
    "50GB/s对应5191.782 µs不变，高9.389%。","","## 4. 独立完整窗口复测","",
    "冻结bench.py仅改ROOT层级，输出独立目录；同GPU、trace绑定、kernel SHA、3pass、128步、冷启动。",
    "逐步均值重算、Q/KV数学校验、绑定/远端引用与unique计数一致性均通过。","",
    "| 原始文件 | 方法 | 原始µs | 复测µs | 变化 |","|---|---|---:|---:|---:|"])
for r in verification["key_repeats"]:
    for mode,t in r["mode_results"].items():
        lines.append(f"| {r['file']} | {mode} | {t['original_us']:.3f} | {t['repeated_us']:.3f} | {t['change_fraction']*100:+.3f}% |")
lines.extend(["",
    "复测支持关键计时可复现；未作归因消融，不能精确分摊LRU、staging、同步和模拟钩子成本。",
    "","## 5. 旧1%与10%来自哪些参数","",
    "- 旧PCIe代理trace单步112请求/远端25%：8468.992 µs，旧upper-middle预算8390.387 µs，比值1.00937。",
    "- 旧表在两组bytes中选较大者；按两组均值8171.432 µs重算，比值1.03641。两者均接近传输参考，但不可直接套入本轮均值。",
    "- 旧PCIe代理trace＋LRU、MTP8/128请求约1.034倍（4步取后2步）。",
    "- 旧50GB/s20us、MTP8/128请求：7634.432/7165.117 µs＝1.0655。",
    "- 旧100GB/s20us同形状：5585.408/3584.339 µs＝1.5583，同样不是10%以内。",
    "- 本轮GLM53 PCIe＋LRU、MTP8 32/64/128请求高8.756%/8.393%/4.778%。",
    "- 本轮50GB/s20us、128请求高9.389%；100GB/s同形状高65.344%，各项固定开销贡献仍待消融。",
    "","## 6. 汇报结论","",
    "中大batch优于Memcpy的证据保持。接近重叠下限应重点展示本次真实trace的PCIe和50GB/s大batch；接近纯device引用低远端比例/快速链路的具体参数。",
    "旧代理数据只用于解释新旧口径，未混入本轮图表。所有核对数字见verification.json。","",
    "原矩阵SHA256："+verification["matrix_unchanged_sha256"]+"。"])
(DEST/"verification_notes.md").write_text("\n".join(lines)+"\n")
print("Verified source bindings, means and source hashes; saved repeats and historical evidence.")
