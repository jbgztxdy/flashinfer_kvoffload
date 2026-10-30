"""Check briefing tables, figure source cells and headline claims against audited data."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from references import for_row

DEST=Path(__file__).resolve().parent
ROOT=DEST.parents[2]
SOURCE=ROOT/"temp/glm_trace/kernel_report_20261029"
DOC=ROOT/"docs/mla_host_glm53_report.md"
load=lambda p:json.loads(p.read_text())
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=load(SOURCE/"matrix.json")
key=lambda r:(r["kind"],r["setting"],r["mtp"],r["requests"],r["host_fraction_target"])
lookup={key(r):r for r in rows}
manifest=load(DEST/"manifest.json")
assert load(SOURCE/"audit.json")["complete"]
assert manifest["source_matrix_sha256"]==digest(SOURCE/"matrix.json")
for name,sha in manifest["artifact_sha256"].items():assert digest(ROOT/name)==sha
meta=load(SOURCE/"experiment.json")


def label(s):
    if s["name"]=="nvlink":return "物理 NVLink"
    if s["name"]=="pcie":return "物理 Host PCIe"
    return (f"{s['bandwidth_GBps']} GB/s" if s["bandwidth_GBps"] else "不限速")+f"＋{s['latency_ns']/1000:g} µs"


settings={label(s):s["name"] for s in meta["settings"]}
text=DOC.read_text();lines=text.splitlines()
mode=None;checked=0;summaries=0;bounds=0;near_rows=0;repeat_rows=0;physical_rows=0;route_rows=0
link_diagnosis_rows=0;parameter_rows=0
routes=load(ROOT/"temp/glm_trace/route_compare_20261030/summary.json")
assert routes["complete"] and len(routes["pairs"])==12
review=load(ROOT/"temp/glm_trace/100gb_review_20261030/summary.json")
assert review["complete"] and review["sweep_cells"]==20 and review["confirmation_cells"]==4
for line in lines:
    if not line.startswith("|"):
        mode=None;continue
    fields=[x.strip() for x in line.strip("|").split("|")]
    if fields[:3]==["链路","MTP","请求"]:
        host="目标远端比例" in fields
        cache="LRU引用命中率" in fields
        mode=("samples",host,cache,len(fields));continue
    if fields[0]=="链路" and fields[1].startswith("无LRU领先"):
        mode=("summary",False,False,len(fields));continue
    if fields[0].startswith("场景：MTP8"):
        mode=("bounds",False,False,len(fields));continue
    if fields[0]=="MTP8请求数":
        mode=("near",False,False,len(fields));continue
    if fields[0]=="MTP8 / 128请求":
        mode=("repeat",False,False,len(fields));continue
    if fields[0]=="物理链路补测":
        mode=("physical",False,False,len(fields));continue
    if fields[:4]==["链路","请求","staged µs","原地读 µs"]:
        mode=("route",False,False,len(fields));continue
    if fields[0]=="主矩阵链路对照":
        mode=("link_diagnosis",False,False,len(fields));continue
    if fields[0]=="参数核查阶段":
        mode=("parameters",False,False,len(fields));continue
    if not mode or all(re.fullmatch(r"[:\-]+",f) for f in fields):continue
    assert len(fields)==mode[3],fields
    if mode[0]=="samples":
        _,host,cache,_=mode
        kind="nocache" if host else "lru"
        s,m,b=settings[fields[0]],int(fields[1]),int(fields[2])
        fraction=int(fields[3].strip('%'))/100 if host else None
        r=lookup[kind,s,m,b,fraction]
        result=[fields[0],str(m),str(b)]+([f"{fraction:.0%}"] if host else [])
        result.extend(f"{r[k]:.2f}" for k in ("ours_mean_us","memcpy_mean_us","local_mean_us"))
        reference=for_row(r)
        result.extend([f"{reference['mean_stepwise_max_us']:.2f}",
                       f"{(1-r['ours_mean_us']/r['memcpy_mean_us'])*100:.1f}%",
                       f"{r['ours_mean_us']/r['local_mean_us']:.3f}×",
                       f"{r['ours_mean_us']/reference['mean_stepwise_max_us']:.3f}×"])
        if cache:result.append(f"{r['ours_reference_cache_hit_rate']*100:.2f}%")
        assert result==fields,(result,fields)
        checked+=1
    elif mode[0]=="summary":
        result=[fields[0]]
        for kind in ("nocache","lru"):
            sub=[r for r in rows if r["kind"]==kind and r["setting"]==settings[fields[0]]]
            win=sum(r["ours_mean_us"]<r["memcpy_mean_us"] for r in sub)
            speed=math.exp(sum(math.log(r['memcpy_mean_us']/r['ours_mean_us']) for r in sub)/len(sub))
            result.extend([str(win),f"{speed:.3f}×"])
        assert result==fields,(result,fields)
        summaries+=1
    elif mode[0]=="bounds":
        kind,link=fields[0].rsplit("，",1)
        kind="nocache" if "无LRU" in kind else "lru"
        r=lookup[kind,settings[link],8,128,.3 if kind=="nocache" else None]
        bound=for_row(r)["mean_stepwise_max_us"]
        result=[fields[0],*[f"{r[k]:.2f}" for k in ("ours_mean_us","memcpy_mean_us","local_mean_us")],
                f"{bound:.2f}",f"{(r['ours_mean_us']/bound-1)*100:.1f}%"]
        assert result==fields,(result,fields)
        bounds+=1
    elif mode[0]=="near":
        r=lookup["lru","g100_l20000",8,int(fields[0]),None]
        near=1-r["host_fraction_actual"]
        cached=r["host_fraction_actual"]-r["ours_remote_reference_fraction"]
        expected=[fields[0],f"{near*100:.4f}%",f"{cached*100:.2f}%",f"{(near+cached)*100:.2f}%"]
        assert fields==expected,(fields,expected)
        near_rows+=1
    elif mode[0]=="repeat":
        setting=settings[fields[0]]
        original=lookup["lru",setting,8,128,None]
        repeated=load(DEST/"recheck/raw"/f"lru_{setting}_mtp8_b128.json")
        assert repeated["complete"]
        ours=repeated["cases"][0]["timings"]["ours"]["mean_us"]
        memcpy=repeated["cases"][0]["timings"]["memcpy"]["mean_us"]
        expected=[fields[0],f"{original['ours_mean_us']:.2f}",f"{ours:.2f}",
            f"{(ours/original['ours_mean_us']-1)*100:+.2f}%",f"{original['memcpy_mean_us']:.2f}",f"{memcpy:.2f}"]
        assert fields==expected,(fields,expected)
        repeat_rows+=1
    elif mode[0]=="physical":
        probe=load(DEST/"physical_bandwidth.json")
        r=next(s for s in probe["settings"] if s["name"]==settings[fields[0]])
        expected=[fields[0],f"{r['measurements']['dma_contiguous']['median_GBps']:.2f}",
                  f"{r['measurements']['sm_gather_random']['median_GBps']:.2f}",f"{r['reference_GBps']:.2f}"]
        assert fields==expected
        physical_rows+=1
    elif mode[0]=="route":
        p=next(p for p in routes["pairs"] if p["setting"]==settings[fields[0]] and p["requests"]==int(fields[1]))
        expected=[fields[0],fields[1],f"{p['staged']['mean_us']:.2f}",f"{p['inplace']['mean_us']:.2f}",
                  f"{p['inplace_to_staged']:.3f}","原地读" if p["faster_route"]=="inplace" else "staged"]
        assert fields==expected,(fields,expected)
        route_rows+=1
    elif mode[0]=="link_diagnosis":
        r=lookup["lru",settings[fields[0]],8,128,None]
        expected=[fields[0],*[f"{r[k]:.2f}" for k in ("ours_mean_us","memcpy_mean_us","local_mean_us")],
                  f"{for_row(r)['mean_stepwise_max_us']:.2f}"]
        assert fields==expected,(fields,expected)
        link_diagnosis_rows+=1
    else:
        confirmed=fields[0].startswith("倒序确认")
        name="default" if fields[0].endswith("默认") else "best"
        pair=next(p for p in review["comparisons"] if p["confirmation"]==confirmed and p["setting"]=="g100_l20000")
        r=pair[name]
        expected=[fields[0],str(r["fetch_ctas"]),str(r["host_item_entries"]),"on" if r["host_split"] else "off",
                  *[f"{r[k]:.2f}" for k in ("mean_us","memcpy_mean_us","local_mean_us","reference_mean_stepwise_max_us")],
                  f"{(r['ours_to_reference']-1)*100:.1f}%",f"{r['reference_cache_hit_rate']*100:.3f}%"]
        assert fields==expected,(fields,expected)
        parameter_rows+=1
assert (checked,summaries,bounds,near_rows,repeat_rows,physical_rows,route_rows)==(26,17,6,4,2,2,12)
assert link_diagnosis_rows==parameter_rows==4

figures=load(DEST/"figure_data.json")
assert len(figures)==6
fig_count=0
for name,figure in figures.items():
    assert (ROOT/figure["file"]).exists()
    assert len(figure["rows"])==manifest["figure_cells"][name]
    assert all(s in figure["caption"] for s in ("trace","staged","LRU"))
    for r in figure["rows"]:
        assert r==lookup[key(r)]
        fig_count+=1
    assert len(figure["overlap_references"])==len(figure["rows"])
    for r,reference in zip(figure["rows"],figure["overlap_references"]):
        assert reference==for_row(r)
        raw=load(ROOT/r["raw"])
        c=next(c for c in raw["cases"] if c["host_fraction_requested"]==r["host_fraction_target"])
        mode="floor" if r["kind"]=="lru" else "snapshots"
        local_file=SOURCE/"raw"/f"{mode}_floor_mtp{r['mtp']}_b{r['requests']}.json"
        local=next(c for c in load(local_file)["cases"] if c["host_fraction_requested"]==r["host_fraction_target"])
        compute=[sum(v)/len(v) for v in local["timings"]["local"]["per_step_samples_us"]]
        transfer=[m*1152/(1000*reference["bandwidth_GBps"]) for m in c["ours_unique_misses"]]
        expected=sum(max(a,b) for a,b in zip(compute,transfer))/len(compute)
        assert abs(expected-reference["mean_stepwise_max_us"])<1e-7
        assert expected>=max(sum(compute)/len(compute),sum(transfer)/len(transfer))-1e-7
assert len(re.findall(r"!\[图",text))==6
for link in re.findall(r"\]\(([^)]+)\)",text):
    assert (DOC.parent/link).resolve().exists(),link

# Independently reproduce the report's main aggregate claims.
for setting,expected in (("g100_l20000",("14.0%","32.1%","1.369")),
                         ("g50_l20000",("29.3%","38.1%","1.541"))):
    sub=[r for r in rows if r["kind"]=="lru" and r["setting"]==setting and r["mtp"]==8 and r["requests"]>=16]
    assert len(sub)==14 and all(r["ours_mean_us"]<r["memcpy_mean_us"] for r in sub)
    reductions=[100*(1-r["ours_mean_us"]/r["memcpy_mean_us"]) for r in sub]
    speed=math.exp(sum(math.log(r["memcpy_mean_us"]/r["ours_mean_us"]) for r in sub)/len(sub))
    actual=(f"{min(reductions):.1f}%",f"{max(reductions):.1f}%",f"{speed:.3f}")
    assert actual==expected and all(value in text for value in actual)
r=lookup["lru","g50_l20000",8,128,None]
assert f"{100*(for_row(r)['ours_to_reference']-1):.1f}%"=="9.4%"
assert f"{r['ours_mean_us']/r['local_mean_us']:.2f}"=="2.30"
assert manifest["physical_bandwidth_sha256"]==digest(DEST/"physical_bandwidth.json")
assert manifest["verification_sha256"]==digest(DEST/"verification.json")
assert manifest["verification_notes_sha256"]==digest(DEST/"verification_notes.md")
assert manifest["route_comparison_sha256"]==digest(ROOT/"temp/glm_trace/route_compare_20261030/summary.json")
assert manifest["parameter_review_sha256"]==digest(ROOT/"temp/glm_trace/100gb_review_20261030/summary.json")
parameter_audit=load(ROOT/"temp/glm_trace/100gb_review_20261030/audit.json")
assert parameter_audit["complete"] and parameter_audit["source_matrix_sha256"]==digest(SOURCE/"matrix.json")
review_dir=ROOT/"temp/glm_trace/100gb_review_20261030"
assert parameter_audit["summary_sha256"]==digest(review_dir/"summary.json")
assert parameter_audit["csv_sha256"]==digest(review_dir/"parameters.csv")
assert parameter_audit["readme_sha256"]==digest(review_dir/"README.md")
for name,sha in parameter_audit["source_sha256"].items():assert digest(review_dir/name)==sha
for f in parameter_audit["files"]:assert digest(ROOT/f["path"])==f["sha256"]
for name,sha in manifest["analysis_source_sha256"].items():assert digest(DEST/name)==sha
verification=load(DEST/"verification.json")
assert verification["matrix_unchanged_sha256"]==digest(SOURCE/"matrix.json")
assert all(abs(t["change_fraction"])<.03 for r in verification["key_repeats"] for t in r["mode_results"].values())
for batch,expected in ((32,"8.8%"),(64,"8.4%"),(128,"4.8%")):
    r=lookup["lru","pcie",8,batch,None]
    assert f"{100*(for_row(r)['ours_to_reference']-1):.1f}%"==expected
summary=dict(complete=True,source_matrix_sha256=digest(SOURCE/"matrix.json"),sample_table_rows=checked,
    full_link_summaries=summaries,ideal_bound_rows=bounds,near_fraction_rows=near_rows,
    independently_repeated_cases=3,repeat_comparison_rows=repeat_rows,physical_bandwidth_rows=physical_rows,
    supplemental_route_comparison_rows=route_rows,
    supplemental_link_diagnosis_rows=link_diagnosis_rows,supplemental_parameter_rows=parameter_rows,
    parameter_sweep_cells=review["sweep_cells"],parameter_confirmation_cells=review["confirmation_cells"],
    reference_definition=manifest["reference_definition"],
    figures=len(figures),figure_source_cells=fig_count,per_step_reference_cells_verified=fig_count,
    artifact_sha256=manifest["artifact_sha256"])
(DEST/"report_check.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(summary,ensure_ascii=False,indent=2))
