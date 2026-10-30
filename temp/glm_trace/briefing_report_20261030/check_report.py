"""Check briefing tables, figure source cells and headline claims against audited data."""
import hashlib
import json
import math
from pathlib import Path
import re

DEST=Path(__file__).resolve().parent
ROOT=DEST.parents[2]
SOURCE=ROOT/"temp/glm_trace/kernel_report_20261029"
DOC=ROOT/"docs/mla_host_glm53_report_external.md"
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
mode=None;checked=0;summaries=0;bounds=0
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
        result.extend([f"{(1-r['ours_mean_us']/r['memcpy_mean_us'])*100:.1f}%",
                       f"{r['ours_mean_us']/r['local_mean_us']:.3f}×"])
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
    else:
        kind,link=fields[0].rsplit("，",1)
        kind="nocache" if kind.startswith("无LRU") else "lru"
        r=lookup[kind,settings[link],8,128,.3 if kind=="nocache" else None]
        bound=r["ideal_compute_transfer_bound_us"]
        result=[fields[0],*[f"{r[k]:.2f}" for k in ("ours_mean_us","memcpy_mean_us","local_mean_us")],
                f"{bound:.2f}",f"{(r['ours_mean_us']/bound-1)*100:.1f}%"]
        assert result==fields,(result,fields)
        bounds+=1
assert (checked,summaries,bounds)==(26,17,4),(checked,summaries,bounds)

figures=load(DEST/"figure_data.json")
assert len(figures)==6
fig_count=0
for name,figure in figures.items():
    assert (ROOT/figure["file"]).exists()
    assert len(figure["rows"])==manifest["figure_cells"][name]
    for r in figure["rows"]:
        assert r==lookup[key(r)]
        fig_count+=1
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
assert f"{100*(r['ours_mean_us']/r['ideal_compute_transfer_bound_us']-1):.1f}%"=="9.4%"
assert f"{r['ours_mean_us']/r['local_mean_us']:.2f}"=="2.30"
summary=dict(complete=True,source_matrix_sha256=digest(SOURCE/"matrix.json"),sample_table_rows=checked,
    full_link_summaries=summaries,ideal_bound_rows=bounds,figures=len(figures),figure_source_cells=fig_count,
    artifact_sha256=manifest["artifact_sha256"])
(DEST/"report_check.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
print(json.dumps(summary,ensure_ascii=False,indent=2))
