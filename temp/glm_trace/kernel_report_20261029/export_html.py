"""Portable, self-contained report with searchable measured-data tables."""
import base64
import html
import json
from pathlib import Path
from markdown_it import MarkdownIt

DEST=Path(__file__).resolve().parent
ROOT=DEST.parents[2]
doc=ROOT/"docs/mla_host_glm53_kernel_matrix_20261029.md"
body=MarkdownIt("default",{"html":False}).render(doc.read_text())
for kind in ("nocache","lru"):
    relative=f"assets/glm53_kernel_matrix_20261029/{kind}.png"
    path=doc.parent/relative
    if path.exists():
        data=base64.b64encode(path.read_bytes()).decode()
        body=body.replace(f'src="{relative}"',f'src="data:image/png;base64,{data}"')
rows=json.loads((DEST/"matrix.json").read_text())
payload=json.dumps(rows,ensure_ascii=False,separators=(",",":")).replace("<","\\u003c")
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GLM-5.3 sparse MLA kernel 实测报告</title>
<style>
:root{color-scheme:light}body{font:15px/1.65 system-ui,sans-serif;color:#182337;margin:24px auto;padding:0 24px;max-width:1560px}
h1,h2,h3,h4{line-height:1.3;color:#122543}h2{margin-top:2em;border-bottom:1px solid #ccd6e2;padding-bottom:.35em}
table{border-collapse:collapse;font-size:12px;width:100%;margin:1em 0;font-variant-numeric:tabular-nums}
th,td{border:1px solid #d2dbe6;padding:5px 8px;text-align:right;white-space:nowrap}th{background:#eef3f9}
tr:nth-child(even){background:#f8fafc}img{max-width:100%;height:auto}a{color:#135da5}
code{background:#edf1f6;padding:1px 3px}#explorer{border:1px solid #b6c8df;background:#f8fbff;border-radius:8px;padding:18px}
label{display:inline-block;margin:4px 12px 4px 0}select,input,button{font:inherit;padding:4px;margin-left:5px}
.tablebox{overflow:auto;max-height:62vh}#count{font-weight:600}.better{background:#e3f2e9}.worse{background:#fff1e5}
nav{position:sticky;top:0;background:#fff;padding:8px 0;border-bottom:1px solid #ccd6e2;z-index:2}
@media print{body{max-width:none;margin:0;padding:0;font-size:10px}nav,#explorer{display:none}table{font-size:7px}tr{break-inside:avoid}h2,h3,h4{break-after:avoid}img{max-height:85vh;object-fit:contain}}
</style>
<nav><a href="#explorer">筛选数据</a> · <a href="#report">完整报告</a> · 本文件包含图表与全部已测数据，可离线打开</nav>
<h1>GLM-5.3 kernel 性能实测</h1>
<section id="explorer"><p>以下筛选只显示实测格子；完整覆盖和审计状态以正文为准。时间单位 µs，比值小于 1 表示本文更快。</p>
<label>场景<select id="kind"><option value="">全部</option><option value="nocache">无 LRU</option><option value="lru">真实 trace＋LRU</option></select></label>
<label>链路<select id="link"><option value="">全部</option></select></label>
<label>MTP<select id="mtp"><option value="">全部</option><option>1</option><option>8</option></select></label>
<label>Host 比例<select id="host"><option value="">全部</option></select></label>
<label>请求数<input id="batch" type="number" min="1" placeholder="全部" style="width:85px"></label>
<button id="download">下载当前筛选 CSV</button><p id="count"></p>
<div class="tablebox"><table><thead><tr><th>场景</th><th>链路</th><th>MTP</th><th>请求</th><th>目标Host%</th><th>实际Host%</th><th>本文</th><th>Memcpy</th><th>纯本地</th><th>本文÷Memcpy</th><th>本文÷本地</th><th>LRU unique命中%</th><th>LRU引用命中%</th><th>最终搬运%</th></tr></thead><tbody id="data"></tbody></table></div></section>
<main id="report">BODY</main>
<script id="measurements" type="application/json">PAYLOAD</script>
<script>
const rows=JSON.parse(document.getElementById('measurements').textContent);
const controls=['kind','link','mtp','host','batch'].map(x=>document.getElementById(x));
const unique=(key)=>[...new Set(rows.map(r=>r[key]))];
for(const link of unique('setting')){const o=document.createElement('option');o.value=link;o.textContent=link;controls[1].append(o)}
for(const f of unique('host_fraction_target').filter(x=>x!==null).sort((a,b)=>a-b)){const o=document.createElement('option');o.value=f;o.textContent=100*f+'%';controls[3].append(o)}
function filtered(){const [kind,link,mtp,host,batch]=controls.map(x=>x.value);return rows.filter(r=>(!kind||r.kind===kind)&&(!link||r.setting===link)&&(!mtp||r.mtp===+mtp)&&(!host||r.host_fraction_target===+host)&&(!batch||r.requests===+batch))}
const fixed=(x,n=2)=>x===null?'—':Number(x).toFixed(n);
function render(){const values=filtered();document.getElementById('count').textContent=values.length+' 个已测组合';const target=document.getElementById('data');target.replaceChildren();
for(const r of values){const tr=document.createElement('tr');const values=[r.kind==='lru'?'trace＋LRU':'无LRU',r.setting,r.mtp,r.requests,r.host_fraction_target===null?'—':fixed(r.host_fraction_target*100,0),fixed(r.host_fraction_actual*100),fixed(r.ours_mean_us),fixed(r.memcpy_mean_us),fixed(r.local_mean_us),fixed(r.ours_to_memcpy,3),fixed(r.ours_to_local_mean,3),r.kind==='lru'?fixed(r.ours_unique_cache_hit_rate*100):'—',r.kind==='lru'?fixed(r.ours_reference_cache_hit_rate*100):'—',fixed(r.final_remote_unique_fraction*100)];
values.forEach((v,i)=>{const td=document.createElement('td');td.textContent=v;if(i===9)td.className=r.ours_to_memcpy<1?'better':'worse';tr.append(td)});target.append(tr)}}
for(const c of controls)c.addEventListener('input',render);render();
document.getElementById('download').addEventListener('click',()=>{const data=filtered();if(!data.length)return;const keys=Object.keys(data[0]);const esc=x=>'"'+String(x===null?'':typeof x==='object'?JSON.stringify(x):x).replaceAll('"','""')+'"';const csv='\\ufeff'+[keys.map(esc).join(','),...data.map(r=>keys.map(k=>esc(r[k])).join(','))].join('\\r\\n');const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='glm53-kernel-filtered.csv';a.click();URL.revokeObjectURL(url)});
</script></html>'''
page=page.replace("BODY",body).replace("PAYLOAD",payload)
target=doc.with_suffix(".html")
temporary=target.with_suffix(".html.new");temporary.write_text(page);temporary.replace(target)
print(json.dumps(dict(file=str(target),measured_cells=len(rows),bytes=target.stat().st_size)))
