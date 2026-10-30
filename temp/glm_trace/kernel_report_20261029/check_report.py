"""Verify complete published tables and exercise offline filtering/export in Chrome."""
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

DEST=Path(__file__).resolve().parent
ROOT=DEST.parents[2]
DOC=ROOT/"docs/mla_host_glm53_kernel_matrix_20261029.md"
meta=json.loads((DEST/"experiment.json").read_text())
audit=json.loads((DEST/"audit.json").read_text())
assert audit["complete"],"Run the full raw-data audit first"
rows=json.loads((DEST/"matrix.json").read_text())
expected={(kind,s["name"],m,b,f) for kind in ("nocache","lru") for s in meta["settings"]
          for m in (1,8) for b in meta["batches"][str(m)]
          for f in (meta["host_fractions"] if kind=="nocache" else [None])}
key=lambda r:(r["kind"],r["setting"],r["mtp"],r["requests"],r["host_fraction_target"])
lookup={key(r):r for r in rows}
assert len(lookup)==len(rows) and set(lookup)==expected
with (DEST/"matrix.csv").open(newline="") as stream:
    published=list(csv.DictReader(stream))
assert len(published)==len(rows)
for actual,reference in zip(published,rows):
    assert set(actual)==set(reference)
    assert all(actual[k]==("" if v is None else str(v)) for k,v in reference.items())

markdown=DOC.read_text()
assert "**状态：全组合实验与审计完成。**" in markdown
published_keys=set()
kind=setting=mtp=None
for line in markdown.splitlines():
    if line.startswith("## 4."):kind="nocache"
    elif line.startswith("## 5."):kind="lru"
    elif line.startswith("## 6."):kind=None
    if not kind:continue
    title=re.match(r"### .*（([^（）]+)）$",line)
    if title:setting=title[1]
    title=re.match(r"#### MTP([18])$",line)
    if title:mtp=int(title[1])
    if not re.match(r"\| \d",line):continue
    values=[v.strip() for v in line.strip("|").split("|")]
    if kind=="nocache":
        assert len(values)==9
        batch,query,target,actual,ours,copy,local,copy_ratio,local_ratio=values
        cell=(kind,setting,mtp,int(batch),int(target)/100)
        r=lookup[cell]
        want=[str(r["requests"]),str(r["query_rows"]),f"{r['host_fraction_target']*100:.0f}",
              f"{r['host_fraction_actual']*100:.2f}",f"{r['ours_mean_us']:.2f}",
              f"{r['memcpy_mean_us']:.2f}",f"{r['local_mean_us']:.2f}",
              f"{r['ours_to_memcpy']:.3f}",f"{r['ours_to_local_mean']:.3f}"]
    else:
        assert len(values)==15
        cell=(kind,setting,int(values[0]),int(values[1]),None)
        r=lookup[cell]
        want=[str(r["mtp"]),str(r["requests"]),str(r["query_rows"]),f"{r['ours_mean_us']:.2f}",
              f"{r['memcpy_mean_us']:.2f}",f"{r['local_mean_us']:.2f}",
              f"{r['ours_to_memcpy']:.3f}",f"{r['ours_to_local_mean']:.3f}",
              *[f"{r[k]*100:.2f}" for k in ("host_fraction_actual","ours_unique_cache_hit_rate",
                "ours_reference_cache_hit_rate","memcpy_unique_cache_hit_rate",
                "ours_remote_reference_fraction","final_remote_unique_fraction")],
              f"{r['ours_misses_per_step']:.1f}"]
    assert cell not in published_keys and values==want,(cell,values,want)
    published_keys.add(cell)
assert published_keys==expected

page=DOC.with_suffix(".html").read_text()
payload=re.search(r'<script id="measurements" type="application/json">(.*?)</script>',page,re.S)
assert payload and json.loads(payload[1])==rows
assert len(re.findall(r'<img src="data:image/png;base64,',page))==2
assert not re.search(r'<(?:script|img)[^>]+(?:src|href)="https?://',page)
browser_check=r'''
<script>
(async()=>{
 const tests=[];
 function ok(test,value){if(!value)throw Error(test);tests.push(test)}
 function choose(kind,link,mtp,host,batch){
   for(const [id,value] of Object.entries({kind,link,mtp,host,batch}))
     document.getElementById(id).value=value;
   document.getElementById('kind').dispatchEvent(new Event('input'));
   return [...document.querySelectorAll('#data tr')];
 }
 try{
   ok('full rendered row count',document.querySelectorAll('#data tr').length===rows.length);
   ok('LRU coverage',choose('lru','','','','').length===663);
   ok('no-LRU coverage',choose('nocache','','','','').length===4641);
   ok('zero Host fraction is selectable',choose('nocache','','','0','').length===663);
   ok('MTP8 request128 all links',choose('nocache','','8','0.3','128').length===17);
   const selected=choose('lru','g100_l20000','8','','128');
   ok('combined LRU filters',selected.length===1);
   const reference=rows.find(r=>r.kind==='lru'&&r.setting==='g100_l20000'&&r.mtp===8&&r.requests===128);
   ok('rendered mean matches source',selected[0].children[6].textContent===reference.ours_mean_us.toFixed(2));
   let savedBlob;
   URL.createObjectURL=b=>{savedBlob=b;return 'blob:report-check'};
   URL.revokeObjectURL=()=>{};
   HTMLAnchorElement.prototype.click=function(){};
   document.getElementById('download').click();
   ok('download creates CSV Blob',savedBlob instanceof Blob);
   const csv=await savedBlob.text();
   ok('CSV has real line breaks',csv.includes('\r\n')&&!csv.includes('\\r\\n'));
   ok('CSV contains selected raw provenance',csv.includes(reference.raw));
   ok('reset all filters',choose('','','','','').length===rows.length);
   const output=document.createElement('pre');output.id='report-verification';
   output.textContent=JSON.stringify({complete:true,tests,rows:rows.length,csv});document.body.append(output);
 }catch(error){
   const output=document.createElement('pre');output.id='report-verification';
   output.textContent=JSON.stringify({complete:false,tests,error:String(error)});document.body.append(output);
 }
})();
</script>
'''
chrome=shutil.which("google-chrome")
assert chrome,"Chrome is required for offline interaction validation"
with tempfile.TemporaryDirectory(prefix="glm53-report-check-") as directory:
    directory=Path(directory)
    target=directory/"report.html"
    target.write_text(page.replace("</html>",browser_check+"</html>"))
    result=subprocess.run([chrome,"--headless","--no-sandbox","--disable-gpu","--disable-dev-shm-usage",
        "--disable-background-networking","--no-first-run",f"--user-data-dir={directory/'profile'}",
        "--virtual-time-budget=5000","--dump-dom",target.as_uri()],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr[-3000:]
    match=re.search(r'<pre id="report-verification">(.*?)</pre>',result.stdout,re.S)
    assert match,"Browser did not finish the verification script"
    browser=json.loads(html.unescape(match[1]))
    assert browser["complete"],browser
    exported=list(csv.DictReader(io.StringIO(browser.pop("csv").lstrip("\ufeff"))))
    assert len(exported)==1 and exported[0]["kind"]=="lru"
    assert exported[0]["setting"]=="g100_l20000" and exported[0]["requests"]=="128"
summary=dict(complete=True,timestamp=time.strftime("%FT%T%z"),markdown_cells=len(published_keys),
             csv_cells=len(published),html_cells=len(rows),browser=browser,
             artifact_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (DOC,DOC.with_suffix(".html"),DEST/"matrix.csv",DEST/"matrix.json",
                     ROOT/"docs/assets/glm53_kernel_matrix_20261029/nocache.png",
                     ROOT/"docs/assets/glm53_kernel_matrix_20261029/lru.png")})
temporary=DEST/"report_check.json.new"
temporary.write_text(json.dumps(summary,indent=2)+"\n");temporary.replace(DEST/"report_check.json")
print(json.dumps(summary,indent=2))
