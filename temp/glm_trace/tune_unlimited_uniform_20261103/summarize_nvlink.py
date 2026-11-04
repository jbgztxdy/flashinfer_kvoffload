import hashlib,json,statistics
from pathlib import Path
p=Path(__file__).resolve().parent
files=sorted(p.glob('nvlink*20261104*json'))+[p/'production_service22_control_nvlink_native_f24.json']
out=[]
for f in files:
 d=json.loads(f.read_text())
 if not isinstance(d,dict) or not d.get('complete'):continue
 passes=d['passes'];means=[statistics.mean(x['us']) for x in passes]
 row=dict(raw=f.name,raw_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),mean_us=statistics.mean(means),pass_means_us=means,header_sha256=d.get('header_sha256'),libraries=d.get('libraries'),launches=d.get('launches'),cache_payload_slots=d.get('cache_payload_slots'))
 if 'unique_misses' in passes[0]:row['unique_MB_per_step']=statistics.mean(passes[0]['unique_misses'])*1152/1e6
 if 'actual_far_read_bytes' in passes[0]:row['far_MB_per_step']=statistics.mean(passes[0]['actual_far_read_bytes'])/1e6
 out.append(row)
(p/'nvlink_summary_20261104.json').write_text(json.dumps(out,indent=2)+'\n')
for r in out:print(r['raw'],round(r['mean_us'],3),r.get('far_MB_per_step'),r.get('unique_MB_per_step'))
