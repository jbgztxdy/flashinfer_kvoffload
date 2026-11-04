from pathlib import Path
import json,statistics
p=Path(__file__).resolve().parent;old=p.parent/'tune_unlimited_uniform_20261103'
f=old/'stall100_full_trace_gpu3.json'
if not f.exists():raise SystemExit('full trace pending')
d=json.loads(f.read_text());base=json.loads((old/'native_lru100_normal_gpu3.json').read_text());assert d['mapping_sha256']==base['mapping_sha256']
r=dict(pure_device_us=statistics.mean(base['pure_device_us']),normal_us=statistics.mean(base['passes'][0]['us']),mapping_sha256=d['mapping_sha256'],settings=[])
for s in d['settings']:
 rows=[]
 for i,(t,st) in enumerate(zip(s['us'],d['steps'])):
  b=st['far_references']*1152;ideal=max(base['pure_device_us'][i],b/100000)
  rows.append(dict(step=i,us=t,far_bytes=b,ideal_us=ideal,effective_GBps=b/t/1000))
 r['settings'].append(dict(name=s['name'],mean_us=statistics.mean(x['us'] for x in rows),mean_step_reference_us=statistics.mean(x['ideal_us'] for x in rows),cold_us=rows[0]['us'],cold_effective_GBps=rows[0]['effective_GBps'],steps=rows))
(p/'full_summary.json').write_text(json.dumps(r,indent=2))
print([(s['name'],s['mean_us'],s['cold_effective_GBps']) for s in r['settings']])
