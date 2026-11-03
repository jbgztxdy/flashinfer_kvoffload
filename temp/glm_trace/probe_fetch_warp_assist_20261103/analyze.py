from pathlib import Path
import hashlib,json,statistics as S
p=Path(__file__).resolve().parent;root=p.parents[2]
old=json.loads((p.parent/'probe_fetch_qk_assist_20261103/result.json').read_text());d=json.loads((p/'result.json').read_text())
assert d['complete'] and d['sampled_QK_exact'] and d['payload_exact'] and d['all_recorded_arrivals_legal']
assert all(not x for x in d['local_bytes']) and d['far_sha256']==old['far_sha256']
rows=[]
for lat in [0,20000]:
 for ctas in [24,32]:
  modes={}
  for mode in [0,1,2]:
   rr=[x for x in d['cases'] if x['latency_ns']==lat and x['ctas']==ctas and x['mode']==mode]
   assert len(rr)==3 and all(x['early']==0 for x in rr)
   us=S.mean(x['us'] for x in rr)
   modes[str(mode)]=dict(mean_us=us,samples_us=[x['us'] for x in rr],payload_GBps=d['bytes']/(us*1000),fetch_span_us=S.mean(x['fetch_span_us'] for x in rr),QK_jobs=S.mean(x['completed_QK_jobs'] for x in rr),helper_tail_us=S.mean(x['helper_tail_us'] for x in rr))
  rows.append(dict(latency_ns=lat,fetch_CTAs=ctas,modes=modes))
serial=[]
for lat in [0,20000]:
 modes={mode:S.mean(x['us'] for x in old['cases'] if x['mode']==mode and x['latency_ns']==lat) for mode in [0,1,2]}
 serial.append(dict(latency_ns=lat,us=modes,plain_GBps=old['bytes']/(modes[0]*1000),interleaved_GBps=old['bytes']/(modes[1]*1000),decision='reject serial wholeQK quantum'))
r=dict(complete=True,serial=serial,spatial=rows,resources=dict(registers=d['registers'],local_bytes=d['local_bytes'],dynamic_shared=d['dynamic_shared']),production_unchanged=True,limits='Microprobe only: no real worker consumes QK, no LRU or simultaneous84workers. Not an end-to-end result. All3361jobs finish during fetch; helper capacity not saturated/measured.')
(p/'analysis.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
