"""CPU-only accounting of existing trace; no counterfactual GPU timing claim."""
import hashlib,json,statistics
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];a=json.loads((p.parent/'route_select_20261102/audit.json').read_text());records=[]
for requests in [16,128]:
 for setting,bw in [('nvlink',281.1237594272116),('g100_l0',100)]:
  x=next(x for x in a['rows'] if x['kind']=='lru' and x['requests']==requests and x['route']=='staged' and x['setting']==setting)
  d=json.loads((root/x['raw']).read_text());c=d['cases'][0];f=json.loads((root/x['reference_raw']).read_text());fc=f['cases'][0];inds={t:i for i,t in enumerate(fc['step_indices'])};compute=[statistics.mean(fc['timings']['local']['per_step_samples_us'][inds[t]]) for t in c['step_indices']]
  nocache_bound=statistics.mean(max(t,n*1152/(1000*bw)) for t,n in zip(compute,c['far_unique']))
  records.append(dict(requests=requests,setting=setting,original_far_fraction=c['host_fraction_actual'],reference_hit_rate=c['ours_reference_cache_hit_rate'],unique_hit_rate=c['ours_unique_cache_hit_rate'],remote_unique_MB_without_LRU=statistics.mean(c['far_unique'])*1152/1e6,remote_unique_MB_with_LRU=statistics.mean(c['ours_unique_misses'])*1152/1e6,compute_us=statistics.mean(compute),reference_without_LRU_us=nocache_bound,reference_with_LRU_us=x['reference_us'],measured_with_LRU_us=x['ours_us'],without_LRU_runtime_not_measured=True,source_raw=x['raw'],source_sha256=hashlib.sha256((root/x['raw']).read_bytes()).hexdigest()))
(p/'existing_accounting.json').write_text(json.dumps(dict(cpu_only=True,records=records,warning='No actual capacity0 timings; derived traffic/reference only. Existing lowfar nocache cases have different residency and cannot be cache toggles.'),indent=2)+'\n');print(json.dumps(records,indent=2))
