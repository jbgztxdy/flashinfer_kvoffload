from pathlib import Path
import json,hashlib,statistics
p=Path(__file__).resolve().parent;root=p.parents[2]
phases=['kernel_start','prepare_grid_done','resolve_done','planner_done','lru_grid_done','service_initialized','last_packet_published','attention_drain_done']
raw={policy:json.loads((p/f'claimrow_timeline_native_unlimited_f{f}.json').read_text()) for policy,f in [('roundrobin',23),('claimrow_quota',24)]}
assert raw['roundrobin']['libraries']==raw['claimrow_quota']['libraries']
assert raw['roundrobin']['header_sha256']==raw['claimrow_quota']['header_sha256']==hashlib.sha256((p/'claimrow_timeline.cuh').read_bytes()).hexdigest()
for lib in raw['roundrobin']['libraries']:assert hashlib.sha256((root/lib['path']).read_bytes()).hexdigest()==lib['sha256']
result=dict(same_actual_library_verified=True,scope='singlebinary normalGPU3 same-deviceHBM diagnostic; one32step coldpass perpolicy; phase costs notadditive causal decomposition; no100GB prediction',phases=phases,policies={})
for policy,r in raw.items():
 assert r['complete'] and r['config']['timeline_diagnostic'] and len(r['passes'])==1
 sample=r['passes'][0];steps=[]
 for step,(us,stamps,miss,refs) in enumerate(zip(sample['us'],sample['phase_timestamps_ns'],sample['unique_misses'],sample['reference_misses'])):
  assert len(stamps)==108 and all(len(a)==8 and min(a)>0 for a in stamps)
  for a in stamps:assert all(a[i]<=a[j] for i,j in [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7)])
  origin=min(a[0] for a in stamps)
  latest=[max(a[i] for a in stamps) for i in range(8)]
  span=(latest[7]-origin)/1000
  assert span<=us+20,('phase clock/event mismatch',step,span,us)
  local_plan=[(a[3]-a[2])/1000 for a in stamps]
  steps.append(dict(step=step,event_us=us,unique=miss,logical_far_bytes=miss*1152,far_references=refs,latest_phase_from_start_us=dict(zip(phases,[(x-origin)/1000 for x in latest])),planner_local_mean_us=statistics.mean(local_plan),planner_local_max_us=max(local_plan),after_all_publish_us=(latest[7]-latest[6])/1000,attention_service_local_mean_us=statistics.mean((a[7]-a[5])/1000 for a in stamps),event_minus_observed_span_us=us-span))
 result['policies'][policy]=dict(mean_event_us=statistics.mean(x['event_us'] for x in steps),mean_unique=statistics.mean(x['unique'] for x in steps),mean_latest_phases_us={k:statistics.mean(x['latest_phase_from_start_us'][k] for x in steps) for k in phases},mean_planner_local_us=statistics.mean(x['planner_local_mean_us'] for x in steps),mean_planner_local_max_us=statistics.mean(x['planner_local_max_us'] for x in steps),mean_after_all_publish_us=statistics.mean(x['after_all_publish_us'] for x in steps),mean_event_minus_observed_span_us=statistics.mean(x['event_minus_observed_span_us'] for x in steps),steps=steps)
a,b=[result['policies'][x] for x in ['roundrobin','claimrow_quota']]
result['quota_minus_roundrobin_us']=dict(event=b['mean_event_us']-a['mean_event_us'],latest_phases={k:b['mean_latest_phases_us'][k]-a['mean_latest_phases_us'][k] for k in phases},planner_local=b['mean_planner_local_us']-a['mean_planner_local_us'],after_all_publish=b['mean_after_all_publish_us']-a['mean_after_all_publish_us'])
(p/'claimrow_timeline_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='steps'} for k,v in result['policies'].items()},indent=2));print('delta',json.dumps(result['quota_minus_roundrobin_us'],indent=2))
