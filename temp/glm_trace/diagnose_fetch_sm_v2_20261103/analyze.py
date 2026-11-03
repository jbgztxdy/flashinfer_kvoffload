"""Audit diagnostic controls and overlapping CTA service budgets; never add overlap twice."""
import hashlib,json,statistics as S,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
load=lambda f:json.loads(f.read_text())
m=load(p/'manifest.json');formal=load(p.parent/'rebench_100gb_v2_20261103/audit.json')
assert hashlib.sha256((root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes()).hexdigest()==m['production_sha256']
rows=[]
for setting in ['g100_l0','g100_l20000']:
 d=load(p/('trace_'+setting+'.json'));assert d['complete'] and d['header_sha256']==m['variants']['trace']
 assert d['cache_payload_check']['unequal_elements']==0
 assert d['resources_start']['foreign']==d['resources_end']['foreign']==[]
 for lib in d['loaded_libraries']:assert hashlib.sha256((root/lib['path']).read_bytes()).hexdigest()==lib['sha256']
 f=next(x for x in formal['rows'] if x['setting']==setting)
 steps=[]
 for t,record in enumerate(d['passes'][0]['timeline']):
  r=record['ctas'];assert len(r)==108 and len({x[10] for x in r})==108
  assert sum(x[13] for x in r)==1024 and sum(x[14] for x in r)==128
  base=min(x[0] for x in r);end=max(x[9] for x in r);fetch=[x for x in r if x[12]]
  lastfetch=max(x[6] for x in fetch);prefix=max(x[4] for x in r)
  critical=max(r,key=lambda x:x[9]);attn=sum(x[16] for x in r);lru=sum(x[15] for x in r);fetchwork=sum(x[6]-x[5] for x in fetch);wait=sum(x[17] for x in r)
  steps.append(dict(step=t,event_us=d['passes'][0]['full_us'][t],total_timeline_us=(end-base)/1000,
   prepare_us=(max(x[2] for x in r)-base)/1000,resolve_barrier_us=(prefix-max(x[2] for x in r))/1000,
   prefix_us=(prefix-base)/1000,fetch_span_us=(lastfetch-min(x[5] for x in fetch))/1000,
   postfetch_tail_us=(end-lastfetch)/1000,last_ready_us=(max(x[24] for x in fetch)-base)/1000,
   deferred_cache_tail_us=(lastfetch-max(x[24] for x in fetch))/1000,
   fetch_compute_rows=sum(x[13] for x in fetch),
   fetch_sm_us=fetchwork/1000,fetch_sm_equivalent_wall_us=fetchwork/108000,
   attention_sm_us=attn/1000,lru_sm_us=lru/1000,wait_call_sm_us=wait/1000,
   attention_per108_us=attn/108000,attention_excluding_wait_per108_us=(attn-wait)/108000,
   far_wait_per_fetch_cta_us=S.mean(x[27] for x in fetch)/1000,
   physical_wait_per_fetch_cta_us=S.mean(x[26] for x in fetch)/1000,
   blocked_calls=sum(x[19] for x in r),wait_calls=sum(x[18] for x in r),
   postprefix_idle_equivalent_us=((end-prefix)*108-attn-lru-fetchwork)/108000,
   critical_cta=r.index(critical),critical_cta_fetch=bool(critical[12]),critical_rows=critical[13],critical_wait_us=critical[17]/1000,
   unique_miss_MB=d['passes'][0]['unique_misses'][t]*1152/1e6,
   actual_transfer_budget_us=d['passes'][0]['unique_misses'][t]*1152/100000,
   fetch_over_actual_budget_us=(lastfetch-min(x[5] for x in fetch))/1000-d['passes'][0]['unique_misses'][t]*1152/100000,
   transport_reference_us=f['step_transfer_us'][t],device_reference_us=f['step_device_us'][t]))
 means={k:S.mean(x[k] for x in steps) for k in steps[0] if k not in ['step','critical_cta','critical_cta_fetch']}
 regime={}
 for name,inds in [('cold',[0]),('warm',list(range(1,32))),('compute',f['regimes']['compute']['steps']),('transfer',f['regimes']['transfer']['steps'])]:
  regime[name]={k:S.mean(steps[i][k] for i in inds) for k in means};regime[name]['steps']=inds
 rows.append(dict(setting=setting,formal_us=f['ours_us'],formal_reference_us=f['reference_us'],instrumentation_slowdown=means['event_us']/f['ours_us']-1,
                  means=means,groups=regime,steps=steps,payload=d['cache_payload_check']))
d=load(p/'capacity_g100_l0.json');assert d['complete'] and d['header_sha256']==m['variants']['capacity']
assert d['resources_start']['foreign']==d['resources_end']['foreign']==[]
assert d['cache_payload_check']['unequal_elements']==0
for lib in d['loaded_libraries']:assert hashlib.sha256((root/lib['path']).read_bytes()).hexdigest()==lib['sha256']
for run in d['passes']:
 assert len(run['capacity_us']['84'])==len(run['capacity_us']['108'])==32
 assert all(a==b==0 for a,b in run['output_unequal_counts'])
 assert all(a==b==0 for a,b in run['state_unequal_counts'])
 assert len(set(run['smids']))==108
c84=[S.mean(run['capacity_us']['84'][i] for run in d['passes']) for i in range(32)]
c108=[S.mean(run['capacity_us']['108'][i] for run in d['passes']) for i in range(32)]
cap=dict(mean84_us=S.mean(c84),mean108_us=S.mean(c108),difference_us=S.mean(a-b for a,b in zip(c84,c108)),slowdown=S.mean(c84)/S.mean(c108)-1,
 pass84_us=[S.mean(run['capacity_us']['84']) for run in d['passes']],pass108_us=[S.mean(run['capacity_us']['108']) for run in d['passes']],
 full_diagnostic_us=S.mean(x for run in d['passes'] for x in run['full_us']),step84_us=c84,step108_us=c108,
 all_output_checks=32*3*2,all_state_checks=32*3*2,exact_equal=True,payload=d['cache_payload_check'])
cap['order_means']={}
for width in ['84','108']:
 for position in [0,1]:
  cap['order_means'][f'{width}_position{position}']=S.mean(run['capacity_us'][width][t] for run in d['passes'] for t,order in enumerate(run['capacity_order']) if order[position]==int(width))
r=dict(complete=True,production_unchanged=True,formal_reference_unchanged=True,trace=rows,capacity=cap,
 interpretation='Capacity control changes only available attention CTAs over completed real remap/cache. No fetch/prepare/LRU replay, no official performance/reference claim. Trace intervals overlap; SM-us/108 is service opportunity, not independently measured causal milliseconds. Wait counter includes even ready check calls and instrumentation; must not call it pure blocking.',
 limitations='32step prefix. Trace onepass each, +1.4-2.8% event perturbation, microsecond globaltimer resolution. 84/108 same-binary static stripe includes load balance and HBM contention effects, not isolated tensor core peak. One benchmark shape/hardware; no new fullmatrix/sanitizer/NVLink/H800.')
(p/'analysis.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'trace':[{k:v for k,v in x.items() if k not in ['steps','groups']} for x in rows], 'capacity':{k:v for k,v in cap.items() if not k.startswith('step')}},indent=2))
