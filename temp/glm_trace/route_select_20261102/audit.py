"""Audit both routes, then select by full-sample means; do not rewrite references or old matrices."""
import csv,hashlib,json,math,statistics,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
load=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
m=load(p/'manifest.json');rows=[];incomplete=[];segments=launches=0
for f in sorted((p/'raw').glob('*.json')):
 d=load(f);cfg=d['configuration']
 if not d['complete']:incomplete.append(str(f.relative_to(root)));continue
 assert d['source_sha256']['include/flashinfer/attention/mla_host_batch.cuh']==m['header_sha256']
 assert d['source_sha256']['include/flashinfer/attention/mla.cuh']==m['math_sha256']
 assert d['resources_start']['foreign']==d['resources_end']['foreign']==[]
 assert all(x['tier']!='far_peer_hbm_gpu3' for x in d['residency'])
 assert cfg['route'] in ['staged','inplace'] and cfg['passes']==3
 for k,h in d['source_sha256'].items():assert sha(p/'source_snapshots'/(h+Path(k).suffix))==h
 refkind='floor' if cfg['kind']=='lru' else 'snapshots'
 ref=root/f"temp/glm_trace/kernel_report_20261029/raw/{refkind}_floor_mtp{cfg['mtp']}_b{cfg['requests']}.json"
 fd=load(ref);assert fd['complete'] and fd['bindings']==d['bindings']
 assert fd['source_sha256']['include/flashinfer/attention/mla.cuh']==m['math_sha256']
 for c in d['cases']:
  fc=next(x for x in fd['cases'] if x['host_fraction_requested']==c['host_fraction_requested'])
  ticks=c['step_indices'];index={t:i for i,t in enumerate(fc['step_indices'])}
  compute=[statistics.mean(fc['timings']['local']['per_step_samples_us'][index[t]]) for t in ticks]
  for mode,timing in c['timings'].items():
   samples=timing['per_step_samples_us'];assert len(samples)==len(ticks)
   assert all(len(x)==cfg['passes'] and all(math.isfinite(y) and y>0 for y in x) for x in samples)
   assert abs(statistics.mean(y for x in samples for y in x)-timing['mean_us'])<1e-7
   for repeat in range(cfg['passes']):
    end=0
    for g in [g for g in c['guards'] if g['mode']==mode and g['pass_index']==repeat]:
     assert g['valid'] and g['first']==end;end=g['last'];segments+=1
    assert end==len(ticks)
   launches+=len(ticks)*cfg['passes']
  assert max(c['numerical_checks']['max_abs_error'].values())<.01
  if cfg['kind']=='lru':
   assert all(x==c['far_unique'][0] for x in c['ours_unique_miss_samples'][0])
   assert all(x==c['far_refs'][0] for x in c['ours_reference_miss_samples'][0])
  bw=cfg['setting'].get('bandwidth_GBps')
  if not bw:bw=next(x['reference_GBps'] for x in load(root/'temp/glm_trace/briefing_report_20261030/physical_bandwidth.json')['settings'] if x['name']==cfg['setting']['name'])
  bound=statistics.mean(max(x,n*1152/(1000*bw)) for x,n in zip(compute,c['ours_unique_misses']))
  rows.append(dict(kind=cfg['kind'],requests=cfg['requests'],mtp=cfg['mtp'],fraction=c['host_fraction_requested'],route=cfg['route'],setting=cfg['setting']['name'],positions=cfg['positions'],passes=cfg['passes'],fetch=cfg['fetch_ctas'],split=cfg['host_split'],ours_us=c['timings']['ours']['mean_us'],pass_mean_us=c['timings']['ours']['pass_mean_us'],memcpy_us=c['timings']['memcpy']['mean_us'],local_us=statistics.mean(compute),reference_us=bound,over_reference=c['timings']['ours']['mean_us']/bound-1,hit_rate=c['ours_reference_cache_hit_rate'],unique_miss_MB=statistics.mean(c['ours_unique_misses'])*1152/1e6,miss_reference_MB=statistics.mean([statistics.mean(x) for x in c['ours_reference_miss_samples']])*1152/1e6 if c['ours_reference_miss_samples'] else None,raw=str(f.relative_to(root)),raw_sha256=sha(f),reference_raw=str(ref.relative_to(root)),reference_sha256=sha(ref),bindings_sha256=hashlib.sha256(json.dumps(d['bindings'],sort_keys=True).encode()).hexdigest()))
index={(x['kind'],x['requests'],x['mtp'],x['fraction'],x['setting'],x['route']):x for x in rows};pairs=[]
for k,s in index.items():
 if k[-1]!='staged':continue
 direct=index.get((*k[:-1],'inplace'))
 if not direct:continue
 assert s['bindings_sha256']==direct['bindings_sha256'] and s['fetch']==direct['fetch'] and s['split']==direct['split']
 chosen=min([s,direct],key=lambda x:x['ours_us'])
 pairs.append(dict(kind=s['kind'],requests=s['requests'],mtp=s['mtp'],fraction=s['fraction'],setting=s['setting'],staged_us=s['ours_us'],inplace_us=direct['ours_us'],selected_route=chosen['route'],selected_us=chosen['ours_us'],local_us=chosen['local_us'],reference_us=chosen['reference_us'],over_reference=chosen['over_reference'],reduction_vs_staged=1-chosen['ours_us']/s['ours_us'],route_difference=abs(s['ours_us']-direct['ours_us'])/max(s['ours_us'],direct['ours_us']),staged_raw=s['raw'],inplace_raw=direct['raw'],selected_raw=chosen['raw'],misses_can_differ='Exact launch LRU ties/admission may differ; each route uses its own actual miss count.',selection_scope='Empirical offline route choice for measured key, not a universally optimal policy.'))
summary=dict(timestamp=time.strftime('%FT%T%z'),rows=rows,pairs=pairs,incomplete=incomplete,timed_launches=launches,accepted_segments=segments,overall_goal_complete=False)
(p/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
(p/'selected_routes.json').write_text(json.dumps(dict(header_sha256=m['header_sha256'],hardware='A100-SXM4-40GB NV12 NUMA0',selection='full-step full-pass arithmetic mean; empirical routing, not source optimization',profiles=pairs),indent=2)+'\n')
print(json.dumps(dict(rows=len(rows),pairs=len(pairs),incomplete=incomplete,timed_launches=launches,accepted_segments=segments),indent=2))
for x in pairs:print(x['kind'],x['requests'],x['fraction'],x['setting'],round(x['staged_us'],3),round(x['inplace_us'],3),x['selected_route'],round(x['over_reference']*100,3))
