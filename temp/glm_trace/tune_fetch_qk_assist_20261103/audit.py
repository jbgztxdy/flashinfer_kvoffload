"""Same-input performance, genuine assistance counts, and diagnostic controls."""
import hashlib,json,statistics as S,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
load=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
m=load(p/'manifest.json');formal=load(p.parent/'rebench_100gb_v2_20261103/audit.json');compute=formal['rows'][0]['step_device_us']
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh')==m['production_sha256']
assert sha(root/'include/flashinfer/attention/far_link_emu.cuh')==m['far_sha256']
rows=[];raws={};files=[];launches=0
for f in sorted(p.glob('*_g100_*_f*.json')):
 d=load(f);assert d['complete'],str(f)
 v=d['variant'];cfg=d['config'];assert cfg['steps']==32 and cfg['mtp']==8 and cfg['requests']==128 and cfg['positions']==256
 assert d['resources_start']['foreign']==d['resources_end']['foreign']==[] and 'membind: 1' in d['numa_policy']
 expected=m['production_sha256'] if v=='base' else m[v+'_sha256'];assert d['header_sha256']==expected
 assert d['far_sha256']==m['far_sha256'];assert len(d['passes'])==cfg['passes']
 assert d['cache_payload_slots']==524288
 for lib in d['libraries']:assert sha(root/lib['path'])==lib['sha256']
 for q in d['passes']:
  assert len(q['us'])==len(q['unique_misses'])==len(q['assist_stats'])==32
  assert all(x[0]==x[1] for x in q['assist_stats']);launches+=32
 samples=[x for q in d['passes'] for x in q['us']];stats=[x for q in d['passes'] for x in q['assist_stats']]
 miss=[S.mean(q['unique_misses'][i] for q in d['passes']) for i in range(32)];ref=[max(c,x*1152/100000) for c,x in zip(compute,miss)]
 r=dict(raw=f.name,variant=v,setting=d['setting']['name'],fetch_ctas=cfg['fetch_ctas'],item=cfg.get('host_item_entries',64),passes=cfg['passes'],mean_us=S.mean(samples),pass_us=[S.mean(q['us']) for q in d['passes']],reference_us=S.mean(ref),unique_miss_MB=S.mean(miss)*1152/1e6,assisted_pairs=S.mean(x[1] for x in stats),QK_pair_fraction=S.mean(x[1] for x in stats)/32768,worker_waited_jobs=S.mean(x[3] for x in stats),canceled_before_compute=S.mean(x[2] for x in stats))
 r['over_reference']=r['mean_us']/r['reference_us']-1
 if v in ['fused','controls']:
  diag=[x for q in d['passes'] for x in q['diagnostic_ns']]
  r.update(wait_equivalent_us=S.mean(x[0] for x in diag)/108000,max_CTA_wait_us=S.mean(x[1] for x in diag)/1000,fetch_span_us=S.mean(x[2] for x in diag)/1000,helper_drain_us=S.mean(x[3] for x in diag)/1000)
  r['fetch_payload_GBps']=r['unique_miss_MB']*1e6/(r['fetch_span_us']*1000)
 rows.append(r);raws[f.name]=d;files.append(dict(path=f.name,sha256=sha(f)))
comparisons=[]
for r in rows:
 if r['variant'] in ['base','controls']:continue
 b=next(x for x in rows if x['variant']=='base' and x['setting']==r['setting'] and x['fetch_ctas']==r['fetch_ctas'])
 assert raws[r['raw']]['bindings']==raws[b['raw']]['bindings']
 comparisons.append(dict(variant=r['variant'],setting=r['setting'],fetch_ctas=r['fetch_ctas'],base_us=b['mean_us'],candidate_us=r['mean_us'],slowdown=r['mean_us']/b['mean_us']-1,decision='reject; not installed'))
controls=[x for x in rows if x['variant']=='controls'];control_result=None
if len(controls)==3:
 a,b,c=(next(x for x in controls if x['item']==i) for i in [63,62,64])
 control_result=dict(no_queue_no_QK_us=a['mean_us'],bookkeeping_no_QK_us=b['mean_us'],full_assist_us=c['mean_us'],bookkeeping_increment_us=b['mean_us']-a['mean_us'],assistance_net_increment_us=c['mean_us']-b['mean_us'],scope='One cold32step pass, same binary and actual inputs, all math/cache operations retained. Descriptive controlled differences, no instruction/HBM causal split.')
r=dict(timestamp=time.strftime('%FT%T%z'),complete=len(controls)==3,rows=rows,comparisons=comparisons,controls=control_result,timed_full_launches=launches,raw_files=files,production_unchanged=True,emulator_unchanged=True,formal_reference_unchanged=True,limits='GPU3/NUMA1 same-cardHBM v2 simulation, not NVLink. BF16MTP8R128LRU4096,32steps. No adoption/full1024/H800/nativephysical/FP16/new sanitizers. Timer budgets are overlapping SM service, not independently removable walltime.')
(p/'audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
