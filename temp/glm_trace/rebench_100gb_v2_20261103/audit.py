"""Audit fresh v2 raw; the formal reference remains mean(stepwise max)."""
import hashlib,json,math,statistics,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
load=lambda f:json.loads(f.read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
m=load(p/'manifest.json');rows=[];files=[];launches=segments=0;pending=[]
raws=[]
for f in sorted((p/'gpu3/raw').glob('*.json')):
 d=load(f)
 if not d['complete']:pending.append(str(f.relative_to(root)));continue
 cfg=d['configuration'];assert cfg['far_emulator_version']==2 and cfg['memcpy_path_version']==3
 assert cfg['hardware_group']=='gpu3_a100_pcie' and cfg['far_peer_is_same_device'] and cfg['far_peer_physical_device']==3
 assert cfg['mtp']==8 and cfg['requests']==128 and cfg['positions']==256 and cfg['steps']==32 and cfg['passes']==3
 assert cfg['memcpy_attention_build']=='normal'
 assert d['extra_cuda_flags']==('' if cfg['kind']=='floor' else '-DFLASHINFER_FAR_EMU')
 if cfg['kind']=='lru':assert cfg['setting']['emulated'] and cfg['setting']['bandwidth_GBps']==100
 assert 'membind: 1' in d['numa_policy']
 assert d['resources_start']['foreign']==d['resources_end']['foreign']==[]
 for rel,digest in m['files'].items():assert d['source_sha256'][rel]==digest
 for rel,digest in d['source_sha256'].items():assert sha(p/'gpu3/source_snapshots'/(digest+Path(rel).suffix))==digest
 for lib in d['loaded_attention_libraries']:
  assert sha(root/lib['path'])==lib['sha256']
 assert all(x['timing_queue'] in ['cuda_graph','eager_guard'] for c in d['cases'] for x in c['guards'])
 for c in d['cases']:
  assert c['step_indices']==list(range(32))
  assert max(c['numerical_checks']['max_abs_error'].values())<.01
  for mode,timing in c['timings'].items():
   samples=timing['per_step_samples_us'];assert len(samples)==32 and all(len(v)==3 for v in samples)
   assert all(math.isfinite(v) and v>0 for row in samples for v in row)
   assert abs(statistics.mean(v for row in samples for v in row)-timing['mean_us'])<1e-7
   for repeat in range(3):
    assert abs(statistics.mean(row[repeat] for row in samples)-timing['pass_mean_us'][repeat])<1e-7
    end=0
    for guard in [g for g in c['guards'] if g['mode']==mode and g['pass_index']==repeat]:
     assert guard['valid'] and guard['first']==end
     if guard['timing_queue']!='cuda_graph':assert guard['guard_ms']>=guard['enqueue_ms']+max(1,.05*guard['enqueue_ms'])
     end=guard['last'];segments+=1
    assert end==32
   launches+=96
  if cfg['kind']=='lru':
   assert cfg['capacity']==4096
   assert all(v==c['far_unique'][0] for v in c['ours_unique_miss_samples'][0])
   hit=1-sum(map(sum,c['ours_reference_miss_samples']))/3/sum(c['far_refs'])
   assert abs(hit-c['ours_reference_cache_hit_rate'])<1e-12
  raws.append((f,d,c))
 files.append(dict(path=str(f.relative_to(root)),sha256=sha(f)))
floors=[z for z in raws if z[1]['configuration']['kind']=='floor']
assert len(floors)==1,(len(floors),pending)
ff,fd,fc=floors[0];compute=[statistics.mean(v) for v in fc['timings']['local']['per_step_samples_us']]
for f,d,c in raws:
 cfg=d['configuration']
 if cfg['kind']=='floor':continue
 assert cfg['kind']=='lru' and fd['bindings']==d['bindings']
 bandwidth=cfg['setting']['bandwidth_GBps'];assert bandwidth==100
 transfer=[v*1152/(bandwidth*1000) for v in c['ours_unique_misses']]
 stepref=[max(a,b) for a,b in zip(compute,transfer)]
 ours,memcpy=(c['timings'][mode]['mean_us'] for mode in ['ours','memcpy'])
 r=dict(setting=cfg['setting']['name'],ours_us=ours,memcpy_us=memcpy,device_us=statistics.mean(compute),transfer_us=statistics.mean(transfer),reference_us=statistics.mean(stepref),over_reference=ours/statistics.mean(stepref)-1,saving_vs_memcpy=1-ours/memcpy,unique_miss_MB=statistics.mean(c['ours_unique_misses'])*1152/1e6,reference_hit_rate=c['ours_reference_cache_hit_rate'],ours_pass_us=c['timings']['ours']['pass_mean_us'],memcpy_pass_us=c['timings']['memcpy']['pass_mean_us'],compute_dominated_steps=sum(a>=b for a,b in zip(compute,transfer)),transfer_dominated_steps=sum(a<b for a,b in zip(compute,transfer)),step_ours_us=[statistics.mean(v) for v in c['timings']['ours']['per_step_samples_us']],step_device_us=compute,step_transfer_us=transfer,step_reference_us=stepref,raw=str(f.relative_to(root)),reference_raw=str(ff.relative_to(root)))
 r['goal110_us']=1.1*r['reference_us'];r['remaining_reduction_to110']=1-r['goal110_us']/ours
 r['regimes']={}
 for regime,indices in [('compute',[i for i,(a,b) in enumerate(zip(compute,transfer)) if a>=b]),('transfer',[i for i,(a,b) in enumerate(zip(compute,transfer)) if a<b])]:
  if not indices:continue
  actual=statistics.mean(r['step_ours_us'][i] for i in indices);reference=statistics.mean(stepref[i] for i in indices)
  r['regimes'][regime]=dict(steps=indices,mean_ours_us=actual,mean_reference_us=reference,mean_excess_us=actual-reference,over_reference=actual/reference-1)
 r['first_step']=dict(ours_us=r['step_ours_us'][0],reference_us=stepref[0],device_us=compute[0],transfer_us=transfer[0])
 rows.append(r)
assert len(rows)==2 and not pending,(len(rows),pending)
result=dict(timestamp=time.strftime('%FT%T%z'),complete=True,hardware='GPU3/A100-PCIE NUMA1 same-card far HBM',emulator_version=2,rows=rows,timed_launches=launches,accepted_segments=segments,raw_files=files,pending=pending,reference_formula='mean_t(max(mean_pass(device_t),mean_pass(actual_unique_miss_t)*1152/(1000*100)))',limits='32steps/256positions x3coldpasses. No full1024trace,physicalNVLink/H800,newkernel mechanism or causalphase attribution. Memcpy usesv2gather+normaldeviceattention; historicalv1totals notstrictnewold comparison.')
(p/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({**result,'rows':[{k:v for k,v in row.items() if not k.startswith('step_')} for row in rows]},indent=2))
