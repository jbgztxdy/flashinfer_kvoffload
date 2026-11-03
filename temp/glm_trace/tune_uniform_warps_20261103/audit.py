"""Audit full uniform-warp negative results, same-input bytes, and calibration limits."""
from pathlib import Path
import hashlib,json,statistics as S,time
p=Path(__file__).resolve().parent;root=p.parents[2]
load=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
m=load(p/'manifest.json');f=load(p.parent/'rebench_100gb_v2_20261103/audit.json');compute=f['rows'][0]['step_device_us']
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh')==m['production_sha256']
assert sha(root/'include/flashinfer/attention/mla.cuh')==m['math_sha256']
assert sha(root/'include/flashinfer/attention/far_link_emu.cuh')==m['far_sha256']
assert sha(p/'candidate.cuh')==m['candidate_sha256']
rows=[];raws={};launches=0
for path in sorted(p.glob('*_f24.json')):
 d=load(path);assert d['complete'] and d['resources_start']['foreign']==d['resources_end']['foreign']==[]
 cfg=d['config'];assert cfg['requests']==128 and cfg['mtp']==8 and cfg['steps']==32 and cfg['positions']==256 and cfg['capacity']==4096
 assert 'membind: 1' in d['numa_policy'] and d['cache_payload_slots']==524288
 assert d['far_sha256']==m['far_sha256'] and d['header_sha256']==m['candidate_sha256' if d['variant']=='candidate' else 'production_sha256']
 for lib in d['libraries']:assert sha(root/lib['path'])==lib['sha256']
 harness='bench_unlimited.py' if 'unlimited' in path.name else 'bench.py';assert d['harness_sha256']==sha(p/harness)
 miss=[S.mean(q['unique_misses'][i] for q in d['passes']) for i in range(32)]
 for q in d['passes']:
  assert len(q['us'])==len(q['unique_misses'])==32
  if d['variant']=='candidate':
   assert all(x[0]==x[1]==(u+9)//10 and x[3]==u for x,u in zip(q['assist_stats'],q['unique_misses']))
  launches+=32
 stats=[x for q in d['passes'] for x in q['assist_stats']];times=[x for q in d['passes'] for x in q['us']]
 r=dict(raw=path.name,variant=d['variant'],setting=d['setting'],mean_us=S.mean(times),pass_us=[S.mean(q['us']) for q in d['passes']],unique_miss_MB=S.mean(miss)*1152/1e6,mean_chunks=S.mean(x[0] for x in stats),mean_wait_progress_loops=S.mean(x[2] for x in stats),mean_fetched_pages=S.mean(x[3] for x in stats),harness=harness,sha256=sha(path))
 if d['setting']['bandwidth_GBps']==100:
  r['reference_us']=S.mean(max(c,u*1152/100000) for c,u in zip(compute,miss));r['over_reference']=r['mean_us']/r['reference_us']-1
 else:r['classification']='Diagnostic only unlimited transport; not formal100GB reference orphysicalNVLink'
 rows.append(r);raws[(d['variant'],d['setting']['name'])]=d
comparisons=[]
for r in rows:
 if r['variant']!='candidate':continue
 b=next(x for x in rows if x['variant']=='base' and x['setting']==r['setting'])
 assert raws[('candidate',r['setting']['name'])]['bindings']==raws[('base',r['setting']['name'])]['bindings']
 comparisons.append(dict(setting=r['setting']['name'],base_us=b['mean_us'],candidate_us=r['mean_us'],slowdown=r['mean_us']/b['mean_us']-1,decision='reject current full candidate; not installed'))
transfer=[]
for filename in ['transfer_result.json','transfer_modes_result.json']:
 d=load(p/filename);assert d['complete'] and d['payload_exact'] and d['far_sha256']==m['far_sha256']
 for cm in sorted({x.get('config_mode',2) for x in d['cases']}):
  for lat in [0,20000]:
   for n in [64,10]:
    samples=[x['us'] for x in d['cases'] if x.get('config_mode',2)==cm and x['latency_ns']==lat and x['rows_per_packet']==n];assert len(samples)==3
    mean=S.mean(samples);transfer.append(dict(raw=filename,mode=cm,mode_name={0:'hooksdisabled',1:'enabledunlimited',2:'enabled100GB'}[cm],latency_ns=lat,rows=n,CTAs=24 if n==64 else 108,mean_us=mean,GBps=d['bytes']/(mean*1000),samples_us=samples))
probe=load(p.parent/'research_sm80_pipeline_20261103/result.json');assert probe['complete'] and probe['sampled_QK_exact'] and probe['all_payload_exact']
probe_rows=[]
for lat in [0,20000]:
 for ctas in [24,108]:
  for mode in [0,1,2]:
   ts=[x['us'] for x in probe['cases'] if x['latency_ns']==lat and x['ctas']==ctas and x['mode']==mode];assert len(ts)==3
   probe_rows.append(dict(latency_ns=lat,CTAs=ctas,mode=mode,mean_us=S.mean(ts),GBps=probe['bytes']/(S.mean(ts)*1000)))
r=dict(complete=True,timestamp=time.strftime('%FT%T%z'),rows=rows,comparisons=comparisons,transfer_diagnostics=transfer,samewarp_probe=probe_rows,timed_full_launches=launches,timed_samewarp_probe_launches=len(probe['cases']),timed_transfer_launches=48,
 focused_checks=2,CPU_ownership_cases=52,source_audit=load(p/'source_audit.json'),production_unchanged=True,canonical_math_unchanged=True,emulator_unchanged=True,formal_reference_unchanged=True,
 decision='Reject current uniform10page implementation, do not install. Samewarp concept remains unproven endtoend; new packetization/lifetime mechanism needed, no trivialcadence/rowcount sweep.',
 limits='BF16 GPU3samecardHBM NUMA1,32step prefix only. TargetactualREG255/STACK48/static1152/dynamic163072. No nativephysical/FP16/normalbuild/sanitizer/full1024/H800/newGraphnodeprobe.100GBv2aggregate calibration does not extend to108CTA10rowpacketization; it achieves31-34GBps even withoutattention. Enabledunlimited diagnostic cannotreplace100GB orbe claimed asproduction gain.')
assert launches==448
(p/'audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['source_audit','samewarp_probe','transfer_diagnostics']},indent=2))
