"""Audit actual normal-build artifacts; preserve every positive/negative measurement."""
from pathlib import Path
import hashlib,json,re,statistics,time
p=Path(__file__).resolve().parent;root=p.parents[2]
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
rows={}
for f in sorted(p.glob('*native_unlimited_f24.json')):
 r=json.loads(f.read_text());assert r['complete'],f
 assert r['config']['normal_build'] and not r['config']['simulator_hooks'],f
 assert r['resources_start']['gpus'][0]['index']==r['resources_end']['gpus'][0]['index']==3
 assert not r['resources_start']['foreign'] and not r['resources_end']['foreign']
 v=r['variant'];assert r['header_sha256']==sha(p/(v+'.cuh'))
 assert r['far_sha256']==sha(root/'include/flashinfer/attention/far_link_emu.cuh')
 libs=r['libraries'];assert any('host_overlap_v2.so' in x['path'] for x in libs)
 for lib in libs:assert sha(root/lib['path'])==lib['sha256'],lib
 flags=[]
 for lib in libs:
  ninja=(root/lib['path']).parent/'build.ninja'
  if ninja.exists():
   text=ninja.read_text();assert '-DFLASHINFER_FAR_EMU' not in text
   flags.append(dict(path=str(ninja.relative_to(root)),sha256=sha(ninja)))
 means=[statistics.mean(q['us']) for q in r['passes']]
 misses=[n for q in r['passes'] for n in q['unique_misses']]
 rows[f.stem]=dict(variant=v,raw=f.name,raw_sha256=sha(f),header_sha256=r['header_sha256'],mean_us=statistics.mean(means),pass_us=means,launches=len(misses),mean_unique_misses=statistics.mean(misses),mean_required_far_MB=statistics.mean(misses)*1152/1e6,staged=r['config']['staged'],diagnostic_only=r['config'].get('diagnostic_only',False),actual_far_MB=statistics.mean([n for q in r['passes'] for n in q.get('actual_far_read_bytes',[m*1152 for m in q['unique_misses']])])/1e6,normal_build=True,simulator_hooks=False,far_fraction=r['config'].get('far_fraction',1.),actual_libraries=libs,compile_flags=flags)
 if v!='base':
  rows[f.stem]['mean_wait_iterations']=statistics.mean(s[2] for q in r['passes'] for s in q['assist_stats'])
  rows[f.stem]['mean_diag5']= statistics.mean(s[5] for q in r['passes'] for s in q['assist_stats'])
comparisons=[]
for key,r in rows.items():
 if key.startswith(('fresh_','mathonly_','confirm_','direct_confirm_')) or r['variant']=='base' or r['diagnostic_only']:continue
 b=rows['base_native_unlimited_f24'];comparisons.append(dict(base=b['raw'],candidate=r['raw'],base_us=b['mean_us'],candidate_us=r['mean_us'],candidate_slower_pct=(r['mean_us']/b['mean_us']-1)*100,mean_far_bytes_change_pct=(r['mean_required_far_MB']/b['mean_required_far_MB']-1)*100))
if 'confirm_clean_native_unlimited_f24' in rows:
 b=rows['confirm_base_native_unlimited_f24'];c=rows['confirm_clean_native_unlimited_f24'];comparisons.append(dict(base=b['raw'],candidate=c['raw'],order='candidate then base',base_us=b['mean_us'],candidate_us=c['mean_us'],candidate_slower_pct=(c['mean_us']/b['mean_us']-1)*100))
if 'direct_confirm_direct_pv_native_unlimited_f24' in rows:
 b=rows['direct_confirm_base_native_unlimited_f24'];c=rows['direct_confirm_direct_pv_native_unlimited_f24'];comparisons.append(dict(base=b['raw'],candidate=c['raw'],order='new direct then old staged',base_us=b['mean_us'],candidate_us=c['mean_us'],candidate_slower_pct=(c['mean_us']/b['mean_us']-1)*100))
if 'fresh_rowcopy_native_unlimited_f24' in rows:
 b=rows['fresh_base_native_unlimited_f24'];c=rows['fresh_rowcopy_native_unlimited_f24'];comparisons.append(dict(base=b['raw'],candidate=c['raw'],order='candidate then base',base_us=b['mean_us'],candidate_us=c['mean_us'],candidate_slower_pct=(c['mean_us']/b['mean_us']-1)*100))
resources={}
for f in sorted(p.glob('*resources*.txt')):
 ls=f.read_text().splitlines()
 for i,line in enumerate(ls):
  if any(w in line for w in ['ServiceKernel','DirectPVKernel','TrycachePVKernel','NativeDiagPVKernel']):resources[f.name]=dict(symbol=line.strip(),actual=ls[i+1].strip(),dynamic_bytes=140160 if 'regstream' in f.name else 147968 if any(w in f.name for w in ['direct_pv','trycache']) else 163072,sha256=sha(f))
logs={f.name:dict(sha256=sha(f),last_lines=f.read_text().splitlines()[-3:]) for f in p.glob('correctness_*.log')}
record=dict(required_byte_basis='Staged: actual unique miss*1152; directPV/trycache: actual_far_read_bytes recorded separately. Bytes never replaced with stagedunique reference.',dispatch_probes={f.name:json.loads(f.read_text()) for f in p.glob('dispatch_probe_*.json')},native_timeline=json.loads((p/'native_timeline_analysis.json').read_text()),timestamp=time.strftime('%FT%T%z'),scope='GPU3 NUMA1 normal-build same-device far HBM. Not NVLink, no rate/latency hooks, no100GB or physical claim.',rows=rows,comparisons=comparisons,resources=resources,correctness_logs=logs,timed_launches=sum(r['launches'] for r in rows.values()),production_header_sha256=sha(root/'include/flashinfer/attention/mla_host_batch.cuh'),math_header_sha256=sha(root/'include/flashinfer/attention/mla.cuh'),far_header_sha256=sha(root/'include/flashinfer/attention/far_link_emu.cuh'),protocol=json.loads((p/'protocol.json').read_text()))
(p/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
for c in comparisons:print(c['candidate'],round(c['candidate_us'],3),round(c['candidate_slower_pct'],3))
print('TIMED_LAUNCHES',record['timed_launches'])
