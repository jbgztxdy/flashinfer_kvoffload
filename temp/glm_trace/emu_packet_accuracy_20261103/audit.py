"""Derive accuracy/performance separately, preserve every failed or negative trial."""
from pathlib import Path
import hashlib,json,statistics,time
p=Path(__file__).resolve().parent;root=p.parents[2]
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
manifest=json.loads((p/'manifest.json').read_text())
assert sha(root/'include/flashinfer/attention/far_link_emu.cuh')==manifest['base_far_sha256']
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh')==manifest['production_host_sha256']
calibration={}
for name in ['backoff','batch','calibrate','calibrate_cta','calibrate_scalar','calibrate_sparse']:
 f=p/(name+'_result.json')
 if not f.exists():continue
 r=json.loads(f.read_text());assert r['complete'] and r['payload_exact'];groups={}
 for c in r['cases']:
  k=(c['config_mode'],c['latency_ns'],c['rows_per_packet'],c['CTAs']);groups.setdefault(k,[]).append(c['us'])
 calibration[name]=[dict(config_mode=k[0],latency_ns=k[1],packet_rows=k[2],CTAs=k[3],mean_us=statistics.mean(v),actual_GBps=r.get('actual_far_bytes',r['bytes'])/statistics.mean(v)/1000,samples_us=v) for k,v in groups.items()]
semantic=json.loads((p/'semantic_result.json').read_text());assert all(c['bad']==c['early']==0 for c in semantic['cases'])
request_audits={}
for name in ['batch_verify','batch_verify50']:
 f=p/(name+'_result.json')
 if not f.exists():continue
 r=json.loads(f.read_text());assert r['complete'] and r['payload_exact'];assert all(c['bad']==c['early']==0 for c in r['cases']);request_audits[name]=r['cases']
reference=root/'temp/glm_trace/tune_scatter_20261031/gpu3/raw/reference_base_staged_f24_i64_soff_floor_floor_mtp8_b128.json';ref=json.loads(reference.read_text());compute=[statistics.mean(z) for z in ref['cases'][0]['timings']['local']['per_step_samples_us'][:32]]
full={}
for f in sorted(p.glob('*_f24.json')):
 r=json.loads(f.read_text());assert r['complete'],f
 assert not r['resources_start']['foreign'] and not r['resources_end']['foreign']
 v=r['variant'];include=p/('include_'+v);assert r['header_sha256']==sha(include/'flashinfer/attention/mla_host_batch.cuh');assert r['far_sha256']==sha(include/'flashinfer/attention/far_link_emu.cuh')
 for lib in r['libraries']:assert sha(root/lib['path'])==lib['sha256']
 direct=r['config']['staged']==False;timing=[t for q in r['passes'] for t in q['us']];bytes_=[q.get('actual_far_read_bytes',[n*1152 for n in q['unique_misses']]) for q in r['passes']]
 row=dict(variant=v,raw_sha256=sha(f),mean_us=statistics.mean(timing),pass_us=[statistics.mean(q['us']) for q in r['passes']],timed_launches=len(timing),actual_far_MB=statistics.mean(n for q in bytes_ for n in q)/1e6,byte_basis='actual direct references' if direct else 'actual unique misses',header_sha256=r['header_sha256'],far_sha256=r['far_sha256'],libraries=r['libraries'])
 if 'g100_' in f.name:
  ideal=[max(compute[t],n/100000) for q in bytes_ for t,n in enumerate(q)];row['stepwise_reference_us']=statistics.mean(ideal);row['ratio_to_reference']=row['mean_us']/statistics.mean(ideal);row['classification']='Prototype fullkernel, not accepted physical100GB equivalence. Calibration envelope and hookoverhead must be considered.'
 else:row['classification']='Samebinary runtimehook control only. Name native_hbm meanshooksdisabled, g0 meansenabledunlimited.'
 full[f.stem]=row
pairs=[]
for setting in ['g100_l0','g100_l20000']:
 b=full['base_'+setting+'_f24'];c=full['staged_'+setting+'_f24'];pairs.append(dict(setting=setting,old_v2_us=b['mean_us'],batch_us=c['mean_us'],slower_pct=(c['mean_us']/b['mean_us']-1)*100,old_over_reference_pct=(b['ratio_to_reference']-1)*100,batch_over_reference_pct=(c['ratio_to_reference']-1)*100))
source_files={str(f.relative_to(p)):sha(f) for f in sorted(p.rglob('*')) if f.is_file() and not any(x.startswith(('include','jit','__pycache__')) for x in f.relative_to(p).parts) and f.suffix in ['.cuh','.cu','.py']}
r=dict(timestamp=time.strftime('%FT%T%z'),production_modified=False,calibration=calibration,semantic=dict(cases=len(semantic['cases']),requests=sum(c['requests'] for c in semantic['cases']),bad=0,early=0,raw_sha256=sha(p/'semantic_result.json')),request_audits=request_audits,fullkernels=full,old_vs_batch_staged=pairs,reference=dict(path=str(reference.relative_to(root)),sha256=sha(reference),scope='Original sameGPU device implementation first32steps, no newreferencebenchmark'),source_files=source_files,timed_fullkernel_launches=sum(x['timed_launches'] for x in full.values()),focused_checks=sum('passed' in f.read_text() for f in p.glob('correctness_*.log'))*2,limitations=['Not physicalNVLink model','Dense copy calibration insufficient for sparsefewfarrows/stage','Softwarehook+compilerresource overhead remains','No futurebytesprefetch or math/cacheomission','No productioninstall/newformalreference/fullmatrix/sanitizers/physicalGPUjobs'])
(p/'audit.json').write_text(json.dumps(r,indent=2)+'\n');print('full_launches',r['timed_fullkernel_launches'],'checks',r['focused_checks']);print(json.dumps(pairs,indent=2))
