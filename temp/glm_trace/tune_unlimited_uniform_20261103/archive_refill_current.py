from pathlib import Path
import ast,hashlib,json,statistics,subprocess
p=Path(__file__).resolve().parent;root=p.parents[2]
expected={'mla_host_batch.cuh':'81093ba5ab33cc5d84b2df04c702975697387b4054ebe8f1991f558f1bd6cdd2','mla.cuh':'34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3','far_link_emu.cuh':'8507eab01aa705c507d0ee389f1216dfc3c29d50844fbf509a0339fd6023d716'}
production={x:hashlib.sha256((root/'include/flashinfer/attention'/x).read_bytes()).hexdigest() for x in expected};assert production==expected
source=json.loads((p/'refill_current_source.json').read_text());assert source['sha256']==hashlib.sha256((p/'refill_current.cuh').read_bytes()).hexdigest();assert source['base_sha256']==hashlib.sha256((p/'clean.cuh').read_bytes()).hexdigest()
results={}
for name in ['refill_current','clean_refill_control','refill_current_reverse']:
 r=json.loads((p/(name+'_native_unlimited_f24.json')).read_text());assert r['complete'] and r['config']['staged'] and r['cache_payload_slots']==524288
 for z in r['libraries']:assert hashlib.sha256((root/z['path']).read_bytes()).hexdigest()==z['sha256'],(name,z)
 results[name]=dict(pass_means_us=[statistics.mean(x['us']) for x in r['passes']],mean_us=statistics.mean(v for x in r['passes'] for v in x['us']),mean_unique=statistics.mean(v for x in r['passes'] for v in x['unique_misses']),header_sha256=r['header_sha256'],actual_library_sha256=next(z['sha256'] for z in r['libraries'] if '_host_overlap_v2' in z['path']))
assert results['refill_current']['header_sha256']==results['refill_current_reverse']['header_sha256']==source['sha256']
assert results['clean_refill_control']['header_sha256']==source['base_sha256']
probe=json.loads((p/'dispatch_probe_refill_current.json').read_text());assert probe['header_sha256']==source['sha256'] and probe['kernel_nodes']==1
launch=probe['launches'][0];assert launch['grid']==[1,108,1] and launch['block']==[32,4,2] and launch['dynamic_shared_bytes']==163072 and launch['active_blocks_per_SM']==1 and 'BatchMLACleanServiceKernel' in launch['kernel_name']
barriers=json.loads((p/'refill_barrier_codegen.json').read_text());assert barriers['clean']['barrier_ops']['BAR.SYNC']==91 and barriers['refill_current']['barrier_ops']['BAR.SYNC']==86
assert barriers['refill_current']['library_sha256']==results['refill_current']['actual_library_sha256']
assert '0 hazards displayed (0 errors, 0 warnings)' in (p/'racecheck_refill_current.log').read_text()
audit=dict(results=results,normal_mean_gain_percent=(1-results['refill_current']['mean_us']/results['clean_refill_control']['mean_us'])*100,reverse_firstpass_delta_us=results['refill_current_reverse']['mean_us']-results['clean_refill_control']['pass_means_us'][0],dispatch=launch,barriers=barriers,focused_tests=2,target_racecheck=dict(scope='K65 one target kernel launch',errors=0,warnings=0),production=production,retained=False,new100GB_result=False)
(p/'refill_current_audit.json').write_text(json.dumps(audit,indent=2)+chr(10))
files=[root/x for x in subprocess.check_output(['git','ls-files','--others','--exclude-standard',str(p.relative_to(root))],cwd=root,text=True).splitlines()]
allowed=('REFILL_CURRENT_','archive_refill_current','bench_clean_refill','bench_refill_current','clean_refill','codegen_refill','compile_refill','correctness_refill','dispatch_probe_refill','dispatch_probes_refill','dispatch_refill','make_refill','probe_refill','racecheck_refill','refill_barrier','refill_current','run_clean_refill','run_refill','test_refill')
assert all(f.relative_to(p).parts[0].startswith(allowed) for f in files),[str(f) for f in files if not f.relative_to(p).parts[0].startswith(allowed)]
for f in files:
 if f.suffix=='.py':ast.parse(f.read_text())
 if f.suffix=='.json':json.loads(f.read_text())
manifest=p/'refill_current_manifest.json';manifest.write_text(json.dumps(dict(files=[dict(path=str(f.relative_to(root)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size) for f in files],production=production),indent=2)+chr(10))
subprocess.run(['git','add','--',*[str(f.relative_to(root)) for f in files],str(manifest.relative_to(root)),'docs/HANDOFF.md'],cwd=root,check=True)
print('verified_files',len(files),'mean_gain_percent',audit['normal_mean_gain_percent'],'reverse_delta_us',audit['reverse_firstpass_delta_us'])
