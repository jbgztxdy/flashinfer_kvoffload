from pathlib import Path
import ast,hashlib,json,statistics,subprocess
p=Path(__file__).resolve().parent;root=p.parents[2]
expected={'mla_host_batch.cuh':'81093ba5ab33cc5d84b2df04c702975697387b4054ebe8f1991f558f1bd6cdd2','mla.cuh':'34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3','far_link_emu.cuh':'8507eab01aa705c507d0ee389f1216dfc3c29d50844fbf509a0339fd6023d716'}
production={x:hashlib.sha256((root/'include/flashinfer/attention'/x).read_bytes()).hexdigest() for x in expected};assert production==expected
source=json.loads((p/'service22_source.json').read_text());assert source['sha256']==hashlib.sha256((p/'service22.cuh').read_bytes()).hexdigest();assert source['base_sha256']==hashlib.sha256((p/'clean.cuh').read_bytes()).hexdigest()
names=['service22','clean_service22_control','service22_reverse','production_service22_control','production_staged_service22_control'];results={}
for name in names:
    r=json.loads((p/(name+'_native_unlimited_f24.json')).read_text());assert r['complete']
    for z in r['libraries']:assert hashlib.sha256((root/z['path']).read_bytes()).hexdigest()==z['sha256'],(name,z)
    results[name]=dict(mean_us=statistics.mean(v for x in r['passes'] for v in x['us']),pass_means_us=[statistics.mean(x['us']) for x in r['passes']],mean_unique=statistics.mean(v for x in r['passes'] for v in x['unique_misses']),staged=r['config']['staged'],header_sha256=r['header_sha256'],actual_library_sha256=next(z['sha256'] for z in r['libraries'] if '_host_overlap_v2' in z['path']))
assert results['service22']['staged'] and results['clean_service22_control']['staged'] and results['service22_reverse']['staged'] and results['production_staged_service22_control']['staged']
assert not results['production_service22_control']['staged']
assert results['service22']['header_sha256']==source['sha256'];assert results['clean_service22_control']['header_sha256']==source['base_sha256']
assert results['production_service22_control']['header_sha256']==production['mla_host_batch.cuh']==results['production_staged_service22_control']['header_sha256']
probe=json.loads((p/'dispatch_probe_service22.json').read_text());assert probe['header_sha256']==source['sha256'] and probe['kernel_nodes']==1
launch=probe['launches'][0];assert launch['grid']==[1,108,1] and launch['block']==[32,4,2] and launch['dynamic_shared_bytes']==165376 and launch['active_blocks_per_SM']==1 and 'BatchMLACleanServiceKernel' in launch['kernel_name']
codegen=json.loads((p/'service22_codegen.json').read_text())['service22'];assert codegen['sha256']==results['service22']['actual_library_sha256'];assert 'REG:255 STACK:16 SHARED:1152' in codegen['resources']
raw=json.loads((p/'service22_native_unlimited_f24.json').read_text());packets={str(rows):statistics.mean((v+rows-1)//rows for x in raw['passes'] for v in x['unique_misses']) for rows in [20,22]}
audit=dict(results=results,mean_packets_per_step=packets,packet_reduction_percent=(1-packets['22']/packets['20'])*100,dispatch=launch,probe_max_abs_error=probe['max_abs_error'],focused_tests=2,cache_payload_slots=raw['cache_payload_slots'],production=production,retained=False,new100GB_result=False,invalid_first_production_control='host_stage=False hardcoded despite --route staged; raw retained and excluded from staged comparisons')
(p/'service22_audit.json').write_text(json.dumps(audit,indent=2)+chr(10))
files=[root/x for x in subprocess.check_output(['git','ls-files','--others','--exclude-standard',str(p.relative_to(root))],cwd=root,text=True).splitlines()]
allowed=('SERVICE22_','archive_service22','bench_clean_service22','bench_production_service22','bench_production_staged_service22','bench_service22','clean_service22','codegen_service22','compile_service22','correctness_service22','dispatch_probe_service22','dispatch_probes_service22','dispatch_service22','make_service22','probe_service22','production_service22','production_staged_service22','run_clean_service22','run_production_service22','run_production_staged_service22','run_service22','service22','test_service22')
assert all(f.relative_to(p).parts[0].startswith(allowed) for f in files),[str(f) for f in files if not f.relative_to(p).parts[0].startswith(allowed)]
for f in files:
    if f.suffix=='.py':ast.parse(f.read_text())
    if f.suffix=='.json':json.loads(f.read_text())
manifest=p/'service22_manifest.json';manifest.write_text(json.dumps(dict(files=[dict(path=str(f.relative_to(root)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size) for f in files],production=production),indent=2)+chr(10))
subprocess.run(['git','add','--',*[str(f.relative_to(root)) for f in files],str(manifest.relative_to(root)),'docs/HANDOFF.md'],cwd=root,check=True)
print('verified_raw',len(names),'new_files',len(files),'means',[(n,round(results[n]['mean_us'],3)) for n in names])
