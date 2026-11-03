from pathlib import Path
import ast,hashlib,json,statistics,subprocess
p=Path(__file__).resolve().parent;root=p.parents[2]
expected={'mla_host_batch.cuh':'81093ba5ab33cc5d84b2df04c702975697387b4054ebe8f1991f558f1bd6cdd2','mla.cuh':'34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3','far_link_emu.cuh':'8507eab01aa705c507d0ee389f1216dfc3c29d50844fbf509a0339fd6023d716'}
production={x:hashlib.sha256((root/'include/flashinfer/attention'/x).read_bytes()).hexdigest() for x in expected};assert production==expected
source=json.loads((p/'stage_tile_uniform_source.json').read_text());assert source['sha256']==hashlib.sha256((p/'stage_tile_uniform.cuh').read_bytes()).hexdigest();assert source['base_sha256']==production['mla_host_batch.cuh']
results={}
for name in ['stage_tile_uniform','production_stage_tile_control']:
 r=json.loads((p/(name+'_native_unlimited_f24.json')).read_text());assert r['complete'] and r['config']['staged'] and r['cache_payload_slots']==524288
 for z in r['libraries']:assert hashlib.sha256((root/z['path']).read_bytes()).hexdigest()==z['sha256'],(name,z)
 results[name]=dict(pass_means_us=[statistics.mean(x['us']) for x in r['passes']],mean_us=statistics.mean(v for x in r['passes'] for v in x['us']),mean_unique=statistics.mean(v for x in r['passes'] for v in x['unique_misses']),header_sha256=r['header_sha256'],actual_library_sha256=next(z['sha256'] for z in r['libraries'] if '_host_overlap_v2' in z['path']))
assert results['stage_tile_uniform']['header_sha256']==source['sha256'] and results['production_stage_tile_control']['header_sha256']==production['mla_host_batch.cuh']
probe=json.loads((p/'dispatch_probe_stage_tile_uniform.json').read_text());assert probe['header_sha256']==source['sha256'] and probe['kernel_nodes']==1
launch=probe['launches'][0];assert launch['grid']==[1,108,1] and launch['block']==[32,4,2] and launch['dynamic_shared_bytes']==147968 and launch['active_blocks_per_SM']==1 and 'BatchMLAPVFeatureKernel' in launch['kernel_name']
codegen=json.loads((p/'stage_tile_uniform_codegen.json').read_text());assert codegen['stage_tile_uniform']['sha256']==results['stage_tile_uniform']['actual_library_sha256'];assert codegen['retained']['sha256']==results['production_stage_tile_control']['actual_library_sha256']
model=json.loads((p/'stage_tile_uniform_model.json').read_text());assert model['tile_cases']>=800000 and model['selected_staged_only_tiles']>0
audit=dict(results=results,candidate_slowdown_percent=(results['stage_tile_uniform']['mean_us']/results['production_stage_tile_control']['mean_us']-1)*100,dispatch=launch,focused_tests=2,cpu_tile_cases=model['tile_cases'],codegen={k:dict(instructions=v['instructions'],resources=v['resources']) for k,v in codegen.items()},production=production,retained=False,new100GB_result=False)
(p/'stage_tile_uniform_audit.json').write_text(json.dumps(audit,indent=2)+chr(10))
files=[root/x for x in subprocess.check_output(['git','ls-files','--others','--exclude-standard',str(p.relative_to(root))],cwd=root,text=True).splitlines()]
allowed=('STAGE_TILE_','archive_stage_tile','bench_production_stage_tile','bench_stage_tile','codegen_stage_tile','compile_stage_tile','correctness_stage_tile','dispatch_probe_stage_tile','dispatch_probes_stage_tile','dispatch_stage_tile','make_stage_tile','model_stage_tile','probe_stage_tile','production_stage_tile','run_production_stage_tile','run_stage_tile','stage_tile','test_stage_tile')
assert all(f.relative_to(p).parts[0].startswith(allowed) for f in files),[str(f) for f in files if not f.relative_to(p).parts[0].startswith(allowed)]
for f in files:
 if f.suffix=='.py':ast.parse(f.read_text())
 if f.suffix=='.json':json.loads(f.read_text())
manifest=p/'stage_tile_uniform_manifest.json';manifest.write_text(json.dumps(dict(files=[dict(path=str(f.relative_to(root)),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size) for f in files],production=production),indent=2)+chr(10))
subprocess.run(['git','add','--',*[str(f.relative_to(root)) for f in files],str(manifest.relative_to(root)),'docs/HANDOFF.md'],cwd=root,check=True)
print('files',len(files),'slowdown_pct',audit['candidate_slowdown_percent'])
