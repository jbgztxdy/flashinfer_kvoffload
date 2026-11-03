from pathlib import Path
import ast,hashlib,json,subprocess,statistics
p=Path(__file__).resolve().parent;root=p.parents[2]
raw_names=['claimrow_timeline_native_unlimited_f23','claimrow_timeline_native_unlimited_f24','claimrow_timeline_workload_native_unlimited_f24','claimrow_timeline_packet_weights_native_unlimited_f24','claimrow_interleave_native_unlimited_f24','claimrow_interleave_sorted_native_unlimited_f24','claimrow_interleave_timeline_native_unlimited_f24']
actual=[]
for name in raw_names:
    f=p/(name+'.json');r=json.loads(f.read_text());assert r['complete']
    assert r['header_sha256']==hashlib.sha256((p/(r['variant']+'.cuh')).read_bytes()).hexdigest(),f
    for lib in r['libraries']:
        assert hashlib.sha256((root/lib['path']).read_bytes()).hexdigest()==lib['sha256'],(f,lib)
    actual.append(dict(raw=name+'.json',mean_us=statistics.mean(v for x in r['passes'] for v in x['us']),library_sha256=next(z['sha256'] for z in r['libraries'] if '_host_overlap_v2' in z['path'])))
assert actual[0]['library_sha256']==actual[1]['library_sha256']==actual[2]['library_sha256']==actual[3]['library_sha256']
for name in ['claimrow_timeline','claimrow_interleave','claimrow_interleave_sorted','claimrow_interleave_timeline']:
    m=json.loads((p/(name+'_source.json')).read_text())
    assert m['sha256']==hashlib.sha256((p/(name+'.cuh')).read_bytes()).hexdigest(),name
assert '0 hazards displayed (0 errors, 0 warnings)' in (p/'racecheck_claimrow_interleave_sorted.log').read_text()
prod={x:hashlib.sha256((root/'include/flashinfer/attention'/x).read_bytes()).hexdigest() for x in ['mla_host_batch.cuh','mla.cuh','far_link_emu.cuh']}
assert prod=={'mla_host_batch.cuh':'81093ba5ab33cc5d84b2df04c702975697387b4054ebe8f1991f558f1bd6cdd2','mla.cuh':'34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3','far_link_emu.cuh':'8507eab01aa705c507d0ee389f1216dfc3c29d50844fbf509a0339fd6023d716'}
audit=dict(raw_actual_library_verified=actual,source_header_verified=True,sorted_target_racecheck_errors=0,sorted_target_racecheck_warnings=0,production=prod,retained=False,new100GB_result=False,scope='bounded normal GPU3 same-device HBM and CPU counterfactual only')
(p/'claimrow_order_audit.json').write_text(json.dumps(audit,indent=2)+chr(10))
paths=set(subprocess.check_output(['git','ls-files','--others','--exclude-standard',str(p.relative_to(root))],cwd=root,text=True).splitlines())
paths.update(subprocess.check_output(['git','diff','--cached','--name-only','--',str(p.relative_to(root))],cwd=root,text=True).splitlines())
files=sorted(root/x for x in paths if x and not x.endswith('/claimrow_order_manifest.json'))
allowed=('CLAIMROW_','analyze_','archive_interleave_','bench_claimrow_','claimrow_','codegen_claimrow_','compile_claimrow_','correctness_claimrow_','fence_claimrow_','make_claimrow_','make_interleave_','model_claimrow_','racecheck_claimrow_','run_claimrow_','test_claimrow_','verify_interleave_')
assert all(f.name.startswith(allowed) for f in files),[f.name for f in files if not f.name.startswith(allowed)]
for f in files:
    if f.suffix=='.py':ast.parse(f.read_text())
    if f.suffix=='.json':json.loads(f.read_text())
manifest=p/'claimrow_order_manifest.json'
manifest.write_text(json.dumps(dict(files=[dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files],production=prod),indent=2)+chr(10))
subprocess.run(['git','add','--',*[str(f.relative_to(root)) for f in files],str(manifest.relative_to(root)),'docs/HANDOFF.md'],cwd=root,check=True)
print('verified_raw',len(actual),'new_files',len(files),'candidate_means',[round(x['mean_us'],3) for x in actual[4:]])
