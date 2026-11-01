"""Verify frozen sources, original code identity, actual dispatch and measured rows."""
import hashlib,json,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
read=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
m=read(p/'manifest.json');v='qkpv_roles'
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh')=='7193255e2fdf83d3baa9347dea1a02b892c29ed746da719818e30fdf4066d469'
assert sha(root/'include/flashinfer/attention/mla.cuh')==m['math_sha256']=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
assert sha(root/'include/flashinfer/attention/far_link_emu.cuh')=='5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
assert sha(root/'temp/glm_trace/kernel_report_20261029/matrix.json')=='e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
for variant,h in [('base',m['base_sha256']),(v,m['candidate_sha256'])]:
    assert sha(p/(variant+'.cuh'))==h
    assert sha(p/(variant+'_mla.cuh'))==m['math_sha256']
    d=p/('include_'+variant)/'flashinfer/attention'
    if d.exists():assert sha(d/'mla_host_batch.cuh')==h and sha(d/'mla.cuh')==m['math_sha256']
proofs=[]
for suffix,count in [('',38),('_normal',76)]:
    f=p/('compiler_'+v+suffix+'_summary.json');d=read(f)
    assert d['complete'] and d['unchanged_original_kernels']==count and d['all_original_instruction_and_control_words_identical']
    assert d['candidate_header_sha256']==m['candidate_sha256'] and d['math_header_sha256']==m['math_sha256']
    for variant,libs in d['libraries'].items():
        directory=p.parent/'tune_cached_pe_20261101'/('jit_cached_pe'+suffix) if variant=='base' else p/('jit_'+variant+suffix)
        for lib in libs:
            files=list(directory.rglob(lib['name']));assert len(files)==1 and sha(files[0])==lib['sha256']
    proofs.append(dict(path=f.name,sha256=sha(f),originals=count,new_resources=[x['resource'] for x in d['new_kernels']]))
probe=read(p/('dispatch_probe_'+v+'.json'))
assert probe['header_sha256']==m['candidate_sha256'] and probe['kernel_nodes']==1
assert probe['resources_start']['foreign']==probe['resources_end']['foreign']==[]
launch=probe['launches'][0]
assert launch['grid']==[1,108,1] and launch['block']==[32,4,3]
assert launch['dynamic_shared_bytes']==147968 and launch['active_blocks_per_SM']==1
sanitizers=[]
for kind in ['synccheck','racecheck','memcheck']:
    f=p/(kind+'_'+v+'.log');text=f.read_text()
    assert 'ERROR SUMMARY: 0 errors' in text or (kind=='racecheck' and 'RACECHECK SUMMARY: 0 hazards' in text)
    sanitizers.append(dict(tool=kind,log=f.name,sha256=sha(f),scope='first matched BF16 K97 QKPV kernel; memcheck API-error reporting disabled, device checks enabled'))
a=read(p/'performance_audit.json');state=read(p/'validation_state.json')
assert state['complete'] and state['phase']=='short_screen_complete' and all(x['exit_code']==0 for x in state['steps'])
assert len(a['rows'])==6 and len(a['pairs'])==3 and not a['incomplete'] and not a['pending_references']
assert a['timed_launches']==1152 and a['accepted_segments']==72
assert sum(x['tests'] for x in a['tests']+a['stress_tests'])==111
pre=read(p/'precompile_review.json');assert sha(p/'precompile_review/qkpv_roles.cuh')==pre['old_header_sha256']
record=dict(complete=True,timestamp=time.strftime('%FT%T%z'),goal_complete=False,candidate_adopted=False,
 headers=dict(base=m['base_sha256'],candidate=m['candidate_sha256'],math=m['math_sha256']),proofs=proofs,
 dispatch=dict(sha256=sha(p/('dispatch_probe_'+v+'.json')),launch=launch,captures=len(list((p/('dispatch_probes_'+v)).glob('*.json')))),
 sanitizers=sanitizers,rows=6,pairs=3,timed_launches=1152,accepted_segments=72,numerical_tests=111,
 precompile_barrier_review_preserved=True,setup_dependency_path_failure_preserved=True,production_unchanged=True,
 limitation='Fixed short GPU3 A100-PCIe/NUMA1 screen256positions/3coldpasses. No fulltrace/nativeNVLink/SXM/adoption proof.',
 parent_cached_pe_phase=read(p.parent/'tune_cached_pe_20261101/broader_validation_state.json')['phase'])
(p/'checkpoint_audit.json').write_text(json.dumps(record,indent=2)+'\n');print('CHECKPOINT_VERIFIED',record['rows'],record['numerical_tests'])
