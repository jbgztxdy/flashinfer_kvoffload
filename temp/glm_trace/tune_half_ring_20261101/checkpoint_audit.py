"""Bind this checkpoint to frozen sources, actual dispatch, and all archived outcomes."""
import hashlib,json,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
read=lambda f:json.loads(f.read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
manifest=read(p/'manifest.json');fixed=read(p/'fixed_control_manifest.json')
headers=dict(base=manifest['base_sha256'],**manifest['candidate_sha256'],no_copy_pe_fixed=fixed['header_sha256'])
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh')=='7193255e2fdf83d3baa9347dea1a02b892c29ed746da719818e30fdf4066d469'
assert sha(root/'include/flashinfer/attention/mla.cuh')==manifest['math_sha256']
assert sha(root/'include/flashinfer/attention/far_link_emu.cuh')=='5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
assert sha(root/'temp/glm_trace/kernel_report_20261029/matrix.json')=='e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
for v,h in headers.items():
    assert sha(p/(v+'.cuh'))==h
    assert sha(p/(v+'_mla.cuh'))==manifest['math_sha256']
    for n,expected in [('mla_host_batch.cuh',h),('mla.cuh',manifest['math_sha256'])]:
        f=p/('include_'+v)/'flashinfer/attention'/n
        if f.exists():assert sha(f)==expected
proofs=[]
for v in ['no_copy_pe','no_copy_pe_fixed','half_ring']:
    for suffix,expected in [('',38),('_normal',76)]:
        f=p/('compiler_'+v+suffix+'_summary.json');d=read(f)
        assert d['complete'] and d['unchanged_original_kernels']==expected
        assert d['all_original_instruction_and_control_words_identical']
        assert d['candidate_header_sha256']==headers[v]
        for name,libs in d['libraries'].items():
            folder=p.parent/'tune_cached_pe_20261101'/('jit_cached_pe'+suffix) if name=='base' else p/('jit_'+name+suffix)
            for lib in libs:
                paths=list(folder.rglob(lib['name']));assert len(paths)==1
                assert sha(paths[0])==lib['sha256']
        proofs.append(dict(file=f.name,sha256=sha(f),unchanged_original_kernels=expected,new_resources=[k['resource'] for k in d['new_kernels']]))
state=read(p/'validation_fixed_state.json');perf=read(p/'performance_audit.json');archived=read(p/'archived_policy_audit.json')
assert state['complete'] and state['phase']=='fixed_short_screen_complete' and all(x['exit_code']==0 for x in state['steps'])
assert len(perf['rows'])==9 and len(perf['pairs'])==6 and not perf['incomplete'] and not perf['pending_references']
assert perf['timed_launches']==1728 and perf['accepted_segments']==108
assert sum(x['tests'] for x in perf['tests']+perf['stress_tests'])==222
assert len(archived['rows'])==9 and archived['out_of_bounds_claim_retracted'] and archived['device_memcheck']['zero_errors']
probes=[]
for v,mem in [('no_copy_pe_fixed',147968),('half_ring',158336)]:
    d=read(p/('dispatch_probe_'+v+'.json'))
    assert d['header_sha256']==headers[v] and d['kernel_nodes']==1
    assert d['launches'][0]['grid']==[1,108,1] and d['launches'][0]['block']==[32,4,2]
    assert d['launches'][0]['dynamic_shared_bytes']==mem and d['launches'][0]['active_blocks_per_SM']==1
    assert d['resources_start']['foreign']==d['resources_end']['foreign']==[]
    for kind in ['synccheck','racecheck','memcheck']:
        f=p/(kind+'_'+v+'.log');text=f.read_text()
        assert 'ERROR SUMMARY: 0 errors' in text or (kind=='racecheck' and 'RACECHECK SUMMARY: 0 hazards' in text)
    probes.append(dict(variant=v,captures=len(list((p/('dispatch_probes_'+v)).glob('*.json'))),dispatch_sha256=sha(p/('dispatch_probe_'+v+'.json'))))
record=dict(complete=True,timestamp=time.strftime('%FT%T%z'),goal_complete=False,candidate_adopted=False,
    headers=headers,proofs=proofs,probes=probes,valid_fixed_cohort=dict(rows=9,pairs=6,timed_launches=1728,segments=108,numerical_tests=222),
    archived_first_cohort=dict(rows=9,pairs=6,timed_launches=1728,segments=108,numerical_tests_including_reused_half_ring=222,
        unique_additional_tests=111,scope='one-buffer control is numerically/memory valid; it changed fetch policy and cannot isolate unused QPE copy removal'),
    total_unique_numerical_tests=333,production_unchanged=True,
    limitation='Short256position/3coldpass GPU3 screen. No fulltrace/nativeNVLink/SXM or adoption proof.',
    parent_cached_pe_state=read(p.parent/'tune_cached_pe_20261101/broader_validation_state.json')['phase'])
(p/'checkpoint_audit.json').write_text(json.dumps(record,indent=2)+'\n')
print('CHECKPOINT_VERIFIED',record['valid_fixed_cohort'])
