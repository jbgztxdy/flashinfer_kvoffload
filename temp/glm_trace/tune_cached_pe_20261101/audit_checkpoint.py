"""Audit current candidate evidence without equating partial coverage with adoption."""
import hashlib
import json
from pathlib import Path
import subprocess

p=Path(__file__).resolve().parent
root=p.parents[2]
read=lambda f:json.loads(f.read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
m=read(p/'manifest.json')
assert sha(p/'base.cuh')==m['base_sha256']=='a2f37fb985fcbc523e740c85e9cc6468afad74fd8716c11aef4221ebab7c4b7c'
assert sha(p/'cached_pe.cuh')==m['candidate_sha256']=='a43617290e405fdcd54b09140dbc29b357d2923e17a3d99a193caf5f7d592b15'
assert sha(p/'cached_pe_mla.cuh')==m['math_sha256']==sha(root/'include/flashinfer/attention/mla.cuh')
assert m['math_sha256']=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
assert sha(root/'include/flashinfer/attention/far_link_emu.cuh')=='5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
assert sha(root/'temp/glm_trace/kernel_report_20261029/matrix.json')=='e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
production=sha(root/'include/flashinfer/attention/mla_host_batch.cuh')
assert production!=m['candidate_sha256']
if production!='b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a':
    retention=read(p.parent/'tune_joint_qk_20261101/joint_qk_retention.json')
    assert retention['candidate_adopted'] and retention['candidate_header_sha256']==production
for suffix,count,stack in [('',36,8),('_normal',72,0)]:
    proof=read(p/('compiler_cached_pe'+suffix+'_summary.json'))
    assert proof['complete'] and proof['unchanged_original_kernels']==count
    assert proof['base_header_sha256']==m['base_sha256'] and proof['candidate_header_sha256']==m['candidate_sha256']
    assert proof['math_header_sha256']==m['math_sha256']
    assert proof['all_new_stack_and_local_zero']==(stack==0)
    assert all(f'REG:255 STACK:{stack} SHARED:2304 LOCAL:0 ' in r['resource'] for r in proof['new_kernels'])
words=read(p/'machine_word_identity.json')
assert words['complete'] and [r['original_kernel_instances'] for r in words['builds']]==[36,72]
assert all(r['all_words_identical'] for r in words['builds'])
probe_files=list((p/'dispatch_probes').glob('*.json'))
assert len(probe_files)==3
for f in probe_files:
    d=read(f);launch=d['launches'][0]
    assert d['header_sha256']==m['candidate_sha256'] and d['math_header_sha256']==m['math_sha256']
    assert d['probe_source_sha256']==sha(p/'probe_cached_pe.py')
    assert d['graph_nodes']==d['kernel_nodes']==1 and d['max_abs_error']<.01
    assert d['resources_start']['foreign']==d['resources_end']['foreign']==[]
    assert launch['grid']==[1,108,1] and launch['block']==[32,4,2]
    assert launch['dynamic_shared_bytes']==147968 and launch['active_blocks_per_SM']==1
    assert 'BatchMLACachedPEKernel' in launch['kernel_name']
for kind in ['synccheck','racecheck']:
    assert 'SUMMARY: 0' in (p/(kind+'_cached_pe.log')).read_text()
for record in read(p/'reused_joint_control.json')['records']:
    assert sha(root/record['path'])==record['sha256']
for name in ['validation_state.json','extension_state.json']:
    s=read(p/name);assert s['complete'] and all(x['exit_code']==0 for x in s['steps'])
for tag in ['cachedreverse1024_f24_']:
    s=read(p/(tag+'gpu3_state.json'));assert len(s)==2 and all(x['exit_code']==0 for x in s)
a=read(p/'performance_audit.json')
assert len(a['rows'])>=22 and len(a['pairs'])>=11 and not a['pending_references'] and not a['incomplete']
assert sum(t['tests'] for t in a['tests'])+sum(t['tests'] for t in a['stress_tests'])==111
for cohort in ['cachedreverse1024_f24_','cachedforward1024_f24_']:
    pairs=[r for r in a['pairs'] if r['cohort']==cohort]
    assert len(pairs)==3 and all(r['reduction']>0 for r in pairs)
native=[r for r in a['pairs'] if r['cohort']=='cachednative1024_f24_']
assert len(native)==2 and all(r['reduction']>-.01 for r in native)
c=read(p/'coverage.json');assert len(c['groups'])==4 and sum(g['expected_cells'] for g in c['groups'])==156
assert c['audit_sha256']==sha(p/'performance_audit.json')
review=read(p/'static_instruction_review.json')
assert review['complete'] and review['cache_liveness']['verified']
assert review['cache_liveness']['registers']==list(range(12,28))
assert review['selected']['LDSM']==dict(base=198,cached_pe=190)
overlap=read(p/'prepare_overlap_review.json')
assert overlap['cpu_only'] and not overlap['kernel_performance_claim']
assert overlap['source_sha256']==sha(p/'inspect_prepare_overlap.py')
assert overlap['layouts']['rows2_cols16']['duplicate_reference_fraction']<.02
assert overlap['layouts']['rows8_cols4']['duplicate_reference_fraction']<.02
state=read(p/'broader_validation_state.json')
assert state['phase']!='failed' and all(x.get('exit_code',0)==0 for x in state['steps'])
live=[]
if not state['complete']:
    ps=subprocess.check_output(['ps','-p',str(state['pid']),'-o','pid,args'],text=True)
    assert 'tune_cached_pe_20261101/validate_more.py' in ps
    live=ps.splitlines()[1:]
decision=read(p/'decisions.json');assert not decision['candidate_adopted'] and not decision['goal_complete']
assert decision['audit_sha256']==sha(p/'performance_audit.json')
snapshots=jsons=0
names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','--',str(p.relative_to(root))],cwd=root,text=True).splitlines()
for name in set(names):
    f=root/name
    if f.suffix=='.json':read(f);jsons+=1
    if f.parent.name=='source_snapshots' and len(f.stem)==64:
        assert sha(f)==f.stem;snapshots+=1
for f in p.glob('*.py'):compile(f.read_text(),str(f),'exec')
result=dict(complete=True,evidence_checkpoint_only=True,goal_complete=False,candidate_adopted=False,
    tests=111,measured_cells=len(a['rows']),pairs=len(a['pairs']),timed_launches=a['timed_launches'],
    accepted_segments=a['accepted_segments'],planned_matrix_cells=156,gpu3_matrix_complete=c['gpu3_matrix_complete'],
    sxm_matrix_complete=c['sxm_matrix_complete'],missing_matrix_cells=sum(len(g['missing_cells']) for g in c['groups']),
    verified_source_snapshots=snapshots,parsed_json=jsons,live_own_producers=live,
    production_header_sha256=production,comparison_base_sha256=m['base_sha256'],candidate_header_sha256=m['candidate_sha256'],
    remaining_retention_gates=c['remaining_retention_gates'])
(p/'checkpoint_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
