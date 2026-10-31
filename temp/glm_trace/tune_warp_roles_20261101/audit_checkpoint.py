"""Validate the completed negative control without discarding v1/preflight failures."""
import hashlib,json,re,gzip,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();read=lambda f:json.loads(f.read_text())
prod='b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
canonical='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
matrix='e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh')==prod
assert sha(root/'include/flashinfer/attention/mla.cuh')==canonical
assert sha(root/'temp/glm_trace/kernel_report_20261029/matrix.json')==matrix
assert not read(p/'continuation_state.json')['active_runners']
a=read(p/'performance_audit.json');d=read(p/'decisions.json')
assert not d['candidate_adopted'] and d['audit_sha256']==sha(p/'performance_audit.json')
assert len(a['rows'])==6 and len(a['pairs'])==3
assert a['timed_launches']==1152 and a['accepted_segments']==72
assert not a['pending_references'] and not a['incomplete']
assert sum(t['tests'] for t in a['tests'])==210 and sum(t['tests'] for t in a['stress_tests'])==8
assert {r['variant'] for r in a['rows']}=={'base','warp_roles_isolated'}
v='warp_roles_isolated';host,math=[sha(p/(v+s+'.cuh')) for s in ['', '_mla']]
m=read(p/'isolated_manifest.json');assert m['candidate_sha256']==host and m['candidate_math_sha256']==math
for suffix,n in [('',34),('_normal',68)]:
    proof=read(p/('compiler_'+v+suffix+'_summary.json'))
    assert proof['complete'] and proof['unchanged_original_kernels']==n
    assert proof['base_header_sha256']==prod and proof['candidate_header_sha256']==host and proof['math_header_sha256']==math
    assert proof['all_new_stack_and_local_zero']
    assert all('REG:246 ' in r['resource'] and 'SHARED:1408 ' in r['resource'] for r in proof['new_kernels'])
for variant in ['warp_roles','warp_roles_isolated']:
    probe=read(p/('dispatch_probe_'+variant+'.json'))
    assert probe['header_sha256']==sha(p/(variant+'.cuh')) and probe['math_header_sha256']==sha(p/(variant+'_mla.cuh'))
    assert probe['graph_nodes']==probe['kernel_nodes']==1
    c=probe['launches'][0]
    assert c['grid']==[1,108,1] and c['block']==[32,2,4]
    assert c['dynamic_shared_bytes']==147968 and c['active_blocks_per_SM']==1
    assert 'BatchMLAWarpRoleKernel' in c['kernel_name'] and probe['max_abs_error']<.01
review=read(p/'v1_codegen_review.json')
assert not review['performance_authorized']
for suffix,n in [('emu',22),('_normal',44)]:
    assert review['builds'][suffix]['changed_count']==n
    assert review['builds'][suffix]['pure_device_exact']
assert not read(p/'compiler_warp_roles_summary.json')['complete']
assert read(p/'screen_missing_plan_failure.json')['timing_launches']==0
assert sha(p/'warp_roles_generation_failed.cuh')==read(p/'generation_failure.json')['header_sha256']
assert all(any('0x1, 0x80' in b for b in r['barriers']) and any('0x2, 0x80' in b for b in r['barriers']) for r in read(p/'role_codegen_counts.json'))
files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','--',str(p.relative_to(root))],cwd=root,text=True).splitlines()
snapshots=0
for name in set(files):
    f=root/name
    if f.suffix=='.json':read(f)
    if f.parent.name=='source_snapshots' and len(f.stem)==64:
        assert sha(f)==f.stem;snapshots+=1
for f in p.glob('*.py'):compile(f.read_text(),str(f),'exec')
result=dict(complete=True,goal_status='active',candidate_adopted=False,normal_tests=218,
    performance_rows=6,paired_comparisons=3,timed_launches=1152,accepted_segments=72,
    verified_source_snapshots=snapshots,production_header_sha256=prod,canonical_math_sha256=canonical,original_matrix_sha256=matrix)
(p/'checkpoint_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
