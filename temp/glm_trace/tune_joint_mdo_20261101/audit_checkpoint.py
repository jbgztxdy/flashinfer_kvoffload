"""Verify all joint-MDO gates, rejected codegen controls and audited performance."""
import hashlib,json,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();read=lambda f:json.loads(f.read_text())
prod='b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a';math='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3';matrix='e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh')==prod and sha(root/'include/flashinfer/attention/mla.cuh')==math
assert sha(root/'temp/glm_trace/kernel_report_20261029/matrix.json')==matrix
assert read(p/'continuation_state.json')['goal_status']=='active'
a=read(p/'performance_audit.json');d=read(p/'decisions.json');v='joint_mdo_typed';host=sha(p/(v+'.cuh'))
assert len(a['rows'])>=6 and len(a['pairs'])>=3
short=read(p/'short_performance_audit.json');assert len(short['rows'])==6 and len(short['pairs'])==3 and short['timed_launches']==1152 and short['accepted_segments']==72
assert not a['pending_references']
assert not d['candidate_adopted']
if 'audit_sha256' in d:assert d['audit_sha256']==sha(p/'performance_audit.json')
assert sum(t['tests'] for t in a['tests'])==315 and sum(t['tests'] for t in a['stress_tests'])==12
assert {r['variant'] for r in a['rows']}=={'base',v}
for suffix,n in [('',34),('_normal',68)]:
 proof=read(p/('compiler_'+v+suffix+'_summary.json'))
 assert proof['complete'] and proof['unchanged_original_kernels']==n
 assert proof['base_header_sha256']==prod and proof['candidate_header_sha256']==host and proof['math_header_sha256']==math
 assert not proof['all_new_stack_and_local_zero']
 assert all('REG:255 STACK:16 SHARED:2304 LOCAL:0 ' in r['resource'] for r in proof['new_kernels'])
for variant in ['joint_mdo','joint_mdo_after','joint_mdo_typed']:
 assert sha(p/(variant+'_mla.cuh'))==math
 probe=read(p/('dispatch_probe_'+variant+'.json'));assert probe['header_sha256']==sha(p/(variant+'.cuh')) and probe['math_header_sha256']==math
 assert probe['graph_nodes']==probe['kernel_nodes']==1
 c=probe['launches'][0];assert c['grid']==[1,108,1] and c['block']==[32,4,2] and c['dynamic_shared_bytes']==147968 and c['active_blocks_per_SM']==1
 assert 'BatchMLAJointMDOKernel' in c['kernel_name'] and probe['max_abs_error']<.01
for name in ['first_codegen_review.json','after_codegen_review.json']:
 r=read(p/name);assert not r['performance_authorized']
 assert r['builds']['emu']['changed_count']==4 and r['builds']['_normal']['changed_count']==8
 assert all(x['pure_device_exact'] for x in r['builds'].values())
assert read(p/'typed_algorithm_identity.json')['algorithm_identical']
for variant in ['joint_mdo','joint_mdo_after']:assert not read(p/('compiler_'+variant+'_summary.json'))['complete']
snapshots=0
files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','--',str(p.relative_to(root))],cwd=root,text=True).splitlines()
for name in set(files):
 f=root/name
 if f.suffix=='.json':read(f)
 if f.parent.name=='source_snapshots' and len(f.stem)==64:assert sha(f)==f.stem;snapshots+=1
for f in p.glob('*.py'):compile(f.read_text(),str(f),'exec')
r=dict(complete=True,goal_status='active',candidate_adopted=False,active_runners=read(p/'continuation_state.json')['active_runners'],performance_incomplete=len(a['incomplete']),normal_tests=327,performance_rows=len(a['rows']),paired_comparisons=len(a['pairs']),timed_launches=a['timed_launches'],accepted_segments=a['accepted_segments'],verified_source_snapshots=snapshots,production_header_sha256=prod,canonical_math_sha256=math,original_matrix_sha256=matrix)
(p/'checkpoint_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
