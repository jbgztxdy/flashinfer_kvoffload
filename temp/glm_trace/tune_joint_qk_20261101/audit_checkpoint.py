"""Validate source/codegen/test provenance and distinguish production from JM comparisons."""
import hashlib,json,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();read=lambda f:json.loads(f.read_text())
prod='b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a';math='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3';matrix='e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
production_sha=sha(root/'include/flashinfer/attention/mla_host_batch.cuh')
if production_sha!=prod:
 retained=read(p/'joint_qk_retention.json');assert retained['candidate_adopted'] and retained['candidate_header_sha256']==production_sha==sha(p/'retained_joint_qk.cuh')
assert sha(root/'include/flashinfer/attention/mla.cuh')==math
assert sha(root/'temp/glm_trace/kernel_report_20261029/matrix.json')==matrix
m=read(p/'manifest.json');host=sha(p/'joint_qk.cuh')
assert m['candidate_sha256']==host and sha(p/'joint_qk_mla.cuh')==m['math_sha256']==math
assert sha(p/'joint_mdo_typed.cuh')==m['joint_mdo_control_sha256']=='1190d4fd5f85d38a489ae67f0fc7e7ad5fdd567f39829eb218e1fb26ef52aa59'
for suffix,n in [('',34),('_normal',68)]:
 proof=read(p/('compiler_joint_qk'+suffix+'_summary.json'))
 assert proof['complete'] and proof['unchanged_original_kernels']==n
 assert proof['base_header_sha256']==prod and proof['candidate_header_sha256']==host and proof['math_header_sha256']==math
 assert proof['all_new_stack_and_local_zero']
 assert all('REG:251 STACK:0 SHARED:2304 LOCAL:0 ' in r['resource'] for r in proof['new_kernels'])
probe=read(p/'dispatch_probe_joint_qk.json');assert probe['header_sha256']==host and probe['math_header_sha256']==math
assert probe['graph_nodes']==probe['kernel_nodes']==1
c=probe['launches'][0];assert c['grid']==[1,108,1] and c['block']==[32,4,2] and c['dynamic_shared_bytes']==147968 and c['active_blocks_per_SM']==1
assert 'BatchMLAJointQKKernel' in c['kernel_name'] and probe['max_abs_error']<.01
reuse=read(p/'reused_joint_control.json');assert reuse['new_tests_claimed']==0
for evidence in reuse['evidence']:assert sha(root/evidence['path'])==evidence['sha256']
assert reuse['control_host_sha256']==m['joint_mdo_control_sha256'] and reuse['control_math_sha256']==math
a=read(p/'performance_audit.json');short=read(p/'short_performance_audit.json')
assert len(short['rows'])==9 and len(short['pairs'])==6 and short['timed_launches']==1728 and short['accepted_segments']==108
assert sum(x['tests'] for x in a['tests']) in (105,210,319) and sum(x['tests'] for x in a['stress_tests']) in (4,8)
assert len(a['rows'])>=18 and not a['pending_references']
index={(r['cohort'],r['setting'],r['variant']):r for r in a['rows']}
for t in read(p/'composition_comparisons.json')['triples']:
 b,j,q=[index[t['cohort'],t['setting'],v]['ours_us'] for v in ['base','joint_mdo_typed','joint_qk']]
 assert t['base_us']==b and t['joint_mdo_us']==j and t['joint_qk_us']==q
 assert abs(t['over_base_reduction']-(1-q/b))<1e-12 and abs(t['over_joint_reduction']-(1-q/j))<1e-12
if read(p/'continuation_state.json')['candidate_adopted']:assert production_sha==sha(p/'retained_joint_qk.cuh')
snapshots=0
files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','--',str(p.relative_to(root))],cwd=root,text=True).splitlines()
for name in set(files):
 f=root/name
 if f.suffix=='.json':read(f)
 if f.parent.name=='source_snapshots' and len(f.stem)==64:assert sha(f)==f.stem;snapshots+=1
for f in p.glob('*.py'):compile(f.read_text(),str(f),'exec')
r=dict(complete=True,goal_status='active',candidate_adopted=production_sha!=prod,new_normal_tests=sum(x['tests'] for x in a['tests'])+sum(x['tests'] for x in a['stress_tests']),reused_control_new_tests=0,
 performance_rows=len(a['rows']),production_paired_comparisons=len(a['pairs']),
 three_way_points=len(read(p/'composition_comparisons.json')['triples']),timed_launches=a['timed_launches'],accepted_segments=a['accepted_segments'],incomplete_raw=len(a['incomplete']),
 active_runners=read(p/'continuation_state.json')['active_runners'],verified_source_snapshots=snapshots,production_header_sha256=production_sha,canonical_math_sha256=math,original_matrix_sha256=matrix)
(p/'checkpoint_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
