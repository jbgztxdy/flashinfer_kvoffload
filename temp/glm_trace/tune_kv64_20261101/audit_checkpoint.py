"""Check the frozen native64KV negative control and unchanged production/reference inputs."""
import hashlib,json,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();read=lambda f:json.loads(f.read_text())
prod='b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a';math='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3';matrix='e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh')==prod
assert sha(root/'include/flashinfer/attention/mla.cuh')==math
assert sha(root/'temp/glm_trace/kernel_report_20261029/matrix.json')==matrix
assert not read(p/'continuation_state.json')['active_runners']
a=read(p/'performance_audit.json');d=read(p/'decisions.json');host=sha(p/'kv64.cuh')
assert len(a['rows'])==6 and len(a['pairs'])==3 and a['timed_launches']==1152 and a['accepted_segments']==72
assert not a['pending_references'] and not a['incomplete']
assert not d['candidate_adopted'] and d['audit_sha256']==sha(p/'performance_audit.json')
assert sum(t['tests'] for t in a['tests'])==105 and sum(t['tests'] for t in a['stress_tests'])==4
for suffix,n in [('',34),('_normal',68)]:
 proof=read(p/('compiler_kv64'+suffix+'_summary.json'))
 assert proof['complete'] and proof['unchanged_original_kernels']==n
 assert proof['base_header_sha256']==prod and proof['candidate_header_sha256']==host and proof['math_header_sha256']==math
 assert not proof['all_new_stack_and_local_zero']
 assert all('REG:255 STACK:8 SHARED:2304 LOCAL:0 ' in r['resource'] for r in proof['new_kernels'])
probe=read(p/'dispatch_probe_kv64.json');assert probe['header_sha256']==host and probe['math_header_sha256']==math
assert probe['graph_nodes']==probe['kernel_nodes']==1
c=probe['launches'][0];assert c['grid']==[1,108,1] and c['block']==[32,4,2] and c['dynamic_shared_bytes']==147968 and c['active_blocks_per_SM']==1
assert 'KV64HostBatchParams' in c['kernel_name'] and probe['max_abs_error']<.01
cg=read(p/'codegen_detail.json');assert cg['hmma']==822 and cg['ldsm']==528 and cg['local_ops']
assert sha(p/'kv64_mla.cuh')==math
snapshots=0
files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','--',str(p.relative_to(root))],cwd=root,text=True).splitlines()
for name in set(files):
 f=root/name
 if f.suffix=='.json':read(f)
 if f.parent.name=='source_snapshots' and len(f.stem)==64:assert sha(f)==f.stem;snapshots+=1
for f in p.glob('*.py'):compile(f.read_text(),str(f),'exec')
r=dict(complete=True,goal_status='active',candidate_adopted=False,normal_tests=109,performance_rows=6,paired_comparisons=3,timed_launches=1152,accepted_segments=72,verified_source_snapshots=snapshots,production_header_sha256=prod,canonical_math_sha256=math,original_matrix_sha256=matrix)
(p/'checkpoint_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
