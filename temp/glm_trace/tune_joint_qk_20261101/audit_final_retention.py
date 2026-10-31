"""Verify completed retention against current production and preserved measurement inputs."""
import hashlib
import json
from pathlib import Path
import subprocess

p=Path(__file__).resolve().parent;root=p.parents[2]
read=lambda name:json.loads((p/name).read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
r=read('joint_qk_retention.json');s=read('completion_state.json');a=read('performance_audit.json');c=read('coverage.json')
assert s['complete'] and s['phase']=='complete' and s['candidate_adopted'] and r['candidate_adopted']
assert not s['goal_complete'] and not r['goal_complete']
assert all(x['exit_code']==0 for x in s['steps'])
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh')==sha(p/'retained_joint_qk.cuh')==r['candidate_header_sha256']=='7193255e2fdf83d3baa9347dea1a02b892c29ed746da719818e30fdf4066d469'
assert sha(p/'joint_qk.cuh')==r['measured_header_sha256']=='a2f37fb985fcbc523e740c85e9cc6468afad74fd8716c11aef4221ebab7c4b7c'
assert sha(root/'include/flashinfer/attention/mla.cuh')=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
assert sha(root/'include/flashinfer/attention/far_link_emu.cuh')=='5e409035aa4edc83f72135c09c453e825ce59e5dcc4bae63c04ff3909d024095'
assert sha(root/'temp/glm_trace/kernel_report_20261029/matrix.json')=='e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
assert len(a['rows'])==210 and len(a['pairs'])==110 and not a['pending_references'] and not a['incomplete']
assert c['gpu3_matrix_complete'] and c['sxm_matrix_complete'] and sum(x['completed_audited_cells'] for x in c['groups'])==156
assert c['audit_sha256']==sha(p/'performance_audit.json')
assert r['matrix_cells']==156 and r['matrix_pairs']==78 and r['changed_lru_pairs']==60
for item in r['frozen']:assert sha(root/item['snapshot'])==item['sha256']
for suffix,count in [('',36),('_normal',72)]:
    proof=read('retained_codegen'+suffix+'_summary.json')
    assert proof['complete'] and proof['unchanged_kernel_instances']==count and proof['all_original_and_new_targets_identical']
t=read('tests_production_gpu3.json');code=read('production_codegen_normal_summary.json');probe=read('dispatch_probe_production_joint_qk.json')
assert t['complete'] and t['exit_code']==0 and t['expected_tests']==109 and t['normal_build']
assert t['production_header_sha256']==r['candidate_header_sha256']
assert t['resources_start']['foreign']==t['resources_end']['foreign']==[]
assert Path(t['include_path'])==root/'include' and 'jit_production_joint_normal' in t['fresh_workspace']
assert code['complete'] and code['unchanged_kernel_instances']==72 and code['all_original_and_new_targets_identical']
assert probe['graph_nodes']==probe['kernel_nodes']==1 and probe['max_abs_error']<.01
assert probe['launches'][0]['block']==[32,4,2] and probe['launches'][0]['grid']==[1,108,1]
assert probe['launches'][0]['dynamic_shared_bytes']==147968 and probe['launches'][0]['active_blocks_per_SM']==1
for key,name in [('sha256','tests_production_gpu3.json'),('codegen_sha256','production_codegen_normal_summary.json'),('dispatch_sha256','dispatch_probe_production_joint_qk.json')]:
    assert r['production_validation'][key]==sha(p/name)
for lib in t['libraries']:assert sha(Path(lib['path']))==lib['sha256']
for setting in ['g100_l0','g100_l20000','g50_l20000']:
    f=p/'raw'/f'compositionsxmforward1024_f24_joint_qk_staged_f24_i64_soff_lru_{setting}_mtp8_b128.json'
    assert json.loads(f.read_text())['complete']
processes=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
live=[line for line in processes if '/python' in line and ('tune_joint_qk_20261101/complete_retention.py' in line or 'tune_joint_qk_20261101/confirm_sxm.py' in line)]
assert not live,live
result=dict(complete=True,retention_complete=True,goal_complete=False,candidate_adopted=True,
    production_header_sha256=r['candidate_header_sha256'],measured_header_sha256=r['measured_header_sha256'],
    matrix_cells=156,changed_pairs=60,total_measured_cells=210,total_pairs=110,
    fresh_root_tests=109,normal_codegen_instances=72,actual_graph_kernel_nodes=1,
    original_handles_terminal={'72075':0,'1398':0},original_producers_live=[],
    current_forward_missing=[],remaining_goal='Large100GB stillabout25-27%above max reference;cachedPE followup independently validating,not installed.',
    note='Producer completion_state retains its last waiting-loop missing/live fields; current raw,coverage,process state and this final audit prove none remain. Historical snapshots are preserved.')
(p/'final_retention_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
