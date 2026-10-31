"""Retain the tested cached-only ID specialization with explicit slow-cell evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import time

p=Path(__file__).resolve().parent;root=p.parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--install',action='store_true');a=parser.parse_args()
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
load=lambda f:json.loads(f.read_text())
base='52919c286398d4668933a772c646bdccd46273a53aafeb8d5c1b14e2d9bff2c5'
candidate='b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
assert sha(p/'base.cuh')==base and sha(p/'cached_ids.cuh')==candidate
assert sha(root/'include/flashinfer/attention/mla.cuh')=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
assert sha(root/'temp/glm_trace/kernel_report_20261029/matrix.json')=='e3fd023ef4f39adc0cf23146d665b1e8f7a4fbe17d7854c246c452ba2d679f99'
coverage=load(p/'coverage.json');assert coverage['cached_planned_complete'] and coverage['cached_measured_complete']
audit=load(p/'performance_audit.json');assert not audit['pending_references']
pairs=[r for r in audit['pairs'] if r['cohort'].startswith('cached_ids_matrix_')]
assert len(pairs)==448
changed=[r for r in pairs if r['kind']=='lru' and r['mtp']*r['requests']!=224 and r['mtp']*r['requests']>=216]
assert len(changed)==96
for normal,count in [(False,9),(True,18)]:
    proof=load(p/('compiler_cached'+('_normal' if normal else '')+'_summary.json'))
    assert proof['candidate_header_sha256']==candidate and proof['unchanged_original_kernels']==count
    assert proof['all_stack_and_local_zero']
tests=load(p/'tests_cached_ids_gpu3.json')
assert tests['complete'] and tests['normal_build'] and tests['exit_code']==0 and tests['header_sha256']==candidate
assert tests['log_sha256']==sha(p/'pytest_cached_ids_gpu3.log') and '105 passed' in (p/'pytest_cached_ids_gpu3.log').read_text()
slow=[r for r in changed if r['reduction']<-.01]
assert {(r['mtp'],r['requests'],r['setting']) for r in slow}=={(8,128,'g100_l'+str(x)) for x in [0,2000,5000,10000]}
brackets=[]
for setting in ['g100_l0','g100_l2000','g100_l5000','g100_l10000','pcie']:
    rows=[r for r in audit['rows'] if r['cohort'].startswith('cached_sxm_slow_') and r['setting']==setting]
    assert len(rows)==3 and all(r['positions']==1024 and r['passes']==5 for r in rows)
    phases={r['cohort']:r for r in rows}
    before,new,after=[phases['cached_sxm_slow_'+x+'_'] for x in ['before','candidate','after']]
    reduction=1-new['ours_us']/statistics.mean([before['ours_us'],after['ours_us']])
    assert reduction>=-.01
    brackets.append(dict(setting=setting,base_before_us=before['ours_us'],candidate_us=new['ours_us'],base_after_us=after['ours_us'],reduction=reduction))
controls=load(p/'controls_audit.json')
assert controls['complete'] and controls['shared_allocation_diagnostic']['distinct_captured_functions_verified']
assert abs(controls['shared_allocation_diagnostic']['reduction'])<.01
allocation=load(p.parent/'tune_cached_runtime_20261031/allocation_audit.json')
assert allocation['complete'] and allocation['actual_distinct_functions_and_equal_launch_shape']
assert all(r['reductions']['cached_ids']>=-.01 for r in allocation['summary'])
assert statistics.mean(r['reduction'] for r in changed)>0
production=root/'include/flashinfer/attention/mla_host_batch.cuh'
assert sha(production) in [base,candidate],'Unrelated production edit'
snapshot=p/'retention_snapshot';snapshot.mkdir(exist_ok=True)
files=['performance_audit.json','coverage.json','tests_cached_ids_gpu3.json','compiler_cached_summary.json',
       'compiler_cached_normal_summary.json','controls_audit.json','same_process_dispatch.json']
frozen=[]
for name in files:
    source=p/name;target=snapshot/(sha(source)+'_'+name)
    if target.exists():assert target.read_bytes()==source.read_bytes()
    else:shutil.copyfile(source,target)
    frozen.append(dict(path=str(source.relative_to(root)),sha256=sha(source),snapshot=str(target.relative_to(root))))
record=dict(timestamp=time.strftime('%FT%T%z'),candidate_adopted=bool(a.install),goal_complete=False,
    base_header_sha256=base,candidate_header_sha256=candidate,matrix_cells=896,matrix_pairs=448,
    changed_lru_pairs=96,unchanged_original_kernel_pairs=352,
    changed_matrix_minimum_reduction=min(r['reduction'] for r in changed),
    changed_matrix_mean_reduction=statistics.mean(r['reduction'] for r in changed),
    changed_matrix_maximum_reduction=max(r['reduction'] for r in changed),brackets=brackets,frozen=frozen,
    slow_cell_policy='All original negatives remain. Changed-path >1% matrix negatives are bracketed at full5pass within1%; unchanged paths retain identical instructions/resources, with explicit representative controls and distinct-function same-allocation checks. This does not promise every independent allocation/timing sample is faster.',
    limitation='Main100GBhistorical32-34%gap remains; large100GB gains are small, nativeNVLink/medium improvements are the retention basis.')
if a.install:
    production.write_bytes((p/'cached_ids.cuh').read_bytes());assert sha(production)==candidate
(p/'cached_ids_retention.json').write_text(json.dumps(record,indent=2)+'\n')
print({k:v for k,v in record.items() if k not in ['frozen','brackets']})
