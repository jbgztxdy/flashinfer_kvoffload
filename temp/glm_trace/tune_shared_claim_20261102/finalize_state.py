import hashlib,json,time
from pathlib import Path
from resources import snapshot
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
m=json.loads((p/'manifest.json').read_text());state=json.loads((p/'focused_sxm_state.json').read_text());audit=json.loads((p/'performance_audit.json').read_text())
assert state['actions'][0]['exit_code']==0 and state['actions'][-1]['exit_code']==1
assert not audit['pairs'] and len(audit['rows'])==3 and len(audit['incomplete'])==1
assert sha(root/'include/flashinfer/attention/mla_host_batch.cuh')==m['base_sha256']
record=dict(timestamp=time.strftime('%FT%T%z'),adopted=False,goal_complete=False,status='awaiting_idle_GPU_for_candidate_timings',
 header_sha256=m['candidate_sha256'],production_sha256=m['base_sha256'],canonical_math_sha256=m['math_sha256'],
 targeted_tests=2,valid_candidate_performance_cells=0,valid_baseline_cells=3,
 incomplete_candidate_raw=audit['incomplete'],resources_current=snapshot(),
 original_runner=dict(exec_handle=35058,exit_code=1,reason='Foreign GPU0/1 processes appeared during candidate run; end resource check rejected the first candidate cell. Original data stays incomplete and excluded.'),
 cpu_builds=dict(handles=[45564,87149],exit_codes=[0,0],first_build='Pre-GPU scratch alias draft archived; second build uses final a47ca8b8.'),
 live_task_runners=[],signals_sent=[],
 next_step='Fresh resource preflight, then only resume_candidate.py; it takes gpu01 lock through bench_sxm, archives incomplete raw and measures3candidate points. Keep existing3basepoints and2correctness tests. Run audit_performance.py afterward. No permission needed.',
 limitations='No valid candidate timing; do not use3440us from contaminated incomplete raw. No exact fullcache-slot LRU tie oracle/sanitizer/exhaustive checks this round. Production unchanged.')
(p/'continuation_state.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
