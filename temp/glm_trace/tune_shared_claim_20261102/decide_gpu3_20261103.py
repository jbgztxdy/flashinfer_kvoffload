import hashlib,json,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
load=lambda n:json.loads((p/n).read_text())
a=load('performance_audit.json');m=load('manifest.json');s=load('gpu3_20261103_state.json')
assert s['complete'] and all(x['exit_code']==0 for x in s['actions']) and len(a['pairs'])==3
assert all(x['hardware']=='GPU3' and x['reduction']<0 for x in a['pairs'])
assert hashlib.sha256((root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes()).hexdigest()==m['base_sha256']
r=dict(timestamp=time.strftime('%FT%T%z'),decision='rejected',adopted=False,header_sha256=m['candidate_sha256'],
 production_unchanged=m['base_sha256'],reason='Fresh same-GPU3 pairs slower5.72/6.65% at100GB0/20us and3.70% at50GB20us. Original incompleteSXM candidate stays excluded; no need to resume it.',
 pairs=a['pairs'],reuse_validation='Original2targetedSXM BF16tests passed; no repetition this round. Same frozen a47ca8b8 header andbinary.',
 fresh_timed_launches=1152,fresh_segments=72,live_runners=[],goal_complete=False,
 scope='GPU3/NUMA1same-devicefarHBM,256positions x3coldpasses,MTP8/R128/LRU4096/f24off; independent freshbase, no SXM absolute-time mixing.')
(p/'decisions_gpu3_20261103.json').write_text(json.dumps(r,indent=2)+'\n')
prior=load('continuation_state.json');(p/'continuation_state_20261102.json').write_text(json.dumps(prior,indent=2)+'\n')
prior.update(timestamp=r['timestamp'],status='rejected_after_valid_GPU3_screen',adopted=False,valid_candidate_performance_cells=3,next_step='Do not rerunshared_claim. Continue new mechanisms from production754706f8.',decision_file='decisions_gpu3_20261103.json',live_task_runners=[])
(p/'continuation_state.json').write_text(json.dumps(prior,indent=2)+'\n');print(json.dumps(r,indent=2))
