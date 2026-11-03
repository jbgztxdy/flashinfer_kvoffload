import hashlib,json,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
load=lambda n:json.loads((p/n).read_text())
a=load('performance_audit.json');m=load('manifest.json');s=load('focused_state.json')
assert s['complete'] and all(x['exit_code']==0 for x in s['actions']) and len(a['pairs'])==3
assert hashlib.sha256((root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes()).hexdigest()==m['base_sha256']
r=dict(timestamp=time.strftime('%FT%T%z'),decision='rejected',adopted=False,header_sha256=m['candidate_sha256'],
 production_unchanged=m['base_sha256'],targeted_tests=3,initial_numerical_failure='failed_lse/',
 reason='100GB0us faster0.292%,20us slower1.674%,50GB flat(-0.003%). No stable gain; no parameter scan or full-window expansion.',
 pairs=a['pairs'],timed_launches=a['timed_launches'],segments=a['accepted_segments'],
 scope='GPU3/NUMA1, same-devicefarHBM EMU; BF16/MTP8/R128/LRU4096/f24off/256positions x3coldpasses. No FP16/SXM/H800/native claims.',
 live_runners=[],goal_complete=False)
(p/'decisions.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
