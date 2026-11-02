import hashlib,json,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
load=lambda name:json.loads((p/name).read_text())
a=load('performance_audit.json');m=load('manifest.json');s=load('focused_sxm_state.json')
assert s['complete'] and all(x['exit_code']==0 for x in s['actions'])
assert len(a['pairs'])==3 and not a['incomplete'] and not a['pending_references']
assert hashlib.sha256((root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes()).hexdigest()==m['base_sha256']
r=dict(timestamp=time.strftime('%FT%T%z'),decision='rejected',adopted=False,production_unchanged=m['base_sha256'],
 candidate_sha256=m['candidate_sha256'],targeted_tests=2,cpu_queue_cases=m['cpu_queue_cases'],
 pairs=a['pairs'],timed_launches=a['timed_launches'],accepted_segments=a['accepted_segments'],
 reason='100GB0us essentially flat (-0.117%);100GB20us slower5.964%;50GB20us faster0.133%. Earlier compute participation does not yield an end-to-end win.',
 scope='Physical SXM0/1 NV12 source, NUMA0, EMU100GB0/20us and50GB20us, MTP8/R128/LRU4096,256positions x3 coldpasses,f24/off.',
 limitations='No full-window/native/FP16/H800/exhaustive tests/sanitizers. GPU3 original attempt stopped before CUDA; the measured continuation ran on newly idle SXM.',
 processes='Build and SXM focused runner terminal0; GPU3 preflight attempt exit1 before CUDA. No background runner.')
(p/'decisions.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
