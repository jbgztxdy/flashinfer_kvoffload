import hashlib,json,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
load=lambda name:json.loads((p/name).read_text())
a=load('performance_audit.json');m=load('manifest.json');s=load('focused_state.json')
assert s['complete'] and all(x['exit_code']==0 for x in s['actions'])
assert len(a['pairs'])==3 and not a['incomplete'] and not a['pending_references']
assert all(x['reduction']<0 for x in a['pairs'])
assert hashlib.sha256((root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes()).hexdigest()==m['base_sha256']
record=dict(timestamp=time.strftime('%FT%T%z'),decision='rejected',adopted=False,production_unchanged=m['base_sha256'],
 mechanism=m['mechanism'],candidate_sha256=m['candidate_sha256'],targeted_tests=2,
 pairs=a['pairs'],timed_launches=a['timed_launches'],accepted_segments=a['accepted_segments'],
 reason='Native64KV/two stages with global query is correct in focused checks but slower by44.5/43.1% at100GB and6.7% at50GB. Do not expand or scan parameters.',
 scope='GPU3 A100-PCIE NUMA1, same-device far-HBM emulation. BF16/MTP8/R128/LRU4096/f24off/256positions x3 cold passes; no NVLink/SXM/H800 claims.',
 limitations='No exhaustive tests/sanitizers/full-window confirmation. Original instances are retained in source; no new full binary-equivalence audit. No strict causal attribution.',
 processes='CPU builds and focused GPU runner terminal0; no background runner.')
(p/'decisions.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
