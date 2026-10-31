"""Reproduce the definition-order control without changing its frozen algorithm."""
import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
s=(p/'joint_mdo.cuh').read_text();a=s.index('// Two native32-row score tiles');b=s.index('template <typename KTraits, typename Params, bool ROW_MERGE>\n__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLAHostBatchKernel(',a)
extra=s[a:b];s=s[:a]+s[b:];key='// partial_capacity bounds the item states of the tail rows, in 64-head tiles.';assert s.count(key)==1;s=s.replace(key,extra+'\n'+key,1)
m=json.loads((p/'after_manifest.json').read_text());sha=lambda s:hashlib.sha256(s.encode()).hexdigest();assert sha(s)==m['candidate_sha256']
for name,value in [('joint_mdo_after.cuh',s),('joint_mdo_after_mla.cuh',(p/'base_mla.cuh').read_text())]:
 f=p/name
 if f.exists():assert f.read_text()==value
 else:f.write_text(value)
print('FROZEN_AFTER_REPRODUCED',sha(s))
