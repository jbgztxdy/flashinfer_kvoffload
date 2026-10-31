"""Keep cached-only ID sharing while restoring the original runtime cache predicate."""
import hashlib
import json
import shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
w=p.parent/'tune_wait_20261031';q=p.parent/'tune_qk_20261031'
base=(w/'base.cuh').read_text();old=(w/'cached_ids.cuh').read_text()
assert hashlib.sha256(base.encode()).hexdigest()=='52919c286398d4668933a772c646bdccd46273a53aafeb8d5c1b14e2d9bff2c5'
assert hashlib.sha256(old.encode()).hexdigest()=='b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
key='  if constexpr (Params::shared_fetch_ids) return true;\n  else return params.pool_capacity != 0;'
assert old.count(key)==1
candidate=old.replace(key,'  return params.pool_capacity != 0;')
for variant,source in [('base',base),('runtime_ids',candidate)]:
    assert not (p/(variant+'.cuh')).exists()
    (p/(variant+'.cuh')).write_text(source)
    shutil.copytree(root/'include',p/('include_'+variant))
    (p/('include_'+variant)/'flashinfer/attention/mla_host_batch.cuh').write_text(source)
for name in ['.gitignore','compile.py','resources.py','resources_gpu3.py','experiment.json','plans.cpp','bench.py','bench_gpu3_native.py','test_candidate.py','candidatepaths.py','screen.py','audit_performance.py']:
    source=((q if name=='screen.py' else w)/name).read_text()
    if name in ['test_candidate.py','candidatepaths.py']:source=source.replace("'cached_ids'","'cached_ids','runtime_ids'")
    if name=='screen.py':source=source.replace('qk_local','runtime_ids')
    (p/name).write_text(source)
source=(q/'compiler_qk.py').read_text().replace('qk_local','runtime_ids').replace('compiler_qk','compiler_runtime')
source=source.replace("assert all('KernelTraitsILb0ELj2ELb0ELj512ELj64ELj64ELj32E' in r['symbol'] for r in added)","assert all('CachedSharedIdsHostBatchParams' in r['symbol'] for r in added)")
(p/'compiler_runtime.py').write_text(source)
(p/'manifest.json').write_text(json.dumps(dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
    predecessor_sha256=hashlib.sha256(old.encode()).hexdigest(),
    mechanism='Only remove constexpr-true cache predicate from cached-only shared-ID type; restore runtimepool_capacity check as in originalfetch_ids. Same dispatch, originaluncached/generic/wholekernels unchanged. No math/emulator/cachepolicy/farbytes changes.',
    candidate_adopted=False,goal_complete=False),indent=2)+'\n')
print((p/'manifest.json').read_text())
