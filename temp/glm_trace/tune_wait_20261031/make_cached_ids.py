"""Confine ID sharing to cached whole rows, keeping every original kernel intact."""
import hashlib
import json
import shutil
from pathlib import Path
p=Path(__file__).resolve().parent
base=(p/'base.cuh').read_text()
source=(p/'fetch_ids.cuh').read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='270c48be82e1d15b70c461b00bac3fe7a124bc75f0b85323960d7e372c4edee9'
s=source.replace('Params::whole_rows && KTraits::CTA_TILE_KV == 32','Params::shared_fetch_ids')
old='  static constexpr bool whole_rows = false;'
assert s.count(old)==1
s=s.replace(old,old+'\n  static constexpr bool shared_fetch_ids = false;')
old='constexpr uint32_t kHostBatchFetchRows = 64;'
assert s.count(old)==1
s=s.replace(old,'''template <typename Params>
struct CachedSharedIdsHostBatchParams : WholeRowHostBatchParams<Params> {
  static constexpr bool shared_fetch_ids = true;
};

'''+old)
old='  return params.pool_capacity != 0;'
assert s.count(old)==1
s=s.replace(old,'''  if constexpr (Params::shared_fetch_ids) return true;
  else return params.pool_capacity != 0;''')
old='''        kernel = reinterpret_cast<decltype(kernel)>(BatchMLAHostBatchKernel<KTraits, WholeParams, false>);'''
assert s.count(old)==1
s=s.replace(old,old+'''
        if constexpr (CTA_TILE_KV == 32) {
          if (params.pool_capacity > 0) {
            using SharedParams = CachedSharedIdsHostBatchParams<Params>;
            static_assert(sizeof(SharedParams)==sizeof(Params));
            static_assert(alignof(SharedParams)==alignof(Params));
            kernel = reinterpret_cast<decltype(kernel)>(BatchMLAHostBatchKernel<KTraits, SharedParams, false>);
          }
        }''')
v='cached_ids'
assert not (p/(v+'.cuh')).exists()
(p/(v+'.cuh')).write_text(s)
shutil.copytree(p/'include_base',p/('include_'+v))
(p/('include_'+v)/'flashinfer/attention/mla_host_batch.cuh').write_text(s)
(p/('manifest_'+v+'.json')).write_text(json.dumps(dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),
    mechanism='Cache-enabled whole-row CTA32 gets a new ID-sharing specialization; every original generic and whole-row kernel remains compile-time shared_fetch_ids=false. No-LRU calls always use original kernels. Cached work reuses IDs in issue/dedup/cache-binding. Same capacity/policy/traffic/math/simulator.'),indent=2)+'\n')
for name in ['test_candidate.py','candidatepaths.py']:
    f=p/name;s=f.read_text().replace("'direct_cache'","'direct_cache','cached_ids'");f.write_text(s)
f=p/'screen.py';s=f.read_text().replace("'direct_cache']","'direct_cache', 'cached_ids']");f.write_text(s)
print((p/('manifest_'+v+'.json')).read_text())
