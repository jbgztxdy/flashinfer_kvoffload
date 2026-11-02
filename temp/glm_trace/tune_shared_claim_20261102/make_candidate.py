import difflib,hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];old=p.parent/'tune_elastic_fetch_20261102'
sha=lambda data:hashlib.sha256(data).hexdigest()
b=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes()
assert sha(b)=='754706f8d8fba14a361ce6dcfc64a358446e5df36b225f392fa8a3e7756ded79'
assert not(p/'shared_claim.cuh').exists()
s=b.decode();math=(root/'include/flashinfer/attention/mla.cuh').read_bytes()
start=s.index('template <typename Q, typename KV, typename O, typename Id>\nstruct StreamSourceTraits');end=s.index('// partial_capacity bounds',start)
section=s[start:end].replace('StreamSource','SharedClaim').replace('stream_source_','shared_claim_')
section=section.replace('fetch_shared_claim_pages','fetch_stream_source_pages')
section=section.replace('prepare_host_batch_rows<KTraits>(params, epoch + 1)','prepare_shared_claim_rows<KTraits>(params, epoch + 1, smem)')
section=section.replace('resolve_host_batch_slots<KTraits>','resolve_shared_claim_slots<KTraits>')
assert section.count('prepare_shared_claim_rows<KTraits>')==2
rstart=s.rindex('template <typename KTraits, typename Params>',0,s.index('__device__ __forceinline__ uint32_t resolve_host_batch_slots('))
rend=s.index('// After the grid barrier, when prepare dropped padding',rstart)
resolve=s[rstart:rend].replace('resolve_host_batch_slots(','resolve_shared_claim_slots(')
oldblock='''      if (owner == row) {
        params.unique_pages[slot] = page;
        // (In place with the cache prepare already marked the claims; this loop does not run.)
        if (host_batch_caching(params) && slot < params.staging_capacity) {
          params.slot_pool[slot] = params.row_pool[row];
        }
      }
'''
assert resolve.count(oldblock)==1;resolve=resolve.replace(oldblock,'')
needle='  if (unique == 0) return 0;'
extra='''
  for(uint32_t row=blockIdx.y;row<params.num_rows;row+=gridDim.y) {
    const uint32_t begin=params.kv_indptr[row],count=row_claims[row],base=row_base[row];
    for(uint32_t rank=tid;rank<count;rank+=THREADS) {
      const uint32_t slot=base+rank;
      params.unique_pages[slot]=params.entry_victim[begin+rank];
      if(slot<params.staging_capacity)params.slot_pool[slot]=params.row_pool[row];
    }
  }
'''
assert resolve.count(needle)==1;resolve=resolve.replace(needle,needle+extra)
at=s.index('    using StreamSourceKernelTraits =');dend=s.index('    using CachedPEKernelTraits =',at)
dispatch=s[at:dend].replace('StreamSource','SharedClaim').replace('stream_source','shared_claim')
dispatch=dispatch.replace('const bool shared_claim = params.stage','const bool shared_claim = max_kv_len > 0 && max_kv_len <= 2048 && params.stage')
s=s[:end]+(p/'prepare.inc').read_text()+'\n'+resolve+section+s[end:]
at=s.index('    using StreamSourceKernelTraits =');s=s[:at]+dispatch+s[at:]
for v,data in [('base',b),('shared_claim',s.encode())]:
 (p/(v+'.cuh')).write_bytes(data);(p/(v+'_mla.cuh')).write_bytes(math)
 tree=p/('include_'+v);shutil.copytree(root/'include',tree)
 (tree/'flashinfer/attention/mla_host_batch.cuh').write_bytes(data)
for name in ['bench_gpu3_native.py','resources_gpu3.py','resources.py','experiment.json','plans.cpp','compile.py','audit_performance.py']:
 shutil.copy2(old/name,p/name)
for name in ['candidatepaths.py','build_candidate.py','run_sxm.py','bench_sxm.py']:
 (p/name).write_text((old/name).read_text().replace('elastic_fetch','shared_claim').replace(",gpu3_rejection='preflight_rejection.json'",''))
test=(old/'test_elastic_fetch.py').read_text().replace('elastic_fetch','shared_claim')
test=test.replace('    for step in range(3):','    if length==65:w.set_host_cache(torch.tensor([0,3,16,31,rows],dtype=torch.int32),512)\n    for step in range(3):')
test=test.replace('        selected[0]=-1;', '        selected[8:16]=selected[:8].clone()\n        selected[0]=-1;')
(p/'test_shared_claim.py').write_text(test)
(p/'candidate.diff').write_text(''.join(difflib.unified_diff(b.decode().splitlines(True),s.splitlines(True),fromfile='base.cuh',tofile='shared_claim.cuh')))
record=dict(base_sha256=sha(b),candidate_sha256=sha(s.encode()),math_sha256=sha(math),
 mechanism='Shared hash dedup within up to4same-pool rows before global CAS; bounded16384key hash for maxK2048. Global unique owners ranked at group firstrow, page IDs temporarily in disjoint entry_victim ranges before LRU; original dense rowprefix resolve/fetch/LRU preserved.',
 invariants='Group split at every cachepool boundary; global CAS still dedups crossgroup/crosspool. Hit stamps before original gridbarrier; no inline wait/newbarrier/fetchjob binarysearch. Identical attention math and sourcefetch/FAR/bytes.',adopted=False)
(p/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
