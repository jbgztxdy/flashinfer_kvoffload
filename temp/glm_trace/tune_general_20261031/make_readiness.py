"""Create immutable, isolated headers for readiness experiments; never edit production."""
import hashlib
import json
import shutil
from pathlib import Path
p=Path(__file__).resolve().parent
base=(p/'base.cuh').read_text()
assert hashlib.sha256(base.encode()).hexdigest()=='81891229df298b90219410b5a4df1e6c20dc4cb1ee9c0233bd17ad383b29225b'
s=base
old='''                                                   const uint32_t stage_idx,
                                                   const uint32_t index_base) {'''
new='''                                                   const uint32_t stage_idx,
                                                   const uint32_t index_base,
                                                   const uint32_t* scratch) {'''
assert s.count(old)==1
s=s.replace(old,new)
old='''    resolve_host_batch_row<KTraits>(params, pages[j], ckv_ptr, kpe_ptr);
'''
new=old+'''    if constexpr (Params::whole_rows) {
      // Each 8-lane row group already holds its page id. Its leader acquires the
      // staging publication, then releases the other loaders in the same warp.
      // Cached and near entries need no publication. No additional CTA barrier.
      if (lane_idx % 8 == 0 && valid && pages[j] < 0) {
        const uint32_t staged = static_cast<uint32_t>(-(pages[j] + 1));
        if (staged >= params.cache_slots) {
          const uint32_t* flag = host_batch_flags(params) +
              (staged - params.cache_slots) / scratch[kHostBatchScratchFetchRows];
          while (host_batch_ld_acquire(flag) != scratch[kHostBatchScratchEpoch] + 1)
            __nanosleep(64);
        }
      }
      __syncwarp();
    }
'''
assert s.count(old)==1
s=s.replace(old,new)
a=s.index('__device__ __forceinline__ void attend_host_batch_tiles(')
b=s.index('// Exclusive scan of one value per thread',a)
part=s[a:b]
assert part.count('wait_host_batch_tile<KTraits>')==3
part=part.replace('wait_host_batch_tile<KTraits>', 'if constexpr (!Params::whole_rows) wait_host_batch_tile<KTraits>')
assert part.count('index_base);')==3
part=part.replace('index_base);','index_base, scratch);')
s=s[:a]+part+s[b:]
name='warp_ready'
target=p/(name+'.cuh')
assert not target.exists()
target.write_text(s)
shutil.copytree(p/'include_base',p/('include_'+name))
(p/('include_'+name)/'flashinfer/attention/mla_host_batch.cuh').write_text(s)
(p/('manifest_'+name+'.json')).write_text(json.dumps(dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),change='Whole-row only: reuse per-thread page ids; 8-lane row leaders acquire staging flags, syncwarp before cp.async; remove duplicate warp0 waits. Generic path unchanged.'),indent=2)+'\n')
