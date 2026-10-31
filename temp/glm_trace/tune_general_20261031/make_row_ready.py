"""Compare row-level readiness hoisting with the existing tile-level publication waits."""
import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent
base=(p/'base.cuh').read_text()
assert hashlib.sha256(base.encode()).hexdigest()=='81891229df298b90219410b5a4df1e6c20dc4cb1ee9c0233bd17ad383b29225b'
s=base
a=s.index('__device__ __forceinline__ void attend_host_batch_tiles(')
b=s.index('// Exclusive scan of one value per thread',a)
part=s[a:b]
assert part.count('wait_host_batch_tile<KTraits>')==3
part=part.replace('wait_host_batch_tile<KTraits>', 'if constexpr (!Params::whole_rows) wait_host_batch_tile<KTraits>')
s=s[:a]+part+s[b:]
old='''    // load_q's copies join the first KV group, as in the native kernel.
    attend_host_batch_tiles<KTraits>'''
new='''    // Whole-row experiment: acquire the row's publications before entering the
    // compute pipeline. This trades resident-prefix overlap for a simpler tile loop.
    if constexpr (Params::whole_rows) {
#pragma unroll 1
      for (uint32_t tile = 0; tile < host_tiles; ++tile)
        wait_host_batch_tile<KTraits>(params, scratch, params.kv_indptr[row], kv_len,
                                     tile * CTA_TILE_KV);
    }
    // load_q's copies join the first KV group, as in the native kernel.
    attend_host_batch_tiles<KTraits>'''
assert s.count(old)==1
s=s.replace(old,new)
name='row_ready'
target=p/(name+'.cuh');assert not target.exists()
target.write_text(s)
shutil.copytree(p/'include_base',p/('include_'+name))
(p/('include_'+name)/'flashinfer/attention/mla_host_batch.cuh').write_text(s)
(p/('manifest_'+name+'.json')).write_text(json.dumps(dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),change='Whole-row only: acquire every Host tile publication once before attend_host_batch_tiles; remove per-tile waits from the compute pipeline. May lose resident-prefix overlap; benchmark before adoption.'),indent=2)+'\n')
