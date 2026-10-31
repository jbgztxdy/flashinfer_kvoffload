"""Move staged cache fill off the fetch pipeline; keep every publication and victim rule."""
import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent
base=(p/'base.cuh').read_text()
assert hashlib.sha256(base.encode()).hexdigest()=='81891229df298b90219410b5a4df1e6c20dc4cb1ee9c0233bd17ad383b29225b'
s=base
a=s.index('__device__ __forceinline__ void fetch_host_batch_pages(')
b=s.index('// Requires the combined [page, 576] layout',a)
part=s[a:b]
old='    if (host_batch_caching(params)) {'
assert part.count(old)==1
part=part.replace(old,'    if (host_batch_caching(params) && !Params::whole_rows) {')
old='''  // Deferred staged chunks: their victims are chosen by now; copy staging -> cache, HBM to HBM,
'''
new='''  if constexpr (Params::whole_rows) {
    // Experiment: do not put victim metadata and cache scatter on the remote-copy
    // pipeline. Re-read the published HBM staging bytes after all copies are done.
    // This adds one HBM read per inserted page; it must earn that traffic in timing.
    if (host_batch_caching(params)) {
#pragma unroll 1
      for (uint32_t c = fetch_id; c * rows < count; c += params.fetch_ctas) {
        if (tid < rows) {
          const uint32_t slot = c * rows + tid;
          int32_t victim = -1;
          if (slot < count) {
            const uint32_t* pool_flag = params.pool_flags + params.slot_pool[slot];
            while (host_batch_ld_acquire(pool_flag) != epoch + 1) __nanosleep(64);
            victim = params.victim_of[slot];
          }
          cache_rows[tid] = victim;
          bind_host_batch_staged_slot(params, slot, victim);
        }
        __syncthreads();
        const auto* source = reinterpret_cast<const uint4*>(
            params.staging + (params.cache_slots + static_cast<int64_t>(c) * rows) * ROW);
#pragma unroll 4
        for (uint32_t v = tid; v < chunk_vecs; v += KTraits::NUM_THREADS) {
          const int32_t victim = cache_rows[v / VECS_PER_ROW];
          if (victim >= 0) {
            reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * ROW)
                [v % VECS_PER_ROW] = source[v];
          }
        }
        __syncthreads();
      }
    }
  }
  // Deferred staged chunks: their victims are chosen by now; copy staging -> cache, HBM to HBM,
'''
assert part.count(old)==1
part=part.replace(old,new)
# Whole-row cache fill above replaces every immediate/deferred cache write in that path.
part=part.replace('for (uint32_t d = 0; d < deferred_count; ++d)', 'for (uint32_t d = 0; !Params::whole_rows && d < deferred_count; ++d)')
s=s[:a]+part+s[b:]
name='defer_cache';target=p/(name+'.cuh');assert not target.exists()
target.write_text(s);shutil.copytree(p/'include_base',p/('include_'+name))
(p/('include_'+name)/'flashinfer/attention/mla_host_batch.cuh').write_text(s)
(p/('manifest_'+name+'.json')).write_text(json.dumps(dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),change='Whole rows only: cache binding/scatter after the remote fetch loop, re-reading published HBM staging; no simulator changes; preserves exact victim selection and publication. Extra HBM traffic/tail are explicit risks.'),indent=2)+'\n')
p=p/'run_readiness.py'
s=p.read_text().replace("'base','warp_ready','row_ready'", "'base','warp_ready','row_ready','defer_cache'")
p.write_text(s)
