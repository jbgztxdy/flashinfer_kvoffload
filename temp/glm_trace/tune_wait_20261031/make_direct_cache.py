"""Avoid a duplicate HBM write for misses admitted into an LRU cache slot.

Ready flags remain indexed by unique staged page. A ready admitted page lives in
its chosen cache victim; an unadmitted page still lives in ordinary staging.
"""
import hashlib
import json
import shutil
from pathlib import Path
p = Path(__file__).resolve().parent
base = (p / 'base.cuh').read_text()
assert hashlib.sha256(base.encode()).hexdigest() == '52919c286398d4668933a772c646bdccd46273a53aafeb8d5c1b14e2d9bff2c5'
s = base
old = '''    if (BUFFERS == 2 && next * rows < count) {
      issue(next, (k + 1) % 2);
      cp_async::wait_group<1>();
    } else {
      cp_async::wait_group<0>();
    }'''
assert s.count(old) == 1
s = s.replace(old, '''    const bool ahead = BUFFERS == 2 && next * rows < count;
    if (ahead) issue(next, (k + 1) % 2);
    if constexpr (Params::whole_rows && KTraits::CTA_TILE_KV == 32) {
      if (host_batch_caching(params)) {
        if (tid < rows) {
          const uint32_t slot = chunk * rows + tid;
          int32_t victim = -1;
          if (slot < count) {
            const uint32_t* flag = params.pool_flags + params.slot_pool[slot];
            while (host_batch_ld_acquire(flag) != epoch + 1) __nanosleep(64);
            victim = params.victim_of[slot];
          }
          cache_rows[tid] = victim;
          bind_host_batch_staged_slot(params, slot, victim);
        }
        __syncthreads();
      }
    }
    if (ahead) cp_async::wait_group<1>();
    else cp_async::wait_group<0>();''')
old = '        destination[v] = buffer[buf * CHUNK_VECS + v];'
assert s.count(old) == 1
s = s.replace(old, '''        if constexpr (Params::whole_rows && KTraits::CTA_TILE_KV == 32) {
          const int32_t victim = host_batch_caching(params) ? cache_rows[v / VECS_PER_ROW] : -1;
          if (victim >= 0) {
            reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * ROW)
                [v % VECS_PER_ROW] = buffer[buf * CHUNK_VECS + v];
          } else destination[v] = buffer[buf * CHUNK_VECS + v];
        } else destination[v] = buffer[buf * CHUNK_VECS + v];''')
old = '    if (host_batch_caching(params)) {\n      bool ready = true;'
assert s.count(old) == 1
s = s.replace(old, '''    if (!(Params::whole_rows && KTraits::CTA_TILE_KV == 32) && host_batch_caching(params)) {
      bool ready = true;''')
old = '''  const int64_t row = staged ? -static_cast<int64_t>(page) - 1 : (page & ~kHostBatchOwnerBit);
  ckv_ptr ='''
assert s.count(old) == 1
s = s.replace(old, '''  int64_t row = staged ? -static_cast<int64_t>(page) - 1 : (page & ~kHostBatchOwnerBit);
  if constexpr (Params::whole_rows && KTraits::CTA_TILE_KV == 32) {
    if (host_batch_caching(params) && staged && row >= params.cache_slots) {
      // The existing tile wait acquired this staging chunk's publication before any
      // load reaches here. Its victim metadata and bytes are therefore both final.
      const int32_t victim = params.victim_of[row - params.cache_slots];
      if (victim >= 0) row = victim;
    }
  }
  ckv_ptr =''')
v = 'direct_cache'
assert not (p / (v + '.cuh')).exists()
(p / (v + '.cuh')).write_text(s)
shutil.copytree(p / 'include_base', p / ('include_' + v))
(p / ('include_' + v) / 'flashinfer/attention/mla_host_batch.cuh').write_text(s)
(p / ('manifest_' + v + '.json')).write_text(json.dumps(dict(
    base_sha256=hashlib.sha256(base.encode()).hexdigest(), candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),
    mechanism='Whole-row CTA32 only: admitted misses go directly shared->cache victim, other misses shared->staging; one HBM store per unique page. Existing chunk flag release/acquire guards both metadata and bytes; attention maps admitted staging entries to victim after that acquire. No changing policy/capacity/far bytes/simulator/reference, one kernel.'), indent=2) + '\n')
for name in ['test_candidate.py','candidatepaths.py','compiler.py']:
    path = p / name
    text = path.read_text().replace("'fetch_ids'", "'fetch_ids','direct_cache'")
    path.write_text(text)
path = p / 'screen.py'
text = path.read_text().replace("'fetch_ids']", "'fetch_ids', 'direct_cache']")
path.write_text(text)
print((p / ('manifest_' + v + '.json')).read_text())
