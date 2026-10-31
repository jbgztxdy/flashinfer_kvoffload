"""Share a chunk's sparse source IDs instead of reloading them for every vector."""
import hashlib
import json
import shutil
from pathlib import Path
p = Path(__file__).resolve().parent
base = (p / 'base.cuh').read_text()
assert hashlib.sha256(base.encode()).hexdigest() == '52919c286398d4668933a772c646bdccd46273a53aafeb8d5c1b14e2d9bff2c5'
first = base.index('__device__ __forceinline__ void fetch_host_batch_pages')
last = base.index('// Requires the combined', first)
prefix, f, suffix = base[:first], base[first:last], base[last:]
old = '  __shared__ int32_t cache_rows[kHostBatchFetchRows];'
assert f.count(old) == 1
f = f.replace(old, old + '\n  __shared__ typename Params::IdType page_ids[BUFFERS][kHostBatchFetchRows];')
old = '  auto issue = [&](uint32_t chunk, uint32_t buf) {\n#pragma unroll'
assert f.count(old) == 1
f = f.replace(old, '''  auto issue = [&](uint32_t chunk, uint32_t buf) {
    if constexpr (Params::whole_rows && KTraits::CTA_TILE_KV == 32) {
      if (tid < rows && chunk * rows + tid < count) {
        page_ids[buf][tid] = params.unique_pages[chunk * rows + tid];
      }
      // This synchronizes the ordinary ID stores without draining the other buffer's
      // outstanding KV cp.async group. Every thread then reuses one shared ID per row.
      __syncthreads();
    }
#pragma unroll''')
old = '''        const auto* source = reinterpret_cast<const uint4*>(
            params.ckv + static_cast<int64_t>(params.unique_pages[slot]) * ROW);'''
assert f.count(old) == 1
f = f.replace(old, '''        const auto page = [&]() {
          if constexpr (Params::whole_rows && KTraits::CTA_TILE_KV == 32)
            return page_ids[buf][v / VECS_PER_ROW];
          else return params.unique_pages[slot];
        }();
        const auto* source = reinterpret_cast<const uint4*>(
            params.ckv + static_cast<int64_t>(page) * ROW);''')
old = '''      params.host_slot[params.unique_pages[chunk * rows + tid] -
                       params.host_page_start] = 0;'''
assert f.count(old) == 1
f = f.replace(old, '''      const auto page = [&]() {
        if constexpr (Params::whole_rows && KTraits::CTA_TILE_KV == 32)
          return page_ids[buf][tid];
        else return params.unique_pages[chunk * rows + tid];
      }();
      params.host_slot[page - params.host_page_start] = 0;''')
# The immediate scatter can also use the IDs already read. Deferred chunks retain
# the original global lookup because the shared ring may have wrapped by then.
old = '          bind_host_batch_staged_slot(params, slot, victim);'
assert f.count(old) == 1
f = f.replace(old, '''          if constexpr (Params::whole_rows && KTraits::CTA_TILE_KV == 32) {
            if (victim >= 0) {
              const auto page = page_ids[buf][tid];
              const int32_t evicted = params.slot_page[victim];
              if (evicted >= 0) params.resident[evicted - params.host_page_start] = -1;
              params.resident[page - params.host_page_start] = victim;
              params.slot_page[victim] = page;
            }
          } else {
            bind_host_batch_staged_slot(params, slot, victim);
          }''')
s = prefix + f + suffix
v = 'fetch_ids'
assert not (p / (v + '.cuh')).exists()
(p / (v + '.cuh')).write_text(s)
shutil.copytree(p / 'include_base', p / ('include_' + v))
(p / ('include_' + v) / 'flashinfer/attention/mla_host_batch.cuh').write_text(s)
(p / ('manifest_' + v + '.json')).write_text(json.dumps(dict(
    base_sha256=hashlib.sha256(base.encode()).hexdigest(), candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),
    mechanism='Whole-row CTA32 only: one coalesced source-ID load per chunk into a shared double ring, reused by vector cp.async issue, dedup clear and immediate cache binding. Same far bytes, waits, publication, immediate shared-buffer scatter and LRU policy; deferred chunks retain original lookup.'), indent=2) + '\n')
for name in ['test_candidate.py','candidatepaths.py','compiler.py']:
    path = p / name
    text = path.read_text().replace("'prebind'", "'prebind','fetch_ids'")
    path.write_text(text)
path = p / 'screen.py'
text = path.read_text().replace("choices=['prebind']", "choices=['prebind', 'fetch_ids']")
text = text.replace("['base', 'prebind']", "['base', 'prebind', 'fetch_ids']")
path.write_text(text)
print((p / ('manifest_' + v + '.json')).read_text())
