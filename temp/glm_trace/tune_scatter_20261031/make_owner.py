"""Preserve the prepare CAS owner through staged remap; avoid per-reference atomics."""
import hashlib
import json
import shutil
from pathlib import Path
p = Path(__file__).resolve().parent
root = p.parents[2]
base = (p / 'base.cuh').read_text()
assert hashlib.sha256(base.encode()).hexdigest() == '52919c286398d4668933a772c646bdccd46273a53aafeb8d5c1b14e2d9bff2c5'

def once(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)

s = once(base, '    if (host_batch_caching(params)) {\n      bool ready = true;',
    '    if (!Params::whole_rows && host_batch_caching(params)) {\n      bool ready = true;')
# In the temporary positive Host-page remap, bit30 marks the original successful CAS.
# After resolve, clear bit30 of the negative staged entry to preserve that one owner.
split = s.index('// prepare_host_batch_rows without staging')
prefix = once(s[:split],
    '          params.remap[begin + host_first + __popc(host_masks[i] & lower)] = pages[i];',
    '''          IdType entry = pages[i];
          if constexpr (Params::whole_rows) {
            if (host_batch_caching(params) && (fresh_masks[i] >> lane & 1)) entry |= kHostBatchOwnerBit;
          }
          params.remap[begin + host_first + __popc(host_masks[i] & lower)] = entry;''')
s = prefix + s[split:]
resolve_first = s.index('__device__ __forceinline__ uint32_t resolve_host_batch_slots')
resolve_last = s.index('// After the grid barrier, when prepare dropped padding', resolve_first)
resolve = s[resolve_first:resolve_last]
resolve = once(resolve, '      const IdType page = params.remap[begin + j];', '''      const IdType marked = params.remap[begin + j];
      const IdType page = Params::whole_rows ? marked & ~kHostBatchOwnerBit : marked;''')
resolve = once(resolve, '        params.remap[begin + j] = host_batch_staged_entry(params, slot);', '''        IdType entry = host_batch_staged_entry(params, slot);
        if constexpr (Params::whole_rows) {
          if (marked & kHostBatchOwnerBit) entry &= ~kHostBatchOwnerBit;
        }
        params.remap[begin + j] = entry;''')
s = s[:resolve_first] + resolve + s[resolve_last:]
s = once(s, '  const bool staged = page < 0;\n  const int64_t row =', '''  const bool staged = page < 0;
  if constexpr (Params::whole_rows) {
    if (staged) page |= kHostBatchOwnerBit;
  }
  const int64_t row =''')
s = once(s, '    const typename KTraits::IdType page = kv_idx < kv_len ? params.remap[index_base + kv_idx] : 0;',
    '''    typename KTraits::IdType page = kv_idx < kv_len ? params.remap[index_base + kv_idx] : 0;
    if constexpr (Params::whole_rows) {
      if (page < 0) page |= kHostBatchOwnerBit;
    }''')
s = once(s, '    if (host_batch_writes_back(params) && lane_idx % 8 == 0) {', '''    if constexpr (Params::whole_rows) {
      // Only the prepare CAS owner writes a unique staged page to its LRU victim.
      // Every other reference uses the same staging bytes without claiming metadata.
      if (host_batch_caching(params) && lane_idx % 8 == 0) {
        int32_t victim = -1;
        if (valid && pages[j] < 0 && !(pages[j] & kHostBatchOwnerBit)) {
          const uint32_t row = static_cast<uint32_t>(-((pages[j] | kHostBatchOwnerBit) + 1));
          const uint32_t staged = row - params.cache_slots;
          const uint32_t* flag = params.pool_flags + params.slot_pool[staged];
          while (host_batch_ld_acquire(flag) != params.state[kHostBatchEpoch] + 1) __nanosleep(64);
          victim = params.victim_of[staged];
          bind_host_batch_staged_slot(params, staged, victim);
        }
        host_batch_tile_victims<KTraits>()[stage_idx * KTraits::CTA_TILE_KV + smem_row] = victim;
      }
    } else if (host_batch_writes_back(params) && lane_idx % 8 == 0) {''')
s = once(s, '  if (!host_batch_writes_back(params)) return;', '''  if constexpr (Params::whole_rows) {
    if (!host_batch_caching(params)) return;
  } else {
    if (!host_batch_writes_back(params)) return;
  }''')
s = once(s, 'const bool whole_rows = params.stage && !params.host_split && params.num_head_tiles == 1 &&',
    '''const bool whole_rows = params.stage && !params.host_split && params.num_head_tiles == 1 &&
          static_cast<uint64_t>(params.cache_slots) + params.staging_capacity <= kHostBatchOwnerBit &&''')
v = 'owner_cache'
assert not (p / (v + '.cuh')).exists()
(p / (v + '.cuh')).write_text(s)
shutil.copytree(p / 'include_base', p / ('include_' + v))
(p / ('include_' + v) / 'flashinfer/attention/mla_host_batch.cuh').write_text(s)
(p / ('manifest_' + v + '.json')).write_text(json.dumps(dict(
    base_sha256=hashlib.sha256(base.encode()).hexdigest(), candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),
    mechanism='Prepare CAS owner survives staged remap via reserved bit30. Only that attention tile fills the cache from shared memory; no per-reference atomic claim. Index-space bound checked on host. Generic paths unchanged.'), indent=2) + '\n')
print((p / ('manifest_' + v + '.json')).read_text())
