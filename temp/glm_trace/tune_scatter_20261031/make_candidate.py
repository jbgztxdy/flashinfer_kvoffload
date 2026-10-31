"""Freeze a new shared-tile cache scatter candidate; never overwrite a build input."""
import hashlib
import json
import shutil
from pathlib import Path
p = Path(__file__).resolve().parent
root = p.parents[2]
previous = p.parent / 'tune_general_20261031'
sha = lambda b: hashlib.sha256(b).hexdigest()
base = (root / 'include/flashinfer/attention/mla_host_batch.cuh').read_text()
assert sha(base.encode()) == '52919c286398d4668933a772c646bdccd46273a53aafeb8d5c1b14e2d9bff2c5'

def replace_once(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)

s = replace_once(base, '    if (host_batch_caching(params)) {\n      bool ready = true;',
    '    if (!Params::whole_rows && host_batch_caching(params)) {\n      bool ready = true;')
anchor = '''    if (host_batch_writes_back(params) && lane_idx % 8 == 0) {'''
s = replace_once(s, anchor, '''    if constexpr (Params::whole_rows) {
      // A whole-row attention item consumes every valid entry exactly once. Duplicate
      // references arbitrate on the unique staged page's victim; one tile copies its bytes.
      // LRU items have no waits and precede attention items in every worker's walk, so
      // waiting on another pool cannot prevent that pool's selection from running.
      if (host_batch_caching(params) && lane_idx % 8 == 0) {
        int32_t victim = -1;
        const uint32_t row = static_cast<uint32_t>(-(pages[j] + 1));
        if (valid && pages[j] < 0 && row >= params.cache_slots) {
          const uint32_t staged = row - params.cache_slots;
          const uint32_t* flag = params.pool_flags + params.slot_pool[staged];
          const uint32_t stamp = params.state[kHostBatchEpoch] + 1;
          while (host_batch_ld_acquire(flag) != stamp) __nanosleep(64);
          victim = atomicExch(params.victim_of + staged, -1);
          bind_host_batch_staged_slot(params, staged, victim);
        }
        host_batch_tile_victims<KTraits>()[stage_idx * KTraits::CTA_TILE_KV + smem_row] = victim;
      }
    } else if (host_batch_writes_back(params) && lane_idx % 8 == 0) {''')
s = replace_once(s, '  if (!host_batch_writes_back(params)) return;', '''  if constexpr (Params::whole_rows) {
    if (!host_batch_caching(params)) return;
  } else {
    if (!host_batch_writes_back(params)) return;
  }''')
for name in ['.gitignore', 'bench.py', 'bench_gpu3_native.py', 'resources.py', 'resources_gpu3.py',
             'experiment.json', 'plans.cpp', 'compile.py', 'test_candidate.py', 'candidatepaths.py']:
    content = (previous / name).read_bytes()
    if name == 'candidatepaths.py':
        content = content.decode().replace("'base','defer_cache','warp_ready','row_ready','threshold2','threshold2_sm80','balanced_sm80','balanced_cached_sm80'", "'base','attention_cache'").encode()
    if name == 'test_candidate.py':
        content = content.decode().replace("'base','defer_cache','warp_ready','row_ready','threshold2','threshold2_sm80','balanced_sm80','balanced_cached_sm80'", "'base','attention_cache'").encode()
    target = p / name
    assert not target.exists()
    target.write_bytes(content)
for variant, text in [('base', base), ('attention_cache', s)]:
    header = p / (variant + '.cuh')
    assert not header.exists()
    header.write_text(text)
    shutil.copytree(root / 'include', p / ('include_' + variant))
    (p / ('include_' + variant) / 'flashinfer/attention/mla_host_batch.cuh').write_text(text)
(p / 'manifest.json').write_text(json.dumps(dict(base_sha256=sha(base.encode()),
    candidate_sha256=sha(s.encode()), mechanism='Whole-row staged cache fill from attention shared KV tiles. One atomic victim claim per unique page; fetch only stages/publishes. All other GPU paths and the retained host dispatch unchanged.',
    retained_production='ce341a02', full_goal_complete=False), indent=2) + '\n')
print((p / 'manifest.json').read_text())
