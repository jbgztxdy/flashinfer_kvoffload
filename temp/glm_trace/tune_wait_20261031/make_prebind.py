"""Overlap LRU victim binding with the current asynchronous far page load.

Unlike rejected late cache fill, this keeps one shared-memory scatter, no reread,
and preserves every ready-flag/emulator wait before staged bytes are published.
"""
import hashlib
import json
import shutil
from pathlib import Path
p = Path(__file__).resolve().parent
base = (p / 'base.cuh').read_text()
assert hashlib.sha256(base.encode()).hexdigest() == '52919c286398d4668933a772c646bdccd46273a53aafeb8d5c1b14e2d9bff2c5'
first = base.index('    if (host_batch_caching(params)) {\n      bool ready = true;')
last = base.index('    if (BUFFERS == 1 && next * rows < count) issue(next, 0);', first)
old_cache = base[first:last]
meta_end = old_cache.index('''#pragma unroll
        for (uint32_t i = 0; i < CHUNK_VECS''')
metadata = old_cache[:meta_end]
assert metadata.endswith('        __syncthreads();\n')
# The outer cache guard and defer/else block close after the victim setup; scattering
# stays after staging publication, using cache_rows prepared above the async wait.
metadata += '        early_fill = true;\n      }\n    }\n'
scatter = old_cache[meta_end:old_cache.index('''      }
      // The buffer is refilled next.''')]
assert scatter.strip().endswith('}')
before = '''    if (BUFFERS == 2 && next * rows < count) {
      issue(next, (k + 1) % 2);
      cp_async::wait_group<1>();
    } else {
      cp_async::wait_group<0>();
    }'''
assert base.count(before) == 1
early = '''    const bool ahead = BUFFERS == 2 && next * rows < count;
    if (ahead) issue(next, (k + 1) % 2);
    bool early_fill = false;
    if constexpr (Params::whole_rows && KTraits::CTA_TILE_KV == 32) {
'''+metadata+'''    }
    if (ahead) cp_async::wait_group<1>();
    else cp_async::wait_group<0>();'''
s = base.replace(before, early)
assert s.count(old_cache) == 1
s = s.replace(old_cache, '''    if constexpr (Params::whole_rows && KTraits::CTA_TILE_KV == 32) {
      if (early_fill) {
'''+scatter+'''      }
      if (host_batch_caching(params)) __syncthreads();
    } else {
'''+old_cache+'''    }
''')
v = 'prebind'
assert not (p / (v + '.cuh')).exists()
(p / (v + '.cuh')).write_text(s)
shutil.copytree(p / 'include_base', p / ('include_' + v))
(p / ('include_' + v) / 'flashinfer/attention/mla_host_batch.cuh').write_text(s)
(p / ('manifest_' + v + '.json')).write_text(json.dumps(dict(
    base_sha256=hashlib.sha256(base.encode()).hexdigest(), candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),
    mechanism='Whole-row CTA32 only: select/check/bind already-chosen LRU victims while cp.async far loads are outstanding. Stage bytes and original emulator wait/publish retain order; cache scatters once from same shared buffer after publication. Initial not-ready pool chunks still defer. No extra HBM read, no changes to math/cache policy/simulator/reference.'), indent=2) + '\n')
q = p.parent / 'tune_scatter_20261031'
for name in ['test_candidate.py', 'candidatepaths.py']:
    text = (q / name).read_text()
    text = text.replace("'tail_wave2'", "'tail_wave2','prebind'")
    (p / name).write_text(text)
(p / 'compile.py').write_bytes((q / 'compile.py').read_bytes())
print((p / ('manifest_' + v + '.json')).read_text())
