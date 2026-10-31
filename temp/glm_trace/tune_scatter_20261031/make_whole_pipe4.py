"""Combine the retained whole-row path with a four-buffer, 32-row fetch pipeline.

The historical fastpipe4 header has no WholeRowHostBatchParams specialization. Its
measurements do not test this combination. Preserve the old six generic kernels.
"""
import hashlib
import json
import shutil
from pathlib import Path
p = Path(__file__).resolve().parent
base = (p / 'base.cuh').read_text()
old_path = p.parent / 'tune_100gb_20261030/fastpipe4.cuh'
old = old_path.read_text()
assert 'WholeRowHostBatchParams' not in old
assert hashlib.sha256(base.encode()).hexdigest() == '52919c286398d4668933a772c646bdccd46273a53aafeb8d5c1b14e2d9bff2c5'
pos = old.index('__device__ __forceinline__ void fetch_host_batch_pages')
first = old.rfind('template <typename KTraits, typename Params>', 0, pos)
last = old.index('// Requires the combined', pos)
helper = old[first:last].replace('fetch_host_batch_pages', 'fetch_whole_pipe4_pages')
helper = helper.replace('kHostBatchFetchRows', '32u')
anchor = 'template <typename KTraits, typename Params>\n__device__ __forceinline__ void fetch_host_batch_pages'
assert base.count(anchor) == 1
s = base.replace(anchor, helper + anchor)
before = '  const uint32_t fetch_rows = host_batch_fetch_rows<ROW_MERGE>(params, unique);'
assert s.count(before) == 1
s = s.replace(before, '''  const uint32_t fetch_rows = Params::whole_rows && KTraits::CTA_TILE_KV == 32
      ? (unique < (params.fetch_ctas <= 8 ? 3u : 1u) * params.fetch_ctas * 32u ? 16u : 32u)
      : host_batch_fetch_rows<ROW_MERGE>(params, unique);''')
before = '  if (unique > 0 && from_back && !host_batch_writes_back(params)) fetch_host_batch_pages<KTraits>(params, smem, epoch, unique, fetch_rows);'
assert s.count(before) == 1
s = s.replace(before, '''  if (unique > 0 && from_back && !host_batch_writes_back(params)) {
    if constexpr (Params::whole_rows && KTraits::CTA_TILE_KV == 32) {
      fetch_whole_pipe4_pages<KTraits>(params, smem, epoch, unique, fetch_rows);
    } else {
      fetch_host_batch_pages<KTraits>(params, smem, epoch, unique, fetch_rows);
    }
  }''')
v = 'whole_pipe4'
assert not (p / (v + '.cuh')).exists()
(p / (v + '.cuh')).write_text(s)
shutil.copytree(p / 'include_base', p / ('include_' + v))
(p / ('include_' + v) / 'flashinfer/attention/mla_host_batch.cuh').write_text(s)
(p / ('manifest_' + v + '.json')).write_text(json.dumps(dict(
    base_sha256=hashlib.sha256(base.encode()).hexdigest(), candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),
    historical_helper_sha256=hashlib.sha256(old.encode()).hexdigest(),
    historical_helper_path=str(old_path.relative_to(p.parents[2])),
    mechanism='Only whole-row CTA_TILE_KV=32 uses 32-row/four-buffer fetch; immediate cache fill stays in fetch. Math, retained dispatch, generic kernels and emulator source unchanged. Historical fastpipe4 lacked whole-row specialization; this combination is unmeasured.'), indent=2) + '\n')
print((p / ('manifest_' + v + '.json')).read_text())
