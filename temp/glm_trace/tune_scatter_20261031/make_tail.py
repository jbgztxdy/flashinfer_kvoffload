"""Split only one final compute wave while preserving the leading whole rows."""
import hashlib
import json
import shutil
from pathlib import Path
p = Path(__file__).resolve().parent
base = (p / 'base.cuh').read_text()
assert hashlib.sha256(base.encode()).hexdigest() == '52919c286398d4668933a772c646bdccd46273a53aafeb8d5c1b14e2d9bff2c5'
s = base
mark = 'constexpr uint32_t kHostBatchFetchRows = 64;'
assert s.count(mark) == 1
s = s.replace(mark, '''// The last SM wave uses two existing partial items per row; leading rows stay direct.
template <typename Params>
struct TailWaveHostBatchParams : FullyStagedHostBatchParams<Params> {
  static constexpr bool host_split = false;
  static constexpr uint32_t num_chunks = 2;
};

''' + mark)
mark = '      auto kernel = row_merge ? BatchMLAHostBatchKernel<KTraits, Params, true>'
assert s.count(mark) == 1
s = s.replace(mark, '''      using TailParams = TailWaveHostBatchParams<Params>;
      const bool split_last_wave = whole_rows && CTA_TILE_KV == 32 &&
          params.num_rows >= 4u * num_sms && partial_capacity >= 3u * num_sms;
''' + mark)
mark = '        kernel = reinterpret_cast<decltype(kernel)>(BatchMLAHostBatchKernel<KTraits, WholeParams, false>);'
assert s.count(mark) == 1
s = s.replace(mark, '''        kernel = split_last_wave
            ? reinterpret_cast<decltype(kernel)>(BatchMLAHostBatchKernel<KTraits, TailParams, false>)
            : reinterpret_cast<decltype(kernel)>(BatchMLAHostBatchKernel<KTraits, WholeParams, false>);''')
mark = '''      if (whole_rows) {
        params.full_rows=params.num_rows;
        tail_tiles=0;
      }'''
assert s.count(mark) == 1
s = s.replace(mark, '''      if (whole_rows) {
        params.full_rows = split_last_wave ? params.num_rows - min(params.num_rows, grid)
                                          : params.num_rows;
        tail_tiles = params.num_rows - params.full_rows;
      }''')
mark = '      void* args[] = {(void*)&params};'
assert s.count(mark) == 1
s = s.replace(mark, '      if (split_last_wave) params.num_chunks = 2;\n' + mark)
v = 'tail_wave2'
assert not (p / (v + '.cuh')).exists()
(p / (v + '.cuh')).write_text(s)
shutil.copytree(p / 'include_base', p / ('include_' + v))
(p / ('include_' + v) / 'flashinfer/attention/mla_host_batch.cuh').write_text(s)
(p / ('manifest_' + v + '.json')).write_text(json.dumps(dict(
    base_sha256=hashlib.sha256(base.encode()).hexdigest(), candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),
    mechanism='For already whole-row, CTA_TILE_KV=32, >=4SM launches with enough partial capacity: retain all but one SM wave as direct whole rows; split only final grid rows into two existing attention items, then existing merge. New fully-staged specialization; generic and original whole math unchanged.'), indent=2) + '\n')
print((p / ('manifest_' + v + '.json')).read_text())
