"""Generate a frozen per-warp fetch-row experiment from retained b6 production."""
import hashlib
import json
from pathlib import Path
import re
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
previous = p.parent / 'tune_qk_pair_20261031'
base = (root / 'include/flashinfer/attention/mla_host_batch.cuh').read_text()
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
assert sha(base) == 'b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
s = base.replace('  static constexpr bool shared_fetch_ids = false;',
    '  static constexpr bool shared_fetch_ids = false;\n  static constexpr bool warp_fetch_rows = false;')
assert s != base
key = 'constexpr uint32_t kHostBatchFetchRows = 64;'
assert s.count(key) == 1
s = s.replace(key, '''template <typename Params>
struct WarpRowsHostBatchParams : CachedSharedIdsHostBatchParams<Params> {
  static constexpr bool warp_fetch_rows = true;
};

''' + key)
begin = s.index('__device__ __forceinline__ void fetch_host_batch_pages(')
end = s.index('// Requires the combined [page, 576] layout', begin)
body = s[begin:end]
issue_begin = body.index('#pragma unroll\n    for (uint32_t i = 0; i < CHUNK_VECS / KTraits::NUM_THREADS; ++i)')
issue_end = body.index('    FAR_EMU_TILE(2, buf,', issue_begin)
native = body[issue_begin:issue_end]
warp_issue = '''    if constexpr (Params::warp_fetch_rows) {
      // Each warp resolves one row, then reuses its base for all 72 vector copies.
      // All lanes keep participating in the original emulation hook below.
#pragma unroll
      for (uint32_t r = tid / 32; r < kHostBatchFetchRows; r += KTraits::NUM_THREADS / 32) {
        if (r < rows && chunk * rows + r < count) {
          const auto page = page_ids[buf][r];
          const auto* source = reinterpret_cast<const uint4*>(
              params.ckv + static_cast<int64_t>(page) * ROW);
#pragma unroll
          for (uint32_t v = tid % 32; v < VECS_PER_ROW; v += 32) {
            cp_async::load_128b<PrefetchMode::kNoPrefetch>(
                buffer + buf * CHUNK_VECS + r * VECS_PER_ROW + v, source + v);
          }
        }
      }
    } else {
''' + native + '    }\n'
body = body[:issue_begin] + warp_issue + body[issue_end:]
old = '''#pragma unroll
    for (uint32_t i = 0; i < CHUNK_VECS / KTraits::NUM_THREADS; ++i) {
      const uint32_t v = i * KTraits::NUM_THREADS + tid;
      if (params.stage && v < chunk_vecs && chunk * rows + v / VECS_PER_ROW < count) {
        destination[v] = buffer[buf * CHUNK_VECS + v];
      }
    }
'''
assert body.count(old) == 1
body = body.replace(old, '''    if constexpr (Params::warp_fetch_rows) {
#pragma unroll
      for (uint32_t r = tid / 32; r < kHostBatchFetchRows; r += KTraits::NUM_THREADS / 32) {
        if (params.stage && r < rows && chunk * rows + r < count) {
#pragma unroll
          for (uint32_t v = tid % 32; v < VECS_PER_ROW; v += 32) {
            destination[r * VECS_PER_ROW + v] =
                buffer[buf * CHUNK_VECS + r * VECS_PER_ROW + v];
          }
        }
      }
    } else {
''' + old + '    }\n')
old = '''#pragma unroll
        for (uint32_t i = 0; i < CHUNK_VECS / KTraits::NUM_THREADS; ++i) {
          const uint32_t v = i * KTraits::NUM_THREADS + tid;
          const int32_t victim = v < chunk_vecs ? cache_rows[v / VECS_PER_ROW] : -1;
          if (victim >= 0) {
            reinterpret_cast<uint4*>(
                params.staging +
                static_cast<int64_t>(victim) * ROW)[v % VECS_PER_ROW] =
                buffer[buf * CHUNK_VECS + v];
          }
        }
'''
assert body.count(old) == 1
body = body.replace(old, '''        if constexpr (Params::warp_fetch_rows) {
#pragma unroll
          for (uint32_t r = tid / 32; r < kHostBatchFetchRows; r += KTraits::NUM_THREADS / 32) {
            const int32_t victim = r < rows ? cache_rows[r] : -1;
            if (victim >= 0) {
              auto* cached = reinterpret_cast<uint4*>(
                  params.staging + static_cast<int64_t>(victim) * ROW);
#pragma unroll
              for (uint32_t v = tid % 32; v < VECS_PER_ROW; v += 32) {
                cached[v] = buffer[buf * CHUNK_VECS + r * VECS_PER_ROW + v];
              }
            }
          }
        } else {
''' + old + '        }\n')
s = s[:begin] + body + s[end:]
key = '            kernel = reinterpret_cast<decltype(kernel)>(BatchMLAHostBatchKernel<KTraits, SharedParams, false>);'
assert s.count(key) == 1
s = s.replace(key, key + '''
            if (params.num_rows >= 4u * num_sms) {
              using WarpParams = WarpRowsHostBatchParams<Params>;
              static_assert(sizeof(WarpParams) == sizeof(Params));
              static_assert(alignof(WarpParams) == alignof(Params));
              kernel = reinterpret_cast<decltype(kernel)>(BatchMLAHostBatchKernel<KTraits, WarpParams, false>);
            }''')
for variant, source in [('base', base), ('warp_rows', s)]:
    assert not (p / (variant + '.cuh')).exists()
    (p / (variant + '.cuh')).write_text(source)
    shutil.copytree(root / 'include', p / ('include_' + variant))
    (p / ('include_' + variant) / 'flashinfer/attention/mla_host_batch.cuh').write_text(source)
for name in ['.gitignore', 'compile.py', 'resources.py', 'resources_gpu3.py', 'experiment.json',
             'plans.cpp', 'bench.py', 'bench_gpu3_native.py', 'test_candidate.py',
             'candidatepaths.py', 'screen.py', 'audit_performance.py']:
    text = (previous / name).read_text()
    if name in ['test_candidate.py', 'candidatepaths.py']:
        text = re.sub(r"\('base','attention_cache'.*?'qk_pair_prefetch'\)", "('base','warp_rows')", text)
    if name == 'screen.py':
        text = text.replace("default=['prebind', 'base']", "default=['warp_rows', 'base']")
        text = text.replace("default='prebind'", "default='warp_rows'")
        text = text.replace("['prebind', 'fetch_ids', 'direct_cache', 'cached_ids','qk_pair','qk_pair_prefetch']", "['warp_rows']")
        text = text.replace("['base', 'prebind', 'fetch_ids', 'direct_cache', 'cached_ids','qk_pair','qk_pair_prefetch']", "['base', 'warp_rows']")
    if name == 'experiment.json':
        obj = json.loads(text)
        obj.update(status='candidate_in_progress', candidate='warp_rows', ours_fetch_ctas=24)
        obj.pop('completed_at', None)
        text = json.dumps(obj, indent=2) + '\n'
    (p / name).write_text(text)
text = (previous / 'compiler_pair.py').read_text().replace('qk_pair', 'warp_rows')
text = text.replace('PairedQKHostBatchParams', 'WarpRowsHostBatchParams').replace('compiler_pair', 'compiler_warp_rows')
(p / 'compiler_warp_rows.py').write_text(text)
text = (previous / 'probe_dispatch.py').read_text().replace('qk_pair', 'warp_rows')
text = text.replace('PairedQKHostBatchParams', 'WarpRowsHostBatchParams').replace('half-head path', 'per-warp fetch-row path')
(p / 'probe_dispatch.py').write_text(text)
manifest = dict(base_sha256=sha(base), candidate_sha256=sha(s),
    mechanism='Only large cached whole-row CTA32 fetch issue/staging store/cache store use one warp per row, reusing its page/victim base across 72 vectors. Attention math/loop, double buffers, original chunk size/flags/fences, far-byte reservation hooks, LRU selection/binding, deferred copies unchanged.',
    tradeoff='72 vectors require three warp instructions with 8/32 lanes in the final instruction, versus contiguous-vector native mapping. Address reuse may not compensate for the extra issued instructions.',
    historical_difference='Not readiness waiting, extra fetch buffers, row/head repartition, or paired/duplicate QK. Source review found no previous per-warp row transfer candidate.',
    candidate_adopted=False, goal_complete=False)
(p / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(manifest)
