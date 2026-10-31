"""Freeze the integrated-fetch prototype and its one-stage attention control."""
import hashlib
import json
from pathlib import Path
import re
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
previous = p.parent/'tune_fetch_warp_20261101'
base = (root/'include/flashinfer/attention/mla_host_batch.cuh').read_text()
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
assert sha(base) == 'b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
s = base.replace('  static constexpr bool shared_fetch_ids = false;',
    '  static constexpr bool shared_fetch_ids = false;\n  static constexpr bool cofetch = false;')
key = 'constexpr uint32_t kHostBatchFetchRows = 64;'
assert s.count(key) == 1
s = s.replace(key, '''template <typename Params>
struct CoFetchHostBatchParams : CachedSharedIdsHostBatchParams<Params> {
  static constexpr bool cofetch = true;
};
template <typename Params>
struct StageOneHostBatchParams : CachedSharedIdsHostBatchParams<Params> {};

// Padding preserves the original dedicated fetcher's 64-row double buffers in
// the one-stage control. It is unused by attention.
template <typename Base>
struct PaddedStageOneHostBatchTraits : Base {
  struct SharedStorage : Base::SharedStorage {
    alignas(16) uint8_t fetch_padding[36864];
  };
};

''' + key)
key = '''  const uint32_t count = params.stage ? min(unique, params.staging_capacity) : unique;
  const uint32_t rounds = params.fetch_ctas <= 8 ? 3 : 1;'''
assert s.count(key) == 1
s = s.replace(key, '''  if constexpr (Params::cofetch) return 16;
''' + key)
key = '// Warp 0 waits until every staged entry of a KV tile has landed;'
assert s.count(key) == 1
s = s.replace(key, (p/'cofetch_helpers.inc').read_text()+'\n'+key)
key = '''  static_assert(KTraits::CTA_TILE_KV <= 64);
  if (threadIdx.y != 0 || threadIdx.z != 0) return;'''
assert s.count(key) == 1
s = s.replace(key, '''  if constexpr (Params::cofetch) {
    wait_host_batch_tile_cofetch<KTraits>(params, scratch, index_base, kv_len, kv_base);
    return;
  }
''' + key)
start = s.index('__device__ __forceinline__ void attend_host_batch_tiles(')
end = s.index('// Exclusive scan of one value per thread over the CTA;', start)
body = s[start:end]
for key in ['''  for (; kv_tile_idx >= mask_tile_idx && kv_tile_idx > 0; --kv_tile_idx) {''',
            '''  for (; kv_tile_idx + 1 > NUM_STAGES; --kv_tile_idx) {''',
            '''  for (; kv_tile_idx >= 0; --kv_tile_idx) {''']:
    assert body.count(key) == 1
    body = body.replace(key, key+'''
    if constexpr (Params::cofetch)
      if ((kv_tile_idx & 7) == 0) step_host_batch_cofetch<KTraits>(params);''')
s = s[:start]+body+s[end:]
key = '  const int lru_items = host_batch_caching(params) && params.stage && unique ? params.num_pools : 0;'
assert s.count(key) == 1
s = s.replace(key, '  const int lru_items = !Params::cofetch && host_batch_caching(params) && params.stage && unique ? params.num_pools : 0;')
key = '''  if (unique > 0 && from_back && !host_batch_writes_back(params)) fetch_host_batch_pages<KTraits>(params, smem, epoch, unique, fetch_rows);
  run_host_batch_items<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, from_back, unique);'''
assert s.count(key) == 1
s = s.replace(key, '''  if constexpr (Params::cofetch) {
    if (unique > 0) {
      // No service buffers are live while a pool uses dynamic shared memory.
      for (uint32_t pool = blockIdx.y; pool < params.num_pools; pool += gridDim.y) {
        run_host_batch_lru<KTraits>(params, smem, pool, epoch, unique);
        __syncthreads();
      }
      cg::this_grid().sync();
    }
    init_host_batch_cofetch<KTraits>(params, unique, epoch);
  } else {
''' + key.split('\n')[0] + '''
  }
  run_host_batch_items<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, from_back, unique);
  if constexpr (Params::cofetch) finish_host_batch_cofetch<KTraits>(params);''')
key = '''      if (whole_rows) {
        static_assert(sizeof(WholeParams) == sizeof(Params));'''
assert s.count(key) == 1
s = s.replace(key, '''      bool selected_one_stage = false;
      using OneStageTraits = KernelTraits<false, 1, true, HEAD_DIM_CKV, HEAD_DIM_KPE,
                                         64, 32, DTypeQ, DTypeKV, DTypeO, IdType>;
''' + key)
key = '            kernel = reinterpret_cast<decltype(kernel)>(BatchMLAHostBatchKernel<KTraits, SharedParams, false>);'
assert s.count(key) == 1
marker = '''
            if (params.num_rows >= 4u * num_sms) {
              using ServiceParams = CoFetchHostBatchParams<Params>;
              static_assert(sizeof(ServiceParams) == sizeof(Params));
              static_assert(alignof(ServiceParams) == alignof(Params));
              kernel = reinterpret_cast<decltype(kernel)>(BatchMLAHostBatchKernel<OneStageTraits, ServiceParams, false>);
              selected_one_stage = true;
            }'''
s = s.replace(key, key+marker)
key = '      const size_t smem_size = sizeof(typename KTraits::SharedStorage);'
assert s.count(key) == 1
s = s.replace(key, '''      const size_t smem_size = selected_one_stage
          ? sizeof(typename OneStageTraits::SharedStorage) : sizeof(typename KTraits::SharedStorage);''')
control = s.replace('using ServiceParams = CoFetchHostBatchParams<Params>;',
                    'using ServiceParams = StageOneHostBatchParams<Params>;')
control = control.replace('BatchMLAHostBatchKernel<OneStageTraits, ServiceParams, false>',
    'BatchMLAHostBatchKernel<PaddedStageOneHostBatchTraits<OneStageTraits>, ServiceParams, false>')
control = control.replace('? sizeof(typename OneStageTraits::SharedStorage)',
    '? sizeof(typename PaddedStageOneHostBatchTraits<OneStageTraits>::SharedStorage)')
for variant, source in [('base', base), ('cofetch', s), ('stage1_control', control)]:
    assert not (p/(variant+'.cuh')).exists()
    (p/(variant+'.cuh')).write_text(source)
    shutil.copytree(root/'include', p/('include_'+variant))
    (p/('include_'+variant)/'flashinfer/attention/mla_host_batch.cuh').write_text(source)
for name in ['.gitignore', 'compile.py', 'resources.py', 'resources_gpu3.py', 'experiment.json',
             'plans.cpp', 'bench.py', 'bench_gpu3_native.py', 'test_candidate.py',
             'candidatepaths.py', 'screen.py', 'audit_performance.py']:
    text = (previous/name).read_text().replace("'warp_rows'", "'cofetch','stage1_control'")
    if name == 'screen.py':
        text = text.replace("default=['cofetch','stage1_control', 'base']", "default=['cofetch', 'base']")
        text = text.replace("default='cofetch','stage1_control'", "default='cofetch'")
    if name == 'experiment.json':
        obj = json.loads(text)
        obj.update(status='candidate_in_progress', candidate='cofetch with stage1_control')
        text = json.dumps(obj, indent=2)+'\n'
    (p/name).write_text(text)
for variant, symbol in [('cofetch','CoFetchHostBatchParams'),('stage1_control','StageOneHostBatchParams')]:
    text = (previous/'compiler_warp_rows.py').read_text().replace('warp_rows',variant)
    text = text.replace('WarpRowsHostBatchParams',symbol)
    (p/('compiler_'+variant+'.py')).write_text(text)
    text = (previous/'probe_dispatch.py').read_text().replace('warp_rows',variant)
    text = text.replace('WarpRowsHostBatchParams',symbol).replace('per-warp fetch-row path',variant+' path')
    text = text.replace("p/'dispatch_probe.json'", "p/('dispatch_probe_"+variant+".json')")
    if variant == 'cofetch': text = text.replace('147968','111104')
    (p/('probe_'+variant+'.py')).write_text(text)
manifest = dict(base_sha256=sha(base), candidate_sha256=sha(s), control_sha256=sha(control),
    mechanism='All CTAs service cyclic disjoint 16-row double-buffered fetch queues during one-stage attention; each eighth KV tile services one chunk, waits make fetch progress, every CTA drains before exit. LRU pools selected first with one uniform grid barrier. Single cooperative kernel; canonical math and emulator unchanged.',
    control='One-stage attention with padding retaining original 64-row dedicated fetch double buffers; same routing threshold, no integrated service.',
    service_payload_bytes=36864, attention_dynamic_bytes=111104, original_attention_dynamic_bytes=147968,
    progress='Every false tile-ready check advances the CTA local queue; exhausted owners spin only after their queues are drained; idle or finished attention CTAs drain all remaining fetches. LRU is complete before any service is issued.',
    candidate_adopted=False, goal_complete=False)
(p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
