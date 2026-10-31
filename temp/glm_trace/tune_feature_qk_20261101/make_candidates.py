"""Generate 2-stage/16-KV feature-partitioned QK and a duplicated-QK layout control."""
import hashlib
import json
from pathlib import Path
import re
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
previous = p.parent/'tune_hybrid_cofetch_20261101'
base = (root/'include/flashinfer/attention/mla_host_batch.cuh').read_text()
math = (root/'include/flashinfer/attention/mla.cuh').read_text()
private_math = (p.parent/'tune_heads_20261031/half_heads_mla.cuh').read_text()
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
assert sha(base) == 'b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
assert sha(math) == '34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
assert sha(private_math) == 'e5968e7605bb657de30e5600c4a8fc09b977a3e2e7163f64e46718de519289a3'
s = base
key = 'constexpr uint32_t kHostBatchFetchRows = 64;'
assert s.count(key) == 1
s = s.replace(key, '''template <typename Q, typename KV, typename O, typename Id, bool FEATURE>
struct SmallKVFeatureHostTraits : KernelTraits<false, 2, false, 512, 64, 32, 16, Q, KV, O, Id> {
  using Base = KernelTraits<false, 2, false, 512, 64, 32, 16, Q, KV, O, Id>;
  static constexpr uint32_t NUM_THREADS = 128;
  static constexpr bool FEATURE_QK = FEATURE;
  struct SharedStorage : Base::SharedStorage {
    // Same allocation/padding in the control even though it does not exchange scores.
    alignas(16) float qk_exchange[2][2][8][32];
  };
};
template <typename Params>
struct SmallKVFeatureHostParams : CachedSharedIdsHostBatchParams<Params> {
  static constexpr uint32_t num_head_tiles = 2;
};

''' + key)
s = s.replace('  constexpr uint32_t CHUNK_VECS = kHostBatchFetchRows * VECS_PER_ROW;',
    '  constexpr uint32_t CHUNK_VECS = (KTraits::CTA_TILE_Q == 32 ? 32u : kHostBatchFetchRows) * VECS_PER_ROW;')
key = 'constexpr uint32_t kHostBatchTileRows = KTraits::NUM_MMA_KV == 1 ? 1 : KTraits::NUM_MMA_KV / 2;'
assert s.count(key) == 1
s = s.replace(key, '''constexpr uint32_t kHostBatchTileRows = KTraits::CTA_TILE_Q == 32
    ? KTraits::CTA_TILE_KV / 16 : (KTraits::NUM_MMA_KV == 1 ? 1 : KTraits::NUM_MMA_KV / 2);''')
key = '  return KTraits::NUM_MMA_KV == 1 ? row : row + (threadIdx.z + j * 2) * 16;'
assert s.count(key) == 1
s = s.replace(key, '''  if constexpr (KTraits::CTA_TILE_Q == 32) return row + (threadIdx.z + j * 2) * 8;
  else return KTraits::NUM_MMA_KV == 1 ? row : row + (threadIdx.z + j * 2) * 16;''')
key = '  if (KTraits::NUM_MMA_KV == 1 && threadIdx.z != 0) return;'
assert s.count(key) == 2  # KV load and the in-place writeback helper
s = s.replace(key, '  if (KTraits::CTA_TILE_Q != 32 && KTraits::NUM_MMA_KV == 1 && threadIdx.z != 0) return;')
key = '''  static_assert(KTraits::CTA_TILE_KV <= 64);
  if (threadIdx.y != 0 || threadIdx.z != 0) return;'''
assert s.count(key) == 1
s = s.replace(key, key+'''
  if constexpr (KTraits::CTA_TILE_Q == 32)
    if (threadIdx.x >= KTraits::CTA_TILE_KV) return;''')
key = '// The native MLA tile loop over the KV tiles of one item, last tile first.'
assert s.count(key) == 1
s = s.replace(key, (p/'feature_helpers.inc').read_text()+'\n'+key)
# Only the host-batch loop calls the new selector. Pure-device math is untouched.
assert s.count('compute_mla_qk<KTraits>(&storage, kv_tile_idx % NUM_STAGES, s_frag);') == 3
s = s.replace('compute_mla_qk<KTraits>(&storage, kv_tile_idx % NUM_STAGES, s_frag);',
              'host_batch_compute_feature_qk<KTraits>(storage, kv_tile_idx % NUM_STAGES, s_frag);')
key = '  const uint32_t fetch_rows = host_batch_fetch_rows<ROW_MERGE>(params, unique);'
assert s.count(key) == 1
s = s.replace(key, '''  const uint32_t fetch_rows = KTraits::CTA_TILE_Q == 32
      ? (unique < params.fetch_ctas * 32u ? 16u : 32u)
      : host_batch_fetch_rows<ROW_MERGE>(params, unique);''')
key = '    DISPATCH_SMEM_CONFIG(smem_limit_per_sm, NUM_STAGES, CTA_TILE_KV, QK_SHARD, {'
assert s.count(key) == 1
s = s.replace(key, '''    using SmallTraits = SmallKVFeatureHostTraits<DTypeQ, DTypeKV, DTypeO, IdType, true>;
    using SmallParams = SmallKVFeatureHostParams<Params>;
    constexpr size_t small_smem = sizeof(typename SmallTraits::SharedStorage);
    static_assert(small_smem == 78080);
    const bool small_kv = params.stage && !params.host_split && params.num_head_tiles == 1 &&
        params.pool_capacity > 0 && params.num_rows >= 4u * num_sms && params.fetch_ctas >= 1 &&
        params.fetch_ctas < num_sms && smem_limit_per_sm >= 2 * (small_smem + 2048) &&
        smem_limit_per_sm < 200000 && static_cast<uint64_t>(params.staging_capacity) >=
            static_cast<uint64_t>(params.num_rows) * max_kv_len;
    if (small_kv) {
      auto small_kernel = BatchMLAHostBatchKernel<SmallTraits, SmallParams, false>;
      FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(small_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, small_smem));
      int small_blocks = 0;
      FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&small_blocks, small_kernel, 128, small_smem));
      if (small_blocks >= 2) {
        const uint32_t small_grid = small_blocks * num_sms;
        if (params.pool_capacity > kHostBatchMaxPool || 2ull * params.pool_capacity * sizeof(uint32_t) > small_smem)
          return cudaErrorInvalidConfiguration;
        static_assert(sizeof(SmallParams) == sizeof(Params));
        static_assert(alignof(SmallParams) == alignof(Params));
        params.full_rows = params.num_rows;
        params.partial_capacity = partial_capacity;
        params.num_chunks = 1;
        const uint32_t length = max(max_kv_len, 1u);
        params.prepare_parts = min(ceil_div(small_grid, max(params.num_rows, 1u)), ceil_div(length, 128u));
        params.prepare_segment = round_up(ceil_div(length, params.prepare_parts), 128u);
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(cudaLaunchCooperativeKernel((void*)small_kernel, dim3(1, small_grid), dim3(32, 2, 2), args, small_smem, stream));
        return cudaSuccess;
      }
    }
''' + key)
control = s.replace('SmallKVFeatureHostTraits<DTypeQ, DTypeKV, DTypeO, IdType, true>',
                    'SmallKVFeatureHostTraits<DTypeQ, DTypeKV, DTypeO, IdType, false>')
for variant, host, mla in [('base',base,math),('dim_split',s,private_math),('duplicate_control',control,private_math)]:
    assert not (p/(variant+'.cuh')).exists()
    (p/(variant+'.cuh')).write_text(host)
    (p/(variant+'_mla.cuh')).write_text(mla)
    shutil.copytree(root/'include', p/('include_'+variant))
    (p/('include_'+variant)/'flashinfer/attention/mla_host_batch.cuh').write_text(host)
    (p/('include_'+variant)/'flashinfer/attention/mla.cuh').write_text(mla)
for name in ['.gitignore','compile.py','resources.py','resources_gpu3.py','experiment.json','plans.cpp',
             'bench.py','bench_gpu3_native.py','test_candidate.py','candidatepaths.py','screen.py',
             'stress_candidate.py','test_cofetch_progress.py','audit_performance.py']:
    text = (previous/name).read_text()
    text = text.replace("'hybrid','hybrid_static'", "'dim_split','duplicate_control'")
    text = text.replace("['hybrid', 'hybrid_static']", "['dim_split', 'duplicate_control']")
    text = text.replace("['base', 'hybrid', 'hybrid_static']", "['base', 'dim_split', 'duplicate_control']")
    text = text.replace("default=['hybrid', 'base']", "default=['dim_split', 'base']")
    text = text.replace("default='hybrid'", "default='dim_split'")
    if name == 'experiment.json':
        obj = json.loads(text);obj.update(candidate='dim_split and duplicate_control',status='candidate_in_progress')
        text = json.dumps(obj,indent=2)+'\n'
    if name in ['bench.py', 'bench_gpu3_native.py']:
        marker = '    files["include/flashinfer/attention/mla_host_batch.cuh"]=digest(VARIANT_INCLUDE/"flashinfer/attention/mla_host_batch.cuh")'
        assert text.count(marker) == 1
        text = text.replace(marker, marker+'\n    files["include/flashinfer/attention/mla.cuh"]=digest(VARIANT_INCLUDE/"flashinfer/attention/mla.cuh")')
        old_source = '((VARIANT_INCLUDE/"flashinfer/attention/mla_host_batch.cuh") if name=="include/flashinfer/attention/mla_host_batch.cuh" else (ROOT/name))'
        new_source = '((VARIANT_INCLUDE/name.removeprefix("include/")) if name in ("include/flashinfer/attention/mla_host_batch.cuh","include/flashinfer/attention/mla.cuh") else (ROOT/name))'
        assert text.count(old_source) == 1
        text = text.replace(old_source, new_source)
    if name == 'test_cofetch_progress.py':
        text = text.replace("@pytest.mark.parametrize('capacity', [512, 4096])", "@pytest.mark.parametrize('capacity', [512, 4096])\n@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float16])")
        text = text.replace('def test_long_fetch_queues_and_cache_payload(capacity):', 'def test_long_fetch_queues_and_cache_payload(capacity, dtype):')
        text = text.replace("device='cuda', dtype=torch.bfloat16", "device='cuda', dtype=dtype")
        text = text.replace('_plan(rows, length, indices)', '_plan(rows, length, indices, dtype=dtype)')
    if name == 'stress_candidate.py': text = text.replace('tests=2,', 'tests=4,')
    if name == 'audit_performance.py':
        text = text.replace("t['tests'] == 2", "t['tests'] == 4").replace("'2 passed'", "'4 passed'")
        text = text.replace("tests=2, sha256=sha(f)", "tests=4, sha256=sha(f)")
        key2 = "            assert floor['source_sha256']['include/flashinfer/attention/mla.cuh']==d['source_sha256']['include/flashinfer/attention/mla.cuh']"
        assert text.count(key2) == 1
        text = text.replace(key2, '''            reference_math = floor['source_sha256']['include/flashinfer/attention/mla.cuh']
            executed_math = d['source_sha256']['include/flashinfer/attention/mla.cuh']
            if reference_math != executed_math:
                assert reference_math == '34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
                assert executed_math == 'e5968e7605bb657de30e5600c4a8fc09b977a3e2e7163f64e46718de519289a3'
                proof = load(p/('compiler_'+cfg['variant']+'_normal_summary.json'))
                assert proof['complete'] and proof['unchanged_original_kernels'] == 68
                assert proof['math_header_sha256'] == executed_math
                assert proof['candidate_header_sha256'] == d['source_sha256']['include/flashinfer/attention/mla_host_batch.cuh']
                assert proof['base_header_sha256'] == 'b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
''')
    (p/name).write_text(text)
for variant in ['dim_split','duplicate_control']:
    text = (previous/'compiler_hybrid_static.py').read_text().replace('hybrid_static',variant)
    text = text.replace('StaticHybridCoFetchHostBatchParams','SmallKVFeatureHostParams')
    (p/('compiler_'+variant+'.py')).write_text(text)
    text = (previous/'probe_hybrid_static.py').read_text().replace('hybrid_static',variant)
    text = text.replace('StaticHybridCoFetchHostBatchParams','SmallKVFeatureHostParams')
    text = text.replace('[1,108,1]', '[1,216,1]').replace('[32,4,2]', '[32,2,2]')
    text = text.replace("'active_blocks_per_SM']==1", "'active_blocks_per_SM']==2").replace('147968','78080')
    text = text.replace("root/'include/flashinfer/attention/mla.cuh'", "p/'include_"+variant+"/flashinfer/attention/mla.cuh'")
    (p/('probe_'+variant+'.py')).write_text(text)
manifest = dict(base_sha256=sha(base),candidate_sha256=sha(s),control_sha256=sha(control),
    base_math_sha256=sha(math),candidate_math_sha256=sha(private_math),
    mechanism='SM80 cached whole-row >=4SM only: two32-head items,128threads,2KVstages/16rows. Warp groups partition each QK dot by feature, exchange4096B partialscores, then original nonsharded softmax/PV. Attention equation unchanged; FP32 reduction order differs.',
    control='Same2stage/16KV/32head layout and4096B padding, native duplicated QK.',
    history='Old half_heads was1stage/32KV, old duplicate-QK was64head/32KV and repeated all features. This keeps2stages and removes feature duplication.',
    expected_dynamic_shared_bytes=78080,expected_active_blocks_per_SM=2,
    candidate_adopted=False,goal_complete=False)
(p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
