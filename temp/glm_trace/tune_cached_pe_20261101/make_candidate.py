"""Keep QPE's 16 packed MMA words per lane across all KV tiles of an item."""
import hashlib
import json
from pathlib import Path
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
jq = p.parent / 'tune_joint_qk_20261101'
wh = p.parent / 'tune_warp_histogram_20261101'
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
base = (jq / 'joint_qk.cuh').read_text()
math = (root / 'include/flashinfer/attention/mla.cuh').read_text()
assert sha(base) == 'a2f37fb985fcbc523e740c85e9cc6468afad74fd8716c11aef4221ebab7c4b7c'
assert sha(math) == '34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
begin = base.index('// Two different KV tiles share each Q fragment.')
end = base.index('// partial_capacity bounds the item states', begin)
clones = base[begin:end]
for old, new in [
    ('host_batch_qk_pair_fragments', 'cached_pe_qk_pair_fragments'),
    ('host_batch_compute_qk_pair', 'cached_pe_compute_qk_pair'),
    ('update_joint_host_mdo', 'cached_pe_update_joint_host_mdo'),
    ('attend_joint_host_tiles', 'cached_pe_attend_joint_host_tiles'),
    ('run_joint_host_item', 'cached_pe_run_joint_host_item'),
    ('BatchMLAJointQKKernel', 'BatchMLACachedPEKernel'),
]:
    clones = clones.replace(old, new)
helper = r'''
// Each lane owns four packed words per 16-feature QPE MMA fragment. The query
// head/feature map is the same map produced by native ldmatrix.x4.
template <typename Params>
__device__ __forceinline__ void cached_pe_load_query(const Params& params, uint32_t row,
                                                   uint32_t (&qpe)[4][4]) {
  const uint32_t lane = threadIdx.x, warp = threadIdx.y;
  const auto* query = params.q_pe + row * params.q_pe_stride_n;
#pragma unroll
  for (uint32_t dim = 0; dim < 4; ++dim) {
    const uint32_t head = warp * 16 + lane / 4, col = dim * 16 + (lane % 4) * 2;
    qpe[dim][0] = *reinterpret_cast<const uint32_t*>(query + head * params.q_pe_stride_h + col);
    qpe[dim][1] = *reinterpret_cast<const uint32_t*>(query + (head + 8) * params.q_pe_stride_h + col);
    qpe[dim][2] = *reinterpret_cast<const uint32_t*>(query + head * params.q_pe_stride_h + col + 8);
    qpe[dim][3] = *reinterpret_cast<const uint32_t*>(query + (head + 8) * params.q_pe_stride_h + col + 8);
  }
}

template <bool PAIR, typename KTraits>
__device__ __forceinline__ void cached_pe_compute_pe(
    typename KTraits::SharedStorage& storage, uint32_t a, uint32_t b,
    uint32_t (&qpe)[4][4], float (*sa)[8], float (*sb)[8]) {
  const uint32_t lane = threadIdx.x, group = threadIdx.z;
  smem_t<KTraits::SWIZZLE_MODE_KPE> ka(storage.kpe_p_smem[a]), kb(storage.kpe_p_smem[b]);
  alignas(16) uint32_t kf[4];
#pragma unroll
  for (uint32_t dim = 0; dim < 4; ++dim) {
    const uint32_t offset = ka.template get_permuted_offset<KTraits::UPCAST_STRIDE_KPE>(
        group * 16 + 8 * (lane / 16) + lane % 8, 2 * dim + (lane % 16) / 8);
    ka.ldmatrix_m8n8x4(offset, kf);
    if (dim == 0)
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(sa[0], qpe[dim], kf);
    else
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sa[0], qpe[dim], kf);
    if constexpr (PAIR) {
      kb.ldmatrix_m8n8x4(offset, kf);
      if (dim == 0)
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(sb[0], qpe[dim], kf);
      else
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sb[0], qpe[dim], kf);
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void cached_pe_compute_one(
    typename KTraits::SharedStorage& storage, uint32_t (&qpe)[4][4], float (*scores)[8]) {
  cached_pe_compute_pe<false, KTraits>(storage, 0, 0, qpe, scores, scores);
  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> q(storage.q_smem_nope);
  smem_t<KTraits::SWIZZLE_MODE_CKV> kv(storage.ckv_smem[0]);
  compute_qk_<false, KTraits, KTraits::NUM_MMA_D_CKV,
      KTraits::UPCAST_STRIDE_Q_NOPE, KTraits::UPCAST_STRIDE_CKV>(q, kv, scores);
}

'''
key = '    float (*sa)[8],float (*sb)[8]) {'
assert clones.count(key) == 1
# Only the wrapper takes a persistent query; the generic CKV pair helper stays literal.
wrapper_begin = clones.index('template <typename KTraits>\n__device__ __forceinline__ void cached_pe_compute_qk_pair(')
wrapper_end = clones.index('// Two native32-row score tiles', wrapper_begin)
wrapper = clones[wrapper_begin:wrapper_end]
wrapper = wrapper.replace('    float (*sa)[8],float (*sb)[8]) {',
                          '    uint32_t (&qpe)[4][4], float (*sa)[8],float (*sb)[8]) {')
pe_begin = wrapper.index('  smem_t<KTraits::SWIZZLE_MODE_Q_PE>')
pe_end = wrapper.index('  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE>', pe_begin)
wrapper = wrapper[:pe_begin] + '  cached_pe_compute_pe<true, KTraits>(storage, a, b, qpe, sa, sb);\n' + wrapper[pe_end:]
clones = clones[:wrapper_begin] + helper + wrapper + clones[wrapper_end:]
key = '    uint32_t kv_len, int host_tiles, float (*scores)[8], float (*o)[8], float* m, float* d) {'
assert clones.count(key) == 1
clones = clones.replace(key, '    uint32_t kv_len, uint32_t row, int host_tiles, float (*scores)[8], float (*o)[8], float* m, float* d) {')
key = '  typename KTraits::IdType pages[kHostBatchTileRows<KTraits>];'
assert clones.count(key) == 1
clones = clones.replace(key, key + '''
  alignas(16) uint32_t qpe[4][4];
  cached_pe_load_query(params, row, qpe);''')
clones = clones.replace('storage, hi%2, lo%2, scores, scores+1);', 'storage, hi%2, lo%2, qpe, scores, scores+1);')
clones = clones.replace('    compute_mla_qk<KTraits>(&storage, 0, scores);', '    cached_pe_compute_one<KTraits>(storage, qpe, scores);')
clones = clones.replace('kv_len, host_tiles, s_frag,', 'kv_len, row, host_tiles, s_frag,')
traits = '''
template <typename Q, typename KV, typename O, typename Id>
struct CachedPETraits : JointQKTraits<Q, KV, O, Id> {};
template <typename Params>
struct CachedPEHostParams : JointQKHostParams<Params> {};

'''
s = base[:end] + traits + clones + base[end:]
start = base.index('    using KV64Traits = JointQKTraits<')
stop = base.index('    DISPATCH_SMEM_CONFIG(', start)
dispatch = base[start:stop].replace('JointQKTraits', 'CachedPETraits').replace('JointQKHostParams', 'CachedPEHostParams').replace('BatchMLAJointQKKernel', 'BatchMLACachedPEKernel').replace('KV64Traits', 'CachedPEKernelTraits').replace('KV64Params', 'CachedPEKernelParams').replace('kv64', 'cached_pe')
s = s.replace(base[start:stop], dispatch + base[start:stop], 1)
for variant, host in [('base', base), ('cached_pe', s)]:
    for name, value in [(variant + '.cuh', host), (variant + '_mla.cuh', math)]:
        f = p / name
        if f.exists(): assert f.read_text() == value, f
        else: f.write_text(value)
    d = p / ('include_' + variant)
    if not d.exists():
        shutil.copytree(root / 'include', d)
        (d / 'flashinfer/attention/mla_host_batch.cuh').write_text(host)
    assert (d / 'flashinfer/attention/mla_host_batch.cuh').read_text() == host
    assert (d / 'flashinfer/attention/mla.cuh').read_text() == math
for name in ['plans.cpp','experiment.json','resources.py','resources_gpu3.py','bench.py','bench_gpu3_native.py','compile.py']:
    f = p / name
    value = (jq / name if name in ['plans.cpp','experiment.json'] else wh / name).read_bytes()
    if f.exists(): assert f.read_bytes() == value
    else: f.write_bytes(value)
for name in ['candidatepaths.py','test_candidate.py','stress_candidate.py','restore_includes.py','verify_machine_words.py']:
    value = (wh / name).read_text().replace('warp_hist', 'cached_pe')
    if name == 'stress_candidate.py': value = value.replace('test_histogram_lru.py','test_cofetch_progress.py')
    if not (p / name).exists(): (p / name).write_text(value)
old = p.parent / 'tune_lane_denominator_20261101'
if not (p / 'test_cofetch_progress.py').exists():
    (p / 'test_cofetch_progress.py').write_bytes((old / 'test_cofetch_progress.py').read_bytes())
value = (wh / 'compiler_warp_hist.py').read_text().replace('warp_hist', 'cached_pe').replace('WarpHistogramHostParams','CachedPEHostParams')
(p / 'compiler_cached_pe.py').write_text(value)
(p / '.gitignore').write_text('/include_*/\n/jit_*/\n/__pycache__/\n/plans.so\n!/*.log\n!/failures/*.log\n')
m = dict(base_sha256=sha(base),candidate_sha256=sha(s),math_sha256=sha(math),
    candidate_adopted=False,goal_complete=False,
    mechanism='Load QPE64 into16 packed MMA words/lane once per attention item; reuse across all paired and odd KV tiles. SamePEthenCKV MMA order, jointMDO/PV/refill, 64heads/256threads/2stage/CTA32/f24, fetch/LRU/math/shared layout.',
    distinction='N9 read QPE global anew per tile as part of a288thread service-warp design. This caches QPE across the whole item and keeps the original256thread dedicated fetch architecture. No unroll/tile/grid/fetch sweep.',
    tradeoff='Sixteen additional long-lived packed words may increase registers/spills. Original shared QPE allocation/load is retained to isolate reuse and keep147968B layout; no memory-capacity/occupancy gain claimed.',
    invariants='All576features and native score accumulation order, exactLRU/cachepayload, simulator and references unchanged. No new kernel/barrier. Only original large cached fully-staged JQ dispatch.')
(p / 'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps(m,indent=2))
