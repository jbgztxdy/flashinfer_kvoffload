"""Full eight-warp attention plus one independent service warp, within one resident CTA."""
import hashlib
import json
from pathlib import Path
import re
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
jq = p.parent / 'tune_joint_qk_20261101'
base = (jq / 'joint_qk.cuh').read_text()
math = (root / 'include/flashinfer/attention/mla.cuh').read_text()
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
assert sha(base) == 'a2f37fb985fcbc523e740c85e9cc6468afad74fd8716c11aef4221ebab7c4b7c'
assert sha(math) == '34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
def function(source, name):
    m = re.search(r'\b(?:void|bool|uint32_t|uint2)\s+' + re.escape(name) + r'\(', source)
    assert m, name
    start = source.rfind('template <', 0, m.start())
    a = source.index('{', m.end()); depth = 1; b = a + 1
    while depth:
        depth += (source[b] == '{') - (source[b] == '}'); b += 1
    return source[start:b]
mathnames = ['load_q', 'compute_qk_', 'logits_mask_', 'update_mdo_states_', 'compute_mla_pv', 'normalize_d_', 'write_o']
hostnames = ['wait_host_batch_tile', 'host_batch_tile_row', 'read_host_batch_pages', 'load_host_batch_kv',
             'host_batch_thread_scan', 'run_host_batch_lru', 'claim_host_batch_item',
             'host_batch_qk_pair_fragments', 'update_joint_host_mdo', 'attend_joint_host_tiles',
             'run_joint_host_item', 'run_joint_host_items']
renames = {n: 'nine_' + n for n in mathnames + hostnames}
renames['host_batch_compute_qk_pair'] = 'nine_compute_pair'
renames['compute_mla_qk'] = 'nine_compute_one'
def rename(s):
    for old, new in renames.items():
        s = re.sub(r'\b' + re.escape(old) + r'\b', new, s)
    return s
def virtual(s):
    return s.replace('threadIdx.y', 'nine_y()').replace('threadIdx.z', 'nine_z()').replace('blockDim.y', '4u').replace('__syncthreads();', 'nine_sync();')
helpers = '''
__device__ __forceinline__ uint32_t nine_y() { return (threadIdx.y - 1) % 4; }
__device__ __forceinline__ uint32_t nine_z() { return (threadIdx.y - 1) / 4; }
__device__ __forceinline__ void nine_sync() { asm volatile("bar.sync 1, 256;" ::: "memory"); }
template <typename Q, typename KV, typename O>
struct NineStorage {
  union {
    struct {
      alignas(128) Q q_smem_nope[64 * 512];
      alignas(128) Q q_smem_pe[64];  // alignment padding; PE is loaded from global into registers
      alignas(128) KV ckv_smem[2][32 * 512];
      alignas(128) KV kpe_p_smem[2][32 * 64];
      union { alignas(16) float m_wg[2][64]; alignas(16) float d_wg[2][64]; };
    };
    alignas(16) O o_smem[64 * 512];
  };
};
template <typename Q, typename KV, typename O, typename Id>
struct NineComputeTraits : KernelTraits<false, 2, true, 512, 64, 64, 32, Q, KV, O, Id> {
  using SharedStorage = NineStorage<Q, KV, O>;
};
template <typename Compute>
struct NineOuterTraits : Compute { static constexpr uint32_t NUM_THREADS = 288; };
template <typename Params> struct NineWarpHostParams : CachedSharedIdsHostBatchParams<Params> {};
'''
clones = []
for name in mathnames:
    s = function(math, name)
    if name == 'load_q':
        a = s.index('#pragma unroll\n    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_KPE / 4;')
        b = s.index('{', a); depth = 1; end = b + 1
        while depth:
            depth += (s[end] == '{') - (s[end] == '}'); end += 1
        s = s[:a] + s[end:]
    if name == 'compute_qk_':
        s = s.replace('#pragma unroll\n  for (uint32_t mma_d', '#pragma unroll 1\n  for (uint32_t mma_d', 1)
    clones.append(rename(virtual(s)))
for name in hostnames:
    s = function(base, name)
    if name == 'host_batch_qk_pair_fragments':
        # The nine-warp launch has at most 227 registers/thread; avoid four-dimension live fragments.
        s = s.replace('#pragma unroll 4', '#pragma unroll 1')
    if name == 'attend_joint_host_tiles':
        s = s.replace('uint32_t kv_len, int host_tiles', 'uint32_t kv_len, uint32_t row, int host_tiles')
        s = s.replace('host_batch_compute_qk_pair<KTraits>(storage,', 'host_batch_compute_qk_pair<KTraits>(params, row, storage,')
        s = s.replace('compute_mla_qk<KTraits>(&storage,', 'compute_mla_qk<KTraits>(params, row, &storage,')
    if name == 'run_joint_host_item':
        s = s.replace('kv_len, host_tiles, s_frag,', 'kv_len, row, host_tiles, s_frag,')
    if name == 'run_joint_host_items':
        s = s.replace('const int workers = gridDim.y - params.fetch_ctas;', 'const int workers = gridDim.y;')
    clones.append(rename(virtual(s)))
qpe = r'''
template <bool PAIR, typename KTraits, typename Params>
__device__ __forceinline__ void nine_pe(const Params& params, uint32_t row,
    typename KTraits::SharedStorage& storage, uint32_t a, uint32_t b, float (*sa)[8], float (*sb)[8]) {
  const uint32_t lane = threadIdx.x, group = nine_z(), warp = nine_y();
  smem_t<KTraits::SWIZZLE_MODE_KPE> ka(storage.kpe_p_smem[a]), kb(storage.kpe_p_smem[b]);
  const auto* query = params.q_pe + row * params.q_pe_stride_n;
  alignas(16) uint32_t qf[4], kf[4];
#pragma unroll 1
  for (uint32_t dim = 0; dim < 4; ++dim) {
    const uint32_t head = warp * 16 + lane / 4, col = dim * 16 + (lane % 4) * 2;
    qf[0] = *reinterpret_cast<const uint32_t*>(query + head * params.q_pe_stride_h + col);
    qf[1] = *reinterpret_cast<const uint32_t*>(query + (head + 8) * params.q_pe_stride_h + col);
    qf[2] = *reinterpret_cast<const uint32_t*>(query + head * params.q_pe_stride_h + col + 8);
    qf[3] = *reinterpret_cast<const uint32_t*>(query + (head + 8) * params.q_pe_stride_h + col + 8);
    const uint32_t offset = ka.template get_permuted_offset<KTraits::UPCAST_STRIDE_KPE>(
        group * 16 + 8 * (lane / 16) + lane % 8, 2 * dim + (lane % 16) / 8);
    ka.ldmatrix_m8n8x4(offset, kf);
    if (dim == 0) mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(sa[0], qf, kf);
    else mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sa[0], qf, kf);
    if constexpr (PAIR) {
      kb.ldmatrix_m8n8x4(offset, kf);
      if (dim == 0) mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(sb[0], qf, kf);
      else mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sb[0], qf, kf);
    }
  }
}
template <typename KTraits, typename Params>
__device__ __forceinline__ void nine_compute_pair(const Params& params, uint32_t row,
    typename KTraits::SharedStorage& storage, uint32_t a, uint32_t b, float (*sa)[8], float (*sb)[8]) {
  nine_pe<true, KTraits>(params, row, storage, a, b, sa, sb);
  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> q(storage.q_smem_nope);
  smem_t<KTraits::SWIZZLE_MODE_CKV> ka(storage.ckv_smem[a]), kb(storage.ckv_smem[b]);
  nine_host_batch_qk_pair_fragments<false, KTraits, KTraits::NUM_MMA_D_CKV,
      KTraits::UPCAST_STRIDE_Q_NOPE, KTraits::UPCAST_STRIDE_CKV>(q, ka, kb, sa, sb);
}
template <typename KTraits, typename Params>
__device__ __forceinline__ void nine_compute_one(const Params& params, uint32_t row,
    typename KTraits::SharedStorage* storage, uint32_t stage, float (*scores)[8]) {
  nine_pe<false, KTraits>(params, row, *storage, stage, stage, scores, scores);
  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> q(storage->q_smem_nope);
  smem_t<KTraits::SWIZZLE_MODE_CKV> kv(storage->ckv_smem[stage]);
  nine_compute_qk_<false, KTraits, KTraits::NUM_MMA_D_CKV,
      KTraits::UPCAST_STRIDE_Q_NOPE, KTraits::UPCAST_STRIDE_CKV>(q, kv, scores);
}
'''
# Pair helpers must precede their wrappers, wrappers must precede the attention tile loop.
at = next(i for i, s in enumerate(clones) if 'void nine_attend_joint_host_tiles(' in s)
clones.insert(at, qpe)
kernel = function(base, 'BatchMLAJointQKKernel')
kernel = kernel.replace('template <typename KTraits, typename Params, bool ROW_MERGE>',
    'template <typename KTraits, typename ComputeTraits, typename Params, bool ROW_MERGE>')
kernel = kernel.replace('BatchMLAJointQKKernel', 'BatchMLANineWarpKernel')
kernel = kernel.replace('host_batch_smem', 'nine_warp_smem')
kernel = kernel.replace('reinterpret_cast<typename KTraits::SharedStorage&>', 'reinterpret_cast<typename ComputeTraits::SharedStorage&>')
kernel = kernel.replace('const uint32_t fetch_rows = host_batch_fetch_rows<ROW_MERGE>(params, unique);', 'const uint32_t fetch_rows = 64;')
start = kernel.index('  if (unique > 0 && from_back && !host_batch_writes_back(params))')
end = kernel.index('  // Uniform over the grid', start)
kernel = kernel[:start] + '''  if (threadIdx.y == 0) {
    if (unique) nine_service<ComputeTraits>(params, smem + sizeof(typename ComputeTraits::SharedStorage), epoch, unique);
  } else {
    nine_run_joint_host_items<ComputeTraits, Params, false>(params, storage, smem, scratch, false, unique);
  }
''' + kernel[end:]
dispatch = '''
    using NineCompute = NineComputeTraits<DTypeQ, DTypeKV, DTypeO, IdType>;
    using NineOuter = NineOuterTraits<NineCompute>;
    using NineParams = NineWarpHostParams<Params>;
    constexpr size_t nine_smem = sizeof(typename NineCompute::SharedStorage) + 2 * 8 * 1152;
    static_assert(sizeof(typename NineCompute::SharedStorage) == 139904);
    static_assert(nine_smem == 158336);
    if (params.stage && !params.host_split && params.num_head_tiles == 1 && params.pool_capacity > 0 &&
        params.num_rows >= 4u * num_sms && smem_limit_per_sm >= nine_smem + 4096 &&
        smem_limit_per_sm < 200000 && static_cast<uint64_t>(params.staging_capacity) >=
            static_cast<uint64_t>(params.num_rows) * max_kv_len) {
      auto nine_kernel = BatchMLANineWarpKernel<NineOuter, NineCompute, NineParams, false>;
      FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(nine_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, nine_smem));
      int nine_blocks = 0;
      FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nine_blocks, nine_kernel, 288, nine_smem));
      if (nine_blocks >= 1) {
        static_assert(sizeof(NineParams) == sizeof(Params));
        static_assert(alignof(NineParams) == alignof(Params));
        if (params.pool_capacity > kHostBatchMaxPool) return cudaErrorInvalidConfiguration;
        params.full_rows = params.num_rows;
        params.partial_capacity = partial_capacity;
        params.num_chunks = 1;
        const uint32_t length = max(max_kv_len, 1u);
        params.prepare_parts = min(ceil_div(num_sms, max(params.num_rows, 1u)), ceil_div(length, 256u));
        params.prepare_segment = round_up(ceil_div(length, params.prepare_parts), 256u);
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(cudaLaunchCooperativeKernel((void*)nine_kernel, dim3(1, num_sms), dim3(32, 9, 1), args, nine_smem, stream));
        return cudaSuccess;
      }
    }
'''
insert = base.index('// partial_capacity bounds the item states')
candidate = base[:insert] + helpers + '\n\n'.join(clones) + '\n' + (p / 'service.inc').read_text() + '\n' + kernel + '\n' + base[insert:]
key = '    using KV64Traits = JointQKTraits<DTypeQ, DTypeKV, DTypeO, IdType>;'
assert candidate.count(key) == 1
candidate = candidate.replace(key, dispatch + key, 1)
for variant, host in [('base', base), ('nine_warp', candidate)]:
    for name, value in [(variant + '.cuh', host), (variant + '_mla.cuh', math)]:
        f = p / name
        if f.exists(): assert f.read_text() == value
        else: f.write_text(value)
    d = p / ('include_' + variant)
    if not d.exists():
        shutil.copytree(root / 'include', d)
        (d / 'flashinfer/attention/mla_host_batch.cuh').write_text(host)
    assert (d / 'flashinfer/attention/mla_host_batch.cuh').read_text() == host
    assert (d / 'flashinfer/attention/mla.cuh').read_text() == math
manifest = dict(base_sha256=sha(base), candidate_sha256=sha(candidate), math_sha256=sha(math),
    candidate_adopted=False, goal_complete=False,
    mechanism='One288threadCTA,warp0service and8compute warps retain64heads. Virtualmathcoordinates/namedbarrier256,all288threads prepare and exit. RemoveunusedsharedQPE8KBvia directglobalMMA-fragment loads. Two8rowcp.asyncbuffers and one8rowregisterbuffer, sameFARmodel/post-readhook,64rowreadyflags,immediate/deferredLRUcachefill.',
    policy_change='All108CTAservicewarps,24rowsinflight/CTA=2592rows versusold24fetchCTA*128rows=3072. Hostfetch_ctas parameter24 retained as input but dedicatedCTA partition replaced. No claimofsameproducerpolicy. QKfeatureunroll1 fixed by288threadregisterbudget<=227,notasweep.',
    budgets=dict(dynamic_shared=158336, compute_shared=139904, service_shared=18432,
                 service_register_payload_bytes_per_warp=9216, block=[32,9,1], grid=[1,108,1]),
    risks='Registercap may spill;globalQPE/L1traffic,more8rowemulatorreservations,andservicewarpbandwidth may offsetSMsharing. Physicalresources/codegen/numerics/cachepayload/synchronization/performance needverification.')
(p / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
