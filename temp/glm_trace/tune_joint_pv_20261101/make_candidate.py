"""Publish the two paired-QK probability tiles together, without changing native PV order."""
import hashlib
import json
import shutil
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parents[2]
jq = p.parent / 'tune_joint_qk_20261101'
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
for name in ['plans.cpp', 'experiment.json']:
    f = p / name
    value = (jq / name).read_bytes()
    if f.exists():
        assert f.read_bytes() == value
    else:
        f.write_bytes(value)
control = (jq / 'joint_qk.cuh').read_text()
base = (jq / 'base.cuh').read_text()
math = (root / 'include/flashinfer/attention/mla.cuh').read_text()
assert sha(control) == 'a2f37fb985fcbc523e740c85e9cc6468afad74fd8716c11aef4221ebab7c4b7c'
assert sha(base) == 'b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
assert sha(math) == '34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'

# Take the canonical native shared-P write and PV-consume expressions literally.
# The dedicated traits keep all pre-existing helper template instances untouched.
pv = math.split('__device__ __forceinline__ void compute_mla_pv(', 1)[1]
publish = pv.split('    __syncthreads();\n    smem_t<KTraits::SWIZZLE_MODE_P>', 1)[1]
publish = '    smem_t<KTraits::SWIZZLE_MODE_P>' + publish.split('    uint32_t p_smem_offset_r', 1)[0]
consume = pv.split('    uint32_t p_smem_offset_r', 1)[1].split('  } else {', 1)[0]
consume = '    uint32_t p_smem_offset_r' + consume
consume = consume.replace('    // wait for p_smem to be filled\n    __syncthreads();\n\n', '')
assert '__syncthreads' not in publish + consume
helper = r'''
// Both KPE stages have finished QK before update_joint_host_mdo's CTA barrier.
// That barrier protects these KPE->P overwrites. Publishing both stages needs only
// one subsequent CTA barrier. Native quantization/rowsum and hi-then-lo PV remain.
template <typename KTraits>
__device__ __forceinline__ void publish_joint_host_p(
    typename KTraits::SharedStorage* smem_storage, const uint32_t stage_idx,
    float (*s_frag)[8], float* d) {
  static_assert(KTraits::QK_SHARD && KTraits::CTA_TILE_KV == 32);
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z,
                 warp_idx_in_wg = threadIdx.y;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  alignas(16) typename KTraits::DTypeKV p_f16[NUM_MMA_KV / 2][8];
#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV / 2; ++mma_kv) {
    vec_cast<typename KTraits::DTypeKV, float>::cast<8>(p_f16[mma_kv], s_frag[mma_kv]);
    mma::m16k16_rowsum_f16f16f32(d, p_f16[mma_kv]);
  }
PUBLISH
}

template <typename KTraits>
__device__ __forceinline__ void consume_joint_host_pv(
    typename KTraits::SharedStorage* smem_storage, const uint32_t stage_idx,
    float (*o_frag)[8]) {
  static_assert(KTraits::QK_SHARD && KTraits::CTA_TILE_KV == 32);
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z,
                 warp_idx_in_wg = threadIdx.y;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t UPCAST_STRIDE_CKV = KTraits::UPCAST_STRIDE_CKV;
  constexpr uint32_t UPCAST_STRIDE_P = KTraits::UPCAST_STRIDE_P;
  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[stage_idx]);
  smem_t<KTraits::SWIZZLE_MODE_P> p_smem(smem_storage->kpe_p_smem[stage_idx]);
  uint32_t ckv_smem_offset_r = ckv_smem.template get_permuted_offset<UPCAST_STRIDE_CKV>(
      lane_idx % 16, warpgroup_idx * NUM_MMA_D_CKV + lane_idx / 16);
CONSUME
}

'''.replace('PUBLISH', publish).replace('CONSUME', consume)
s = control.replace('JointQKTraits', 'JointPVTraits').replace('JointQKHostParams', 'JointPVHostParams').replace('BatchMLAJointQKKernel', 'BatchMLAJointPVKernel')
key = 'template <typename KTraits, typename Params>\n__device__ __forceinline__ void attend_joint_host_tiles('
assert s.count(key) == 1
s = s.replace(key, helper + key, 1)
key = '    compute_mla_pv<KTraits>(&storage, hi%2, scores, d, o);'
assert s.count(key) == 1
s = s.replace(key, '''    publish_joint_host_p<KTraits>(&storage, hi%2, scores, d);
    publish_joint_host_p<KTraits>(&storage, lo%2, scores+1, d);
    __syncthreads();
    consume_joint_host_pv<KTraits>(&storage, hi%2, o);''', 1)
key = '    compute_mla_pv<KTraits>(&storage, lo%2, scores+1, d, o);'
assert s.count(key) == 1
s = s.replace(key, '    consume_joint_host_pv<KTraits>(&storage, lo%2, o);', 1)
for v, host in [('base', base), ('joint_qk', control), ('joint_pv', s)]:
    for filename, value in [(v + '.cuh', host), (v + '_mla.cuh', math)]:
        f = p / filename
        if f.exists():
            assert f.read_text() == value
        else:
            f.write_text(value)
    d = p / ('include_' + v)
    if not d.exists():
        shutil.copytree(root / 'include', d)
        (d / 'flashinfer/attention/mla_host_batch.cuh').write_text(host)
    assert (d / 'flashinfer/attention/mla_host_batch.cuh').read_text() == host
    assert (d / 'flashinfer/attention/mla.cuh').read_text() == math
manifest = dict(base_sha256=sha(base), joint_qk_control_sha256=sha(control),
    candidate_sha256=sha(s), math_sha256=sha(math), candidate_adopted=False, goal_complete=False,
    mechanism='After paired QK and joint-MDO barrier, quantize/rowsum/store both probability tiles. One CTA barrier publishes both, then native hi-PV/refill/lo-PV/refill. Four native PV barriers per pair become one. Same cast, rowsum, per-output MMA sequence, cache/LRU, staging, 32KV/2stage and 108CTA scheduling.',
    synchronization_proof='Both QK KPE reads precede joint-MDO barrier. P writes target separate KPE/P stage arrays, disjoint from m_wg. New barrier precedes all P reads. Existing pre-refill barriers protect each released CKV/P stage. P_lo remains untouched during stage_hi refill. Odd final tile retains canonical PV.',
    performance_hypothesis='Reduce repeated CTA synchronization after QK pairing; not a warp-role/tile/unroll/fetch/cadence scan.')
(p / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(manifest, indent=2))
