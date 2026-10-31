"""Accumulate quantized P in each lane; reduce denominator replicas once per attention item."""
import hashlib
import json
from pathlib import Path
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
jq = p.parent / 'tune_joint_qk_20261101'
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
base = (jq / 'base.cuh').read_text()
control = (jq / 'joint_qk.cuh').read_text()
math = (root / 'include/flashinfer/attention/mla.cuh').read_text()
assert sha(base) == 'b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
assert sha(control) == 'a2f37fb985fcbc523e740c85e9cc6468afad74fd8716c11aef4221ebab7c4b7c'
assert sha(math) == '34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
begin = math.index('template <typename KTraits>\n__device__ __forceinline__ void compute_mla_pv(')
end = math.index('template <typename KTraits>\n__device__ __forceinline__ void normalize_d_', begin)
pv = math[begin:end].replace('compute_mla_pv(', 'compute_lane_denominator_pv(')
key = '      mma::m16k16_rowsum_f16f16f32(d, p_f16[mma_kv]);'
assert pv.count(key) == 2
pv = pv.replace(key, '''      // Keep exactly the probabilities consumed by PV. Four lanes own disjoint
      // KV columns of the same two query rows; postpone their reduction to item end.
      float probability[8];
      vec_cast<float, typename KTraits::DTypeKV>::cast<8>(probability, p_f16[mma_kv]);
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        d[j] += (probability[j*2] + probability[j*2+1]) +
                (probability[j*2+4] + probability[j*2+5]);
      }''')
s = control.replace('JointQKTraits', 'LaneDenominatorTraits').replace('JointQKHostParams', 'LaneDenominatorHostParams').replace('BatchMLAJointQKKernel', 'BatchMLALaneDenominatorKernel')
begin = s.index('// Two different KV tiles share each Q fragment.')
end = s.index('// partial_capacity bounds the item states', begin)
body = s[begin:end]
assert body.count('compute_mla_pv<KTraits>') == 3
body = body.replace('compute_mla_pv<KTraits>', 'compute_lane_denominator_pv<KTraits>')
key = '  init_states_<KTraits>(o_frag, m, d);'
assert body.count(key) == 1
body = body.replace(key, key + '''
  // Native empty-item d=1 has four identical lane replicas. Each partial starts
  // at 1/4, preserving that empty-item state after the final quad reduction.
  d[0] = d[1] = 0.25f;''')
key = '  normalize_d_<KTraits>(&storage, 0, o_frag, m, d);'
assert body.count(key) == 1
body = body.replace(key, '''#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    d[j] += math::shfl_xor_sync(d[j], 2);
    d[j] += math::shfl_xor_sync(d[j], 1);
  }
''' + key)
s = s[:begin] + pv + body + s[end:]
for variant, host in [('base', base), ('joint_qk', control), ('lane_d', s)]:
    for name, value in [(variant + '.cuh', host), (variant + '_mla.cuh', math)]:
        f = p / name
        if f.exists():
            assert f.read_text() == value
        else:
            f.write_text(value)
    d = p / ('include_' + variant)
    if not d.exists():
        shutil.copytree(root / 'include', d)
        (d / 'flashinfer/attention/mla_host_batch.cuh').write_text(host)
    assert (d / 'flashinfer/attention/mla_host_batch.cuh').read_text() == host
    assert (d / 'flashinfer/attention/mla.cuh').read_text() == math
for name in ['plans.cpp', 'experiment.json']:
    f = p / name
    value = (jq / name).read_bytes()
    if f.exists():
        assert f.read_bytes() == value
    else:
        f.write_bytes(value)
m = dict(base_sha256=sha(base), joint_qk_control_sha256=sha(control), candidate_sha256=sha(s),
    math_sha256=sha(math), candidate_adopted=False, goal_complete=False,
    mechanism='Replace only per-tile quantized-P MMA rowsum with lane-private FP32 sums of identical BF16/FP16 probabilities; final quad shuffles combine once before native cross-warpgroup normalize. Same jointMDO/PV/refill/barriers/fetch/LRU/32KV/2stage/f24.',
    equation='For each query, P columns split across four lanes. Joint max/rescale factor is identical across that quad, so summing lane-private d at item end gives the same softmax denominator in real arithmetic. FP32 sum/rescale grouping differs. d starts1/4 perlane to preserve native empty-item d=1 perwarpgroup. P quantization and PV are unchanged.',
    tradeoff='Removes one TensorCore denominator instruction per32KV tile/warp, adds probability decode and scalar adds plus four final shuffles. No assumed gain; registers or scalar work may dominate.',
    excluded='No delayed/upper-bound softmax anchor, no probability shift, no P publication/barrier changes, no parameter sweep.')
(p / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
print(json.dumps(m, indent=2))
