"""Skip only exact-unit online-softmax output rescaling in a distinct cached kernel."""
import hashlib
import json
from pathlib import Path
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
previous = p.parent/'tune_feature_qk_20261101'
base = (root/'include/flashinfer/attention/mla_host_batch.cuh').read_text()
math = (root/'include/flashinfer/attention/mla.cuh').read_text()
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
assert sha(base) == 'b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
assert sha(math) == '34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
private = math.replace('#include ', '#include ', 1)
# The detection trait leaves every existing KTraits (including non-KernelTraits
# custom device traits) false without changing its definition or layout.
key = 'template <typename KTraits>\n__device__ __forceinline__ void update_mdo_states_('
assert private.count(key) == 1
private = private.replace(key, '''template <typename Traits, typename = void>
struct MLAUnitOScaleGuard : std::false_type {};
template <typename Traits>
struct MLAUnitOScaleGuard<Traits, std::void_t<decltype(Traits::skip_unit_o_scale)>>
    : std::bool_constant<Traits::skip_unit_o_scale> {};

''' + key)
key = '''      d[j] *= o_scale;
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 2; ++mma_d) {
        o_frag[mma_d][j * 2 + 0] *= o_scale;
        o_frag[mma_d][j * 2 + 1] *= o_scale;
        o_frag[mma_d][j * 2 + 4] *= o_scale;
        o_frag[mma_d][j * 2 + 5] *= o_scale;
      }'''
assert private.count(key) == 2
private = private.replace(key, '''      // The computed scale, including NaN/non-unit cases, is unchanged.
      // Exact multiplication by 1 leaves finite d/O unchanged.
      if (!MLAUnitOScaleGuard<KTraits>::value || o_scale != 1.0f) {
''' + key + '\n      }')
base_header = base
key = 'constexpr uint32_t kHostBatchFetchRows = 64;'
assert base.count(key) == 1
candidate = base.replace(key, '''template <typename Base>
struct UnitScaleHostTraits : Base {
  static constexpr bool skip_unit_o_scale = true;
};
template <typename Params>
struct UnitScaleHostBatchParams : CachedSharedIdsHostBatchParams<Params> {};

''' + key)
key = '            kernel = reinterpret_cast<decltype(kernel)>(BatchMLAHostBatchKernel<KTraits, SharedParams, false>);'
assert candidate.count(key) == 1
candidate = candidate.replace(key, key+'''
            if (params.num_rows >= 4u * num_sms) {
              using ScaleTraits = UnitScaleHostTraits<KTraits>;
              using ScaleParams = UnitScaleHostBatchParams<Params>;
              static_assert(sizeof(ScaleParams) == sizeof(Params));
              static_assert(alignof(ScaleParams) == alignof(Params));
              kernel = reinterpret_cast<decltype(kernel)>(BatchMLAHostBatchKernel<ScaleTraits, ScaleParams, false>);
            }''')
for variant, host, mla in [('base',base_header,math),('unit_scale',candidate,private)]:
    assert not (p/(variant+'.cuh')).exists()
    (p/(variant+'.cuh')).write_text(host)
    (p/(variant+'_mla.cuh')).write_text(mla)
    shutil.copytree(root/'include',p/('include_'+variant))
    (p/('include_'+variant)/'flashinfer/attention/mla_host_batch.cuh').write_text(host)
    (p/('include_'+variant)/'flashinfer/attention/mla.cuh').write_text(mla)
for name in ['.gitignore','compile.py','resources.py','resources_gpu3.py','experiment.json','plans.cpp',
             'bench.py','bench_gpu3_native.py','test_candidate.py','candidatepaths.py','screen.py',
             'stress_candidate.py','test_cofetch_progress.py','audit_performance.py']:
    text = (previous/name).read_text()
    text = text.replace("'dim_split','duplicate_control'", "'unit_scale'")
    text = text.replace("['dim_split', 'duplicate_control']", "['unit_scale']")
    text = text.replace("['base', 'dim_split', 'duplicate_control']", "['base', 'unit_scale']")
    text = text.replace("default=['dim_split', 'base']", "default=['unit_scale', 'base']")
    text = text.replace("default='dim_split'", "default='unit_scale'")
    if name == 'experiment.json':
        obj = json.loads(text);obj.update(candidate='unit_scale',status='candidate_in_progress')
        text = json.dumps(obj,indent=2)+'\n'
    if name == 'audit_performance.py':
        text = text.replace('e5968e7605bb657de30e5600c4a8fc09b977a3e2e7163f64e46718de519289a3',sha(private))
    (p/name).write_text(text)
text = (previous/'compiler_dim_split.py').read_text().replace('dim_split','unit_scale')
text = text.replace('SmallKVFeatureHostParams','UnitScaleHostBatchParams')
(p/'compiler_unit_scale.py').write_text(text)
# Use the already validated full-head Graph probe; only the type and private math differ.
old = p.parent/'tune_hybrid_cofetch_20261101/probe_hybrid_static.py'
text = old.read_text().replace('hybrid_static','unit_scale')
text = text.replace('StaticHybridCoFetchHostBatchParams','UnitScaleHostBatchParams')
text = text.replace("root/'include/flashinfer/attention/mla.cuh'", "p/'include_unit_scale/flashinfer/attention/mla.cuh'")
(p/'probe_unit_scale.py').write_text(text)
manifest = dict(base_sha256=sha(base_header),candidate_sha256=sha(candidate),
    base_math_sha256=sha(math),candidate_math_sha256=sha(private),
    mechanism='Same cached whole-row >=4SM,CTA32/2stages/64heads/256threads and fetch/LRU. Compute original o_scale exactly; when it equals1 skip d/O multiplication only. All QK, score exponentials, m, P, PV, normalization and tile order remain native. No QK partition/reduction change.',
    baseline_control='Detection trait false for every original KTraits; require all34EMU/68normaloriginalkernel instructions/resources unchanged. Originalcachedkernel remains compiled as the unguarded control.',
    tradeoff='Warp divergence/predicate cost may exceed skipped unit multiplications; test rather than infer runtime gain.',
    candidate_adopted=False,goal_complete=False)
(p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
