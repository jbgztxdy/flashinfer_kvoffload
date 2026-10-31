"""Give each warp its own radix histogram in the existing small cached path."""
import hashlib
import json
from pathlib import Path
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
jq = p.parent / 'tune_joint_qk_20261101'
old = p.parent / 'tune_lane_denominator_20261101'
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
base = (jq / 'joint_qk.cuh').read_text()
math = (root / 'include/flashinfer/attention/mla.cuh').read_text()
assert sha(base) == 'a2f37fb985fcbc523e740c85e9cc6468afad74fd8716c11aef4221ebab7c4b7c'
assert sha(math) == '34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
s = base
key = 'constexpr uint32_t kHostBatchFetchRows = 64;'
s = s.replace(key, '''// A distinct type keeps every existing kernel specialization unchanged.
template <typename Params>
struct WarpHistogramHostParams : Params {};

''' + key)
begin = base.index('template <typename KTraits, typename Params>\n__device__ __noinline__ void run_host_batch_lru(')
end = base.index('\ntemplate <', begin + 20)
helper = base[begin:end]
helper = helper.replace('const Params& params, uint8_t* smem,',
                        'const WarpHistogramHostParams<Params>& params, uint8_t* smem,', 1)
key = '  __shared__ uint32_t histogram[256], warp_sums[THREADS / 32], selected[4];'
assert helper.count(key) == 1
helper = helper.replace(key, key + '''
  // Warp aggregation still combines equal keys before each atomic. Separate
  // rows remove serialization between warps that count the same launch epoch.
  __shared__ uint32_t warp_histogram[THREADS / 32][256];
  uint32_t* local_histogram = warp_histogram[tid / 32];''')
key = '      for (uint32_t b = tid; b < 256; b += THREADS) histogram[b] = 0;'
assert helper.count(key) == 1
helper = helper.replace(key, '      for (uint32_t b = lane; b < 256; b += 32) local_histogram[b] = 0;')
assert helper.count('atomicAdd(histogram + bin,') == 2
helper = helper.replace('atomicAdd(histogram + bin,', 'atomicAdd(local_histogram + bin,')
key = '        for (uint32_t b = 0; b < 8; ++b) sum += bins[b] = histogram[lane * 8 + b];'
assert helper.count(key) == 1
helper = helper.replace(key, '''        for (uint32_t b = 0; b < 8; ++b) {
          uint32_t count = 0;
#pragma unroll
          for (uint32_t w = 0; w < THREADS / 32; ++w)
            count += warp_histogram[w][lane * 8 + b];
          // Each lane owns these eight bins throughout the following rescan.
          // No other lane reads them before the existing CTA barrier.
          sum += bins[b] = histogram[lane * 8 + b] = count;
        }''')
# Define the overload before either kernel's dependent call is instantiated.
s = s.replace(base[begin:end], base[begin:end] + '\n' + helper, 1)
key = '''      const size_t smem_size = sizeof(typename KTraits::SharedStorage);'''
assert s.count(key) == 1
s = s.replace(key, '''      if constexpr (NUM_STAGES == 2 && CTA_TILE_KV == 32 && QK_SHARD) {
        if (row_merge && smem_limit_per_sm >= 160000 && smem_limit_per_sm < 200000) {
          using HistParams = WarpHistogramHostParams<Params>;
          static_assert(sizeof(HistParams) == sizeof(Params));
          static_assert(alignof(HistParams) == alignof(Params));
          kernel = reinterpret_cast<decltype(kernel)>(
              BatchMLAHostBatchKernel<KTraits, HistParams, true>);
        }
      }
''' + key)
for variant, host in [('base', base), ('warp_hist', s)]:
    for name, value in [(variant + '.cuh', host), (variant + '_mla.cuh', math)]:
        f = p / name
        if f.exists():
            assert f.read_text() == value, f
        else:
            f.write_text(value)
    d = p / ('include_' + variant)
    if not d.exists():
        shutil.copytree(root / 'include', d)
        (d / 'flashinfer/attention/mla_host_batch.cuh').write_text(host)
    assert (d / 'flashinfer/attention/mla_host_batch.cuh').read_text() == host
    assert (d / 'flashinfer/attention/mla.cuh').read_text() == math
for name in ['plans.cpp', 'experiment.json', 'resources.py', 'resources_gpu3.py',
             'bench.py', 'bench_gpu3_native.py', 'compile.py']:
    f = p / name
    value = (jq / name if name in ['plans.cpp', 'experiment.json'] else old / name).read_bytes()
    if f.exists():
        assert f.read_bytes() == value
    else:
        f.write_bytes(value)
for name in ['candidatepaths.py', 'test_candidate.py']:
    value = (old / name).read_text().replace('lane_d', 'warp_hist')
    (p / name).write_text(value)
nine = p.parent / 'tune_nine_warp_static_20261101'
value = (nine / 'compiler_nine_warp_static.py').read_text().replace('nine_warp_static', 'warp_hist')
value = value.replace("'BatchMLANineStaticWarpKernel' in r['symbol'] or 'nine_static_run_host_batch_lru' in r['symbol']",
                      "'WarpHistogramHostParams' in r['symbol']")
(p / 'compiler_warp_hist.py').write_text(value)
(p / 'verify_machine_words.py').write_text((nine / 'verify_machine_words.py').read_text().replace('nine_warp_static', 'warp_hist'))
(p / '.gitignore').write_text('/include_*/\n/jit_*/\n/__pycache__/\n/plans.so\n!/*.log\n')
m = dict(base_sha256=sha(base), candidate_sha256=sha(s), math_sha256=sha(math),
    base='Frozen JQ a2f; production remains b6 until its original retention gate finishes',
    candidate_adopted=False, goal_complete=False,
    mechanism='Eight warp-private 256-bin radix histograms, same warp-aggregated integer counts; warp0 reduces private rows into its original eight owned bins before the existing prefix scan.',
    dispatch='Only existing ROW_MERGE cached staged rows*headtiles<=32, 2stage/CTA32/QK_SHARD, 160000<=shared_limit<200000; all other existing JQ instantiations remain literal.',
    invariants='Same stamp loads, radix rank/prefix, victim threshold, tie traversal, scans, payload, attention, fetch, chunk planner and simulator. No added CTA or grid barriers. One cooperative kernel.',
    tradeoff='Adds8192 static shared bytes and64 histogram loads per warp0 lane per radix byte; only measured evidence can decide whether avoided contention pays.')
(p / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
print(json.dumps(m, indent=2))
