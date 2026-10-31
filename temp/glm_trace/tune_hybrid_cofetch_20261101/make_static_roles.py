"""Remove unreachable 64-row service paths from attention's live accumulator scope."""
import hashlib
import json
from pathlib import Path
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
source = (p/'hybrid.cuh').read_text()
sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
assert sha(source) == '9168c83fdb5ea1e5f0281aa42f820d433135269922225c133d824f21a3588259'
begin = source.index('// Workers retain two attention stages and use a separate 2x4-row service buffer.')
end = source.index('// Warp 0 waits until every staged entry of a KV tile has landed;', begin)
helpers = source[begin:end]
for name in ['issue_host_batch_cofetch', 'init_host_batch_cofetch',
             'step_host_batch_cofetch', 'finish_host_batch_cofetch']:
    key = 'template <typename KTraits, typename Params>\n__device__ __forceinline__ '
    signature = next(line for line in helpers.splitlines() if '__device__ __forceinline__ ' in line and name+'(' in line)
    old = key+signature.split('__device__ __forceinline__ ', 1)[1]
    new = old.replace('template <typename KTraits, typename Params>',
                      'template <typename KTraits, typename Params, bool DEDICATED = false>')
    assert helpers.count(old) == 1, name
    helpers = helpers.replace(old, new)
helpers = helpers.replace('if (service.dedicated)', 'if constexpr (DEDICATED)')
helpers = helpers.replace('service.dedicated ? 64u : 4u', 'DEDICATED ? 64u : 4u')
helpers = helpers.replace('service.dedicated = blockIdx.y >= gridDim.y - params.fetch_ctas;',
                          'service.dedicated = DEDICATED;')
helpers = helpers.replace('issue_host_batch_cofetch<KTraits>(params,',
                          'issue_host_batch_cofetch<KTraits, Params, DEDICATED>(params,')
helpers = helpers.replace('while (step_host_batch_cofetch<KTraits>(params))',
                          'while (step_host_batch_cofetch<KTraits, Params, DEDICATED>(params))')
candidate = source[:begin]+helpers+source[end:]
old = '''    init_host_batch_cofetch<KTraits>(params, unique, epoch, smem);
    if (from_back) finish_host_batch_cofetch<KTraits>(params);'''
assert candidate.count(old) == 1
candidate = candidate.replace(old, '''    if (from_back) {
      init_host_batch_cofetch<KTraits, Params, true>(params, unique, epoch, smem);
      finish_host_batch_cofetch<KTraits, Params, true>(params);
    } else {
      init_host_batch_cofetch<KTraits, Params, false>(params, unique, epoch, smem);
    }''')
candidate = candidate.replace('HybridCoFetchHostBatchParams', 'StaticHybridCoFetchHostBatchParams')
variant = 'hybrid_static'
assert not (p/(variant+'.cuh')).exists()
(p/(variant+'.cuh')).write_text(candidate)
shutil.copytree(root/'include', p/('include_'+variant))
(p/('include_'+variant)/'flashinfer/attention/mla_host_batch.cuh').write_text(candidate)
for name in ['test_candidate.py', 'candidatepaths.py', 'screen.py', 'stress_candidate.py']:
    f = p/name
    text = f.read_text()
    if name == 'screen.py':
        text = text.replace("choices=['hybrid']", "choices=['hybrid', 'hybrid_static']")
        text = text.replace("['base', 'hybrid']", "['base', 'hybrid', 'hybrid_static']")
    elif name == 'stress_candidate.py':
        text = text.replace("choices=['hybrid']", "choices=['hybrid', 'hybrid_static']")
    else:
        text = text.replace("'hybrid'", "'hybrid','hybrid_static'")
    f.write_text(text)
for stem in ['compiler', 'probe']:
    text = (p/(stem+'_hybrid.py')).read_text().replace('hybrid', variant)
    text = text.replace('HybridCoFetchHostBatchParams', 'StaticHybridCoFetchHostBatchParams')
    (p/(stem+'_'+variant+'.py')).write_text(text)
manifest = dict(predecessor_sha256=sha(source), candidate_sha256=sha(candidate),
    mechanism='Same two-stage hybrid policy, jobs, payload buffers, cadence8, math, LRU and emulator. Compile dedicated64-row service separately from worker4-row service, so attention never carries unreachable64-row paths while its output accumulators are live. Dedicated queues finish before attention.',
    evidence='Predecessor passes107normaltests but EMU newkernel REG255 STACK440. Static specialization tests a compiler-liveness hypothesis, not an unroll/cadence sweep.',
    candidate_adopted=False, goal_complete=False)
(p/'static_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(manifest)
