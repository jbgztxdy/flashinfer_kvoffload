"""Bound duplicate-QK feature unrolling after the fully unrolled candidate spills."""
import hashlib
import json
import shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
base=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_text()
assert hashlib.sha256(base.encode()).hexdigest()=='52919c286398d4668933a772c646bdccd46273a53aafeb8d5c1b14e2d9bff2c5'
candidate=(p.parent/'tune_qk_20261031/qk_local.cuh').read_text()
assert hashlib.sha256(candidate.encode()).hexdigest()=='030da403e7d11d4bb2544684fad08676d5ffa1d9c6d571a8fcb00fcba5e8fe18'
math=(root/'include/flashinfer/attention/mla.cuh').read_text()
key='''  // compute q*k^T
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_QK; ++mma_d)'''
assert math.count(key)==1
new_math=math.replace(key,'''  // Keep the existing specialization constants for every original kernel.
  // The new duplicate-QK CTA32 path limits feature unrolling to bound live fragments.
  // compute q*k^T
#pragma unroll (KTraits::CTA_TILE_KV == 32 && !KTraits::QK_SHARD ? 4 : NUM_MMA_D_QK)
  for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_QK; ++mma_d)''')
for variant,header,math_header in [('base',base,math),('qk_unroll4',candidate,new_math)]:
    assert not (p/(variant+'.cuh')).exists()
    (p/(variant+'.cuh')).write_text(header)
    (p/(variant+'_mla.cuh')).write_text(math_header)
    shutil.copytree(root/'include',p/('include_'+variant))
    for name,source in [('mla_host_batch.cuh',header),('mla.cuh',math_header)]:
        (p/('include_'+variant)/'flashinfer/attention'/name).write_text(source)
h=p.parent/'tune_heads_20261031'
q=p.parent/'tune_qk_20261031'
for name in ['.gitignore','compile.py','resources.py','resources_gpu3.py','experiment.json','plans.cpp','bench.py','bench_gpu3_native.py','test_candidate.py','candidatepaths.py','screen.py','audit_performance.py']:
    source=((h if name in ['bench.py','bench_gpu3_native.py','audit_performance.py'] else q)/name).read_text()
    if name in ['test_candidate.py','candidatepaths.py','screen.py']:
        source=source.replace('qk_local','qk_unroll4')
    if name=='audit_performance.py':
        source=source.replace('compiler_heads_normal_summary','compiler_loop_normal_summary').replace('half_heads','qk_unroll4')
    (p/name).write_text(source)
source=(h/'compiler_heads.py').read_text().replace('half_heads','qk_unroll4').replace('HalfHeadHostTraits','KernelTraits').replace('compiler_heads','compiler_loop')
source=source.replace("libs = sorted((p / ('jit_' + variant + suffix)).rglob('*.so'))", "library_root=p.parent/'tune_heads_20261031' if variant=='base' else p\n    libs = sorted((library_root / ('jit_' + variant + suffix)).rglob('*.so'))")
source=source.replace("assert all('KernelTraits' in r['symbol'] and 'STACK:0' in r['resource'] and 'LOCAL:0' in r['resource'] for r in added)", "no_spill=all('STACK:0' in r['resource'] and 'LOCAL:0' in r['resource'] for r in added)")
source=source.replace('unique_new_kernel_symbols=len({r[\'symbol\'] for r in added}),', "unique_new_kernel_symbols=len({r['symbol'] for r in added}), all_new_stack_and_local_zero=no_spill,")
(p/'compiler_loop.py').write_text(source)
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
(p/'manifest.json').write_text(json.dumps(dict(base_sha256=sha(base),candidate_sha256=sha(candidate),
    base_math_sha256=sha(math),candidate_math_sha256=sha(new_math),
    mechanism='Same duplicate-QK CTA32 as030da403; bound outer QK feature unroll to4 instead of full32/4 to try to avoid192B stack. All original types use their full original unroll factor; require complete codegen equivalence.',
    candidate_adopted=False,goal_complete=False),indent=2)+'\n')
print((p/'manifest.json').read_text())
