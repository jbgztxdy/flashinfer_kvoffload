"""Isolate duplicate-QK warpgroups to remove softmax/P exchange on large CTA32."""
import hashlib
import json
import shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
base=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_text()
assert hashlib.sha256(base.encode()).hexdigest()=='52919c286398d4668933a772c646bdccd46273a53aafeb8d5c1b14e2d9bff2c5'
key='''        kernel = reinterpret_cast<decltype(kernel)>(BatchMLAHostBatchKernel<KTraits, WholeParams, false>);
      }
      const size_t smem_size'''
assert base.count(key)==1
candidate=base.replace(key,'''        kernel = reinterpret_cast<decltype(kernel)>(BatchMLAHostBatchKernel<KTraits, WholeParams, false>);
        if constexpr (CTA_TILE_KV == 32) {
          // Both warpgroups compute identical logits/P. PV still partitions features.
          // The original false-QK_SHARD math avoids m_wg/P shared-memory exchange.
          if (params.num_rows >= 4u * num_sms) {
            using LocalQKTraits = KernelTraits<false, NUM_STAGES, false, HEAD_DIM_CKV,
                HEAD_DIM_KPE, 64, CTA_TILE_KV, DTypeQ, DTypeKV, DTypeO, IdType>;
            static_assert(sizeof(typename LocalQKTraits::SharedStorage)==sizeof(typename KTraits::SharedStorage));
            kernel = reinterpret_cast<decltype(kernel)>(BatchMLAHostBatchKernel<LocalQKTraits, WholeParams, false>);
          }
        }
      }
      const size_t smem_size''')
for variant,header in [('base',base),('qk_local',candidate)]:
    assert not (p/(variant+'.cuh')).exists()
    (p/(variant+'.cuh')).write_text(header)
    shutil.copytree(root/'include',p/('include_'+variant))
    (p/('include_'+variant)/'flashinfer/attention/mla_host_batch.cuh').write_text(header)
w=p.parent/'tune_wait_20261031'
for name in ['.gitignore','compile.py','resources.py','resources_gpu3.py','experiment.json','plans.cpp','bench.py','bench_gpu3_native.py','test_candidate.py','candidatepaths.py','screen.py','audit_performance.py']:
    source=(w/name).read_text()
    if name in ['test_candidate.py','candidatepaths.py']:
        source=source.replace("'cached_ids'","'cached_ids','qk_local'")
    if name=='screen.py':source=source.replace("'cached_ids']","'cached_ids', 'qk_local']")
    (p/name).write_text(source)
manifest=dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
    mechanism='64-head CTA32/2stages,duplicate QK logits in both warpgroups; native false-QK_SHARD softmax/PV skips shared m/P exchange. Single cooperative kernel, same attention/LRU/traffic/emulator. Only original large whole-row conditions with>=4SMrows.',
    candidate_adopted=False,goal_complete=False)
(p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
