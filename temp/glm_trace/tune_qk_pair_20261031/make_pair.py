"""Generate an independent paired-tile QK candidate from retained b6 production."""
import hashlib
import json
from pathlib import Path
import shutil
p=Path(__file__).resolve().parent;root=p.parents[2]
base=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_text()
assert hashlib.sha256(base.encode()).hexdigest()=='b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
s=base.replace('  static constexpr bool shared_fetch_ids = false;',
               '  static constexpr bool shared_fetch_ids = false;\n  static constexpr bool paired_qk = false;')
assert s!=base
key='constexpr uint32_t kHostBatchFetchRows = 64;'
assert s.count(key)==1
s=s.replace(key,'''template <typename Params>
struct PairedQKHostBatchParams : CachedSharedIdsHostBatchParams<Params> {
  static constexpr bool paired_qk = true;
};

'''+key)
key='// The native MLA tile loop over the KV tiles of one item, last tile first.'
assert s.count(key)==1
s=s.replace(key,(p/'pair_helpers.inc').read_text()+'\n'+key)
key='''  constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  constexpr int32_t NUM_STAGES = KTraits::NUM_STAGES;
  typename KTraits::IdType pages[kHostBatchTileRows<KTraits>];'''
assert s.count(key)==1
s=s.replace(key,'''  if constexpr (Params::paired_qk) {
    attend_host_batch_tile_pairs<KTraits>(params,storage,variant,scratch,index_base,kv_len,
        host_tiles,s_frag,o_frag,m,d);
    return;
  }
'''+key)
key='''            kernel = reinterpret_cast<decltype(kernel)>(BatchMLAHostBatchKernel<KTraits, SharedParams, false>);'''
assert s.count(key)==1
s=s.replace(key,key+'''
            if (params.num_rows >= 4u*num_sms) {
              using PairParams=PairedQKHostBatchParams<Params>;
              static_assert(sizeof(PairParams)==sizeof(Params));
              kernel=reinterpret_cast<decltype(kernel)>(BatchMLAHostBatchKernel<KTraits,PairParams,false>);
            }''')
for variant,source in [('base',base),('qk_pair',s)]:
    assert not (p/(variant+'.cuh')).exists()
    (p/(variant+'.cuh')).write_text(source)
    shutil.copytree(root/'include',p/('include_'+variant))
    (p/('include_'+variant)/'flashinfer/attention/mla_host_batch.cuh').write_text(source)
w=p.parent/'tune_wait_20261031';r=p.parent/'tune_cached_runtime_20261031'
for name in ['.gitignore','compile.py','resources.py','resources_gpu3.py','experiment.json','plans.cpp','bench.py','bench_gpu3_native.py','test_candidate.py','candidatepaths.py','screen.py']:
    source=(w/name).read_text()
    if name in ['test_candidate.py','candidatepaths.py']:
        source=source.replace("'cached_ids'","'cached_ids','qk_pair'")
    if name=='screen.py':
        source=source.replace("'cached_ids']","'cached_ids','qk_pair']")
    (p/name).write_text(source)
compiler=(r/'compiler_runtime.py').read_text().replace('runtime_ids','qk_pair').replace('compiler_runtime','compiler_pair')
compiler=compiler.replace("library_root=p.parent/'tune_heads_20261031' if variant=='base' else p",'library_root=p')
compiler=compiler.replace("assert all('CachedSharedIdsHostBatchParams' in r['symbol'] for r in added)","assert all('PairedQKHostBatchParams' in r['symbol'] for r in added)")
(p/'compiler_pair.py').write_text(compiler)
manifest=dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),
    mechanism='Two different KV tiles share each Q fragment and independent score accumulators; PE then CKV MMA order per score and descending softmax/PV order retained. Full-staging cachedwholeCTA32>=4SMrows only; samefetch/LRU/simulator/singlekernel.',
    historical_regpipe3_sha256=hashlib.sha256((p.parent/'tune_100gb_20261030/regpipe3.cuh').read_bytes()).hexdigest(),
    historical_difference='regpipe3 adds a third register fetch buffer; its attention QK remains one tile at a time. This changes attention QK, not fetch buffering.',
    unroll=4,candidate_adopted=False,goal_complete=False)
(p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest)
