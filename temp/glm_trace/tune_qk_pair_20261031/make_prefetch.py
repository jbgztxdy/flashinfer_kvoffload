"""Restore the native loop's early page-ID reads in the paired-tile schedule."""
import hashlib
import json
from pathlib import Path
import shutil
p=Path(__file__).resolve().parent;root=p.parents[2]
source=(p/'qk_pair.cuh').read_text()
assert hashlib.sha256(source.encode()).hexdigest()=='75ea7044f3ba677b1dc67b828651c8cb013dfd6bb88a7d2ff43a3f81922d6f9f'
begin=source.index('__device__ __forceinline__ void attend_host_batch_tile_pairs(')
end=source.index('// The native MLA tile loop',begin)
body=source[begin:end]
key='''  for (;tile>=0;tile-=2) {
    cp_async::wait_group<0>();'''
assert body.count(key)==1
body=body.replace(key,'''  for (;tile>=0;tile-=2) {
    // The native loop reads next IDs before QK/PV to hide their global latency.
    typename KTraits::IdType pages_a[kHostBatchTileRows<KTraits>];
    typename KTraits::IdType pages_b[kHostBatchTileRows<KTraits>];
    const int next_a=tile-2,next_b=tile-3;
    if (next_a>=0) read_host_batch_pages<KTraits>(params,index_base,kv_len,next_a*32,pages_a);
    if (next_b>=0) read_host_batch_pages<KTraits>(params,index_base,kv_len,next_b*32,pages_b);
    cp_async::wait_group<0>();''')
body=body.replace('    const int next_a=tile-2;\n','').replace('      const int next_b=tile-3;\n','')
body=body.replace('''      read_host_batch_pages<KTraits>(params,index_base,kv_len,next_a*32,pages);
      load_host_batch_kv<KTraits>(&storage,params,pages,kv_len,next_a*32,next_a%2,index_base);''','''      load_host_batch_kv<KTraits>(&storage,params,pages_a,kv_len,next_a*32,next_a%2,index_base);''')
body=body.replace('''        read_host_batch_pages<KTraits>(params,index_base,kv_len,next_b*32,pages);
        load_host_batch_kv<KTraits>(&storage,params,pages,kv_len,next_b*32,next_b%2,index_base);''','''        load_host_batch_kv<KTraits>(&storage,params,pages_b,kv_len,next_b*32,next_b%2,index_base);''')
candidate=source[:begin]+body+source[end:]
assert candidate!=source and 'params,pages_a' in candidate and 'params,pages_b' in candidate
variant='qk_pair_prefetch';assert not (p/(variant+'.cuh')).exists()
(p/(variant+'.cuh')).write_text(candidate)
shutil.copytree(root/'include',p/('include_'+variant))
(p/('include_'+variant)/'flashinfer/attention/mla_host_batch.cuh').write_text(candidate)
for name in ['test_candidate.py','candidatepaths.py','screen.py']:
    f=p/name;s=f.read_text().replace("'qk_pair'","'qk_pair','qk_pair_prefetch'");f.write_text(s)
s=(p/'compiler_pair.py').read_text().replace("for variant in ['base', 'qk_pair']:","for variant in ['base', 'qk_pair_prefetch']:").replace("records['qk_pair']","records['qk_pair_prefetch']").replace("sha(p/'qk_pair.cuh')","sha(p/'qk_pair_prefetch.cuh')").replace("p/'include_qk_pair/","p/'include_qk_pair_prefetch/").replace('compiler_pair','compiler_prefetch')
(p/'compiler_prefetch.py').write_text(s)
(p/'prefetch_manifest.json').write_text(json.dumps(dict(predecessor_sha256=hashlib.sha256(source.encode()).hexdigest(),candidate_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
    mechanism='Only hoist the next two remap/page-ID reads before the paired QK/PV work, as the native one-tile loop does. Same pair helper/unroll4/math/fetch/cache; do not conflate with first late-ID schedule.',
    candidate_adopted=False,goal_complete=False),indent=2)+'\n')
print((p/'prefetch_manifest.json').read_text())
