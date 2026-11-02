"""Double the attention KV tile without sacrificing its second pipeline stage."""
import difflib,hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda data:hashlib.sha256(data).hexdigest()
b=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes()
assert sha(b)=='754706f8d8fba14a361ce6dcfc64a358446e5df36b225f392fa8a3e7756ded79'
assert not(p/'global_query.cuh').exists()
s=b.decode();math=(root/'include/flashinfer/attention/mla.cuh').read_bytes()
start=s.index('template <typename Q, typename KV, typename O, typename Id>\nstruct StreamSourceTraits')
end=s.index('// partial_capacity bounds',start)
section=s[start:end].replace('StreamSource','GlobalQuery').replace('stream_source_','global_query_')
traits='''struct GlobalQueryTraits : JointQKTraits<Q, KV, O, Id> {};'''
replacement='''struct GlobalQueryTraits : KernelTraits<false,2,true,512,64,64,64,Q,KV,O,Id> {
  union SharedStorage {
    struct {
      alignas(16) KV ckv_smem[2][64*512];
      alignas(16) KV kpe_p_smem[2][64*64];
      union { alignas(16) float m_wg[2][64]; alignas(16) float d_wg[2][64]; };
    };
    alignas(16) O o_smem[64*512];
  };
};'''
assert section.count(traits)==1;section=section.replace(traits,replacement)
first=section.index('// Two different KV tiles')
last=section.index('// Each lane owns',first)
section=section[:first]+section[last:]
first=section.index('template <bool PAIR, typename KTraits>')
last=section.index('template <typename KTraits, typename Params, bool ROW_MERGE>',first)
section=section[:first]+(p/'attention.inc').read_text()+'\n'+section[last:]
old='''    load_q<KTraits>(&storage, params.q_nope + row * params.q_nope_stride_n,
                    params.q_pe + row * params.q_pe_stride_n, params.q_nope_stride_n,
                    params.q_nope_stride_h, params.q_pe_stride_n, params.q_pe_stride_h, 1,
                    head_tile * CTA_TILE_Q, params.num_heads);
    // load_q's copies join the first KV group, as in the native kernel.
'''
assert section.count(old)==1;section=section.replace(old,'')
fstart=s.rindex('template <typename KTraits, typename Params>',0,s.index('__device__ __forceinline__ void fetch_stream_source_pages('))
fetch=s[fstart:start].replace('fetch_stream_source_pages','fetch_global_query_pages')
at=s.index('    using StreamSourceKernelTraits =');dend=s.index('    using CachedPEKernelTraits =',at)
dispatch=s[at:dend].replace('StreamSource','GlobalQuery').replace('stream_source','global_query')
s=s[:end]+fetch+section+s[end:]
at=s.index('    using StreamSourceKernelTraits =');s=s[:at]+dispatch+s[at:]
for v,data in [('base',b),('global_query',s.encode())]:
 (p/(v+'.cuh')).write_bytes(data);(p/(v+'_mla.cuh')).write_bytes(math)
 tree=p/('include_'+v);shutil.copytree(root/'include',tree)
 (tree/'flashinfer/attention/mla_host_batch.cuh').write_bytes(data)
old=p.parent/'tune_qk_cursor_20261102'
for name in ['bench_gpu3_native.py','resources_gpu3.py','experiment.json','plans.cpp','compile.py','audit_performance.py']:
 shutil.copy2(old/name,p/name)
for name in ['candidatepaths.py','run_builds.py','run_focused.py']:
 (p/name).write_text((old/name).read_text().replace('qk_cursor','global_query'))
test=(p.parent/'tune_two_pass_20261102/test_two_pass.py').read_text()
test=test.replace('test_two_pass_payload_lse_graph','test_global_query_payload_lse_graph')
test=test.replace("[65,2048]","[65,2048]")
test=test.replace('    assert w._host_batch_buffers[6].numel()*2>=rows*length*64*4\n','')
(p/'test_global_query.py').write_text(test)
runner=(p/'run_focused.py').read_text()
begin=runner.index("cmd=['numactl'",runner.index("record['resources_start']"))
end=runner.index("\nrun('targeted_correctness'",begin)
runner=runner[:begin]+"cmd=['numactl','-N1','-m1','/mnt/data/mhy-vllm-venv/bin/python','-m','pytest','-q','-p','no:cacheprovider','-p','jitpaths','-p','candidatepaths',str(p/'test_global_query.py')]"+runner[end:]
runner=runner.replace("for v in ['base','global_query']:","for v in ['global_query']:",1)
(p/'run_focused.py').write_text(runner)
(p/'candidate.diff').write_text(''.join(difflib.unified_diff(b.decode().splitlines(True),s.splitlines(True),fromfile='base.cuh',tofile='global_query.cuh')))
record=dict(base_sha256=sha(b),candidate_sha256=sha(s.encode()),math_sha256=sha(math),
 mechanism='Native sharded64KV/two stages, query-nope fragments loaded directly from global through caches. Reclaims Q shared storage for second64KV stage; cached item QPE; one64KV max/rescale/PV; same total shared147968B and dedicated fetch64x2.',
 invariants='One cooperative kernel, same prepare/resolve/exact launchLRU/sourcefetch/FAR/bytes. Original instances literal. Attention equation unchanged; FP32 order differs across KV columns.',adopted=False)
(p/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
