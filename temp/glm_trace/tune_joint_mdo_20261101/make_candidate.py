"""Native32KV/2stage with a joint MDO over two score tiles and pipelined stage refill."""
import hashlib,json,re,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
base=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_text();math=(root/'include/flashinfer/attention/mla.cuh').read_text()
assert sha(base)=='b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
assert sha(math)=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
def function(s,name):
 m=re.search(r'\bvoid '+re.escape(name)+r'\(',s);assert m,name
 a=s.rfind('template <',0,m.start());i=s.index('{',m.end());depth=1;j=i+1
 while depth:depth+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[a:j]
item=function(base,'run_host_batch_item').replace('run_host_batch_item(', 'run_joint_host_item(').replace('attend_host_batch_tiles<KTraits>','attend_joint_host_tiles<KTraits>')
items=function(base,'run_host_batch_items').replace('run_host_batch_items(', 'run_joint_host_items(').replace('run_host_batch_item<KTraits','run_joint_host_item<KTraits').replace('float s_frag[KTraits::QK_SHARD ? NUM_MMA_KV / 2 : NUM_MMA_KV][8];','float s_frag[2][8];')
kernel=function(base,'BatchMLAHostBatchKernel').replace('BatchMLAHostBatchKernel(', 'BatchMLAJointMDOKernel(').replace('run_host_batch_items<KTraits','run_joint_host_items<KTraits')
key='template <typename KTraits, typename Params, bool ROW_MERGE>\n__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLAHostBatchKernel('
assert base.count(key)==1
s=base.replace(key,(p/'joint_attention.inc').read_text()+'\n'+item+'\n'+items+'\n'+kernel+'\n\n'+key,1)
s=s.replace('constexpr uint32_t kHostBatchFetchRows = 64;','template <typename Params>\nstruct JointMDOHostParams : CachedSharedIdsHostBatchParams<Params> {};\n\nconstexpr uint32_t kHostBatchFetchRows = 64;',1)
# Reuse the fully frozen, allocation-matched dispatcher from the KV64 experiment. Its only
# changes here are the original32KV/2stage traits and the new single-kernel symbol/params.
launch=(p.parent/'tune_kv64_20261101/kv64.cuh').read_text()
a=launch.index('    using KV64Traits =');b=launch.index('    DISPATCH_SMEM_CONFIG(',a)
launch=launch[a:b].replace('KernelTraits<false, 1, true, 512, 64, 64, 64,','KernelTraits<false, 2, true, 512, 64, 64, 32,').replace('BatchMLAHostBatchKernel<KV64Traits','BatchMLAJointMDOKernel<KV64Traits').replace('KV64HostBatchParams','JointMDOHostParams')
key='    DISPATCH_SMEM_CONFIG(smem_limit_per_sm, NUM_STAGES, CTA_TILE_KV, QK_SHARD, {'
s=s.replace(key,launch+key,1)
for v,host in [('base',base),('joint_mdo',s)]:
 for name,value in [(v+'.cuh',host),(v+'_mla.cuh',math)]:
  f=p/name
  if f.exists():assert f.read_text()==value
  else:f.write_text(value)
 for suffix in ['', '_normal']:
  d=p/('include_'+v+suffix)
  if not d.exists():
   shutil.copytree(root/'include',d);(d/'flashinfer/attention/mla_host_batch.cuh').write_text(host)
  assert sha((d/'flashinfer/attention/mla_host_batch.cuh').read_text())==sha(host)
  assert sha((d/'flashinfer/attention/mla.cuh').read_text())==sha(math)
m=dict(base_sha256=sha(base),candidate_sha256=sha(s),math_sha256=sha(math),
 mechanism='Native64head/256thread32KV/2stage. TwoQK scoretiles share one joint maximum and d/O rescale; native PV per tile in descending order. Refill first stage between PVs so future firstKV overlaps secondPV. Exact originalfetch/LRU/layout/147968B retained. Canonical math header untouched; new host helper changes FP32 online grouping.',
 distinction='Earlier QKpair retained separate MDO updates. KV64/1stage reduces update frequency but loses double buffering, measured100GB1.576/1.795% slower. This tests joint MDO while retaining native two-stage storage and partial refill overlap; not a cadence/unroll/fetch scan.',
 candidate_adopted=False,goal_complete=False)
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
