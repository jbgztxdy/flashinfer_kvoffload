"""Publish the unique owner's encoded claim with a GPU-scope atomic store."""
import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
base=(p/'base.cuh').read_text();math=(p/'base_mla.cuh').read_text()
assert sha(base)=='7193255e2fdf83d3baa9347dea1a02b892c29ed746da719818e30fdf4066d469'
def function(source,name):
 pos=source.index('void '+name+'(');start=source.rfind('template <',0,pos);first=source.index('{',pos);depth=1;end=first+1
 while depth:depth+=(source[end]=='{')-(source[end]=='}');end+=1
 return source[start:end]+'\n\n'
prepare=function(base,'prepare_host_batch_rows').replace('void prepare_host_batch_rows(', 'void claim_store_prepare_rows(')
old='''          atomicExch(params.host_slot + (pages[i] - params.host_page_start),
                     (row << kHostBatchRankBits | rank) + 2);'''
new='''          uint32_t* claim_slot = params.host_slot + (pages[i] - params.host_page_start);
          const uint32_t encoded = (row << kHostBatchRankBits | rank) + 2;
          // The CAS winner is the only writer. Other CAS(0->1) calls cannot modify nonzero.
          // GPU-scope atomic store keeps per-location coherence; resolve follows grid.sync.
          asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" :: "l"(claim_slot), "r"(encoded) : "memory");'''
assert prepare.count(old)==1;prepare=prepare.replace(old,new)
traits='''template <typename Q, typename KV, typename O, typename Id>
struct ClaimStoreTraits : JointQKTraits<Q, KV, O, Id> {};
template <typename Params>
struct ClaimStoreHostParams : JointQKHostParams<Params> {};

'''
kernel=function(base,'BatchMLAJointQKKernel').replace('BatchMLAJointQKKernel','BatchMLAClaimStoreKernel').replace('prepare_host_batch_rows<KTraits>','claim_store_prepare_rows<KTraits>')
insert=base.index('// partial_capacity bounds the item states');s=base[:insert]+traits+prepare+kernel+base[insert:]
a=base.index('    using JointTraits = JointQKTraits<');b=base.index('    DISPATCH_SMEM_CONFIG(',a);dispatch=base[a:b]
new=dispatch.replace('JointTraits','ClaimStoreKernelTraits').replace('JointParams','ClaimStoreKernelParams').replace('JointQKTraits','ClaimStoreTraits').replace('JointQKHostParams','ClaimStoreHostParams').replace('BatchMLAJointQKKernel','BatchMLAClaimStoreKernel').replace('joint_','claim_store_')
s=s.replace(dispatch,new+dispatch,1)
for name,value in [('claim_store.cuh',s),('claim_store_mla.cuh',math)]:
 f=p/name
 if f.exists():assert f.read_text()==value
 else:f.write_text(value)
d=p/'include_claim_store'
if not d.exists():
 shutil.copytree(root/'include',d);(d/'flashinfer/attention/mla_host_batch.cuh').write_text(s)
assert (d/'flashinfer/attention/mla_host_batch.cuh').read_text()==s
m=dict(base_sha256=sha(base),candidate_sha256=sha(s),math_sha256=sha(math),candidate_adopted=False,goal_complete=False,
 mechanism='Keep every original claimingCAS. Only successful owner publishes encoded(row,rank) using st.relaxed.gpu.global.u32 instead of unused-result atomicExch. No claim-state probe,extra read or spin.',
 correctness='Owner CAS0->1 is unique. FailedCAS(0->1)cannotmodify nonzero. Owner is sole encodedclaim writer; fetchclearing occurs only afterprepare/resolve gridbarriers. RelaxedGPU atomicstore remains aligned32bitcoherent; everyowner finishes publicationbeforegrid.sync exposesrank toresolve.',
 scope='SamecurrentproductionJQ7193cachedwhole32KV2stageSM80dispatch;unchangedmath/fetch/LRU/barriers/256threads/147968dynamic/f24. Preservealloriginal36-72words.',
 risk='GPU store vsRMW may change compiler scheduling; exactresources/actualbranch/numerics/cachepayload/performance mustbe measured. No promisedbenefit.')
(p/'store_manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(m)
