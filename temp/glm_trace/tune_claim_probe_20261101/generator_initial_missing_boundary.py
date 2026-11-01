"""Isolate a relaxed claim-state probe before the existing owning CAS."""
import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];prior=p.parent/'tune_cached_pe_20261101'
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
base=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_text();math=(root/'include/flashinfer/attention/mla.cuh').read_text()
assert sha(base)=='7193255e2fdf83d3baa9347dea1a02b892c29ed746da719818e30fdf4066d469'
assert sha(math)=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
def function(source,name):
 pos=source.index('void '+name+'(');start=source.rfind('template <',0,pos);first=source.index('{',pos);depth=1;end=first+1
 while depth:depth+=(source[end]=='{')-(source[end]=='}');end+=1
 return source[start:end]+'\n\n'
prepare=function(base,'prepare_host_batch_rows').replace('void prepare_host_batch_rows(', 'void claim_probe_prepare_rows(')
old='''        claimed[i] = pages[i] >= params.host_page_start
                         ? atomicCAS(params.host_slot + (pages[i] - params.host_page_start), 0u, 1u)
                         : 2u;'''
new='''        claimed[i] = 2u;
        if (pages[i] >= params.host_page_start) {
          uint32_t* slot = params.host_slot + (pages[i] - params.host_page_start);
          uint32_t observed;
          // This is a GPU-scope relaxed atomic load, not a non-atomic C++ data race.
          // Claim state cannot become zero again until prepare/resolve/grid barriers finish.
          asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(observed) : "l"(slot) : "memory");
          claimed[i] = observed != 0 ? observed : atomicCAS(slot, 0u, 1u);
        }'''
assert prepare.count(old)==1;prepare=prepare.replace(old,new)
traits='''template <typename Q, typename KV, typename O, typename Id>
struct ClaimProbeTraits : JointQKTraits<Q, KV, O, Id> {};
template <typename Params>
struct ClaimProbeHostParams : JointQKHostParams<Params> {};

'''
kernel=function(base,'BatchMLAJointQKKernel').replace('BatchMLAJointQKKernel','BatchMLAClaimProbeKernel').replace('prepare_host_batch_rows<KTraits>','claim_probe_prepare_rows<KTraits>')
insert=base.index('// partial_capacity bounds the item states')
s=base[:insert]+traits+prepare+kernel+base[insert:]
a=base.index('    using JointTraits = JointQKTraits<');b=base.index('    const auto launch =',a)
dispatch=base[a:b]
new=dispatch.replace('JointTraits','ClaimProbeKernelTraits').replace('JointParams','ClaimProbeKernelParams').replace('JointQKTraits','ClaimProbeTraits').replace('JointQKHostParams','ClaimProbeHostParams').replace('BatchMLAJointQKKernel','BatchMLAClaimProbeKernel').replace('joint_', 'claim_probe_')
s=s.replace(dispatch,new+dispatch,1)
for v,host in [('base',base),('claim_probe',s)]:
 for name,value in [(v+'.cuh',host),(v+'_mla.cuh',math)]:
  f=p/name
  if f.exists():assert f.read_text()==value,f
  else:f.write_text(value)
 d=p/('include_'+v)
 if not d.exists():
  shutil.copytree(root/'include',d);(d/'flashinfer/attention/mla_host_batch.cuh').write_text(host)
 assert (d/'flashinfer/attention/mla_host_batch.cuh').read_text()==host
 assert (d/'flashinfer/attention/mla.cuh').read_text()==math
for n in ['plans.cpp','experiment.json','resources.py','resources_gpu3.py','bench.py','bench_gpu3_native.py','compile.py','test_cofetch_progress.py']:
 shutil.copy2(prior/n,p/n)
for n in ['candidatepaths.py','test_candidate.py','stress_candidate.py']:
 s=(prior/n).read_text().replace("'cached_pe'","'claim_probe'")
 if n=='candidatepaths.py':s=s.replace("'base','claim_probe','joint_qk'","'base','claim_probe'")
 if n=='stress_candidate.py':s=s.replace('tests=6','tests=8')
 (p/n).write_text(s)
(p/'.gitignore').write_text('/include_*/\n/jit_*/\n/__pycache__/\n/plans.so\n!/*.log\n')
m=dict(base_sha256=sha(base),candidate_sha256=sha((p/'claim_probe.cuh').read_text()),math_sha256=sha(math),candidate_adopted=False,goal_complete=False,
 mechanism='GPU-scope relaxed atomic claim-state load before existing atomicCAS(0->1). SkipCAS if nonzero; if zero,theoriginalCAS still elects exactlyoneowner. Existing rankpublish/resolve/gridbarrier/fetchclearing unchanged.',
 scope='Onlycachedwhole32KV/2stage/largeSM80 shape alreadyusingproductionJQ;newtypedkernel. Keepallolddefinitions/math/attention/fetch/LRU/simulator byteforbyte.',
 correctness='host_slot state monotonic duringprepare:0->1->encodedowner>=2. Any nonzero observation implies alreadyclaimed; stalereadzero merelyfallsbackCAS. No owner/rank is consumed beforeoriginalgridbarrier. Originalendfetch clearsalluniqueincloverflowbeforenextlaunch.',
 resources='Same256threads/64heads/2stage32KV/147968dynamic/f24originalpolicy;actualresources/codegen/dispatchmustbeverified.',
 risk='An extra random read on firstclaims can exceed avoidedduplicate atomics. No promisedbenefit;shortsameparameterthreelinkscreen aftercorrectnessandoriginal36-72wordidentity.')
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(m)
