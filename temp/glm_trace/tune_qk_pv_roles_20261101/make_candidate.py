"""Freeze a12-warp QK/PV role kernel without changing any original definition."""
import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];prior=p.parent/'tune_cached_pe_20261101'
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
base=(prior/'cached_pe.cuh').read_text();math=(root/'include/flashinfer/attention/mla.cuh').read_text()
assert sha(base)=='a43617290e405fdcd54b09140dbc29b357d2923e17a3d99a193caf5f7d592b15'
assert sha(math)=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
def function(source,name):
    pos=source.index('void '+name+'(');start=source.rfind('template <',0,pos)
    first=source.index('{',pos);depth=1;end=first+1
    while depth:depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[start:end]+'\n\n'
helpers=(p/'qkpv.inc').read_text();index=helpers.index('template <typename KTraits, typename Params>\n__device__ __forceinline__ void qkpv_load_q(')
write=function(math,'write_o').replace('void write_o(', 'void qkpv_write_o(').replace('threadIdx.z','(threadIdx.z - 1)')
query=function(base,'cached_pe_load_query').replace('cached_pe_load_query','qkpv_load_query')
helpers=helpers[:index]+write+query+helpers[index:]
kernel=function(base,'BatchMLACachedPEKernel').replace('BatchMLACachedPEKernel','BatchMLAQKPVKernel').replace('cached_pe_run_joint_host_items','qkpv_run_host_items')
insert=base.index('// partial_capacity bounds the item states')
s=base[:insert]+helpers+kernel+base[insert:]
a=base.index('    using CachedPEKernelTraits = CachedPETraits<');b=base.index('    using KV64Traits = JointQKTraits<',a)
dispatch=base[a:b]
new=dispatch.replace('CachedPEKernelTraits','QKPVKernelTraits').replace('CachedPEKernelParams','QKPVKernelParams').replace('CachedPETraits','QKPVTraits').replace('CachedPEHostParams','QKPVHostParams').replace('BatchMLACachedPEKernel','BatchMLAQKPVKernel').replace('cached_pe','qkpv_roles').replace(', 256,',', 384,').replace('dim3(32, 4, 2)','dim3(32, 4, 3)')
s=s.replace(dispatch,new+dispatch,1)
for v,host in [('base',base),('qkpv_roles',s)]:
    for name,value in [(v+'.cuh',host),(v+'_mla.cuh',math)]:
        f=p/name
        if f.exists():assert f.read_text()==value,f
        else:f.write_text(value)
    d=p/('include_'+v)
    if not d.exists():
        shutil.copytree(root/'include',d);(d/'flashinfer/attention/mla_host_batch.cuh').write_text(host)
    assert (d/'flashinfer/attention/mla_host_batch.cuh').read_text()==host
    assert (d/'flashinfer/attention/mla.cuh').read_text()==math
for name in ['plans.cpp','experiment.json','resources.py','resources_gpu3.py','bench.py','bench_gpu3_native.py','compile.py','test_cofetch_progress.py']:
    f=p/name;value=(prior/name).read_bytes()
    if f.exists():assert f.read_bytes()==value
    else:f.write_bytes(value)
for name in ['candidatepaths.py','test_candidate.py','stress_candidate.py']:
    s=(prior/name).read_text().replace("'cached_pe'","'qkpv_roles'")
    if name=='candidatepaths.py':s=s.replace("'base','qkpv_roles','joint_qk'","'base','qkpv_roles'")
    f=p/name
    if not f.exists():f.write_text(s)
(p/'.gitignore').write_text('/include_*/\n/jit_*/\n/__pycache__/\n/plans.so\n!/*.log\n')
m=dict(base_sha256=sha(base),candidate_sha256=sha((p/'qkpv_roles.cuh').read_text()),math_sha256=sha(math),candidate_adopted=False,goal_complete=False,
 mechanism='Four full32column QK/softmax warps and eight256feature PV warps, within one384-thread cooperative CTA. Share eachQ fragment across both16KV halves and both staged tiles. QK-role loads next-hi while PV-role computes current-lo. Original64row/f24 dedicatedfetch/realLRU/post-read FAR unchanged.',
 block=[32,4,3],dynamic_shared_bytes=147968,compute_fields_bytes=140032,
 change='Joint softmax across full32KV/two tiles local to four QK warps; denominator uses original quantizedP/native rowsum then total state broadcast. Same attention equation and PE->CKV per-score MMA order; FP32 reduction/grouping differs.',
 protocol='Namedbarrier1/all384:initialstate,Ppublication,PVhi-release,PVlo-release. Namedbarrier2/QK128:waiterandcp.async visibility. All384 originalprepare/LRU/grid/exit barriers remain outside divergent roles.',
 risks='Uniform168-register budget; possible spills, producer load issue cost, synchronization overhead and TensorCore contention. No performance or correctness claim before actual validation.')
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(m)
