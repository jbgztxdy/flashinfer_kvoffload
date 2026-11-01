"""Create an attention half-buffer ring and a two-stage unused-QPE-copy control."""
import hashlib
import json
from pathlib import Path
import re
import shutil

p=Path(__file__).resolve().parent
root=p.parents[2]
prior=p.parent/'tune_cached_pe_20261101'
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
base=(prior/'cached_pe.cuh').read_text()
math=(root/'include/flashinfer/attention/mla.cuh').read_text()
assert sha(base)=='a43617290e405fdcd54b09140dbc29b357d2923e17a3d99a193caf5f7d592b15'
assert sha(math)=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'

def function(source,name):
    pos=source.index('void '+name+'(')
    start=source.rfind('template <',0,pos)
    opening=source.index('{',pos)
    depth=1;end=opening+1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[start:end]+'\n\n'

def no_pe_query(prefix):
    s=function(math,'load_q').replace('void load_q(', 'void '+prefix+'load_q(')
    s=s.replace('  smem_t<KTraits::SWIZZLE_MODE_Q_PE> q_smem_pe(smem_storage->q_smem_pe);\n','')
    s=re.sub(r'    DTypeQ\* q_pe_ptr =\n.*?;\n','',s,flags=re.S)
    key='    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_KPE / 4; ++mma_d) {'
    start=s.index(key);opening=s.index('{',start);depth=1;end=opening+1
    while depth:
        depth+=(s[end]=='{')-(s[end]=='}');end+=1
    pragma=s.rfind('#pragma unroll',0,start)
    s=s[:pragma]+s[end:]
    assert 'q_smem_pe' not in s and 'q_pe_ptr' not in s
    return s

no_copy_storage='''
template <typename Q, typename KV, typename O>
struct NoCopyPEStorage {
  union {
    struct {
      alignas(16) Q q_smem_nope[64 * 512];
      alignas(16) uint8_t q_alignment_padding[128];
      alignas(16) KV ckv_smem[2][32 * 512];
      alignas(16) KV kpe_p_smem[2][32 * 64];
      union {
        alignas(16) float m_wg[2][64];
        alignas(16) float d_wg[2][64];
      };
    };
    alignas(16) O o_smem[64 * 512];
  };
};
'''
ring=(p/'half_ring.inc').read_text()
boundary=ring.index('__device__ __forceinline__ uint32_t half_ring_index(')
ring_storage,ring_helpers=ring[:boundary],ring[boundary:]
names=['cached_pe_qk_pair_fragments','cached_pe_load_query','cached_pe_compute_pe',
       'cached_pe_compute_one','cached_pe_compute_qk_pair','cached_pe_update_joint_host_mdo',
       'cached_pe_attend_joint_host_tiles','cached_pe_run_joint_host_item','cached_pe_run_joint_host_items',
       'BatchMLACachedPEKernel']
insert=base.index('// partial_capacity bounds the item states')
dispatch_start=base.index('    using CachedPEKernelTraits = CachedPETraits<')
dispatch_end=base.index('    using KV64Traits = JointQKTraits<',dispatch_start)
dispatch=base[dispatch_start:dispatch_end]
headers={}
for variant,prefix,trait,params,kernel,size in [
    ('no_copy_pe','no_copy_pe_','NoCopyPETraits','NoCopyPEHostParams','BatchMLANoCopyPEKernel',139904),
    ('half_ring','half_ring_','HalfRingTraits','HalfRingHostParams','BatchMLAHalfRingKernel',158336)]:
    def rename(s):
        return s.replace('cached_pe_',prefix).replace('BatchMLACachedPEKernel',kernel)
    storage=no_copy_storage if variant=='no_copy_pe' else ring_storage
    storage_name='NoCopyPEStorage' if variant=='no_copy_pe' else 'HalfRingStorage'
    traits=f'''
template <typename Q, typename KV, typename O, typename Id>
struct {trait} : CachedPETraits<Q, KV, O, Id> {{
  using SharedStorage = {storage_name}<Q, KV, O>;
}};
template <typename Params>
struct {params} : CachedPEHostParams<Params> {{}};

'''
    if variant=='no_copy_pe':
        helpers=''.join(rename(function(base,name)) for name in names)
        helpers=helpers.replace('    load_q<KTraits>(', '    no_copy_pe_load_q<KTraits>(')
        helpers=no_pe_query(prefix)+helpers
    else:
        helpers=no_pe_query(prefix)
        helpers+=rename(function(base,'cached_pe_load_query'))
        helpers+=rename(function(base,'cached_pe_update_joint_host_mdo'))
        helpers+=ring_helpers+'\n'
        for name in ['cached_pe_run_joint_host_item','cached_pe_run_joint_host_items','BatchMLACachedPEKernel']:
            s=rename(function(base,name)).replace('    load_q<KTraits>(', '    half_ring_load_q<KTraits>(')
            helpers+=s
    s=base[:insert]+storage+traits+helpers+base[insert:]
    new_dispatch=dispatch.replace('CachedPEKernelTraits',trait+'Alias').replace('CachedPEKernelParams',params+'Alias')
    new_dispatch=new_dispatch.replace('CachedPETraits',trait).replace('CachedPEHostParams',params)
    new_dispatch=new_dispatch.replace('BatchMLACachedPEKernel',kernel).replace('cached_pe',variant)
    new_dispatch=new_dispatch.replace('static_assert('+variant+'_smem == 147968);','static_assert('+variant+f'_smem == {size});')
    assert f'static_assert({variant}_smem == {size});' in new_dispatch
    s=s.replace(dispatch,new_dispatch+dispatch,1)
    headers[variant]=s
headers['base']=base
for variant,host in headers.items():
    for name,value in [(variant+'.cuh',host),(variant+'_mla.cuh',math)]:
        f=p/name
        if f.exists():assert f.read_text()==value,f
        else:f.write_text(value)
    d=p/('include_'+variant)
    if not d.exists():
        shutil.copytree(root/'include',d)
        (d/'flashinfer/attention/mla_host_batch.cuh').write_text(host)
    assert (d/'flashinfer/attention/mla_host_batch.cuh').read_text()==host
    assert (d/'flashinfer/attention/mla.cuh').read_text()==math
for name in ['plans.cpp','experiment.json','resources.py','resources_gpu3.py','bench.py','bench_gpu3_native.py','compile.py','test_cofetch_progress.py']:
    f=p/name;value=(prior/name).read_bytes()
    if f.exists():assert f.read_bytes()==value
    else:f.write_bytes(value)
for name in ['candidatepaths.py','test_candidate.py','stress_candidate.py']:
    s=(prior/name).read_text().replace("'cached_pe'","'no_copy_pe','half_ring'")
    if name=='candidatepaths.py':s=s.replace("'base','no_copy_pe','half_ring','joint_qk'","'base','no_copy_pe','half_ring'")
    if not (p/name).exists():(p/name).write_text(s)
(p/'.gitignore').write_text('/include_*/\n/jit_*/\n/__pycache__/\n/plans.so\n!/*.log\n')
m=dict(base_sha256=sha(base),candidate_sha256={v:sha(s) for v,s in headers.items() if v!='base'},
    math_sha256=sha(math),candidate_adopted=False,goal_complete=False,
    base='Frozen cachedPEa436,not currentproductionJQ7193; parent cachedPE SXM runner remains unchanged.',
    mechanism='Five16rowphysicalhalves represent two32KVtiles plus one16rowprefetch. Rotate halves after descending hi/lo PV. Prefetch only tiles beyond host_tiles, so no earlier remote wait. Same native quantizedP/MMA order, jointMDO, QPEregisterreuse,256threads/f24/onecooperativekernel.',
    control='no_copy_pe retains2x32KV attention stages but sizeof storage139904 automatically selects ONE fetch buffer. It cannot isolate unusedQPEcopy removal; see fixed_control_manifest.json and corrected invalid_budget_review.json. Frozen header unchanged.',
    resources=dict(no_copy_pe_dynamic=139904,half_ring_dynamic=158336,prior_static=2304,half_ring_total_if_static_unchanged=160640),
    changes='Private32Brow P-half address mapping; balanced16threads/row half loader; no far read/traffic hook changes. Initial QNOPE copies join first half group. Wait_group1 only after an additional early prefetch commit, otherwise wait_group0.',
    risks='Additional metadata, buffer IDs, cp.async groups or register spills may offset overlap. All original38EMU/76normal instances must remain exact; numerical/cache/Graph/synchronization and performance gates precede any adoption.')
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps(m,indent=2))
