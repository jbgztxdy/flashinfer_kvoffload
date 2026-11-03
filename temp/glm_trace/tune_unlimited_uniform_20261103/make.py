"""One-time barrier-fused allwarp candidate, normal build first."""
from pathlib import Path
import hashlib,json,shutil,difflib
p=Path(__file__).resolve().parent;root=p.parents[2];old=p.parent/'tune_uniform_warps_20261103'
s=(old/'candidate.cuh').read_text();base=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_text();assert hashlib.sha256(base.encode()).hexdigest()=='cab508dcb6e83b7a953d94a8e54e30e0aaf60bb1d7ba98480fc66893edcf4384';assert not(p/'fused.cuh').exists()
a=s.index('// Single-CTA state lives');b=s.index('// Two different KV tiles',a);s=s[:a]+(p/'service.cuh').read_text()+'\n'+s[b:]
# QK is the original production body again: no service params or conditionalrounds.
a=s.index('// Two different KV tiles');b=s.index('// Each lane owns',a)
x=base[base.index('// Two different KV tiles'):base.index('// Each lane owns')].replace('pv_feature_','uniform_');s=s[:a]+x+s[b:]
a=s.index('template <typename KTraits,typename Params>\n__device__ __forceinline__ void uniform_compute_qk_pair');b=s.index('// Two native32-row',a)
x=base[base.index('template <typename KTraits>\n__device__ __forceinline__ void pv_feature_compute_qk_pair'):base.index('// Two native32-row',base.index('void pv_feature_compute_qk_pair'))].replace('pv_feature_','uniform_');s=s[:a]+x+s[b:]
# Rename old outertypedpath, leaving service helperprefixalreadyfinal.
s=s.replace('UniformTraits','FusedServiceTraits').replace('UniformHostParams','FusedServiceHostParams').replace('BatchMLAUniformKernel','BatchMLAFusedServiceKernel').replace('uniform_','fused_service_')
# preQK realwait/barrier is original; do stores then QK. FirstjointMDO barrier
# ensures stores complete, next nativePV barrier ensures nextmetadata complete.
oldtext='''    fused_service_progress<KTraits>(params,storage);
    fused_service_compute_qk_pair<KTraits>(storage, hi%2, lo%2, qpe, scores, scores+1,params);'''
new='''    const uint32_t service_tid=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x;
    if(service_tid==0)storage.service.current=storage.service.done;
    cp_async::wait_group<0>();
    __syncthreads();
    fused_service_stores<KTraits>(params,storage);
    fused_service_compute_qk_pair<KTraits>(storage, hi%2, lo%2, qpe, scores, scores+1);'''
assert oldtext in s;s=s.replace(oldtext,new)
a=s.index('void fused_service_update_joint_host_mdo');b=s.index('template <typename KTraits, typename Params>\n__device__ __forceinline__ void fused_service_attend',a);t=s[a:b]
t=t.replace('template <typename KTraits>','template <typename KTraits,typename Params>') if False else t
# Template declaration occurs beforefunctionname: explicitreplacementoutside slice.
s=s[:a]+t+s[b:]
s=s.replace('template <typename KTraits>\n__device__ __forceinline__ void fused_service_update_joint_host_mdo','template <typename KTraits,typename Params>\n__device__ __forceinline__ void fused_service_update_joint_host_mdo')
a=s.index('void fused_service_update_joint_host_mdo');b=s.index('template <typename KTraits, typename Params>\n__device__ __forceinline__ void fused_service_attend',a);t=s[a:b];t=t.replace('float (*s0)[8], float (*s1)[8], float (*o)[8], float* m, float* d) {','float (*s0)[8], float (*s1)[8], float (*o)[8], float* m, float* d,const Params& params) {');t=t.replace('  __syncthreads();','  __syncthreads();\n  fused_service_publish_prepare<KTraits>(params,*storage);',1);s=s[:a]+t+s[b:]
s=s.replace('fused_service_update_joint_host_mdo<KTraits>(&storage, variant, scores, scores+1, o, m, d);','fused_service_update_joint_host_mdo<KTraits>(&storage, variant, scores, scores+1, o, m, d,params);')
# PV firstonlyhi copiesnextpacket afterexistingbarrier.
s=s.replace('template <typename KTraits>\n__device__ __forceinline__ void fused_service_pv(', 'template <bool SERVICE,typename KTraits,typename Params>\n__device__ __forceinline__ void fused_service_pv(')
a=s.index('void fused_service_pv(');b=s.index('// All256threads compute',a);t=s[a:b];t=t.replace('float (*o_frag)[8]) {','float (*o_frag)[8],const Params& params) {');t=t.replace('    __syncthreads();','    __syncthreads();\n    if constexpr(SERVICE)fused_service_refill<KTraits>(params,*smem_storage);',1);s=s[:a]+t+s[b:]
# servicehelpers mustbe declared beforePVbody references; movehelperregion.
a=s.index('// All256threads compute');b=s.index('// Two different KV tiles',a);helpers=s[a:b];s=s[:a]+s[b:];a=s.index('template <bool SERVICE,typename KTraits,typename Params>\n__device__ __forceinline__ void fused_service_pv');s=s[:a]+helpers+'\n'+s[a:]
s=s.replace('fused_service_pv<KTraits>(&storage, hi%2, scores, d, o);','fused_service_pv<true,KTraits>(&storage, hi%2, scores, d, o,params);').replace('fused_service_pv<KTraits>(&storage, lo%2, scores+1, d, o);','fused_service_pv<false,KTraits>(&storage, lo%2, scores+1, d, o,params);').replace('fused_service_pv<KTraits>(&storage, 0, scores, d, o);','fused_service_pv<false,KTraits>(&storage, 0, scores, d, o,params);')
# Absorb the caller's original postwaitbarrier into wait'sbroadcast barrier.
for tile in ['next_hi','next_lo']:
 oldtext=f'''      if ({tile} < host_tiles)
        fused_service_wait_tile<KTraits>(params, storage, scratch, index_base, kv_len, {tile}*32);
      __syncthreads();'''
 new=f'''      if ({tile} < host_tiles)
        fused_service_wait_tile<KTraits>(params, storage, scratch, index_base, kv_len, {tile}*32);
      else __syncthreads();'''
 assert oldtext in s;s=s.replace(oldtext,new)
# Sourcepredictedsizealignment same163072 (contextnow196B).
s=s.replace('static_assert(fused_service_smem == 163072);','static_assert(fused_service_smem == 163072);')
for v,value in [('base',base),('fused',s)]:
 tree=p/('include_'+v);shutil.copytree(root/'include',tree);(tree/'flashinfer/attention/mla_host_batch.cuh').write_text(value);(p/(v+'.cuh')).write_text(value)
for f in ['compile.py','resources_gpu3.py','experiment.json','plans.cpp']:shutil.copy2(old/f,p/f)
(p/'.gitignore').write_text('include_*\njit_*\n__pycache__/\nplans.so\n!*.log\n')
(p/'fused.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='production',tofile='fused',n=0)))
(p/'manifest.json').write_text(json.dumps(dict(production_sha256=hashlib.sha256(base.encode()).hexdigest(),fused_sha256=hashlib.sha256(s.encode()).hexdigest(),math_sha256=hashlib.sha256((root/'include/flashinfer/attention/mla.cuh').read_bytes()).hexdigest(),far_sha256=hashlib.sha256((root/'include/flashinfer/attention/far_link_emu.cuh').read_bytes()).hexdigest(),mode='normal no emulation compiled',mechanism='AllCTAservice retire/publish/refill reusepreQK,jointMDO,PVbarriers; nativeQKinnerlooprestored; staticuniqueownership'),indent=2)+'\n')
