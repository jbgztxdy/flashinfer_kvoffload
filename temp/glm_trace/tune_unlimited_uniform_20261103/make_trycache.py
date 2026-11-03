"""Nonblocking same-launch reuse in direct MLA; exact actual-source counting."""
from pathlib import Path
import hashlib,json,shutil
p=Path(__file__).resolve().parent;root=p.parents[2];assert not(p/'trycache.cuh').exists()
s=(p/'direct_pv.cuh').read_text()
a=s.index('template <typename KTraits>\n__device__ __forceinline__ void direct_pv_pv(')
helper='''template<typename Params>__device__ __forceinline__ uint32_t* trycache_flags(const Params& p){return reinterpret_cast<uint32_t*>(p.partial_o);}
template<typename Params>__device__ __forceinline__ unsigned long long* trycache_counts(const Params& p){return reinterpret_cast<unsigned long long*>(trycache_flags(p)+round_up(p.cache_slots,2u))+blockIdx.y*8;}
template<typename KTraits,typename Params>
__device__ __forceinline__ void trycache_publish(const Params& p,uint32_t a,uint32_t b){
 if(threadIdx.x%8!=0)return;
#pragma unroll
 for(uint32_t stage=0;stage<2;++stage){uint32_t at=(stage?b:a)*32+host_batch_tile_row<KTraits>(0);int32_t victim=host_batch_tile_victims<KTraits>()[at];
  if(victim>=0)asm volatile("st.release.gpu.global.u32 [%0],%1;"::"l"(trycache_flags(p)+victim),"r"(p.state[kHostBatchEpoch]+1):"memory");
 }
}
template<typename KTraits,typename Params>
__device__ __forceinline__ void trycache_read_pages(const Params& p,uint32_t base,uint32_t len,uint32_t tile,typename KTraits::IdType* pages,uint32_t& far_count){
#pragma unroll
 for(uint32_t j=0;j<kHostBatchTileRows<KTraits>;++j){
  const uint32_t k=tile+host_batch_tile_row<KTraits>(j);auto page=k<len?p.remap[base+k]:0;
  if(k<len&&page>=0&&(page&~kHostBatchOwnerBit)>=p.host_page_start){
   if(!(page&kHostBatchOwnerBit)){
    int32_t slot=p.resident[page-p.host_page_start];
    if(slot>=0&&host_batch_ld_acquire(trycache_flags(p)+slot)==p.state[kHostBatchEpoch]+1)page=-slot-1;
   }
   if(page>=0&&threadIdx.x%8==0)far_count++;
  }
  pages[j]=page;
 }
}
'''
s=s[:a]+helper+s[a:]
a=s.index('void direct_pv_update_joint_host_mdo(');t=s.rfind('template <typename KTraits>',0,a);s=s[:t]+s[t:].replace('template <typename KTraits>','template <typename KTraits,typename Params>',1)
a=s.index('void direct_pv_update_joint_host_mdo(');b=s.index('template <typename KTraits, typename Params>\n__device__ __forceinline__ void direct_pv_attend',a);x=s[a:b]
x=x.replace('float (*s0)[8], float (*s1)[8], float (*o)[8], float* m, float* d) {','float (*s0)[8], float (*s1)[8], float (*o)[8], float* m, float* d,const Params& p,uint32_t a,uint32_t b) {').replace('  __syncthreads();','  __syncthreads();\n  trycache_publish<KTraits>(p,a,b);',1)
s=s[:a]+x+s[b:]
a=s.index('void direct_pv_attend_joint_host_tiles(');b=s.index('// partial_capacity bounds',a);x=s[a:b]
x=x.replace('float (*o)[8], float* m, float* d) {','float (*o)[8], float* m, float* d,uint32_t& far_count) {',1)
x=x.replace('read_host_batch_pages<KTraits>(params, index_base, kv_len, tile*32, pages);','trycache_read_pages<KTraits>(params,index_base,kv_len,tile*32,pages,far_count);\n      __syncthreads();')
for tile in ['next_hi','next_lo']:
 x=x.replace(f'read_host_batch_pages<KTraits>(params, index_base, kv_len, {tile}*32, pages);',f'trycache_read_pages<KTraits>(params,index_base,kv_len,{tile}*32,pages,far_count);')
x=x.replace('direct_pv_update_joint_host_mdo<KTraits>(&storage, variant, scores, scores+1, o, m, d);','direct_pv_update_joint_host_mdo<KTraits>(&storage, variant, scores, scores+1, o, m, d,params,hi%2,lo%2);')
# Single oddtile: update_mdo has a CTA barrier before P writes; publish afterward.
x=x.replace('    update_mdo_states_<KTraits>(&storage, 0, variant, scores, o, m, d);','    update_mdo_states_<KTraits>(&storage, 0, variant, scores, o, m, d);\n    __syncthreads();trycache_publish<KTraits>(params,0,0);')
x=x.replace('float (*o_frag)[8], float* m, float* d) {','float (*o_frag)[8], float* m, float* d,uint32_t& far_count) {',1)
x=x.replace('                                     o_frag, m, d);','                                     o_frag, m, d,far_count);',1)
a2=x.index('void direct_pv_run_joint_host_items(')
at=x.index('  constexpr uint32_t NUM_MMA_KV',a2);x=x[:at]+'  uint32_t far_count=0;\n'+x[at:]
x=x.replace('                                 m, d);','                                 m, d,far_count);',1)
a2=x.index('__global__ __launch_bounds__');pre=x[:a2];last=pre.rfind('\n}');pre=pre[:last]+'''
  uint32_t total=far_count;
#pragma unroll
  for(uint32_t offset=16;offset>0;offset/=2)total+=__shfl_down_sync(0xffffffffu,total,offset);
  if(threadIdx.x==0)trycache_counts(params)[threadIdx.z*blockDim.y+threadIdx.y]=total;
'''+pre[last:];x=pre+x[a2:]
x=x.replace('  const uint32_t epoch = params.state[kHostBatchEpoch];','''  const uint32_t epoch = params.state[kHostBatchEpoch];
  for(uint32_t c=blockIdx.y*KTraits::NUM_THREADS+tid;c<params.cache_slots;c+=gridDim.y*KTraits::NUM_THREADS)trycache_flags(params)[c]=0;
''',1)
s=s[:a]+x+s[b:]
# Normal-only gate: all original production branches remain after this guard.
a=s.index('    using DirectPVKernelTraits');b=s.index('    using PVFeatureKernelTraits',a);s=s[:a]+'#ifndef FLASHINFER_FAR_EMU\n'+s[a:b]+'#endif\n'+s[b:]
s=s.replace('DirectPV','TrycachePV').replace('direct_pv_','trycache_pv_')
(p/'trycache.cuh').write_text(s);shutil.copytree(root/'include',p/'include_trycache');(p/'include_trycache/flashinfer/attention/mla_host_batch.cuh').write_text(s)
m=json.loads((p/'manifest.json').read_text());m['trycache_sha256']=hashlib.sha256(s.encode()).hexdigest();m['trycache_mechanism']='Normal-only directPV + nonblocking same-launch cache reuse; ownerwrites -> jointMDO CTAbarrier -> pervictim release, candidate readacquire before existingCTAbarrier; all actualfarread selections counted perwarp atkernelend. No waiting for futureowner and no assumed unique traffic.';(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
x=(p/'bench_direct_pv.py').read_text().replace("'direct_pv']","'direct_pv','trycache']")
x=x.replace("refs[t]=((w._host_batch_buffers[3]>=0)&((w._host_batch_buffers[3]&~(1<<30))>=near)).sum()", "refs[t]=w._host_batch_buffers[6].view(torch.int64).reshape(-1)[R*CAP//2:R*CAP//2+108*8].sum()")
(p/'bench_trycache.py').write_text(x)
x=(p/'test_direct_pv.py').read_text().replace('test_direct_pv_output','test_trycache_output');(p/'test_trycache.py').write_text(x)
(p/'run_trycache.py').write_text((p/'run_direct_pv.py').read_text().replace('direct_pv','trycache'))
