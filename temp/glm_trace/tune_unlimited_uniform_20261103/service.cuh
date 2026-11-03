// All256threads compute attention and move unique KV. The normal build contains
// no virtual-link instructions. Service phases reuse existing math barriers.
struct FusedService {
 uint32_t count,epoch,done,current,waiting;
 int32_t pages[2][10],victims[2][10];
 uint32_t produced,issues,wait_loops,rows;
};
template<typename Q,typename KV,typename O>
struct FusedServiceStorage {
 union {
  struct {
   alignas(16) Q q_smem_nope[64*512];
   alignas(16) KV ckv_smem[2][32*512];
   alignas(16) KV kpe_p_smem[2][32*64];
   union {alignas(16) float m_wg[2][64];alignas(16) float d_wg[2][64];};
  };
  alignas(16) O o_smem[64*512];
 };
 alignas(128) KV remote[2][10*576];
 FusedService service;
};
template<typename Q,typename KV,typename O,typename Id>
struct FusedServiceTraits : JointQKTraits<Q,KV,O,Id> {using SharedStorage=FusedServiceStorage<Q,KV,O>;};
template<typename Params> struct FusedServiceHostParams : JointQKHostParams<Params> {};
template<typename Params> __device__ __forceinline__ uint32_t* fused_service_flags(const Params& p){return reinterpret_cast<uint32_t*>(p.partial_o);}
template<typename Params> __device__ __forceinline__ unsigned long long* fused_service_diag(const Params& p){return reinterpret_cast<unsigned long long*>(fused_service_flags(p)+round_up(ceil_div(p.staging_capacity,10u),2u))+blockIdx.y*8;}
template<typename KTraits,typename Params>
__device__ __forceinline__ void fused_service_ids(const Params& p,typename KTraits::SharedStorage& s,uint32_t ticket){
 uint32_t t=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x,chunk=blockIdx.y+ticket*gridDim.y,b=ticket%2;
 if(t<10&&chunk*10+t<s.service.count){uint32_t slot=chunk*10+t;s.service.pages[b][t]=p.unique_pages[slot];s.service.victims[b][t]=p.victim_of[slot];}
}
template<typename KTraits,typename Params>
__device__ __forceinline__ void fused_service_issue(const Params& p,typename KTraits::SharedStorage& s,uint32_t ticket){
 uint32_t t=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x,chunk=blockIdx.y+ticket*gridDim.y,b=ticket%2;
 if(chunk*10>=s.service.count)return;
 uint64_t policy;asm volatile("createpolicy.fractional.L2::evict_first.b64 %0,1.0;":"=l"(policy));
#pragma unroll
 for(uint32_t r=0;r<3;++r){uint32_t v=r*256+t,slot=chunk*10+v/72;if(v<720&&slot<s.service.count){auto page=s.service.pages[b][v/72];auto* src=reinterpret_cast<const uint4*>(p.ckv+static_cast<int64_t>(page)*576)+v%72;uint32_t dst=static_cast<uint32_t>(__cvta_generic_to_shared(reinterpret_cast<uint4*>(s.remote[b])+v));asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0],[%1],16,%2;"::"r"(dst),"l"(src),"l"(policy));}}
 FAR_EMU_TILE(2,b,t==0?min(10u,s.service.count-chunk*10):0,1152);cp_async::commit_group();if(t==0)s.service.issues++;
}
// Every lane has already completed cp.async.wait_group0 and the original QK
// barrier. No stage metadata is overwritten before the subsequent MDO barrier.
template<typename KTraits,typename Params>
__device__ __forceinline__ void fused_service_stores(const Params& p,typename KTraits::SharedStorage& s){
 uint32_t t=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x,ticket=s.service.current,chunk=blockIdx.y+ticket*gridDim.y,b=ticket%2;
 if(chunk*10>=s.service.count)return;
#ifdef FLASHINFER_FAR_EMU
 ::flashinfer::far_emu::State state{};::flashinfer::far_emu::wait_stage<2>(b,state);__syncthreads();
#endif
#pragma unroll
 for(uint32_t r=0;r<3;++r){uint32_t v=r*256+t,slot=chunk*10+v/72;if(v<720&&slot<s.service.count){uint4 value=reinterpret_cast<uint4*>(s.remote[b])[v];reinterpret_cast<uint4*>(p.staging+(p.cache_slots+static_cast<int64_t>(slot))*576)[v%72]=value;int32_t victim=s.service.victims[b][v/72];if(victim>=0)reinterpret_cast<uint4*>(p.staging+static_cast<int64_t>(victim)*576)[v%72]=value;}}
 if(t<10&&chunk*10+t<s.service.count){auto page=s.service.pages[b][t];p.host_slot[page-p.host_page_start]=0;int32_t victim=s.service.victims[b][t];if(victim>=0){int32_t evicted=p.slot_page[victim];if(evicted>=0)p.resident[evicted-p.host_page_start]=-1;p.resident[page-p.host_page_start]=victim;p.slot_page[victim]=page;}}
 __threadfence();
}
// Called immediately after jointMDO's original barrier; all payload stores are
// fenced and visible before the leader releases the ready flag.
template<typename KTraits,typename Params>
__device__ __forceinline__ void fused_service_publish_prepare(const Params& p,typename KTraits::SharedStorage& s){
 uint32_t t=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x,ticket=s.service.current,chunk=blockIdx.y+ticket*gridDim.y;
 if(chunk*10>=s.service.count)return;
 if(t==0){asm volatile("st.release.gpu.global.u32 [%0],%1;"::"l"(fused_service_flags(p)+chunk),"r"(s.service.epoch+1):"memory");s.service.produced++;s.service.rows+=min(10u,s.service.count-chunk*10);}
 fused_service_ids<KTraits>(p,s,ticket+2);
}
// The first original PV barrier published the next packet's metadata. Its
// second original barrier also publishes the leader's done increment.
template<typename KTraits,typename Params>
__device__ __forceinline__ void fused_service_refill(const Params& p,typename KTraits::SharedStorage& s){
 uint32_t t=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x,ticket=s.service.current;
 if((blockIdx.y+ticket*gridDim.y)*10>=s.service.count)return;
 fused_service_issue<KTraits>(p,s,ticket+2);if(t==0)s.service.done=ticket+1;
}
template<typename KTraits,typename Params>
__device__ __forceinline__ void fused_service_step(const Params& p,typename KTraits::SharedStorage& s){
 uint32_t t=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x;if(t==0)s.service.current=s.service.done;
 cp_async::wait_group<0>();__syncthreads();fused_service_stores<KTraits>(p,s);__syncthreads();
 fused_service_publish_prepare<KTraits>(p,s);__syncthreads();fused_service_refill<KTraits>(p,s);__syncthreads();
}
template<typename KTraits,typename Params>
__device__ __forceinline__ void fused_service_wait_tile(const Params& p,typename KTraits::SharedStorage& s,const uint32_t* scratch,uint32_t base,uint32_t len,uint32_t tile){
 uint32_t t=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x;
 while(true){if(threadIdx.y==0&&threadIdx.z==0){uint32_t k=tile+threadIdx.x;auto page=k<len?p.remap[base+k]:0;uint32_t row=static_cast<uint32_t>(-(page+1));bool wait=page<0&&row>=p.cache_slots&&host_batch_ld_acquire(fused_service_flags(p)+(row-p.cache_slots)/10)!=scratch[kHostBatchScratchEpoch]+1;bool any=__any_sync(0xffffffffu,wait);if(threadIdx.x==0)s.service.waiting=any;}
 __syncthreads();if(!s.service.waiting)return;fused_service_step<KTraits>(p,s);if(t==0)s.service.wait_loops++;__nanosleep(64);}
}
template<typename KTraits,typename Params>
__device__ __forceinline__ void fused_service_initialize(const Params& p,typename KTraits::SharedStorage& s,uint32_t unique,uint32_t epoch){
 uint32_t t=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x;if(t==0){s.service.count=unique;s.service.epoch=epoch;s.service.done=0;s.service.current=0;s.service.waiting=0;s.service.produced=s.service.issues=s.service.wait_loops=s.service.rows=0;}
 FAR_EMU_INIT(2);__syncthreads();fused_service_ids<KTraits>(p,s,0);fused_service_ids<KTraits>(p,s,1);__syncthreads();fused_service_issue<KTraits>(p,s,0);fused_service_issue<KTraits>(p,s,1);__syncthreads();
}
template<typename KTraits,typename Params>
__device__ __forceinline__ void fused_service_drain(const Params& p,typename KTraits::SharedStorage& s){while((blockIdx.y+s.service.done*gridDim.y)*10<s.service.count)fused_service_step<KTraits>(p,s);}
template<typename KTraits>
__device__ __forceinline__ void fused_service_load_q(typename KTraits::SharedStorage* s,typename KTraits::DTypeQ* qn,uint32_t stride_n,uint32_t stride_h,uint_fastdiv heads){
 smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> q(s->q_smem_nope);uint32_t lane=threadIdx.x,wg=threadIdx.z,w=threadIdx.y;
#pragma unroll
 for(uint32_t a=0;a<2;++a){uint32_t row,head;heads.divmod(lane/8+(wg+a*2)*16+w*4,row,head);auto* ptr=qn+row*stride_n+head*stride_h+(lane%8)*8;
#pragma unroll
 for(uint32_t dim=0;dim<8;++dim){uint32_t at=q.template get_permuted_offset<KTraits::UPCAST_STRIDE_Q_NOPE>(32*a+wg*16+w*4+lane/8,dim*8+lane%8);q.template load_128b_async<SharedMemFillMode::kFillZero>(at,ptr,row<1);ptr+=64;}}
}
