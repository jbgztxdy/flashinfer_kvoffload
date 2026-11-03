// Single-CTA state lives in shared, avoiding a long-lived per-thread service
// payload/context across the128 FP32 O accumulators. All256threads compute+copy.
struct UniformService {
  uint32_t count,epoch,head,issue_stage,ready,waiting;
  uint32_t chunk[2],pending[2];
  int32_t pages[2][10],victims[10];
  uint32_t produced,rounds,wait_loops,rows;
};
template<typename Q,typename KV,typename O>
struct UniformStorage {
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
 UniformService service;
};
template<typename Q,typename KV,typename O,typename Id>
struct UniformTraits : JointQKTraits<Q,KV,O,Id> {
 using SharedStorage=UniformStorage<Q,KV,O>;
};
template<typename Params> struct UniformHostParams : JointQKHostParams<Params> {};
template<typename Params>
__device__ __forceinline__ uint32_t* uniform_flags(const Params& p){return reinterpret_cast<uint32_t*>(p.partial_o);}
template<typename Params>
__device__ __forceinline__ unsigned long long* uniform_diag(const Params& p){return reinterpret_cast<unsigned long long*>(uniform_flags(p)+round_up(ceil_div(p.staging_capacity,10u),2u))+blockIdx.y*8;}
// Prepare one empty stage for distributed copy. The following barrier publishes
// IDs/selector; callers never reuse it while another warp still reads them.
template<typename KTraits,typename Params>
__device__ __forceinline__ void uniform_prepare_issue(const Params& p,typename KTraits::SharedStorage& s){
 auto& c=s.service;uint32_t tid=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x;
 if(tid==0){c.issue_stage=2;for(uint32_t off=0;off<2;++off){uint32_t b=(c.head+off)%2;if(!c.pending[b]&&c.chunk[b]*10<c.count){c.issue_stage=b;break;}}}
 __syncthreads();uint32_t b=c.issue_stage;
 if(b<2&&tid<10){uint32_t slot=c.chunk[b]*10+tid;c.pages[b][tid]=slot<c.count?p.unique_pages[slot]:0;}
 __syncthreads();
}
template<uint32_t ROUND,typename KTraits,typename Params>
__device__ __forceinline__ void uniform_issue_round(const Params& p,typename KTraits::SharedStorage& s){
 auto& c=s.service;uint32_t b=c.issue_stage;if(b>=2)return;
 uint32_t tid=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x;
 const uint32_t v=ROUND*256+tid,slot=c.chunk[b]*10+v/72;
 if(v<720 && slot<c.count){
  const auto page=c.pages[b][v/72];
  const uint4* src=reinterpret_cast<const uint4*>(p.ckv+static_cast<int64_t>(page)*576)+v%72;
  uint32_t dst=static_cast<uint32_t>(__cvta_generic_to_shared(reinterpret_cast<uint4*>(s.remote[b])+v));
  uint64_t policy;asm volatile("createpolicy.fractional.L2::evict_first.b64 %0,1.0;":"=l"(policy));
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0],[%1],16,%2;"::"r"(dst),"l"(src),"l"(policy));
 }
 if constexpr(ROUND==2){
  FAR_EMU_TILE(2,b,tid==0?min(10u,c.count-c.chunk[b]*10):0,1152);
  cp_async::commit_group();
  if(tid==0){c.pending[b]=1;c.rounds++;}
 }
}
// A ready poll never parks on a virtual link deadline. It retires at most one
// stage; blocking progress while waiting/draining uses the same legal hook.
template<typename KTraits,typename Params>
__device__ __forceinline__ void uniform_retire(const Params& p,typename KTraits::SharedStorage& s){
 auto& c=s.service;uint32_t tid=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x,b=c.head;
 if(tid==0){c.ready=c.pending[b];
#ifdef FLASHINFER_FAR_EMU
  if(c.ready&&::flashinfer::far_emu::config.enabled)c.ready=::flashinfer::far_emu::now_ns()>=*(volatile unsigned long long*)&::flashinfer::far_emu::pending_ready[::flashinfer::far_emu::cta_slot()][b];
#endif
 }
 __syncthreads();if(!c.ready)return;
 // cp.async groups from attention and service share the hardware queue.
 // Caller/drain explicitly waits all committed real reads before publication.
#ifdef FLASHINFER_FAR_EMU
 ::flashinfer::far_emu::State state{};::flashinfer::far_emu::wait_stage<2>(b,state);
#endif
 __syncthreads();uint32_t chunk=c.chunk[b],first=chunk*10;
 if(tid<10){uint32_t slot=first+tid;int32_t victim=slot<c.count?p.victim_of[slot]:-1;c.victims[tid]=victim;
  if(slot<c.count){auto page=c.pages[b][tid];p.host_slot[page-p.host_page_start]=0;
   if(victim>=0){int32_t evicted=p.slot_page[victim];if(evicted>=0)p.resident[evicted-p.host_page_start]=-1;p.resident[page-p.host_page_start]=victim;p.slot_page[victim]=page;}
  }
 }
 __syncthreads();
#pragma unroll
 for(uint32_t r=0;r<3;++r){uint32_t v=r*256+tid;uint32_t slot=first+v/72;
  if(v<720&&slot<c.count){uint4 value=reinterpret_cast<uint4*>(s.remote[b])[v];
   reinterpret_cast<uint4*>(p.staging+(p.cache_slots+static_cast<int64_t>(slot))*576)[v%72]=value;
   int32_t victim=c.victims[v/72];if(victim>=0)reinterpret_cast<uint4*>(p.staging+static_cast<int64_t>(victim)*576)[v%72]=value;
  }
 }
 __threadfence();__syncthreads();
 if(tid==0){asm volatile("st.release.gpu.global.u32 [%0],%1;"::"l"(uniform_flags(p)+chunk),"r"(c.epoch+1):"memory");c.pending[b]=0;c.chunk[b]+=2*gridDim.y;c.head=b^1;c.produced++;c.rows+=min(10u,c.count-first);}
 __syncthreads();
}
template<typename KTraits,typename Params>
__device__ __forceinline__ void uniform_progress(const Params& p,typename KTraits::SharedStorage& s){
 cp_async::wait_group<0>();__syncthreads();uniform_retire<KTraits>(p,s);uniform_prepare_issue<KTraits>(p,s);
}
template<typename KTraits,typename Params>
__device__ __forceinline__ void uniform_issue_full(const Params& p,typename KTraits::SharedStorage& s){
 uniform_issue_round<0,KTraits>(p,s);uniform_issue_round<1,KTraits>(p,s);uniform_issue_round<2,KTraits>(p,s);__syncthreads();
}
template<typename KTraits,typename Params>
__device__ __forceinline__ void uniform_wait_tile(const Params& p,typename KTraits::SharedStorage& s,const uint32_t* scratch,uint32_t base,uint32_t len,uint32_t tile){
 auto& c=s.service;uint32_t tid=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x;
 while(true){bool waiting=false;
  if(threadIdx.y==0&&threadIdx.z==0){uint32_t k=tile+threadIdx.x;auto page=k<len?p.remap[base+k]:0;uint32_t row=static_cast<uint32_t>(-(page+1));if(page<0&&row>=p.cache_slots)waiting=host_batch_ld_acquire(uniform_flags(p)+(row-p.cache_slots)/10)!=scratch[kHostBatchScratchEpoch]+1;bool any=__any_sync(0xffffffffu,waiting);if(threadIdx.x==0)c.waiting=any;}
  __syncthreads();if(!c.waiting)return;
  uniform_progress<KTraits>(p,s);uniform_issue_full<KTraits>(p,s);
  if(tid==0)c.wait_loops++;
  __nanosleep(64);__syncthreads();
 }
}
template<typename KTraits,typename Params>
__device__ __forceinline__ void uniform_initialize(const Params& p,typename KTraits::SharedStorage& s,uint32_t unique,uint32_t epoch){
 uint32_t tid=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x;auto& c=s.service;
 if(tid==0){c.count=unique;c.epoch=epoch;c.head=0;c.pending[0]=c.pending[1]=0;c.chunk[0]=blockIdx.y;c.chunk[1]=blockIdx.y+gridDim.y;c.produced=c.rounds=c.wait_loops=c.rows=0;}
 FAR_EMU_INIT(2);__syncthreads();
 uniform_prepare_issue<KTraits>(p,s);uniform_issue_full<KTraits>(p,s);
 uniform_prepare_issue<KTraits>(p,s);uniform_issue_full<KTraits>(p,s);
}
template<typename KTraits,typename Params>
__device__ __forceinline__ void uniform_drain(const Params& p,typename KTraits::SharedStorage& s){
 auto& c=s.service;
 while(c.chunk[c.head]*10<c.count){uniform_progress<KTraits>(p,s);uniform_issue_full<KTraits>(p,s);__nanosleep(64);}
}
// Private no-PE-shared load: native PE fragments already come from q_pe regs.
template<typename KTraits>
__device__ __forceinline__ void uniform_load_q(typename KTraits::SharedStorage* s,typename KTraits::DTypeQ* qn,uint32_t stride_n,uint32_t stride_h,uint_fastdiv heads){
 smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> q(s->q_smem_nope);uint32_t lane=threadIdx.x,wg=threadIdx.z,w=threadIdx.y;
#pragma unroll
 for(uint32_t a=0;a<2;++a){uint32_t row,head;heads.divmod(lane/8+(wg+a*2)*16+w*4,row,head);auto* ptr=qn+row*stride_n+head*stride_h+(lane%8)*8;
#pragma unroll
  for(uint32_t dim=0;dim<8;++dim){uint32_t at=q.template get_permuted_offset<KTraits::UPCAST_STRIDE_Q_NOPE>(32*a+wg*16+w*4+lane/8,dim*8+lane%8);q.template load_128b_async<SharedMemFillMode::kFillZero>(at,ptr,row<1);ptr+=64;}
 }
}
