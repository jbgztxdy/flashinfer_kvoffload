// Private experimental work exchange. Whole-row partial_o is otherwise unused.
constexpr uint32_t kAssistCTAs=128,kAssistPairs=128;
struct alignas(128) AssistControl {
  uint32_t gen,row,index_base,kv_len,pairs,progress,lease[2];
  uint32_t helper_done,fetch_done,produced,consumed,canceled,waited;
  unsigned long long job[kAssistPairs];
};
__device__ __forceinline__ void assist_loader_sync(){asm volatile("bar.sync 2,128;":::"memory");}
__device__ __forceinline__ void assist_math_sync(){asm volatile("bar.sync 3,128;":::"memory");}
__device__ __forceinline__ uint64_t assist_ld64(const uint64_t* p){uint64_t x;asm volatile("ld.acquire.gpu.global.u64 %0,[%1];":"=l"(x):"l"(p):"memory");return x;}
__device__ __forceinline__ void assist_st64(uint64_t* p,uint64_t x){asm volatile("st.release.gpu.global.u64 [%0],%1;"::"l"(p),"l"(x):"memory");}
template<typename Params> __device__ __forceinline__ AssistControl* assist_controls(const Params& p){return reinterpret_cast<AssistControl*>(p.partial_o);}
template<typename Params> __device__ __forceinline__ float* assist_scores(const Params& p,uint32_t owner,uint32_t slot){return reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(p.partial_o)+kAssistCTAs*sizeof(AssistControl))+(owner*2+slot)*4096;}
__device__ __forceinline__ void assist_copy16(void* dst,const void* src,bool valid=true){uint32_t a=static_cast<uint32_t>(__cvta_generic_to_shared(dst));asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(a),"l"(src),"r"(valid?16:0));}
template<typename Params> __device__ __forceinline__ const typename Params::DTypeKV* assist_kv(const Params& p,typename Params::IdType page){return (page<0?p.staging:p.ckv)+((page<0?-static_cast<int64_t>(page)-1:static_cast<int64_t>(page&~kHostBatchOwnerBit))*576);}
// All128 math lanes execute this helper, with no cross-role shared alias.
template<typename Params>
__device__ __forceinline__ void assist_qk32(const Params& p,uint32_t row,uint32_t base,uint32_t len,uint32_t tile,float* out,typename Params::DTypeKV* work){
 using T=typename Params::DTypeKV;uint32_t t=threadIdx.y*32+threadIdx.x,lane=threadIdx.x,warp=threadIdx.y;
 T* qs=work;T* ks=work+4096;float s[2][8];
#pragma unroll 1
 for(uint32_t b=0;b<9;++b){
  const auto* q=b==0?p.q_pe+row*p.q_pe_stride_n:p.q_nope+row*p.q_nope_stride_n;
  const uint32_t stride=b==0?p.q_pe_stride_h:p.q_nope_stride_h,feat=b==0?0:(b-1)*64;
  for(uint32_t v=t;v<512;v+=128)assist_copy16(qs+(v/8)*64+((v%8)^((v/8)%8))*8,q+(v/8)*stride+feat+(v%8)*8);
  for(uint32_t v=t;v<256;v+=128){uint32_t key=tile*32+v/8;bool valid=key<len;auto page=valid?p.remap[base+key]:0;assist_copy16(ks+(v/8)*64+((v%8)^((v/8)%8))*8,assist_kv(p,page)+(b==0?512:(b-1)*64)+(v%8)*8,valid);}
  cp_async::commit_group();cp_async::wait_group<0>();assist_math_sync();
#pragma unroll
  for(uint32_t dim=0;dim<4;++dim){uint32_t a[4],k[4];mma::ldmatrix_m8n8x4(a,qs+(warp*16+lane%16)*64+((dim*16+8*(lane/16))^((lane%8)*8)));
#pragma unroll
   for(uint32_t g=0;g<2;++g){mma::ldmatrix_m8n8x4(k,ks+(g*16+8*(lane/16)+lane%8)*64+((dim*16+8*((lane%16)/8))^((lane%8)*8)));if(b==0&&dim==0)mma::mma_sync_m16n16k16_row_col_f16f16f32<T,mma::MMAMode::kInit>(s[g],a,k);else mma::mma_sync_m16n16k16_row_col_f16f16f32<T>(s[g],a,k);}
  }
  assist_math_sync();
 }
#pragma unroll
 for(uint32_t g=0;g<2;++g)for(uint32_t j=0;j<8;++j)out[(g*128+t)*8+j]=s[g][j];
}
// Called only by the worker leader, after its own cp.async wait. Its decision
// is broadcast by the ORIGINAL pre-QK CTA barrier (no per-pair extra barrier).
template<typename Params>
__device__ __forceinline__ void assist_take(const Params& p,uint32_t* scratch,uint32_t row,uint32_t pair){
 auto* c=assist_controls(p)+blockIdx.y;uint32_t gen=host_batch_ld_acquire(&c->gen);scratch[kHostBatchScratchClaimed]=0;
 if(!gen)return;
 c->progress=pair;__threadfence();auto* word=&c->job[pair];uint64_t prefix=static_cast<uint64_t>(gen)<<32;
 bool waited=false;
 while(true){uint64_t val=assist_ld64(word),state=val&7;
  if((val>>32)!=gen)return;
  if(state==0){if(atomicCAS(word,prefix,prefix|3)==prefix)return;continue;}
  if(state==1){waited=true;__nanosleep(64);continue;}
  if(state==2){scratch[kHostBatchScratchClaimed]=static_cast<uint32_t>((val>>3)&1)+1;c->consumed++;c->waited+=waited;return;}
  return;
 }
}
template<typename Params>
__device__ __forceinline__ void assist_begin_row(const Params& p,uint32_t row,uint32_t base,uint32_t len){
 auto* all=assist_controls(p);auto* c=all+blockIdx.y;
 if(host_batch_ld_acquire(&all->helper_done)>=p.fetch_ctas || !p.state[kHostBatchUnique]){c->gen=0;return;}
 const uint32_t gen=(p.state[kHostBatchEpoch]+1)*65536+row+1;
 c->gen=0;__threadfence();c->row=row;c->index_base=base;c->kv_len=len;c->pairs=ceil_div(len,32u)/2;c->progress=0;
 for(uint32_t i=0;i<c->pairs;++i)c->job[i]=static_cast<uint64_t>(gen)<<32;
 __threadfence();asm volatile("st.release.gpu.global.u32 [%0],%1;"::"l"(&c->gen),"r"(gen):"memory");
}
template<typename KTraits,typename Params>
__device__ __forceinline__ void assist_math_service(const Params& p,uint8_t* smem,volatile uint32_t* finished){
 // Only the four z1 warps enter. Each producer owns a disjoint set of worker
 // queues; current row's64-bit generation prevents delayed claims on reused rows.
 const uint32_t t=threadIdx.y*32+threadIdx.x,fetchid=blockIdx.y-(gridDim.y-p.fetch_ctas),workers=gridDim.y-p.fetch_ctas;
 __shared__ uint32_t job_owner,job_pair,job_slot,job_gen,job_row,job_base,job_len,run,ready[4];
 auto* all=assist_controls(p);uint32_t scan=0;
 while(true){
  if(t==0){run=!*finished;job_owner=0xffffffffu;
   if(run){uint32_t count=fetchid<workers?ceil_div(workers-fetchid,p.fetch_ctas):0;
    for(uint32_t n=0;n<count;++n){uint32_t owner=fetchid+((scan+n)%count)*p.fetch_ctas;auto* c=all+owner;uint32_t gen=host_batch_ld_acquire(&c->gen);if(!gen)continue;
     uint32_t slot=2;if(atomicCAS(&c->lease[0],0u,1u)==0)slot=0;else if(atomicCAS(&c->lease[1],0u,1u)==0)slot=1;if(slot==2)continue;
     uint32_t first=host_batch_ld_acquire(&c->progress)+4,limit=c->pairs;bool found=false;
     for(uint32_t pair=first;pair<limit;++pair){uint64_t expected=static_cast<uint64_t>(gen)<<32;if(atomicCAS(&c->job[pair],expected,expected|1|(slot<<3))!=expected)continue;
      job_owner=owner;job_pair=pair;job_slot=slot;job_gen=gen;job_row=c->row;job_base=c->index_base;job_len=c->kv_len;found=true;break;}
     if(found){scan=(scan+n+1)%count;break;}atomicExch(&c->lease[slot],0u);
    }
   }
  }
  assist_math_sync();if(!run)break;
  if(job_owner==0xffffffffu){__nanosleep(64);continue;}
  uint32_t hi=ceil_div(job_len,32u)-1-job_pair*2;bool ok=true;
  if(t<64){uint32_t key=(hi-1)*32+t;if(key<job_len){auto page=p.remap[job_base+key];uint32_t r=static_cast<uint32_t>(-(page+1));if(page<0&&r>=p.cache_slots){auto* flag=host_batch_flags(p)+(r-p.cache_slots)/64;ok=host_batch_ld_acquire(flag)==p.state[kHostBatchEpoch]+1;}}}
  bool warp_ok=__all_sync(0xffffffffu,ok);if(threadIdx.x==0)ready[threadIdx.y]=warp_ok;assist_math_sync();
  if(t==0)run=ready[0]&&ready[1]&&ready[2]&&ready[3];assist_math_sync();
  auto* c=all+job_owner;auto* word=&c->job[job_pair];uint64_t prefix=static_cast<uint64_t>(job_gen)<<32;
  if(!run){if(t==0){assist_st64(word,prefix);atomicExch(&c->lease[job_slot],0u);c->canceled++;}assist_math_sync();continue;}
  float* output=assist_scores(p,job_owner,job_slot);
  assist_qk32(p,job_row,job_base,job_len,hi,output,reinterpret_cast<typename Params::DTypeKV*>(smem+147968));
  assist_qk32(p,job_row,job_base,job_len,hi-1,output+2048,reinterpret_cast<typename Params::DTypeKV*>(smem+147968));
  __threadfence();assist_math_sync();
  if(t==0){assist_st64(word,prefix|2|(job_slot<<3));(all+blockIdx.y)->produced++;}
  assist_math_sync();
 }
 if(t==0)atomicAdd(&all->helper_done,1u);assist_math_sync();
}
