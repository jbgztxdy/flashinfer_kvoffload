from pathlib import Path
import shutil
p=Path(__file__).resolve().parent;root=p.parents[2];old=root/'temp/glm_trace/tune_unlimited_uniform_20261103'
s=(root/'include/flashinfer/attention/far_link_emu.cuh').read_text()
a=s.index('// Called by every thread');b=s.index('// Host: load the settings',a)
s=s[:a]+'''// Scenario-calibrated local deadlines, NOT a work-conserving shared-link model.
__device__ unsigned long long stage_rows[kSlots][kMaxStages][8];
__device__ unsigned long long stage_issue[kSlots][kMaxStages][8];
// ps_per_byte is interpreted by this PRIVATE model as calibrated local delay ps/B.
template <uint32_t STAGES>
__device__ __forceinline__ void tile(uint32_t stage,uint32_t far_rows,uint32_t row_bytes){
 if(!config.enabled)return;
 const uint32_t rows=__reduce_add_sync(0xffffffffu,far_rows);
 const uint32_t warp=threadIdx.y+blockDim.y*threadIdx.z;
 if(threadIdx.x==0){stage_rows[cta_slot()][stage][warp]=(unsigned long long)rows*row_bytes;stage_issue[cta_slot()][stage][warp]=now_ns();}
}
struct State {unsigned long long bytes=0,issue=0;int stage=-1;};
__device__ __forceinline__ bool waiter(){return threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0;}
template<uint32_t STAGES>
__device__ __forceinline__ void wait_stage(uint32_t stage,State&){
 if(!config.enabled)return;
 __syncthreads();
 if(waiter()){
  unsigned long long bytes=0,issue=0;
  for(int w=0;w<8;w++){bytes+=stage_rows[cta_slot()][stage][w];auto t=stage_issue[cta_slot()][stage][w];if(t>issue)issue=t;stage_rows[cta_slot()][stage][w]=0;}
  if(bytes)wait_until(issue+(bytes*config.ps_per_byte+999)/1000+config.latency_ns);
 }
}
template<uint32_t STAGES>__device__ __forceinline__ void prefetch(uint32_t,State&){}
template<uint32_t STAGES>__device__ __forceinline__ void wait_all(){State s{};for(uint32_t i=0;i<STAGES;i++)wait_stage<STAGES>(i,s);}
template<uint32_t STAGES>__device__ __forceinline__ void init(){
 if(config.enabled&&waiter())for(uint32_t s=0;s<STAGES;s++)for(int w=0;w<8;w++){stage_rows[cta_slot()][s][w]=0;stage_issue[cta_slot()][s][w]=0;}
}

''' + s[b:]
# old configure GBps argument becomes explicit calibrated ps/B via private env override
s=s.replace('cudaStreamSynchronize(stream);','if(const char* rate=std::getenv("FLASHINFER_STALL_PS_PER_BYTE"))c.ps_per_byte=std::strtoull(rate,nullptr,10);\n  cudaStreamSynchronize(stream);')
(p/'stall.cuh').write_text(s)
shutil.copytree(old/'include_retained',p/'include',symlinks=True)
(p/'include/flashinfer/attention/far_link_emu.cuh').write_text(s)
