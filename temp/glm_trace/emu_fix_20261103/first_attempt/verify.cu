// Focused simulator tests. All records are diagnostics, not MLA benchmark timings.
#define FLASHINFER_FAR_EMU
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "flashinfer/attention/far_link_emu.cuh"
namespace cg=cooperative_groups;using U=unsigned long long;
__device__ U issue_time;
void check(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA:%s\n",cudaGetErrorString(e));exit(1);}}
__global__ void reservations(U* out,int trials,int busy,int mixed){
 auto grid=cg::this_grid();
 for(int n=0;n<trials;++n){
  if(!blockIdx.x&&!threadIdx.x){issue_time=flashinfer::far_emu::now_ns();flashinfer::far_emu::link_clock=busy?issue_time+1000000:0;}
  grid.sync();
  if(!threadIdx.x){U bytes=mixed?(blockIdx.x%3==0?1152:(blockIdx.x%3==1?18432:73728)):73728;out[n*(gridDim.x+2)+blockIdx.x]=flashinfer::far_emu::reserve(bytes,issue_time);}
  grid.sync();
  if(!blockIdx.x&&!threadIdx.x){out[n*(gridDim.x+2)+gridDim.x]=issue_time;out[n*(gridDim.x+2)+gridDim.x+1]=flashinfer::far_emu::link_clock;}
  grid.sync();
 }
}
__global__ void arithmetic(U* out){
 if(threadIdx.x||blockIdx.x)return;
 // Adversarial arrival sequence includes idle gaps, nonmonotone timestamps,
 // zero-byte and unlimited-bandwidth requests. Compare the recurrence on CPU.
 const U ts[8]={1000,1000,800,100000,99900,100001,1000000,1000000};
 const U bytes[8]={1,1152,73728,18432,0,73728,73728,1152};
 flashinfer::far_emu::link_clock=0;
 for(int i=0;i<8;++i)out[i]=flashinfer::far_emu::reserve(bytes[i],ts[i]);
 out[8]=flashinfer::far_emu::link_clock;
}
// Exercise actual stage hooks with uneven/late warps and two pipeline stages.
__global__ void stages(U* records){
 FAR_EMU_INIT(2);FAR_EMU_SYNC();
 for(int i=0;i<16;++i){
  const int stage=i%2;const bool late=threadIdx.x>=32;
  if(late)flashinfer::far_emu::wait_until(flashinfer::far_emu::now_ns()+2000);
  FAR_EMU_TILE(2,stage,threadIdx.x%32==0?1:0,1152);
  if(threadIdx.x%32==0)records[i*10+threadIdx.x/32]=flashinfer::far_emu::pending_ready[flashinfer::far_emu::cta_slot()][stage];
  FAR_EMU_WAIT(2,stage);
  __syncthreads();
  if(threadIdx.x==0){records[i*10+8]=flashinfer::far_emu::now_ns();records[i*10+9]=flashinfer::far_emu::pending_ready[flashinfer::far_emu::cta_slot()][stage];}
 }
}
// Actual payload,64row packets,2stages,24CTAs. Diagnostic values are recorded
// after issue/wait and never used to alter deadlines or timing.
__global__ void payload(const uint4* src,uint4* dst,int rows,U* times,int hooks){
 extern __shared__ uint4 buffer[];
 FAR_EMU_INIT(2);FAR_EMU_SYNC();
 const int chunk_bytes=64*1152,vecs=chunk_bytes/16;
 auto issue=[&](int chunk,int buf){
  for(int v=threadIdx.x;v<vecs && chunk*64+v/72<rows;v+=blockDim.x){
   unsigned addr=(unsigned)__cvta_generic_to_shared(buffer+buf*vecs+v);
   asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(addr),"l"(src+chunk*vecs+v));
  }
  if(hooks){FAR_EMU_TILE(2,buf,threadIdx.x==0?min(64,rows-chunk*64):0,1152);}
  asm volatile("cp.async.commit_group;");
 };
 int chunk=blockIdx.x;int k=0;
 if(chunk*64<rows)issue(chunk,0);
 for(;chunk*64<rows;chunk+=gridDim.x,++k){
  int next=chunk+gridDim.x;
  if(next*64<rows){issue(next,(k+1)%2);asm volatile("cp.async.wait_group 1;");}else asm volatile("cp.async.wait_group 0;");
  if(hooks){FAR_EMU_WAIT(2,k%2);}
  __syncthreads();
  if(!threadIdx.x){times[chunk]=flashinfer::far_emu::now_ns();}
  for(int v=threadIdx.x;v<vecs && chunk*64+v/72<rows;v+=blockDim.x)dst[chunk*vecs+v]=buffer[(k%2)*vecs+v];
  __syncthreads();
 }
}
__global__ void reset(){if(!threadIdx.x&&!blockIdx.x)flashinfer::far_emu::link_clock=0;}
int main(){
 check(cudaSetDevice(0));bool first=true;printf("{\"reservation_cases\":[");
 constexpr int trials=500;
 for(U ps:{10ull,20ull})for(U lat:{0ull,20000ull})for(int ctas:{2,24})for(int busy:{0,1})for(int mixed:{0,1}){
  flashinfer::far_emu::Config cfg{ps,lat,0,0,1};check(cudaMemcpyToSymbol(flashinfer::far_emu::config,&cfg,sizeof(cfg)));
  U* dev;size_t size=trials*(ctas+2)*sizeof(U);check(cudaMalloc(&dev,size));void*args[]={&dev,(void*)&trials,&busy,&mixed};check(cudaLaunchCooperativeKernel((void*)reservations,dim3(ctas),dim3(32),args));check(cudaDeviceSynchronize());std::vector<U>h(size/sizeof(U));check(cudaMemcpy(h.data(),dev,size,cudaMemcpyDeviceToHost));check(cudaFree(dev));
  int bad=0,early=0;U total=0;for(int c=0;c<ctas;++c){U b=mixed?(c%3==0?1152:(c%3==1?18432:73728)):73728;total+=(b*ps+999)/1000;}
  for(int n=0;n<trials;++n){auto z=h.data()+n*(ctas+2);U start=z[ctas]+(busy?1000000:0);if(z[ctas+1]!=start+total)++bad;
   std::vector<U>ends(z,z+ctas);std::sort(ends.begin(),ends.end());if(ends.back()!=start+total+lat)++bad;
   for(int c=0;c<ctas;++c)if(z[c]<start+lat)++early;
   if(!mixed)for(int c=0;c<ctas;++c)if(ends[c]!=start+(c+1)*(73728*ps+999)/1000+lat)++bad;
  }
  if(!first)printf(",");first=false;printf("{\"GBps\":%llu,\"latency_ns\":%llu,\"ctas\":%d,\"busy\":%d,\"mixed\":%d,\"trials\":%d,\"bad\":%d,\"early\":%d}",1000/ps,lat,ctas,busy,mixed,trials,bad,early);if(bad||early)return 2;
 }
 printf("],\"arithmetic_cases\":[");first=true;
 for(U ps:{0ull,10ull,20ull}){
  flashinfer::far_emu::Config cfg{ps,20000,0,0,1};check(cudaMemcpyToSymbol(flashinfer::far_emu::config,&cfg,sizeof(cfg)));U*dev;check(cudaMalloc(&dev,9*sizeof(U)));arithmetic<<<1,1>>>(dev);check(cudaDeviceSynchronize());U h[9];check(cudaMemcpy(h,dev,sizeof(h),cudaMemcpyDeviceToHost));check(cudaFree(dev));
  U ts[8]={1000,1000,800,100000,99900,100001,1000000,1000000};U bytes[8]={1,1152,73728,18432,0,73728,73728,1152};U clock=0;int bad=0;
  for(int i=0;i<8;++i){U expected=0;if(bytes[i]){if(!ps)expected=ts[i]+20000;else{clock=std::max(clock,ts[i])+(bytes[i]*ps+999)/1000;expected=clock+20000;}}if(h[i]!=expected)++bad;}
  if(h[8]!=clock)++bad;if(!first)printf(",");first=false;printf("{\"ps_per_byte\":%llu,\"bad\":%d}",ps,bad);if(bad)return 3;
 }
 printf("],\"stage_cases\":[");first=true;
 for(U lat:{0ull,20000ull}){flashinfer::far_emu::Config cfg{10,lat,0,0,1};check(cudaMemcpyToSymbol(flashinfer::far_emu::config,&cfg,sizeof(cfg)));U*dev;check(cudaMalloc(&dev,160*sizeof(U)));reset<<<1,1>>>();stages<<<1,256>>>(dev);check(cudaDeviceSynchronize());U h[160];check(cudaMemcpy(h,dev,sizeof(h),cudaMemcpyDeviceToHost));check(cudaFree(dev));int bad=0;for(int i=0;i<16;++i){U latest=0;for(int w=0;w<8;++w)latest=std::max(latest,h[i*10+w]);if(h[i*10+8]<latest||h[i*10+9])++bad;}if(!first)printf(",");first=false;printf("{\"latency_ns\":%llu,\"bad\":%d}",lat,bad);if(bad)return 4;}
 printf("],\"payload_cases\":[");first=true;
 const int rows=262144+13,chunks=(rows+63)/64;size_t bytes=(size_t)rows*1152;uint4*src,*dst;U*times;check(cudaMalloc(&src,bytes));check(cudaMalloc(&dst,bytes));check(cudaMalloc(&times,chunks*sizeof(U)));check(cudaMemset(src,0x5a,bytes));
 check(cudaFuncSetAttribute(payload,cudaFuncAttributeMaxDynamicSharedMemorySize,147456));
 cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));
 for(U ps:{0ull,10ull,20ull})for(U lat:{0ull,20000ull}){
  flashinfer::far_emu::Config cfg{ps,lat,0,0,1};check(cudaMemcpyToSymbol(flashinfer::far_emu::config,&cfg,sizeof(cfg)));check(cudaMemset(dst,0,bytes));
  std::vector<float>samples;
  for(int rep=0;rep<4;++rep){reset<<<1,1>>>();check(cudaEventRecord(start));payload<<<24,256,147456>>>(src,dst,rows,times,1);check(cudaEventRecord(end));check(cudaEventSynchronize(end));float ms;check(cudaEventElapsedTime(&ms,start,end));if(rep)samples.push_back(ms*1000);}
  std::vector<unsigned char>h(bytes);check(cudaMemcpy(h.data(),dst,bytes,cudaMemcpyDeviceToHost));bool ok=std::all_of(h.begin(),h.end(),[](unsigned char v){return v==0x5a;});if(!ok)return 5;
  if(!first)printf(",");first=false;printf("{\"GBps\":%llu,\"latency_ns\":%llu,\"bytes\":%zu,\"payload_ok\":true,\"samples_us\":[%.3f,%.3f,%.3f]}",ps?1000/ps:0,lat,bytes,samples[0],samples[1],samples[2]);
 }
 printf("]}\n");check(cudaFree(src));check(cudaFree(dst));check(cudaFree(times));return 0;
}
