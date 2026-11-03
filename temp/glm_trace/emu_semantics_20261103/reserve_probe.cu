// Diagnostic only: invoke unchanged production reserve(). No payload/performance claim.
#define FLASHINFER_FAR_EMU
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "flashinfer/attention/far_link_emu.cuh"
namespace cg=cooperative_groups;
using U=unsigned long long;
__device__ U common_issue;
__global__ void probe(U* records,int repeats,int busy) {
 auto grid=cg::this_grid();
 for(int i=0;i<repeats;++i) {
  if(blockIdx.x==0 && threadIdx.x==0) {
   common_issue=flashinfer::far_emu::now_ns();
   flashinfer::far_emu::link_clock=busy?common_issue+1000000:0;
  }
  grid.sync();
  if(threadIdx.x==0)records[i*(gridDim.x+2)+blockIdx.x]=flashinfer::far_emu::reserve(73728,common_issue);
  grid.sync();
  if(blockIdx.x==0 && threadIdx.x==0) {
   records[i*(gridDim.x+2)+gridDim.x]=common_issue;
   records[i*(gridDim.x+2)+gridDim.x+1]=flashinfer::far_emu::link_clock;
  }
  grid.sync();
 }
}
void check(cudaError_t x) {if(x!=cudaSuccess) {std::fprintf(stderr,"CUDA %s\n",cudaGetErrorString(x));std::exit(1);}}
int main() {
 check(cudaSetDevice(0));
 flashinfer::far_emu::Config cfg{10,0,0,0,1};check(cudaMemcpyToSymbol(flashinfer::far_emu::config,&cfg,sizeof(cfg)));
 constexpr int repeats=2000;
 std::printf("{\"scope\":\"Unchanged reserve helper, synchronized synthetic calls; no payload/timing measurement\",\"cases\":[");
 bool first=true;
 for(int ctas:{2,24})for(int busy:{0,1}) {
  U* device;size_t bytes=(size_t)repeats*(ctas+2)*sizeof(U);check(cudaMalloc(&device,bytes));
  void* args[]={&device,(void*)&repeats,&busy};
  check(cudaLaunchCooperativeKernel((void*)probe,dim3(ctas),dim3(32),args,0,nullptr));check(cudaDeviceSynchronize());
  std::vector<U> h(repeats*(ctas+2));check(cudaMemcpy(h.data(),device,bytes,cudaMemcpyDeviceToHost));check(cudaFree(device));
  int bad=0,duplicates=0;U max_under=0,min_span=~0ull,max_span=0;
  for(int i=0;i<repeats;++i) {
   U* z=h.data()+i*(ctas+2);U issue=z[ctas],clock=z[ctas+1];U expected=issue+(busy?1000000:0)+(U)ctas*737;
   if(clock!=expected)++bad;
   if(clock<expected && expected-clock>max_under)max_under=expected-clock;
   U span=clock-issue-(busy?1000000:0);if(span<min_span)min_span=span;if(span>max_span)max_span=span;
   bool dup=false;for(int a=0;a<ctas;++a)for(int b=0;b<a;++b)if(z[a]==z[b])dup=true;
   if(dup)++duplicates;
  }
  if(!first)std::printf(",");first=false;
  std::printf("{\"ctas\":%d,\"initially_busy\":%s,\"trials\":%d,\"clock_mismatch_trials\":%d,\"duplicate_deadline_trials\":%d,\"max_undercount_ns\":%llu,\"min_service_span_ns\":%llu,\"max_service_span_ns\":%llu,\"expected_service_span_ns\":%llu}",ctas,busy?"true":"false",repeats,bad,duplicates,max_under,min_span,max_span,(U)ctas*737);
 }
 std::printf("]}\n");return 0;
}
