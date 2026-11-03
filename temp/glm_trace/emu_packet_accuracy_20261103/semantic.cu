#define FLASHINFER_FAR_EMU
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "flashinfer/attention/far_link_emu.cuh"
using U=unsigned long long;namespace F=flashinfer::far_emu;namespace cg=cooperative_groups;
void ck(cudaError_t x){if(x!=cudaSuccess){fprintf(stderr,"CUDA:%s\n",cudaGetErrorString(x));exit(2);}}
__global__ void reset(){if(!threadIdx.x){F::link_clock=0;F::broker_lock=0;F::broker_count=0;}}
__global__ void exercise(int rounds,int active){
 extern __shared__ unsigned char reserved[];if(threadIdx.x==0)reserved[0]=0;
 auto grid=cg::this_grid();const unsigned warp=threadIdx.x/32,lane=threadIdx.x%32;
 for(int i=0;i<rounds;++i){
  grid.sync();
  if(i%8==0&&threadIdx.x==0)F::wait_until(F::now_ns()+100000);__syncthreads();
  if(warp<unsigned(active)){
   if((blockIdx.x+warp+i)%3==0&&lane==0)F::wait_until(F::now_ns()+2000);__syncwarp();
   U bytes=(blockIdx.x+warp+i)%4==0?1ull:(blockIdx.x+warp+i)%4==1?1152ull:(blockIdx.x+warp+i)%4==2?11520ull:73728ull;
   U end=F::reserve_warp(bytes);
   if(lane==0){F::wait_until(end);U id=F::broker_requests[blockIdx.x*8+warp].record;F::broker_records[id].consume=F::now_ns();}
   __syncwarp();
  }
 }
 grid.sync();
}
int main(){ck(cudaSetDevice(0));unsigned trace=1;ck(cudaMemcpyToSymbol(F::broker_trace,&trace,sizeof(trace)));ck(cudaFuncSetAttribute(exercise,cudaFuncAttributeMaxDynamicSharedMemorySize,163072));bool first=true;printf("{\"cases\":[");
 for(U ps:{10ull,20ull})for(U latency:{0ull,20000ull})for(int ctas:{2,24,108})for(int active:{1,8}){
  const int rounds=32;F::Config config{ps,latency,0,0,1};ck(cudaMemcpyToSymbol(F::config,&config,sizeof(config)));reset<<<1,1>>>();void*args[]={(void*)&rounds,&active};ck(cudaLaunchCooperativeKernel((void*)exercise,dim3(ctas),dim3(256),args,163072));ck(cudaDeviceSynchronize());U count;ck(cudaMemcpyFromSymbol(&count,F::broker_count,sizeof(count)));std::vector<F::BrokerRecord>data(count);ck(cudaMemcpyFromSymbol(data.data(),F::broker_records,count*sizeof(data[0])));U clock=0,cost=0,required=0;int bad=0,early=0;
  for(auto& r:data){U expected=std::max(clock,r.issue)+r.cost;if(r.end!=expected||r.prior!=expected-r.cost)++bad;if(r.consume<r.end+latency)++early;clock=r.end;cost+=r.cost;}
  for(int i=0;i<rounds;++i)for(int c=0;c<ctas;++c)for(int w=0;w<active;++w){U bytes=(c+w+i)%4==0?1ull:(c+w+i)%4==1?1152ull:(c+w+i)%4==2?11520ull:73728ull;required+=(bytes*ps+999)/1000;}
  if(cost!=required||count!=U(rounds*ctas*active))++bad;if(!first)printf(",");first=false;printf("{\"GBps\":%llu,\"latency_ns\":%llu,\"CTAs\":%d,\"active_warps\":%d,\"requests\":%llu,\"bad\":%d,\"early\":%d}",1000/ps,latency,ctas,active,count,bad,early);if(bad||early)return 3;
 }
 printf("],\"complete\":true}\n");
}
