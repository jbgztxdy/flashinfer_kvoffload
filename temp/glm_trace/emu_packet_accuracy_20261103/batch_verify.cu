#define FLASHINFER_FAR_EMU
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "flashinfer/attention/far_link_emu.cuh"
using U=unsigned long long;
void ck(cudaError_t x){if(x!=cudaSuccess){fprintf(stderr,"CUDA:%s\n",cudaGetErrorString(x));exit(2);}}
__global__ void reset(){if(threadIdx.x==0){flashinfer::far_emu::link_clock=0;flashinfer::far_emu::broker_lock=0;flashinfer::far_emu::broker_count=0;}}
template<int ROWS>
__global__ void copy(const uint4* src,uint4* dst,int count){
 extern __shared__ __align__(128) uint4 payload[];constexpr int V=ROWS*72;FAR_EMU_INIT(2);FAR_EMU_SYNC();__shared__ U rec[2];
 auto issue=[&](int chunk,int stage){for(int v=threadIdx.x;v<V&&chunk*ROWS+v/72<count;v+=256){uint32_t a=(uint32_t)__cvta_generic_to_shared(payload+stage*V+v);asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(a),"l"(src+size_t(chunk)*V+v));}FAR_EMU_TILE(2,stage,threadIdx.x==0?min(ROWS,count-chunk*ROWS):0,1152);if(threadIdx.x==0)rec[stage]=flashinfer::far_emu::broker_requests[blockIdx.x*8].record;asm volatile("cp.async.commit_group;");};
 int chunk=blockIdx.x,step=0;if(chunk*ROWS<count)issue(chunk,0);
 for(;chunk*ROWS<count;chunk+=gridDim.x,++step){int next=chunk+gridDim.x;if(next*ROWS<count){issue(next,(step+1)%2);asm volatile("cp.async.wait_group 1;");}else asm volatile("cp.async.wait_group 0;");FAR_EMU_WAIT(2,step%2);__syncthreads();if(threadIdx.x==0)flashinfer::far_emu::broker_records[rec[step%2]].consume=flashinfer::far_emu::now_ns();for(int v=threadIdx.x;v<V&&chunk*ROWS+v/72<count;v+=256)dst[size_t(chunk)*V+v]=payload[(step%2)*V+v];__syncthreads();}
}
int main(){ck(cudaSetDevice(0));unsigned tracing=1;ck(cudaMemcpyToSymbol(flashinfer::far_emu::broker_trace,&tracing,sizeof(tracing)));int count=215040+13;size_t bytes=size_t(count)*1152;uint4 *src,*dst;ck(cudaMalloc(&src,bytes));ck(cudaMalloc(&dst,bytes));ck(cudaMemset(src,0x5a,bytes));ck(cudaFuncSetAttribute(copy<64>,cudaFuncAttributeMaxDynamicSharedMemorySize,147968));ck(cudaFuncSetAttribute(copy<10>,cudaFuncAttributeMaxDynamicSharedMemorySize,163072));cudaEvent_t st,en;ck(cudaEventCreate(&st));ck(cudaEventCreate(&en));bool first=true;printf("{\"classification\":\"transfer diagnostic only\",\"bytes\":%zu,\"cases\":[",bytes);
 for(int config_mode:{2})for(U lat:{0ull,20000ull})for(int rep=0;rep<2;++rep)for(int i=0;i<2;++i){int mode=rep%2?1-i:i;flashinfer::far_emu::Config c{config_mode==2?10ull:0ull,lat,0,0,config_mode?1:0};ck(cudaMemcpyToSymbol(flashinfer::far_emu::config,&c,sizeof(c)));reset<<<1,1>>>();ck(cudaEventRecord(st));if(mode==0)copy<64><<<24,256,147968>>>(src,dst,count);else copy<10><<<108,256,163072>>>(src,dst,count);ck(cudaEventRecord(en));ck(cudaEventSynchronize(en));float ms;ck(cudaEventElapsedTime(&ms,st,en));U entries;ck(cudaMemcpyFromSymbol(&entries,flashinfer::far_emu::broker_count,sizeof(entries)));std::vector<flashinfer::far_emu::BrokerRecord> records(entries);ck(cudaMemcpyFromSymbol(records.data(),flashinfer::far_emu::broker_records,entries*sizeof(records[0])));U clock=0,charged=0;int bad=0,early=0;for(auto& x:records){U expected=std::max(clock,x.issue)+x.cost;if(x.end!=expected||x.prior!=expected-x.cost)++bad;if(x.consume<x.end+lat)++early;clock=x.end;charged+=x.cost;}U expectedCost=0;for(int row=0;row<count;row+=(mode?10:64))expectedCost+=(U(std::min(mode?10:64,count-row))*1152*10+999)/1000;if(charged!=expectedCost||entries!=(count+(mode?9:63))/(mode?10:64))++bad;if(bad||early){fprintf(stderr,"auditbad=%d early=%d\n",bad,early);return 4;}if(rep){if(!first)printf(",");first=false;printf("{\"config_mode\":%d,\"latency_ns\":%llu,\"rows_per_packet\":%d,\"CTAs\":%d,\"repeat\":%d,\"us\":%.6f,\"request_records\":%llu,\"charged_ns\":%llu,\"bad\":%d,\"early\":%d}",config_mode,lat,mode?10:64,mode?108:24,rep,ms*1000,entries,charged,bad,early);}if(rep==0){std::vector<unsigned char>data(bytes);ck(cudaMemcpy(data.data(),dst,bytes,cudaMemcpyDeviceToHost));if(!std::all_of(data.begin(),data.end(),[](unsigned char x){return x==0x5a;}))return 3;}}
 printf("],\"payload_exact\":true}\n");}
