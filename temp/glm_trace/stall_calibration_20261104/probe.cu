#define FLASHINFER_FAR_EMU
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include "stall.cuh"
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
__global__ void copy(const uint4* src,uint4* dst,int count,int farrows){
 extern __shared__ __align__(128) uint4 data[];int tid=(threadIdx.z*4+threadIdx.y)*32+threadIdx.x;FAR_EMU_INIT(2);__syncthreads();
 auto issue=[&](int chunk,int stage){for(int v=tid;v<2304;v+=256){unsigned sm=(unsigned)__cvta_generic_to_shared(data+stage*2304+v);asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(sm),"l"(src+size_t(chunk)*2304+v));}int warp=tid/32;int rows=max(0,min(4,farrows-warp*4));FAR_EMU_TILE(2,stage,threadIdx.x==0?rows:0,1152);asm volatile("cp.async.commit_group;");};
 int chunk=blockIdx.y,step=0;if(chunk<count)issue(chunk,0);
 for(;chunk<count;chunk+=gridDim.y,step++){int next=chunk+gridDim.y;if(next<count){issue(next,(step+1)%2);asm volatile("cp.async.wait_group 1;");}else asm volatile("cp.async.wait_group 0;");FAR_EMU_WAIT(2,step%2);__syncthreads();for(int v=tid;v<2304;v+=256)dst[size_t(chunk)*2304+v]=data[(step%2)*2304+v];__syncthreads();}
}
int main(){ck(cudaSetDevice(0));int chunks=108*64;size_t bytes=size_t(chunks)*32*1152;uint4 *s,*d;ck(cudaMalloc(&s,bytes));ck(cudaMalloc(&d,bytes));ck(cudaMemset(s,0x5a,bytes));ck(cudaFuncSetAttribute(copy,cudaFuncAttributeMaxDynamicSharedMemorySize,147968));cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));printf("{\"cases\":[");bool first=true;
 for(int far:{8,32})for(int ps:{0,200,400,600,800,1000,1400,2000})for(int rep=0;rep<3;rep++){
 flashinfer::far_emu::Config c{(unsigned long long)ps,0,0,0,1};ck(cudaMemcpyToSymbol(flashinfer::far_emu::config,&c,sizeof(c)));ck(cudaEventRecord(a));copy<<<dim3(1,108),dim3(32,4,2),147968>>>(s,d,chunks,far);ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));if(rep){if(!first)printf(",");first=false;printf("{\"far_rows\":%d,\"ps_per_byte\":%d,\"us\":%.3f,\"GBps\":%.3f}",far,ps,ms*1000,double(chunks)*far*1152/ms/1e6);}}
 std::vector<unsigned char>h(bytes);ck(cudaMemcpy(h.data(),d,bytes,cudaMemcpyDeviceToHost));if(!std::all_of(h.begin(),h.end(),[](unsigned char x){return x==0x5a;}))return 3;printf("],\"full_payload_exact\":true}\n");}
