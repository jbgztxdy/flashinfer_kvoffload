#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdint>
#define CK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 1;}}while(0)
// All workers keep draining even after their synthetic demand sequence ends.
__global__ void probe(const uint4* src,uint4* stage,uint4* cache,unsigned* claim,unsigned* ready,unsigned* copies,unsigned* cursor,unsigned* done,unsigned* errors,unsigned pages,unsigned packets){
 __shared__ uint4 buffer[20*72];__shared__ unsigned job;
 unsigned t=threadIdx.x,turn=0;
 while(true){
  if(t==0){job=packets;
   unsigned demand=(blockIdx.x*97+turn*31)%packets;
   if(turn<64&&atomicCAS(claim+demand,0,1)==0)job=demand;
   if(job==packets){while(true){unsigned k=atomicAdd(cursor,1);if(k>=packets)break;if(atomicCAS(claim+k,0,1)==0){job=k;break;}}}
  }
  __syncthreads();unsigned k=job;
  if(k==packets){if(t==0)job=atomicAdd(done,0)==packets?packets+1:packets;__syncthreads();if(job==packets+1)break;__nanosleep(64);++turn;continue;}
  unsigned n=min(20u,pages-k*20),vectors=n*72;
  for(unsigned v=t;v<vectors;v+=blockDim.x)buffer[v]=src[k*20*72+v];
  __syncthreads();
  for(unsigned v=t;v<vectors;v+=blockDim.x){uint4 x=buffer[v];stage[k*20*72+v]=x;cache[k*20*72+v]=x;}
  // Every writer fences, then leader release publishes the complete payload.
  __threadfence();__syncthreads();
  if(t==0){atomicAdd(copies+k,1);asm volatile("st.release.gpu.global.u32 [%0],1;"::"l"(ready+k):"memory");atomicAdd(done,1);}
  __syncthreads();++turn;
 }
 // Concurrent acquire consumers verify completed packets before kernel exit.
 for(unsigned k=blockIdx.x;k<packets;k+=gridDim.x){
  unsigned flag;do{asm volatile("ld.acquire.gpu.global.u32 %0,[%1];":"=r"(flag):"l"(ready+k):"memory");}while(!flag);
  unsigned vectors=min(20u,pages-k*20)*72;
  for(unsigned v=t;v<vectors;v+=blockDim.x){unsigned at=k*20*72+v;uint4 a=src[at],b=stage[at],c=cache[at];if(a.x!=b.x||a.y!=b.y||a.z!=b.z||a.w!=b.w||a.x!=c.x||a.y!=c.y||a.z!=c.z||a.w!=c.w)atomicAdd(errors,1);}
 }
}
int main(){unsigned pages=215053,packets=(pages+19)/20;size_t bytes=size_t(pages)*1152;
 std::vector<uint4> input(size_t(pages)*72);for(size_t i=0;i<input.size();++i)input[i]=make_uint4(i,i^0x12345678u,i*17,i+93);
 uint4 *src,*stage,*cache;unsigned *claim,*ready,*copies,*cursor,*done,*errors;
 CK(cudaMalloc(&src,bytes));CK(cudaMalloc(&stage,bytes));CK(cudaMalloc(&cache,bytes));CK(cudaMemcpy(src,input.data(),bytes,cudaMemcpyHostToDevice));
 CK(cudaMalloc(&claim,packets*4));CK(cudaMalloc(&ready,packets*4));CK(cudaMalloc(&copies,packets*4));CK(cudaMalloc(&cursor,4));CK(cudaMalloc(&done,4));CK(cudaMalloc(&errors,4));
 for(unsigned blocks: {2u,24u,108u})for(unsigned epoch=0;epoch<3;++epoch){
 CK(cudaMemset(claim,0,packets*4));CK(cudaMemset(ready,0,packets*4));CK(cudaMemset(copies,0,packets*4));CK(cudaMemset(cursor,0,4));CK(cudaMemset(done,0,4));CK(cudaMemset(errors,0,4));CK(cudaMemset(stage,0,bytes));CK(cudaMemset(cache,0,bytes));
 probe<<<blocks,256>>>(src,stage,cache,claim,ready,copies,cursor,done,errors,pages,packets);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
 std::vector<unsigned> c(packets);unsigned err,finished;CK(cudaMemcpy(c.data(),copies,packets*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&err,errors,4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&finished,done,4,cudaMemcpyDeviceToHost));for(auto x:c)if(x!=1){fprintf(stderr,"duplicate or omitted packet\n");return 2;}if(err||finished!=packets)return 3;
 printf("blocks=%u epoch=%u packets=%u pages=%u far_bytes=%zu errors=%u copies_exact1=true\n",blocks,epoch,packets,pages,bytes,err);
 }
 return 0;
}
