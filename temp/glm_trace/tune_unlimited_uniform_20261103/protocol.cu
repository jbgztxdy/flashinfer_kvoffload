// Cross-CTA message-passing test for the actual barrier->release->acquire relay.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cooperative_groups.h>
void ck(cudaError_t s){if(s!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(s));exit(2);}}
__device__ __forceinline__ unsigned acq(const unsigned* p){unsigned x;asm volatile("ld.acquire.gpu.global.u32 %0,[%1];":"=r"(x):"l"(p):"memory");return x;}
__device__ __forceinline__ void rel(unsigned* p,unsigned x){asm volatile("st.release.gpu.global.u32 [%0],%1;"::"l"(p),"r"(x):"memory");}
__global__ void relay(uint4* data,unsigned* flags,unsigned* errors,unsigned trials){
 extern __shared__ unsigned char reserved[];if(threadIdx.x==0)reserved[0]=0;
 unsigned pair=blockIdx.x%54,t=threadIdx.x;constexpr unsigned N=20*72;
 if(blockIdx.x<54){for(unsigned seq=1;seq<=trials;++seq){if(t==0)while(acq(flags+pair*2+1)!=seq-1)__nanosleep(64);__syncthreads();
  for(unsigned v=t;v<N;v+=256){unsigned x=seq*9176+v+pair*37;data[pair*N+v]=make_uint4(x,x^0xf00df00d,x+0x12345678,x*7);}
  // No perwriter threadfence: CTA sync then one GPU release publishes all lanes.
  __syncthreads();if(t==0)rel(flags+pair*2,seq);
 }}else{for(unsigned seq=1;seq<=trials;++seq){if(t==0)while(acq(flags+pair*2)!=seq)__nanosleep(64);__syncthreads();
  for(unsigned v=t;v<N;v+=256){unsigned x=seq*9176+v+pair*37;uint4 got=data[pair*N+v];if(got.x!=x||got.y!=(x^0xf00df00d)||got.z!=x+0x12345678||got.w!=x*7)atomicAdd(errors,1u);}
  __syncthreads();if(t==0)rel(flags+pair*2+1,seq);
 }}
}
int main(){ck(cudaSetDevice(0));unsigned trials=500,*flags,*errors;uint4*data;ck(cudaMalloc(&data,54*20*1152));ck(cudaMalloc(&flags,108*sizeof(unsigned)));ck(cudaMalloc(&errors,sizeof(unsigned)));ck(cudaMemset(flags,0,108*sizeof(unsigned)));ck(cudaMemset(errors,0,sizeof(unsigned)));ck(cudaFuncSetAttribute(relay,cudaFuncAttributeMaxDynamicSharedMemorySize,163072));void*args[]={&data,&flags,&errors,&trials};ck(cudaLaunchCooperativeKernel((void*)relay,dim3(108),dim3(256),args,163072));ck(cudaDeviceSynchronize());unsigned bad;ck(cudaMemcpy(&bad,errors,sizeof(unsigned),cudaMemcpyDeviceToHost));printf("{\"pairs\":54,\"trials_per_pair\":%u,\"payload_bytes_per_trial\":23040,\"checked_vectors\":%llu,\"errors\":%u,\"protocol\":\"allwriterstores -> bar.sync -> leaderst.release.gpu -> consumerld.acquire.gpu -> bar.sync -> read\"}\n",trials,54ull*trials*20*72,bad);return bad?3:0;}
