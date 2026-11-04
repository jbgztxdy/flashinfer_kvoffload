#include <initializer_list>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
using U=unsigned long long;
__device__ U now(){U x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x));return x;}
__global__ void probe(const uint4* src,uint4* dst,int rounds,int mode,int assumed,int delay){
 const int tid=threadIdx.x;U deadline=now();
 for(int r=0;r<rounds;r++){
  U issue=now();uint4 x=src[(size_t(blockIdx.x)*rounds+r)*288+tid];
  // 256 threads each read 16B =4096B/CTA. Static share is intentionally diagnostic.
  if(mode==1){if(tid==0)__nanosleep(delay);__syncthreads();}
  if(mode==2){if(tid==0){U cost=(4096ull*assumed+99)/100;deadline=(deadline>issue?deadline:issue)+cost;while(now()<deadline)__nanosleep(64);}__syncthreads();}
  dst[(size_t(blockIdx.x)*rounds+r)*288+tid]=x;
 }
}
int main(){cudaSetDevice(0);uint4 *s,*d;int rounds=512;size_t n=size_t(108)*rounds*288*16;cudaMalloc(&s,n);cudaMalloc(&d,n);cudaMemset(s,0x5a,n);cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);printf("{\"cases\":[");bool first=true;
 for(int mode=0;mode<3;mode++)for(int ct:{2,24,108})for(int rep=0;rep<2;rep++){
 cudaEventRecord(a);probe<<<ct,256>>>(s,d,rounds,mode,108,4424);cudaEventRecord(b);cudaEventSynchronize(b);float ms;cudaEventElapsedTime(&ms,a,b);if(rep){if(!first)printf(",");first=false;printf("{\"mode\":%d,\"ctas\":%d,\"us\":%.3f,\"GBps\":%.3f}",mode,ct,ms*1000,4096.*ct*rounds/ms/1e6);}}
 cudaError_t e=cudaGetLastError();printf("],\"cuda_error\":\"%s\"}\n",cudaGetErrorString(e));return e!=cudaSuccess;
}
