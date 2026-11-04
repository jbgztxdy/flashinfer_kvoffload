#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#define CK(x) do {auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
__device__ __forceinline__ unsigned long long ns(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__global__ void init(uint4* p,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)p[i]=make_uint4(i,i^0x12345678u,i*17u,~i);}
__global__ void plain(const uint4* s,uint4* d,size_t n){for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)d[i]=s[i];}

// 1152B KV pages, unique random source permutation. Each CTA owns 64-page
// packets, double shared buffers, just like the production fetch shape.
__global__ void sparse(const uint4* s,uint4* d,const uint32_t* pages,uint32_t count){
 extern __shared__ __align__(128) uint4 buffer[];
 constexpr unsigned V=64*72;
 unsigned t=threadIdx.x;
 auto issue=[&](unsigned packet,unsigned b){
  for(unsigned v=t;v<V;v+=blockDim.x){unsigned row=packet*64+v/72;
   if(row<count){const uint4* src=s+size_t(pages[row])*72+v%72;
    unsigned dst=(unsigned)__cvta_generic_to_shared(buffer+b*V+v);
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(dst),"l"(src):"memory");
   }
  }
  asm volatile("cp.async.commit_group;":::"memory");
 };
 unsigned packet=blockIdx.x,k=0;
 if(packet*64<count)issue(packet,0);
 for(;packet*64<count;packet+=gridDim.x,++k){
  unsigned next=packet+gridDim.x;
  if(next*64<count){issue(next,(k+1)%2);asm volatile("cp.async.wait_group 1;":::"memory");}
  else asm volatile("cp.async.wait_group 0;":::"memory");
  __syncthreads();
  for(unsigned v=t;v<V;v+=blockDim.x)if(packet*64+v/72<count)d[size_t(packet)*V+v]=buffer[(k%2)*V+v];
  __syncthreads();
 }
}
__global__ void verify(const uint4* d,size_t n,const uint32_t* pages,unsigned* bad){
 for(size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){
  size_t j=pages?size_t(pages[i/72])*72+i%72:i;
  uint4 a=d[i],b=make_uint4(j,j^0x12345678u,j*17u,~j);
  if(a.x!=b.x||a.y!=b.y||a.z!=b.z||a.w!=b.w)atomicAdd(bad,1u);
 }
}
// One lane, randomized single cycle, one dependent load per hop. .cg avoids
// L1; large working set exceeds A100 L2. This is load-use latency, not copy
// launch latency and not latency of a fully parallel 1152B page transfer.
__global__ void chase(const uint64_t* links,unsigned hops,uint64_t* result){
 uint64_t at=0;
 for(unsigned i=0;i<1024;++i)asm volatile("ld.global.cg.u64 %0,[%1];":"=l"(at):"l"(links+at*16):"memory");
 auto begin=ns();
 for(unsigned i=0;i<hops;++i)asm volatile("ld.global.cg.u64 %0,[%1];":"=l"(at):"l"(links+at*16):"memory");
 auto end=ns();result[0]=end-begin;result[1]=at;
}

template<class F> void measure(const char* name,size_t bytes,F fn){
 for(int i=0;i<3;++i)fn();CK(cudaDeviceSynchronize());
 cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));
 printf("{\"kind\":\"bandwidth\",\"name\":\"%s\",\"payload_bytes\":%zu,\"us\":[",name,bytes);
 for(int sample=0;sample<7;++sample){CK(cudaEventRecord(a));for(int i=0;i<10;++i)fn();CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));printf("%s%.6f",sample?",":"",ms*1000/10);}
 printf("]}\n");CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));
}
int main(){
 CK(cudaSetDevice(0));cudaDeviceProp prop;CK(cudaGetDeviceProperties(&prop,0));
 printf("{\"kind\":\"device\",\"name\":\"%s\",\"sms\":%d,\"l2_bytes\":%d}\n",prop.name,prop.multiProcessorCount,prop.l2CacheSize);
 constexpr uint32_t rows=262144;constexpr size_t vectors=size_t(rows)*72,bytes=vectors*16;
 uint4 *src,*dst;uint32_t* ids;unsigned* bad;
 CK(cudaMalloc(&src,bytes));CK(cudaMalloc(&dst,bytes));CK(cudaMalloc(&ids,rows*4));CK(cudaMalloc(&bad,4));
 std::vector<uint32_t> order(rows);std::iota(order.begin(),order.end(),0);std::mt19937 rng(20261104);std::shuffle(order.begin(),order.end(),rng);
 CK(cudaMemcpy(ids,order.data(),rows*4,cudaMemcpyHostToDevice));init<<<prop.multiProcessorCount,256>>>(src,vectors);CK(cudaGetLastError());
 auto check=[&](const uint32_t* map){CK(cudaMemset(bad,0,4));verify<<<prop.multiProcessorCount,256>>>(dst,vectors,map,bad);unsigned h;CK(cudaMemcpy(&h,bad,4,cudaMemcpyDeviceToHost));if(h){fprintf(stderr,"payload mismatch %u\n",h);exit(2);}};
 measure("cudaMemcpyAsync_D2D",bytes,[&](){CK(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice));});check(nullptr);
 measure("SM_contiguous_copy",bytes,[&](){plain<<<prop.multiProcessorCount,256>>>(src,dst,vectors);});check(nullptr);
 CK(cudaFuncSetAttribute(sparse,cudaFuncAttributeMaxDynamicSharedMemorySize,147456));
 for(unsigned ctas:{24u,(unsigned)prop.multiProcessorCount}){
  char name[96];snprintf(name,sizeof(name),"SM_sparse_unique_1152B_64pages_2buffers_%uCTA",ctas);
  measure(name,bytes,[&](){sparse<<<ctas,256,147456>>>(src,dst,ids,rows);});check(ids);
 }
 CK(cudaFree(src));CK(cudaFree(dst));CK(cudaFree(ids));CK(cudaFree(bad));
 uint64_t* result;CK(cudaMalloc(&result,16));
 for(size_t working:{size_t(1)<<20,size_t(256)<<20}){
  size_t nodes=working/128;std::vector<uint32_t> cycle(nodes);std::iota(cycle.begin(),cycle.end(),0);std::shuffle(cycle.begin(),cycle.end(),rng);
  std::vector<uint64_t> host(working/8,0);for(size_t i=0;i<nodes;++i)host[size_t(cycle[i])*16]=cycle[(i+1)%nodes];
  uint64_t* links;CK(cudaMalloc(&links,working));CK(cudaMemcpy(links,host.data(),working,cudaMemcpyHostToDevice));
  // Traverse the entire large ring every sample: repeating a short fixed
  // prefix would fit in L2 and incorrectly report a cache-hit latency.
  const unsigned hops=std::max(size_t(65536),nodes);
  uint64_t expected=0;for(size_t i=0;i<size_t(hops)+1024;++i)expected=host[expected*16];
  printf("{\"kind\":\"dependent_load_latency\",\"working_set_bytes\":%zu,\"hops\":%u,\"ns_per_hop\":[",working,hops);
  for(int sample=0;sample<7;++sample){chase<<<1,1>>>(links,hops,result);uint64_t got[2];CK(cudaMemcpy(got,result,16,cudaMemcpyDeviceToHost));if(got[1]!=expected)return 3;printf("%s%.6f",sample?",":"",double(got[0])/hops);}
  printf("]}\n");CK(cudaFree(links));
 }
 CK(cudaFree(result));CK(cudaGetLastError());printf("{\"kind\":\"complete\",\"payload_checks\":true}\n");
}
