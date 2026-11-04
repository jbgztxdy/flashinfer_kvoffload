#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <cstdint>
using U=unsigned long long;
constexpr int MAXP=864,SLOTS=MAXP*2;
struct Box{U issue,cost,end,seq;unsigned req,ack;};
struct Record{U issue,cost,end,consume;unsigned p,seq;};
__device__ Box boxes[SLOTS];__device__ unsigned done[MAXP];
__device__ Record records[MAXP*128];__device__ U rec_count,epochs;
__device__ U now(){U t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
__device__ unsigned acquire(unsigned* p){unsigned v;asm volatile("ld.acquire.gpu.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;}
__device__ void release(unsigned* p,unsigned v){asm volatile("st.release.gpu.global.u32 [%0],%1;"::"l"(p),"r"(v):"memory");}
__global__ void reset(){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<SLOTS)boxes[i]={};if(i<MAXP)done[i]=0;if(i==0){rec_count=0;epochs=0;}}
// Dedicated broker CTA only in this protocol diagnostic; NOT a production architectural change.
__global__ void run(const uint4* src,uint4* dst,int workers,int rounds,int window,int latency,int pattern,int trace){
 int lane=threadIdx.x%32,warp=threadIdx.x/32;int producers=workers*8;
 if(blockIdx.x==0){if(warp)return;U clock=0,count=0,tick=now();
  while(true){
   while(now()<tick)__nanosleep(64);tick=now()+window;
   for(int base=0;base<producers*2;base+=32){int id=base+lane;bool valid=id<producers*2;Box* b=boxes+(valid?id:0);unsigned req=valid?acquire(&b->req):0;bool active=valid&&req!=b->ack;
    U cost=active?b->cost:0,issue=active?b->issue:0,prefix=cost;
    for(int d=1;d<32;d*=2){U x=__shfl_up_sync(0xffffffff,prefix,d);if(lane>=d)prefix+=x;}
    long long anchor=active?(long long)issue-(long long)(prefix-cost):(-0x7fffffffffffffffll);anchor=max(anchor,(long long)clock);
    for(int d=1;d<32;d*=2){long long x=__shfl_up_sync(0xffffffff,anchor,d);if(lane>=d)anchor=max(anchor,x);}
    U end=anchor+prefix;unsigned mask=__ballot_sync(0xffffffff,active);int last=31-__clz(mask|1);U final=__shfl_sync(0xffffffff,end,last);
    if(mask)clock=final;
    if(active){b->end=end+latency;release(&b->ack,req);}
    count+=__popc(mask);
   }
   unsigned all=1;for(int p=lane;p<producers;p+=32)all&=acquire(done+p);all=__all_sync(0xffffffff,all);
   if(lane==0)epochs++;if(all)break;
  }
  if(lane==0)rec_count=count;return;
 }
 int p=(blockIdx.x-1)*8+warp;extern __shared__ __align__(128) uint4 raw[];auto payload=reinterpret_cast<uint4 (*)[2][288]>(raw);
 auto issue=[&](int seq){int stage=seq%2;size_t at=(size_t(p)*rounds+seq)*288;
  for(int v=lane;v<288;v+=32){unsigned sm=(unsigned)__cvta_generic_to_shared(&payload[warp][stage][v]);asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(sm),"l"(src+at+v));}
  asm volatile("cp.async.commit_group;");__syncwarp();
  if(lane==0){Box* b=boxes+p*2+stage;b->issue=now();b->cost=47; // ceil(4608B/100GB)=47ns; retain rounding
   b->seq=seq;release(&b->req,seq/2+1);}
 };
 if(pattern==1&&p%3==0){U until=now()+100000;while(now()<until)__nanosleep(64);}
 issue(0);issue(1);
 for(int seq=0;seq<rounds;seq++){
  Box* b=boxes+p*2+seq%2;unsigned generation=seq/2+1;U deadline=0;
  if(lane==0){while(acquire(&b->ack)!=generation)__nanosleep(64);deadline=b->end;while(now()<deadline)__nanosleep(64);}
  __syncwarp();if(seq+1<rounds)asm volatile("cp.async.wait_group 1;");else asm volatile("cp.async.wait_group 0;");
  for(int v=lane;v<288;v+=32)dst[(size_t(p)*rounds+seq)*288+v]=payload[warp][seq%2][v];
  __syncwarp();if(trace&&lane==0)records[p*128+seq]={b->issue,b->cost,b->end,now(),(unsigned)p,(unsigned)seq};
  if(pattern==2&&seq==rounds/2){U until=now()+100000;while(now()<until)__nanosleep(64);}
  if(seq+2<rounds)issue(seq+2);
 }
 if(lane==0)release(done+p,1);
}
void ck(cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(2);}}
int main(){ck(cudaSetDevice(0));ck(cudaFuncSetAttribute(run,cudaFuncAttributeMaxDynamicSharedMemorySize,73728));int rounds=128;size_t bytes=size_t(MAXP)*rounds*4608;uint4 *s,*d;ck(cudaMalloc(&s,bytes));ck(cudaMalloc(&d,bytes));ck(cudaMemset(s,0x5a,bytes));cudaEvent_t a,b;ck(cudaEventCreate(&a));ck(cudaEventCreate(&b));printf("{\"cases\":[");bool first=true;
 for(int workers:{24,108})for(int window:{500})for(int lat:{0,20000})for(int pat:{0}){
  reset<<<8,256>>>();ck(cudaMemset(d,0,bytes));ck(cudaEventRecord(a));run<<<workers+1,256,73728>>>(s,d,workers,rounds,window,lat,pat,0);ck(cudaEventRecord(b));ck(cudaEventSynchronize(b));float ms;ck(cudaEventElapsedTime(&ms,a,b));
  int n=workers*8*rounds;std::vector<unsigned char>h(size_t(n)*4608);ck(cudaMemcpy(h.data(),d,h.size(),cudaMemcpyDeviceToHost));bool exact=std::all_of(h.begin(),h.end(),[](unsigned char x){return x==0x5a;});
  U count,ep;ck(cudaMemcpyFromSymbol(&count,rec_count,sizeof(count)));ck(cudaMemcpyFromSymbol(&ep,epochs,sizeof(ep)));int bad=0;
  if(!first)printf(",");first=false;printf("{\"workers\":%d,\"window_ns\":%d,\"latency_ns\":%d,\"pattern\":%d,\"us\":%.3f,\"GBps\":%.3f,\"requests\":%llu,\"epochs\":%llu,\"bad\":%d,\"payload_exact\":%s}",workers,window,lat,pat,ms*1000,h.size()/ms/1e6,count,ep,bad,exact?"true":"false");fflush(stdout);if(bad||!exact||count!=n)return 3;
 }
 printf("]}\n");}
