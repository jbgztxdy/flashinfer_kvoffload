// Fetch-only feasibility: 4 loader warps +4 useful-QK warps. Not full MLA.
#define FLASHINFER_FAR_EMU
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include "flashinfer/mma.cuh"
#include "flashinfer/attention/far_link_emu.cuh"
using B=__nv_bfloat16;using U=unsigned long long;
void ck(cudaError_t s){if(s!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(s));exit(2);}}
constexpr int V=64*1152/16,SMEM=147968+12288;
__device__ __forceinline__ void loader_sync(){asm volatile("bar.sync 2,128;":::"memory");}
__device__ __forceinline__ void math_sync(){asm volatile("bar.sync 3,128;":::"memory");}
__device__ __forceinline__ void copy16(void* dst,const void* src){unsigned a=(unsigned)__cvta_generic_to_shared(dst);asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(a),"l"(src));}
// Exactly the original v2's pre-wait producer visibility,latest-stage deadline,
// waiter and post-wait visibility. Core reserve/tile/wait_until/header unchanged.
// Only128 actual loaders participate; helper warps issue NO far requests.
__device__ __forceinline__ U loader_wait(int stage){
 loader_sync();U due=0;
 if(threadIdx.x==0){auto c=flashinfer::far_emu::cta_slot();due=*(volatile U*)&flashinfer::far_emu::pending_ready[c][stage];flashinfer::far_emu::wait_until(due);flashinfer::far_emu::pending_ready[c][stage]=0;}
 loader_sync();return due;
}
__device__ __forceinline__ void qk_shared(const B* q,const B* k,float* out,int job,B* work){
 int tid=threadIdx.x-128,lane=tid%32,warp=tid/32;B* qs=work;B* ks=work+64*64;
 q+=size_t(job)*64*576;k+=size_t(job)*32*576;float s[2][8];
 #pragma unroll 1
 for(int block=0;block<9;++block){
  int feat=block==0?512:(block-1)*64;
  for(int v=tid;v<64*8;v+=128)copy16(qs+(v/8)*64+((v%8)^((v/8)%8))*8,q+(v/8)*576+feat+(v%8)*8);
  for(int v=tid;v<32*8;v+=128)copy16(ks+(v/8)*64+((v%8)^((v/8)%8))*8,k+(v/8)*576+feat+(v%8)*8);
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;");math_sync();
  #pragma unroll
  for(int dim=0;dim<4;++dim){
   uint32_t a[4],b[4];
   flashinfer::mma::ldmatrix_m8n8x4(a,qs+(warp*16+lane%16)*64+((dim*16+8*(lane/16))^((lane%8)*8)));
   #pragma unroll
   for(int group=0;group<2;++group){
    flashinfer::mma::ldmatrix_m8n8x4(b,ks+(group*16+8*(lane/16)+lane%8)*64+((dim*16+8*((lane%16)/8))^((lane%8)*8)));
    if(block==0 && dim==0)flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<B,flashinfer::mma::MMAMode::kInit>(s[group],a,b);
    else flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<B>(s[group],a,b);
   }
  }
  math_sync();
 }
 int head=warp*16+lane/4,col=(lane%4)*2;
 #pragma unroll
 for(int group=0;group<2;++group)for(int i=0;i<8;++i){int h=head+((i%4)/2)*8,kv=group*16+col+(i%2)+(i/4)*8;out[(size_t(job)*64+h)*32+kv]=s[group][i];}
 math_sync();
}
__global__ void reset(){if(!threadIdx.x)flashinfer::far_emu::link_clock=0;}
template<int MODE> //0=8warp original fetch,1=4warp fetch only,2=4+4 overlap
__global__ void pipeline(const uint4* src,uint4* dst,const B* q,const B* k,float* out,int rows,int jobs,U* rec){
 extern __shared__ __align__(128) uint4 sm[];
 __shared__ volatile int finished;
 if(threadIdx.x==0)finished=0;
 FAR_EMU_INIT(2);__syncthreads();
 if constexpr(MODE==0){
  auto issue=[&](int chunk,int stage){for(int v=threadIdx.x;v<V && chunk*64+v/72<rows;v+=256)copy16(sm+stage*V+v,src+size_t(chunk)*V+v);FAR_EMU_TILE(2,stage,threadIdx.x==0?min(64,rows-chunk*64):0,1152);asm volatile("cp.async.commit_group;");};
  int chunk=blockIdx.x,step=0;if(threadIdx.x==0)rec[blockIdx.x*8]=flashinfer::far_emu::now_ns();if(chunk*64<rows)issue(chunk,0);
  for(;chunk*64<rows;chunk+=gridDim.x,++step){int next=chunk+gridDim.x;if(next*64<rows){issue(next,(step+1)%2);asm volatile("cp.async.wait_group 1;");}else asm volatile("cp.async.wait_group 0;");
   FAR_EMU_WAIT(2,step%2);__syncthreads();
   for(int v=threadIdx.x;v<V && chunk*64+v/72<rows;v+=256)dst[size_t(chunk)*V+v]=sm[(step%2)*V+v];__syncthreads();}
  if(threadIdx.x==0)rec[blockIdx.x*8+1]=flashinfer::far_emu::now_ns();
 }else{
  if(threadIdx.x<128){
   auto issue=[&](int chunk,int stage){for(int v=threadIdx.x;v<V && chunk*64+v/72<rows;v+=128)copy16(sm+stage*V+v,src+size_t(chunk)*V+v);FAR_EMU_TILE(2,stage,threadIdx.x==0?min(64,rows-chunk*64):0,1152);asm volatile("cp.async.commit_group;");};
   int chunk=blockIdx.x,step=0;if(threadIdx.x==0)rec[blockIdx.x*8]=flashinfer::far_emu::now_ns();if(chunk*64<rows)issue(chunk,0);
   for(;chunk*64<rows;chunk+=gridDim.x,++step){int next=chunk+gridDim.x;if(next*64<rows){issue(next,(step+1)%2);asm volatile("cp.async.wait_group 1;");}else asm volatile("cp.async.wait_group 0;");
    U due=loader_wait(step%2);
    if(threadIdx.x==0 && flashinfer::far_emu::now_ns()<due)rec[blockIdx.x*8+4]++;
    for(int v=threadIdx.x;v<V && chunk*64+v/72<rows;v+=128)dst[size_t(chunk)*V+v]=sm[(step%2)*V+v];loader_sync();}
   if(threadIdx.x==0){rec[blockIdx.x*8+1]=flashinfer::far_emu::now_ns();__threadfence_block();finished=1;}
  }else if constexpr(MODE==2){
   int done=0;
   for(int job=blockIdx.x;job<jobs;job+=gridDim.x){
    // A single leader publishes the stop decision; reuse a record word in shared?
    // All128 helper lanes observe same decision through a named group barrier.
    __shared__ int run;
    if(threadIdx.x==128)run=!finished;
    math_sync();if(!run)break;
    qk_shared(q,k,out,job,reinterpret_cast<B*>(reinterpret_cast<uint8_t*>(sm)+147968));
    ++done;
   }
   if(threadIdx.x==128){rec[blockIdx.x*8+2]=done;rec[blockIdx.x*8+3]=flashinfer::far_emu::now_ns();}
  }
 }
 __syncthreads();
}
int main(){
 ck(cudaSetDevice(0));constexpr int rows=215040+13,jobs=(rows+63)/64;
 size_t bytes=size_t(rows)*1152;uint4 *src,*dst;B *q,*k;float*out;U*records;
 ck(cudaMalloc(&src,bytes));ck(cudaMalloc(&dst,bytes));ck(cudaMalloc(&q,size_t(jobs)*64*576*2));ck(cudaMalloc(&k,size_t(jobs)*32*576*2));ck(cudaMalloc(&out,size_t(jobs)*64*32*4));ck(cudaMalloc(&records,32*8*sizeof(U)));
 std::vector<B> hq(size_t(jobs)*64*576),hk(size_t(jobs)*32*576);
 for(size_t i=0;i<hq.size();++i)hq[i]=__float2bfloat16(float(int((i*17+i/576)%37)-18)/16);
 for(size_t i=0;i<hk.size();++i)hk[i]=__float2bfloat16(float(int((i*11+i/576)%29)-14)/16);
 ck(cudaMemcpy(q,hq.data(),hq.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(k,hk.data(),hk.size()*2,cudaMemcpyHostToDevice));ck(cudaMemset(src,0x5a,bytes));
 ck(cudaFuncSetAttribute(pipeline<0>,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM));ck(cudaFuncSetAttribute(pipeline<1>,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM));ck(cudaFuncSetAttribute(pipeline<2>,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM));
 cudaFuncAttributes a[3];ck(cudaFuncGetAttributes(a,pipeline<0>));ck(cudaFuncGetAttributes(a+1,pipeline<1>));ck(cudaFuncGetAttributes(a+2,pipeline<2>));
 cudaEvent_t st,en;ck(cudaEventCreate(&st));ck(cudaEventCreate(&en));
 printf("{\"classification\":\"microprobe: not fullMLA or genuine QK replacement\",\"rows\":%d,\"jobs\":%d,\"bytes\":%zu,\"dynamic_shared\":%d,\"registers\":[%d,%d,%d],\"local_bytes\":[%zu,%zu,%zu],\"cases\":[",rows,jobs,bytes,SMEM,a[0].numRegs,a[1].numRegs,a[2].numRegs,a[0].localSizeBytes,a[1].localSizeBytes,a[2].localSizeBytes);
 bool first=true;
 for(U lat:{0ull,20000ull})for(int ctas:{24,32}){
  flashinfer::far_emu::Config c{10,lat,0,0,1};ck(cudaMemcpyToSymbol(flashinfer::far_emu::config,&c,sizeof(c)));
  for(int repeat=0;repeat<4;++repeat)for(int mi=0;mi<3;++mi){int mode=repeat%2==0?mi:2-mi;
   reset<<<1,1>>>();ck(cudaMemset(records,0,32*8*sizeof(U)));ck(cudaMemset(out,0x7f,size_t(jobs)*64*32*4));ck(cudaEventRecord(st));
   if(mode==0)pipeline<0><<<ctas,256,SMEM>>>(src,dst,q,k,out,rows,jobs,records);
   if(mode==1)pipeline<1><<<ctas,256,SMEM>>>(src,dst,q,k,out,rows,jobs,records);
   if(mode==2)pipeline<2><<<ctas,256,SMEM>>>(src,dst,q,k,out,rows,jobs,records);
   ck(cudaEventRecord(en));ck(cudaEventSynchronize(en));float ms;ck(cudaEventElapsedTime(&ms,st,en));
   std::vector<U> rec(ctas*8);ck(cudaMemcpy(rec.data(),records,rec.size()*sizeof(U),cudaMemcpyDeviceToHost));U begin=~0ull,end=0,helpers_end=0;int completed=0,early=0;
   for(int b=0;b<ctas;++b){begin=std::min(begin,rec[b*8]);end=std::max(end,rec[b*8+1]);helpers_end=std::max(helpers_end,rec[b*8+3]);completed+=rec[b*8+2];early+=rec[b*8+4];}
   if(early){fprintf(stderr,"Early far arrival %d\n",early);return 4;}
   if(repeat){if(!first)printf(",");first=false;printf("{\"latency_ns\":%llu,\"ctas\":%d,\"repeat\":%d,\"mode\":%d,\"us\":%.6f,\"fetch_span_us\":%.6f,\"completed_QK_jobs\":%d,\"helper_tail_us\":%.6f,\"early\":%d}",lat,ctas,repeat,mode,ms*1000,(end-begin)/1000.,completed,helpers_end>end?(helpers_end-end)/1000.:0,early);}
   if(repeat==0){std::vector<unsigned char> data(bytes);ck(cudaMemcpy(data.data(),dst,bytes,cudaMemcpyDeviceToHost));if(!std::all_of(data.begin(),data.end(),[](unsigned char x){return x==0x5a;}))return 5;
    if(mode==2){std::vector<float> got(size_t(jobs)*64*32);ck(cudaMemcpy(got.data(),out,got.size()*4,cudaMemcpyDeviceToHost));
     for(int b:{0,1,ctas/2,ctas-1})for(int seq:{0,int(rec[b*8+2])-1}){if(seq<0)continue;int job=b+seq*ctas;for(int head=0;head<64;++head)for(int key=0;key<32;++key){float sum=0;for(int d=0;d<576;++d){int f=d<64?512+d:d-64;sum+=__bfloat162float(hq[(size_t(job)*64+head)*576+f])*__bfloat162float(hk[(size_t(job)*32+key)*576+f]);}float x=got[(size_t(job)*64+head)*32+key];if(x!=sum){fprintf(stderr,"QK mismatch b%d job%d h%d k%d %f vs%f\n",b,job,head,key,x,sum);return 6;}}}
    }
   }
  }
 }
 printf("],\"sampled_QK_exact\":true,\"payload_exact\":true,\"all_recorded_arrivals_legal\":true}\n");return 0;
}
