// Feasibility microprobe ONLY. Not integrated MLA, not an optimized attention result.
#define FLASHINFER_FAR_EMU
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include "flashinfer/mma.cuh"
#include "flashinfer/attention/far_link_emu.cuh"
using B=__nv_bfloat16;using U=unsigned long long;
void ck(cudaError_t s){if(s!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(s));exit(2);}}
constexpr int V=64*1152/16;
__device__ __forceinline__ void qk(const B* q,const B* k,float* out,int job){
 int lane=threadIdx.x%32,warp=threadIdx.x/32,head=(warp%4)*16+lane/4,col=(lane%4)*2,key=(warp/4)*16+lane/4;
 q+=size_t(job)*64*576;k+=size_t(job)*32*576;float s[8];
 #pragma unroll 1
 for(int dim=0;dim<36;++dim){
  int feat=dim<4?512+dim*16:(dim-4)*16;
  uint32_t a[4],b[4];
  a[0]=*reinterpret_cast<const uint32_t*>(q+head*576+feat+col);
  a[1]=*reinterpret_cast<const uint32_t*>(q+(head+8)*576+feat+col);
  a[2]=*reinterpret_cast<const uint32_t*>(q+head*576+feat+col+8);
  a[3]=*reinterpret_cast<const uint32_t*>(q+(head+8)*576+feat+col+8);
  b[0]=*reinterpret_cast<const uint32_t*>(k+key*576+feat+col);
  b[1]=*reinterpret_cast<const uint32_t*>(k+key*576+feat+col+8);
  b[2]=*reinterpret_cast<const uint32_t*>(k+(key+8)*576+feat+col);
  b[3]=*reinterpret_cast<const uint32_t*>(k+(key+8)*576+feat+col+8);
  if(dim==0)flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<B,flashinfer::mma::MMAMode::kInit>(s,a,b);
  else flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<B>(s,a,b);
 }
 #pragma unroll
 for(int i=0;i<8;++i){int h=head+((i%4)/2)*8,kv=(warp/4)*16+col+(i%2)+(i/4)*8;out[(size_t(job)*64+h)*32+kv]=s[i];}
}
__global__ void reset(){if(!threadIdx.x)flashinfer::far_emu::link_clock=0;}
__global__ void standalone(const B* q,const B* k,float* out,int jobs){for(int j=blockIdx.x;j<jobs;j+=gridDim.x)qk(q,k,out,j);}
template<bool ASSIST>
__global__ void pipeline(const uint4* src,uint4* dst,const B* q,const B* k,float* out,int rows){
 extern __shared__ __align__(128) uint4 sm[];
 FAR_EMU_INIT(2);FAR_EMU_SYNC();
 auto issue=[&](int chunk,int stage){
  for(int v=threadIdx.x;v<V && chunk*64+v/72<rows;v+=256){unsigned a=(unsigned)__cvta_generic_to_shared(sm+stage*V+v);asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(a),"l"(src+size_t(chunk)*V+v));}
  FAR_EMU_TILE(2,stage,threadIdx.x==0?min(64,rows-chunk*64):0,1152);
  asm volatile("cp.async.commit_group;");
 };
 int chunk=blockIdx.x,step=0;if(chunk*64<rows)issue(chunk,0);
 for(;chunk*64<rows;chunk+=gridDim.x,++step){
  int next=chunk+gridDim.x;
  if(next*64<rows){issue(next,(step+1)%2);asm volatile("cp.async.wait_group 1;");}
  else asm volatile("cp.async.wait_group 0;");
  // Operands are independent already-local data. Never consume far bytes before deadline.
  // FAR hooks and serializer are unchanged; full output publication is charged here.
  if constexpr(ASSIST)qk(q,k,out,chunk);
  FAR_EMU_WAIT(2,step%2);__syncthreads();
  for(int v=threadIdx.x;v<V && chunk*64+v/72<rows;v+=256)dst[size_t(chunk)*V+v]=sm[(step%2)*V+v];
  __syncthreads();
 }
}
int main(){
 ck(cudaSetDevice(0));constexpr int rows=215040+13,jobs=(rows+63)/64;
 size_t bytes=size_t(rows)*1152;uint4 *src,*dst;B *q,*k;float*out;
 ck(cudaMalloc(&src,bytes));ck(cudaMalloc(&dst,bytes));ck(cudaMalloc(&q,size_t(jobs)*64*576*sizeof(B)));ck(cudaMalloc(&k,size_t(jobs)*32*576*sizeof(B)));ck(cudaMalloc(&out,size_t(jobs)*64*32*sizeof(float)));
 std::vector<B> hq(size_t(jobs)*64*576),hk(size_t(jobs)*32*576);
 for(size_t i=0;i<hq.size();++i)hq[i]=__float2bfloat16(float(int((i*17+i/576)%37)-18)/16);
 for(size_t i=0;i<hk.size();++i)hk[i]=__float2bfloat16(float(int((i*11+i/576)%29)-14)/16);
 ck(cudaMemcpy(q,hq.data(),hq.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(k,hk.data(),hk.size()*2,cudaMemcpyHostToDevice));ck(cudaMemset(src,0x5a,bytes));
 ck(cudaFuncSetAttribute(pipeline<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,147968));ck(cudaFuncSetAttribute(pipeline<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,147968));
 cudaFuncAttributes a,b;ck(cudaFuncGetAttributes(&a,pipeline<false>));ck(cudaFuncGetAttributes(&b,pipeline<true>));
 cudaEvent_t st,en;ck(cudaEventCreate(&st));ck(cudaEventCreate(&en));
 printf("{\"classification\":\"standalone feasibility only; no MLA prepare/LRU/concurrent84CTA or mailbox consumer\",\"rows\":%d,\"jobs\":%d,\"bytes\":%zu,\"registers\":[%d,%d],\"local_bytes\":[%zu,%zu],\"cases\":[",rows,jobs,bytes,a.numRegs,b.numRegs,a.localSizeBytes,b.localSizeBytes);
 bool first=true;
 for(U lat:{0ull,20000ull}){
  flashinfer::far_emu::Config c{10,lat,0,0,1};ck(cudaMemcpyToSymbol(flashinfer::far_emu::config,&c,sizeof(c)));
  for(int repeat=0;repeat<4;++repeat){
   for(int mode_i=0;mode_i<3;++mode_i){int mode=repeat%2==0?mode_i:2-mode_i;
    reset<<<1,1>>>();ck(cudaMemset(out,0x7f,size_t(jobs)*64*32*sizeof(float)));ck(cudaEventRecord(st));
    if(mode==0)pipeline<false><<<24,256,147968>>>(src,dst,q,k,out,rows);
    if(mode==1)pipeline<true><<<24,256,147968>>>(src,dst,q,k,out,rows);
    if(mode==2)standalone<<<24,256>>>(q,k,out,jobs);
    ck(cudaEventRecord(en));ck(cudaEventSynchronize(en));float ms;ck(cudaEventElapsedTime(&ms,st,en));
    if(repeat){if(!first)printf(",");first=false;printf("{\"latency_ns\":%llu,\"repeat\":%d,\"mode\":%d,\"us\":%.6f}",lat,repeat,mode,ms*1000);}
    if(repeat==0 && mode!=0){
     std::vector<float> got(size_t(jobs)*64*32);ck(cudaMemcpy(got.data(),out,got.size()*4,cudaMemcpyDeviceToHost));
     for(int job=0;job<jobs;++job)for(int head=0;head<64;++head)for(int key=0;key<32;++key){float sum=0;for(int d=0;d<576;++d){int f=d<64?512+d:d-64;sum+=__bfloat162float(hq[(size_t(job)*64+head)*576+f])*__bfloat162float(hk[(size_t(job)*32+key)*576+f]);}float x=got[(size_t(job)*64+head)*32+key];if(x!=sum){fprintf(stderr,"QK mismatch job%d h%d k%d got%f expected%f\n",job,head,key,x,sum);return 3;}}
    }
    if(repeat==0 && mode!=2){std::vector<unsigned char> data(bytes);ck(cudaMemcpy(data.data(),dst,bytes,cudaMemcpyDeviceToHost));if(!std::all_of(data.begin(),data.end(),[](unsigned char x){return x==0x5a;}))return 4;}
   }
  }
 }
 printf("],\"all_QK_outputs_exact\":true,\"all_payload_bytes_equal\":true}\n");return 0;
}
