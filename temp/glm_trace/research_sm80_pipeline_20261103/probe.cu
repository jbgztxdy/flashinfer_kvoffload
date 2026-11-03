// Same-warp memory/MMA scheduling microprobe; no end-to-end MLA claim.
#define FLASHINFER_FAR_EMU
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "flashinfer/mma.cuh"
#include "flashinfer/attention/far_link_emu.cuh"
using B=__nv_bfloat16;using U=unsigned long long;
void ck(cudaError_t x){if(x!=cudaSuccess){fprintf(stderr,"CUDA:%s\n",cudaGetErrorString(x));exit(2);}}
constexpr int V=64*1152/16,SMEM=147968+12288;
__device__ __forceinline__ void issue_round(const uint4* src,uint4* sm,int rows,int chunk,int round){int v=round*256+threadIdx.x;if(v<V && chunk*64+v/72<rows){unsigned dst=(unsigned)__cvta_generic_to_shared(sm+v);asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(dst),"l"(src+size_t(chunk)*V+v));}}
template<bool DISTRIBUTE>
__device__ __forceinline__ void qk_and_copy(const B* q,const B* k,float* out,int job,B* work,const uint4* src,uint4* nextsm,int rows,int next){
 int t=threadIdx.x,lane=t%32,warp=t/32;B* qs=work;B* ks=work+4096;float score[8];
 q+=size_t(job)*64*576;k+=size_t(job)*32*576;
#pragma unroll 1
 for(int block=0;block<9;++block){int feat=block==0?512:(block-1)*64;
  // Synchronous coalesced local operand loads keep the far cp.async groups
  // independent. Identical operand movement in bulk/distributed controls.
  for(int v=t;v<512;v+=256)*reinterpret_cast<uint4*>(qs+(v/8)*64+((v%8)^((v/8)%8))*8)=*reinterpret_cast<const uint4*>(q+(v/8)*576+feat+(v%8)*8);
  for(int v=t;v<256;v+=256)*reinterpret_cast<uint4*>(ks+(v/8)*64+((v%8)^((v/8)%8))*8)=*reinterpret_cast<const uint4*>(k+(v/8)*576+feat+(v%8)*8);
  __syncthreads();
#pragma unroll
  for(int dim=0;dim<4;++dim){uint32_t a[4],b[4];
   flashinfer::mma::ldmatrix_m8n8x4(a,qs+((warp%4)*16+lane%16)*64+((dim*16+8*(lane/16))^((lane%8)*8)));
   flashinfer::mma::ldmatrix_m8n8x4(b,ks+((warp/4)*16+8*(lane/16)+lane%8)*64+((dim*16+8*((lane%16)/8))^((lane%8)*8)));
   if(block==0&&dim==0)flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<B,flashinfer::mma::MMAMode::kInit>(score,a,b);
   else flashinfer::mma::mma_sync_m16n16k16_row_col_f16f16f32<B>(score,a,b);
   if constexpr(DISTRIBUTE)if((block*4+dim)%2==0 && next*64<rows)issue_round(src,nextsm,rows,next,(block*4+dim)/2);
  }
  __syncthreads();
 }
#pragma unroll
 for(int i=0;i<8;++i)out[(size_t(job)*256+t)*8+i]=score[i];
}
__global__ void reset(){if(threadIdx.x==0)flashinfer::far_emu::link_clock=0;}
template<int MODE> //0=plainfetch,1=bulk async then local QK,2=spreadcpasync acrossMMA
__global__ void pipeline(const uint4* src,uint4* dst,const B* q,const B* k,float* scores,int rows){
 extern __shared__ __align__(128) uint4 sm[];B* work=reinterpret_cast<B*>(reinterpret_cast<uint8_t*>(sm)+147968);
 FAR_EMU_INIT(2);FAR_EMU_SYNC();
 auto issue=[&](int chunk,int stage){
  for(int round=0;round<18;++round)issue_round(src,sm+stage*V,rows,chunk,round);
  FAR_EMU_TILE(2,stage,threadIdx.x==0?min(64,rows-chunk*64):0,1152);asm volatile("cp.async.commit_group;");
 };
 int chunk=blockIdx.x,step=0;if(chunk*64<rows)issue(chunk,0);
 for(;chunk*64<rows;chunk+=gridDim.x,++step){int next=chunk+gridDim.x,stage=(step+1)%2;
  if constexpr(MODE!=2){if(next*64<rows)issue(next,stage);}
  if constexpr(MODE==1)qk_and_copy<false>(q,k,scores,chunk,work,src,sm+stage*V,rows,next);
  if constexpr(MODE==2){
   qk_and_copy<true>(q,k,scores,chunk,work,src,sm+stage*V,rows,next);
   if(next*64<rows){FAR_EMU_TILE(2,stage,threadIdx.x==0?min(64,rows-next*64):0,1152);asm volatile("cp.async.commit_group;");}
  }
  if(next*64<rows){asm volatile("cp.async.wait_group 1;");}else asm volatile("cp.async.wait_group 0;");
  FAR_EMU_WAIT(2,step%2);__syncthreads();
  for(int v=threadIdx.x;v<V&&chunk*64+v/72<rows;v+=256)dst[size_t(chunk)*V+v]=sm[(step%2)*V+v];__syncthreads();
 }
}
int main(){
 ck(cudaSetDevice(0));constexpr int rows=215040+13,jobs=(rows+63)/64;size_t bytes=size_t(rows)*1152;
 uint4 *src,*dst;B*q,*k;float*out;ck(cudaMalloc(&src,bytes));ck(cudaMalloc(&dst,bytes));ck(cudaMalloc(&q,size_t(jobs)*64*576*2));ck(cudaMalloc(&k,size_t(jobs)*32*576*2));ck(cudaMalloc(&out,size_t(jobs)*64*32*4));
 std::vector<B>hq(size_t(jobs)*64*576),hk(size_t(jobs)*32*576);
 for(size_t i=0;i<hq.size();++i)hq[i]=__float2bfloat16(float(int((i*17+i/576)%37)-18)/16);
 for(size_t i=0;i<hk.size();++i)hk[i]=__float2bfloat16(float(int((i*11+i/576)%29)-14)/16);
 ck(cudaMemcpy(q,hq.data(),hq.size()*2,cudaMemcpyHostToDevice));ck(cudaMemcpy(k,hk.data(),hk.size()*2,cudaMemcpyHostToDevice));ck(cudaMemset(src,0x5a,bytes));
 ck(cudaFuncSetAttribute(pipeline<0>,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM));ck(cudaFuncSetAttribute(pipeline<1>,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM));ck(cudaFuncSetAttribute(pipeline<2>,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM));
 cudaFuncAttributes a[3];ck(cudaFuncGetAttributes(a,pipeline<0>));ck(cudaFuncGetAttributes(a+1,pipeline<1>));ck(cudaFuncGetAttributes(a+2,pipeline<2>));
 cudaEvent_t st,en;ck(cudaEventCreate(&st));ck(cudaEventCreate(&en));
 printf("{\"classification\":\"instruction schedule microprobe, not full MLA nor real trace work replacement\",\"rows\":%d,\"jobs\":%d,\"bytes\":%zu,\"registers\":[%d,%d,%d],\"local_bytes\":[%zu,%zu,%zu],\"cases\":[",rows,jobs,bytes,a[0].numRegs,a[1].numRegs,a[2].numRegs,a[0].localSizeBytes,a[1].localSizeBytes,a[2].localSizeBytes);bool first=true;
 for(U lat:{0ull,20000ull})for(int ctas:{24,108}){
  flashinfer::far_emu::Config c{10,lat,0,0,1};ck(cudaMemcpyToSymbol(flashinfer::far_emu::config,&c,sizeof(c)));
  for(int rep=0;rep<4;++rep)for(int i=0;i<3;++i){int mode=rep%2==0?i:2-i;
   reset<<<1,1>>>();ck(cudaMemset(out,0x7f,size_t(jobs)*64*32*4));ck(cudaEventRecord(st));
   if(mode==0)pipeline<0><<<ctas,256,SMEM>>>(src,dst,q,k,out,rows);
   if(mode==1)pipeline<1><<<ctas,256,SMEM>>>(src,dst,q,k,out,rows);
   if(mode==2)pipeline<2><<<ctas,256,SMEM>>>(src,dst,q,k,out,rows);
   ck(cudaEventRecord(en));ck(cudaEventSynchronize(en));float ms;ck(cudaEventElapsedTime(&ms,st,en));
   if(rep){if(!first)printf(",");first=false;printf("{\"latency_ns\":%llu,\"ctas\":%d,\"repeat\":%d,\"mode\":%d,\"us\":%.6f}",lat,ctas,rep,mode,ms*1000);}
   if(rep==0){std::vector<unsigned char> payload(bytes);ck(cudaMemcpy(payload.data(),dst,bytes,cudaMemcpyDeviceToHost));if(!std::all_of(payload.begin(),payload.end(),[](unsigned char x){return x==0x5a;}))return 3;
    if(mode){std::vector<float>got(size_t(jobs)*2048);ck(cudaMemcpy(got.data(),out,got.size()*4,cudaMemcpyDeviceToHost));
     for(int job:{0,1,jobs/2,jobs-1})for(int t=0;t<256;++t)for(int j=0;j<8;++j){int lane=t%32,warp=t/32,h=(warp%4)*16+lane/4+((j%4)/2)*8,key=(warp/4)*16+2*(lane%4)+j%2+(j/4)*8;float sum=0;
      for(int d=0;d<576;++d){int f=d<64?512+d:d-64;sum+=__bfloat162float(hq[(size_t(job)*64+h)*576+f])*__bfloat162float(hk[(size_t(job)*32+key)*576+f]);}
      if(got[(size_t(job)*256+t)*8+j]!=sum){fprintf(stderr,"QK mismatch mode%d job%d t%d j%d\n",mode,job,t,j);return 4;}
     }
    }
   }
  }
 }
 printf("],\"sampled_QK_exact\":true,\"all_payload_exact\":true}\n");return 0;
}
