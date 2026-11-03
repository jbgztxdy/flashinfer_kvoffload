#define main verified_main
#include "verify.cu"
#undef main
int main(){
 check(cudaSetDevice(0));const int rows=262144+13,chunks=(rows+63)/64;size_t bytes=(size_t)rows*1152;uint4*src,*dst;U*times;
 check(cudaMalloc(&src,bytes));check(cudaMalloc(&dst,bytes));check(cudaMalloc(&times,chunks*sizeof(U)));check(cudaMemset(src,0x5a,bytes));check(cudaFuncSetAttribute(payload,cudaFuncAttributeMaxDynamicSharedMemorySize,147456));
 cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));printf("{\"scope\":\"Transport microkernel,notMLAoverheaddecomposition\",\"cases\":[");bool first=true;
 for(int mode=0;mode<3;++mode){int hooks=mode!=0;flashinfer::far_emu::Config cfg{0,0,0,0,mode==2};check(cudaMemcpyToSymbol(flashinfer::far_emu::config,&cfg,sizeof(cfg)));std::vector<float>samples;
  for(int rep=0;rep<4;++rep){reset<<<1,1>>>();check(cudaEventRecord(start));payload<<<24,256,147456>>>(src,dst,rows,times,hooks);check(cudaEventRecord(end));check(cudaEventSynchronize(end));float ms;check(cudaEventElapsedTime(&ms,start,end));if(rep)samples.push_back(ms*1000);}
  if(!first)printf(",");first=false;printf("{\"mode\":%d,\"hooks\":%d,\"config_enabled\":%d,\"bytes\":%zu,\"samples_us\":[%.3f,%.3f,%.3f]}",mode,hooks,mode==2,bytes,samples[0],samples[1],samples[2]);
 }
 printf("]}\n");check(cudaFree(src));check(cudaFree(dst));check(cudaFree(times));return 0;
}
