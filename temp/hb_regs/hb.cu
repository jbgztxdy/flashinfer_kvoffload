// Standalone instantiation of the batched Host kernel for register/stack inspection.
#include <flashinfer/attention/mla_host_batch.cuh>
using namespace flashinfer;
using P = MLAHostBatchParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>;
using T = mla::KernelTraits<false, 2, true, 512, 64, 64, 32, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>;
template __global__ void mla::BatchMLAHostBatchKernel<T, P>(const __grid_constant__ P);
