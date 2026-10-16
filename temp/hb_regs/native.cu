// Standalone native MLA kernel instantiation for register comparison.
#include <flashinfer/attention/mla.cuh>
using namespace flashinfer;
using P = MLAParams<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>;
using T = mla::KernelTraits<false, 2, true, 512, 64, 64, 32, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int32_t>;
template __global__ void mla::BatchMLAPagedAttentionKernel<T, P>(const __grid_constant__ P);
