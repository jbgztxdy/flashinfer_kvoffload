// Purpose: Measure mapped Host reads for the actual sparse MLA addresses.
// Usage: TMPDIR=$PWD/temp nvcc -O3 -arch=sm_80 -cubin temp/bench_sparse_host_fetch.cu -o temp/bench_sparse_host_fetch.cubin
#include <cuda_runtime.h>

extern "C" __global__ void sparse_fetch(const uint4* cache, const int* indices, uint4* output, int boundary) {
  uint4 values[16];
  int positions[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    int position = blockIdx.x * 256 + threadIdx.x + i * gridDim.x * 256;
    int page = position < 2048 * 72 ? indices[position / 72] : -1;
    positions[i] = page >= boundary ? position : -1;
    if (positions[i] >= 0) {
      asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(values[i].x), "=r"(values[i].y), "=r"(values[i].z), "=r"(values[i].w)
                   : "l"(cache + page * 72 + position % 72) : "memory");
    }
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) if (positions[i] >= 0) output[positions[i]] = values[i];
}
