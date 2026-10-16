// Purpose: Isolate why scattered 1152 B Host reads fall short of the 27 GB/s bulk ceiling.
//          Sweeps access pattern (sequential / strided / random) at fixed total bytes, so
//          concurrency and volume are held constant and only locality varies.
// Usage: nvcc -O3 -arch=sm_80 -cubin temp/bench_host_gather_bw.cu -o temp/bench_host_gather_bw.cubin
#include <cuda_runtime.h>

// Each warp pulls `rows` rows of 1152 B (72 uint4) from mapped Host memory into registers.
extern "C" __global__ void gather(const uint4* __restrict__ cache,
                                  const int* __restrict__ index, uint4* __restrict__ sink,
                                  int rows, int vecs_per_row) {
  const int lane = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int warps = (gridDim.x * blockDim.x) >> 5;
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (int r = warp; r < rows; r += warps) {
    const uint4* src = cache + (long long)index[r] * vecs_per_row;
    for (int v = lane; v < vecs_per_row; v += 32) {
      uint4 x;
      asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w) : "l"(src + v) : "memory");
      acc.x ^= x.x; acc.y ^= x.y; acc.z ^= x.z; acc.w ^= x.w;
    }
  }
  if (acc.x == 0xdeadbeef) sink[warp] = acc;  // keep the loads live
}
