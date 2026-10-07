// Purpose: Verify mapped pinned Host memory can be read directly by a CUDA kernel.
// Usage: nvcc -O2 temp/probe_mapped_host_tensor.cu -o temp/probe_mapped_host_tensor && temp/probe_mapped_host_tensor
#include <cuda_runtime.h>
#include <cstdio>
__global__ void read_kernel(const float* p, float* out) { out[0] = p[0] + p[1]; }
int main() {
  cudaSetDeviceFlags(cudaDeviceMapHost);
  float *h = nullptr, *d = nullptr, *out = nullptr;
  auto e = cudaHostAlloc(&h, 4096, cudaHostAllocMapped | cudaHostAllocPortable);
  if (e != cudaSuccess) { printf("alloc=%s\n", cudaGetErrorString(e)); return 1; }
  h[0] = 3.0f; h[1] = 4.0f;
  e = cudaHostGetDevicePointer((void**)&d, h, 0);
  if (e != cudaSuccess) { printf("map=%s\n", cudaGetErrorString(e)); return 1; }
  cudaMalloc(&out, sizeof(float));
  read_kernel<<<1,1>>>(d, out);
  cudaDeviceSynchronize();
  float answer = 0; cudaMemcpy(&answer, out, sizeof(float), cudaMemcpyDeviceToHost);
  printf("host=%p device=%p answer=%f error=%s\n", h, d, answer, cudaGetErrorString(cudaGetLastError()));
  cudaFree(out); cudaFreeHost(h); return answer == 7.0f ? 0 : 1;
}
