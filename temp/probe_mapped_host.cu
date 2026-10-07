// Purpose: Verify CUDA mapped pinned Host memory and obtain its device pointer.
// Usage: nvcc -O2 temp/probe_mapped_host.cu -o temp/probe_mapped_host && temp/probe_mapped_host

#include <cuda_runtime.h>
#include <cstdio>

int main() {
  cudaError_t err = cudaSetDeviceFlags(cudaDeviceMapHost);
  if (err != cudaSuccess && err != cudaErrorSetOnActiveProcess) {
    std::printf("cudaSetDeviceFlags: %s\n", cudaGetErrorString(err));
    return 1;
  }
  int can_map = 0;
  err = cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, 0);
  if (err != cudaSuccess) {
    std::printf("cudaDeviceGetAttribute: %s\n", cudaGetErrorString(err));
    return 1;
  }
  void* host_ptr = nullptr;
  err = cudaHostAlloc(&host_ptr, 4096, cudaHostAllocMapped | cudaHostAllocPortable);
  if (err != cudaSuccess) {
    std::printf("cudaHostAlloc: %s\n", cudaGetErrorString(err));
    return 1;
  }
  void* device_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
  std::printf("canMapHostMemory=%d host=%p device=%p same=%d\n", can_map,
              host_ptr, device_ptr, host_ptr == device_ptr);
  if (err != cudaSuccess) {
    std::printf("cudaHostGetDevicePointer: %s\n", cudaGetErrorString(err));
    cudaFreeHost(host_ptr);
    return 1;
  }
  cudaFreeHost(host_ptr);
  return 0;
}
