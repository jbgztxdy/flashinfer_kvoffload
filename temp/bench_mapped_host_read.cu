// Purpose: Measure GPU kernel read bandwidth from CUDA mapped pinned Host DRAM.
// Usage: nvcc -O3 -arch=sm_80 temp/bench_mapped_host_read.cu -o temp/bench_mapped_host_read && taskset -c 0-31,64-95 temp/bench_mapped_host_read 1 > temp/mapped_host_read_bandwidth.csv

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#define CHECK(call) do { cudaError_t status = (call); if (status != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(status)); std::exit(1); } } while (0)

__global__ void read_mapped(const uint4* data, unsigned long long* sums, size_t count) {
  size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  unsigned long long sum = 0;
  for (size_t i = tid; i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
    uint4 value;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(data + i) : "memory");
    sum += static_cast<unsigned long long>(value.x) + value.y + value.z + value.w;
  }
  sums[tid] = sum;
}

int main(int argc, char** argv) {
  int device = argc > 1 ? std::atoi(argv[1]) : 1;
  CHECK(cudaSetDevice(device));
  CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  cudaDeviceProp properties{};
  CHECK(cudaGetDeviceProperties(&properties, device));
  std::fprintf(stderr, "device=%d name=%s canMapHostMemory=%d SMs=%d\n", device, properties.name, properties.canMapHostMemory, properties.multiProcessorCount);
  std::printf("mib,blocks,host_pointer,device_pointer,median_us,p10_us,p90_us,median_GB_per_s,verified\n");
  for (size_t mib : {64UL, 256UL}) {
    size_t bytes = mib * 1024 * 1024;
    size_t count = bytes / sizeof(uint4);
    uint4* host = nullptr;
    uint4* mapped = nullptr;
    CHECK(cudaHostAlloc(&host, bytes, cudaHostAllocMapped | cudaHostAllocPortable));
    CHECK(cudaHostGetDevicePointer(&mapped, host, 0));
    for (size_t i = 0; i < count; ++i) host[i] = make_uint4(1, 2, 3, 4);
    for (int blocks_per_sm : {4, 8, 16}) {
      int blocks = properties.multiProcessorCount * blocks_per_sm;
      size_t threads = static_cast<size_t>(blocks) * 256;
      unsigned long long* output = nullptr;
      CHECK(cudaMalloc(&output, threads * sizeof(unsigned long long)));
      read_mapped<<<blocks, 256>>>(mapped, output, count);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      std::vector<unsigned long long> reference(threads);
      CHECK(cudaMemcpy(reference.data(), output, reference.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
      unsigned long long total = std::accumulate(reference.begin(), reference.end(), 0ULL);
      if (total != 10ULL * count) { std::fprintf(stderr, "Incorrect read sum\n"); return 1; }
      cudaStream_t stream;
      CHECK(cudaStreamCreate(&stream));
      cudaGraph_t graph;
      cudaGraphExec_t executable;
      CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      constexpr int repeats = 5;
      for (int i = 0; i < repeats; ++i) read_mapped<<<blocks, 256, 0, stream>>>(mapped, output, count);
      CHECK(cudaStreamEndCapture(stream, &graph));
      CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
      cudaEvent_t start, end;
      CHECK(cudaEventCreate(&start));
      CHECK(cudaEventCreate(&end));
      CHECK(cudaGraphLaunch(executable, stream));
      CHECK(cudaStreamSynchronize(stream));
      std::vector<float> times;
      for (int round = 0; round < 20; ++round) {
        CHECK(cudaEventRecord(start, stream));
        CHECK(cudaGraphLaunch(executable, stream));
        CHECK(cudaEventRecord(end, stream));
        CHECK(cudaEventSynchronize(end));
        float ms;
        CHECK(cudaEventElapsedTime(&ms, start, end));
        times.push_back(ms * 1000 / repeats);
      }
      std::sort(times.begin(), times.end());
      double median_us = (times[9] + times[10]) / 2;
      std::printf("%zu,%d,%p,%p,%.3f,%.3f,%.3f,%.6f,true\n", mib, blocks, static_cast<void*>(host), static_cast<void*>(mapped), median_us, times[1], times[17], bytes / (median_us * 1000));
      std::fflush(stdout);
      CHECK(cudaEventDestroy(start));
      CHECK(cudaEventDestroy(end));
      CHECK(cudaGraphExecDestroy(executable));
      CHECK(cudaGraphDestroy(graph));
      CHECK(cudaStreamDestroy(stream));
      CHECK(cudaFree(output));
    }
    CHECK(cudaFreeHost(host));
  }
  return 0;
}
