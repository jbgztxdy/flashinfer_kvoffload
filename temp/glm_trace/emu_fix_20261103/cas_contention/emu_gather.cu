// Experiment-only row gather (dst[i] = src[ids[i]], 1152-byte rows) through the same emulated far
// link as the kernels (include/flashinfer/attention/far_link_emu.cuh): the Memcpy baselines'
// gather of missed rows over the emulated link. Settings from FLASHINFER_FAR_EMU; far rows are
// all rows (the gather only reads the far tier).
#define FLASHINFER_FAR_EMU
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include "flashinfer/attention/far_link_emu.cuh"

constexpr int ROW_VECS = 72, ROWS = 16, THREADS = 256, PER = (ROWS * ROW_VECS + THREADS - 1) / THREADS;

__global__ void gather(const uint4* src, const int64_t* ids, uint4* dst, int64_t n) {
  __shared__ unsigned long long ready;
  for (int64_t base = (int64_t)blockIdx.x * ROWS; base < n; base += (int64_t)gridDim.x * ROWS) {
    const int rows = n - base < ROWS ? (int)(n - base) : ROWS;
    // Match the kernel model: reserve when this packet is issued, before its loads;
    // real source reads and the virtual serialization/latency then overlap.
    if (threadIdx.x == 0) ready = flashinfer::far_emu::reserve((unsigned long long)rows * ROW_VECS * 16, flashinfer::far_emu::now_ns());
    uint4 v[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int k = i * THREADS + threadIdx.x;
      if (k < rows * ROW_VECS) v[i] = src[ids[base + k / ROW_VECS] * ROW_VECS + k % ROW_VECS];
    }
    __syncthreads();
    if (threadIdx.x == 0) flashinfer::far_emu::wait_until(ready);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < PER; ++i) {
      const int k = i * THREADS + threadIdx.x;
      if (k < rows * ROW_VECS) dst[(base + k / ROW_VECS) * ROW_VECS + k % ROW_VECS] = v[i];
    }
    __syncthreads();
  }
}

// One CTA, `steps` dependent reads of one row each through the emulated link: per-access latency.
__global__ void chase(const uint4* src, const int64_t* ids, int steps, unsigned long long* out) {
  __shared__ unsigned long long ready;
  __shared__ uint4 sink[ROW_VECS];
  const unsigned long long t0 = flashinfer::far_emu::now_ns();
  int64_t row = ids[0];
  for (int s = 0; s < steps; ++s) {
    if (threadIdx.x == 0) ready = flashinfer::far_emu::reserve(ROW_VECS * 16, flashinfer::far_emu::now_ns());
    if (threadIdx.x < ROW_VECS) sink[threadIdx.x] = src[row * ROW_VECS + threadIdx.x];
    __syncthreads();
    if (threadIdx.x == 0) flashinfer::far_emu::wait_until(ready);
    __syncthreads();
    row = ids[(s + 1 + (sink[0].x & 1)) % steps];  // the next row depends on the data read
  }
  if (threadIdx.x == 0) *out = (flashinfer::far_emu::now_ns() - t0) / steps;
}

int64_t latency(torch::Tensor src, torch::Tensor ids, int64_t steps) {
  auto stream = c10::cuda::getCurrentCUDAStream();
  FAR_EMU_CONFIGURE(stream);
  auto out = torch::zeros({1}, ids.options());
  chase<<<1, 128, 0, stream>>>(reinterpret_cast<const uint4*>(src.data_ptr()), ids.data_ptr<int64_t>(), steps,
                               reinterpret_cast<unsigned long long*>(out.data_ptr()));
  return out.item<int64_t>();
}

void run(torch::Tensor src, torch::Tensor ids, torch::Tensor dst) {
  const int64_t n = ids.numel();
  if (!n) return;
  auto stream = c10::cuda::getCurrentCUDAStream();
  FAR_EMU_CONFIGURE(stream);
  gather<<<432, THREADS, 0, stream>>>(reinterpret_cast<const uint4*>(src.data_ptr()), ids.data_ptr<int64_t>(),
                                       reinterpret_cast<uint4*>(dst.data_ptr()), n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gather", &run); m.def("latency", &latency); }
