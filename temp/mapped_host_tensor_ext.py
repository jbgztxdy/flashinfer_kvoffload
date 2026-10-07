# Purpose: Allocate CUDA mapped pinned Host memory and expose it as a CUDA tensor.
# Usage: Imported by the mapped Host DRAM benchmark.

import torch
from torch.utils.cpp_extension import load_inline

_EXT = None

def _extension():
    global _EXT
    if _EXT is None:
        src = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

struct Holder { void* ptr; size_t bytes; };

void release(void* ctx) {
  auto* h = static_cast<Holder*>(ctx);
  cudaFreeHost(h->ptr);
  delete h;
}

torch::Tensor allocate(std::vector<int64_t> sizes, int64_t device) {
  TORCH_CHECK(device >= 0, "device must be nonnegative");
  c10::cuda::CUDAGuard guard(device);
  auto st = torch::prod(torch::tensor(sizes, torch::kInt64)).item<int64_t>();
  void* host = nullptr;
  auto e = cudaHostAlloc(&host, st * 2, cudaHostAllocMapped | cudaHostAllocPortable);
  TORCH_CHECK(e == cudaSuccess, "cudaHostAlloc failed: ", cudaGetErrorString(e));
  void* dev = nullptr;
  e = cudaHostGetDevicePointer(&dev, host, 0);
  if (e != cudaSuccess) { cudaFreeHost(host); TORCH_CHECK(false, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(e)); }
  auto* holder = new Holder{host, static_cast<size_t>(st * 2)};
  auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA, device);
  return torch::from_blob(dev, sizes, [holder](void*) { release(holder); }, options);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("allocate", &allocate); }
'''
        _EXT = load_inline(name='mapped_host_tensor_ext_v1', cpp_sources=src, functions=None, with_cuda=True, extra_cflags=['-O2'], extra_cuda_cflags=['-O2'], verbose=False)
    return _EXT

def allocate(shape, device):
    return _extension().allocate(list(shape), int(device))
