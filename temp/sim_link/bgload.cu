// Purpose: occupy part of the GPU1 -> GPU0 NVLink direction, to emulate a slower far tier for
// kernels on GPU0 that read GPU1 memory. CTAs on GPU1 store into a small buffer on GPU0 (default
// 2 MB, which stays resident in GPU0's 40 MB write-back L2, so GPU0 HBM sees almost no traffic).
// Rate knobs: CTA count and a nanosleep between each warp's 512-byte store rounds.
// Usage: ./bgload <ctas> <threads> <sleep_ns> <seconds> [buffer_MB]   (runs until killed if seconds <= 0)
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t;
}

// Each launch runs ~`span_ns`; every warp stores 16 B per lane (512 B) per round to the remote
// buffer, then sleeps `sleep_ns`.
__global__ void store_loop(uint4* dst, size_t vecs, unsigned sleep_ns, unsigned long long span_ns,
                           unsigned long long* bytes) {
  const unsigned long long end = now_ns() + span_ns;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const uint4 v = make_uint4(i, i + 1, i + 2, i + 3);
  unsigned long long n = 0;
  while (now_ns() < end) {
#pragma unroll 4
    for (int k = 0; k < 4; ++k) {
      dst[i % vecs] = v;
      i += stride;
    }
    n += 4;
    if (sleep_ns) __nanosleep(sleep_ns);
  }
  atomicAdd(bytes, n * 16);
}

static volatile sig_atomic_t stop = 0;
static void on_signal(int) { stop = 1; }

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr, "usage: %s ctas threads sleep_ns seconds [buffer_MB]\n", argv[0]); return 2; }
  const int ctas = atoi(argv[1]);
  const int threads = atoi(argv[2]);
  const unsigned sleep_ns = atoi(argv[3]);
  const double seconds = atof(argv[4]);
  const size_t mb = argc > 5 ? atoi(argv[5]) : 2;
  signal(SIGTERM, on_signal);
  signal(SIGINT, on_signal);
  uint4* dst;
  CK(cudaSetDevice(0));
  CK(cudaMalloc(&dst, mb << 20));
  CK(cudaSetDevice(1));
  CK(cudaDeviceEnablePeerAccess(0, 0));
  unsigned long long* bytes;
  CK(cudaMallocManaged(&bytes, sizeof(*bytes)));
  *bytes = 0;
  cudaEvent_t a, b;
  CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a));
  int launches = 0;
  // 20 ms launches back to back keep the link busy and leave the process killable.
  while (!stop && (seconds <= 0 || launches * 0.02 < seconds)) {
    store_loop<<<ctas, threads>>>(dst, (mb << 20) / 16, sleep_ns, 20000000ull, bytes);
    ++launches;
    if (launches % 8 == 0) CK(cudaDeviceSynchronize());
  }
  CK(cudaEventRecord(b));
  CK(cudaDeviceSynchronize());
  float ms; CK(cudaEventElapsedTime(&ms, a, b));
  printf("bgload ctas=%d threads=%d sleep_ns=%u buffer=%zuMB: %.1f GB/s over %.2f s\n", ctas, threads, sleep_ns, mb,
         *bytes / (ms * 1e6), ms / 1e3);
  return 0;
}
