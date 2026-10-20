// Purpose: per-CTA read bandwidth of random 1152-byte rows from a far tier (peer GPU over NVLink,
// or mapped Host memory over PCIe) into shared memory, as the fetch CTAs do, versus CTA count and
// in-flight depth (cp.async stages of ROWS_PER_STAGE rows), with or without writing the rows back
// to local HBM. Usage: ./fetch_bw peer|host
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#define CK(x) do { cudaError_t e = (x); if (e) { printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
constexpr int ROW = 1152, VECS = ROW / 16;
__device__ int THREADS_D;

__device__ __forceinline__ void cp16(void* s, const void* g) {
  unsigned sa = __cvta_generic_to_shared(s);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa), "l"(g));
}
template <int N> __device__ __forceinline__ void wait_group() { asm volatile("cp.async.wait_group %0;" ::"n"(N)); }

// Each CTA walks `per_cta` rows of `ids` in stages of R rows; S stages in flight.
template <int S>
__global__ void fetch(const uint4* far, const int* ids, uint4* out, int per_cta, int R, bool write) {
  extern __shared__ uint4 buf[];
  const int base = blockIdx.x * per_cta, stage_vecs = R * VECS, stages = per_cta / R;
  auto issue = [&](int st) {
    if (st >= stages) { asm volatile("cp.async.commit_group;"); return; }
    uint4* b = buf + (st % S) * stage_vecs;
    for (int v = threadIdx.x; v < stage_vecs; v += blockDim.x) {
      const int row = ids[base + st * R + v / VECS];
      cp16(b + v, far + (long)row * VECS + v % VECS);
    }
    asm volatile("cp.async.commit_group;");
  };
  for (int st = 0; st < S - 1; ++st) issue(st);
  for (int st = 0; st < stages; ++st) {
    issue(st + S - 1);
    wait_group<S - 1>();
    __syncthreads();
    if (write) {
      uint4* b = buf + (st % S) * stage_vecs;
      for (int v = threadIdx.x; v < stage_vecs; v += blockDim.x) out[(long)(base + st * R) * VECS + v] = b[v];
    }
    __syncthreads();
  }
}

int main(int argc, char** argv) {
  const bool host = argc > 1 && argv[1][0] == 'h';
  const long far_rows = 4L << 20;  // 4.8 GB pool
  CK(cudaSetDevice(0));
  uint4* far;
  if (host) {
    CK(cudaHostAlloc((void**)&far, far_rows * ROW, cudaHostAllocMapped));
  } else {
    CK(cudaSetDevice(1)); CK(cudaMalloc(&far, far_rows * ROW)); CK(cudaMemset(far, 1, far_rows * ROW));
    CK(cudaSetDevice(0)); CK(cudaDeviceEnablePeerAccess(1, 0));
  }
  const int max_cta = 108, per_cta = 2048;  // rows per CTA (2.36 MB)
  std::vector<int> h_ids(max_cta * per_cta);
  srand(1);
  for (auto& x : h_ids) x = (int)(((long)rand() * 7919) % far_rows);
  int* ids; uint4* out;
  CK(cudaMalloc(&ids, h_ids.size() * 4)); CK(cudaMemcpy(ids, h_ids.data(), h_ids.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&out, (long)h_ids.size() * ROW));
  cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  printf("tier=%s rows/CTA=%d row=%dB\n", host ? "host" : "peer", per_cta, ROW);
  printf("ctas stages rows/stage inflightKB write | GB/s total  GB/s per CTA\n");
  auto run = [&](auto kernel, int S, int R, int ctas, bool write, int THREADS = 256) {
    const int smem = S * R * ROW;
    CK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<ctas, THREADS, smem>>>((const uint4*)far, ids, out, per_cta, R, write);
    CK(cudaDeviceSynchronize());
    std::vector<float> t;
    for (int i = 0; i < 5; ++i) {
      CK(cudaEventRecord(a)); kernel<<<ctas, THREADS, smem>>>((const uint4*)far, ids, out, per_cta, R, write);
      CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms, a, b)); t.push_back(ms);
    }
    std::sort(t.begin(), t.end());
    const double gbs = (double)ctas * per_cta * ROW / (t[2] * 1e-3) / 1e9;
    printf("%4d %6d %9d %10.0f %5d thr=%4d | %9.1f %12.2f\n", ctas, S, R, S * R * ROW / 1024.0, write, THREADS, gbs, gbs / ctas);
  };
  // Threads per CTA, one CTA per SM (smem pinned high so two never share an SM).
  for (int thr : {128, 256, 512, 1024}) for (int ctas : {1, 8, 16}) run(fetch<4>, 4, 32, ctas, false, thr);
  // Several small fetch CTAs per SM: 2 x 16 rows = 36 KB each, so up to 4-5 fit one SM.
  printf("-- small CTAs (36 KB smem), ctas = SMs x per-SM residency\n");
  for (int ctas : {16, 32, 64, 128}) run(fetch<2>, 2, 16, ctas, false, 256);
  return 0;
}
