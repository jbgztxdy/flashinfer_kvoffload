// Experiment-only emulation of a slower far link (bandwidth and access latency) for the far-KV
// MLA kernels, compiled in only with -DFLASHINFER_FAR_EMU; otherwise every hook below is empty.
//
// Model v2: each far packet reserves its bytes at issue on one virtual link clock shared by
// the module/device. start = max(link clock, issue time), link clock = start + ceil(bytes /
// bandwidth), and the data may be used at link clock + latency. Reservations are linearized by
// CAS; latency does not occupy the serializer. Service time rounds UP to nanoseconds (less
// than 1ns per packet), never dropping bandwidth budget. The real reads are still issued (the real link
// is faster than the emulated one); consumers wait for the emulated time before using a tile, so
// requests stay in flight together as on a real link with that latency. This models payload
// serialization/availability, not all physical-link resources. Concurrent independent modules
// have independent clocks and must not be interpreted as sharing one physical link.
// Runtime settings come from FLASHINFER_FAR_EMU="<GB/s>:<latency ns>:<far begin>:<far end>"
// (addresses of the far tier; read by the launcher on eager launches, see configure); bandwidth 0
// means unlimited.
#pragma once

#ifdef FLASHINFER_FAR_EMU
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace flashinfer {
namespace far_emu {

struct Config {
  unsigned long long ps_per_byte;  // 1000 / (GB/s); 0 = unlimited
  unsigned long long latency_ns;
  unsigned long long far_lo, far_hi;  // far tier address range (native kernel: reads inside are far)
  int enabled;
};

__device__ Config config;
__device__ unsigned long long link_clock;

__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Time at which `bytes` of far data issued at `t` may be used.
__device__ __forceinline__ unsigned long long reserve(unsigned long long bytes, unsigned long long t) {
  if (!config.enabled || bytes == 0) return 0;
  if (config.ps_per_byte == 0) return t + config.latency_ns;
  const unsigned long long cost = (bytes * config.ps_per_byte + 999) / 1000;
  // Updating max(clock,t)+cost must be ONE atomic transaction. atomicAdd followed by
  // atomicMax loses other callers' reservations when several CTAs resume an idle link.
  unsigned long long old = atomicAdd(&link_clock, 0ull);
  while (true) {
    const unsigned long long end = (old > t ? old : t) + cost;
    const unsigned long long observed = atomicCAS(&link_clock, old, end);
    if (observed == old) return end + config.latency_ns;
    old = observed;
  }
}

__device__ __forceinline__ void wait_until(unsigned long long t) {
  // Do not truncate a valid long transfer/deadline: doing so releases data too early.
  while (now_ns() < t) __nanosleep(64);
}

// Per CTA/stage, the latest deadline of its warp-issued packets. Reserve immediately, not at
// consume time: retrospective reservations can repay an idle gap with a burst of early data.
// Warp granularity avoids a block barrier in loaders where only some warps load (e.g. KV16).
constexpr uint32_t kSlots = 8192, kMaxStages = 4;
__device__ unsigned long long pending_ready[kSlots][kMaxStages];

__device__ __forceinline__ uint32_t cta_slot() {
  return (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) % kSlots;
}

// Scenario-calibrated local deadlines, NOT a work-conserving shared-link model.
__device__ unsigned long long stage_rows[kSlots][kMaxStages][8];
__device__ unsigned long long stage_issue[kSlots][kMaxStages][8];
// ps_per_byte is interpreted by this PRIVATE model as calibrated local delay ps/B.
template <uint32_t STAGES>
__device__ __forceinline__ void tile(uint32_t stage,uint32_t far_rows,uint32_t row_bytes){
 if(!config.enabled)return;
 const uint32_t rows=__reduce_add_sync(0xffffffffu,far_rows);
 const uint32_t warp=threadIdx.y+blockDim.y*threadIdx.z;
 if(threadIdx.x==0){stage_rows[cta_slot()][stage][warp]=(unsigned long long)rows*row_bytes;stage_issue[cta_slot()][stage][warp]=now_ns();}
}
struct State {unsigned long long bytes=0,issue=0;int stage=-1;};
__device__ __forceinline__ bool waiter(){return threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0;}
template<uint32_t STAGES>
__device__ __forceinline__ void wait_stage(uint32_t stage,State&){
 if(!config.enabled)return;
 __syncthreads();
 if(waiter()){
  unsigned long long bytes=0,issue=0;
  for(int w=0;w<8;w++){bytes+=stage_rows[cta_slot()][stage][w];auto t=stage_issue[cta_slot()][stage][w];if(t>issue)issue=t;stage_rows[cta_slot()][stage][w]=0;}
  if(bytes)wait_until(issue+(bytes*config.ps_per_byte+999)/1000+config.latency_ns);
 }
}
template<uint32_t STAGES>__device__ __forceinline__ void prefetch(uint32_t,State&){}
template<uint32_t STAGES>__device__ __forceinline__ void wait_all(){State s{};for(uint32_t i=0;i<STAGES;i++)wait_stage<STAGES>(i,s);}
template<uint32_t STAGES>__device__ __forceinline__ void init(){
 if(config.enabled&&waiter())for(uint32_t s=0;s<STAGES;s++)for(int w=0;w<8;w++){stage_rows[cta_slot()][s][w]=0;stage_issue[cta_slot()][s][w]=0;}
}

// Host: load the settings from FLASHINFER_FAR_EMU into the device (per launch, so a benchmark
// can change them between runs). Internal linkage: several extension libraries include this
// header, and an exported inline definition was interposed across them, so one library's
// launches configured another library's `config`.
static void configure(cudaStream_t stream) {
  // Never inside a CUDA graph capture: a captured copy would replay from this function's stack.
  // The value uploaded by the last eager launch stays in effect for the replays.
  cudaStreamCaptureStatus capture;
  if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone) return;
  // Copied only when the setting changes: a per-launch copy added a few us to every launch.
  static std::string last = "unset";
  const char* env = std::getenv("FLASHINFER_FAR_EMU");
  if (last == (env ? env : "")) return;
  last = env ? env : "";
  Config c{0, 0, 0, 0, 0};
  if (const char* s = env) {
    double gbps = 0;
    unsigned long long lat = 0, lo = 0, hi = 0;
    if (std::sscanf(s, "%lf:%llu:%llu:%llu", &gbps, &lat, &lo, &hi) == 4) {
      c = Config{gbps > 0 ? static_cast<unsigned long long>(1000.0 / gbps + 0.5) : 0, lat, lo, hi, 1};
    }
  }
  if(const char* rate=std::getenv("FLASHINFER_STALL_PS_PER_BYTE"))c.ps_per_byte=std::strtoull(rate,nullptr,10);
  cudaStreamSynchronize(stream);
  cudaMemcpyToSymbol(config, &c, sizeof(c));
  if (std::getenv("FLASHINFER_FAR_EMU_VERBOSE")) {
    std::fprintf(stderr, "far_emu: %s -> enabled=%d ps/B=%llu latency=%llu ns far=[%llx, %llx)\n",
                 last.c_str(), c.enabled, c.ps_per_byte, c.latency_ns, c.far_lo, c.far_hi);
  }
}

}  // namespace far_emu
}  // namespace flashinfer

#define FAR_EMU_INIT(STAGES)                              \
  ::flashinfer::far_emu::State far_emu_state{}; \
  ::flashinfer::far_emu::init<STAGES>()
#define FAR_EMU_TILE(STAGES, stage, far_rows, row_bytes) \
  ::flashinfer::far_emu::tile<STAGES>(stage, far_rows, row_bytes)
#define FAR_EMU_WAIT(STAGES, stage) ::flashinfer::far_emu::wait_stage<STAGES>(stage, far_emu_state)
#define FAR_EMU_PREFETCH(STAGES, stage) ::flashinfer::far_emu::prefetch<STAGES>(stage, far_emu_state)
#define FAR_EMU_WAIT_ALL(STAGES) ::flashinfer::far_emu::wait_all<STAGES>()
#define FAR_EMU_CONFIGURE(stream) ::flashinfer::far_emu::configure(stream)
#define FAR_EMU_SYNC() __syncthreads()
#define FAR_EMU_IS_FAR(ptr)                                                        \
  (reinterpret_cast<unsigned long long>(ptr) >= ::flashinfer::far_emu::config.far_lo && \
   reinterpret_cast<unsigned long long>(ptr) < ::flashinfer::far_emu::config.far_hi)
#else
#define FAR_EMU_INIT(STAGES)
#define FAR_EMU_TILE(STAGES, stage, far_rows, row_bytes)
#define FAR_EMU_WAIT(STAGES, stage)
#define FAR_EMU_PREFETCH(STAGES, stage)
#define FAR_EMU_WAIT_ALL(STAGES)
#define FAR_EMU_CONFIGURE(stream)
#define FAR_EMU_SYNC()
#endif
