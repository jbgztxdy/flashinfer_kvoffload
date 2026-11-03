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


// Isolated model candidate. Only already-issued requests enter these mailboxes.
// A winning requester warp combines all currently published mailboxes; there is
// no dedicated CTA/SM, no future byte reservation and no fixed perCTA bandwidth.
constexpr uint32_t kBrokerCTAs=8192,kBrokerWarps=8;
struct BrokerRequest {unsigned long long issue,cost,deadline,record;};
__device__ BrokerRequest broker_requests[kBrokerCTAs*kBrokerWarps];
__device__ uint32_t broker_masks[kBrokerCTAs],broker_lock;
struct BrokerRecord {unsigned long long issue,cost,prior,end,consume,queue;};
constexpr uint32_t kBrokerRecords=100000;
__device__ BrokerRecord broker_records[kBrokerRecords];
__device__ unsigned long long broker_count;
__device__ uint32_t broker_trace;
__device__ __forceinline__ uint32_t broker_acquire(const uint32_t* p){uint32_t v;asm volatile("ld.acquire.gpu.global.u32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;}
__device__ __forceinline__ unsigned long long broker_acquire64(const unsigned long long* p){unsigned long long v;asm volatile("ld.acquire.gpu.global.u64 %0,[%1];":"=l"(v):"l"(p):"memory");return v;}
__device__ __forceinline__ void broker_release(uint32_t* p,uint32_t v){asm volatile("st.release.gpu.global.u32 [%0],%1;"::"l"(p),"r"(v):"memory");}
__device__ __forceinline__ void broker_release64(unsigned long long* p,unsigned long long v){asm volatile("st.release.gpu.global.u64 [%0],%1;"::"l"(p),"l"(v):"memory");}
__device__ __forceinline__ unsigned long long broker_shfl(unsigned long long v,int lane){return __shfl_sync(0xffffffffu,v,lane);}
__device__ __forceinline__ void broker_combine(){
 const uint32_t lane=threadIdx.x%32,ctas=gridDim.x*gridDim.y*gridDim.z;
 unsigned long long clock=lane==0?link_clock:0;clock=broker_shfl(clock,0);
 unsigned long long record=lane==0?broker_count:0;record=broker_shfl(record,0);
 for(uint32_t base=0;base<ctas;base+=32){
  uint32_t c=base+lane,mask=c<ctas?broker_acquire(broker_masks+c):0;
  uint32_t any=mask;
#pragma unroll
  for(uint32_t d=16;d>0;d/=2)any|=__shfl_xor_sync(0xffffffffu,any,d);
  while(any){
   uint32_t w=__ffs(any)-1;any&=any-1;bool active=(mask>>w)&1u;
   auto* q=broker_requests+c*kBrokerWarps+w;
   unsigned long long cost=active?q->cost:0,issue=active?q->issue:0,prefix=cost;
#pragma unroll
   for(uint32_t d=1;d<32;d*=2){auto a=__shfl_up_sync(0xffffffffu,prefix,d);if(lane>=d)prefix+=a;}
   // Prefix max-plus scan: end_i=prefixCost_i+max(clock,max_j<=i(issue_j-prefixCostBefore_j)).
   long long anchor=active?static_cast<long long>(issue)-static_cast<long long>(prefix-cost):(-0x7fffffffffffffffll);
   anchor=max(anchor,static_cast<long long>(clock));
#pragma unroll
   for(uint32_t d=1;d<32;d*=2){auto a=__shfl_up_sync(0xffffffffu,anchor,d);if(lane>=d)anchor=max(anchor,a);}
   unsigned long long end=static_cast<unsigned long long>(anchor)+prefix;
   const uint32_t votes=__ballot_sync(0xffffffffu,active),below=__popc(votes&((1u<<lane)-1));
   const unsigned long long id=record+below;
   if(active){
    // Remove this request before publishing its answer. The producer may reuse
    // its mailbox only after acquiring that answer, so the next bit cannot be cleared here.
    atomicAnd(broker_masks+c,~(1u<<w));
    if(broker_trace&&id<kBrokerRecords)broker_records[id]={issue,cost,end-cost,end,0,c*kBrokerWarps+w};
    q->record=id;
    broker_release64(&q->deadline,end+config.latency_ns);
   }
   clock=broker_shfl(end,31);record+=__popc(votes);
  }
 }
 if(lane==0){link_clock=clock;broker_count=record;broker_release(&broker_lock,0);}
}
// Every lane in the loading warp calls this function. The request is published
// after real copies have been issued, and receives exactly its own service cost.
__device__ __forceinline__ unsigned long long reserve_warp(unsigned long long bytes){
 if(!bytes)return 0;
 const uint32_t lane=threadIdx.x%32,t=(threadIdx.z*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
 const uint32_t c=blockIdx.x+gridDim.x*(blockIdx.y+gridDim.y*blockIdx.z),w=t/32;
 auto* q=broker_requests+c*kBrokerWarps+w;
 if(lane==0){
  q->deadline=0;q->issue=now_ns();q->cost=(bytes*config.ps_per_byte+999)/1000;
  __threadfence();atomicOr(broker_masks+c,1u<<w);
 }
 __syncwarp();
 unsigned long long answer=0;
 while(true){
  if(lane==0)answer=broker_acquire64(&q->deadline);answer=broker_shfl(answer,0);
  if(answer)return answer;
  uint32_t won=lane==0?atomicCAS(&broker_lock,0u,1u):0;won=__shfl_sync(0xffffffffu,won,0);
  if(won==0)broker_combine();else __nanosleep(64);
 }
}
// Per CTA/stage, the latest deadline of its warp-issued packets. Reserve immediately, not at
// consume time: retrospective reservations can repay an idle gap with a burst of early data.
// Warp granularity avoids a block barrier in loaders where only some warps load (e.g. KV16).
constexpr uint32_t kSlots = 8192, kMaxStages = 4;
__device__ unsigned long long pending_ready[kSlots][kMaxStages];

__device__ __forceinline__ uint32_t cta_slot() {
  return (blockIdx.x + gridDim.x * (blockIdx.y + gridDim.y * blockIdx.z)) % kSlots;
}

// Called by every thread of a loading warp after it issued its rows of a stage's tile:
// `far_rows` is the number of far rows this thread speaks for.
template <uint32_t STAGES>
__device__ __forceinline__ void tile(uint32_t stage, uint32_t far_rows, uint32_t row_bytes) {
  static_assert(STAGES <= kMaxStages);
  if (!config.enabled) return;
  const uint32_t rows = __reduce_add_sync(0xffffffffu, far_rows);
  if(rows){
    unsigned long long deadline;
    if(config.ps_per_byte==0){deadline=now_ns()+config.latency_ns;deadline=broker_shfl(deadline,0);}
    else deadline=reserve_warp(static_cast<unsigned long long>(rows)*row_bytes);
    if(threadIdx.x%32==0)atomicMax(&pending_ready[cta_slot()][stage],deadline);
  }
}

// Retain aggregate layout for existing callers that explicitly initialize State{0,0,-1}.
// V2 no longer reads these deferred-accounting fields.
struct State {
  unsigned long long bytes = 0, issue = 0;
  int stage = -1;
};

__device__ __forceinline__ bool waiter() { return threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0; }

// All CTA threads participate. The pre-wait barrier is necessary even after per-thread
// cp.async.wait_group: another warp might not yet have published its packet's deadline.
// Callers retain their post-wait barrier before consuming or reusing the stage.
template <uint32_t STAGES>
__device__ __forceinline__ void wait_stage(uint32_t stage, State&) {
  if (!config.enabled) return;
  __syncthreads();
  if (waiter()) {
    const uint32_t c = cta_slot();
    wait_until(*(volatile unsigned long long*)&pending_ready[c][stage]);
    pending_ready[c][stage] = 0;
  }
}

// Compatibility hook: deadlines are already reserved at issue; no deferred byte snapshot.
template <uint32_t STAGES>
__device__ __forceinline__ void prefetch(uint32_t, State&) {}

template <uint32_t STAGES>
__device__ __forceinline__ void wait_all() {
  State none{};
#pragma unroll
  for (uint32_t s = 0; s < STAGES; ++s) wait_stage<STAGES>(s, none);
}

template <uint32_t STAGES>
__device__ __forceinline__ void init() {
  if (config.enabled && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
#pragma unroll
    for (uint32_t s = 0; s < STAGES; ++s) {
      pending_ready[cta_slot()][s] = 0;
    }
  }
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
