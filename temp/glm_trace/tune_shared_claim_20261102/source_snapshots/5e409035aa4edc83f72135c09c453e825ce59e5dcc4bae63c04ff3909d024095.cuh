// Experiment-only emulation of a slower far link (bandwidth and access latency) for the far-KV
// MLA kernels, compiled in only with -DFLASHINFER_FAR_EMU; otherwise every hook below is empty.
//
// Model: every far read reserves its bytes on one virtual link clock shared by the whole GPU,
// start = max(link clock, now), link clock = start + bytes / bandwidth, and the data may be used
// at start + bytes / bandwidth + latency. The real reads are still issued at once (the real link
// is faster than the emulated one); consumers wait for the emulated time before using a tile, so
// requests stay in flight together as on a real link with that latency.
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
  const unsigned long long cost = bytes * config.ps_per_byte / 1000;
  // Busy link (clock ahead of now): one add queues the bytes exactly. An idle link must not bank
  // credit for the time it was idle, so its clock is moved up to now (a CAS loop here capped the
  // emulated link at ~50 GB/s from contention).
  unsigned long long end = atomicAdd(&link_clock, cost) + cost;
  if (end < t + cost) {
    atomicMax(&link_clock, t + cost);
    end = t + cost;
  }
  return end + config.latency_ns;
}

__device__ __forceinline__ void wait_until(unsigned long long t) {
  const unsigned long long limit = now_ns() + 1000000000ull;  // a stale time must not hang
  if (t > limit) t = limit;
  while (now_ns() < t) __nanosleep(64);
}

// Per CTA and pipeline stage: far bytes of the tile being issued (summed with reductions, which
// return nothing and so do not stall the load path) and its issue time (stored by the CTA's first
// thread). The thread that waits for the stage reserves the bytes on the link as of the issue
// time. A returning atomic per loading warp cost ~25% with an unlimited link.
constexpr uint32_t kSlots = 8192, kMaxStages = 4;
__device__ unsigned long long pending_bytes[kSlots][kMaxStages];
__device__ unsigned long long pending_issue[kSlots][kMaxStages];

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
  if (threadIdx.x % 32 == 0 && rows) {
    atomicAdd(&pending_bytes[cta_slot()][stage], static_cast<unsigned long long>(rows) * row_bytes);
  }
  // The tile's issue time, from the CTA's first thread (a loader in every caller).
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) pending_issue[cta_slot()][stage] = now_ns();
}

// The waiting thread's copy of one stage's counters, read right after a barrier (see prefetch):
// a global load right before each tile's barrier cost ~25% of the native kernel with an unlimited
// link, even with no far bytes at all.
struct State {
  unsigned long long bytes, issue;
  int stage;
};

__device__ __forceinline__ bool waiter() { return threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0; }

template <uint32_t STAGES>
__device__ __forceinline__ void consume(uint32_t stage, unsigned long long bytes, unsigned long long issue) {
  if (bytes) {
    wait_until(reserve(bytes, issue));
    pending_bytes[cta_slot()][stage] = 0;
  }
}

// One thread waits for a stage's tile, before the barrier that releases the CTA to use it.
template <uint32_t STAGES>
__device__ __forceinline__ void wait_stage(uint32_t stage, State& st) {
  if (!config.enabled || !waiter()) return;
  const uint32_t c = cta_slot();
  if (st.stage == static_cast<int>(stage)) {
    consume<STAGES>(stage, st.bytes, st.issue);
  } else {
    consume<STAGES>(stage, *(volatile unsigned long long*)&pending_bytes[c][stage],
                    *(volatile unsigned long long*)&pending_issue[c][stage]);
  }
  st.stage = -1;
}

// Right after a tile's barrier: every warp has issued the next tile (stage `stage`), so its
// counters are final; read them now, their latency hidden behind this tile's compute.
template <uint32_t STAGES>
__device__ __forceinline__ void prefetch(uint32_t stage, State& st) {
  if (!config.enabled || !waiter()) return;
  const uint32_t c = cta_slot();
  st.bytes = *(volatile unsigned long long*)&pending_bytes[c][stage];
  st.issue = *(volatile unsigned long long*)&pending_issue[c][stage];
  st.stage = static_cast<int>(stage);
}

template <uint32_t STAGES>
__device__ __forceinline__ void wait_all() {
  State none{0, 0, -1};
#pragma unroll
  for (uint32_t s = 0; s < STAGES; ++s) wait_stage<STAGES>(s, none);
}

template <uint32_t STAGES>
__device__ __forceinline__ void init() {
  if (config.enabled && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
#pragma unroll
    for (uint32_t s = 0; s < STAGES; ++s) {
      pending_bytes[cta_slot()][s] = 0;
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
  ::flashinfer::far_emu::State far_emu_state{0, 0, -1}; \
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
