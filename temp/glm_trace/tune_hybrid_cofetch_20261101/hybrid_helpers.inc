// Workers retain two attention stages and use a separate 2x4-row service buffer.
// Dedicated CTAs use the original dynamic region as 2x64-row service buffers,
// drain their queue, then join attention. A shared global counter assigns disjoint
// 16-row jobs (one per worker claim, four per dedicated claim).
template <typename KTraits>
struct HostBatchCoFetchStorage {
  using DType = typename KTraits::DTypeKV;
  static constexpr uint32_t ROW = KTraits::HEAD_DIM_CKV + KTraits::HEAD_DIM_KPE;
  alignas(16) DType payload[2][4 * ROW];
  typename KTraits::IdType pages[2][64];
  int32_t victims[64];
  uint4* dedicated_memory;
  uint32_t count, epoch, sequence, job_base, producer_part;
  uint32_t begin[2], rows[2];
  bool dedicated, active[2];
};

template <typename KTraits>
__device__ __forceinline__ HostBatchCoFetchStorage<KTraits>& host_batch_cofetch_storage() {
  __shared__ HostBatchCoFetchStorage<KTraits> service;
  return service;
}

template <typename KTraits, typename Params, uint32_t MAX_ROWS>
__device__ __forceinline__ void issue_host_batch_hybrid_payload(const Params& params,
                                                               uint32_t buf) {
  auto& service = host_batch_cofetch_storage<KTraits>();
  constexpr uint32_t ROW = HostBatchCoFetchStorage<KTraits>::ROW;
  constexpr uint32_t VECS = ROW * sizeof(typename KTraits::DTypeKV) / 16;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t begin = service.begin[buf], rows = service.rows[buf];
  if (tid < rows) service.pages[buf][tid] = params.unique_pages[begin + tid];
  __syncthreads();
  auto* buffer = MAX_ROWS == 64 ? service.dedicated_memory + buf * 64 * VECS
                               : reinterpret_cast<uint4*>(service.payload[buf]);
#pragma unroll
  for (uint32_t i = 0; i < ceil_div(MAX_ROWS * VECS, KTraits::NUM_THREADS); ++i) {
    const uint32_t v = i * KTraits::NUM_THREADS + tid;
    if (v < rows * VECS) {
      const auto page = service.pages[buf][v / VECS];
      const auto* source = reinterpret_cast<const uint4*>(params.ckv + int64_t(page) * ROW);
      cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffer + v, source + v % VECS);
    }
  }
  FAR_EMU_TILE(2, buf, tid == 0 ? rows : 0, ROW * sizeof(typename KTraits::DTypeKV));
  cp_async::commit_group();
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void issue_host_batch_cofetch(const Params& params, uint32_t buf) {
  auto& service = host_batch_cofetch_storage<KTraits>();
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  if (tid == 0) {
    uint32_t begin;
    if (service.dedicated) {
      begin = atomicAdd(params.state + kHostBatchGaps, 4u) * 16;
    } else {
      if (service.producer_part == 4) {
        service.job_base = atomicAdd(params.state + kHostBatchGaps, 1u) * 16;
        service.producer_part = 0;
      }
      begin = service.job_base + service.producer_part++ * 4;
    }
    service.begin[buf] = begin;
    service.active[buf] = begin < service.count;
    service.rows[buf] = begin < service.count
        ? min(service.dedicated ? 64u : 4u, service.count - begin) : 0;
  }
  __syncthreads();
  if (!service.active[buf]) return;
  if (service.dedicated) issue_host_batch_hybrid_payload<KTraits, Params, 64>(params, buf);
  else issue_host_batch_hybrid_payload<KTraits, Params, 4>(params, buf);
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void init_host_batch_cofetch(const Params& params,
                                                       uint32_t unique, uint32_t epoch,
                                                       uint8_t* smem) {
  auto& service = host_batch_cofetch_storage<KTraits>();
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    service.count = min(unique, params.staging_capacity);
    service.epoch = epoch;
    service.sequence = 0;
    service.producer_part = 4;
    service.job_base = service.count;
    service.dedicated = blockIdx.y >= gridDim.y - params.fetch_ctas;
    service.dedicated_memory = reinterpret_cast<uint4*>(smem);
  }
  FAR_EMU_INIT(2);
  __syncthreads();
  issue_host_batch_cofetch<KTraits>(params, 0);
  issue_host_batch_cofetch<KTraits>(params, 1);
}

template <typename KTraits, typename Params, uint32_t MAX_ROWS>
__device__ __forceinline__ void finish_host_batch_hybrid_payload(const Params& params,
                                                               uint32_t buf) {
  auto& service = host_batch_cofetch_storage<KTraits>();
  constexpr uint32_t ROW = HostBatchCoFetchStorage<KTraits>::ROW;
  constexpr uint32_t VECS = ROW * sizeof(typename KTraits::DTypeKV) / 16;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t begin = service.begin[buf], rows = service.rows[buf];
  const auto* buffer = MAX_ROWS == 64 ? service.dedicated_memory + buf * 64 * VECS
                                     : reinterpret_cast<uint4*>(service.payload[buf]);
  auto* destination = reinterpret_cast<uint4*>(
      params.staging + (params.cache_slots + int64_t(begin)) * ROW);
#pragma unroll
  for (uint32_t i = 0; i < ceil_div(MAX_ROWS * VECS, KTraits::NUM_THREADS); ++i) {
    const uint32_t v = i * KTraits::NUM_THREADS + tid;
    if (v < rows * VECS) destination[v] = buffer[v];
  }
  if (tid < rows) params.host_slot[service.pages[buf][tid] - params.host_page_start] = 0;
  __threadfence();
  __syncthreads();
  // A worker publishes only after all four subchunks of its 16-row job landed.
  // Dedicated claims start on 16-row boundaries and publish every covered flag.
  if (MAX_ROWS == 64 || begin % 16 == 12 || begin + rows == service.count) {
    if (tid < (MAX_ROWS == 64 ? ceil_div(rows, 16u) : 1u)) {
      uint32_t* flag = host_batch_flags(params) + begin / 16 + tid;
      asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(flag), "r"(service.epoch + 1)
                   : "memory");
    }
  }
  if (tid < MAX_ROWS) {
    const int32_t victim = tid < rows ? params.victim_of[begin + tid] : -1;
    service.victims[tid] = victim;
    if (victim >= 0) {
      const auto page = service.pages[buf][tid];
      const int32_t evicted = params.slot_page[victim];
      if (evicted >= 0) params.resident[evicted - params.host_page_start] = -1;
      params.resident[page - params.host_page_start] = victim;
      params.slot_page[victim] = page;
    }
  }
  __syncthreads();
#pragma unroll
  for (uint32_t i = 0; i < ceil_div(MAX_ROWS * VECS, KTraits::NUM_THREADS); ++i) {
    const uint32_t v = i * KTraits::NUM_THREADS + tid;
    const int32_t victim = v < rows * VECS ? service.victims[v / VECS] : -1;
    if (victim >= 0)
      reinterpret_cast<uint4*>(params.staging + int64_t(victim) * ROW)[v % VECS] = buffer[v];
  }
  __syncthreads();
}

template <typename KTraits, typename Params>
__device__ __forceinline__ bool step_host_batch_cofetch(const Params& params) {
  auto& service = host_batch_cofetch_storage<KTraits>();
  const uint32_t buf = service.sequence % 2;
  if (!service.active[buf]) return false;
  cp_async::wait_group<0>();
#ifdef FLASHINFER_FAR_EMU
  ::flashinfer::far_emu::State far_emu_state{0, 0, -1};
#endif
  FAR_EMU_WAIT(2, buf);
  __syncthreads();
  if (service.dedicated) finish_host_batch_hybrid_payload<KTraits, Params, 64>(params, buf);
  else finish_host_batch_hybrid_payload<KTraits, Params, 4>(params, buf);
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) ++service.sequence;
  __syncthreads();
  issue_host_batch_cofetch<KTraits>(params, buf);
  return true;
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void finish_host_batch_cofetch(const Params& params) {
#pragma unroll 1
  while (step_host_batch_cofetch<KTraits>(params)) {}
  cp_async::wait_group<0>();
  __syncthreads();
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void wait_host_batch_tile_cofetch(
    const Params& params, const uint32_t* scratch, uint32_t index_base, uint32_t kv_len,
    uint32_t kv_base) {
  const uint32_t lane = threadIdx.x;
#pragma unroll 1
  while (true) {
    bool ready = true;
    if (threadIdx.y == 0 && threadIdx.z == 0 && kv_base + lane < kv_len) {
      const auto page = params.remap[index_base + kv_base + lane];
      const uint32_t row = static_cast<uint32_t>(-(page + 1));
      if (page < 0 && row >= params.cache_slots)
        ready = host_batch_ld_acquire(host_batch_flags(params) +
            (row - params.cache_slots) / 16) == scratch[kHostBatchScratchEpoch] + 1;
    }
    if (__syncthreads_and(ready)) return;
    if (!step_host_batch_cofetch<KTraits>(params)) __nanosleep(64);
  }
}
