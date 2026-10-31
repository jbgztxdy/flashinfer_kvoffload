// A CTA keeps its attention state while advancing a disjoint cyclic fetch queue.
// The one-stage attention storage and two 16-row service buffers together use the
// same payload bytes as the original two-stage attention allocation.
template <typename KTraits>
struct HostBatchCoFetchStorage {
  using DType = typename KTraits::DTypeKV;
  static constexpr uint32_t ROW = KTraits::HEAD_DIM_CKV + KTraits::HEAD_DIM_KPE;
  alignas(16) DType payload[2][16 * ROW];
  typename KTraits::IdType pages[2][16];
  int32_t victims[16];
  uint32_t count, epoch, sequence;
};

template <typename KTraits>
__device__ __forceinline__ HostBatchCoFetchStorage<KTraits>& host_batch_cofetch_storage() {
  __shared__ HostBatchCoFetchStorage<KTraits> service;
  return service;
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void issue_host_batch_cofetch(const Params& params, uint32_t sequence) {
  auto& service = host_batch_cofetch_storage<KTraits>();
  constexpr uint32_t ROW = HostBatchCoFetchStorage<KTraits>::ROW;
  constexpr uint32_t VECS = ROW * sizeof(typename KTraits::DTypeKV) / 16;
  constexpr uint32_t CHUNK_VECS = 16 * VECS;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t chunk = blockIdx.y + sequence * gridDim.y, buf = sequence % 2;
  if (chunk * 16 >= service.count) return;
  if (tid < 16 && chunk * 16 + tid < service.count)
    service.pages[buf][tid] = params.unique_pages[chunk * 16 + tid];
  __syncthreads();
  auto* destination = reinterpret_cast<uint4*>(service.payload[buf]);
#pragma unroll
  for (uint32_t i = 0; i < ceil_div(CHUNK_VECS, KTraits::NUM_THREADS); ++i) {
    const uint32_t v = i * KTraits::NUM_THREADS + tid;
    if (v < CHUNK_VECS && chunk * 16 + v / VECS < service.count) {
      const auto page = service.pages[buf][v / VECS];
      const auto* source = reinterpret_cast<const uint4*>(params.ckv + int64_t(page) * ROW);
      cp_async::load_128b<PrefetchMode::kNoPrefetch>(destination + v, source + v % VECS);
    }
  }
  FAR_EMU_TILE(2, buf, tid == 0 ? min(16u, service.count - chunk * 16) : 0,
               ROW * sizeof(typename KTraits::DTypeKV));
  cp_async::commit_group();
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void init_host_batch_cofetch(const Params& params,
                                                       uint32_t unique, uint32_t epoch) {
  auto& service = host_batch_cofetch_storage<KTraits>();
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    service.count = min(unique, params.staging_capacity);
    service.epoch = epoch;
    service.sequence = 0;
  }
  FAR_EMU_INIT(2);
  __syncthreads();
  issue_host_batch_cofetch<KTraits>(params, 0);
  issue_host_batch_cofetch<KTraits>(params, 1);
}

// Every call either completes one chunk, including its cache payload, or reports
// that this CTA's queue is exhausted. The same emulation hooks and byte accounting
// as the dedicated fetcher are used; no bandwidth/latency model is changed.
template <typename KTraits, typename Params>
__device__ __forceinline__ bool step_host_batch_cofetch(const Params& params) {
  auto& service = host_batch_cofetch_storage<KTraits>();
  constexpr uint32_t ROW = HostBatchCoFetchStorage<KTraits>::ROW;
  constexpr uint32_t VECS = ROW * sizeof(typename KTraits::DTypeKV) / 16;
  constexpr uint32_t CHUNK_VECS = 16 * VECS;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t sequence = service.sequence;
  const uint32_t chunk = blockIdx.y + sequence * gridDim.y, buf = sequence % 2;
  if (chunk * 16 >= service.count) return false;
  // Attention and service have disjoint payload storage but share the cp.async
  // FIFO. Draining it also covers any attention copies already issued.
  cp_async::wait_group<0>();
#ifdef FLASHINFER_FAR_EMU
  ::flashinfer::far_emu::State far_emu_state{0, 0, -1};
#endif
  FAR_EMU_WAIT(2, buf);
  __syncthreads();
  const auto* source = reinterpret_cast<const uint4*>(service.payload[buf]);
  auto* destination = reinterpret_cast<uint4*>(
      params.staging + (params.cache_slots + int64_t(chunk) * 16) * ROW);
#pragma unroll
  for (uint32_t i = 0; i < ceil_div(CHUNK_VECS, KTraits::NUM_THREADS); ++i) {
    const uint32_t v = i * KTraits::NUM_THREADS + tid;
    if (v < CHUNK_VECS && chunk * 16 + v / VECS < service.count) destination[v] = source[v];
  }
  if (tid < 16 && chunk * 16 + tid < service.count) {
    const auto page = service.pages[buf][tid];
    params.host_slot[page - params.host_page_start] = 0;
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    uint32_t* flag = host_batch_flags(params) + chunk;
    asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(flag), "r"(service.epoch + 1)
                 : "memory");
  }
  // All pool selections completed before service initialization. Each selected
  // victim belongs to one unique slot, so its payload and binding have one writer.
  if (tid < 16) {
    const uint32_t slot = chunk * 16 + tid;
    const int32_t victim = slot < service.count ? params.victim_of[slot] : -1;
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
  for (uint32_t i = 0; i < ceil_div(CHUNK_VECS, KTraits::NUM_THREADS); ++i) {
    const uint32_t v = i * KTraits::NUM_THREADS + tid;
    const int32_t victim = v < CHUNK_VECS ? service.victims[v / VECS] : -1;
    if (victim >= 0)
      reinterpret_cast<uint4*>(params.staging + int64_t(victim) * ROW)[v % VECS] = source[v];
  }
  __syncthreads();
  if (tid == 0) service.sequence = sequence + 1;
  __syncthreads();
  issue_host_batch_cofetch<KTraits>(params, sequence + 2);
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
    // Waiting never prevents this CTA from completing the chunks other CTAs need.
    if (!step_host_batch_cofetch<KTraits>(params)) __nanosleep(64);
  }
}
