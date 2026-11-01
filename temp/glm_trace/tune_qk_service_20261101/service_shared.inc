// Three shared service stages; seven rows is the largest equal packing at the measured limit.
template <typename Id>
struct QKSMetadata {
  Id pages[3][7];
  int32_t victims[7];
  uint32_t tile_ready;
};
struct QKSContext { uint32_t job = 0; };
__device__ __forceinline__ uint32_t qks_service_base(uint32_t job) {
  return (blockIdx.y + job / 10 * gridDim.y) * 64 + (job % 10) * 7;
}
__device__ __forceinline__ uint32_t qks_service_rows(uint32_t job, uint32_t count) {
  return min(7u, min(64u - (job % 10) * 7, count - qks_service_base(job)));
}

template <uint32_t STAGE, typename KTraits, typename Params>
__device__ __forceinline__ void qks_service_issue(
    const Params& params, typename KTraits::SharedStorage& storage,
    QKSMetadata<typename Params::IdType>& metadata, QKSContext& ctx,
    uint32_t job, uint32_t count) {
  const uint32_t tid = threadIdx.y * 32 + threadIdx.x, first = qks_service_base(job);
  const uint32_t rows = qks_service_rows(job, count);
  if (tid < rows) metadata.pages[STAGE][tid] = params.unique_pages[first + tid];
  qks_producer_sync();
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t v = i * 128 + tid;
    if (v < rows * 72) {
      const auto page = metadata.pages[STAGE][v / 72];
      const auto* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
      cp_async::load_128b<PrefetchMode::kNoPrefetch>(storage.service_payload[STAGE] + v, source + v % 72);
    }
  }
  FAR_EMU_TILE(3, STAGE, tid == 0 ? rows : 0, 1152);
  cp_async::commit_group();qks_producer_sync();
}

template <uint32_t STAGE, typename KTraits, typename Params>
__device__ __forceinline__ void qks_service_consume(
    const Params& params, typename KTraits::SharedStorage& storage,
    QKSMetadata<typename Params::IdType>& metadata, QKSContext& ctx,
    uint32_t epoch, uint32_t count) {
  const uint32_t tid = threadIdx.y * 32 + threadIdx.x, first = qks_service_base(ctx.job);
  const uint32_t rows = qks_service_rows(ctx.job, count);
  cp_async::wait_group<0>();
#ifdef FLASHINFER_FAR_EMU
  ::flashinfer::far_emu::State far_emu_state{0, 0, -1};
#endif
  FAR_EMU_WAIT(3, STAGE);qks_producer_sync();
  auto* destination = reinterpret_cast<uint4*>(params.staging + (params.cache_slots + static_cast<int64_t>(first)) * 576);
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t v = i * 128 + tid;
    if (v < rows * 72) destination[v] = storage.service_payload[STAGE][v];
  }
  if (tid < rows) params.host_slot[metadata.pages[STAGE][tid] - params.host_page_start] = 0;
  if (ctx.job % 10 == 9 || first + rows >= count) {
    __threadfence();qks_producer_sync();
    if (tid == 0) asm volatile("st.release.gpu.global.u32 [%0], %1;" ::
        "l"(host_batch_flags(params) + first / 64), "r"(epoch + 1) : "memory");
  }
  if (tid < 7) {
    int32_t victim = -1;
    if (tid < rows) {
      victim = params.victim_of[first + tid];bind_host_batch_staged_slot(params, first + tid, victim);
    }
    metadata.victims[tid] = victim;
  }
  qks_producer_sync();
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t v = i * 128 + tid;
    if (v < rows * 72) {
      const int32_t victim = metadata.victims[v / 72];
      if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % 72] = storage.service_payload[STAGE][v];
    }
  }
  qks_producer_sync();
  if (qks_service_base(ctx.job + 3) < count)
    qks_service_issue<STAGE, KTraits>(params, storage, metadata, ctx, ctx.job + 3, count);
}

template <typename KTraits, typename Params>
__device__ __forceinline__ bool qks_service_step(
    const Params& params, typename KTraits::SharedStorage& storage,
    QKSMetadata<typename Params::IdType>& metadata, QKSContext& ctx, uint32_t epoch, uint32_t count) {
  if (qks_service_base(ctx.job) >= count) return false;
  const uint32_t stage = ctx.job % 3;
  if (stage == 0) qks_service_consume<0, KTraits>(params, storage, metadata, ctx, epoch, count);
  else if (stage == 1) qks_service_consume<1, KTraits>(params, storage, metadata, ctx, epoch, count);
  else qks_service_consume<2, KTraits>(params, storage, metadata, ctx, epoch, count);
  ++ctx.job;return true;
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void qks_wait_tile(
    const Params& params, typename KTraits::SharedStorage& storage,
    QKSMetadata<typename Params::IdType>& metadata, QKSContext& ctx,
    const uint32_t* scratch, uint32_t index_base, uint32_t length, uint32_t base,
    uint32_t epoch, uint32_t count) {
#pragma unroll 1
  while (true) {
    if (threadIdx.y == 0) {
      const uint32_t k = base + threadIdx.x;
      const auto page = k < length ? params.remap[index_base + k] : 0;
      const uint32_t row = static_cast<uint32_t>(-(page + 1));
      bool ready = true;
      if (page < 0 && row >= params.cache_slots)
        ready = host_batch_ld_acquire(host_batch_flags(params) + (row - params.cache_slots) / 64) == epoch + 1;
      const bool all_ready = __all_sync(0xffffffffu, ready);
      if (threadIdx.x == 0) metadata.tile_ready = all_ready;
    }
    qks_producer_sync();
    const bool tile_ready = metadata.tile_ready;qks_producer_sync();
    if (tile_ready) break;
    if (!qks_service_step<KTraits>(params, storage, metadata, ctx, epoch, count)) __nanosleep(64);
  }
}
