// One independent service warp; two cp.async/shared stages and one register stage.
// Ready flags retain the original 64-row granularity. Every subjob has eight rows.
template <typename KTraits, typename Params>
__device__ __forceinline__ void nine_service(const Params& params, uint8_t* auxiliary,
                                            uint32_t epoch, uint32_t unique) {
  constexpr uint32_t VECS = 72, SUB = 8, WORDS = SUB * VECS / 32;
  static_assert(WORDS == 18);
  const uint32_t lane = threadIdx.x;
  const uint32_t count = min(unique, params.staging_capacity);
  uint4* buffers = reinterpret_cast<uint4*>(auxiliary);
  __shared__ typename Params::IdType nine_pages[3][SUB];
  __shared__ int32_t nine_victims[SUB];
  __shared__ uint32_t nine_deferred[kHostBatchMaxDeferred], nine_deferred_count;
  if (lane == 0) nine_deferred_count = 0;
  __syncwarp();
  uint4 register_stage[WORDS];
  FAR_EMU_INIT(3);
  __syncwarp();
  const auto base = [&](uint32_t job) {
    return (blockIdx.y + job / 8 * gridDim.y) * 64 + job % 8 * SUB;
  };
  auto issue = [&](uint32_t job, uint32_t stage) {
    const uint32_t first = base(job);
    if (lane < SUB && first + lane < count) nine_pages[stage][lane] = params.unique_pages[first + lane];
    __syncwarp();
#pragma unroll
    for (uint32_t i = 0; i < WORDS; ++i) {
      const uint32_t v = i * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[stage][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (stage == 2) register_stage[i] = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + stage * SUB * VECS + v,
                                                          source + v % VECS);
      }
    }
    // Same unmodified model/hook after actual reads, including the synchronous register stage.
    FAR_EMU_TILE(3, stage, lane == 0 ? min(SUB, count - first) : 0, 1152);
    if (stage != 2) cp_async::commit_group();
    __syncwarp();
  };
#pragma unroll
  for (uint32_t job = 0; job < 3; ++job) if (base(job) < count) issue(job, job);
  bool chunk_deferred = false;
#pragma unroll 1
  for (uint32_t job = 0; base(job) < count; ++job) {
    const uint32_t first = base(job), stage = job % 3, chunk = first / 64;
    if (stage != 2) cp_async::wait_group<0>();
    FAR_EMU_WAIT(3, stage);
    __syncwarp();
    auto* destination = reinterpret_cast<uint4*>(params.staging +
        (params.cache_slots + static_cast<int64_t>(first)) * 576);
#pragma unroll
    for (uint32_t i = 0; i < WORDS; ++i) {
      const uint32_t v = i * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = stage == 2 ? register_stage[i] : buffers[stage * SUB * VECS + v];
    }
    if (lane < SUB && first + lane < count)
      params.host_slot[nine_pages[stage][lane] - params.host_page_start] = 0;
    const bool last_subjob = job % 8 == 7 || first + SUB >= count;
    if (last_subjob) {
      __threadfence();
      __syncwarp();
      if (lane == 0) asm volatile("st.release.gpu.global.u32 [%0], %1;" ::
          "l"(host_batch_flags(params) + chunk), "r"(epoch + 1) : "memory");
    }
    bool ready = true;
    if (lane < SUB && first + lane < count)
      ready = host_batch_ld_acquire(params.pool_flags + params.slot_pool[first + lane]) == epoch + 1;
    const bool defer = !__all_sync(0xffffffffu, ready) && nine_deferred_count < kHostBatchMaxDeferred;
    if (defer) chunk_deferred = true;
    else {
      if (lane < SUB) {
        const uint32_t slot = first + lane;
        int32_t victim = -1;
        if (slot < count) {
          while (host_batch_ld_acquire(params.pool_flags + params.slot_pool[slot]) != epoch + 1)
            __nanosleep(64);
          victim = params.victim_of[slot];
          bind_host_batch_staged_slot(params, slot, victim);
        }
        nine_victims[lane] = victim;
      }
      __syncwarp();
#pragma unroll
      for (uint32_t i = 0; i < WORDS; ++i) {
        const uint32_t v = i * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            stage == 2 ? register_stage[i] : buffers[stage * SUB * VECS + v];
      }
      __syncwarp();
    }
    if (last_subjob) {
      if (chunk_deferred && lane == 0) nine_deferred[nine_deferred_count++] = chunk;
      __syncwarp();
      chunk_deferred = false;
    }
    if (base(job + 3) < count) issue(job + 3, stage);
  }
  cp_async::wait_group<0>();
  __syncwarp();
  // Preserve the original deferred-cache rule; already-filled subjobs may be copied again.
  // Each unique staged page has one selected victim, so this repeats identical bindings/data.
#pragma unroll 1
  for (uint32_t d = 0; d < nine_deferred_count; ++d) {
#pragma unroll 1
    for (uint32_t sub = 0; sub < 8; ++sub) {
      const uint32_t first = nine_deferred[d] * 64 + sub * SUB;
      if (lane < SUB) {
        int32_t victim = -1;
        if (first + lane < count) {
          while (host_batch_ld_acquire(params.pool_flags + params.slot_pool[first + lane]) != epoch + 1)
            __nanosleep(64);
          victim = params.victim_of[first + lane];
          bind_host_batch_staged_slot(params, first + lane, victim);
        }
        nine_victims[lane] = victim;
      }
      __syncwarp();
      const uint4* source = reinterpret_cast<const uint4*>(params.staging +
          (params.cache_slots + static_cast<int64_t>(first)) * 576);
#pragma unroll
      for (uint32_t i = 0; i < WORDS; ++i) {
        const uint32_t v = i * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] = source[v];
      }
      __syncwarp();
    }
  }
  for (uint32_t slot = count + blockIdx.y * 32 + lane; slot < unique; slot += gridDim.y * 32)
    params.host_slot[params.unique_pages[slot] - params.host_page_start] = 0;
  __threadfence();
  __syncwarp();
}
