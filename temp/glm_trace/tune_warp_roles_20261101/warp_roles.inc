// Four fetch warps have their own barrier and payload region. Their physical z=0/1 keeps
// far_emu's unmodified waiter and issue-time thread valid. Compute uses physical z=2/3.
__device__ __forceinline__ void warp_role_fetch_sync() {
  asm volatile("bar.sync 2, 128;" ::: "memory");
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void fetch_warp_role_pages(const Params& params, uint8_t* smem,
                                                     uint32_t unique) {
  using DTypeKV = typename Params::DTypeKV;
  constexpr uint32_t ROW = KTraits::HEAD_DIM_CKV + KTraits::HEAD_DIM_KPE;
  constexpr uint32_t ROWS = 16, THREADS = 128, VECS = ROW * sizeof(DTypeKV) / 16;
  constexpr uint32_t CHUNK = ROWS * VECS;
  static_assert(CHUNK % THREADS == 0);
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t epoch = params.state[kHostBatchEpoch];
  const uint32_t count = min(unique, params.staging_capacity);
  auto* buffer = reinterpret_cast<uint4*>(smem);
  __shared__ typename Params::IdType page_ids[2][ROWS];
  __shared__ int32_t victims[ROWS];
  FAR_EMU_INIT(2);
  warp_role_fetch_sync();
  auto issue = [&](uint32_t chunk, uint32_t buf) {
    if (tid < ROWS && chunk * ROWS + tid < count)
      page_ids[buf][tid] = params.unique_pages[chunk * ROWS + tid];
    warp_role_fetch_sync();
#pragma unroll
    for (uint32_t i = 0; i < CHUNK / THREADS; ++i) {
      const uint32_t v = i * THREADS + tid;
      if (chunk * ROWS + v / VECS < count) {
        const auto page = page_ids[buf][v / VECS];
        const auto* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * ROW);
        cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffer + buf * CHUNK + v, source + v % VECS);
      }
    }
    FAR_EMU_TILE(2, buf, tid == 0 ? min(ROWS, count - chunk * ROWS) : 0, ROW * sizeof(DTypeKV));
    cp_async::commit_group();
  };
  uint32_t chunk = blockIdx.y;
  if (chunk * ROWS < count) issue(chunk, 0);
#pragma unroll 1
  for (uint32_t k = 0; chunk * ROWS < count; ++k, chunk += gridDim.y) {
    const uint32_t next = chunk + gridDim.y;
    if (next * ROWS < count) {
      issue(next, (k + 1) % 2);
      cp_async::wait_group<1>();
    } else {
      cp_async::wait_group<0>();
    }
    FAR_EMU_WAIT(2, k % 2);
    warp_role_fetch_sync();
    const uint32_t buf = k % 2;
    auto* destination = reinterpret_cast<uint4*>(params.staging +
        (params.cache_slots + static_cast<int64_t>(chunk) * ROWS) * ROW);
#pragma unroll
    for (uint32_t i = 0; i < CHUNK / THREADS; ++i) {
      const uint32_t v = i * THREADS + tid;
      if (chunk * ROWS + v / VECS < count) destination[v] = buffer[buf * CHUNK + v];
    }
    if (tid < ROWS && chunk * ROWS + tid < count)
      params.host_slot[page_ids[buf][tid] - params.host_page_start] = 0;
    __threadfence();
    warp_role_fetch_sync();
    if (tid == 0) {
      asm volatile("st.release.gpu.global.u32 [%0], %1;" ::
          "l"(host_batch_flags(params) + chunk), "r"(epoch + 1) : "memory");
    }
    // Every pool's exact LRU selection completed before role separation; no deferred queue,
    // readiness vote or pool spin can serialize the independent compute warps here.
    if (tid < ROWS) {
      const uint32_t slot = chunk * ROWS + tid;
      const int32_t victim = slot < count ? params.victim_of[slot] : -1;
      victims[tid] = victim;
      if (victim >= 0) {
        const auto page = page_ids[buf][tid];
        const int32_t evicted = params.slot_page[victim];
        if (evicted >= 0) params.resident[evicted - params.host_page_start] = -1;
        params.resident[page - params.host_page_start] = victim;
        params.slot_page[victim] = page;
      }
    }
    warp_role_fetch_sync();
#pragma unroll
    for (uint32_t i = 0; i < CHUNK / THREADS; ++i) {
      const uint32_t v = i * THREADS + tid;
      const int32_t victim = victims[v / VECS];
      if (victim >= 0)
        reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * ROW)[v % VECS] =
            buffer[buf * CHUNK + v];
    }
    warp_role_fetch_sync();
  }
  // Full staging is a dispatch requirement, so every unique page is represented above.
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void compute_warp_role_items(const Params& params,
    typename KTraits::SharedStorage& storage, uint8_t* smem, const uint32_t* scratch) {
  float s_frag[KTraits::NUM_MMA_KV / 2][8];
  alignas(16) float o_frag[KTraits::NUM_MMA_D_CKV / 2][8];
  float m[2], d[2];
#pragma unroll 1
  for (uint32_t item = blockIdx.y; item < params.num_rows * params.num_head_tiles; item += gridDim.y) {
    run_host_batch_item<KTraits, Params, false>(params, storage, smem, scratch, item, s_frag, o_frag, m, d);
  }
}
