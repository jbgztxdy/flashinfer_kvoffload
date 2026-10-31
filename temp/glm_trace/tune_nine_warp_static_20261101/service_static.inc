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
  uint4 register_stage_0;
  uint4 register_stage_1;
  uint4 register_stage_2;
  uint4 register_stage_3;
  uint4 register_stage_4;
  uint4 register_stage_5;
  uint4 register_stage_6;
  uint4 register_stage_7;
  uint4 register_stage_8;
  uint4 register_stage_9;
  uint4 register_stage_10;
  uint4 register_stage_11;
  uint4 register_stage_12;
  uint4 register_stage_13;
  uint4 register_stage_14;
  uint4 register_stage_15;
  uint4 register_stage_16;
  uint4 register_stage_17;
  FAR_EMU_INIT(3);
  __syncwarp();
  const auto base = [&](uint32_t job) {
    return (blockIdx.y + job / 8 * gridDim.y) * 64 + job % 8 * SUB;
  };
  if (base(0) < count) {
    const uint32_t issue_job = 0;

    const uint32_t first = base(issue_job);
    if (lane < SUB && first + lane < count) nine_pages[0][lane] = params.unique_pages[first + lane];
    __syncwarp();
    {
      const uint32_t v = 0 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_0 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 1 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_1 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 2 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_2 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 3 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_3 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 4 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_4 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 5 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_5 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 6 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_6 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 7 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_7 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 8 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_8 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 9 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_9 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 10 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_10 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 11 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_11 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 12 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_12 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 13 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_13 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 14 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_14 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 15 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_15 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 16 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_16 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 17 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_17 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    // Same unmodified model/hook after actual reads, including the synchronous register 0.
    FAR_EMU_TILE(3, 0, lane == 0 ? min(SUB, count - first) : 0, 1152);
    if (0 != 2) cp_async::commit_group();
    __syncwarp();
  
  }
  if (base(1) < count) {
    const uint32_t issue_job = 1;

    const uint32_t first = base(issue_job);
    if (lane < SUB && first + lane < count) nine_pages[1][lane] = params.unique_pages[first + lane];
    __syncwarp();
    {
      const uint32_t v = 0 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_0 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 1 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_1 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 2 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_2 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 3 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_3 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 4 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_4 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 5 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_5 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 6 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_6 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 7 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_7 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 8 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_8 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 9 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_9 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 10 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_10 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 11 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_11 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 12 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_12 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 13 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_13 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 14 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_14 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 15 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_15 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 16 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_16 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 17 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_17 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    // Same unmodified model/hook after actual reads, including the synchronous register 1.
    FAR_EMU_TILE(3, 1, lane == 0 ? min(SUB, count - first) : 0, 1152);
    if (1 != 2) cp_async::commit_group();
    __syncwarp();
  
  }
  if (base(2) < count) {
    const uint32_t issue_job = 2;

    const uint32_t first = base(issue_job);
    if (lane < SUB && first + lane < count) nine_pages[2][lane] = params.unique_pages[first + lane];
    __syncwarp();
    {
      const uint32_t v = 0 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_0 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 1 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_1 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 2 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_2 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 3 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_3 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 4 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_4 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 5 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_5 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 6 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_6 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 7 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_7 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 8 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_8 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 9 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_9 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 10 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_10 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 11 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_11 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 12 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_12 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 13 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_13 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 14 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_14 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 15 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_15 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 16 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_16 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 17 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_17 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    // Same unmodified model/hook after actual reads, including the synchronous register 2.
    FAR_EMU_TILE(3, 2, lane == 0 ? min(SUB, count - first) : 0, 1152);
    if (2 != 2) cp_async::commit_group();
    __syncwarp();
  
  }
  bool chunk_deferred = false;
#pragma unroll 1
  for (uint32_t cycle = 0; base(cycle) < count; cycle += 3) {
    if (base(cycle + 0) < count) {
      const uint32_t job = cycle + 0;

    const uint32_t first = base(job), chunk = first / 64;
    if (0 != 2) cp_async::wait_group<0>();
    FAR_EMU_WAIT(3, 0);
    __syncwarp();
    auto* destination = reinterpret_cast<uint4*>(params.staging +
        (params.cache_slots + static_cast<int64_t>(first)) * 576);
    {
      const uint32_t v = 0 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 0 == 2 ? register_stage_0 : buffers[0 * SUB * VECS + v];
        }
    {
      const uint32_t v = 1 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 0 == 2 ? register_stage_1 : buffers[0 * SUB * VECS + v];
        }
    {
      const uint32_t v = 2 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 0 == 2 ? register_stage_2 : buffers[0 * SUB * VECS + v];
        }
    {
      const uint32_t v = 3 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 0 == 2 ? register_stage_3 : buffers[0 * SUB * VECS + v];
        }
    {
      const uint32_t v = 4 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 0 == 2 ? register_stage_4 : buffers[0 * SUB * VECS + v];
        }
    {
      const uint32_t v = 5 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 0 == 2 ? register_stage_5 : buffers[0 * SUB * VECS + v];
        }
    {
      const uint32_t v = 6 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 0 == 2 ? register_stage_6 : buffers[0 * SUB * VECS + v];
        }
    {
      const uint32_t v = 7 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 0 == 2 ? register_stage_7 : buffers[0 * SUB * VECS + v];
        }
    {
      const uint32_t v = 8 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 0 == 2 ? register_stage_8 : buffers[0 * SUB * VECS + v];
        }
    {
      const uint32_t v = 9 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 0 == 2 ? register_stage_9 : buffers[0 * SUB * VECS + v];
        }
    {
      const uint32_t v = 10 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 0 == 2 ? register_stage_10 : buffers[0 * SUB * VECS + v];
        }
    {
      const uint32_t v = 11 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 0 == 2 ? register_stage_11 : buffers[0 * SUB * VECS + v];
        }
    {
      const uint32_t v = 12 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 0 == 2 ? register_stage_12 : buffers[0 * SUB * VECS + v];
        }
    {
      const uint32_t v = 13 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 0 == 2 ? register_stage_13 : buffers[0 * SUB * VECS + v];
        }
    {
      const uint32_t v = 14 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 0 == 2 ? register_stage_14 : buffers[0 * SUB * VECS + v];
        }
    {
      const uint32_t v = 15 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 0 == 2 ? register_stage_15 : buffers[0 * SUB * VECS + v];
        }
    {
      const uint32_t v = 16 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 0 == 2 ? register_stage_16 : buffers[0 * SUB * VECS + v];
        }
    {
      const uint32_t v = 17 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 0 == 2 ? register_stage_17 : buffers[0 * SUB * VECS + v];
        }
    if (lane < SUB && first + lane < count)
      params.host_slot[nine_pages[0][lane] - params.host_page_start] = 0;
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
    {
        const uint32_t v = 0 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            0 == 2 ? register_stage_0 : buffers[0 * SUB * VECS + v];
          }
    {
        const uint32_t v = 1 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            0 == 2 ? register_stage_1 : buffers[0 * SUB * VECS + v];
          }
    {
        const uint32_t v = 2 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            0 == 2 ? register_stage_2 : buffers[0 * SUB * VECS + v];
          }
    {
        const uint32_t v = 3 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            0 == 2 ? register_stage_3 : buffers[0 * SUB * VECS + v];
          }
    {
        const uint32_t v = 4 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            0 == 2 ? register_stage_4 : buffers[0 * SUB * VECS + v];
          }
    {
        const uint32_t v = 5 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            0 == 2 ? register_stage_5 : buffers[0 * SUB * VECS + v];
          }
    {
        const uint32_t v = 6 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            0 == 2 ? register_stage_6 : buffers[0 * SUB * VECS + v];
          }
    {
        const uint32_t v = 7 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            0 == 2 ? register_stage_7 : buffers[0 * SUB * VECS + v];
          }
    {
        const uint32_t v = 8 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            0 == 2 ? register_stage_8 : buffers[0 * SUB * VECS + v];
          }
    {
        const uint32_t v = 9 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            0 == 2 ? register_stage_9 : buffers[0 * SUB * VECS + v];
          }
    {
        const uint32_t v = 10 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            0 == 2 ? register_stage_10 : buffers[0 * SUB * VECS + v];
          }
    {
        const uint32_t v = 11 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            0 == 2 ? register_stage_11 : buffers[0 * SUB * VECS + v];
          }
    {
        const uint32_t v = 12 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            0 == 2 ? register_stage_12 : buffers[0 * SUB * VECS + v];
          }
    {
        const uint32_t v = 13 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            0 == 2 ? register_stage_13 : buffers[0 * SUB * VECS + v];
          }
    {
        const uint32_t v = 14 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            0 == 2 ? register_stage_14 : buffers[0 * SUB * VECS + v];
          }
    {
        const uint32_t v = 15 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            0 == 2 ? register_stage_15 : buffers[0 * SUB * VECS + v];
          }
    {
        const uint32_t v = 16 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            0 == 2 ? register_stage_16 : buffers[0 * SUB * VECS + v];
          }
    {
        const uint32_t v = 17 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            0 == 2 ? register_stage_17 : buffers[0 * SUB * VECS + v];
          }
      __syncwarp();
    }
    if (last_subjob) {
      if (chunk_deferred && lane == 0) nine_deferred[nine_deferred_count++] = chunk;
      __syncwarp();
      chunk_deferred = false;
    }
    if (base(job + 3) < count) {
    const uint32_t issue_job = job + 3;

    const uint32_t first = base(issue_job);
    if (lane < SUB && first + lane < count) nine_pages[0][lane] = params.unique_pages[first + lane];
    __syncwarp();
    {
      const uint32_t v = 0 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_0 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 1 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_1 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 2 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_2 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 3 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_3 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 4 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_4 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 5 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_5 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 6 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_6 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 7 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_7 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 8 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_8 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 9 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_9 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 10 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_10 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 11 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_11 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 12 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_12 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 13 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_13 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 14 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_14 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 15 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_15 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 16 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_16 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 17 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[0][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (0 == 2) register_stage_17 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 0 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    // Same unmodified model/hook after actual reads, including the synchronous register 0.
    FAR_EMU_TILE(3, 0, lane == 0 ? min(SUB, count - first) : 0, 1152);
    if (0 != 2) cp_async::commit_group();
    __syncwarp();
  
  }
  
    }
    if (base(cycle + 1) < count) {
      const uint32_t job = cycle + 1;

    const uint32_t first = base(job), chunk = first / 64;
    if (1 != 2) cp_async::wait_group<0>();
    FAR_EMU_WAIT(3, 1);
    __syncwarp();
    auto* destination = reinterpret_cast<uint4*>(params.staging +
        (params.cache_slots + static_cast<int64_t>(first)) * 576);
    {
      const uint32_t v = 0 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 1 == 2 ? register_stage_0 : buffers[1 * SUB * VECS + v];
        }
    {
      const uint32_t v = 1 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 1 == 2 ? register_stage_1 : buffers[1 * SUB * VECS + v];
        }
    {
      const uint32_t v = 2 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 1 == 2 ? register_stage_2 : buffers[1 * SUB * VECS + v];
        }
    {
      const uint32_t v = 3 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 1 == 2 ? register_stage_3 : buffers[1 * SUB * VECS + v];
        }
    {
      const uint32_t v = 4 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 1 == 2 ? register_stage_4 : buffers[1 * SUB * VECS + v];
        }
    {
      const uint32_t v = 5 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 1 == 2 ? register_stage_5 : buffers[1 * SUB * VECS + v];
        }
    {
      const uint32_t v = 6 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 1 == 2 ? register_stage_6 : buffers[1 * SUB * VECS + v];
        }
    {
      const uint32_t v = 7 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 1 == 2 ? register_stage_7 : buffers[1 * SUB * VECS + v];
        }
    {
      const uint32_t v = 8 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 1 == 2 ? register_stage_8 : buffers[1 * SUB * VECS + v];
        }
    {
      const uint32_t v = 9 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 1 == 2 ? register_stage_9 : buffers[1 * SUB * VECS + v];
        }
    {
      const uint32_t v = 10 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 1 == 2 ? register_stage_10 : buffers[1 * SUB * VECS + v];
        }
    {
      const uint32_t v = 11 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 1 == 2 ? register_stage_11 : buffers[1 * SUB * VECS + v];
        }
    {
      const uint32_t v = 12 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 1 == 2 ? register_stage_12 : buffers[1 * SUB * VECS + v];
        }
    {
      const uint32_t v = 13 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 1 == 2 ? register_stage_13 : buffers[1 * SUB * VECS + v];
        }
    {
      const uint32_t v = 14 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 1 == 2 ? register_stage_14 : buffers[1 * SUB * VECS + v];
        }
    {
      const uint32_t v = 15 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 1 == 2 ? register_stage_15 : buffers[1 * SUB * VECS + v];
        }
    {
      const uint32_t v = 16 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 1 == 2 ? register_stage_16 : buffers[1 * SUB * VECS + v];
        }
    {
      const uint32_t v = 17 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 1 == 2 ? register_stage_17 : buffers[1 * SUB * VECS + v];
        }
    if (lane < SUB && first + lane < count)
      params.host_slot[nine_pages[1][lane] - params.host_page_start] = 0;
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
    {
        const uint32_t v = 0 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            1 == 2 ? register_stage_0 : buffers[1 * SUB * VECS + v];
          }
    {
        const uint32_t v = 1 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            1 == 2 ? register_stage_1 : buffers[1 * SUB * VECS + v];
          }
    {
        const uint32_t v = 2 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            1 == 2 ? register_stage_2 : buffers[1 * SUB * VECS + v];
          }
    {
        const uint32_t v = 3 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            1 == 2 ? register_stage_3 : buffers[1 * SUB * VECS + v];
          }
    {
        const uint32_t v = 4 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            1 == 2 ? register_stage_4 : buffers[1 * SUB * VECS + v];
          }
    {
        const uint32_t v = 5 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            1 == 2 ? register_stage_5 : buffers[1 * SUB * VECS + v];
          }
    {
        const uint32_t v = 6 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            1 == 2 ? register_stage_6 : buffers[1 * SUB * VECS + v];
          }
    {
        const uint32_t v = 7 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            1 == 2 ? register_stage_7 : buffers[1 * SUB * VECS + v];
          }
    {
        const uint32_t v = 8 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            1 == 2 ? register_stage_8 : buffers[1 * SUB * VECS + v];
          }
    {
        const uint32_t v = 9 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            1 == 2 ? register_stage_9 : buffers[1 * SUB * VECS + v];
          }
    {
        const uint32_t v = 10 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            1 == 2 ? register_stage_10 : buffers[1 * SUB * VECS + v];
          }
    {
        const uint32_t v = 11 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            1 == 2 ? register_stage_11 : buffers[1 * SUB * VECS + v];
          }
    {
        const uint32_t v = 12 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            1 == 2 ? register_stage_12 : buffers[1 * SUB * VECS + v];
          }
    {
        const uint32_t v = 13 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            1 == 2 ? register_stage_13 : buffers[1 * SUB * VECS + v];
          }
    {
        const uint32_t v = 14 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            1 == 2 ? register_stage_14 : buffers[1 * SUB * VECS + v];
          }
    {
        const uint32_t v = 15 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            1 == 2 ? register_stage_15 : buffers[1 * SUB * VECS + v];
          }
    {
        const uint32_t v = 16 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            1 == 2 ? register_stage_16 : buffers[1 * SUB * VECS + v];
          }
    {
        const uint32_t v = 17 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            1 == 2 ? register_stage_17 : buffers[1 * SUB * VECS + v];
          }
      __syncwarp();
    }
    if (last_subjob) {
      if (chunk_deferred && lane == 0) nine_deferred[nine_deferred_count++] = chunk;
      __syncwarp();
      chunk_deferred = false;
    }
    if (base(job + 3) < count) {
    const uint32_t issue_job = job + 3;

    const uint32_t first = base(issue_job);
    if (lane < SUB && first + lane < count) nine_pages[1][lane] = params.unique_pages[first + lane];
    __syncwarp();
    {
      const uint32_t v = 0 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_0 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 1 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_1 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 2 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_2 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 3 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_3 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 4 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_4 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 5 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_5 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 6 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_6 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 7 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_7 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 8 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_8 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 9 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_9 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 10 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_10 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 11 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_11 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 12 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_12 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 13 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_13 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 14 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_14 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 15 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_15 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 16 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_16 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 17 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[1][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (1 == 2) register_stage_17 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 1 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    // Same unmodified model/hook after actual reads, including the synchronous register 1.
    FAR_EMU_TILE(3, 1, lane == 0 ? min(SUB, count - first) : 0, 1152);
    if (1 != 2) cp_async::commit_group();
    __syncwarp();
  
  }
  
    }
    if (base(cycle + 2) < count) {
      const uint32_t job = cycle + 2;

    const uint32_t first = base(job), chunk = first / 64;
    if (2 != 2) cp_async::wait_group<0>();
    FAR_EMU_WAIT(3, 2);
    __syncwarp();
    auto* destination = reinterpret_cast<uint4*>(params.staging +
        (params.cache_slots + static_cast<int64_t>(first)) * 576);
    {
      const uint32_t v = 0 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 2 == 2 ? register_stage_0 : buffers[2 * SUB * VECS + v];
        }
    {
      const uint32_t v = 1 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 2 == 2 ? register_stage_1 : buffers[2 * SUB * VECS + v];
        }
    {
      const uint32_t v = 2 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 2 == 2 ? register_stage_2 : buffers[2 * SUB * VECS + v];
        }
    {
      const uint32_t v = 3 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 2 == 2 ? register_stage_3 : buffers[2 * SUB * VECS + v];
        }
    {
      const uint32_t v = 4 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 2 == 2 ? register_stage_4 : buffers[2 * SUB * VECS + v];
        }
    {
      const uint32_t v = 5 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 2 == 2 ? register_stage_5 : buffers[2 * SUB * VECS + v];
        }
    {
      const uint32_t v = 6 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 2 == 2 ? register_stage_6 : buffers[2 * SUB * VECS + v];
        }
    {
      const uint32_t v = 7 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 2 == 2 ? register_stage_7 : buffers[2 * SUB * VECS + v];
        }
    {
      const uint32_t v = 8 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 2 == 2 ? register_stage_8 : buffers[2 * SUB * VECS + v];
        }
    {
      const uint32_t v = 9 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 2 == 2 ? register_stage_9 : buffers[2 * SUB * VECS + v];
        }
    {
      const uint32_t v = 10 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 2 == 2 ? register_stage_10 : buffers[2 * SUB * VECS + v];
        }
    {
      const uint32_t v = 11 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 2 == 2 ? register_stage_11 : buffers[2 * SUB * VECS + v];
        }
    {
      const uint32_t v = 12 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 2 == 2 ? register_stage_12 : buffers[2 * SUB * VECS + v];
        }
    {
      const uint32_t v = 13 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 2 == 2 ? register_stage_13 : buffers[2 * SUB * VECS + v];
        }
    {
      const uint32_t v = 14 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 2 == 2 ? register_stage_14 : buffers[2 * SUB * VECS + v];
        }
    {
      const uint32_t v = 15 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 2 == 2 ? register_stage_15 : buffers[2 * SUB * VECS + v];
        }
    {
      const uint32_t v = 16 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 2 == 2 ? register_stage_16 : buffers[2 * SUB * VECS + v];
        }
    {
      const uint32_t v = 17 * 32 + lane;
      if (first + v / VECS < count)
        destination[v] = 2 == 2 ? register_stage_17 : buffers[2 * SUB * VECS + v];
        }
    if (lane < SUB && first + lane < count)
      params.host_slot[nine_pages[2][lane] - params.host_page_start] = 0;
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
    {
        const uint32_t v = 0 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            2 == 2 ? register_stage_0 : buffers[2 * SUB * VECS + v];
          }
    {
        const uint32_t v = 1 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            2 == 2 ? register_stage_1 : buffers[2 * SUB * VECS + v];
          }
    {
        const uint32_t v = 2 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            2 == 2 ? register_stage_2 : buffers[2 * SUB * VECS + v];
          }
    {
        const uint32_t v = 3 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            2 == 2 ? register_stage_3 : buffers[2 * SUB * VECS + v];
          }
    {
        const uint32_t v = 4 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            2 == 2 ? register_stage_4 : buffers[2 * SUB * VECS + v];
          }
    {
        const uint32_t v = 5 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            2 == 2 ? register_stage_5 : buffers[2 * SUB * VECS + v];
          }
    {
        const uint32_t v = 6 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            2 == 2 ? register_stage_6 : buffers[2 * SUB * VECS + v];
          }
    {
        const uint32_t v = 7 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            2 == 2 ? register_stage_7 : buffers[2 * SUB * VECS + v];
          }
    {
        const uint32_t v = 8 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            2 == 2 ? register_stage_8 : buffers[2 * SUB * VECS + v];
          }
    {
        const uint32_t v = 9 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            2 == 2 ? register_stage_9 : buffers[2 * SUB * VECS + v];
          }
    {
        const uint32_t v = 10 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            2 == 2 ? register_stage_10 : buffers[2 * SUB * VECS + v];
          }
    {
        const uint32_t v = 11 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            2 == 2 ? register_stage_11 : buffers[2 * SUB * VECS + v];
          }
    {
        const uint32_t v = 12 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            2 == 2 ? register_stage_12 : buffers[2 * SUB * VECS + v];
          }
    {
        const uint32_t v = 13 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            2 == 2 ? register_stage_13 : buffers[2 * SUB * VECS + v];
          }
    {
        const uint32_t v = 14 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            2 == 2 ? register_stage_14 : buffers[2 * SUB * VECS + v];
          }
    {
        const uint32_t v = 15 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            2 == 2 ? register_stage_15 : buffers[2 * SUB * VECS + v];
          }
    {
        const uint32_t v = 16 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            2 == 2 ? register_stage_16 : buffers[2 * SUB * VECS + v];
          }
    {
        const uint32_t v = 17 * 32 + lane;
        const int32_t victim = nine_victims[v / VECS];
        if (victim >= 0) reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576)[v % VECS] =
            2 == 2 ? register_stage_17 : buffers[2 * SUB * VECS + v];
          }
      __syncwarp();
    }
    if (last_subjob) {
      if (chunk_deferred && lane == 0) nine_deferred[nine_deferred_count++] = chunk;
      __syncwarp();
      chunk_deferred = false;
    }
    if (base(job + 3) < count) {
    const uint32_t issue_job = job + 3;

    const uint32_t first = base(issue_job);
    if (lane < SUB && first + lane < count) nine_pages[2][lane] = params.unique_pages[first + lane];
    __syncwarp();
    {
      const uint32_t v = 0 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_0 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 1 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_1 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 2 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_2 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 3 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_3 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 4 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_4 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 5 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_5 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 6 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_6 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 7 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_7 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 8 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_8 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 9 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_9 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 10 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_10 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 11 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_11 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 12 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_12 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 13 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_13 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 14 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_14 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 15 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_15 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 16 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_16 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    {
      const uint32_t v = 17 * 32 + lane;
      if (first + v / VECS < count) {
        const auto page = nine_pages[2][v / VECS];
        const uint4* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
        if (2 == 2) register_stage_17 = source[v % VECS];
        else cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffers + 2 * SUB * VECS + v,
                                                          source + v % VECS);
      }
        }
    // Same unmodified model/hook after actual reads, including the synchronous register 2.
    FAR_EMU_TILE(3, 2, lane == 0 ? min(SUB, count - first) : 0, 1152);
    if (2 != 2) cp_async::commit_group();
    __syncwarp();
  
  }
  
    }
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
