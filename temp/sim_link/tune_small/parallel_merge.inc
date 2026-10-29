// Experimental: one CTA per head, four groups reduce disjoint subsets of states.
template <typename KTraits, typename Params>
__device__ __forceinline__ void merge_host_batch_chunks_parallel(const Params& params,
                                                                uint8_t* smem) {
  using DTypeO = typename KTraits::DTypeO;
  constexpr uint32_t VEC = 8;
  constexpr uint32_t WIDTH = KTraits::HEAD_DIM_CKV / VEC;
  constexpr uint32_t PARTS = KTraits::NUM_THREADS / WIDTH;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t part = tid / WIDTH, feature = tid % WIDTH * VEC;
  const uint32_t tail_tiles = (params.num_rows - params.full_rows) * params.num_head_tiles;
  const uint32_t max_items = host_batch_max_host_items(params);
  auto* values = reinterpret_cast<float*>(smem);
  auto* maxima = values + PARTS * KTraits::HEAD_DIM_CKV;
  auto* denominators = maxima + PARTS;
#pragma unroll 1
  for (uint32_t unit = blockIdx.y; unit < tail_tiles * KTraits::CTA_TILE_Q; unit += gridDim.y) {
    const uint32_t tile = unit / KTraits::CTA_TILE_Q, h = unit % KTraits::CTA_TILE_Q;
    const uint32_t row = params.full_rows + tile / params.num_head_tiles;
    const uint32_t head_tile = tile % params.num_head_tiles;
    const uint32_t length = host_batch_row_len(params, row);
    const uint32_t host = host_batch_split_host(params, row);
    const uint32_t chunk = round_up(ceil_div(length - host, params.num_chunks), KTraits::CTA_TILE_KV);
    const uint32_t hbm_items = chunk ? ceil_div(length - host, chunk) : 0;
    const uint32_t host_items =
        host_batch_host_chunks<KTraits::CTA_TILE_KV>(host, max_items, params.host_item_entries).x;
    const uint32_t host_first = tail_tiles * params.num_chunks +
                               host_batch_host_items(params)[row] + head_tile * host_items;
    const uint32_t states = hbm_items + host_items;
    state_t<VEC> local;
#pragma unroll 1
    for (uint32_t k = part; k < states; k += PARTS) {
      const uint32_t partial = (k < hbm_items ? tile * params.num_chunks + k
                                             : host_first + k - hbm_items) * KTraits::CTA_TILE_Q + h;
      const uint4 raw = __ldcg(reinterpret_cast<const uint4*>(
          params.partial_o + static_cast<int64_t>(partial) * KTraits::HEAD_DIM_CKV + feature));
      vec_t<float, VEC> value;
#pragma unroll
      for (uint32_t j = 0; j < VEC; ++j) value[j] = float(reinterpret_cast<const DTypeO*>(&raw)[j]);
      local.merge(value, __ldcg(params.partial_lse + partial), 1.f);
    }
    local.o.store(values + part * KTraits::HEAD_DIM_CKV + feature);
    if (tid % WIDTH == 0) {
      maxima[part] = local.m;
      denominators[part] = local.d;
    }
    __syncthreads();
    if (part == 0) {
      state_t<VEC> merged;
#pragma unroll
      for (uint32_t p = 0; p < PARTS; ++p) {
        if (p < states) {
          vec_t<float, VEC> value;
          value.load(values + p * KTraits::HEAD_DIM_CKV + feature);
          merged.merge(value, maxima[p], denominators[p]);
        }
      }
      merged.normalize();
      const uint32_t head = head_tile * KTraits::CTA_TILE_Q + h;
      merged.o.cast_store(params.final_o + row * params.o_stride_n + head * params.o_stride_h + feature);
      if (params.final_lse && tid == 0) {
        params.final_lse[row * params.num_heads + head] =
            merged.get_lse() * (params.return_lse_base_on_e ? math::loge2 : 1.f);
      }
    }
    __syncthreads();
  }
}
