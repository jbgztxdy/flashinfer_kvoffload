// Two native32-row score tiles share one maximum, denominator/output rescale, then each
// runs native PV in descending tile order. This changes FP32 reduction grouping, not the
// attention equation. It is distinct from the earlier paired-QK-only schedule.
template <typename KTraits>
__device__ __forceinline__ void update_joint_host_mdo(
    typename KTraits::SharedStorage* storage, typename KTraits::AttentionVariant variant,
    float (*s0)[8], float (*s1)[8], float (*o)[8], float* m, float* d) {
  static_assert(KTraits::QK_SHARD && KTraits::CTA_TILE_KV == 32);
  const uint32_t lane = threadIdx.x, wg = threadIdx.z, warp = threadIdx.y;
  const float scale = variant.sm_scale_log2;
  float previous[2];
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    previous[j] = m[j];
#pragma unroll
    for (uint32_t t = 0; t < 2; ++t) {
      const float* s = t == 0 ? s0[0] : s1[0];
      m[j] = max(m[j], max(max(s[j*2], s[j*2+1]), max(s[j*2+4], s[j*2+5])));
    }
    m[j] = max(m[j], math::shfl_xor_sync(m[j], 2));
    m[j] = max(m[j], math::shfl_xor_sync(m[j], 1));
    if (lane % 4 == 0) storage->m_wg[wg][warp*16+j*8+lane/4] = m[j];
  }
  __syncthreads();
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    m[j] = max(storage->m_wg[0][warp*16+j*8+lane/4], storage->m_wg[1][warp*16+j*8+lane/4]);
    const float factor = math::ptx_exp2(previous[j]*scale - m[j]*scale);
    d[j] *= factor;
#pragma unroll
    for (uint32_t feature = 0; feature < KTraits::NUM_MMA_D_CKV/2; ++feature) {
      o[feature][j*2] *= factor;
      o[feature][j*2+1] *= factor;
      o[feature][j*2+4] *= factor;
      o[feature][j*2+5] *= factor;
    }
#pragma unroll
    for (uint32_t t = 0; t < 2; ++t) {
      float* s = t == 0 ? s0[0] : s1[0];
      s[j*2] = math::ptx_exp2(s[j*2]*scale - m[j]*scale);
      s[j*2+1] = math::ptx_exp2(s[j*2+1]*scale - m[j]*scale);
      s[j*2+4] = math::ptx_exp2(s[j*2+4]*scale - m[j]*scale);
      s[j*2+5] = math::ptx_exp2(s[j*2+5]*scale - m[j]*scale);
    }
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void attend_joint_host_tiles(
    const Params& params, typename KTraits::SharedStorage& storage,
    typename KTraits::AttentionVariant& variant, const uint32_t* scratch, uint32_t index_base,
    uint32_t kv_len, int host_tiles, float (*scores)[8], float (*o)[8], float* m, float* d) {
  static_assert(KTraits::NUM_STAGES == 2 && KTraits::CTA_TILE_KV == 32 && Params::fully_staged);
  typename KTraits::IdType pages[kHostBatchTileRows<KTraits>];
  int hi = ceil_div(kv_len, 32u) - 1;
  const int mask_tile = kv_len / 32;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int tile = hi - i;
    if (tile >= 0 && tile < host_tiles)
      wait_host_batch_tile<KTraits>(params, scratch, index_base, kv_len, tile*32);
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int tile = hi - i;
    if (tile >= 0) {
      read_host_batch_pages<KTraits>(params, index_base, kv_len, tile*32, pages);
      load_host_batch_kv<KTraits>(&storage, params, pages, kv_len, tile*32, tile%2, index_base);
      cp_async::commit_group();
    }
  }
#pragma unroll 1
  for (; hi >= 1; hi -= 2) {
    const int lo = hi - 1, next_hi = hi - 2, next_lo = hi - 3;
    if (next_hi >= 0) read_host_batch_pages<KTraits>(params, index_base, kv_len, next_hi*32, pages);
    // The second current tile may still be in flight while the first native QK runs.
    cp_async::wait_group<1>();
    __syncthreads();
    compute_mla_qk<KTraits>(&storage, hi%2, scores);
    cp_async::wait_group<0>();
    __syncthreads();
    compute_mla_qk<KTraits>(&storage, lo%2, scores+1);
    if (hi >= mask_tile)
      logits_mask_<KTraits>(0, hi*32, 1, kv_len, kv_len, params.num_heads, scores);
    update_joint_host_mdo<KTraits>(&storage, variant, scores, scores+1, o, m, d);
    compute_mla_pv<KTraits>(&storage, hi%2, scores, d, o);
    if (next_hi >= 0) {
      if (next_hi < host_tiles)
        wait_host_batch_tile<KTraits>(params, scratch, index_base, kv_len, next_hi*32);
      __syncthreads();
      load_host_batch_kv<KTraits>(&storage, params, pages, kv_len, next_hi*32, next_hi%2, index_base);
      cp_async::commit_group();
    }
    // Refill the first released stage before PV of the second tile, preserving overlap.
    if (next_lo >= 0) read_host_batch_pages<KTraits>(params, index_base, kv_len, next_lo*32, pages);
    compute_mla_pv<KTraits>(&storage, lo%2, scores+1, d, o);
    if (next_lo >= 0) {
      if (next_lo < host_tiles)
        wait_host_batch_tile<KTraits>(params, scratch, index_base, kv_len, next_lo*32);
      __syncthreads();
      load_host_batch_kv<KTraits>(&storage, params, pages, kv_len, next_lo*32, next_lo%2, index_base);
      cp_async::commit_group();
    }
  }
  cp_async::wait_group<0>();
  __syncthreads();
  if (hi == 0) {
    compute_mla_qk<KTraits>(&storage, 0, scores);
    logits_mask_<KTraits>(0, 0, 1, kv_len, kv_len, params.num_heads, scores);
    update_mdo_states_<KTraits>(&storage, 0, variant, scores, o, m, d);
    compute_mla_pv<KTraits>(&storage, 0, scores, d, o);
  }
}
