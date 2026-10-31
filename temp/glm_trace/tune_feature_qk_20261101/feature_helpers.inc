// Each warp group computes disjoint feature dimensions for the same 16 KV rows.
// Sum both partial scores before the original false-QK_SHARD softmax/PV path.
// The attention equation is unchanged; FP32 dot-product reduction order differs.
template <bool INIT, typename KTraits, uint32_t D, uint32_t Q_STRIDE, uint32_t K_STRIDE,
          SwizzleMode Q_SWIZZLE, SwizzleMode K_SWIZZLE>
__device__ __forceinline__ void host_batch_qk_feature_half(
    smem_t<Q_SWIZZLE> q, smem_t<K_SWIZZLE> k, float (*score)[8]) {
  static_assert(D % 2 == 0 && KTraits::NUM_MMA_KV == 1);
  const uint32_t lane = threadIdx.x, group = threadIdx.z, warp = threadIdx.y;
  alignas(16) uint32_t qf[4], kf[4];
#pragma unroll
  for (uint32_t local = 0; local < D / 2; ++local) {
    const uint32_t dim = local + group * (D / 2);
    const uint32_t qo = q.template get_permuted_offset<Q_STRIDE>(
        warp * 16 + lane % 16, dim * 2 + lane / 16);
    const uint32_t ko = k.template get_permuted_offset<K_STRIDE>(
        8 * (lane / 16) + lane % 8, dim * 2 + (lane % 16) / 8);
    q.ldmatrix_m8n8x4(qo, qf);
    k.ldmatrix_m8n8x4(ko, kf);
    if (INIT && local == 0)
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(
          score[0], qf, kf);
    else
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(score[0], qf, kf);
  }
}

template <typename KTraits>
__device__ __forceinline__ void host_batch_compute_feature_qk(
    typename KTraits::SharedStorage& storage, uint32_t stage, float (*score)[8]) {
  if constexpr (KTraits::CTA_TILE_Q == 32 && KTraits::CTA_TILE_KV == 16) {
    if constexpr (KTraits::FEATURE_QK) {
      smem_t<KTraits::SWIZZLE_MODE_Q_PE> qp(storage.q_smem_pe);
      smem_t<KTraits::SWIZZLE_MODE_KPE> kp(storage.kpe_p_smem[stage]);
      host_batch_qk_feature_half<true, KTraits, KTraits::NUM_MMA_D_KPE,
          KTraits::UPCAST_STRIDE_Q_PE, KTraits::UPCAST_STRIDE_KPE>(qp, kp, score);
      smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> qn(storage.q_smem_nope);
      smem_t<KTraits::SWIZZLE_MODE_CKV> kn(storage.ckv_smem[stage]);
      host_batch_qk_feature_half<false, KTraits, KTraits::NUM_MMA_D_CKV,
          KTraits::UPCAST_STRIDE_Q_NOPE, KTraits::UPCAST_STRIDE_CKV>(qn, kn, score);
      const uint32_t group = threadIdx.z, warp = threadIdx.y, lane = threadIdx.x;
#pragma unroll
      for (uint32_t r = 0; r < 8; ++r) storage.qk_exchange[group][warp][r][lane] = score[0][r];
      __syncthreads();
#pragma unroll
      for (uint32_t r = 0; r < 8; ++r)
        score[0][r] = storage.qk_exchange[0][warp][r][lane] + storage.qk_exchange[1][warp][r][lane];
      // Both groups must have consumed the exchange before the next tile writes it,
      // including the final loop's consecutive stages without an outer barrier.
      __syncthreads();
    } else {
      compute_mla_qk<KTraits>(&storage, stage, score);
    }
  } else {
    compute_mla_qk<KTraits>(&storage, stage, score);
  }
}
