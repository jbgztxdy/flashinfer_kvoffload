// Separate full-column QK/softmax from the two native PV feature groups.
template <typename Q, typename KV, typename O>
struct QKSStorage {
  union {
    struct {
      alignas(16) Q q_smem_nope[64 * 512];
      alignas(16) KV ckv_smem[2][32 * 512];
      alignas(16) KV kpe_p_smem[2][32 * 64];
      alignas(16) float o_scale[64], m_final[64], d_final[64];
      alignas(128) uint4 service_payload[2][576];
    };
    alignas(16) O o_smem[64 * 512];
  };
};
template <typename Q, typename KV, typename O, typename Id>
struct QKSTraits : CachedPETraits<Q, KV, O, Id> {
  static constexpr uint32_t NUM_THREADS = 384;
  using SharedStorage = QKSStorage<Q, KV, O>;
};
template <typename Params>
struct QKSHostParams : CachedPEHostParams<Params> {};

__device__ __forceinline__ void qks_all_sync() {
  asm volatile("bar.sync 1, 384;" ::: "memory");
}
__device__ __forceinline__ void qks_producer_sync() {
  asm volatile("bar.sync 2, 128;" ::: "memory");
}

template <typename KTraits, bool SYNC_WARP = false>
__device__ __forceinline__ void qks_write_o(typename KTraits::SharedStorage* smem_storage,
                                        typename KTraits::DTypeO* final_o, float* final_lse,
                                        typename KTraits::DTypeO* partial_o, float* partial_lse,
                                        float (*o_frag)[8], typename KTraits::DTypeQKAccum* m,
                                        float* d, const uint32_t o_stride_n,
                                        const uint32_t o_stride_h, const uint32_t q_len,
                                        const uint32_t packed_offset, const uint_fastdiv& num_heads,
                                        const bool& return_lse_base_on_e) {
  using DTypeO = typename KTraits::DTypeO;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t HEAD_DIM_CKV = KTraits::HEAD_DIM_CKV;
  constexpr uint32_t UPCAST_STRIDE_FINAL_O = KTraits::UPCAST_STRIDE_FINAL_O;
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = (threadIdx.z - 1), warp_idx_in_wg = threadIdx.y;
  smem_t<KTraits::SWIZZLE_MODE_O> o_smem(smem_storage->o_smem);
  // The output tile aliases query storage, so finish prior reads first.
  if constexpr (SYNC_WARP) __syncwarp();
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 2; ++mma_d) {
    uint32_t o_frag_f16[8 / 2];
    vec_cast<DTypeO, float>::cast<8>((DTypeO*)o_frag_f16, o_frag[mma_d]);
#ifdef FLASHINFER_STMATRIX_M8N8X4_ENABLED
    uint32_t o_smem_offset_w = o_smem.template get_permuted_offset<UPCAST_STRIDE_FINAL_O>(
        warp_idx_in_wg * 16 + lane_idx % 16,
        warpgroup_idx * NUM_MMA_D_CKV + mma_d * 2 + lane_idx / 16);
    o_smem.template stmatrix_m8n8x4(o_smem_offset_w, o_frag_f16);
#else
    uint32_t o_smem_offset_w = o_smem.template get_permuted_offset<UPCAST_STRIDE_FINAL_O>(
        warp_idx_in_wg * 16 + lane_idx / 4, warpgroup_idx * NUM_MMA_D_CKV + mma_d * 2);
    ((uint32_t*)(o_smem.base + o_smem_offset_w))[lane_idx % 4] = o_frag_f16[0];
    ((uint32_t*)(o_smem.base + o_smem_offset_w + 8 * UPCAST_STRIDE_FINAL_O))[lane_idx % 4] =
        o_frag_f16[1];
    ((uint32_t*)(o_smem.base + (o_smem_offset_w ^ 0x1)))[lane_idx % 4] = o_frag_f16[2];
    ((uint32_t*)(o_smem.base + (o_smem_offset_w ^ 0x1) + 8 * UPCAST_STRIDE_FINAL_O))[lane_idx % 4] =
        o_frag_f16[3];
#endif
  }

  // Host partial writes may reuse this shared tile within the same warp.
  if constexpr (SYNC_WARP) __syncwarp();
  if (partial_o != nullptr) {
    // write to partial_o
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      uint32_t q_idx = (packed_offset + warp_idx_in_wg * 16 + 8 * j + lane_idx / 4) / num_heads;
      if (lane_idx % 4 == 0 && q_idx < q_len) {
        partial_lse[(blockIdx.x * 4 + warp_idx_in_wg) * 16 + 8 * j + lane_idx / 4] =
            math::ptx_log2(d[j]) + float(m[j]);
      }
    }

    // step 1. smem to gmem
    uint32_t o_smem_offset_w = o_smem.template get_permuted_offset<UPCAST_STRIDE_FINAL_O>(
        warp_idx_in_wg * 16 + lane_idx / 8, warpgroup_idx * NUM_MMA_D_CKV + lane_idx % 8);
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      uint32_t q_idx = (packed_offset + warp_idx_in_wg * 16 + 4 * j + lane_idx / 8) / num_heads;
      DTypeO* o_partial_ptr =
          partial_o +
          ((blockIdx.x * 4 + warp_idx_in_wg) * 16 + 4 * j + lane_idx / 8) * HEAD_DIM_CKV +
          warpgroup_idx * (HEAD_DIM_CKV / 2) + (lane_idx % 8) * upcast_size<DTypeO>();
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 8; ++mma_d) {
        if (q_idx < q_len) {
          o_smem.template store_128b(o_smem_offset_w, o_partial_ptr);
        }
        o_partial_ptr += 8 * upcast_size<DTypeO>();
        o_smem_offset_w = o_smem.template advance_offset_by_column<8>(o_smem_offset_w, mma_d);
      }
      o_smem_offset_w =
          o_smem.template advance_offset_by_row<4, UPCAST_STRIDE_FINAL_O>(o_smem_offset_w) -
          NUM_MMA_D_CKV;
    }
  } else {
    // write to final_o

    if (final_lse) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        uint32_t q, r;
        num_heads.divmod(packed_offset + warp_idx_in_wg * 16 + 8 * j + lane_idx / 4, q, r);
        if (lane_idx % 4 == 0 && q < q_len) {
          final_lse[q * num_heads + r] = math::ptx_log2(d[j]) + float(m[j]);
          if (return_lse_base_on_e) {
            final_lse[q * num_heads + r] *= math::loge2;
          }
        }
      }
    }

    // step 1. smem to gmem
    uint32_t o_smem_offset_w = o_smem.template get_permuted_offset<UPCAST_STRIDE_FINAL_O>(
        warp_idx_in_wg * 16 + lane_idx / 8, warpgroup_idx * NUM_MMA_D_CKV + lane_idx % 8);
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      uint32_t q, r;
      num_heads.divmod(packed_offset + warp_idx_in_wg * 16 + 4 * j + lane_idx / 8, q, r);
      DTypeO* o_final_ptr = final_o + q * o_stride_n + r * o_stride_h +
                            warpgroup_idx * (HEAD_DIM_CKV / 2) +
                            (lane_idx % 8) * upcast_size<DTypeO>();
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 8; ++mma_d) {
        if (q < q_len) {
          o_smem.template store_128b(o_smem_offset_w, o_final_ptr);
        }
        o_final_ptr += 8 * upcast_size<DTypeO>();
        o_smem_offset_w = o_smem.template advance_offset_by_column<8>(o_smem_offset_w, mma_d);
      }
      o_smem_offset_w =
          o_smem.template advance_offset_by_row<4, UPCAST_STRIDE_FINAL_O>(o_smem_offset_w) -
          NUM_MMA_D_CKV;
    }
  }
  if constexpr (SYNC_WARP) __syncwarp();
}

template <typename Params>
__device__ __forceinline__ void qks_load_query(const Params& params, uint32_t row,
                                                   uint32_t (&qpe)[4][4]) {
  const uint32_t lane = threadIdx.x, warp = threadIdx.y;
  const auto* query = params.q_pe + row * params.q_pe_stride_n;
#pragma unroll
  for (uint32_t dim = 0; dim < 4; ++dim) {
    const uint32_t head = warp * 16 + lane / 4, col = dim * 16 + (lane % 4) * 2;
    qpe[dim][0] = *reinterpret_cast<const uint32_t*>(query + head * params.q_pe_stride_h + col);
    qpe[dim][1] = *reinterpret_cast<const uint32_t*>(query + (head + 8) * params.q_pe_stride_h + col);
    qpe[dim][2] = *reinterpret_cast<const uint32_t*>(query + head * params.q_pe_stride_h + col + 8);
    qpe[dim][3] = *reinterpret_cast<const uint32_t*>(query + (head + 8) * params.q_pe_stride_h + col + 8);
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void qks_load_q(
    typename KTraits::SharedStorage& storage, const Params& params, uint32_t row) {
  const uint32_t tid = threadIdx.y * 32 + threadIdx.x;
  smem_t<SwizzleMode::k128B> q(storage.q_smem_nope);
#pragma unroll
  for (uint32_t i = 0; i < 32; ++i) {
    const uint32_t v = tid + i * 128, head = v / 64, column = v % 64;
    const auto* source = params.q_nope + row * params.q_nope_stride_n +
                         head * params.q_nope_stride_h + column * 8;
    q.load_128b_async<SharedMemFillMode::kFillZero>(q.template get_permuted_offset<64>(head, column), source, true);
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void qks_load_kv(
    typename KTraits::SharedStorage& storage, const Params& params,
    uint32_t index_base, uint32_t kv_len, uint32_t kv_base, uint32_t stage) {
  static_assert(Params::fully_staged);
  const uint32_t tid = threadIdx.y * 32 + threadIdx.x;
  smem_t<SwizzleMode::k128B> ckv(storage.ckv_smem[stage]), kpe(storage.kpe_p_smem[stage]);
#pragma unroll
  for (uint32_t half = 0; half < 2; ++half) {
    const uint32_t r = tid / 8 + half * 16, column = tid % 8;
    const bool valid = kv_base + r < kv_len;
    const typename KTraits::IdType page = valid ? params.remap[index_base + kv_base + r] : 0;
    const bool staged = page < 0;
    const int64_t source_row = staged ? -static_cast<int64_t>(page) - 1 : (page & ~kHostBatchOwnerBit);
    const auto* source = (staged ? params.staging : params.ckv) + source_row * 576 + column * 8;
#pragma unroll
    for (uint32_t chunk = 0; chunk < 8; ++chunk) {
      ckv.load_128b_async<SharedMemFillMode::kFillZero>(
          ckv.template get_permuted_offset<64>(r, chunk * 8 + column), source + chunk * 64, valid);
    }
    kpe.load_128b_async<SharedMemFillMode::kFillZero>(
        kpe.template get_permuted_offset<8>(r, column), source + 512, valid);
  }
}

template <bool PAIR, typename KTraits>
__device__ __forceinline__ void qks_compute_qk(
    typename KTraits::SharedStorage& storage, uint32_t (&qpe)[4][4], float (&s)[2][2][8]) {
  const uint32_t lane = threadIdx.x, warp = threadIdx.y;
  smem_t<SwizzleMode::k128B> pe0(storage.kpe_p_smem[0]), pe1(storage.kpe_p_smem[1]);
  uint32_t kf[4];
#pragma unroll
  for (uint32_t dim = 0; dim < 4; ++dim) {
#pragma unroll
    for (uint32_t n = 0; n < 2; ++n) {
      const uint32_t ko = pe0.template get_permuted_offset<8>(n * 16 + 8 * (lane / 16) + lane % 8, 2 * dim + (lane % 16) / 8);
      pe0.ldmatrix_m8n8x4(ko, kf);
      if (dim == 0)
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(s[0][n], qpe[dim], kf);
      else
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(s[0][n], qpe[dim], kf);
      if constexpr (PAIR) {
        pe1.ldmatrix_m8n8x4(ko, kf);
        if (dim == 0)
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(s[1][n], qpe[dim], kf);
        else
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(s[1][n], qpe[dim], kf);
      }
    }
  }
  smem_t<SwizzleMode::k128B> q(storage.q_smem_nope), k0(storage.ckv_smem[0]), k1(storage.ckv_smem[1]);
#pragma unroll 4
  for (uint32_t dim = 0; dim < 32; ++dim) {
    uint32_t qf[4];
    q.ldmatrix_m8n8x4(q.template get_permuted_offset<64>(warp * 16 + lane % 16, dim * 2 + lane / 16), qf);
#pragma unroll
    for (uint32_t n = 0; n < 2; ++n) {
      const uint32_t ko = k0.template get_permuted_offset<64>(n * 16 + 8 * (lane / 16) + lane % 8, 2 * dim + (lane % 16) / 8);
      k0.ldmatrix_m8n8x4(ko, kf);
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(s[0][n], qf, kf);
      if constexpr (PAIR) {
        k1.ldmatrix_m8n8x4(ko, kf);
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(s[1][n], qf, kf);
      }
    }
  }
}

template <bool PAIR, typename KTraits>
__device__ __forceinline__ void qks_softmax_publish(
    typename KTraits::SharedStorage& storage, uint32_t hi, uint32_t kv_len,
    float sm_scale, float (&s)[2][2][8], float* m, float* d) {
  const uint32_t lane = threadIdx.x, warp = threadIdx.y;
#pragma unroll
  for (uint32_t t = 0; t < (PAIR ? 2 : 1); ++t) {
#pragma unroll
    for (uint32_t n = 0; n < 2; ++n) {
#pragma unroll
      for (uint32_t r = 0; r < 8; ++r) {
        const uint32_t col = (hi - t) * 32 + n * 16 + 2 * (lane % 4) + 8 * (r / 4) + r % 2;
        if (col >= kv_len) s[t][n][r] = KTraits::MaskFillValue;
      }
    }
  }
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    const float previous = m[j];
#pragma unroll
    for (uint32_t t = 0; t < (PAIR ? 2 : 1); ++t) {
#pragma unroll
      for (uint32_t n = 0; n < 2; ++n) {
        m[j] = max(m[j], max(max(s[t][n][j * 2], s[t][n][j * 2 + 1]), max(s[t][n][j * 2 + 4], s[t][n][j * 2 + 5])));
      }
    }
    m[j] = max(m[j], math::shfl_xor_sync(m[j], 2));
    m[j] = max(m[j], math::shfl_xor_sync(m[j], 1));
    const float scale = math::ptx_exp2(previous * sm_scale - m[j] * sm_scale);
    d[j] *= scale;
    if (lane % 4 == 0) storage.o_scale[warp * 16 + lane / 4 + j * 8] = scale;
#pragma unroll
    for (uint32_t t = 0; t < (PAIR ? 2 : 1); ++t) {
#pragma unroll
      for (uint32_t n = 0; n < 2; ++n) {
#pragma unroll
        for (uint32_t x = 0; x < 2; ++x) {
          s[t][n][j * 2 + x] = math::ptx_exp2(s[t][n][j * 2 + x] * sm_scale - m[j] * sm_scale);
          s[t][n][j * 2 + x + 4] = math::ptx_exp2(s[t][n][j * 2 + x + 4] * sm_scale - m[j] * sm_scale);
        }
      }
    }
  }
#pragma unroll
  for (uint32_t t = 0; t < (PAIR ? 2 : 1); ++t) {
    smem_t<SwizzleMode::k64B> p(storage.kpe_p_smem[t]);
#pragma unroll
    for (uint32_t n = 0; n < 2; ++n) {
      alignas(16) typename KTraits::DTypeKV ph[8];
      vec_cast<typename KTraits::DTypeKV, float>::cast<8>(ph, s[t][n]);
      mma::m16k16_rowsum_f16f16f32(d, ph);
      const uint32_t off = p.template get_permuted_offset<4>(warp * 16 + lane / 4, n * 2);
      reinterpret_cast<uint32_t*>(p.base + off)[lane % 4] = *reinterpret_cast<uint32_t*>(&ph[0]);
      reinterpret_cast<uint32_t*>(p.base + off + 32)[lane % 4] = *reinterpret_cast<uint32_t*>(&ph[2]);
      reinterpret_cast<uint32_t*>(p.base + (off ^ 1))[lane % 4] = *reinterpret_cast<uint32_t*>(&ph[4]);
      reinterpret_cast<uint32_t*>(p.base + (off ^ 1) + 32)[lane % 4] = *reinterpret_cast<uint32_t*>(&ph[6]);
    }
  }
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    if (lane % 4 == 0) {
      const uint32_t q = warp * 16 + lane / 4 + j * 8;
      storage.m_final[q] = m[j];storage.d_final[q] = d[j];
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void qks_compute_pv(
    typename KTraits::SharedStorage& storage, uint32_t stage, float (*o)[8]) {
  const uint32_t lane = threadIdx.x, group = threadIdx.z - 1, warp = threadIdx.y;
  smem_t<SwizzleMode::k128B> v(storage.ckv_smem[stage]);
  smem_t<SwizzleMode::k64B> p(storage.kpe_p_smem[stage]);
  uint32_t vo = v.template get_permuted_offset<64>(lane % 16, group * 32 + lane / 16);
  uint32_t po = p.template get_permuted_offset<4>(warp * 16 + lane % 16, lane / 16);
#pragma unroll
  for (uint32_t n = 0; n < 2; ++n) {
    uint32_t pf[4];p.ldmatrix_m8n8x4(po, pf);
    po = p.template advance_offset_by_column<2>(po, n);
#pragma unroll
    for (uint32_t dim = 0; dim < 16; ++dim) {
      uint32_t vf[4];v.ldmatrix_m8n8x4_trans(vo, vf);
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(o[dim], pf, vf);
      vo = v.template advance_offset_by_column<2>(vo, dim);
    }
    vo = v.template advance_offset_by_row<16, 64>(vo) - 32;
  }
}

// Shared subjob metadata; only the128 QK-role lanes access it after upfront LRU.
template <typename Id>
struct QKSMetadata {
  Id pages[3][8];
  int32_t victims[8];
  uint32_t tile_ready;
};
struct QKSContext {
  uint32_t job = 0;
  uint4 r0, r1, r2, r3, r4;
};

__device__ __forceinline__ uint32_t qks_service_base(uint32_t job) {
  return (blockIdx.y + job / 8 * gridDim.y) * 64 + (job % 8) * 8;
}

template <uint32_t STAGE, typename KTraits, typename Params>
__device__ __forceinline__ void qks_service_issue(
    const Params& params, typename KTraits::SharedStorage& storage,
    QKSMetadata<typename Params::IdType>& metadata, QKSContext& ctx,
    uint32_t job, uint32_t count) {
  const uint32_t tid = threadIdx.y * 32 + threadIdx.x, first = qks_service_base(job);
  if (tid < 8 && first + tid < count) metadata.pages[STAGE][tid] = params.unique_pages[first + tid];
  qks_producer_sync();
  auto issue_word = [&](uint32_t v, uint4& reg) {
    if (v < 576 && first + v / 72 < count) {
      const auto page = metadata.pages[STAGE][v / 72];
      const auto* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * 576);
      if constexpr (STAGE == 2) reg = source[v % 72];
      else cp_async::load_128b<PrefetchMode::kNoPrefetch>(storage.service_payload[STAGE] + v, source + v % 72);
    }
  };
  issue_word(tid, ctx.r0);issue_word(tid + 128, ctx.r1);issue_word(tid + 256, ctx.r2);
  issue_word(tid + 384, ctx.r3);issue_word(tid + 512, ctx.r4);
  // Exactly the existing model hook after the real reads; only physical tid0 speaks for rows.
  FAR_EMU_TILE(3, STAGE, tid == 0 ? min(8u, count - first) : 0, 1152);
  if constexpr (STAGE != 2) cp_async::commit_group();
  qks_producer_sync();
}

template <uint32_t STAGE, typename KTraits, typename Params>
__device__ __forceinline__ void qks_service_consume(
    const Params& params, typename KTraits::SharedStorage& storage,
    QKSMetadata<typename Params::IdType>& metadata, QKSContext& ctx,
    uint32_t epoch, uint32_t count) {
  const uint32_t tid = threadIdx.y * 32 + threadIdx.x, first = qks_service_base(ctx.job);
  if constexpr (STAGE != 2) cp_async::wait_group<0>();
#ifdef FLASHINFER_FAR_EMU
  ::flashinfer::far_emu::State far_emu_state{0, 0, -1};
#endif
  FAR_EMU_WAIT(3, STAGE);
  qks_producer_sync();
  auto* destination = reinterpret_cast<uint4*>(params.staging + (params.cache_slots + static_cast<int64_t>(first)) * 576);
  auto write_stage = [&](uint32_t v, const uint4& reg) {
    if (v < 576 && first + v / 72 < count) {
      if constexpr (STAGE == 2) destination[v] = reg;
      else destination[v] = storage.service_payload[STAGE][v];
    }
  };
  write_stage(tid, ctx.r0);write_stage(tid + 128, ctx.r1);write_stage(tid + 256, ctx.r2);
  write_stage(tid + 384, ctx.r3);write_stage(tid + 512, ctx.r4);
  if (tid < 8 && first + tid < count) params.host_slot[metadata.pages[STAGE][tid] - params.host_page_start] = 0;
  if (ctx.job % 8 == 7 || first + 8 >= count) {
    __threadfence();qks_producer_sync();
    if (tid == 0) asm volatile("st.release.gpu.global.u32 [%0], %1;" ::
        "l"(host_batch_flags(params) + first / 64), "r"(epoch + 1) : "memory");
  }
  // Every pool victim was selected before the grid barrier. No cachefill is deferred.
  if (tid < 8) {
    const uint32_t slot = first + tid;
    int32_t victim = -1;
    if (slot < count) {
      victim = params.victim_of[slot];bind_host_batch_staged_slot(params, slot, victim);
    }
    metadata.victims[tid] = victim;
  }
  qks_producer_sync();
  auto write_cache = [&](uint32_t v, const uint4& reg) {
    if (v < 576) {
      const int32_t victim = metadata.victims[v / 72];
      if (victim >= 0) {
        auto* dest = reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * 576);
        if constexpr (STAGE == 2) dest[v % 72] = reg;
        else dest[v % 72] = storage.service_payload[STAGE][v];
      }
    }
  };
  write_cache(tid, ctx.r0);write_cache(tid + 128, ctx.r1);write_cache(tid + 256, ctx.r2);
  write_cache(tid + 384, ctx.r3);write_cache(tid + 512, ctx.r4);
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
    const bool tile_ready = metadata.tile_ready;
    qks_producer_sync();  // Every lane captures readiness before the next loop can overwrite it.
    if (tile_ready) break;
    if (!qks_service_step<KTraits>(params, storage, metadata, ctx, epoch, count)) __nanosleep(64);
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void qks_produce(
    const Params& params, typename KTraits::SharedStorage& storage,
    const uint32_t* scratch, uint32_t row, uint32_t kv_len,
    QKSMetadata<typename Params::IdType>& metadata, QKSContext& ctx, uint32_t epoch, uint32_t count) {
  const uint32_t lane = threadIdx.x, warp = threadIdx.y;
  float m[2] = {-math::inf, -math::inf}, d[2] = {2.f, 2.f};
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    if (lane % 4 == 0) {
      const uint32_t q = warp * 16 + lane / 4 + j * 8;
      storage.m_final[q] = m[j];storage.d_final[q] = d[j];
    }
  }
  qks_all_sync();
  if (kv_len == 0) return;
  const uint32_t index_base = params.kv_indptr[row];
  const int host_tiles = ceil_div(params.state[kHostBatchHostCounts + row], 32u);
  alignas(16) uint32_t qpe[4][4];qks_load_query(params, row, qpe);
  qks_load_q<KTraits>(storage, params, row);
  int hi = ceil_div(kv_len, 32u) - 1;
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int tile = hi - t;
    if (tile >= 0) {
      if (tile < host_tiles) qks_wait_tile<KTraits>(params, storage, metadata, ctx, scratch, index_base, kv_len, tile * 32, epoch, count);
      qks_producer_sync();
      qks_load_kv<KTraits>(storage, params, index_base, kv_len, tile * 32, t);
      cp_async::commit_group();
    }
  }
#pragma unroll 1
  for (; hi >= 0; hi -= 2) {
    float scores[2][2][8];
    cp_async::wait_group<0>();qks_producer_sync();
    if (hi >= 1) {
      qks_compute_qk<true, KTraits>(storage, qpe, scores);
      qks_producer_sync();  // All QK warps finish KPE reads before any P store.
      qks_softmax_publish<true, KTraits>(storage, hi, kv_len, params.sm_scale * math::log2e, scores, m, d);
    } else {
      qks_compute_qk<false, KTraits>(storage, qpe, scores);
      qks_producer_sync();  // All QK warps finish KPE reads before any P store.
      qks_softmax_publish<false, KTraits>(storage, hi, kv_len, params.sm_scale * math::log2e, scores, m, d);
    }
    qks_all_sync();  // P and scale are ready; all KPE readers have finished.
    qks_service_step<KTraits>(params, storage, metadata, ctx, epoch, count);  // One subjob during PV-hi.
    qks_all_sync();  // PV-hi has released stage0.
    const int next_hi = hi - 2, next_lo = hi - 3;
    if (next_hi >= 0) {
      if (next_hi < host_tiles) qks_wait_tile<KTraits>(params, storage, metadata, ctx, scratch, index_base, kv_len, next_hi * 32, epoch, count);
      qks_producer_sync();
      qks_load_kv<KTraits>(storage, params, index_base, kv_len, next_hi * 32, 0);
      cp_async::commit_group();
    }
    qks_all_sync();  // PV-lo has released stage1; stage0 loading overlaps it.
    if (next_lo >= 0) {
      if (next_lo < host_tiles) qks_wait_tile<KTraits>(params, storage, metadata, ctx, scratch, index_base, kv_len, next_lo * 32, epoch, count);
      qks_producer_sync();
      qks_load_kv<KTraits>(storage, params, index_base, kv_len, next_lo * 32, 1);
      cp_async::commit_group();
    }
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void qks_consume(
    const Params& params, typename KTraits::SharedStorage& storage, uint32_t row, uint32_t kv_len) {
  const uint32_t lane = threadIdx.x, warp = threadIdx.y;
  alignas(16) float o[16][8];float m[2], d[2];
  init_states_<KTraits>(o, m, d);
  qks_all_sync();
#pragma unroll 1
  for (int hi = ceil_div(kv_len, 32u) - 1; hi >= 0; hi -= 2) {
    qks_all_sync();
    float scale[2];
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) scale[j] = storage.o_scale[warp * 16 + lane / 4 + j * 8];
#pragma unroll
    for (uint32_t dim = 0; dim < 16; ++dim) {
#pragma unroll
      for (uint32_t r = 0; r < 8; ++r) o[dim][r] *= scale[(r % 4) / 2];
    }
    qks_compute_pv<KTraits>(storage, 0, o);
    qks_all_sync();
    if (hi >= 1) qks_compute_pv<KTraits>(storage, 1, o);
    qks_all_sync();
  }
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    const uint32_t q = warp * 16 + lane / 4 + j * 8;
    m[j] = storage.m_final[q];d[j] = storage.d_final[q];
    const float reciprocal = m[j] != -math::inf ? math::ptx_rcp(d[j]) : 0.f;
#pragma unroll
    for (uint32_t dim = 0; dim < 16; ++dim) {
#pragma unroll
      for (uint32_t x = 0; x < 2; ++x) {
        o[dim][j * 2 + x] *= reciprocal;o[dim][j * 2 + x + 4] *= reciprocal;
      }
    }
  }
  finalize_m_<KTraits>(typename KTraits::AttentionVariant(params, row, nullptr), m);
  qks_write_o<KTraits, true>(&storage, params.final_o + row * params.o_stride_n,
      params.final_lse ? params.final_lse + row * params.num_heads : nullptr,
      nullptr, nullptr, o, m, d, params.o_stride_n, params.o_stride_h, 1, 0,
      params.num_heads, params.return_lse_base_on_e);
}

template <typename KTraits, typename Params, bool ROW_MERGE>
__device__ __forceinline__ void qks_run_host_items(
    const Params& params, typename KTraits::SharedStorage& storage, uint8_t* smem,
    uint32_t* scratch, bool fetcher, uint32_t unique) {
  static_assert(Params::whole_rows && !ROW_MERGE);
  __shared__ QKSMetadata<typename Params::IdType> metadata;
  const uint32_t epoch = scratch[kHostBatchScratchEpoch];
  const uint32_t count = min(unique, params.staging_capacity);
  if (threadIdx.z == 0) {
    const uint32_t tid = threadIdx.y * 32 + threadIdx.x;
    QKSContext ctx;
    FAR_EMU_INIT(3);
    qks_producer_sync();
    if (qks_service_base(0) < count) qks_service_issue<0, KTraits>(params, storage, metadata, ctx, 0, count);
    if (qks_service_base(1) < count) qks_service_issue<1, KTraits>(params, storage, metadata, ctx, 1, count);
    if (qks_service_base(2) < count) qks_service_issue<2, KTraits>(params, storage, metadata, ctx, 2, count);
#pragma unroll 1
    for (uint32_t row = blockIdx.y; row < params.num_rows; row += gridDim.y) {
      qks_produce<KTraits>(params, storage, scratch, row, host_batch_row_len(params, row), metadata, ctx, epoch, count);
      qks_all_sync();  // Output alias is released before the next query load.
    }
#pragma unroll 1
    while (qks_service_step<KTraits>(params, storage, metadata, ctx, epoch, count)) {}
    cp_async::wait_group<0>();
    for (uint32_t slot = count + blockIdx.y * 128 + tid; slot < unique; slot += gridDim.y * 128)
      params.host_slot[params.unique_pages[slot] - params.host_page_start] = 0;
    __threadfence();qks_producer_sync();
  } else {
#pragma unroll 1
    for (uint32_t row = blockIdx.y; row < params.num_rows; row += gridDim.y) {
      qks_consume<KTraits>(params, storage, row, host_batch_row_len(params, row));
      qks_all_sync();
    }
  }
}
