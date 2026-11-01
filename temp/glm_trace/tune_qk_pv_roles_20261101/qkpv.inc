// Separate full-column QK/softmax from the two native PV feature groups.
template <typename Q, typename KV, typename O>
struct QKPVStorage {
  union {
    struct {
      alignas(16) Q q_smem_nope[64 * 512];
      alignas(16) KV ckv_smem[2][32 * 512];
      alignas(16) KV kpe_p_smem[2][32 * 64];
      alignas(16) float o_scale[64], m_final[64], d_final[64];
    };
    alignas(16) O o_smem[64 * 512];
    // Fetch must retain the original two64-row buffers, selected by sizeof storage.
    alignas(16) uint8_t fetch_budget[147968];
  };
};
template <typename Q, typename KV, typename O, typename Id>
struct QKPVTraits : CachedPETraits<Q, KV, O, Id> {
  static constexpr uint32_t NUM_THREADS = 384;
  using SharedStorage = QKPVStorage<Q, KV, O>;
};
template <typename Params>
struct QKPVHostParams : CachedPEHostParams<Params> {};

__device__ __forceinline__ void qkpv_all_sync() {
  asm volatile("bar.sync 1, 384;" ::: "memory");
}
__device__ __forceinline__ void qkpv_producer_sync() {
  asm volatile("bar.sync 2, 128;" ::: "memory");
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void qkpv_load_q(
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
__device__ __forceinline__ void qkpv_load_kv(
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
__device__ __forceinline__ void qkpv_compute_qk(
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
__device__ __forceinline__ void qkpv_softmax_publish(
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
__device__ __forceinline__ void qkpv_compute_pv(
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

template <typename KTraits, typename Params>
__device__ __forceinline__ void qkpv_produce(
    const Params& params, typename KTraits::SharedStorage& storage,
    const uint32_t* scratch, uint32_t row, uint32_t kv_len) {
  const uint32_t lane = threadIdx.x, warp = threadIdx.y;
  float m[2] = {-math::inf, -math::inf}, d[2] = {2.f, 2.f};
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    if (lane % 4 == 0) {
      const uint32_t q = warp * 16 + lane / 4 + j * 8;
      storage.m_final[q] = m[j];storage.d_final[q] = d[j];
    }
  }
  qkpv_all_sync();
  if (kv_len == 0) return;
  const uint32_t index_base = params.kv_indptr[row];
  const int host_tiles = ceil_div(params.state[kHostBatchHostCounts + row], 32u);
  alignas(16) uint32_t qpe[4][4];qkpv_load_query(params, row, qpe);
  qkpv_load_q<KTraits>(storage, params, row);
  int hi = ceil_div(kv_len, 32u) - 1;
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int tile = hi - t;
    if (tile >= 0) {
      if (tile < host_tiles) wait_host_batch_tile<KTraits>(params, scratch, index_base, kv_len, tile * 32);
      qkpv_producer_sync();
      qkpv_load_kv<KTraits>(storage, params, index_base, kv_len, tile * 32, t);
      cp_async::commit_group();
    }
  }
#pragma unroll 1
  for (; hi >= 0; hi -= 2) {
    float scores[2][2][8];
    cp_async::wait_group<0>();qkpv_producer_sync();
    if (hi >= 1) {
      qkpv_compute_qk<true, KTraits>(storage, qpe, scores);
      qkpv_producer_sync();  // All QK warps finish KPE reads before any P store.
      qkpv_softmax_publish<true, KTraits>(storage, hi, kv_len, params.sm_scale * math::log2e, scores, m, d);
    } else {
      qkpv_compute_qk<false, KTraits>(storage, qpe, scores);
      qkpv_producer_sync();  // All QK warps finish KPE reads before any P store.
      qkpv_softmax_publish<false, KTraits>(storage, hi, kv_len, params.sm_scale * math::log2e, scores, m, d);
    }
    qkpv_all_sync();  // P and scale are ready; all KPE readers have finished.
    qkpv_all_sync();  // PV-hi has released stage0.
    const int next_hi = hi - 2, next_lo = hi - 3;
    if (next_hi >= 0) {
      if (next_hi < host_tiles) wait_host_batch_tile<KTraits>(params, scratch, index_base, kv_len, next_hi * 32);
      qkpv_producer_sync();
      qkpv_load_kv<KTraits>(storage, params, index_base, kv_len, next_hi * 32, 0);
      cp_async::commit_group();
    }
    qkpv_all_sync();  // PV-lo has released stage1; stage0 loading overlaps it.
    if (next_lo >= 0) {
      if (next_lo < host_tiles) wait_host_batch_tile<KTraits>(params, scratch, index_base, kv_len, next_lo * 32);
      qkpv_producer_sync();
      qkpv_load_kv<KTraits>(storage, params, index_base, kv_len, next_lo * 32, 1);
      cp_async::commit_group();
    }
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void qkpv_consume(
    const Params& params, typename KTraits::SharedStorage& storage, uint32_t row, uint32_t kv_len) {
  const uint32_t lane = threadIdx.x, warp = threadIdx.y;
  alignas(16) float o[16][8];float m[2], d[2];
  init_states_<KTraits>(o, m, d);
  qkpv_all_sync();
#pragma unroll 1
  for (int hi = ceil_div(kv_len, 32u) - 1; hi >= 0; hi -= 2) {
    qkpv_all_sync();
    float scale[2];
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) scale[j] = storage.o_scale[warp * 16 + lane / 4 + j * 8];
#pragma unroll
    for (uint32_t dim = 0; dim < 16; ++dim) {
#pragma unroll
      for (uint32_t r = 0; r < 8; ++r) o[dim][r] *= scale[(r % 4) / 2];
    }
    qkpv_compute_pv<KTraits>(storage, 0, o);
    qkpv_all_sync();
    if (hi >= 1) qkpv_compute_pv<KTraits>(storage, 1, o);
    qkpv_all_sync();
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
  qkpv_write_o<KTraits, true>(&storage, params.final_o + row * params.o_stride_n,
      params.final_lse ? params.final_lse + row * params.num_heads : nullptr,
      nullptr, nullptr, o, m, d, params.o_stride_n, params.o_stride_h, 1, 0,
      params.num_heads, params.return_lse_base_on_e);
}

template <typename KTraits, typename Params, bool ROW_MERGE>
__device__ __forceinline__ void qkpv_run_host_items(
    const Params& params, typename KTraits::SharedStorage& storage, uint8_t* smem,
    uint32_t* scratch, bool fetcher, uint32_t unique) {
  static_assert(Params::whole_rows && !ROW_MERGE);
  const int lru_items = host_batch_caching(params) && params.stage && unique ? params.num_pools : 0;
  const int items = lru_items + params.num_rows;
  const int workers = gridDim.y - params.fetch_ctas;
  const int first = fetcher ? items - 1 - (blockIdx.y - workers) : blockIdx.y;
  const int step = fetcher ? -static_cast<int>(params.fetch_ctas) : workers;
#pragma unroll 1
  for (int item = first; item >= 0 && item < items; item += step) {
    if (!claim_host_batch_item(params, scratch, item)) {
      if (fetcher) break;
      continue;
    }
    if (item < lru_items) {
      run_host_batch_lru<KTraits>(params, smem, item, scratch[kHostBatchScratchEpoch], unique);
      __syncthreads();continue;
    }
    const uint32_t row = item - lru_items, length = host_batch_row_len(params, row);
    __syncthreads();
    if (threadIdx.z == 0) qkpv_produce<KTraits>(params, storage, scratch, row, length);
    else qkpv_consume<KTraits>(params, storage, row, length);
    __syncthreads();
  }
}
