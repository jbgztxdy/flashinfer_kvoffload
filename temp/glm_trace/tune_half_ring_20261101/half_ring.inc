// Two logical 32KV tiles and one 16-row prefetch half share five physical halves.
template <typename Q, typename KV, typename O>
struct HalfRingStorage {
  union {
    struct {
      alignas(16) Q q_smem_nope[64 * 512];
      alignas(16) uint8_t q_alignment_padding[128];
      alignas(16) KV ckv_half[5][16 * 512];
      alignas(16) KV kpe_half[5][16 * 64];
      union {
        alignas(16) float m_wg[2][64];
        alignas(16) float d_wg[2][64];
      };
    };
    alignas(16) O o_smem[64 * 512];
  };
};

__device__ __forceinline__ uint32_t half_ring_index(uint32_t first, uint32_t offset) {
  const uint32_t index = first + offset;
  return index < 5 ? index : index - 5;
}

// A P half is 64 query rows x16 KV columns, with two uint4s per row. Toggle
// the column at each group of four rows: each ldmatrix 8x8 block spans all32 banks.
__device__ __forceinline__ uint32_t half_ring_p_offset(uint32_t row, uint32_t column) {
  return row * 2 + (column ^ ((row / 4) & 1));
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void half_ring_load_half(
    typename KTraits::SharedStorage& storage, const Params& params,
    uint32_t index_base, uint32_t kv_len, uint32_t kv_base, uint32_t half) {
  static_assert(Params::fully_staged && KTraits::NUM_THREADS == 256);
  using KV = typename KTraits::DTypeKV;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t row = tid / 16, vector = tid % 16;
  const bool valid = kv_base + row < kv_len;
  const typename KTraits::IdType page = valid ? params.remap[index_base + kv_base + row] : 0;
  const bool staged = page < 0;
  const int64_t source_row = staged ? -static_cast<int64_t>(page) - 1 : (page & ~kHostBatchOwnerBit);
  KV* source = (staged ? params.staging : params.ckv) + source_row * 576 + vector * 8;
  smem_t<SwizzleMode::k128B> ckv(storage.ckv_half[half]), kpe(storage.kpe_half[half]);
#pragma unroll
  for (uint32_t chunk = 0; chunk < 4; ++chunk) {
    const uint32_t offset = ckv.template get_permuted_offset<64>(row, chunk * 16 + vector);
    ckv.load_128b_async<SharedMemFillMode::kFillZero>(offset, source + chunk * 128, valid);
  }
  if (vector < 8) {
    const uint32_t offset = kpe.template get_permuted_offset<8>(row, vector);
    kpe.load_128b_async<SharedMemFillMode::kFillZero>(offset, source + 512, valid);
  }
  // Attention only reads near/resident/staged HBM. The dedicated fetch loop
  // retains the original post-read FAR hooks; no simulator request is moved here.
}

template <bool PAIR, typename KTraits>
__device__ __forceinline__ void half_ring_compute(
    typename KTraits::SharedStorage& storage, uint32_t first,
    uint32_t (&qpe)[4][4], float (*sa)[8], float (*sb)[8]) {
  const uint32_t lane = threadIdx.x, group = threadIdx.z, warp = threadIdx.y;
  const uint32_t a = half_ring_index(first, group), b = half_ring_index(first, 2 + group);
  smem_t<SwizzleMode::k128B> pa(storage.kpe_half[a]), pb(storage.kpe_half[b]);
  alignas(16) uint32_t kf[4], qf[4];
#pragma unroll
  for (uint32_t dim = 0; dim < 4; ++dim) {
    const uint32_t offset = pa.template get_permuted_offset<8>(
        8 * (lane / 16) + lane % 8, 2 * dim + (lane % 16) / 8);
    pa.ldmatrix_m8n8x4(offset, kf);
    if (dim == 0)
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(sa[0], qpe[dim], kf);
    else
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sa[0], qpe[dim], kf);
    if constexpr (PAIR) {
      pb.ldmatrix_m8n8x4(offset, kf);
      if (dim == 0)
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(sb[0], qpe[dim], kf);
      else
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sb[0], qpe[dim], kf);
    }
  }
  smem_t<SwizzleMode::k128B> q(storage.q_smem_nope), ka(storage.ckv_half[a]), kb(storage.ckv_half[b]);
#pragma unroll 4
  for (uint32_t dim = 0; dim < 32; ++dim) {
    const uint32_t qo = q.template get_permuted_offset<64>(warp * 16 + lane % 16, dim * 2 + lane / 16);
    q.ldmatrix_m8n8x4(qo, qf);
    const uint32_t ko = ka.template get_permuted_offset<64>(
        8 * (lane / 16) + lane % 8, 2 * dim + (lane % 16) / 8);
    ka.ldmatrix_m8n8x4(ko, kf);
    mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sa[0], qf, kf);
    if constexpr (PAIR) {
      kb.ldmatrix_m8n8x4(ko, kf);
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sb[0], qf, kf);
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void half_ring_pv(
    typename KTraits::SharedStorage& storage, uint32_t first, uint32_t tile_half,
    float (*scores)[8], float* d, float (*o)[8]) {
  const uint32_t lane = threadIdx.x, group = threadIdx.z, warp = threadIdx.y;
  alignas(16) typename KTraits::DTypeKV p_f16[8];
  vec_cast<typename KTraits::DTypeKV, float>::cast<8>(p_f16, scores[0]);
  mma::m16k16_rowsum_f16f16f32(d, p_f16);
  __syncthreads();
  auto* pbase = reinterpret_cast<uint32_t*>(storage.kpe_half[half_ring_index(first, tile_half + group)]);
  const uint32_t offset = half_ring_p_offset(warp * 16 + lane / 4, 0);
  pbase[offset * 4 + lane % 4] = *reinterpret_cast<uint32_t*>(&p_f16[0]);
  pbase[(offset + 16) * 4 + lane % 4] = *reinterpret_cast<uint32_t*>(&p_f16[2]);
  pbase[(offset ^ 1) * 4 + lane % 4] = *reinterpret_cast<uint32_t*>(&p_f16[4]);
  pbase[((offset ^ 1) + 16) * 4 + lane % 4] = *reinterpret_cast<uint32_t*>(&p_f16[6]);
  __syncthreads();
#pragma unroll
  for (uint32_t kv = 0; kv < 2; ++kv) {
    const uint32_t half = half_ring_index(first, tile_half + kv);
    smem_t<SwizzleMode::k128B> p(storage.kpe_half[half]), v(storage.ckv_half[half]);
    uint32_t pf[4];
    p.ldmatrix_m8n8x4(half_ring_p_offset(warp * 16 + lane % 16, lane / 16), pf);
    uint32_t vo = v.template get_permuted_offset<64>(lane % 16, group * 32 + lane / 16);
#pragma unroll
    for (uint32_t dim = 0; dim < 16; ++dim) {
      uint32_t vf[4];
      v.ldmatrix_m8n8x4_trans(vo, vf);
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(o[dim], pf, vf);
      vo = v.template advance_offset_by_column<2>(vo, dim);
    }
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void half_ring_attend_joint_host_tiles(
    const Params& params, typename KTraits::SharedStorage& storage,
    typename KTraits::AttentionVariant& variant, const uint32_t* scratch, uint32_t index_base,
    uint32_t kv_len, uint32_t row, int host_tiles, float (*scores)[8], float (*o)[8], float* m, float* d) {
  alignas(16) uint32_t qpe[4][4];
  half_ring_load_query(params, row, qpe);
  int hi = ceil_div(kv_len, 32u) - 1;
  const int mask_tile = kv_len / 32;
  uint32_t first = 0;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int tile = hi - i;
    if (tile >= 0 && tile < host_tiles)
      wait_host_batch_tile<KTraits>(params, scratch, index_base, kv_len, tile * 32);
  }
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const int tile = hi - i;
    if (tile >= 0) {
      half_ring_load_half<KTraits>(storage, params, index_base, kv_len, tile * 32, i * 2);
      cp_async::commit_group();
      half_ring_load_half<KTraits>(storage, params, index_base, kv_len, tile * 32 + 16, i * 2 + 1);
      cp_async::commit_group();
    }
  }
#pragma unroll 1
  for (; hi >= 1; hi -= 2) {
    const int next_hi = hi - 2, next_lo = hi - 3;
    const bool early_hi = next_hi >= host_tiles && next_hi >= 0;
    const bool early_lo = next_lo >= host_tiles && next_lo >= 0;
    if (early_hi) {
      half_ring_load_half<KTraits>(storage, params, index_base, kv_len, next_hi * 32, half_ring_index(first, 4));
      cp_async::commit_group();
      cp_async::wait_group<1>();
    } else {
      cp_async::wait_group<0>();
    }
    __syncthreads();
    half_ring_compute<true, KTraits>(storage, first, qpe, scores, scores + 1);
    if (hi >= mask_tile)
      logits_mask_<KTraits>(0, hi * 32, 1, kv_len, kv_len, params.num_heads, scores);
    half_ring_update_joint_host_mdo<KTraits>(&storage, variant, scores, scores + 1, o, m, d);
    half_ring_pv<KTraits>(storage, first, 0, scores, d, o);
    if (next_hi >= 0) {
      if (!early_hi && next_hi < host_tiles)
        wait_host_batch_tile<KTraits>(params, scratch, index_base, kv_len, next_hi * 32);
      __syncthreads();
      if (!early_hi) {
        half_ring_load_half<KTraits>(storage, params, index_base, kv_len, next_hi * 32, half_ring_index(first, 4));
        cp_async::commit_group();
      }
      half_ring_load_half<KTraits>(storage, params, index_base, kv_len, next_hi * 32 + 16, first);
      cp_async::commit_group();
      if (early_lo) {
        half_ring_load_half<KTraits>(storage, params, index_base, kv_len, next_lo * 32, half_ring_index(first, 1));
        cp_async::commit_group();
      }
    }
    half_ring_pv<KTraits>(storage, first, 2, scores + 1, d, o);
    if (next_lo >= 0) {
      if (!early_lo && next_lo < host_tiles)
        wait_host_batch_tile<KTraits>(params, scratch, index_base, kv_len, next_lo * 32);
      __syncthreads();
      if (!early_lo) {
        half_ring_load_half<KTraits>(storage, params, index_base, kv_len, next_lo * 32, half_ring_index(first, 1));
        cp_async::commit_group();
      }
      half_ring_load_half<KTraits>(storage, params, index_base, kv_len, next_lo * 32 + 16, half_ring_index(first, 2));
      cp_async::commit_group();
    }
    first = first ? first - 1 : 4;
  }
  cp_async::wait_group<0>();
  __syncthreads();
  if (hi == 0) {
    half_ring_compute<false, KTraits>(storage, first, qpe, scores, scores);
    logits_mask_<KTraits>(0, 0, 1, kv_len, kv_len, params.num_heads, scores);
    update_mdo_states_<KTraits>(&storage, 0, variant, scores, o, m, d);
    half_ring_pv<KTraits>(storage, first, 0, scores, d, o);
  }
}
