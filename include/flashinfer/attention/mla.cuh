/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_MLA_FA2_CUH_
#define FLASHINFER_MLA_FA2_CUH_
#include <cooperative_groups.h>

#include <cstdint>
#include <sstream>

#include "../profiler.cuh"
#include "mla_params.cuh"
#include "prefill.cuh"
#include "variant_helper.cuh"

namespace flashinfer {

namespace mla {

struct StandardAttention : AttentionVariantBase {
  float sm_scale_log2;

  PROFILER_CLOSURE_PARAMS_DECL

  template <typename Params>
  __device__ __host__ StandardAttention(const Params& params, uint32_t batch_idx,
                                        uint8_t* smem_ptr) {
    sm_scale_log2 = params.sm_scale * math::log2e;
  }
};

template <uint32_t NUM_STAGES, uint32_t CTA_TILE_Q, uint32_t CTA_TILE_KV, uint32_t HEAD_DIM_CKV,
          uint32_t HEAD_DIM_KPE, typename DTypeQ, typename DTypeKV, typename DTypeO>
struct SharedStorageQKVO {
  union {
    struct {
      alignas(16) DTypeQ q_smem_nope[CTA_TILE_Q * HEAD_DIM_CKV];
      alignas(16) DTypeQ q_smem_pe[CTA_TILE_Q * HEAD_DIM_KPE];
      alignas(16) DTypeKV ckv_smem[NUM_STAGES][CTA_TILE_KV * HEAD_DIM_CKV];
      alignas(16) DTypeKV
          kpe_p_smem[NUM_STAGES]
                    [CTA_TILE_KV * (HEAD_DIM_KPE > CTA_TILE_Q ? HEAD_DIM_KPE : CTA_TILE_Q)];
      union {
        alignas(16) float m_wg[2][CTA_TILE_Q];  // cross warpgroup synchronization
        alignas(16) float d_wg[2][CTA_TILE_Q];  // cross warpgroup synchronization
      };
    };
    alignas(16) DTypeO o_smem[CTA_TILE_Q * HEAD_DIM_CKV];
  };
};

template <bool CAUSAL_, uint32_t NUM_STAGES_, bool QK_SHARD_, uint32_t HEAD_DIM_CKV_,
          uint32_t HEAD_DIM_KPE_, uint32_t CTA_TILE_Q_, uint32_t CTA_TILE_KV_, typename DTypeQ_,
          typename DTypeKV_, typename DTypeO_, typename IdType_>
struct KernelTraits {
  static constexpr bool CAUSAL = CAUSAL_;
  static constexpr uint32_t NUM_STAGES = NUM_STAGES_;
  // NOTE(Zihao): whether to shard Q*K computation across warpgroups
  // if true, each warpgroup will compute a subset of Q*K (sharded on the KV dimension)
  // if false, each warpgroup will compute the full Q*K, which is duplicated across warpgroups
  static constexpr bool QK_SHARD = QK_SHARD_;
  static constexpr uint32_t NUM_MMA_KV = CTA_TILE_KV_ / 16;
  static constexpr uint32_t HEAD_DIM_CKV = HEAD_DIM_CKV_;
  static constexpr uint32_t HEAD_DIM_KPE = HEAD_DIM_KPE_;
  static constexpr uint32_t HEAD_DIM_ALL = HEAD_DIM_CKV + HEAD_DIM_KPE;
  static constexpr uint32_t NUM_MMA_D_CKV = HEAD_DIM_CKV / 16;
  static constexpr uint32_t NUM_MMA_D_KPE = HEAD_DIM_KPE / 16;
  static constexpr uint32_t NUM_THREADS = 256;
  static constexpr uint32_t CTA_TILE_Q = CTA_TILE_Q_;
  static constexpr uint32_t CTA_TILE_KV = CTA_TILE_KV_;

  static constexpr SwizzleMode SWIZZLE_MODE_Q_NOPE = SwizzleMode::k128B;
  static constexpr SwizzleMode SWIZZLE_MODE_Q_PE = SwizzleMode::k128B;
  static constexpr SwizzleMode SWIZZLE_MODE_CKV = SwizzleMode::k128B;
  static constexpr SwizzleMode SWIZZLE_MODE_KPE = SwizzleMode::k128B;
  static constexpr SwizzleMode SWIZZLE_MODE_P =
      CTA_TILE_KV >= 64 ? SwizzleMode::k128B : SwizzleMode::k64B;
  static constexpr SwizzleMode SWIZZLE_MODE_O = SwizzleMode::k128B;
  static constexpr uint32_t UPCAST_STRIDE_Q_NOPE = HEAD_DIM_CKV / upcast_size<DTypeQ_>();
  static constexpr uint32_t UPCAST_STRIDE_Q_PE = HEAD_DIM_KPE / upcast_size<DTypeQ_>();
  static constexpr uint32_t UPCAST_STRIDE_CKV = HEAD_DIM_CKV / upcast_size<DTypeKV_>();
  static constexpr uint32_t UPCAST_STRIDE_KPE = HEAD_DIM_KPE / upcast_size<DTypeKV_>();
  static constexpr uint32_t UPCAST_STRIDE_FINAL_O = HEAD_DIM_CKV / upcast_size<DTypeO_>();
  static constexpr uint32_t UPCAST_STRIDE_P = CTA_TILE_KV / upcast_size<DTypeKV_>();

  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;
  using DTypeQKAccum = float;

  using SharedStorage = SharedStorageQKVO<NUM_STAGES, CTA_TILE_Q, CTA_TILE_KV, HEAD_DIM_CKV,
                                          HEAD_DIM_KPE, DTypeQ, DTypeKV, DTypeO>;
  using AttentionVariant = StandardAttention;

  static constexpr DTypeQKAccum MaskFillValue = -math::inf;
};

template <typename KTraits>
__device__ __forceinline__ void init_states_(float (*o_frag)[8], typename KTraits::DTypeQKAccum* m,
                                             float* d) {
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 2; ++mma_d) {
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
      o_frag[mma_d][reg_id] = 0.f;
    }
  }

#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    m[j] = typename KTraits::DTypeQKAccum(-math::inf);
    d[j] = 1.f;
  }
}

template <typename KTraits>
__device__ __forceinline__ void load_q(
    typename KTraits::SharedStorage* smem_storage, typename KTraits::DTypeQ* q_nope,
    typename KTraits::DTypeQ* q_pe, const uint32_t q_nope_stride_n, const uint32_t q_nope_stride_h,
    const uint32_t q_pe_stride_n, const uint32_t q_pe_stride_h, const uint32_t q_len,
    const uint32_t packed_offset, const uint_fastdiv& num_heads) {
  using DTypeQ = typename KTraits::DTypeQ;
  constexpr uint32_t UPCAST_STRIDE_Q_NOPE = KTraits::UPCAST_STRIDE_Q_NOPE;
  constexpr uint32_t UPCAST_STRIDE_Q_PE = KTraits::UPCAST_STRIDE_Q_PE;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t NUM_MMA_D_KPE = KTraits::NUM_MMA_D_KPE;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;

  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> q_smem_nope(smem_storage->q_smem_nope);
  smem_t<KTraits::SWIZZLE_MODE_Q_PE> q_smem_pe(smem_storage->q_smem_pe);

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < 2; ++mma_q) {
    uint32_t q, r;
    num_heads.divmod(
        packed_offset + lane_idx / 8 + (warpgroup_idx + mma_q * 2) * 16 + warp_idx_in_wg * 4, q, r);
    DTypeQ* q_nope_ptr =
        q_nope + q * q_nope_stride_n + r * q_nope_stride_h + (lane_idx % 8) * upcast_size<DTypeQ>();
    DTypeQ* q_pe_ptr =
        q_pe + q * q_pe_stride_n + r * q_pe_stride_h + (lane_idx % 8) * upcast_size<DTypeQ>();
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 4; ++mma_d) {
      uint32_t q_smem_nope_offset_w =
          q_smem_nope.template get_permuted_offset<UPCAST_STRIDE_Q_NOPE>(
              32 * mma_q + warpgroup_idx * 16 + warp_idx_in_wg * 4 + lane_idx / 8,
              mma_d * 8 + lane_idx % 8);
      q_smem_nope.load_128b_async<SharedMemFillMode::kFillZero>(q_smem_nope_offset_w, q_nope_ptr,
                                                                q < q_len);
      q_nope_ptr += 8 * upcast_size<DTypeQ>();
    }
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_KPE / 4; ++mma_d) {
      uint32_t q_smem_pe_offset_w = q_smem_pe.template get_permuted_offset<UPCAST_STRIDE_Q_PE>(
          32 * mma_q + warpgroup_idx * 16 + warp_idx_in_wg * 4 + lane_idx / 8,
          mma_d * 8 + lane_idx % 8);

      q_smem_pe.load_128b_async<SharedMemFillMode::kFillZero>(q_smem_pe_offset_w, q_pe_ptr,
                                                              q < q_len);
      q_pe_ptr += 8 * upcast_size<DTypeQ>();
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void load_kv(
    typename KTraits::SharedStorage* smem_storage, typename KTraits::DTypeKV* ckv,
    typename KTraits::DTypeKV* kpe, typename KTraits::IdType* indices, const uint32_t ckv_stride_n,
    const uint32_t ckv_stride_page, const uint32_t kpe_stride_n, const uint32_t kpe_stride_page,
    const uint32_t packed_kv_bound, const uint32_t packed_block_iter_base,
    const uint_fastdiv& block_size, const uint32_t stage_idx) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t UPCAST_STRIDE_CKV = KTraits::UPCAST_STRIDE_CKV;
  constexpr uint32_t UPCAST_STRIDE_KPE = KTraits::UPCAST_STRIDE_KPE;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t NUM_MMA_D_KPE = KTraits::NUM_MMA_D_KPE;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;

  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[stage_idx]);
  smem_t<KTraits::SWIZZLE_MODE_KPE> kpe_smem(smem_storage->kpe_p_smem[stage_idx]);

  if constexpr (KTraits::NUM_MMA_KV == 1) {
    if (warpgroup_idx == 0) {
      uint32_t q, r;
      uint32_t packed_block_iter = packed_block_iter_base + lane_idx / 8 + warp_idx_in_wg * 4;
      block_size.divmod(packed_block_iter, q, r);

      DTypeKV* ckv_ptr = ckv +
                         (packed_block_iter < packed_kv_bound ? indices[q] : 0) * ckv_stride_page +
                         r * ckv_stride_n + (lane_idx % 8) * upcast_size<DTypeKV>();
      DTypeKV* kpe_ptr = kpe +
                         (packed_block_iter < packed_kv_bound ? indices[q] : 0) * kpe_stride_page +
                         r * kpe_stride_n + (lane_idx % 8) * upcast_size<DTypeKV>();

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 4; ++mma_d) {
        uint32_t ckv_smem_offset_w = ckv_smem.template get_permuted_offset<UPCAST_STRIDE_CKV>(
            warp_idx_in_wg * 4 + lane_idx / 8, 8 * mma_d + lane_idx % 8);
        ckv_smem.load_128b_async<SharedMemFillMode::kFillZero>(ckv_smem_offset_w, ckv_ptr,
                                                               packed_block_iter < packed_kv_bound);
        ckv_ptr += 8 * upcast_size<DTypeKV>();
      }

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_KPE / 4; ++mma_d) {
        uint32_t kpe_smem_offset_w = kpe_smem.template get_permuted_offset<UPCAST_STRIDE_KPE>(
            warp_idx_in_wg * 4 + lane_idx / 8, 8 * mma_d + lane_idx % 8);
        kpe_smem.load_128b_async<SharedMemFillMode::kFillZero>(kpe_smem_offset_w, kpe_ptr,
                                                               packed_block_iter < packed_kv_bound);
        kpe_ptr += 8 * upcast_size<DTypeKV>();
      }
    }
  } else {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
      uint32_t q, r;
      uint32_t packed_block_iter = packed_block_iter_base + lane_idx / 8 +
                                   (warpgroup_idx + mma_kv * 2) * 16 + warp_idx_in_wg * 4;
      block_size.divmod(packed_block_iter, q, r);

      DTypeKV* ckv_ptr = ckv +
                         (packed_block_iter < packed_kv_bound ? indices[q] : 0) * ckv_stride_page +
                         r * ckv_stride_n + (lane_idx % 8) * upcast_size<DTypeKV>();
      DTypeKV* kpe_ptr = kpe +
                         (packed_block_iter < packed_kv_bound ? indices[q] : 0) * kpe_stride_page +
                         r * kpe_stride_n + (lane_idx % 8) * upcast_size<DTypeKV>();

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 4; ++mma_d) {
        uint32_t ckv_smem_offset_w = ckv_smem.template get_permuted_offset<UPCAST_STRIDE_CKV>(
            32 * mma_kv + warpgroup_idx * 16 + warp_idx_in_wg * 4 + lane_idx / 8,
            8 * mma_d + lane_idx % 8);
        ckv_smem.load_128b_async<SharedMemFillMode::kFillZero>(ckv_smem_offset_w, ckv_ptr,
                                                               packed_block_iter < packed_kv_bound);
        ckv_ptr += 8 * upcast_size<DTypeKV>();
      }

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_KPE / 4; ++mma_d) {
        uint32_t kpe_smem_offset_w = kpe_smem.template get_permuted_offset<UPCAST_STRIDE_KPE>(
            32 * mma_kv + warpgroup_idx * 16 + warp_idx_in_wg * 4 + lane_idx / 8,
            8 * mma_d + lane_idx % 8);
        kpe_smem.load_128b_async<SharedMemFillMode::kFillZero>(kpe_smem_offset_w, kpe_ptr,
                                                               packed_block_iter < packed_kv_bound);
        kpe_ptr += 8 * upcast_size<DTypeKV>();
      }
    }
  }
}

template <bool init, typename KTraits, uint32_t NUM_MMA_D_QK, uint32_t UPCAST_STRIDE_Q,
          uint32_t UPCAST_STRIDE_K, SwizzleMode SWIZZLE_MODE_Q, SwizzleMode SWIZZLE_MODE_KV,
          uint32_t MMA_D_START = 0>
__device__ __forceinline__ void compute_qk_(smem_t<SWIZZLE_MODE_Q> q_smem,
                                            smem_t<SWIZZLE_MODE_KV> k_smem,
                                            typename KTraits::DTypeQKAccum (*s_frag)[8]) {
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  alignas(16) uint32_t q_frag[4], k_frag[4];
  // compute q*k^T
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_QK; ++mma_d) {
    uint32_t q_smem_offset_r = q_smem.template get_permuted_offset<UPCAST_STRIDE_Q>(
        warp_idx_in_wg * 16 + lane_idx % 16, (mma_d + MMA_D_START) * 2 + lane_idx / 16);
    q_smem.ldmatrix_m8n8x4(q_smem_offset_r, q_frag);

    if constexpr (KTraits::QK_SHARD) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
        uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
            (warpgroup_idx * (KTraits::NUM_MMA_KV / 2) + mma_kv) * 16 + 8 * (lane_idx / 16) +
                lane_idx % 8,
            2 * (mma_d + MMA_D_START) + (lane_idx % 16) / 8);

        k_smem.ldmatrix_m8n8x4(k_smem_offset_r, k_frag);

        if (init && mma_d == 0) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(
              s_frag[mma_kv], q_frag, k_frag);
        } else {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(s_frag[mma_kv],
                                                                              q_frag, k_frag);
        }
      }
    } else {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
            mma_kv * 16 + 8 * (lane_idx / 16) + lane_idx % 8,
            2 * (mma_d + MMA_D_START) + (lane_idx % 16) / 8);

        k_smem.ldmatrix_m8n8x4(k_smem_offset_r, k_frag);

        if (init && mma_d == 0) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(
              s_frag[mma_kv], q_frag, k_frag);
        } else {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(s_frag[mma_kv],
                                                                              q_frag, k_frag);
        }
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void logits_mask_(const uint32_t qo_packed_idx_base,
                                             const uint32_t kv_idx_base, const uint32_t qo_len,
                                             const uint32_t kv_len, const uint32_t kv_end,
                                             const uint_fastdiv num_heads,
                                             typename KTraits::DTypeQKAccum (*s_frag)[8]) {
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  uint32_t q[2];
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    q[j] = (qo_packed_idx_base + warp_idx_in_wg * 16 + lane_idx / 4 + 8 * j) / num_heads;
  }

  if constexpr (KTraits::QK_SHARD) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV / 2; ++mma_kv) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const uint32_t q_idx = q[(reg_id % 4) / 2],
                       kv_idx = kv_idx_base + warpgroup_idx * (NUM_MMA_KV / 2) * 16 + mma_kv * 16 +
                                2 * (lane_idx % 4) + 8 * (reg_id / 4) + reg_id % 2;
        const bool mask =
            (!(KTraits::CAUSAL ? (kv_idx + qo_len > kv_len + q_idx || (kv_idx >= kv_end))
                               : kv_idx >= kv_end));
        s_frag[mma_kv][reg_id] = (mask) ? s_frag[mma_kv][reg_id] : (KTraits::MaskFillValue);
      }
    }
  } else {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
        const uint32_t q_idx = q[(reg_id % 4) / 2], kv_idx = kv_idx_base + mma_kv * 16 +
                                                             2 * (lane_idx % 4) + 8 * (reg_id / 4) +
                                                             reg_id % 2;
        const bool mask =
            (!(KTraits::CAUSAL ? (kv_idx + qo_len > kv_len + q_idx || (kv_idx >= kv_end))
                               : kv_idx >= kv_end));
        s_frag[mma_kv][reg_id] = (mask) ? s_frag[mma_kv][reg_id] : (KTraits::MaskFillValue);
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void update_mdo_states_(typename KTraits::SharedStorage* smem_storage,
                                                   const uint32_t stage_idx,
                                                   typename KTraits::AttentionVariant variant,
                                                   typename KTraits::DTypeQKAccum (*s_frag)[8],
                                                   float (*o_frag)[8],
                                                   typename KTraits::DTypeQKAccum* m, float* d) {
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  using AttentionVariant = typename KTraits::AttentionVariant;
  const float sm_scale = variant.sm_scale_log2;
  const uint32_t warpgroup_idx = threadIdx.z, lane_idx = threadIdx.x, warp_idx_in_wg = threadIdx.y;
  float m_prev[2];
  if constexpr (KTraits::QK_SHARD) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      m_prev[j] = m[j];
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
        float m_local = max(max(s_frag[mma_kv][j * 2 + 0], s_frag[mma_kv][j * 2 + 1]),
                            max(s_frag[mma_kv][j * 2 + 4], s_frag[mma_kv][j * 2 + 5]));
        m[j] = max(m[j], m_local);
      }
      m[j] = max(m[j], math::shfl_xor_sync(m[j], 0x2));
      m[j] = max(m[j], math::shfl_xor_sync(m[j], 0x1));
      if (lane_idx % 4 == 0) {
        smem_storage->m_wg[warpgroup_idx][warp_idx_in_wg * 16 + j * 8 + lane_idx / 4] = m[j];
      }
    }

    __syncthreads();

#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      m[j] = max(smem_storage->m_wg[0][warp_idx_in_wg * 16 + j * 8 + lane_idx / 4],
                 smem_storage->m_wg[1][warp_idx_in_wg * 16 + j * 8 + lane_idx / 4]);
      float o_scale = math::ptx_exp2(m_prev[j] * sm_scale - m[j] * sm_scale);
      d[j] *= o_scale;
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 2; ++mma_d) {
        o_frag[mma_d][j * 2 + 0] *= o_scale;
        o_frag[mma_d][j * 2 + 1] *= o_scale;
        o_frag[mma_d][j * 2 + 4] *= o_scale;
        o_frag[mma_d][j * 2 + 5] *= o_scale;
      }
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
        s_frag[mma_kv][j * 2 + 0] =
            math::ptx_exp2(s_frag[mma_kv][j * 2 + 0] * sm_scale - m[j] * sm_scale);
        s_frag[mma_kv][j * 2 + 1] =
            math::ptx_exp2(s_frag[mma_kv][j * 2 + 1] * sm_scale - m[j] * sm_scale);
        s_frag[mma_kv][j * 2 + 4] =
            math::ptx_exp2(s_frag[mma_kv][j * 2 + 4] * sm_scale - m[j] * sm_scale);
        s_frag[mma_kv][j * 2 + 5] =
            math::ptx_exp2(s_frag[mma_kv][j * 2 + 5] * sm_scale - m[j] * sm_scale);
      }
    }
  } else {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      m_prev[j] = m[j];
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        float m_local = max(max(s_frag[mma_kv][j * 2 + 0], s_frag[mma_kv][j * 2 + 1]),
                            max(s_frag[mma_kv][j * 2 + 4], s_frag[mma_kv][j * 2 + 5]));
        m[j] = max(m[j], m_local);
      }
      m[j] = max(m[j], math::shfl_xor_sync(m[j], 0x2));
      m[j] = max(m[j], math::shfl_xor_sync(m[j], 0x1));
    }

#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      float o_scale = math::ptx_exp2(m_prev[j] * sm_scale - m[j] * sm_scale);
      d[j] *= o_scale;
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 2; ++mma_d) {
        o_frag[mma_d][j * 2 + 0] *= o_scale;
        o_frag[mma_d][j * 2 + 1] *= o_scale;
        o_frag[mma_d][j * 2 + 4] *= o_scale;
        o_frag[mma_d][j * 2 + 5] *= o_scale;
      }
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        s_frag[mma_kv][j * 2 + 0] =
            math::ptx_exp2(s_frag[mma_kv][j * 2 + 0] * sm_scale - m[j] * sm_scale);
        s_frag[mma_kv][j * 2 + 1] =
            math::ptx_exp2(s_frag[mma_kv][j * 2 + 1] * sm_scale - m[j] * sm_scale);
        s_frag[mma_kv][j * 2 + 4] =
            math::ptx_exp2(s_frag[mma_kv][j * 2 + 4] * sm_scale - m[j] * sm_scale);
        s_frag[mma_kv][j * 2 + 5] =
            math::ptx_exp2(s_frag[mma_kv][j * 2 + 5] * sm_scale - m[j] * sm_scale);
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void compute_mla_qk(typename KTraits::SharedStorage* smem_storage,
                                               const uint32_t stage_idx,
                                               typename KTraits::DTypeQKAccum (*s_frag)[8]) {
  constexpr uint32_t UPCAST_STRIDE_Q_NOPE = KTraits::UPCAST_STRIDE_Q_NOPE;
  constexpr uint32_t UPCAST_STRIDE_Q_PE = KTraits::UPCAST_STRIDE_Q_PE;
  constexpr uint32_t UPCAST_STRIDE_CKV = KTraits::UPCAST_STRIDE_CKV;
  constexpr uint32_t UPCAST_STRIDE_KPE = KTraits::UPCAST_STRIDE_KPE;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> q_smem_nope(smem_storage->q_smem_nope);
  smem_t<KTraits::SWIZZLE_MODE_Q_PE> q_smem_pe(smem_storage->q_smem_pe);
  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[stage_idx]);
  smem_t<KTraits::SWIZZLE_MODE_KPE> kpe_smem(smem_storage->kpe_p_smem[stage_idx]);
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  compute_qk_</*init=*/true, KTraits, KTraits::NUM_MMA_D_KPE, KTraits::UPCAST_STRIDE_Q_PE,
              KTraits::UPCAST_STRIDE_KPE>(q_smem_pe, kpe_smem, s_frag);
  compute_qk_</*init=*/false, KTraits, KTraits::NUM_MMA_D_CKV, KTraits::UPCAST_STRIDE_Q_NOPE,
              KTraits::UPCAST_STRIDE_CKV>(q_smem_nope, ckv_smem, s_frag);
}

template <typename KTraits>
__device__ __forceinline__ void compute_mla_pv(typename KTraits::SharedStorage* smem_storage,
                                               const uint32_t stage_idx,
                                               typename KTraits::DTypeQKAccum (*s_frag)[8],
                                               typename KTraits::DTypeQKAccum* d,
                                               float (*o_frag)[8]) {
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t UPCAST_STRIDE_CKV = KTraits::UPCAST_STRIDE_CKV;
  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[stage_idx]);
  uint32_t ckv_smem_offset_r = ckv_smem.template get_permuted_offset<UPCAST_STRIDE_CKV>(
      lane_idx % 16, warpgroup_idx * NUM_MMA_D_CKV + lane_idx / 16);
  if constexpr (KTraits::QK_SHARD) {
    // shard s_frag computation on KV dimension across warpgroups, need allgather
    alignas(16) typename KTraits::DTypeKV p_f16[NUM_MMA_KV / 2][8];
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV / 2; ++mma_kv) {
      vec_cast<typename KTraits::DTypeKV, float>::cast<8>(p_f16[mma_kv], s_frag[mma_kv]);
      mma::m16k16_rowsum_f16f16f32(d, p_f16[mma_kv]);
    }

    __syncthreads();
    smem_t<KTraits::SWIZZLE_MODE_P> p_smem(smem_storage->kpe_p_smem[stage_idx]);
    constexpr uint32_t UPCAST_STRIDE_P = KTraits::UPCAST_STRIDE_P;
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV / 2; ++mma_kv) {
#ifdef FLASHINFER_STMATRIX_M8N8X4_ENABLED
      uint32_t p_smem_offset_w = p_smem.template get_permuted_offset<UPCAST_STRIDE_P>(
          warp_idx_in_wg * 16 + lane_idx % 16,
          warpgroup_idx * NUM_MMA_KV + mma_kv * 2 + lane_idx / 16);
      p_smem.stmatrix_m8n8x4(p_smem_offset_w, (uint32_t*)p_f16[mma_kv]);
#else
      uint32_t p_smem_offset_w = p_smem.template get_permuted_offset<UPCAST_STRIDE_P>(
          warp_idx_in_wg * 16 + lane_idx / 4, warpgroup_idx * NUM_MMA_KV + mma_kv * 2);
      ((uint32_t*)(p_smem.base + p_smem_offset_w))[lane_idx % 4] = *(uint32_t*)&p_f16[mma_kv][0];
      ((uint32_t*)(p_smem.base + p_smem_offset_w + 8 * UPCAST_STRIDE_P))[lane_idx % 4] =
          *(uint32_t*)&p_f16[mma_kv][2];
      ((uint32_t*)(p_smem.base + (p_smem_offset_w ^ 0x1)))[lane_idx % 4] =
          *(uint32_t*)&p_f16[mma_kv][4];
      ((uint32_t*)(p_smem.base + (p_smem_offset_w ^ 0x1) + 8 * UPCAST_STRIDE_P))[lane_idx % 4] =
          *(uint32_t*)&p_f16[mma_kv][6];
#endif
    }
    uint32_t p_smem_offset_r = p_smem.template get_permuted_offset<UPCAST_STRIDE_P>(
        warp_idx_in_wg * 16 + lane_idx % 16, lane_idx / 16);

    // wait for p_smem to be filled
    __syncthreads();

#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
      uint32_t p_frag[4];
      p_smem.ldmatrix_m8n8x4(p_smem_offset_r, p_frag);
      p_smem_offset_r = p_smem.template advance_offset_by_column<2>(p_smem_offset_r, mma_kv);

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 2; ++mma_d) {
        uint32_t v_frag[4];
        ckv_smem.ldmatrix_m8n8x4_trans(ckv_smem_offset_r, v_frag);
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(o_frag[mma_d], p_frag,
                                                                             v_frag);
        ckv_smem_offset_r = ckv_smem.template advance_offset_by_column<2>(ckv_smem_offset_r, mma_d);
      }
      ckv_smem_offset_r =
          ckv_smem.template advance_offset_by_row<16, UPCAST_STRIDE_CKV>(ckv_smem_offset_r) -
          NUM_MMA_D_CKV;
    }
  } else {
    // no need to store p_smem because all warpgroups are working on the same p
    alignas(16) typename KTraits::DTypeKV p_f16[NUM_MMA_KV][8];
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
      vec_cast<typename KTraits::DTypeKV, float>::cast<8>(p_f16[mma_kv], s_frag[mma_kv]);
      mma::m16k16_rowsum_f16f16f32(d, p_f16[mma_kv]);
    }
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 2; ++mma_d) {
        uint32_t v_frag[4];
        ckv_smem.ldmatrix_m8n8x4_trans(ckv_smem_offset_r, v_frag);
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(
            o_frag[mma_d], (uint32_t*)p_f16[mma_kv], v_frag);
        ckv_smem_offset_r = ckv_smem.template advance_offset_by_column<2>(ckv_smem_offset_r, mma_d);
      }
      ckv_smem_offset_r =
          ckv_smem.template advance_offset_by_row<16, UPCAST_STRIDE_CKV>(ckv_smem_offset_r) -
          NUM_MMA_D_CKV;
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void normalize_d_(typename KTraits::SharedStorage* smem_storage,
                                             const uint32_t stage_idx, float (*o_frag)[8],
                                             typename KTraits::DTypeQKAccum* m, float* d) {
  const uint32_t warpgroup_idx = threadIdx.z, lane_idx = threadIdx.x, warp_idx_in_wg = threadIdx.y;
  if constexpr (KTraits::QK_SHARD) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      if (lane_idx % 4 == 0) {
        smem_storage->d_wg[warpgroup_idx][warp_idx_in_wg * 16 + j * 8 + lane_idx / 4] = d[j];
      }
    }
    __syncthreads();
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      d[j] = smem_storage->d_wg[0][warp_idx_in_wg * 16 + j * 8 + lane_idx / 4] +
             smem_storage->d_wg[1][warp_idx_in_wg * 16 + j * 8 + lane_idx / 4];
    }
  }

  float d_rcp[2];
  // compute reciprocal of d
#pragma unroll
  for (uint32_t j = 0; j < 2; ++j) {
    d_rcp[j] = (m[j] != typename KTraits::DTypeQKAccum(-math::inf)) ? math::ptx_rcp(d[j]) : 0.f;
  }

#pragma unroll
  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 2; ++mma_d) {
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
      o_frag[mma_d][reg_id] = o_frag[mma_d][reg_id] * d_rcp[(reg_id % 4) / 2];
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void finalize_m_(typename KTraits::AttentionVariant variant,
                                            typename KTraits::DTypeQKAccum* m) {
  if constexpr (variant.use_softmax) {
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      if (m[j] != typename KTraits::DTypeQKAccum(-math::inf)) {
        m[j] *= variant.sm_scale_log2;
      }
    }
  }
}

template <typename KTraits>
__device__ void DevicePersistentMergeStates(
    typename KTraits::IdType* merge_packed_offset_start,
    typename KTraits::IdType* merge_packed_offset_end,
    typename KTraits::IdType* merge_partial_packed_offset_start,
    typename KTraits::IdType* merge_partial_packed_offset_end,
    typename KTraits::IdType* merge_partial_stride, typename KTraits::DTypeO* partial_o,
    float* partial_lse, typename KTraits::DTypeO* final_o, float* final_lse,
    const uint32_t o_stride_n, const uint32_t o_stride_h, const uint_fastdiv& num_heads,
    const bool& return_lse_base_on_e) {
  constexpr uint32_t VEC_SIZE = 8;  // partial o has data type float
  constexpr uint32_t NUM_THRS_PER_ROW = KTraits::HEAD_DIM_CKV / VEC_SIZE;
  constexpr uint32_t ROWS_PER_ITERATION = (KTraits::NUM_THREADS) / NUM_THRS_PER_ROW;
  const uint32_t cta_idx = (gridDim.x * blockIdx.y + blockIdx.x);
  const uint32_t thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t offset_start = merge_packed_offset_start[cta_idx];
  const uint32_t len = merge_packed_offset_end[cta_idx] - offset_start;
  const uint32_t partial_offset_start = merge_partial_packed_offset_start[cta_idx];
  const uint32_t partial_offset_end = merge_partial_packed_offset_end[cta_idx];
  const uint32_t stride = merge_partial_stride[cta_idx];
#pragma unroll 1
  for (uint32_t local_packed_offset = thread_id / NUM_THRS_PER_ROW; local_packed_offset < len;
       local_packed_offset += ROWS_PER_ITERATION) {
    uint32_t final_packed_offset = offset_start + local_packed_offset;
    uint32_t q, r;
    num_heads.divmod(final_packed_offset, q, r);
    state_t<VEC_SIZE> st;
#pragma unroll 8
    for (uint32_t partial_packed_offset = partial_offset_start + local_packed_offset;
         partial_packed_offset < partial_offset_end; partial_packed_offset += stride) {
      vec_t<float, VEC_SIZE> o_partial;
      float lse_partial;
      o_partial.cast_load(partial_o + partial_packed_offset * KTraits::HEAD_DIM_CKV +
                          (thread_id % NUM_THRS_PER_ROW) * VEC_SIZE);
      lse_partial = partial_lse[partial_packed_offset];
      st.merge(o_partial, lse_partial, 1);
    }
    st.normalize();
    st.o.cast_store(final_o +
                    (q * o_stride_n + r * o_stride_h + (thread_id % NUM_THRS_PER_ROW) * VEC_SIZE));
    if (final_lse) {
      final_lse[q * num_heads + r] = st.get_lse();
      if (return_lse_base_on_e) {
        final_lse[q * num_heads + r] *= math::loge2;
      }
    }
  }
}

template <typename KTraits, bool SYNC_WARP = false>
__device__ __forceinline__ void write_o(typename KTraits::SharedStorage* smem_storage,
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
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
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

template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLAPagedAttentionKernel(
    const __grid_constant__ Params params) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;

  extern __shared__ __align__(alignof(typename KTraits::SharedStorage)) uint8_t smem[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);

  typename KTraits::AttentionVariant variant(params, blockIdx.y, smem);

  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q_NOPE = KTraits::SWIZZLE_MODE_Q_NOPE;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q_PE = KTraits::SWIZZLE_MODE_Q_PE;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_CKV = KTraits::SWIZZLE_MODE_CKV;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KPE = KTraits::SWIZZLE_MODE_KPE;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  [[maybe_unused]] constexpr int32_t NUM_STAGES = KTraits::NUM_STAGES;
  [[maybe_unused]] constexpr bool CAUSAL = KTraits::CAUSAL;

  DTypeQ* q_nope = params.q_nope;
  DTypeQ* q_pe = params.q_pe;
  DTypeKV* ckv = params.ckv;
  DTypeKV* kpe = params.kpe;
  IdType* kv_indices = params.kv_indices;
  DTypeO* partial_o = params.partial_o;
  float* partial_lse = params.partial_lse;
  DTypeO* final_o = params.final_o;
  float* final_lse = params.final_lse;
  IdType* work_indptr = params.work_indptr;

  float s_frag[KTraits::QK_SHARD ? NUM_MMA_KV / 2 : NUM_MMA_KV][8];
  alignas(16) float o_frag[NUM_MMA_D_CKV / 2][8];
  float m[2];
  float d[2];

  const uint_fastdiv& num_heads = params.num_heads;
  const uint_fastdiv& block_size = params.block_size;
  const uint32_t q_nope_stride_n = params.q_nope_stride_n;
  const uint32_t q_nope_stride_h = params.q_nope_stride_h;
  const uint32_t q_pe_stride_n = params.q_pe_stride_n;
  const uint32_t q_pe_stride_h = params.q_pe_stride_h;
  const uint32_t ckv_stride_page = params.ckv_stride_page;
  const uint32_t ckv_stride_n = params.ckv_stride_n;
  const uint32_t kpe_stride_page = params.kpe_stride_page;
  const uint32_t kpe_stride_n = params.kpe_stride_n;
  const uint32_t o_stride_n = params.o_stride_n;
  const uint32_t o_stride_h = params.o_stride_h;
  const uint32_t cluster_tile_q = gridDim.x * KTraits::CTA_TILE_Q;

#pragma unroll 1
  for (IdType work_idx = work_indptr[blockIdx.y]; work_idx < work_indptr[blockIdx.y + 1];
       ++work_idx) {
    const uint32_t q_indptr = params.q_indptr[work_idx];
    const uint32_t kv_indptr = params.kv_indptr[work_idx];
    const int32_t partial_indptr = params.partial_indptr[work_idx];
    const uint32_t q_len = params.q_len[work_idx];
    const uint32_t kv_len = params.kv_len[work_idx];
    const uint32_t packed_qo_start = params.q_start[work_idx];
    const uint32_t kv_start = params.kv_start[work_idx];
    const uint32_t kv_end = params.kv_end[work_idx];

    const uint32_t qo_packed_idx_base = packed_qo_start + blockIdx.x * KTraits::CTA_TILE_Q;
    const uint32_t qo_upperbound =
        min(q_len, ceil_div(qo_packed_idx_base + KTraits::CTA_TILE_Q, num_heads));

    init_states_<KTraits>(o_frag, m, d);

    __syncthreads();
    load_q<KTraits>(&smem_storage, q_nope + q_indptr * q_nope_stride_n,
                    q_pe + q_indptr * q_pe_stride_n, q_nope_stride_n, q_nope_stride_h,
                    q_pe_stride_n, q_pe_stride_h, qo_upperbound, qo_packed_idx_base,
                    params.num_heads);

    int kv_tile_idx =
        ceil_div(
            (CAUSAL ? min(kv_end, kv_len - q_len + (packed_qo_start + cluster_tile_q) / num_heads)
                    : kv_end),
            CTA_TILE_KV) -
        1 - (kv_start / CTA_TILE_KV);

    int mask_tile_idx =
        (CAUSAL ? min(kv_end, kv_len - q_len + packed_qo_start / num_heads) : kv_end) /
            CTA_TILE_KV -
        (kv_start / CTA_TILE_KV);

    uint32_t block_iter_base = kv_indptr * block_size + kv_start;
    // last kv tile
    __syncthreads();
    uint32_t packed_kv_bound = kv_indptr * block_size + kv_len;
    load_kv<KTraits>(&smem_storage, ckv, kpe, kv_indices, ckv_stride_n, ckv_stride_page,
                     kpe_stride_n, kpe_stride_page, packed_kv_bound,
                     block_iter_base + kv_tile_idx * CTA_TILE_KV, block_size,
                     kv_tile_idx % NUM_STAGES);
    cp_async::commit_group();
#pragma unroll
    for (int stage_idx = 1; stage_idx < NUM_STAGES; ++stage_idx) {
      if (kv_tile_idx - stage_idx >= 0) {
        load_kv<KTraits>(&smem_storage, ckv, kpe, kv_indices, ckv_stride_n, ckv_stride_page,
                         kpe_stride_n, kpe_stride_page, packed_kv_bound,
                         block_iter_base + (kv_tile_idx - stage_idx) * CTA_TILE_KV, block_size,
                         (kv_tile_idx - stage_idx) % NUM_STAGES);
        cp_async::commit_group();
      }
    }

    // loop with mask
#pragma unroll 1
    for (; kv_tile_idx >= mask_tile_idx && kv_tile_idx > 0; --kv_tile_idx) {
      cp_async::wait_group<NUM_STAGES - 1>();
      __syncthreads();

      // compute mla qk
      compute_mla_qk<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag);

      // logits mask
      logits_mask_<KTraits>(qo_packed_idx_base, kv_start + kv_tile_idx * CTA_TILE_KV, q_len, kv_len,
                            kv_end, num_heads, s_frag);

      // compute m,d states in online softmax
      update_mdo_states_<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, variant, s_frag, o_frag,
                                  m, d);

      // compute sfm * v
      compute_mla_pv<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag, d, o_frag);

      if (kv_tile_idx - NUM_STAGES >= 0) {
        __syncthreads();
        load_kv<KTraits>(&smem_storage, ckv, kpe, kv_indices, ckv_stride_n, ckv_stride_page,
                         kpe_stride_n, kpe_stride_page, packed_kv_bound,
                         block_iter_base + (kv_tile_idx - NUM_STAGES) * CTA_TILE_KV, block_size,
                         (kv_tile_idx - NUM_STAGES) % NUM_STAGES);
        cp_async::commit_group();
      }
    }

    // loop without mask
#pragma unroll 1
    for (; kv_tile_idx + 1 > NUM_STAGES; --kv_tile_idx) {
      cp_async::wait_group<NUM_STAGES - 1>();
      __syncthreads();

      // compute mla qk
      compute_mla_qk<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag);

      // compute m,d states in online softmax
      update_mdo_states_<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, variant, s_frag, o_frag,
                                  m, d);
      // compute sfm * v
      compute_mla_pv<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag, d, o_frag);

      __syncthreads();
      load_kv<KTraits>(&smem_storage, ckv, kpe, kv_indices, ckv_stride_n, ckv_stride_page,
                       kpe_stride_n, kpe_stride_page, packed_kv_bound,
                       block_iter_base + (kv_tile_idx - NUM_STAGES) * CTA_TILE_KV, block_size,
                       (kv_tile_idx - NUM_STAGES) % NUM_STAGES);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    __syncthreads();

    // last tiles
#pragma unroll
    for (; kv_tile_idx >= 0; --kv_tile_idx) {
      // compute mla qk
      compute_mla_qk<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag);

      logits_mask_<KTraits>(qo_packed_idx_base, kv_start + kv_tile_idx * CTA_TILE_KV, q_len, kv_len,
                            kv_end, num_heads, s_frag);

      // compute m,d states in online softmax
      update_mdo_states_<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, variant, s_frag, o_frag,
                                  m, d);

      // compute sfm * v
      compute_mla_pv<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag, d, o_frag);
    }

    __syncthreads();

    // normalize and write back
    normalize_d_<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, o_frag, m, d);

    finalize_m_<KTraits>(variant, m);

    write_o<KTraits>(
        &smem_storage, final_o + q_indptr * o_stride_n,
        final_lse ? final_lse + q_indptr * num_heads : nullptr,
        (partial_indptr == -1) ? nullptr : partial_o + partial_indptr * KTraits::HEAD_DIM_CKV,
        (partial_indptr == -1) ? nullptr : partial_lse + partial_indptr, o_frag, m, d, o_stride_n,
        o_stride_h, qo_upperbound, qo_packed_idx_base, num_heads, params.return_lse_base_on_e);
  }

  auto grid = cg::this_grid();
  grid.sync();

  // the second stage, merge partial outputs
  DevicePersistentMergeStates<KTraits>(
      params.merge_packed_offset_start, params.merge_packed_offset_end,
      params.merge_partial_packed_offset_start, params.merge_partial_packed_offset_end,
      params.merge_partial_stride, partial_o, partial_lse, final_o, final_lse, o_stride_n,
      o_stride_h, num_heads, params.return_lse_base_on_e);
}


template <typename Params>
__device__ __forceinline__ void record_overlap_phase(const Params& params, uint32_t slot,
                                                     bool producer = false) {
#ifdef FLASHINFER_MLA_HOST_TRACE
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == uint32_t(producer)) {
    unsigned long long time;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time));
    params.host_prefetch_state[3608 + blockIdx.y * 4 + slot] = time;
  }
#endif
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void run_host_fetch_warps(const Params& params,
                                                     unsigned long long epoch) {
  const uint32_t lane = threadIdx.x;
  const uint32_t rank = blockIdx.y - 44;
  const uint32_t warp = threadIdx.y;
  extern __shared__ uint8_t agent_smem[];
  auto* selected_pages = reinterpret_cast<uint32_t*>(agent_smem + 65536);
  auto* selected_positions = selected_pages + 32;
  auto* warp_counts = selected_positions + 32;
  typename KTraits::IdType pages[16];
  uint32_t preceding[16], masks[16], count = 0;
#pragma unroll
  for (uint32_t i = 0; i < 16; ++i) {
    pages[i] = params.kv_indices[warp * 512 + i * 32 + lane];
    preceding[i] = count;
    masks[i] = __ballot_sync(0xffffffff, pages[i] >= params.host_page_start);
    count += __popc(masks[i]);
  }
  if (lane == 0) warp_counts[warp] = count;
  asm volatile("bar.sync 2, 128;" ::: "memory");
  uint32_t prefix = 0, total = 0;
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    if (i < warp) prefix += warp_counts[i];
    total += warp_counts[i];
  }
#pragma unroll
  for (uint32_t i = 0; i < 16; ++i) {
    const uint32_t host_rank = prefix + preceding[i] + __popc(masks[i] & ((1U << lane) - 1));
    if (pages[i] >= params.host_page_start && host_rank % 64 == rank) {
      selected_pages[host_rank / 64] = pages[i];
      selected_positions[host_rank / 64] = warp * 512 + i * 32 + lane;
    }
  }
  asm volatile("bar.sync 2, 128;" ::: "memory");
  const uint32_t selected = (total + 63 - rank) / 64;
  record_overlap_phase(params, 1, true);
  uint32_t positions[8];
  uint4 first[8], second[8], rope[8];
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const uint32_t slot = i * 4 + warp;
    positions[i] = slot < selected ? selected_positions[slot] : 2048;
    if (positions[i] < 2048) {
      const auto page = selected_pages[slot];
      const auto* source = params.ckv + page * params.ckv_stride_page;
      asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(first[i].x), "=r"(first[i].y), "=r"(first[i].z), "=r"(first[i].w)
                   : "l"(source + lane * 8)
                   : "memory");
      asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(second[i].x), "=r"(second[i].y), "=r"(second[i].z), "=r"(second[i].w)
                   : "l"(source + 256 + lane * 8)
                   : "memory");
      if (lane < 8) {
        asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                     : "=r"(rope[i].x), "=r"(rope[i].y), "=r"(rope[i].z), "=r"(rope[i].w)
                     : "l"(params.kpe + page * params.kpe_stride_page + lane * 8)
                     : "memory");
      }
    }
  }
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    if (positions[i] < 2048) {
      auto* destination = params.host_prefetch_buffer + positions[i] * 576;
      *reinterpret_cast<uint4*>(destination + lane * 8) = first[i];
      *reinterpret_cast<uint4*>(destination + 256 + lane * 8) = second[i];
      if (lane < 8) *reinterpret_cast<uint4*>(destination + 512 + lane * 8) = rope[i];
    }
  }
  __threadfence();
  __syncwarp();
  if (lane == 0) {
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      if (positions[i] < 2048) {
        asm volatile("st.release.gpu.global.u64 [%0], %1;"
                     :
                     : "l"(params.host_prefetch_state + gridDim.y + positions[i]), "l"(epoch)
                     : "memory");
      }
    }
  }
  record_overlap_phase(params, 2, true);
}

template <typename BaseTraits>
// Shared index compaction for one-query, 64-head, top-2048 decode.
struct HostSplitKernelTraits : BaseTraits {
  struct SharedStorage : BaseTraits::SharedStorage {
    typename BaseTraits::IdType indices[128];
    uint32_t counts[8];
    uint32_t count;
    unsigned long long epoch;
  };
};

template <typename KTraits, uint32_t HOST_CHUNKS = 28, bool RAW_HOST_PAGES = false,
          uint32_t LOCAL_CHUNKS = 16, typename Params>
__device__ __forceinline__ uint32_t prepare_host_split(typename KTraits::SharedStorage* storage,
                                                       const Params& params) {
  const uint32_t lane = threadIdx.x;
  const uint32_t warp = threadIdx.z * 4 + threadIdx.y;
  const bool host = blockIdx.y >= LOCAL_CHUNKS;
  const uint32_t chunk = host ? blockIdx.y - LOCAL_CHUNKS : blockIdx.y;
  const uint32_t chunks = host ? HOST_CHUNKS : LOCAL_CHUNKS;
  typename KTraits::IdType pages[8];
  uint32_t preceding[8], masks[8], count = 0;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    pages[i] = params.kv_indices[warp * 256 + i * 32 + lane];
    preceding[i] = count;
    masks[i] = __ballot_sync(0xffffffff, pages[i] >= params.host_page_start);
    count += __popc(masks[i]);
  }
  if (lane == 0) storage->counts[warp] = count;
  __syncthreads();
  uint32_t prefix = 0, total_host = 0;
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    if (i < warp) prefix += storage->counts[i];
    total_host += storage->counts[i];
  }
#pragma unroll
  for (uint32_t i = 0; i < 8; ++i) {
    const uint32_t position = warp * 256 + i * 32 + lane;
    const uint32_t host_rank = prefix + preceding[i] + __popc(masks[i] & ((1U << lane) - 1));
    const uint32_t rank = host ? host_rank : position - host_rank;
    if ((pages[i] >= params.host_page_start) == host && rank % chunks == chunk) {
      storage->indices[rank / chunks] = host && !RAW_HOST_PAGES ? position : pages[i];
    }
  }
  const uint32_t length = ((host ? total_host : 2048 - total_host) + chunks - 1 - chunk) / chunks;
  if (lane == 0 && warp == 0) storage->count = length;
  __syncthreads();
  return length;
}

template <typename BaseTraits>
struct HostNarrowKernelTraits : BaseTraits {
  static constexpr bool QK_SHARD = false;
  static constexpr uint32_t CTA_TILE_KV = 16;
  static constexpr uint32_t NUM_MMA_KV = 1;
  static constexpr uint32_t UPCAST_STRIDE_P = 2;
};

template <uint32_t PASS, typename KTraits>
__device__ __forceinline__ void compute_small_host_pv(typename KTraits::SharedStorage* storage,
                                                      float (*s_frag)[8], float* d,
                                                      float (*o_frag)[8]) {
  const uint32_t lane = threadIdx.x;
  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(storage->ckv_smem[0]);
  uint32_t offset = ckv_smem.template get_permuted_offset<KTraits::UPCAST_STRIDE_CKV>(
      lane % 16, PASS * KTraits::NUM_MMA_D_CKV + lane / 16);
  alignas(16) typename KTraits::DTypeKV p[8];
  vec_cast<typename KTraits::DTypeKV, float>::template cast<8>(p, s_frag[0]);
  if constexpr (PASS == 0) mma::m16k16_rowsum_f16f16f32(d, p);
#pragma unroll
  for (uint32_t feature = 0; feature < 16; ++feature) {
    uint32_t value[4];
    ckv_smem.ldmatrix_m8n8x4_trans(offset, value);
    mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(
        o_frag[feature], reinterpret_cast<uint32_t*>(p), value);
    offset = ckv_smem.template advance_offset_by_column<2>(offset, feature);
  }
}

template <uint32_t BEGIN, uint32_t END, bool ROPE, typename KTraits, typename Params>
__device__ __forceinline__ void load_private_host_segment(
    typename KTraits::SharedStorage& storage, const Params& params, uint32_t length) {
  const uint32_t lane = threadIdx.x;
  const uint32_t group = threadIdx.z;
  if constexpr (!KTraits::QK_SHARD) {
    if (group != 0) return;
  }
  const uint32_t row = (KTraits::QK_SHARD ? group * 16 : 0) + threadIdx.y * 4 + lane / 8;
  const uint32_t page = row < length ? storage.indices[row] : 0;
  if constexpr (ROPE) {
    smem_t<KTraits::SWIZZLE_MODE_KPE> shared(storage.kpe_p_smem[0]);
    auto* source = params.kpe + page * params.kpe_stride_page + (lane % 8) * 8;
    const uint32_t offset = shared.template get_permuted_offset<KTraits::UPCAST_STRIDE_KPE>(
        row, lane % 8);
    shared.template load_128b_async<SharedMemFillMode::kFillZero>(offset, source, row < length);
  } else {
    smem_t<KTraits::SWIZZLE_MODE_CKV> shared(storage.ckv_smem[0]);
#pragma unroll
    for (uint32_t feature = BEGIN; feature < END; ++feature) {
      auto* source = params.ckv + page * params.ckv_stride_page + feature * 64 + (lane % 8) * 8;
      const uint32_t offset = shared.template get_permuted_offset<KTraits::UPCAST_STRIDE_CKV>(
          row, feature * 8 + lane % 8);
      shared.template load_128b_async<SharedMemFillMode::kFillZero>(offset, source, row < length);
    }
  }
}

template <bool HOST, bool SINGLE_TILE, bool DIRECT_HOST, typename KTraits, bool PIPELINE = false,
          typename Params>
__device__ __forceinline__ void compute_host_split(typename KTraits::SharedStorage& storage,
                                                   Params params, uint32_t length, uint8_t* smem) {
  load_q<KTraits>(&storage, params.q_nope, params.q_pe, params.q_nope_stride_n,
                  params.q_nope_stride_h, params.q_pe_stride_n, params.q_pe_stride_h, 1, 0,
                  params.num_heads);
  cp_async::commit_group();
  if constexpr (HOST && !DIRECT_HOST) {
    const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    if (tid < length) {
      unsigned long long ready;
      do {
        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];"
                     : "=l"(ready)
                     : "l"(params.host_prefetch_state + gridDim.y + storage.indices[tid])
                     : "memory");
        if (ready != storage.epoch) __nanosleep(32);
      } while (ready != storage.epoch);
    }
    __syncthreads();
    params.ckv = params.host_prefetch_buffer;
    params.kpe = params.host_prefetch_buffer + 512;
    params.ckv_stride_page = 576;
    params.kpe_stride_page = 576;
  }
  if constexpr (!DIRECT_HOST) record_overlap_phase(params, 2);
  float s_frag[KTraits::QK_SHARD ? KTraits::NUM_MMA_KV / 2 : KTraits::NUM_MMA_KV][8], o_frag[16][8],
      m[2], d[2];
  init_states_<KTraits>(o_frag, m, d);
  typename KTraits::AttentionVariant variant(params, 0, smem);
  int tile = SINGLE_TILE ? 0 : int(ceil_div(length, KTraits::CTA_TILE_KV)) - 1;
  if constexpr (PIPELINE) {
    static_assert(HOST && DIRECT_HOST && SINGLE_TILE);
    load_private_host_segment<0, 4, false, KTraits>(storage, params, length);
    cp_async::commit_group();
    load_private_host_segment<4, 8, false, KTraits>(storage, params, length);
    cp_async::commit_group();
    load_private_host_segment<0, 1, true, KTraits>(storage, params, length);
    cp_async::commit_group();
    smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> query(storage.q_smem_nope);
    smem_t<KTraits::SWIZZLE_MODE_CKV> key(storage.ckv_smem[0]);
    cp_async::wait_group<2>();
    __syncthreads();
    if (KTraits::QK_SHARD || threadIdx.z == 0)
      compute_qk_<true, KTraits, 16, KTraits::UPCAST_STRIDE_Q_NOPE,
                  KTraits::UPCAST_STRIDE_CKV, KTraits::SWIZZLE_MODE_Q_NOPE,
                  KTraits::SWIZZLE_MODE_CKV>(query, key, s_frag);
    cp_async::wait_group<1>();
    __syncthreads();
    if (KTraits::QK_SHARD || threadIdx.z == 0)
      compute_qk_<false, KTraits, 16, KTraits::UPCAST_STRIDE_Q_NOPE,
                  KTraits::UPCAST_STRIDE_CKV, KTraits::SWIZZLE_MODE_Q_NOPE,
                  KTraits::SWIZZLE_MODE_CKV, 16>(query, key, s_frag);
    cp_async::wait_group<0>();
    __syncthreads();
    if (KTraits::QK_SHARD || threadIdx.z == 0) {
      smem_t<KTraits::SWIZZLE_MODE_Q_PE> rope_query(storage.q_smem_pe);
      smem_t<KTraits::SWIZZLE_MODE_KPE> rope_key(storage.kpe_p_smem[0]);
      compute_qk_<false, KTraits, KTraits::NUM_MMA_D_KPE, KTraits::UPCAST_STRIDE_Q_PE,
                  KTraits::UPCAST_STRIDE_KPE, KTraits::SWIZZLE_MODE_Q_PE,
                  KTraits::SWIZZLE_MODE_KPE>(rope_query, rope_key, s_frag);
    }
  } else {
    load_kv<KTraits>(&storage, params.ckv, params.kpe, storage.indices, params.ckv_stride_n,
                    params.ckv_stride_page, params.kpe_stride_n, params.kpe_stride_page, length,
                    tile * KTraits::CTA_TILE_KV, params.block_size, tile % 2);
    cp_async::commit_group();
    if (tile > 0) {
      load_kv<KTraits>(&storage, params.ckv, params.kpe, storage.indices, params.ckv_stride_n,
                      params.ckv_stride_page, params.kpe_stride_n, params.kpe_stride_page, length,
                      (tile - 1) * KTraits::CTA_TILE_KV, params.block_size, (tile - 1) % 2);
      cp_async::commit_group();
    }
    cp_async::wait_group<0>();
    __syncthreads();
  }
  if constexpr (DIRECT_HOST) record_overlap_phase(params, 2);
  if constexpr (!KTraits::QK_SHARD) {
    if (length <= 16) {
      if (threadIdx.z == 0) {
        if constexpr (!PIPELINE) compute_mla_qk<KTraits>(&storage, 0, s_frag);
        logits_mask_<KTraits>(0, 0, 1, length, length, params.num_heads, s_frag);
        update_mdo_states_<KTraits>(&storage, 0, variant, s_frag, o_frag, m, d);
        compute_small_host_pv<0, KTraits>(&storage, s_frag, d, o_frag);
        normalize_d_<KTraits>(&storage, 0, o_frag, m, d);
        finalize_m_<KTraits>(variant, m);
        write_o<KTraits, true>(&storage, params.final_o, params.final_lse,
                         params.partial_o + blockIdx.y * 64 * 512,
                         params.partial_lse + blockIdx.y * 64, o_frag, m, d, params.o_stride_n,
                         params.o_stride_h, 1, 0, params.num_heads, false);
#pragma unroll
        for (uint32_t feature = 0; feature < 16; ++feature) {
#pragma unroll
          for (uint32_t reg = 0; reg < 8; ++reg) o_frag[feature][reg] = 0.f;
        }
        compute_small_host_pv<1, KTraits>(&storage, s_frag, d, o_frag);
        normalize_d_<KTraits>(&storage, 0, o_frag, m, d);
        write_o<KTraits, true>(&storage, params.final_o, params.final_lse,
                         params.partial_o + blockIdx.y * 64 * 512 + 256,
                         params.partial_lse + blockIdx.y * 64, o_frag, m, d, params.o_stride_n,
                         params.o_stride_h, 1, 0, params.num_heads, false);
      }
      return;
    }
  }
  for (; tile >= 0; --tile) {
    if (tile > 0)
      cp_async::wait_group<1>();
    else
      cp_async::wait_group<0>();
    __syncthreads();
    if constexpr (!PIPELINE) compute_mla_qk<KTraits>(&storage, tile % 2, s_frag);
    logits_mask_<KTraits>(0, tile * KTraits::CTA_TILE_KV, 1, length, length, params.num_heads,
                          s_frag);
    update_mdo_states_<KTraits>(&storage, tile % 2, variant, s_frag, o_frag, m, d);
    compute_mla_pv<KTraits>(&storage, tile % 2, s_frag, d, o_frag);
    __syncthreads();
    if (tile >= 2) {
      load_kv<KTraits>(&storage, params.ckv, params.kpe, storage.indices, params.ckv_stride_n,
                       params.ckv_stride_page, params.kpe_stride_n, params.kpe_stride_page, length,
                       (tile - 2) * KTraits::CTA_TILE_KV, params.block_size, (tile - 2) % 2);
      cp_async::commit_group();
    }
  }
  normalize_d_<KTraits>(&storage, 0, o_frag, m, d);
  finalize_m_<KTraits>(variant, m);
  write_o<KTraits, true>(&storage, params.final_o, params.final_lse,
                   params.partial_o + blockIdx.y * 64 * 512, params.partial_lse + blockIdx.y * 64,
                   o_frag, m, d, params.o_stride_n, params.o_stride_h, 1, 0, params.num_heads,
                   false);
}

template <uint32_t START, uint32_t COUNT, typename KTraits, typename Params>
__device__ __forceinline__ void merge_host_group(typename KTraits::SharedStorage& storage,
                                                 const Params& params, float* output,
                                                 float& lse_output) {
  const uint32_t tid = threadIdx.y * 32 + threadIdx.x, lane = threadIdx.x;
  const uint32_t head = blockIdx.y - 44;
  extern __shared__ uint8_t smem[];
  auto* weights = reinterpret_cast<float*>(smem + 36864);
  if (tid < 32) {
    uint32_t mask;
    do {
      unsigned long long ready = storage.epoch;
      if (lane < COUNT) {
        asm volatile("ld.global.cg.u64 %0, [%1];"
                     : "=l"(ready)
                     : "l"(params.host_prefetch_state + gridDim.y + 2048 + START + lane)
                     : "memory");
      }
      mask = __ballot_sync(0xffffffff, ready == storage.epoch);
      if (mask != 0xffffffff) __nanosleep(128);
    } while (mask != 0xffffffff);
    if (lane < COUNT) {
      unsigned long long ready;
      asm volatile("ld.acquire.gpu.global.u64 %0, [%1];"
                   : "=l"(ready)
                   : "l"(params.host_prefetch_state + gridDim.y + 2048 + START + lane)
                   : "memory");
    }
    const float lse = lane < COUNT ? params.partial_lse[(START + lane) * 64 + head] : -math::inf;
    float maximum = lse;
#pragma unroll
    for (uint32_t offset = 16; offset > 0; offset /= 2)
      maximum = max(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    const float weight = maximum == -math::inf ? 0.f : math::ptx_exp2(lse - maximum);
    float sum = weight;
#pragma unroll
    for (uint32_t offset = 16; offset > 0; offset /= 2)
      sum += __shfl_xor_sync(0xffffffff, sum, offset);
    if (lane < COUNT) weights[lane] = sum > 0.f ? weight / sum : 0.f;
    if (lane == 0) weights[32] = maximum == -math::inf ? -math::inf : maximum + math::ptx_log2(sum);
  }
  asm volatile("bar.sync 1, 128;" ::: "memory");
  lse_output = weights[32];
  uint2 packed_values[COUNT];
#pragma unroll
  for (uint32_t split = 0; split < COUNT; ++split) {
    asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];"
                 : "=r"(packed_values[split].x), "=r"(packed_values[split].y)
                 : "l"(params.partial_o + ((START + split) * 64 + head) * 512 + tid * 4)
                 : "memory");
  }
#pragma unroll
  for (uint32_t feature = 0; feature < 4; ++feature) output[feature] = 0.f;
#pragma unroll
  for (uint32_t split = 0; split < COUNT; ++split) {
    const float weight = weights[split];
    if (weight > 0.f) {
      float values[4];
      vec_cast<float, typename KTraits::DTypeO>::template cast<4>(
          values, reinterpret_cast<const typename KTraits::DTypeO*>(packed_values + split));
#pragma unroll
      for (uint32_t feature = 0; feature < 4; ++feature)
        output[feature] = fmaf(values[feature], weight, output[feature]);
    }
  }
  asm volatile("bar.sync 1, 128;" ::: "memory");
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void merge_ready_host_states(typename KTraits::SharedStorage& storage,
                                                        const Params& params) {
  const uint32_t tid = threadIdx.y * 32 + threadIdx.x, head = blockIdx.y - 44;
  float local[4], host[4], local_lse, host_lse;
  merge_host_group<0, 16, KTraits>(storage, params, local, local_lse);
  merge_host_group<16, 28, KTraits>(storage, params, host, host_lse);
  const float maximum = max(local_lse, host_lse);
  const float local_weight = math::ptx_exp2(local_lse - maximum);
  const float host_weight = math::ptx_exp2(host_lse - maximum);
  const float denominator = local_weight + host_weight;
#pragma unroll
  for (uint32_t feature = 0; feature < 4; ++feature) {
    params.final_o[head * params.o_stride_h + tid * 4 + feature] = typename KTraits::DTypeO(
        (local[feature] * local_weight + host[feature] * host_weight) / denominator);
  }
  if (tid == 0 && params.final_lse) {
    float value = maximum + math::ptx_log2(denominator);
    if (params.return_lse_base_on_e) value *= math::loge2;
    params.final_lse[head] = value;
  }
}

template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLAHostOverlapKernel(Params params) {
  extern __shared__ uint8_t smem[];
  auto& storage = *reinterpret_cast<typename KTraits::SharedStorage*>(smem);
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  record_overlap_phase(params, 0);
  if (tid == 0) storage.epoch = atomicAdd(params.host_prefetch_state + blockIdx.y, 1ULL) + 1;
  __syncthreads();
  params.partial_o = params.host_prefetch_buffer + 2048 * 576;
  params.partial_lse = reinterpret_cast<float*>(params.host_prefetch_state + gridDim.y + 2048 + 44);
  if (blockIdx.y < 44) {
    const uint32_t length = prepare_host_split<KTraits>(&storage, params);
    record_overlap_phase(params, 1);
    if (length > 0) {
      if (blockIdx.y < 16)
        compute_host_split<false, false, false, KTraits>(storage, params, length, smem);
      else if (length > 16 && length <= 32)
        compute_host_split<true, true, false, KTraits>(storage, params, length, smem);
      else
        compute_host_split<true, false, false, HostNarrowKernelTraits<KTraits>>(storage, params, length, smem);
    } else {
      if (tid < 64) params.partial_lse[blockIdx.y * 64 + tid] = -math::inf;
    }
    record_overlap_phase(params, 3);
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      asm volatile("st.release.gpu.global.u64 [%0], %1;"
                   :
                   : "l"(params.host_prefetch_state + gridDim.y + 2048 + blockIdx.y),
                     "l"(storage.epoch)
                   : "memory");
    }
  } else {
    if (threadIdx.z == 1)
      run_host_fetch_warps<KTraits>(params, storage.epoch);
    else
      merge_ready_host_states<KTraits>(storage, params);
  }
}

template <uint32_t START, uint32_t COUNT, uint32_t HOST_CHUNKS, typename KTraits, typename Params>
__device__ __forceinline__ void merge_private_host_group(
    typename KTraits::SharedStorage& storage, const Params& params, float* output,
    float& lse_output, uint32_t head) {
  const uint32_t tid = threadIdx.y * 32 + threadIdx.x, lane = threadIdx.x;
  extern __shared__ uint8_t smem[];
  auto* weights = reinterpret_cast<float*>(smem + 36864 + threadIdx.z * 512);
  constexpr uint32_t BATCHES = (COUNT + 31) / 32;
  if (tid < 32) {
    uint32_t mask;
    do {
      bool complete = true;
#pragma unroll
      for (uint32_t batch = 0; batch < BATCHES; ++batch) {
        const uint32_t split = batch * 32 + lane;
        if (split < COUNT) {
          unsigned long long ready;
          asm volatile("ld.global.cg.u64 %0, [%1];"
                       : "=l"(ready)
                       : "l"(params.host_prefetch_state + gridDim.y + START + split)
                       : "memory");
          complete = complete && ready == storage.epoch;
        }
      }
      mask = __ballot_sync(0xffffffff, complete);
      if (mask != 0xffffffff) __nanosleep(128);
    } while (mask != 0xffffffff);
    float lse[BATCHES], weight[BATCHES], maximum = -math::inf;
#pragma unroll
    for (uint32_t batch = 0; batch < BATCHES; ++batch) {
      const uint32_t split = batch * 32 + lane;
      lse[batch] = -math::inf;
      if (split < COUNT) {
        unsigned long long ready;
        asm volatile("ld.acquire.gpu.global.u64 %0, [%1];"
                     : "=l"(ready)
                     : "l"(params.host_prefetch_state + gridDim.y + START + split)
                     : "memory");
        lse[batch] = params.partial_lse[(START + split) * 64 + head];
      }
      maximum = max(maximum, lse[batch]);
    }
#pragma unroll
    for (uint32_t offset = 16; offset > 0; offset /= 2)
      maximum = max(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    float sum = 0.f;
#pragma unroll
    for (uint32_t batch = 0; batch < BATCHES; ++batch) {
      weight[batch] = maximum == -math::inf ? 0.f : math::ptx_exp2(lse[batch] - maximum);
      sum += weight[batch];
    }
#pragma unroll
    for (uint32_t offset = 16; offset > 0; offset /= 2)
      sum += __shfl_xor_sync(0xffffffff, sum, offset);
#pragma unroll
    for (uint32_t batch = 0; batch < BATCHES; ++batch) {
      const uint32_t split = batch * 32 + lane;
      if (split < COUNT) weights[split] = sum > 0.f ? weight[batch] / sum : 0.f;
    }
    if (lane == 0) weights[64] = maximum == -math::inf ? -math::inf : maximum + math::ptx_log2(sum);
  }
  const uint32_t barrier = threadIdx.z + 1;
  asm volatile("bar.sync %0, 128;" :: "r"(barrier) : "memory");
  lse_output = weights[64];
#ifdef FLASHINFER_MLA_HOST_STREAM_MERGE
#pragma unroll
  for (uint32_t feature = 0; feature < 4; ++feature) output[feature] = 0.f;
#pragma unroll 1
  for (uint32_t batch = 0; batch < COUNT; batch += 8) {
    uint2 values[8];
#pragma unroll
    for (uint32_t slot = 0; slot < 8; ++slot) {
      const uint32_t split = batch + slot;
      if (split < COUNT) {
        asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];"
                     : "=r"(values[slot].x), "=r"(values[slot].y)
                     : "l"(params.partial_o + ((START + split) * 64 + head) * 512 + tid * 4)
                     : "memory");
      }
    }
#pragma unroll
    for (uint32_t slot = 0; slot < 8; ++slot) {
      const uint32_t split = batch + slot;
      if (split < COUNT && weights[split] > 0.f) {
        float decoded[4];
        vec_cast<float, typename KTraits::DTypeO>::template cast<4>(
            decoded, reinterpret_cast<const typename KTraits::DTypeO*>(values + slot));
#pragma unroll
        for (uint32_t feature = 0; feature < 4; ++feature)
          output[feature] = fmaf(decoded[feature], weights[split], output[feature]);
      }
    }
  }
#else
  uint2 values[COUNT];
#pragma unroll
  for (uint32_t split = 0; split < COUNT; ++split) {
    asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];"
                 : "=r"(values[split].x), "=r"(values[split].y)
                 : "l"(params.partial_o + ((START + split) * 64 + head) * 512 + tid * 4)
                 : "memory");
  }
#pragma unroll
  for (uint32_t feature = 0; feature < 4; ++feature) output[feature] = 0.f;
#pragma unroll
  for (uint32_t split = 0; split < COUNT; ++split) {
    const float weight = weights[split];
    if (weight > 0.f) {
      float decoded[4];
      vec_cast<float, typename KTraits::DTypeO>::template cast<4>(
          decoded, reinterpret_cast<const typename KTraits::DTypeO*>(values + split));
#pragma unroll
      for (uint32_t feature = 0; feature < 4; ++feature)
        output[feature] = fmaf(decoded[feature], weight, output[feature]);
    }
  }
#endif
  asm volatile("bar.sync %0, 128;" :: "r"(barrier) : "memory");
}

template <uint32_t HOST_CHUNKS, typename KTraits, typename Params, uint32_t LOCAL_CHUNKS = 16>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLAPrivateHostKernel(Params params) {
  static_assert(LOCAL_CHUNKS >= 16 && LOCAL_CHUNKS + HOST_CHUNKS <= 84);
#if defined(FLASHINFER_MLA_HOST_PRIVATE_PIPELINE) || !defined(FLASHINFER_MLA_HOST_PRIVATE_SHARED)
  constexpr bool PIPELINE = true;
#else
  constexpr bool PIPELINE = false;
#endif
  extern __shared__ uint8_t smem[];
  auto& storage = *reinterpret_cast<typename KTraits::SharedStorage*>(smem);
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  record_overlap_phase(params, 0);
  if (tid == 0) storage.epoch = atomicAdd(params.host_prefetch_state + blockIdx.y, 1ULL) + 1;
  __syncthreads();
  params.partial_o = params.host_prefetch_buffer;
  params.partial_lse = reinterpret_cast<float*>(params.host_prefetch_state + gridDim.y + LOCAL_CHUNKS + HOST_CHUNKS);
  if (blockIdx.y < LOCAL_CHUNKS + HOST_CHUNKS) {
    const uint32_t length = prepare_host_split<KTraits, HOST_CHUNKS, true, LOCAL_CHUNKS>(&storage, params);
    record_overlap_phase(params, 1);
    if (length > 0) {
      if (blockIdx.y < LOCAL_CHUNKS)
        compute_host_split<false, false, false, KTraits>(storage, params, length, smem);
      else if (length <= 16)
        compute_host_split<true, true, true, HostNarrowKernelTraits<KTraits>, PIPELINE>(
            storage, params, length, smem);
      else if (length <= 32)
        compute_host_split<true, true, true, KTraits, PIPELINE>(storage, params, length, smem);
      else
        compute_host_split<true, false, true, KTraits>(storage, params, length, smem);
    } else if (tid < 64) {
      params.partial_lse[blockIdx.y * 64 + tid] = -math::inf;
    }
    record_overlap_phase(params, 3);
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      asm volatile("st.release.gpu.global.u64 [%0], %1;"
                   :: "l"(params.host_prefetch_state + gridDim.y + blockIdx.y),
                      "l"(storage.epoch) : "memory");
    }
  } else {
    constexpr uint32_t MERGE_CTAS = 108 - LOCAL_CHUNKS - HOST_CHUNKS;
    constexpr uint32_t HEADS_PER_CTA = MERGE_CTAS >= 64 ? 1 : (MERGE_CTAS >= 32 ? 2 : 4);
    if (HEADS_PER_CTA == 1 && threadIdx.z != 0) return;
    const uint32_t feature_base = (threadIdx.y * 32 + threadIdx.x) * 4;
#pragma unroll
    for (uint32_t pass = 0; pass < (HEADS_PER_CTA == 1 ? 1 : HEADS_PER_CTA / 2); ++pass) {
      const uint32_t offset = pass * 2 + threadIdx.z;
      const uint32_t head = (blockIdx.y - LOCAL_CHUNKS - HOST_CHUNKS) * HEADS_PER_CTA + offset;
      if (head >= 64) return;
      float local[4], host[4], local_lse, host_lse;
      merge_private_host_group<0, LOCAL_CHUNKS, HOST_CHUNKS, KTraits>(
          storage, params, local, local_lse, head);
      merge_private_host_group<LOCAL_CHUNKS, HOST_CHUNKS, HOST_CHUNKS, KTraits>(
          storage, params, host, host_lse, head);
      const float maximum = max(local_lse, host_lse);
      const float local_weight = math::ptx_exp2(local_lse - maximum);
      const float host_weight = math::ptx_exp2(host_lse - maximum);
      const float denominator = local_weight + host_weight;
#pragma unroll
      for (uint32_t feature = 0; feature < 4; ++feature)
        params.final_o[head * params.o_stride_h + feature_base + feature] = typename KTraits::DTypeO(
            (local[feature] * local_weight + host[feature] * host_weight) / denominator);
      if (feature_base == 0 && params.final_lse) {
        float lse = maximum + math::ptx_log2(denominator);
        if (params.return_lse_base_on_e) lse *= math::loge2;
        params.final_lse[head] = lse;
      }
      record_overlap_phase(params, 3);
    }
  }
}

#define DISPATCH_SMEM_CONFIG(smem_limit_per_sm, NUM_STAGES, CTA_TILE_KV, QK_SHARD, ...) \
  if (smem_limit_per_sm >= 221696) {                                                    \
    constexpr uint32_t NUM_STAGES = 2;                                                  \
    constexpr uint32_t CTA_TILE_KV = 64;                                                \
    constexpr bool QK_SHARD = true;                                                     \
    __VA_ARGS__;                                                                        \
  } else if (smem_limit_per_sm >= 147968) {                                             \
    constexpr uint32_t NUM_STAGES = 2;                                                  \
    constexpr uint32_t CTA_TILE_KV = 32;                                                \
    constexpr bool QK_SHARD = true;                                                     \
    __VA_ARGS__;                                                                        \
  } else if (smem_limit_per_sm >= 92672) {                                              \
    constexpr uint32_t NUM_STAGES = 1;                                                  \
    constexpr uint32_t CTA_TILE_KV = 16;                                                \
    constexpr bool QK_SHARD = false;                                                    \
    __VA_ARGS__;                                                                        \
  } else {                                                                              \
    std::ostringstream err;                                                             \
    err << "Unsupported shared memory size: " << smem_limit_per_sm;                     \
    FLASHINFER_ERROR(err.str());                                                        \
    return cudaErrorNotSupported;                                                       \
  }

template <MaskMode MASK_MODE, uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename Params>
cudaError_t BatchMLAPagedAttention(Params params, uint32_t num_blks_x, uint32_t num_blks_y,
                                   cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  if (MASK_MODE == MaskMode::kCustom) {
    return cudaErrorNotSupported;
  }
  constexpr bool CAUSAL = MASK_MODE == MaskMode::kCausal;

  dim3 nblks(num_blks_x, num_blks_y);
  dim3 nthrs(32, 4, 2);

  // get GPU shared memory size
  int device;
  int smem_limit_per_sm;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&smem_limit_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);

  if constexpr (!CAUSAL && HEAD_DIM_CKV == 512 && HEAD_DIM_KPE == 64 &&
                sizeof(DTypeQ) == 2 && sizeof(DTypeKV) == 2) {
    using HostTraits = HostSplitKernelTraits<KernelTraits<
        false, 2, true, HEAD_DIM_CKV, HEAD_DIM_KPE, 64, 32, DTypeQ, DTypeKV, DTypeO, IdType>>;
    if (params.host_page_start >= 0 && params.block_size == 1 && num_blks_y >= 108 &&
        smem_limit_per_sm >= sizeof(typename HostTraits::SharedStorage)) {
#ifdef FLASHINFER_MLA_HOST_STAGED_BASELINE
      auto kernel = BatchMLAHostOverlapKernel<HostTraits, Params>;
#elif defined(FLASHINFER_MLA_HOST_PRIVATE_SHARED)
#ifndef FLASHINFER_MLA_HOST_LOCAL_CHUNKS
#define FLASHINFER_MLA_HOST_LOCAL_CHUNKS 16
#endif
      auto kernel = BatchMLAPrivateHostKernel<FLASHINFER_MLA_HOST_PRIVATE_CHUNKS, HostTraits, Params,
                                            FLASHINFER_MLA_HOST_LOCAL_CHUNKS>;
#else
      auto kernel = BatchMLAPrivateHostKernel<32, HostTraits, Params>;
#endif
      size_t smem_size = sizeof(typename HostTraits::SharedStorage);
      void* args[] = {(void*)&params};
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      FLASHINFER_CUDA_CALL(
          cudaLaunchCooperativeKernel((void*)kernel, dim3(1, 108), nthrs, args, smem_size, stream));
      return cudaSuccess;
    }
  }

  if (params.host_page_start >= 0) return cudaErrorNotSupported;

  DISPATCH_SMEM_CONFIG(smem_limit_per_sm, NUM_STAGES, CTA_TILE_KV, QK_SHARD, {
    using KTraits = KernelTraits<CAUSAL, NUM_STAGES, QK_SHARD, HEAD_DIM_CKV, HEAD_DIM_KPE,
                                 /*CTA_TILE_Q_=*/64, CTA_TILE_KV, DTypeQ, DTypeKV, DTypeO, IdType>;
    size_t smem_size = sizeof(typename KTraits::SharedStorage);
    auto kernel = BatchMLAPagedAttentionKernel<KTraits, Params>;
    void* args[] = {(void*)&params};

    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    FLASHINFER_CUDA_CALL(
        cudaLaunchCooperativeKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  });

  return cudaSuccess;
}

}  // namespace mla

}  // namespace flashinfer

#endif  // FLASHINFER_MLA_FA2_CUH_
