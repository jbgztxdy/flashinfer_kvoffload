/*
 * Copyright (c) 2025 by FlashInfer team.
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
#include <cstdint>
#include <flashinfer/attention/mla.cuh>
#include <flashinfer/attention/mla_host_batch.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/fastdiv.cuh>

#include "batch_mla_config.inc"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Array;
using tvm::ffi::Optional;

void BatchMLAPagedAttentionRunHostPrefetch(
    TensorView float_workspace_buffer, TensorView int_workspace_buffer,
    Array<int64_t> plan_info_vec, TensorView q_nope, TensorView q_pe, TensorView ckv_cache,
    TensorView kpe_cache, TensorView kv_indices, TensorView o, Optional<TensorView> maybe_lse,
    int64_t mask_mode_code, int64_t num_heads, int64_t page_size, double sm_scale,
    bool return_lse_base_on_e, int64_t host_page_start, Optional<TensorView> maybe_host_buffer,
    Optional<TensorView> maybe_host_state) {
  // q_nope: [n, num_heads, head_dim_ckv]
  // q_pe: [n, num_heads, head_dim_kpe]
  // ckv_cache: [num_pages, page_size, head_dim_ckv]
  // kpe_cache: [num_pages, page_size, head_dim_kpe]
  CHECK_INPUT_TYPE(kv_indices, dl_int32);

  MLAPlanInfo plan_info;
  plan_info.FromVector(std::vector<int64_t>(plan_info_vec.begin(), plan_info_vec.end()));

  if (host_page_start >= 0) {
    TVM_FFI_ICHECK(maybe_host_buffer.has_value() && maybe_host_state.has_value())
        << "Host overlap requires initialized scratch buffers";
    TVM_FFI_ICHECK_EQ(mask_mode_code, static_cast<int64_t>(MaskMode::kNone));
    TVM_FFI_ICHECK_EQ(num_heads, 64);
    TVM_FFI_ICHECK_EQ(page_size, 1);
    TVM_FFI_ICHECK_EQ(q_nope.ndim(), 3);
    TVM_FFI_ICHECK_EQ(q_nope.size(0), 1);
    TVM_FFI_ICHECK_EQ(q_nope.size(1), 64);
    TVM_FFI_ICHECK_EQ(q_nope.size(2), 512);
    TVM_FFI_ICHECK_EQ(q_pe.ndim(), 3);
    TVM_FFI_ICHECK_EQ(q_pe.size(0), 1);
    TVM_FFI_ICHECK_EQ(q_pe.size(1), 64);
    TVM_FFI_ICHECK_EQ(q_pe.size(2), 64);
    TVM_FFI_ICHECK_EQ(ckv_cache.ndim(), 3);
    TVM_FFI_ICHECK_EQ(ckv_cache.size(1), 1);
    TVM_FFI_ICHECK_EQ(ckv_cache.size(2), 512);
    TVM_FFI_ICHECK_EQ(kpe_cache.ndim(), 3);
    TVM_FFI_ICHECK_EQ(kpe_cache.size(0), ckv_cache.size(0));
    TVM_FFI_ICHECK_EQ(kpe_cache.size(1), 1);
    TVM_FFI_ICHECK_EQ(kpe_cache.size(2), 64);
    TVM_FFI_ICHECK_LE(host_page_start, ckv_cache.size(0));
    TVM_FFI_ICHECK_GE(kv_indices.size(0), 2048);
    const auto buffer = maybe_host_buffer.value();
    const auto state = maybe_host_state.value();
    CHECK_INPUT(buffer);
    CHECK_INPUT(state);
    CHECK_DEVICE(buffer, q_nope);
    CHECK_DEVICE(state, q_nope);
    CHECK_SAME_DTYPE(buffer, ckv_cache);
    CHECK_INPUT_TYPE(state, dl_int64);
    TVM_FFI_ICHECK_EQ(buffer.ndim(), 2);
    TVM_FFI_ICHECK_GE(buffer.size(0), 5376);
    TVM_FFI_ICHECK_EQ(buffer.size(1), 512);
    TVM_FFI_ICHECK_EQ(state.ndim(), 1);
    TVM_FFI_ICHECK_GE(state.size(0), 4096);
  }

  void* float_buffer_ptr = float_workspace_buffer.data_ptr();
  void* int_buffer_ptr = int_workspace_buffer.data_ptr();

  const MaskMode mask_mode = static_cast<MaskMode>(mask_mode_code);

  unsigned int q_nope_stride_n = q_nope.stride(0);
  unsigned int q_nope_stride_h = q_nope.stride(1);
  unsigned int q_pe_stride_n = q_pe.stride(0);
  unsigned int q_pe_stride_h = q_pe.stride(1);
  unsigned int ckv_stride_page = ckv_cache.stride(0);
  unsigned int ckv_stride_n = ckv_cache.stride(1);
  unsigned int kpe_stride_page = kpe_cache.stride(0);
  unsigned int kpe_stride_n = kpe_cache.stride(1);
  unsigned int o_stride_n = o.stride(0);
  unsigned int o_stride_h = o.stride(1);

  ffi::CUDADeviceGuard device_guard(q_nope.device().device_id);
  const cudaStream_t stream = get_stream(q_nope.device());

  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, MASK_MODE, HEAD_DIM_CKV, HEAD_DIM_KPE, Params, [&] {
        Params params;

        params.q_nope = static_cast<DTypeQ*>(q_nope.data_ptr());
        params.q_pe = static_cast<DTypeQ*>(q_pe.data_ptr());
        params.ckv = static_cast<DTypeKV*>(ckv_cache.data_ptr());
        params.kpe = static_cast<DTypeKV*>(kpe_cache.data_ptr());

        params.q_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.q_indptr_offset);
        params.kv_indptr = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_indptr_offset);
        params.partial_indptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.partial_indptr_offset);
        params.kv_indices = static_cast<IdType*>(kv_indices.data_ptr());
        params.q_len = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.q_len_offset);
        params.kv_len = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_len_offset);
        params.q_start = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.q_start_offset);
        params.kv_start = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_start_offset);
        params.kv_end = GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.kv_end_offset);
        params.work_indptr =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.work_indptr_offset);
        params.merge_packed_offset_start = GetPtrFromBaseOffset<IdType>(
            int_buffer_ptr, plan_info.merge_packed_offset_start_offset);
        params.merge_packed_offset_end =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_packed_offset_end_offset);
        params.merge_partial_packed_offset_start = GetPtrFromBaseOffset<IdType>(
            int_buffer_ptr, plan_info.merge_partial_packed_offset_start_offset);
        params.merge_partial_packed_offset_end = GetPtrFromBaseOffset<IdType>(
            int_buffer_ptr, plan_info.merge_partial_packed_offset_end_offset);
        params.merge_partial_stride =
            GetPtrFromBaseOffset<IdType>(int_buffer_ptr, plan_info.merge_partial_stride_offset);
        params.final_o = static_cast<DTypeO*>(o.data_ptr());
        params.final_lse =
            maybe_lse.has_value() ? static_cast<float*>(maybe_lse.value().data_ptr()) : nullptr;
        params.partial_o =
            GetPtrFromBaseOffset<DTypeO>(float_buffer_ptr, plan_info.partial_o_offset);
        params.partial_lse =
            GetPtrFromBaseOffset<float>(float_buffer_ptr, plan_info.partial_lse_offset);

        params.num_heads = uint_fastdiv(num_heads);
        params.block_size = uint_fastdiv(page_size);

        params.q_nope_stride_n = q_nope_stride_n;
        params.q_nope_stride_h = q_nope_stride_h;
        params.q_pe_stride_n = q_pe_stride_n;
        params.q_pe_stride_h = q_pe_stride_h;
        params.ckv_stride_page = ckv_stride_page;
        params.ckv_stride_n = ckv_stride_n;
        params.kpe_stride_page = kpe_stride_page;
        params.kpe_stride_n = kpe_stride_n;
        params.o_stride_n = o_stride_n;
        params.o_stride_h = o_stride_h;

        params.sm_scale = sm_scale;
        params.return_lse_base_on_e = return_lse_base_on_e;
        params.host_page_start = host_page_start;
        params.host_prefetch_buffer =
            maybe_host_buffer.has_value()
                ? static_cast<DTypeKV*>(maybe_host_buffer.value().data_ptr())
                : nullptr;
        params.host_prefetch_state =
            maybe_host_state.has_value()
                ? static_cast<unsigned long long*>(maybe_host_state.value().data_ptr())
                : nullptr;

        cudaError_t status = mla::BatchMLAPagedAttention<MASK_MODE, HEAD_DIM_CKV, HEAD_DIM_KPE>(
            params, plan_info.num_blks_x, plan_info.num_blks_y, stream);

        TVM_FFI_ICHECK(status == cudaSuccess)
            << "Failed to run MLA, error: " << cudaGetErrorString(status);
      });
}

void BatchMLAPagedAttentionRun(TensorView float_workspace_buffer, TensorView int_workspace_buffer,
                               Array<int64_t> plan_info_vec, TensorView q_nope, TensorView q_pe,
                               TensorView ckv_cache, TensorView kpe_cache, TensorView kv_indices,
                               TensorView o, Optional<TensorView> maybe_lse, int64_t mask_mode_code,
                               int64_t num_heads, int64_t page_size, double sm_scale,
                               bool return_lse_base_on_e) {
  BatchMLAPagedAttentionRunHostPrefetch(
      float_workspace_buffer, int_workspace_buffer, plan_info_vec, q_nope, q_pe, ckv_cache,
      kpe_cache, kv_indices, o, maybe_lse, mask_mode_code, num_heads, page_size, sm_scale,
      return_lse_base_on_e, -1, Optional<TensorView>(), Optional<TensorView>());
}

void BatchMLAPagedAttentionRunHostBatch(
    TensorView q_nope, TensorView q_pe, TensorView ckv_cache, TensorView kpe_cache,
    TensorView kv_indptr, TensorView kv_indices, TensorView o, Optional<TensorView> maybe_lse,
    int64_t num_heads, double sm_scale, bool return_lse_base_on_e, int64_t host_page_start,
    int64_t max_kv_len, TensorView staging, TensorView host_slot, TensorView remap,
    TensorView unique_pages, TensorView state, TensorView partial_o, TensorView partial_lse,
    int64_t fetch_ctas, int64_t num_pools, int64_t pool_capacity, TensorView cache_rows,
    TensorView cache_persist, TensorView cache_scratch) {
  // q_nope: [rows, num_heads, 512], q_pe: [rows, num_heads, 64], one query token per row
  // ckv_cache: [num_pages, 1, 512], kpe_cache: [num_pages, 1, 64]
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_nope);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(q_pe);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(o);
  CHECK_INPUT(kv_indptr);
  CHECK_INPUT(kv_indices);
  CHECK_INPUT(staging);
  CHECK_INPUT(host_slot);
  CHECK_INPUT(remap);
  CHECK_INPUT(unique_pages);
  CHECK_INPUT(state);
  CHECK_INPUT(partial_o);
  CHECK_INPUT(partial_lse);
  CHECK_DEVICE(q_pe, q_nope);
  CHECK_DEVICE(ckv_cache, q_nope);
  CHECK_DEVICE(kpe_cache, q_nope);
  CHECK_DEVICE(o, q_nope);
  CHECK_DEVICE(kv_indptr, q_nope);
  CHECK_DEVICE(kv_indices, q_nope);
  CHECK_DEVICE(staging, q_nope);
  CHECK_DEVICE(host_slot, q_nope);
  CHECK_DEVICE(state, q_nope);
  CHECK_DEVICE(partial_o, q_nope);
  CHECK_INPUT_TYPE(kv_indptr, dl_int32);
  CHECK_INPUT_TYPE(kv_indices, dl_int32);
  CHECK_INPUT_TYPE(remap, dl_int32);
  CHECK_INPUT_TYPE(unique_pages, dl_int32);
  CHECK_INPUT_TYPE(host_slot, dl_int32);
  CHECK_INPUT_TYPE(state, dl_int32);
  CHECK_INPUT_TYPE(partial_lse, dl_float32);
  CHECK_SAME_DTYPE(staging, ckv_cache);
  CHECK_SAME_DTYPE(partial_o, o);
  const int64_t rows = q_nope.size(0);
  const int64_t head_tiles = num_heads / 64;
  TVM_FFI_ICHECK(num_heads % 64 == 0 && q_nope.size(1) == num_heads && q_pe.size(1) == num_heads)
      << "Host batch requires a multiple of 64 heads";
  TVM_FFI_ICHECK(q_nope.size(2) == 512 && q_pe.size(0) == rows && q_pe.size(2) == 64);
  TVM_FFI_ICHECK(ckv_cache.ndim() == 3 && ckv_cache.size(1) == 1 && ckv_cache.size(2) == 512);
  TVM_FFI_ICHECK(kpe_cache.ndim() == 3 && kpe_cache.size(0) == ckv_cache.size(0) &&
                 kpe_cache.size(1) == 1 && kpe_cache.size(2) == 64);
  TVM_FFI_ICHECK(ckv_cache.stride(0) == 576 && kpe_cache.stride(0) == 576 &&
                 ckv_cache.stride(2) == 1 && kpe_cache.stride(2) == 1 &&
                 static_cast<char*>(kpe_cache.data_ptr()) ==
                     static_cast<char*>(ckv_cache.data_ptr()) + 512 * sizeof(DTypeKV))
      << "Host batch requires ckv/kpe views of one [num_pages, 1, 576] cache";
  TVM_FFI_ICHECK(0 <= host_page_start && host_page_start <= ckv_cache.size(0));
  TVM_FFI_ICHECK_EQ(kv_indptr.size(0), rows + 1);
  TVM_FFI_ICHECK_GE(remap.size(0), kv_indices.size(0));
  TVM_FFI_ICHECK_GE(unique_pages.size(0), kv_indices.size(0));
  TVM_FFI_ICHECK_GE(host_slot.size(0), ckv_cache.size(0) - host_page_start);
  TVM_FFI_ICHECK(staging.ndim() == 2 && staging.size(1) == 576 && staging.size(0) >= 1);
  // LRU cache (pool_capacity 0 disables it): cache slots follow the staging slots in `staging`.
  // cache_rows: row_pool [rows] | pool_row_start [num_pools + 1].
  // cache_persist (kept across launches, per layer): resident [Host pages] init -1 |
  //   slot_page [slots] init -1 | slot_stamp [slots] init 0.
  // cache_scratch: victim_of [capacity] | slot_pool [capacity] | row_slot_base [rows + 1] |
  //   pool_flags [num_pools] (epoch-tagged, init 0).
  const int64_t cache_slots = num_pools * pool_capacity;
  TVM_FFI_ICHECK(num_pools >= 0 && pool_capacity >= 0 && staging.size(0) > cache_slots);
  const int64_t capacity = staging.size(0) - cache_slots;
  // Items: at most one per row tile plus the split tail, bounded by the partial capacity.
  // Per row: Host count, claims, first Host item; the row of each Host item; fetch flags; item
  // claims (row tiles plus tail item states).
  const int64_t partials = partial_lse.size(0) / 64;
  TVM_FFI_ICHECK_GE(state.size(0), mla::kHostBatchHostCounts + 3 * rows + 1 + partials +
                                       (capacity + mla::kHostBatchFetchRows - 1) /
                                           mla::kHostBatchFetchRows +
                                       rows * head_tiles + partials +
                                       (pool_capacity > 0 ? num_pools : 0));
  // host_slot packs (row, rank within the row's claims); see prepare_host_batch_rows.
  TVM_FFI_ICHECK(rows < (1 << (31 - mla::kHostBatchRankBits)) &&
                 max_kv_len < (1 << mla::kHostBatchRankBits))
      << "Host batch supports fewer than 32768 rows of fewer than 65536 pages";
  const int64_t host_pages = ckv_cache.size(0) - host_page_start;
  if (pool_capacity > 0) {
    TVM_FFI_ICHECK(num_pools > 0);
    CHECK_INPUT(cache_rows);
    CHECK_INPUT(cache_persist);
    CHECK_INPUT(cache_scratch);
    CHECK_INPUT_TYPE(cache_rows, dl_int32);
    CHECK_INPUT_TYPE(cache_persist, dl_int32);
    CHECK_INPUT_TYPE(cache_scratch, dl_int32);
    TVM_FFI_ICHECK_GE(cache_rows.size(0), rows + num_pools + 1);
    TVM_FFI_ICHECK_GE(cache_persist.size(0), host_pages + 2 * cache_slots);
    TVM_FFI_ICHECK_GE(cache_scratch.size(0), 2 * capacity + rows + 1 + num_pools);
  }
  TVM_FFI_ICHECK(partial_lse.size(0) % 64 == 0 &&
                 partial_o.size(0) * partial_o.size(1) >= partial_lse.size(0) * 512);

  ffi::CUDADeviceGuard device_guard(q_nope.device().device_id);
  const cudaStream_t stream = get_stream(q_nope.device());

  MLAHostBatchParams<DTypeQ, DTypeKV, DTypeO, IdType> params;
  params.q_nope = static_cast<DTypeQ*>(q_nope.data_ptr());
  params.q_pe = static_cast<DTypeQ*>(q_pe.data_ptr());
  params.ckv = static_cast<DTypeKV*>(ckv_cache.data_ptr());
  params.final_o = static_cast<DTypeO*>(o.data_ptr());
  params.final_lse =
      maybe_lse.has_value() ? static_cast<float*>(maybe_lse.value().data_ptr()) : nullptr;
  params.kv_indptr = static_cast<IdType*>(kv_indptr.data_ptr());
  params.kv_indices = static_cast<IdType*>(kv_indices.data_ptr());
  params.remap = static_cast<IdType*>(remap.data_ptr());
  params.unique_pages = static_cast<IdType*>(unique_pages.data_ptr());
  params.host_slot = static_cast<uint32_t*>(host_slot.data_ptr());
  params.state = static_cast<uint32_t*>(state.data_ptr());
  params.staging = static_cast<DTypeKV*>(staging.data_ptr());
  params.partial_o = static_cast<DTypeO*>(partial_o.data_ptr());
  params.partial_lse = static_cast<float*>(partial_lse.data_ptr());
  params.num_heads = uint_fastdiv(num_heads);
  params.num_rows = rows;
  params.num_head_tiles = head_tiles;
  params.staging_capacity = capacity;
  params.fetch_ctas = fetch_ctas;
  params.num_pools = pool_capacity > 0 ? num_pools : 0;
  params.pool_capacity = pool_capacity;
  if (pool_capacity > 0) {
    auto* rows_data = static_cast<IdType*>(cache_rows.data_ptr());
    params.row_pool = rows_data;
    params.pool_row_start = rows_data + rows;
    auto* persist = static_cast<int32_t*>(cache_persist.data_ptr());
    params.resident = persist;
    params.slot_page = persist + host_pages;
    params.slot_stamp = reinterpret_cast<uint32_t*>(persist + host_pages + cache_slots);
    auto* scratch = static_cast<int32_t*>(cache_scratch.data_ptr());
    params.victim_of = scratch;
    params.slot_pool = scratch + capacity;
    params.row_slot_base = reinterpret_cast<uint32_t*>(scratch + 2 * capacity);
    params.pool_flags = reinterpret_cast<uint32_t*>(scratch + 2 * capacity + rows + 1);
  } else {
    params.row_pool = params.pool_row_start = nullptr;
    params.resident = params.slot_page = params.victim_of = params.slot_pool = nullptr;
    params.slot_stamp = params.row_slot_base = params.pool_flags = nullptr;
  }
  params.q_nope_stride_n = q_nope.stride(0);
  params.q_nope_stride_h = q_nope.stride(1);
  params.q_pe_stride_n = q_pe.stride(0);
  params.q_pe_stride_h = q_pe.stride(1);
  params.o_stride_n = o.stride(0);
  params.o_stride_h = o.stride(1);
  params.host_page_start = host_page_start;
  params.sm_scale = sm_scale;
  params.return_lse_base_on_e = return_lse_base_on_e;

  cudaError_t status = mla::BatchMLAHostBatchAttention<HEAD_DIM_CKV, HEAD_DIM_KPE>(
      params, max_kv_len, partial_lse.size(0) / 64, stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "Failed to run batched Host MLA, error: " << cudaGetErrorString(status);
}
