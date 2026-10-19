/*
 * Copyright (c) 2026 by FlashInfer team.
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
#ifndef FLASHINFER_MLA_HOST_BATCH_CUH_
#define FLASHINFER_MLA_HOST_BATCH_CUH_

#include "mla.cuh"

namespace flashinfer {

// Batched sparse MLA decode over a KV cache whose pages at or above host_page_start live in
// mapped Host memory. Each row is one query token with its own selected pages.
//
// One cooperative launch runs these phases:
//  1. Every CTA lays out its rows with Host entries first and deduplicates Host pages across the
//     whole batch into HBM staging slots, rewriting staged entries as -(slot + 1).
//  2. After a grid barrier, Host entries get staging slots, numbered in row order. After a
//     second barrier, the last fetch_ctas CTAs copy staged pages in slot order and publish a
//     ready flag per 64 slots, then take attention items from the end of the list. The
//     other CTAs take items (row, 64-head tile, KV chunk) from the front. Each item computes its
//     HBM tiles first and only waits on the slots its Host tiles use.
//  3. Small batches split rows into KV chunks; after a third barrier the grid merges them.
//
// Deduplication is the main gain once PCIe is saturated: MTP tokens of one request select largely
// the same pages, and native direct reads move every reference separately. Pages beyond the
// staging capacity keep their Host page id and are read directly, so overflow stays correct.
// The cache must use the combined [page, 576] layout.
template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
struct MLAHostBatchParams {
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;

  DTypeQ* q_nope;
  DTypeQ* q_pe;
  DTypeKV* ckv;            // combined cache [num_pages, 576]; kpe is ckv + 512
  DTypeO* final_o;
  float* final_lse;
  IdType* kv_indptr;
  IdType* kv_indices;

  IdType* remap;           // [nnz] rewritten indices, Host entries first within each row
  IdType* unique_pages;    // [nnz] Host page of each staging slot
  uint32_t* host_slot;     // [num_pages - host_page_start] 0 free, 1 claiming, slot + 2
  uint32_t* state;         // counters, per-row data, per-slot flags and claims, item claims
  DTypeKV* staging;        // [staging_capacity + num_pools * pool_capacity, 576]; cache slots last
  DTypeO* partial_o;       // [items, CTA_TILE_Q, HEAD_DIM_CKV] when rows are split
  float* partial_lse;      // [items, CTA_TILE_Q]

  // Cross-launch LRU cache of Host pages (all null and pool_capacity 0 when disabled). Pool p owns
  // cache slots [p * pool_capacity, (p + 1) * pool_capacity), stored after the staging slots, and
  // serves the contiguous rows [pool_row_start[p], pool_row_start[p + 1]), e.g. one request's
  // MTP tokens. Persistent state is per layer; stamps hold the epoch + 1 of a slot's last use.
  IdType* row_pool;        // [num_rows] pool of each row
  IdType* pool_row_start;  // [num_pools + 1]
  int32_t* resident;       // [num_pages - host_page_start] cache slot of a Host page, or -1
  int32_t* slot_page;      // [num_pools * pool_capacity] Host page in a cache slot, or -1
  uint32_t* slot_stamp;    // [num_pools * pool_capacity] 0 = empty
  int32_t* victim_of;      // [staging_capacity] cache slot a staged page is copied to, or -1
  int32_t* slot_pool;      // [staging_capacity] pool of the row that claimed a staged page
  uint32_t* row_slot_base; // [num_rows + 1] first staging slot of each row's claims
  uint32_t* pool_flags;    // [num_pools] epoch + 1 once the pool's victims are chosen

  uint_fastdiv num_heads;
  uint32_t num_rows;
  uint32_t num_head_tiles;
  uint32_t full_rows;       // rows [0, full_rows) run as one item per head tile
  uint32_t num_chunks;      // HBM chunks per later (tail) row
  uint32_t partial_capacity;
  uint32_t staging_capacity;
  uint32_t fetch_ctas;
  uint32_t num_pools;
  uint32_t pool_capacity;
  uint32_t host_item_entries;  // smallest Host item of a tail row (see host_batch_host_chunks)
  bool host_split;             // tail rows put Host entries in separate items
  // false: attention reads uncached Host pages in place, in their original order, with no
  // staging or wait (fast far tiers such as NVLink); with the cache on, fetch CTAs only copy the
  // misses into their victims for later launches. Without the cache the launch skips prepare.
  bool stage;

  uint32_t q_nope_stride_n;
  uint32_t q_nope_stride_h;
  uint32_t q_pe_stride_n;
  uint32_t q_pe_stride_h;
  uint32_t o_stride_n;
  uint32_t o_stride_h;

  IdType host_page_start;
  float sm_scale;
  bool return_lse_base_on_e;
};

namespace mla {

// Staging slots copied and published together.
constexpr uint32_t kHostBatchFetchRows = 64;
// state layout: counters; per row its Host entry count, its claimed page count and its first Host
// item; the row of each Host item; one ready flag per kHostBatchFetchRows staging slots; one claim
// per attention item. Flags and claims hold epoch + 1, so they never need clearing.
// Experiment: a nonzero state[kHostBatchReuseMeta] makes the launch reuse the remap, slots and
// item lists the previous launch left, skipping prepare, resolve and their two grid barriers.
// This bounds what moving that work into the top-k stage could save. Set from the host only.
constexpr uint32_t kHostBatchReuseMeta = 0;
constexpr uint32_t kHostBatchUnique = 1;
constexpr uint32_t kHostBatchExit = 2;
constexpr uint32_t kHostBatchEpoch = 3;
constexpr uint32_t kHostBatchHostCounts = 4;
// Shared scratch slots used after the partition phase.
constexpr uint32_t kHostBatchScratchEpoch = 0;
constexpr uint32_t kHostBatchScratchClaimed = 1;
// host_slot packs the claiming row and the page's rank among that row's claims (see
// prepare_host_batch_rows); kv lengths and row counts are bounded accordingly.
constexpr uint32_t kHostBatchRankBits = 16;
// Largest LRU pool: victim selection keeps two scans of the pool's slots in shared memory.
constexpr uint32_t kHostBatchMaxPool = 8192;

template <typename Params>
__device__ __forceinline__ bool host_batch_caching(const Params& params) {
  return params.pool_capacity != 0;
}

// Cache slot c is staging row staging_capacity + c, so cached entries take the staged encoding
// -(row + 1) and the tile loads need no change; only staged rows below staging_capacity wait.
template <typename Params>
__device__ __forceinline__ typename Params::IdType host_batch_cached_entry(const Params& params,
                                                                           int32_t slot) {
  return -static_cast<typename Params::IdType>(params.staging_capacity + slot) - 1;
}

template <typename Params>
__device__ __forceinline__ uint32_t* host_batch_row_claims(const Params& params) {
  return params.state + kHostBatchHostCounts + params.num_rows;
}

template <typename Params>
__device__ __forceinline__ uint32_t* host_batch_host_items(const Params& params) {
  return host_batch_row_claims(params) + params.num_rows;
}

template <typename Params>
__device__ __forceinline__ uint32_t* host_batch_host_item_rows(const Params& params) {
  return host_batch_host_items(params) + params.num_rows + 1;
}

template <typename Params>
__device__ __forceinline__ uint32_t* host_batch_flags(const Params& params) {
  return host_batch_host_item_rows(params) + params.partial_capacity;
}

template <typename Params>
__device__ __forceinline__ uint32_t* host_batch_claims(const Params& params) {
  return host_batch_flags(params) + ceil_div(params.staging_capacity, kHostBatchFetchRows);
}

__device__ __forceinline__ uint32_t host_batch_ld_acquire(const uint32_t* address) {
  uint32_t value;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

// Tail rows (see BatchMLAHostBatchAttention) split their HBM entries into num_chunks items and
// their Host entries into Host items of at least `min_entries` (params.host_item_entries), at most
// `max_items` per head tile. HBM items come first in the item list and Host items follow in row
// order, matching the fetch order, so most Host work runs while later rows are still in flight
// and little is left once the last pages land. Placing Host entries in each row's leading KV
// chunk instead left a ~22-tile item per row for after its data arrived (64 rows: ~100 us tail);
// spreading them over every chunk made every item wait while holding its SM.
// Small items pay off only while the link is slow: each also loads Q and writes a partial state
// for the merge, so over PCIe 64 entries keep the tail short, but over NVLink (~10x the bandwidth)
// that overhead dominates. There the extra items also overflow the wave the HBM chunks were sized
// for, so host_split off, which leaves Host entries in the row's chunks, is faster still.
template <uint32_t CTA_TILE_KV>
__device__ __forceinline__ uint2 host_batch_host_chunks(uint32_t host, uint32_t max_items,
                                                        uint32_t min_entries) {
  if (host == 0) return make_uint2(0, 1);
  const uint32_t items = max(1u, min(max_items, ceil_div(host, min_entries)));
  const uint32_t length = round_up(ceil_div(host, items), CTA_TILE_KV);
  return make_uint2(ceil_div(host, length), length);
}

// Host entries a tail row splits off into Host items; with host_split off they stay in the row's
// chunks (see host_batch_host_chunks for when each wins).
template <typename Params>
__device__ __forceinline__ uint32_t host_batch_split_host(const Params& params, uint32_t row) {
  return params.host_split ? params.state[kHostBatchHostCounts + row] : 0;
}

template <typename Params>
__device__ __forceinline__ uint32_t host_batch_max_host_items(const Params& params) {
  const uint32_t tail_tiles = (params.num_rows - params.full_rows) * params.num_head_tiles;
  return tail_tiles ? params.partial_capacity / tail_tiles - params.num_chunks : 0;
}

// Rewrites each row of kv_indices into remap with its Host entries first, as page ids for
// resolve_host_batch_slots, and its HBM entries after. The first row to CAS a page claims it and records (row,
// rank among the row's claims) in host_slot; after the grid barrier a page's staging slot is the
// claims of earlier rows plus its rank. Slots, and so the fetch order, then follow row order even
// though rows are prepared concurrently: slots in claim order made every item wait for the whole
// transfer at small batch, and per-row fetch units were sparse after dedup and slower (~21 vs
// ~26 GB/s). Each thread handles PER_THREAD entries per pass so its loads and CASes overlap, and
// each warp takes its ranks with one atomicAdd; per-entry claims serialized eight contended round
// trips per warp. Waiting here for another row's claim instead of resolving after the barrier took
// 20-30 us for one request's 8 MTP rows, which share half their pages.
// With the LRU cache, a Host entry whose page is resident becomes a cached entry, placed with the
// HBM entries (it needs no wait) and stamped with this launch's `stamp`; only misses are claimed.
// Nothing changes residency before the grid barrier, so both passes see the same hits.
template <typename KTraits, typename Params>
__device__ __forceinline__ void prepare_host_batch_rows(const Params& params, uint32_t* scratch,
                                                        uint32_t stamp) {
  using IdType = typename Params::IdType;
  constexpr uint32_t THREADS = KTraits::NUM_THREADS;
  constexpr uint32_t WARPS = THREADS / 32;
  constexpr uint32_t PER_THREAD = 8;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t lane = tid % 32, warp = tid / 32;
  const uint32_t lower = (1u << lane) - 1;
#pragma unroll 1
  for (uint32_t row = blockIdx.y; row < params.num_rows; row += gridDim.y) {
    const IdType begin = params.kv_indptr[row];
    const uint32_t length = params.kv_indptr[row + 1] - begin;

    // HBM entries are placed after all Host entries. Counts Host entries in the high half and
    // the uncached ones among them in the low half (lengths are below 2^16, see the launcher).
    uint32_t count = 0;
#pragma unroll 1
    for (uint32_t j = tid; j < length; j += THREADS) {
      const IdType page = params.kv_indices[begin + j];
      if (page >= params.host_page_start) {
        const bool cached =
            host_batch_caching(params) && params.resident[page - params.host_page_start] >= 0;
        count += (1u << 16) + !cached;
      }
    }
#pragma unroll
    for (uint32_t offset = 16; offset > 0; offset /= 2) {
      count += __shfl_xor_sync(0xffffffff, count, offset);
    }
    if (lane == 0) scratch[warp] = count;
    __syncthreads();
    uint32_t counts = 0;
#pragma unroll
    for (uint32_t w = 0; w < WARPS; ++w) counts += scratch[w];
    __syncthreads();
    const uint32_t host_total = counts & 0xffff;

    // Rows with no Host entry need no reordering: remap is then kv_indices verbatim (HBM
    // entries land at begin + local_rank). Skipping the ballots and the cross-warp prefix sum
    // is what makes an all-HBM launch cost about the same as the native kernel.
    if (counts == 0) {
#pragma unroll 1
      for (uint32_t j = tid; j < length; j += THREADS) {
        params.remap[begin + j] = params.kv_indices[begin + j];
      }
      if (tid == 0) params.state[kHostBatchHostCounts + row] = 0;
      __syncthreads();
      continue;
    }

    uint32_t host_base = 0, local_base = 0;
#pragma unroll 1
    for (uint32_t offset = 0; offset < length; offset += PER_THREAD * THREADS) {
      IdType pages[PER_THREAD];
      uint32_t claimed[PER_THREAD], host_masks[PER_THREAD], local_masks[PER_THREAD];
#pragma unroll
      for (uint32_t i = 0; i < PER_THREAD; ++i) {
        const uint32_t j = offset + i * THREADS + tid;
        pages[i] = j < length ? params.kv_indices[begin + j] : -1;
      }
      if (host_batch_caching(params)) {
#pragma unroll
        for (uint32_t i = 0; i < PER_THREAD; ++i) {
          if (pages[i] >= params.host_page_start) {
            const int32_t slot = params.resident[pages[i] - params.host_page_start];
            if (slot >= 0) {
              params.slot_stamp[slot] = stamp;
              pages[i] = host_batch_cached_entry(params, slot);
            }
          }
        }
      }
      // host_slot: 0 free, 1 claiming, (row << kHostBatchRankBits | rank) + 2 claimed.
#pragma unroll
      for (uint32_t i = 0; i < PER_THREAD; ++i) {
        claimed[i] = pages[i] >= params.host_page_start
                         ? atomicCAS(params.host_slot + (pages[i] - params.host_page_start), 0u, 1u)
                         : 2u;
      }
      uint32_t fresh_masks[PER_THREAD], fresh_total = 0;
#pragma unroll
      for (uint32_t i = 0; i < PER_THREAD; ++i) {
        fresh_masks[i] =
            __ballot_sync(0xffffffff, pages[i] >= params.host_page_start && claimed[i] == 0);
        fresh_total += __popc(fresh_masks[i]);
      }
      uint32_t first = 0;
      if (lane == 0 && fresh_total) {
        first = atomicAdd(host_batch_row_claims(params) + row, fresh_total);
      }
      first = __shfl_sync(0xffffffff, first, 0);
#pragma unroll
      for (uint32_t i = 0; i < PER_THREAD; ++i) {
        if (fresh_masks[i] >> lane & 1) {
          const uint32_t rank = first + __popc(fresh_masks[i] & lower);
          atomicExch(params.host_slot + (pages[i] - params.host_page_start),
                     (row << kHostBatchRankBits | rank) + 2);
        }
        first += __popc(fresh_masks[i]);
        host_masks[i] = __ballot_sync(0xffffffff, pages[i] >= params.host_page_start);
        // Cached entries are negative but never -1, the padding of short passes.
        local_masks[i] =
            __ballot_sync(0xffffffff, pages[i] != -1 && pages[i] < params.host_page_start);
        if (lane == 0) {
          scratch[(i * WARPS + warp) * 2] = __popc(host_masks[i]);
          scratch[(i * WARPS + warp) * 2 + 1] = __popc(local_masks[i]);
        }
      }
      __syncthreads();
#pragma unroll
      for (uint32_t i = 0; i < PER_THREAD; ++i) {
        uint32_t host_rank = host_base + __popc(host_masks[i] & lower);
        uint32_t local_rank = local_base + __popc(local_masks[i] & lower);
#pragma unroll
        for (uint32_t w = 0; w < WARPS; ++w) {
          const uint32_t host_count = scratch[(i * WARPS + w) * 2];
          const uint32_t local_count = scratch[(i * WARPS + w) * 2 + 1];
          if (w < warp) {
            host_rank += host_count;
            local_rank += local_count;
          }
          host_base += host_count;
          local_base += local_count;
        }
        if (pages[i] >= params.host_page_start) {
          params.remap[begin + host_rank] = pages[i];
        } else if (pages[i] != -1) {
          params.remap[begin + host_total + local_rank] = pages[i];
        }
      }
      __syncthreads();
    }
    if (tid == 0) params.state[kHostBatchHostCounts + row] = host_total;
  }
}

// prepare_host_batch_rows without staging (params.stage false, cache on): entries keep their
// order, cached ones are rewritten and stamped, and uncached Host pages stay page ids for the
// attention to read in place; they are only claimed, so the fetch CTAs can copy each once into
// the cache. Rows then have no Host part, so items split them like all-HBM rows.
template <typename KTraits, typename Params>
__device__ __forceinline__ void prepare_host_batch_rows_in_place(const Params& params,
                                                                 uint32_t stamp) {
  using IdType = typename Params::IdType;
  constexpr uint32_t THREADS = KTraits::NUM_THREADS;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t lane = tid % 32;
#pragma unroll 1
  for (uint32_t row = blockIdx.y; row < params.num_rows; row += gridDim.y) {
    const IdType begin = params.kv_indptr[row];
    const uint32_t length = params.kv_indptr[row + 1] - begin;
#pragma unroll 1
    for (uint32_t base = 0; base < length; base += THREADS) {
      const uint32_t j = base + tid;
      IdType page = j < length ? params.kv_indices[begin + j] : -1;
      bool fresh = false;
      if (page >= params.host_page_start) {
        const int32_t slot = params.resident[page - params.host_page_start];
        if (slot >= 0) {
          params.slot_stamp[slot] = stamp;
          params.remap[begin + j] = host_batch_cached_entry(params, slot);
        } else {
          fresh = atomicCAS(params.host_slot + (page - params.host_page_start), 0u, 1u) == 0;
          params.remap[begin + j] = page;
        }
      } else if (j < length) {
        params.remap[begin + j] = page;
      }
      const uint32_t mask = __ballot_sync(0xffffffff, fresh);
      uint32_t first = 0;
      if (lane == 0 && mask) first = atomicAdd(host_batch_row_claims(params) + row, __popc(mask));
      first = __shfl_sync(0xffffffff, first, 0);
      if (fresh) {
        const uint32_t rank = first + __popc(mask & ((1u << lane) - 1));
        atomicExch(params.host_slot + (page - params.host_page_start),
                   (row << kHostBatchRankBits | rank) + 2);
      }
    }
    if (tid == 0) params.state[kHostBatchHostCounts + row] = 0;
  }
}

// Exclusive scan of value(i) for i < n into out[i] by one CTA; returns the total.
template <uint32_t THREADS, typename Value>
__device__ __forceinline__ uint32_t host_batch_block_scan(uint32_t n, Value value, uint32_t* out) {
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  __shared__ uint32_t warp_sums[THREADS / 32], carry;
  if (tid == 0) carry = 0;
#pragma unroll 1
  for (uint32_t base = 0; base < n; base += THREADS) {
    const uint32_t i = base + tid;
    const uint32_t count = i < n ? value(i) : 0;
    uint32_t sum = count;
#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset *= 2) {
      const uint32_t other = __shfl_up_sync(0xffffffff, sum, offset);
      if (tid % 32 >= offset) sum += other;
    }
    if (tid % 32 == 31) warp_sums[tid / 32] = sum;
    __syncthreads();
    uint32_t prefix = carry;
    for (uint32_t w = 0; w < tid / 32; ++w) prefix += warp_sums[w];
    if (i < n) out[i] = prefix + sum - count;
    __syncthreads();
    if (tid == THREADS - 1) carry = prefix + sum;
    __syncthreads();
  }
  return carry;
}

// After the grid barrier: turn each Host entry's page id into its staging slot, -(slot + 1), and
// let claiming rows record the page of each slot. Every CTA scans the per-row claim counts into
// shared memory itself, which saves a grid barrier. Returns the number of unique Host pages.
template <typename KTraits, typename Params>
__device__ __forceinline__ uint32_t resolve_host_batch_slots(const Params& params, uint8_t* smem) {
  using IdType = typename Params::IdType;
  constexpr uint32_t THREADS = KTraits::NUM_THREADS;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  uint32_t* row_base = reinterpret_cast<uint32_t*>(smem);
  const uint32_t* row_claims = host_batch_row_claims(params);
  const uint32_t unique = host_batch_block_scan<THREADS>(
      params.num_rows, [&](uint32_t row) { return row_claims[row]; }, row_base);
  // No Host page anywhere: nothing to rewrite, and no Host items to list. The caller then skips
  // the second grid barrier and the fetch entirely.
  if (unique == 0) return 0;
#pragma unroll 1
  for (uint32_t row = blockIdx.y; row < params.num_rows; row += gridDim.y) {
    const IdType begin = params.kv_indptr[row];
    // Host entries fill each row's leading positions, or anywhere when not staging.
    const uint32_t host = params.stage ? params.state[kHostBatchHostCounts + row]
                                       : params.kv_indptr[row + 1] - begin;
#pragma unroll 1
    for (uint32_t j = tid; j < host; j += THREADS) {
      const IdType page = params.remap[begin + j];
      if (!params.stage && page < params.host_page_start) continue;
      const uint32_t claim = params.host_slot[page - params.host_page_start] - 2;
      const uint32_t owner = claim >> kHostBatchRankBits;
      const uint32_t slot = row_base[owner] + (claim & ((1u << kHostBatchRankBits) - 1));
      if (owner == row) {
        params.unique_pages[slot] = page;
        if (host_batch_caching(params) && slot < params.staging_capacity) {
          params.slot_pool[slot] = params.row_pool[row];
        }
      }
      if (params.stage && slot < params.staging_capacity) {
        params.remap[begin + j] = -static_cast<IdType>(slot) - 1;
      }
    }
  }
  // CTA 0 lists the Host items of the tail rows, and publishes the row bases that delimit each
  // LRU pool's staged pages.
  if (blockIdx.y == 0) {
    if (host_batch_caching(params)) {
      for (uint32_t row = tid; row < params.num_rows; row += THREADS) {
        params.row_slot_base[row] = row_base[row];
      }
      if (tid == 0) params.row_slot_base[params.num_rows] = unique;
    }
    const uint32_t max_items = host_batch_max_host_items(params);
    const uint32_t tail_rows = params.num_rows - params.full_rows;
    uint32_t* host_items = host_batch_host_items(params) + params.full_rows;
    const auto items = [&](uint32_t t) {
      return host_batch_host_chunks<KTraits::CTA_TILE_KV>(
                 host_batch_split_host(params, params.full_rows + t), max_items,
                 params.host_item_entries)
                 .x *
             params.num_head_tiles;
    };
    const uint32_t total = host_batch_block_scan<THREADS>(tail_rows, items, host_items);
    if (tid == 0) host_batch_host_items(params)[params.num_rows] = total;
    __threadfence_block();
    __syncthreads();
    uint32_t* item_rows = host_batch_host_item_rows(params);
    // A warp per tail row.
#pragma unroll 1
    for (uint32_t t = tid / 32; t < tail_rows; t += THREADS / 32) {
      const uint32_t first = host_items[t], count = items(t);
      for (uint32_t k = tid % 32; k < count; k += 32) item_rows[first + k] = params.full_rows + t;
    }
  }
  return unique;
}

// Copy staged pages Host -> HBM in slot order, kHostBatchFetchRows slots per chunk, publishing a
// ready flag per chunk. Slots follow row order (see prepare_host_batch_rows), as do attention
// items. cp.async into shared memory keeps enough PCIe reads in flight: register loads
// (ld.global.cg) reached only ~15 GB/s on sparse 1152-byte rows, cp.async ~26 GB/s. Two buffers
// overlap the next chunk's reads with the copy-out of the current one when shared memory allows.
template <typename KTraits, typename Params>
__device__ __forceinline__ void fetch_host_batch_pages(const Params& params, uint8_t* smem,
                                                       uint32_t epoch, uint32_t unique) {
  using DTypeKV = typename Params::DTypeKV;
  constexpr uint32_t ROW = KTraits::HEAD_DIM_CKV + KTraits::HEAD_DIM_KPE;
  constexpr uint32_t VECS_PER_ROW = ROW * sizeof(DTypeKV) / 16;
  constexpr uint32_t CHUNK_VECS = kHostBatchFetchRows * VECS_PER_ROW;
  constexpr uint32_t BUFFERS =
      sizeof(typename KTraits::SharedStorage) >= 2 * CHUNK_VECS * 16 ? 2 : 1;
  static_assert(sizeof(typename KTraits::SharedStorage) >= CHUNK_VECS * 16);
  static_assert(CHUNK_VECS % KTraits::NUM_THREADS == 0);
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t fetch_id = blockIdx.y - (gridDim.y - params.fetch_ctas);
  const uint32_t count = params.stage ? min(unique, params.staging_capacity) : unique;
  uint32_t* flags = host_batch_flags(params);
  uint4* buffer = reinterpret_cast<uint4*>(smem);

  auto issue = [&](uint32_t chunk, uint32_t buf) {
#pragma unroll
    for (uint32_t i = 0; i < CHUNK_VECS / KTraits::NUM_THREADS; ++i) {
      const uint32_t v = i * KTraits::NUM_THREADS + tid;
      const uint32_t slot = chunk * kHostBatchFetchRows + v / VECS_PER_ROW;
      if (slot < count) {
        const auto* source = reinterpret_cast<const uint4*>(
            params.ckv + static_cast<int64_t>(params.unique_pages[slot]) * ROW);
        cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffer + buf * CHUNK_VECS + v,
                                                       source + v % VECS_PER_ROW);
      }
    }
    cp_async::commit_group();
  };

  uint32_t chunk = fetch_id;
  if (chunk * kHostBatchFetchRows < count) issue(chunk, 0);
#pragma unroll 1
  for (uint32_t k = 0; chunk * kHostBatchFetchRows < count; ++k, chunk += params.fetch_ctas) {
    const uint32_t next = chunk + params.fetch_ctas;
    if (BUFFERS == 2 && next * kHostBatchFetchRows < count) {
      issue(next, (k + 1) % 2);
      cp_async::wait_group<1>();
    } else {
      cp_async::wait_group<0>();
    }
    __syncthreads();
    const uint32_t buf = BUFFERS == 2 ? k % 2 : 0;
    auto* destination = reinterpret_cast<uint4*>(
        params.staging + static_cast<int64_t>(chunk) * kHostBatchFetchRows * ROW);
#pragma unroll
    for (uint32_t i = 0; i < CHUNK_VECS / KTraits::NUM_THREADS; ++i) {
      const uint32_t v = i * KTraits::NUM_THREADS + tid;
      if (params.stage && chunk * kHostBatchFetchRows + v / VECS_PER_ROW < count) {
        destination[v] = buffer[buf * CHUNK_VECS + v];
      }
    }
    // The resolve step is done, so the dedup entries can be cleared for the next launch.
    if (tid < kHostBatchFetchRows && chunk * kHostBatchFetchRows + tid < count) {
      params.host_slot[params.unique_pages[chunk * kHostBatchFetchRows + tid] -
                       params.host_page_start] = 0;
    }
    if (params.stage) {
      __threadfence();
      __syncthreads();
      if (tid == 0) {
        asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(flags + chunk), "r"(epoch + 1)
                     : "memory");
      }
    }
    // Fill the LRU cache from the same shared-memory copy, after the attention items waiting on
    // this chunk were released: one more HBM write per staged page, and no reread. The victims
    // come from the pools' LRU items, which workers run first (see run_host_batch_items); the
    // next chunk's reads are already in flight while this one waits for them.
    if (host_batch_caching(params)) {
      __shared__ int32_t cache_rows[kHostBatchFetchRows];
      if (tid < kHostBatchFetchRows) {
        const uint32_t slot = chunk * kHostBatchFetchRows + tid;
        int32_t victim = -1;
        if (slot < count) {
          const uint32_t* pool_flag = params.pool_flags + params.slot_pool[slot];
          while (host_batch_ld_acquire(pool_flag) != epoch + 1) __nanosleep(64);
          victim = params.victim_of[slot];
        }
        cache_rows[tid] = victim;
      }
      __syncthreads();
#pragma unroll
      for (uint32_t i = 0; i < CHUNK_VECS / KTraits::NUM_THREADS; ++i) {
        const uint32_t v = i * KTraits::NUM_THREADS + tid;
        const int32_t victim = cache_rows[v / VECS_PER_ROW];
        if (victim >= 0) {
          reinterpret_cast<uint4*>(
              params.staging +
              static_cast<int64_t>(params.staging_capacity + victim) * ROW)[v % VECS_PER_ROW] =
              buffer[buf * CHUNK_VECS + v];
        }
      }
      // The buffer is refilled next.
      __syncthreads();
    }
    if (BUFFERS == 1 && next * kHostBatchFetchRows < count) issue(next, 0);
  }
  // Overflow pages were never staged; clear their dedup entries too.
#pragma unroll 1
  for (uint32_t slot = count + fetch_id * KTraits::NUM_THREADS + tid; slot < unique;
       slot += params.fetch_ctas * KTraits::NUM_THREADS) {
    params.host_slot[params.unique_pages[slot] - params.host_page_start] = 0;
  }
}

// Requires the combined [page, 576] layout (kpe right after ckv), so one base select and one
// 64-bit offset serve both parts.
template <typename KTraits, typename Params>
__device__ __forceinline__ void resolve_host_batch_row(const Params& params,
                                                       typename KTraits::IdType page,
                                                       typename KTraits::DTypeKV*& ckv_ptr,
                                                       typename KTraits::DTypeKV*& kpe_ptr) {
  constexpr uint32_t ROW = KTraits::HEAD_DIM_CKV + KTraits::HEAD_DIM_KPE;
  const bool staged = page < 0;
  const int64_t row = staged ? -static_cast<int64_t>(page) - 1 : page;
  ckv_ptr = (staged ? params.staging : params.ckv) + row * ROW +
            (threadIdx.x % 8) * upcast_size<typename KTraits::DTypeKV>();
  kpe_ptr = ckv_ptr + KTraits::HEAD_DIM_CKV;
}

// Warp 0 waits until every staged entry of a KV tile has landed; the barrier that precedes each
// tile load then releases the CTA. Keeping the wait out of the per-thread load path and out of a
// second copy of the tile loop avoids spilling the loop's shared-memory offsets.
template <typename KTraits, typename Params>
__device__ __forceinline__ void wait_host_batch_tile(const Params& params, const uint32_t* scratch,
                                                     uint32_t index_base, uint32_t kv_len,
                                                     uint32_t kv_base) {
  static_assert(KTraits::CTA_TILE_KV <= 64);
  if (threadIdx.y != 0 || threadIdx.z != 0) return;
#pragma unroll
  for (uint32_t i = 0; i < KTraits::CTA_TILE_KV; i += 32) {
    const uint32_t kv_idx = kv_base + i + threadIdx.x;
    const typename KTraits::IdType page = kv_idx < kv_len ? params.remap[index_base + kv_idx] : 0;
    // Cached entries (rows past staging_capacity) are resident already.
    if (page < 0 && static_cast<uint32_t>(-(page + 1)) < params.staging_capacity) {
      const uint32_t* flag = host_batch_flags(params) + (-(page + 1)) / kHostBatchFetchRows;
      while (host_batch_ld_acquire(flag) != scratch[kHostBatchScratchEpoch] + 1) __nanosleep(64);
    }
  }
}

// Rows of a KV tile one thread loads, following load_kv's thread layout.
template <typename KTraits>
constexpr uint32_t kHostBatchTileRows = KTraits::NUM_MMA_KV == 1 ? 1 : KTraits::NUM_MMA_KV / 2;

template <typename KTraits>
__device__ __forceinline__ uint32_t host_batch_tile_row(uint32_t j) {
  const uint32_t row = threadIdx.x / 8 + threadIdx.y * 4;
  return KTraits::NUM_MMA_KV == 1 ? row : row + (threadIdx.z + j * 2) * 16;
}

// Page ids are read one tile ahead (see attend_host_batch_tiles), so the load below does not
// wait on them.
template <typename KTraits, typename Params>
__device__ __forceinline__ void read_host_batch_pages(const Params& params, uint32_t index_base,
                                                      uint32_t kv_len, uint32_t kv_base,
                                                      typename KTraits::IdType* pages) {
#pragma unroll
  for (uint32_t j = 0; j < kHostBatchTileRows<KTraits>; ++j) {
    const uint32_t kv_idx = kv_base + host_batch_tile_row<KTraits>(j);
    pages[j] = kv_idx < kv_len ? params.remap[index_base + kv_idx] : 0;
  }
}

// load_kv for page size one with staged entries resolved per row.
template <typename KTraits, typename Params>
__device__ __forceinline__ void load_host_batch_kv(typename KTraits::SharedStorage* smem_storage,
                                                   const Params& params,
                                                   const typename KTraits::IdType* pages,
                                                   const uint32_t kv_len, const uint32_t kv_base,
                                                   const uint32_t stage_idx) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t UPCAST_STRIDE_CKV = KTraits::UPCAST_STRIDE_CKV;
  constexpr uint32_t UPCAST_STRIDE_KPE = KTraits::UPCAST_STRIDE_KPE;
  const uint32_t lane_idx = threadIdx.x;
  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[stage_idx]);
  smem_t<KTraits::SWIZZLE_MODE_KPE> kpe_smem(smem_storage->kpe_p_smem[stage_idx]);
  if (KTraits::NUM_MMA_KV == 1 && threadIdx.z != 0) return;
#pragma unroll
  for (uint32_t j = 0; j < kHostBatchTileRows<KTraits>; ++j) {
    const uint32_t smem_row = host_batch_tile_row<KTraits>(j);
    const bool valid = kv_base + smem_row < kv_len;
    DTypeKV *ckv_ptr, *kpe_ptr;
    resolve_host_batch_row<KTraits>(params, pages[j], ckv_ptr, kpe_ptr);
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 4; ++mma_d) {
      uint32_t ckv_smem_offset_w = ckv_smem.template get_permuted_offset<UPCAST_STRIDE_CKV>(
          smem_row, 8 * mma_d + lane_idx % 8);
      ckv_smem.load_128b_async<SharedMemFillMode::kFillZero>(ckv_smem_offset_w, ckv_ptr, valid);
      ckv_ptr += 8 * upcast_size<DTypeKV>();
    }
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_KPE / 4; ++mma_d) {
      uint32_t kpe_smem_offset_w = kpe_smem.template get_permuted_offset<UPCAST_STRIDE_KPE>(
          smem_row, 8 * mma_d + lane_idx % 8);
      kpe_smem.load_128b_async<SharedMemFillMode::kFillZero>(kpe_smem_offset_w, kpe_ptr, valid);
      kpe_ptr += 8 * upcast_size<DTypeKV>();
    }
  }
}

// Merge the tail rows' item states after the grid barrier; every CTA takes (row tile, head) units
// so the partial reads spread over all SMs. Loads are batched because each state is an independent
// L2 read. Partial states: HBM item c of tail row tile t at t * num_chunks + c, Host items after
// all HBM states in item order.
template <typename KTraits, typename Params>
__device__ __forceinline__ void merge_host_batch_chunks(const Params& params) {
  using DTypeO = typename KTraits::DTypeO;
  // 32 features per thread: every pass pays a chain of dependent metadata loads, so fewer, wider
  // passes are faster.
  constexpr uint32_t VEC_SIZE = 32;
  constexpr uint32_t BATCH = 8;
  constexpr uint32_t THREADS_PER_HEAD = KTraits::HEAD_DIM_CKV / VEC_SIZE;
  constexpr uint32_t HEADS_PER_PASS = KTraits::NUM_THREADS / THREADS_PER_HEAD;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t feature = (tid % THREADS_PER_HEAD) * VEC_SIZE;
  const uint32_t tail_tiles = (params.num_rows - params.full_rows) * params.num_head_tiles;
  const uint32_t max_items = host_batch_max_host_items(params);
  const uint32_t units = tail_tiles * KTraits::CTA_TILE_Q;
#pragma unroll 1
  for (uint32_t unit = blockIdx.y * HEADS_PER_PASS + tid / THREADS_PER_HEAD; unit < units;
       unit += gridDim.y * HEADS_PER_PASS) {
    const uint32_t tile = unit / KTraits::CTA_TILE_Q, h = unit % KTraits::CTA_TILE_Q;
    const uint32_t row = params.full_rows + tile / params.num_head_tiles;
    const uint32_t head_tile = tile % params.num_head_tiles;
    const uint32_t length = params.kv_indptr[row + 1] - params.kv_indptr[row];
    const uint32_t host = host_batch_split_host(params, row);
    const uint32_t hbm_chunk = round_up(ceil_div(length - host, params.num_chunks), KTraits::CTA_TILE_KV);
    const uint32_t hbm_items = hbm_chunk ? ceil_div(length - host, hbm_chunk) : 0;
    const uint32_t host_items =
        host_batch_host_chunks<KTraits::CTA_TILE_KV>(host, max_items, params.host_item_entries).x;
    const uint32_t host_first = tail_tiles * params.num_chunks +
                                host_batch_host_items(params)[row] + head_tile * host_items;
    const uint32_t states = hbm_items + host_items;
    state_t<VEC_SIZE> st;
#pragma unroll 1
    for (uint32_t base = 0; base < states; base += BATCH) {
      constexpr uint32_t VECS = VEC_SIZE * sizeof(DTypeO) / 16;
      uint4 raw[BATCH][VECS];
      float lse[BATCH];
#pragma unroll
      for (uint32_t i = 0; i < BATCH; ++i) {
        const uint32_t k = min(base + i, states - 1);
        const uint32_t partial =
            (k < hbm_items ? tile * params.num_chunks + k : host_first + k - hbm_items) *
                KTraits::CTA_TILE_Q + h;
        const auto* source = reinterpret_cast<const uint4*>(
            params.partial_o + static_cast<int64_t>(partial) * KTraits::HEAD_DIM_CKV + feature);
#pragma unroll
        for (uint32_t v = 0; v < VECS; ++v) raw[i][v] = __ldcg(source + v);
        lse[i] = __ldcg(params.partial_lse + partial);
      }
#pragma unroll
      for (uint32_t i = 0; i < BATCH; ++i) {
        if (base + i < states) {
          vec_t<float, VEC_SIZE> o;
#pragma unroll
          for (uint32_t j = 0; j < VEC_SIZE; ++j) {
            o[j] = float(reinterpret_cast<const DTypeO*>(raw[i])[j]);
          }
          st.merge(o, lse[i], 1.f);
        }
      }
    }
    st.normalize();
    const uint32_t head = head_tile * KTraits::CTA_TILE_Q + h;
    st.o.cast_store(params.final_o + row * params.o_stride_n + head * params.o_stride_h + feature);
    if (params.final_lse && tid % THREADS_PER_HEAD == 0) {
      params.final_lse[row * params.num_heads + head] =
          st.get_lse() * (params.return_lse_base_on_e ? math::loge2 : 1.f);
    }
  }
}

// The native MLA tile loop over the KV tiles of one item, last tile first. Tiles below
// host_tiles hold Host entries; HBM tiles therefore run before any wait on PCIe. The next tile's
// page ids are read at the top of each iteration: native load_kv hides that latency behind the
// PV MMAs it schedules past the barrier, which the conditional Host wait here prevents.
template <typename KTraits, typename Params>
__device__ __forceinline__ void attend_host_batch_tiles(
    const Params& params, typename KTraits::SharedStorage& storage,
    typename KTraits::AttentionVariant& variant, const uint32_t* scratch, uint32_t index_base,
    uint32_t kv_len, int host_tiles, float (*s_frag)[8], float (*o_frag)[8], float* m, float* d) {
  constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  constexpr int32_t NUM_STAGES = KTraits::NUM_STAGES;
  typename KTraits::IdType pages[kHostBatchTileRows<KTraits>];
  int kv_tile_idx = ceil_div(kv_len, CTA_TILE_KV) - 1;
  const int mask_tile_idx = kv_len / CTA_TILE_KV;
#pragma unroll
  for (int stage_idx = 0; stage_idx < NUM_STAGES; ++stage_idx) {
    if (kv_tile_idx - stage_idx >= 0 && kv_tile_idx - stage_idx < host_tiles) {
      wait_host_batch_tile<KTraits>(params, scratch, index_base, kv_len,
                                    (kv_tile_idx - stage_idx) * CTA_TILE_KV);
    }
  }
  __syncthreads();
#pragma unroll
  for (int stage_idx = 0; stage_idx < NUM_STAGES; ++stage_idx) {
    if (kv_tile_idx - stage_idx >= 0) {
      read_host_batch_pages<KTraits>(params, index_base, kv_len,
                                     (kv_tile_idx - stage_idx) * CTA_TILE_KV, pages);
      load_host_batch_kv<KTraits>(&storage, params, pages, kv_len,
                                  (kv_tile_idx - stage_idx) * CTA_TILE_KV,
                                  (kv_tile_idx - stage_idx) % NUM_STAGES);
      cp_async::commit_group();
    }
  }

  // The noncausal mask only depends on the KV position, so no query offset is passed.
#pragma unroll 1
  for (; kv_tile_idx >= mask_tile_idx && kv_tile_idx > 0; --kv_tile_idx) {
    const int next = kv_tile_idx - NUM_STAGES;
    if (next >= 0) read_host_batch_pages<KTraits>(params, index_base, kv_len, next * CTA_TILE_KV, pages);
    cp_async::wait_group<NUM_STAGES - 1>();
    __syncthreads();
    compute_mla_qk<KTraits>(&storage, kv_tile_idx % NUM_STAGES, s_frag);
    logits_mask_<KTraits>(0, kv_tile_idx * CTA_TILE_KV, 1, kv_len, kv_len, params.num_heads,
                          s_frag);
    update_mdo_states_<KTraits>(&storage, kv_tile_idx % NUM_STAGES, variant, s_frag, o_frag, m, d);
    compute_mla_pv<KTraits>(&storage, kv_tile_idx % NUM_STAGES, s_frag, d, o_frag);
    if (next >= 0) {
      if (next < host_tiles) {
        wait_host_batch_tile<KTraits>(params, scratch, index_base, kv_len, next * CTA_TILE_KV);
      }
      __syncthreads();
      load_host_batch_kv<KTraits>(&storage, params, pages, kv_len, next * CTA_TILE_KV,
                                  next % NUM_STAGES);
      cp_async::commit_group();
    }
  }

#pragma unroll 1
  for (; kv_tile_idx + 1 > NUM_STAGES; --kv_tile_idx) {
    const int next = kv_tile_idx - NUM_STAGES;
    read_host_batch_pages<KTraits>(params, index_base, kv_len, next * CTA_TILE_KV, pages);
    cp_async::wait_group<NUM_STAGES - 1>();
    __syncthreads();
    compute_mla_qk<KTraits>(&storage, kv_tile_idx % NUM_STAGES, s_frag);
    update_mdo_states_<KTraits>(&storage, kv_tile_idx % NUM_STAGES, variant, s_frag, o_frag, m, d);
    compute_mla_pv<KTraits>(&storage, kv_tile_idx % NUM_STAGES, s_frag, d, o_frag);
    if (next < host_tiles) {
      wait_host_batch_tile<KTraits>(params, scratch, index_base, kv_len, next * CTA_TILE_KV);
    }
    __syncthreads();
    load_host_batch_kv<KTraits>(&storage, params, pages, kv_len, next * CTA_TILE_KV,
                                next % NUM_STAGES);
    cp_async::commit_group();
  }
  cp_async::wait_group<0>();
  __syncthreads();

#pragma unroll
  for (; kv_tile_idx >= 0; --kv_tile_idx) {
    compute_mla_qk<KTraits>(&storage, kv_tile_idx % NUM_STAGES, s_frag);
    logits_mask_<KTraits>(0, kv_tile_idx * CTA_TILE_KV, 1, kv_len, kv_len, params.num_heads,
                          s_frag);
    update_mdo_states_<KTraits>(&storage, kv_tile_idx % NUM_STAGES, variant, s_frag, o_frag, m, d);
    compute_mla_pv<KTraits>(&storage, kv_tile_idx % NUM_STAGES, s_frag, d, o_frag);
  }
}

// LRU item of one pool: give each page the pool's rows missed this launch a cache slot, taking
// the least recently used slots that no entry of this launch hits, and publish the pool's flag
// for the fetch loop, which copies the pages in (see fetch_host_batch_pages). Runs after the
// second grid barrier, when every hit of this launch is stamped and every staged page has its
// slot. Stamps are launch epochs, so this is exact LRU at launch granularity: entries hit in the
// same launch are equally recent, as in a per-step LRU list. A radix select over the C stamps
// finds the m-th oldest, and two block scans rank the victims in slot order. Not inlined: the
// tile loop that shares this item loop must keep its registers (see run_host_batch_items).
template <typename KTraits, typename Params>
__device__ __noinline__ void run_host_batch_lru(const Params& params, uint8_t* smem,
                                                uint32_t pool, uint32_t epoch, uint32_t unique) {
  constexpr uint32_t THREADS = KTraits::NUM_THREADS;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t stamp = epoch + 1;
  const uint32_t capacity = params.pool_capacity;
  const uint32_t count = params.stage ? min(unique, params.staging_capacity) : unique;
  const uint32_t first = min(params.row_slot_base[params.pool_row_start[pool]], count);
  const uint32_t last = min(params.row_slot_base[params.pool_row_start[pool + 1]], count);
  const uint32_t* stamps = params.slot_stamp + pool * capacity;
  // Slots hit this launch sort last and are never taken.
  const auto key = [&](uint32_t i) {
    const uint32_t value = stamps[i];
    return value == stamp ? ~0u : value;
  };
  __shared__ uint32_t histogram[256], selected[2];

  uint32_t evictable = 0;
  for (uint32_t i = tid; i < capacity; i += THREADS) evictable += key(i) != ~0u;
#pragma unroll
  for (uint32_t offset = 16; offset > 0; offset /= 2) {
    evictable += __shfl_xor_sync(0xffffffff, evictable, offset);
  }
  if (tid == 0) selected[0] = 0;
  __syncthreads();
  if (tid % 32 == 0) atomicAdd(selected, evictable);
  __syncthreads();
  const uint32_t victims = min(last - first, selected[0]);
  __syncthreads();

  if (victims > 0) {
    // The victims-th smallest key, eight bits per pass, most significant first.
    uint32_t prefix = 0, mask = 0, rank = victims;
#pragma unroll 1
    for (int shift = 24; shift >= 0; shift -= 8) {
      for (uint32_t b = tid; b < 256; b += THREADS) histogram[b] = 0;
      __syncthreads();
      for (uint32_t i = tid; i < capacity; i += THREADS) {
        const uint32_t k = key(i);
        if ((k & mask) == prefix) atomicAdd(histogram + (k >> shift & 255), 1u);
      }
      __syncthreads();
      if (tid == 0) {
        uint32_t below = 0, b = 0;
        while (below + histogram[b] < rank) below += histogram[b++];
        selected[0] = prefix | b << shift;
        selected[1] = rank - below;
      }
      __syncthreads();
      prefix = selected[0];
      rank = selected[1];
      mask |= 255u << shift;
      __syncthreads();
    }
    // Every key below the threshold, then the first `rank` keys equal to it.
    uint32_t* equal_rank = reinterpret_cast<uint32_t*>(smem);
    uint32_t* victim_rank = equal_rank + capacity;
    host_batch_block_scan<THREADS>(
        capacity, [&](uint32_t i) { return static_cast<uint32_t>(key(i) == prefix); }, equal_rank);
    const auto taken = [&](uint32_t i) {
      const uint32_t k = key(i);
      return k < prefix || (k == prefix && equal_rank[i] < rank);
    };
    host_batch_block_scan<THREADS>(
        capacity, [&](uint32_t i) { return static_cast<uint32_t>(taken(i)); }, victim_rank);
#pragma unroll 1
    for (uint32_t i = tid; i < capacity; i += THREADS) {
      if (!taken(i)) continue;
      const uint32_t staged = first + victim_rank[i];
      const int32_t slot = pool * capacity + i;
      const typename Params::IdType page = params.unique_pages[staged];
      const int32_t evicted = params.slot_page[slot];
      // An evicted page is not referenced this launch: a reference would have hit and stamped it.
      if (evicted >= 0) params.resident[evicted - params.host_page_start] = -1;
      params.resident[page - params.host_page_start] = slot;
      params.slot_page[slot] = page;
      params.slot_stamp[slot] = stamp;
      params.victim_of[staged] = slot;
    }
  }
  // Pages beyond the pool's free slots stay uncached.
  for (uint32_t staged = first + victims + tid; staged < last; staged += THREADS) {
    params.victim_of[staged] = -1;
  }
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(params.pool_flags + pool),
                 "r"(epoch + 1)
                 : "memory");
  }
}

// Claims an item for this CTA; workers and stealing fetch CTAs may both reach it.
template <typename Params>
__device__ __forceinline__ bool claim_host_batch_item(const Params& params, uint32_t* scratch,
                                                      uint32_t item) {
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  if (tid == 0) {
    const uint32_t mark = scratch[kHostBatchScratchEpoch] + 1;
    scratch[kHostBatchScratchClaimed] = atomicExch(host_batch_claims(params) + item, mark) != mark;
  }
  __syncthreads();
  const bool claimed = scratch[kHostBatchScratchClaimed];
  // tid 0 rewrites the slot on the next claim.
  __syncthreads();
  return claimed;
}

// Items: [full rows: one per row tile | tail HBM chunks | tail Host chunks]. Full items cover a
// whole row and wait only in their leading (Host) tiles.
template <typename KTraits, typename Params>
__device__ __forceinline__ void run_host_batch_item(const Params& params,
                                                    typename KTraits::SharedStorage& storage,
                                                    uint8_t* smem, const uint32_t* scratch,
                                                    const uint32_t item, float (*s_frag)[8],
                                                    float (*o_frag)[8], float* m, float* d) {
  using IdType = typename KTraits::IdType;
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  const uint32_t full_items = params.full_rows * params.num_head_tiles;
  const uint32_t tail_tiles = (params.num_rows - params.full_rows) * params.num_head_tiles;
  const uint32_t hbm_end = full_items + tail_tiles * params.num_chunks;
  uint32_t row, head_tile, partial = 0, kv_start, kv_len, host_tiles;
  if (item < full_items) {
    row = item / params.num_head_tiles;
    head_tile = item % params.num_head_tiles;
    kv_start = 0;
    kv_len = params.kv_indptr[row + 1] - params.kv_indptr[row];
    host_tiles = ceil_div(params.state[kHostBatchHostCounts + row], CTA_TILE_KV);
  } else if (item < hbm_end) {
    const uint32_t tile = (item - full_items) / params.num_chunks;
    const uint32_t chunk = (item - full_items) % params.num_chunks;
    row = params.full_rows + tile / params.num_head_tiles;
    head_tile = tile % params.num_head_tiles;
    partial = item - full_items;
    const uint32_t length = params.kv_indptr[row + 1] - params.kv_indptr[row];
    const uint32_t host = host_batch_split_host(params, row);
    const uint32_t chunk_len = round_up(ceil_div(length - host, params.num_chunks), CTA_TILE_KV);
    if (chunk * chunk_len >= length - host) return;
    kv_start = host + chunk * chunk_len;
    kv_len = min(chunk_len, length - kv_start);
    // Without Host items the leading chunks carry the row's Host entries and wait on them.
    const uint32_t row_host = params.state[kHostBatchHostCounts + row];
    host_tiles = kv_start < row_host ? ceil_div(min(row_host, kv_start + kv_len) - kv_start, CTA_TILE_KV)
                                     : 0;
  } else {
    const uint32_t k = item - hbm_end;
    row = host_batch_host_item_rows(params)[k];
    const uint2 chunks = host_batch_host_chunks<CTA_TILE_KV>(
        params.state[kHostBatchHostCounts + row], host_batch_max_host_items(params),
        params.host_item_entries);
    const uint32_t local = k - host_batch_host_items(params)[row];
    head_tile = local / chunks.x;
    partial = tail_tiles * params.num_chunks + k;
    kv_start = local % chunks.x * chunks.y;
    kv_len = min(chunks.y, params.state[kHostBatchHostCounts + row] - kv_start);
    host_tiles = ceil_div(kv_len, CTA_TILE_KV);
  }

  typename KTraits::AttentionVariant variant(params, row, smem);
  init_states_<KTraits>(o_frag, m, d);
  __syncthreads();
  load_q<KTraits>(&storage, params.q_nope + row * params.q_nope_stride_n,
                  params.q_pe + row * params.q_pe_stride_n, params.q_nope_stride_n,
                  params.q_nope_stride_h, params.q_pe_stride_n, params.q_pe_stride_h, 1,
                  head_tile * CTA_TILE_Q, params.num_heads);
  // load_q's copies join the first KV group, as in the native kernel.
  attend_host_batch_tiles<KTraits>(params, storage, variant, scratch,
                                   params.kv_indptr[row] + kv_start, kv_len, host_tiles, s_frag,
                                   o_frag, m, d);
  __syncthreads();
  normalize_d_<KTraits>(&storage, 0, o_frag, m, d);
  finalize_m_<KTraits>(variant, m);

  // Each warp reads back output rows other lanes stored to shared memory.
  if (item < full_items) {
    write_o<KTraits, /*SYNC_WARP=*/true>(&storage, params.final_o + row * params.o_stride_n,
                     params.final_lse ? params.final_lse + row * params.num_heads : nullptr,
                     nullptr, nullptr, o_frag, m, d, params.o_stride_n, params.o_stride_h, 1,
                     head_tile * CTA_TILE_Q, params.num_heads, params.return_lse_base_on_e);
  } else {
    write_o<KTraits, /*SYNC_WARP=*/true>(&storage, nullptr, nullptr,
                     params.partial_o + static_cast<int64_t>(partial) * CTA_TILE_Q *
                                            KTraits::HEAD_DIM_CKV,
                     params.partial_lse + partial * CTA_TILE_Q, o_frag, m, d, params.o_stride_n,
                     params.o_stride_h, 1, head_tile * CTA_TILE_Q, params.num_heads,
                     params.return_lse_base_on_e);
  }
}

// Worker CTAs walk the items in a static round robin from the front; fetch CTAs, once their
// copies are done, walk from the back and stop at the first item a worker already claimed. The
// launcher sizes the full part one item short of an even share, so the two walks meet among the
// small tail items. Both walks derive from
// blockIdx and the loop counter, which keeps the item warp-uniform for the compiler: a
// shared-memory work queue, or a second inlined copy of the item, spilled the tile loop's
// shared-memory offsets (~11% slower on HBM pages).
// With the LRU cache, one item per pool precedes the attention items when any page is staged.
// Each worker reaches its LRU items before any attention item and they never wait, so the fetch
// loop, which waits on the pools' flags, cannot deadlock with them.
template <typename KTraits, typename Params>
__device__ __forceinline__ void run_host_batch_items(const Params& params,
                                                     typename KTraits::SharedStorage& storage,
                                                     uint8_t* smem, uint32_t* scratch,
                                                     bool fetcher, uint32_t unique) {
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  // With no Host page the item list has no Host items, and host_batch_host_items was never
  // written this launch, so it must not be read.
  const int lru_items = host_batch_caching(params) && unique ? params.num_pools : 0;
  const int items = lru_items +
                    (params.full_rows + (params.num_rows - params.full_rows) * params.num_chunks) *
                        params.num_head_tiles +
                    (unique ? host_batch_host_items(params)[params.num_rows] : 0);
  const int workers = gridDim.y - params.fetch_ctas;
  const int first = fetcher ? items - 1 - (blockIdx.y - workers) : blockIdx.y;
  const int step = fetcher ? -static_cast<int>(params.fetch_ctas) : workers;

  float s_frag[KTraits::QK_SHARD ? NUM_MMA_KV / 2 : NUM_MMA_KV][8];
  alignas(16) float o_frag[NUM_MMA_D_CKV / 2][8];
  float m[2];
  float d[2];

#pragma unroll 1
  for (int item = first; item >= 0 && item < items; item += step) {
    if (!claim_host_batch_item(params, scratch, item)) {
      if (fetcher) break;
      continue;
    }
    if (item < lru_items) {
      run_host_batch_lru<KTraits>(params, smem, item, scratch[kHostBatchScratchEpoch], unique);
      // The next item reuses the shared memory the selection scanned into.
      __syncthreads();
      continue;
    }
    run_host_batch_item<KTraits>(params, storage, smem, scratch, item - lru_items, s_frag, o_frag,
                                 m, d);
  }
}

template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLAHostBatchKernel(
    const __grid_constant__ Params params) {
  // cp.async moves full 128-byte lines only when the shared destination is aligned like the
  // global source. The static scratch below would otherwise leave the dynamic buffer 16-byte
  // aligned, which split every request in two: twice the sysmem sectors on the fetch (~22 vs
  // ~26 GB/s) and slower HBM tile loads.
  // A distinct name: extern shared arrays are one symbol per translation unit, and the native MLA
  // kernels compiled alongside declare `smem` with 16-byte alignment, which would win.
  extern __shared__ __align__(128) uint8_t host_batch_smem[];
  uint8_t* smem = host_batch_smem;
  auto& storage = reinterpret_cast<typename KTraits::SharedStorage&>(*smem);
  // Partition counts: 2 per warp for each of the 8 entries a thread handles per pass.
  __shared__ uint32_t scratch[2 * 8 * KTraits::NUM_THREADS / 32];
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  // The previous launch's last CTA advanced the epoch; ready flags from it no longer match.
  const uint32_t epoch = params.state[kHostBatchEpoch];

  // Not staging and no cache: nothing to prepare; remap is kv_indices (see the launcher) and
  // the host counts stay zero from allocation.
  const bool reuse_meta = params.state[kHostBatchReuseMeta] != 0 ||
                          (!params.stage && !host_batch_caching(params));
  uint32_t unique;
  if (!params.stage && !host_batch_caching(params)) {
    unique = 0;
  } else if (reuse_meta) {
    unique = params.state[kHostBatchUnique];
  } else {
    if (params.stage) {
      prepare_host_batch_rows<KTraits>(params, scratch, epoch + 1);
    } else {
      prepare_host_batch_rows_in_place<KTraits>(params, epoch + 1);
    }
    cg::this_grid().sync();
    unique = resolve_host_batch_slots<KTraits>(params, smem);
    if (blockIdx.y == 0 && tid == 0) params.state[kHostBatchUnique] = unique;
  }
  // unique is the same on every CTA (each runs the same scan), so the branch is grid-uniform and
  // the cooperative barriers below stay matched.
  // Which end of the item list this CTA walks from. This is a partition of the grid and must not
  // depend on `unique`: if the back CTAs also walked from the front they would retrace the same
  // items and contend on the claims (0% Host, 512 rows: 1514 -> 1784 us).
  const bool from_back = blockIdx.y >= gridDim.y - params.fetch_ctas;
  if (unique > 0 && !reuse_meta) {
    // Rows resolved by other CTAs must be complete before any item or fetch reads them, and fetch
    // CTAs clear the dedup slots the resolve step reads.
    cg::this_grid().sync();
    // Claim counts were read by every CTA's scan; reset them for the next launch.
    for (uint32_t row = blockIdx.y * KTraits::NUM_THREADS + tid; row < params.num_rows;
         row += gridDim.y * KTraits::NUM_THREADS) {
      host_batch_row_claims(params)[row] = 0;
    }
  }
  if (tid == 0) scratch[kHostBatchScratchEpoch] = epoch;
  __syncthreads();
  // Fetch CTAs take attention items once their copies are issued, so low Host shares and the end
  // of the PCIe stream keep every SM busy; with no Host page there is nothing to fetch.
  if (unique > 0 && from_back) fetch_host_batch_pages<KTraits>(params, smem, epoch, unique);
  run_host_batch_items<KTraits>(params, storage, smem, scratch, from_back, unique);
  // Uniform over the grid, so every CTA takes the same branch.
  if (params.full_rows < params.num_rows) {
    cg::this_grid().sync();
    merge_host_batch_chunks<KTraits>(params);
  }

  // The last CTA to leave resets the shared counters for the next launch.
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    if (atomicAdd(params.state + kHostBatchExit, 1u) == gridDim.y - 1) {
      params.state[kHostBatchExit] = 0;
      __threadfence();
      params.state[kHostBatchEpoch] = epoch + 1;
    }
  }
}

// partial_capacity bounds the item states of the tail rows, in 64-head tiles.
template <uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename Params>
cudaError_t BatchMLAHostBatchAttention(Params params, uint32_t max_kv_len,
                                       uint32_t partial_capacity, cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  if constexpr (HEAD_DIM_CKV != 512 || HEAD_DIM_KPE != 64 || sizeof(DTypeKV) != 2 ||
                sizeof(DTypeQ) != 2) {
    return cudaErrorNotSupported;
  } else {
    int device, smem_limit_per_sm, num_sms;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
    FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(
        &smem_limit_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    FLASHINFER_CUDA_CALL(
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
    DISPATCH_SMEM_CONFIG(smem_limit_per_sm, NUM_STAGES, CTA_TILE_KV, QK_SHARD, {
      using KTraits = KernelTraits<false, NUM_STAGES, QK_SHARD, HEAD_DIM_CKV, HEAD_DIM_KPE,
                                   /*CTA_TILE_Q_=*/64, CTA_TILE_KV, DTypeQ, DTypeKV, DTypeO, IdType>;
      auto kernel = BatchMLAHostBatchKernel<KTraits, Params>;
      const size_t smem_size = sizeof(typename KTraits::SharedStorage);
      FLASHINFER_CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      int blocks_per_sm = 0;
      FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm, kernel, KTraits::NUM_THREADS, smem_size));
      const uint32_t grid = blocks_per_sm * num_sms;
      if (blocks_per_sm < 1 || params.fetch_ctas < 1 || params.fetch_ctas >= grid) {
        return cudaErrorInvalidConfiguration;
      }
      // LRU victim selection scans a pool's slots twice into the dynamic shared memory.
      if (params.pool_capacity > kHostBatchMaxPool ||
          2 * params.pool_capacity * sizeof(uint32_t) > smem_size) {
        return cudaErrorInvalidConfiguration;
      }
      // Full rows give each worker one item less than an even share of all row tiles over the
      // grid. The remaining tail rows are split into HBM and Host items (see
      // host_batch_host_chunks), which both item walks consume (see run_host_batch_items).
      const uint32_t workers = grid - params.fetch_ctas;
      const uint32_t row_tiles = params.num_rows * params.num_head_tiles;
      const uint32_t per_worker = row_tiles / grid > 0 ? row_tiles / grid - 1 : 0;
      params.full_rows = min(params.num_rows, workers * per_worker / params.num_head_tiles);
      uint32_t tail_tiles = (params.num_rows - params.full_rows) * params.num_head_tiles;
      // Each tail row tile needs an HBM state and at least one Host state.
      if (2 * tail_tiles > partial_capacity) {
        params.full_rows = params.num_rows;
        tail_tiles = 0;
      }
      params.partial_capacity = partial_capacity;
      const uint32_t kv_len = max(max_kv_len, 1u);
      params.num_chunks = 1;
      uint64_t best_cost = UINT64_MAX;
      for (uint32_t chunks = 1; chunks <= 32 && tail_tiles > 0; ++chunks) {
        const uint32_t chunk_len = round_up(ceil_div(kv_len, chunks), CTA_TILE_KV);
        if (chunks > 1 && (tail_tiles * (chunks + 1) > partial_capacity ||
                           chunk_len * (chunks - 1) >= kv_len)) {
          break;
        }
        // In KV tiles over the whole row, Host share unknown: rounds of items over the grid,
        // each also loading Q (about two tiles of bytes) and writing a partial state that the
        // merge re-reads (about four).
        const uint64_t item_cost = chunk_len / CTA_TILE_KV + 6;
        const uint64_t cost = ceil_div(tail_tiles * chunks, grid) * item_cost;
        if (cost < best_cost) {
          best_cost = cost;
          params.num_chunks = chunks;
        }
      }
      void* args[] = {(void*)&params};
      FLASHINFER_CUDA_CALL(cudaLaunchCooperativeKernel((void*)kernel, dim3(1, grid),
                                                       dim3(32, 4, 2), args, smem_size, stream));
    });
    return cudaSuccess;
  }
}

}  // namespace mla

}  // namespace flashinfer

#endif  // FLASHINFER_MLA_HOST_BATCH_CUH_
