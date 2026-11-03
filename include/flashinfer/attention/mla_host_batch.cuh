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

#include <queue>
#include <vector>
#include "mla.cuh"
#include "far_link_emu.cuh"

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
  static constexpr bool fully_staged = false;
  static constexpr bool whole_rows = false;
  static constexpr bool shared_fetch_ids = false;

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
  DTypeKV* staging;        // [cache_slots + staging_capacity, 576]: LRU cache slots, then staging
  DTypeO* partial_o;       // [items, CTA_TILE_Q, HEAD_DIM_CKV] when rows are split
  float* partial_lse;      // [items, CTA_TILE_Q]

  // Cross-launch LRU cache of Host pages (all null and pool_capacity 0 when disabled). Each
  // persistent pool (a request slot, as SGLang's req_pool_idx) owns cache slots
  // [id * pool_capacity, (id + 1) * pool_capacity). A launch serves num_pools of them: batch pool
  // p is persistent pool pool_id[p] and covers the contiguous rows [pool_row_start[p],
  // pool_row_start[p + 1]), e.g. one request's MTP tokens, so requests keep their cache when the
  // batch is reordered or resized. Hits are global (resident maps any Host page to its slot);
  // only placement and eviction are per pool. Persistent state is per layer; stamps hold the
  // epoch + 1 of a slot's last use, 0 when empty.
  IdType* row_pool;        // [num_rows] batch pool of each row
  IdType* pool_row_start;  // [num_pools + 1]
  IdType* pool_id;         // [num_pools] persistent pool of each batch pool
  int32_t* resident;       // [num_pages - host_page_start] cache slot of a Host page, or -1
  int32_t* slot_page;      // [cache_slots] Host page in a cache slot, or -1
  uint32_t* slot_stamp;    // [cache_slots] 0 = empty
  int32_t* victim_of;      // [staging_capacity] cache slot a staged page is copied to, or -1
  int32_t* slot_pool;      // [staging_capacity] pool of the row that claimed a staged page
  int32_t* claim_entry;    // [nnz] in place: entry of each row's k-th claim, at kv_indptr[row] + k
  int32_t* entry_victim;   // [nnz] in place: LRU victim of a claiming entry, or -1
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
  uint32_t cache_slots;    // persistent pools x pool_capacity
  uint32_t host_item_entries;  // smallest Host item of a tail row (see host_batch_host_chunks)
  bool host_split;             // tail rows put Host entries in separate items
  // false: attention reads uncached Host pages in place, in their original order, with no
  // staging or wait (fast far tiers such as NVLink); with the cache on, fetch CTAs only copy the
  // misses into their victims for later launches. Without the cache the launch skips prepare.
  bool stage;
  uint32_t prepare_parts;    // prepare units per row (see prepare_host_batch_rows)
  uint32_t prepare_segment;  // entries per prepare unit

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
// A launch whose staging can hold every selected entry never reads far KV in attention.
// Specialize that invariant while retaining the generic overflow/in-place path.
template <typename Params>
struct FullyStagedHostBatchParams : Params {
  static constexpr bool stage = true;
  static constexpr bool fully_staged = true;
};

template <typename Params>
struct WholeRowHostBatchParams : FullyStagedHostBatchParams<Params> {
  static constexpr bool whole_rows = true;
  static constexpr bool host_split = false;
  static constexpr uint32_t num_chunks = 1;
};

template <typename Params>
struct CachedSharedIdsHostBatchParams : WholeRowHostBatchParams<Params> {
  static constexpr bool shared_fetch_ids = true;
};

template <typename Q, typename KV, typename O, typename Id>
struct JointQKTraits : KernelTraits<false, 2, true, 512, 64, 64, 32, Q, KV, O, Id> {};

template <typename Params>
struct JointQKHostParams : CachedSharedIdsHostBatchParams<Params> {};

constexpr uint32_t kHostBatchFetchRows = 64;
// Smallest chunk; ready flags are allocated per this many staging slots.
constexpr uint32_t kHostBatchMinFetchRows = 16;
// Staged chunks a fetch CTA may set aside until the LRU victims are chosen (see the fetch loop).
constexpr uint32_t kHostBatchMaxDeferred = 64;
// state layout: counters; per row its Host entry count (the front placement cursor), its claimed
// page count, its HBM (back) placement cursor and its first Host item; the row of each Host item;
// one ready flag per kHostBatchMinFetchRows staging slots (one per fetch chunk is used); one claim
// per LRU and attention item; then one completion count per row for the small staged merge.
// Flags and claims hold epoch + 1, so they never need clearing.
// Nonzero once prepare dropped an invalid (negative) entry, so rows need compacting; cleared
// by the launch's last CTA.
constexpr uint32_t kHostBatchGaps = 0;
// Unique Host pages of the last launch, for diagnostics.
constexpr uint32_t kHostBatchUnique = 1;
constexpr uint32_t kHostBatchExit = 2;
constexpr uint32_t kHostBatchEpoch = 3;
constexpr uint32_t kHostBatchHostCounts = 4;
// Shared scratch slots used after the partition phase.
constexpr uint32_t kHostBatchScratchEpoch = 0;
constexpr uint32_t kHostBatchScratchClaimed = 1;
constexpr uint32_t kHostBatchScratchFetchRows = 2;
// host_slot packs the claiming row and the page's rank among that row's claims (see
// prepare_host_batch_rows); kv lengths and row counts are bounded accordingly.
constexpr uint32_t kHostBatchRankBits = 16;
// In place with the cache, the claiming row's entry of a missed Host page carries this bit; the
// LRU step stores its victim in entry_victim and the attention CTA that loads it copies it there
// (see writeback_host_batch_tile). Page ids stay below it (checked by the launcher).
constexpr int32_t kHostBatchOwnerBit = 1 << 30;
// Pads the passes of prepare_host_batch_rows; cached entries encode slot 0 as -1.
constexpr int32_t kHostBatchNoEntry = INT32_MIN;
// Largest LRU pool: victim selection keeps two scans of the pool's slots in shared memory.
constexpr uint32_t kHostBatchMaxPool = 8192;

template <typename Params>
__device__ __forceinline__ bool host_batch_caching(const Params& params) {
  if constexpr (Params::shared_fetch_ids) return true;
  else return params.pool_capacity != 0;
}

// In place with the cache: attention CTAs write missed pages into the cache as they load them.
template <typename Params>
__device__ __forceinline__ bool host_batch_writes_back(const Params& params) {
  return host_batch_caching(params) && !params.stage;
}

// One buffer holds the cache slots, then the staging slots: cache slot c is row c and staging
// slot s is row cache_slots + s, and an entry -(row + 1) reads that row, so the tile loads need no
// second base. Only rows at or past cache_slots wait for the fetch. Cache slots first keep their
// rows fixed when the staging part is resized for a different batch.
template <typename Params>
__device__ __forceinline__ typename Params::IdType host_batch_cached_entry(const Params& params,
                                                                           int32_t slot) {
  return -static_cast<typename Params::IdType>(slot) - 1;
}

template <typename Params>
__device__ __forceinline__ typename Params::IdType host_batch_staged_entry(const Params& params,
                                                                           uint32_t slot) {
  return -static_cast<typename Params::IdType>(params.cache_slots + slot) - 1;
}

template <typename Params>
__device__ __forceinline__ uint32_t* host_batch_row_claims(const Params& params) {
  return params.state + kHostBatchHostCounts + params.num_rows;
}

template <typename Params>
__device__ __forceinline__ uint32_t* host_batch_local_cursors(const Params& params) {
  return host_batch_row_claims(params) + params.num_rows;
}

// Valid entries of a row after prepare: negative entries of kv_indices (top-k padding, as SGLang
// emits -1) are dropped and the rest packed to the front, so this is the row's attention length.
// Without prepare (no staging, no cache) entries are used as given.
template <typename Params>
__device__ __forceinline__ uint32_t host_batch_row_len(const Params& params, uint32_t row) {
  if (!params.stage && !host_batch_caching(params)) {
    return params.kv_indptr[row + 1] - params.kv_indptr[row];
  }
  return params.state[kHostBatchHostCounts + row] + host_batch_local_cursors(params)[row];
}

template <typename Params>
__device__ __forceinline__ uint32_t* host_batch_host_items(const Params& params) {
  return host_batch_local_cursors(params) + params.num_rows;
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
  return host_batch_flags(params) + ceil_div(params.staging_capacity, kHostBatchMinFetchRows);
}

__device__ __forceinline__ uint32_t host_batch_ld_acquire(const uint32_t* address) {
  uint32_t value;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

template <typename Params>
__device__ __forceinline__ uint32_t* host_batch_row_completed(const Params& params) {
  // Dedicated counters: claim counts are still being reset as workers leave the resolve barrier.
  return host_batch_claims(params) + params.num_rows * params.num_head_tiles +
         params.partial_capacity + params.num_pools;
}

template <bool ROW_MERGE, typename Params>
__device__ __forceinline__ void wait_host_batch_row(const Params& params, uint32_t row,
                                                    uint32_t states) {
  if constexpr (ROW_MERGE) {
    const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    if (tid == 0) {
      while (host_batch_ld_acquire(host_batch_row_completed(params) + row) <
             states * params.num_head_tiles)
        __nanosleep(64);
    }
    __syncthreads();
  }
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
// Host entries fill each row from the front and HBM entries from the back through two per-row
// cursors, so no pass has to count the Host entries first, and several CTAs can share a row: the
// work is (row, segment) units sized by the launcher to cover the grid, so a one-row launch no
// longer prepares its 2048 entries on a single CTA. The order within each part is arbitrary.
// The cursors start at zero; the launch's last CTA clears them.
// With the LRU cache, a Host entry whose page is resident becomes a cached entry, placed with the
// HBM entries (it needs no wait) and stamped with this launch's `stamp`; only misses are claimed.
// Nothing changes residency before the grid barrier, so every unit sees the same hits.
template <typename KTraits, typename Params>
__device__ __forceinline__ void prepare_host_batch_rows(const Params& params, uint32_t stamp) {
  using IdType = typename Params::IdType;
  constexpr uint32_t THREADS = KTraits::NUM_THREADS;
  constexpr uint32_t PER_THREAD = 8;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t lane = tid % 32;
  const uint32_t lower = (1u << lane) - 1;
  const uint32_t units = params.num_rows * params.prepare_parts;
  uint32_t* host_cursors = params.state + kHostBatchHostCounts;
  uint32_t* local_cursors = host_batch_local_cursors(params);
#pragma unroll 1
  for (uint32_t unit = blockIdx.y; unit < units; unit += gridDim.y) {
    const uint32_t row = unit / params.prepare_parts;
    const IdType begin = params.kv_indptr[row];
    const uint32_t length = params.kv_indptr[row + 1] - begin;
    const uint32_t part_begin = unit % params.prepare_parts * params.prepare_segment;
    const uint32_t part_end = min(part_begin + params.prepare_segment, length);
#pragma unroll 1
    for (uint32_t offset = part_begin; offset < part_end; offset += PER_THREAD * THREADS) {
      IdType pages[PER_THREAD];
      uint32_t claimed[PER_THREAD];
#pragma unroll
      for (uint32_t i = 0; i < PER_THREAD; ++i) {
        const uint32_t j = offset + i * THREADS + tid;
        pages[i] = j < part_end ? params.kv_indices[begin + j] : kHostBatchNoEntry;
      }
      // Negative entries (padding) are dropped; their positions leave a gap between the Host
      // part and the HBM part that resolve_host_batch_slots closes.
      bool dropped = false;
#pragma unroll
      for (uint32_t i = 0; i < PER_THREAD; ++i) {
        if (pages[i] < 0 && pages[i] != kHostBatchNoEntry) {
          pages[i] = kHostBatchNoEntry;
          dropped = true;
        }
      }
      if (__any_sync(0xffffffff, dropped) && lane == 0) params.state[kHostBatchGaps] = 1;
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
      uint32_t fresh_masks[PER_THREAD], host_masks[PER_THREAD], local_masks[PER_THREAD];
      uint32_t fresh_total = 0, host_total = 0, local_total = 0;
#pragma unroll
      for (uint32_t i = 0; i < PER_THREAD; ++i) {
        fresh_masks[i] =
            __ballot_sync(0xffffffff, pages[i] >= params.host_page_start && claimed[i] == 0);
        host_masks[i] = __ballot_sync(0xffffffff, pages[i] >= params.host_page_start);
        // Cached entries are negative; kHostBatchNoEntry pads short passes.
        local_masks[i] = __ballot_sync(
            0xffffffff, pages[i] != kHostBatchNoEntry && pages[i] < params.host_page_start);
        fresh_total += __popc(fresh_masks[i]);
        host_total += __popc(host_masks[i]);
        local_total += __popc(local_masks[i]);
      }
      uint32_t first = 0, host_first = 0, local_first = 0;
      if (lane == 0) {
        if (fresh_total) first = atomicAdd(host_batch_row_claims(params) + row, fresh_total);
        if (host_total) host_first = atomicAdd(host_cursors + row, host_total);
        if (local_total) local_first = atomicAdd(local_cursors + row, local_total);
      }
      first = __shfl_sync(0xffffffff, first, 0);
      host_first = __shfl_sync(0xffffffff, host_first, 0);
      local_first = __shfl_sync(0xffffffff, local_first, 0);
#pragma unroll
      for (uint32_t i = 0; i < PER_THREAD; ++i) {
        if (fresh_masks[i] >> lane & 1) {
          const uint32_t rank = first + __popc(fresh_masks[i] & lower);
          atomicExch(params.host_slot + (pages[i] - params.host_page_start),
                     (row << kHostBatchRankBits | rank) + 2);
        }
        if (pages[i] >= params.host_page_start) {
          params.remap[begin + host_first + __popc(host_masks[i] & lower)] = pages[i];
        } else if (pages[i] != kHostBatchNoEntry) {
          params.remap[begin + length - 1 - local_first - __popc(local_masks[i] & lower)] = pages[i];
        }
        first += __popc(fresh_masks[i]);
        host_first += __popc(host_masks[i]);
        local_first += __popc(local_masks[i]);
      }
    }
  }
}

// prepare_host_batch_rows without staging (params.stage false, cache on): valid entries are packed
// to the row's front, cached ones are rewritten and stamped, and uncached Host pages stay page ids
// for the attention to read in place. The entry that claims a page (first CAS) is marked with
// kHostBatchOwnerBit, and the row's k-th claim records its entry in claim_entry; the LRU step
// gives it a victim and the attention that loads it copies the page there
// (writeback_host_batch_tile), so no resolve pass over the entries is needed (it took ~100 us at
// 1024 rows of 2048). Eight entries per thread
// per pass overlap the dependent resident/CAS round trips (one per pass took ~225 us there).
template <typename KTraits, typename Params>
__device__ __forceinline__ void prepare_host_batch_rows_in_place(const Params& params,
                                                                 uint32_t stamp) {
  using IdType = typename Params::IdType;
  constexpr uint32_t THREADS = KTraits::NUM_THREADS;
  constexpr uint32_t PER_THREAD = 8;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t lane = tid % 32;
  const uint32_t lower = (1u << lane) - 1;
  const uint32_t units = params.num_rows * params.prepare_parts;
#pragma unroll 1
  for (uint32_t unit = blockIdx.y; unit < units; unit += gridDim.y) {
    const uint32_t row = unit / params.prepare_parts;
    const IdType begin = params.kv_indptr[row];
    const uint32_t length = params.kv_indptr[row + 1] - begin;
    const uint32_t part_begin = unit % params.prepare_parts * params.prepare_segment;
    const uint32_t part_end = min(part_begin + params.prepare_segment, length);
#pragma unroll 1
    for (uint32_t offset = part_begin; offset < part_end; offset += PER_THREAD * THREADS) {
      IdType entries[PER_THREAD];
      int32_t slots[PER_THREAD];
      uint32_t valid_masks[PER_THREAD];
#pragma unroll
      for (uint32_t i = 0; i < PER_THREAD; ++i) {
        const uint32_t j = offset + i * THREADS + tid;
        entries[i] = j < part_end ? params.kv_indices[begin + j] : -1;
        // Padding (negative input) is dropped. Decided before rewriting: cache slot 0 is -1.
        valid_masks[i] = __ballot_sync(0xffffffff, entries[i] >= 0);
      }
#pragma unroll
      for (uint32_t i = 0; i < PER_THREAD; ++i) {
        slots[i] = entries[i] >= params.host_page_start
                       ? params.resident[entries[i] - params.host_page_start]
                       : -1;
      }
      bool fresh[PER_THREAD];
#pragma unroll
      for (uint32_t i = 0; i < PER_THREAD; ++i) {
        fresh[i] = false;
        if (entries[i] >= params.host_page_start) {
          if (slots[i] >= 0) {
            params.slot_stamp[slots[i]] = stamp;
            entries[i] = host_batch_cached_entry(params, slots[i]);
          } else {
            fresh[i] =
                atomicCAS(params.host_slot + (entries[i] - params.host_page_start), 0u, 1u) == 0;
          }
        }
      }
      uint32_t fresh_masks[PER_THREAD], fresh_total = 0, valid_total = 0;
#pragma unroll
      for (uint32_t i = 0; i < PER_THREAD; ++i) {
        fresh_masks[i] = __ballot_sync(0xffffffff, fresh[i]);
        fresh_total += __popc(fresh_masks[i]);
        valid_total += __popc(valid_masks[i]);
      }
      uint32_t first = 0, position = 0;
      if (lane == 0) {
        if (fresh_total) first = atomicAdd(host_batch_row_claims(params) + row, fresh_total);
        if (valid_total) position = atomicAdd(host_batch_local_cursors(params) + row, valid_total);
      }
      first = __shfl_sync(0xffffffff, first, 0);
      position = __shfl_sync(0xffffffff, position, 0);
#pragma unroll
      for (uint32_t i = 0; i < PER_THREAD; ++i) {
        if (valid_masks[i] >> lane & 1) {
          const uint32_t at = begin + position + __popc(valid_masks[i] & lower);
          if (fresh[i]) {
            params.remap[at] = entries[i] | kHostBatchOwnerBit;
            params.claim_entry[begin + first + __popc(fresh_masks[i] & lower)] = at;
          } else {
            params.remap[at] = entries[i];
          }
        }
        first += __popc(fresh_masks[i]);
        position += __popc(valid_masks[i]);
      }
    }
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
  // Same (row, segment) units as the prepare step, so a few rows still spread over the grid. In
  // place with the cache, prepare already marked the claiming entries (no page ids to resolve).
  const uint32_t units = host_batch_writes_back(params) ? 0 : params.num_rows * params.prepare_parts;
#pragma unroll 1
  for (uint32_t unit = blockIdx.y; unit < units; unit += gridDim.y) {
    const uint32_t row = unit / params.prepare_parts;
    const IdType begin = params.kv_indptr[row];
    // Host entries fill each row's leading positions, or anywhere when not staging.
    const uint32_t host = params.stage ? params.state[kHostBatchHostCounts + row]
                                       : host_batch_row_len(params, row);
    const uint32_t part_begin = unit % params.prepare_parts * params.prepare_segment;
    const uint32_t part_end = min(part_begin + params.prepare_segment, host);
#pragma unroll 1
    for (uint32_t j = part_begin + tid; j < part_end; j += THREADS) {
      const IdType page = params.remap[begin + j];
      if (!params.stage && page < params.host_page_start) continue;
      const uint32_t claim = params.host_slot[page - params.host_page_start] - 2;
      const uint32_t owner = claim >> kHostBatchRankBits;
      const uint32_t slot = row_base[owner] + (claim & ((1u << kHostBatchRankBits) - 1));
      if (owner == row) {
        params.unique_pages[slot] = page;
        // (In place with the cache prepare already marked the claims; this loop does not run.)
        if (host_batch_caching(params) && slot < params.staging_capacity) {
          params.slot_pool[slot] = params.row_pool[row];
        }
      }
      if (params.stage && slot < params.staging_capacity) {
        params.remap[begin + j] = host_batch_staged_entry(params, slot);
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

// After the grid barrier, when prepare dropped padding: a staged row holds its Host entries at
// [0, host), its HBM entries at [length - local, length) and a gap of dropped positions between.
// Move the HBM entries that lie at or past host + local into the gap, so the row's valid entries
// are [0, host + local) (see host_batch_row_len). Holes and movers are disjoint ranges of equal
// size, so the copy needs no ordering. Rows without padding have no gap and copy nothing.
template <typename KTraits, typename Params>
__device__ __forceinline__ void compact_host_batch_rows(const Params& params) {
  constexpr uint32_t THREADS = KTraits::NUM_THREADS;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
#pragma unroll 1
  for (uint32_t row = blockIdx.y; row < params.num_rows; row += gridDim.y) {
    const auto begin = params.kv_indptr[row];
    const uint32_t length = params.kv_indptr[row + 1] - begin;
    const uint32_t host = params.state[kHostBatchHostCounts + row];
    const uint32_t local = host_batch_local_cursors(params)[row];
    const uint32_t valid = host + local;
    if (valid == length) continue;
    const uint32_t local_begin = length - local;
    const uint32_t holes = min(valid, local_begin) - host;
    const uint32_t movers = max(valid, local_begin);
    for (uint32_t k = tid; k < holes; k += THREADS) {
      params.remap[begin + host + k] = params.remap[begin + movers + k];
    }
  }
}

// Staging slots per fetch chunk, the same on every CTA (unique is grid-uniform). With at most
// eight fetch CTAs, keep 16-row chunks below three rounds of 64-row chunks: this lets the first
// pages arrive early on PCIe. More fetch CTAs supply enough concurrent reads to use 64-row chunks
// after one round. Small batches with more than eight fetch CTAs switch as soon as 16-row chunks
// would take a second round: avoiding that extra dependent wait matters on high-latency links.
// Larger batches keep the 64-row-per-CTA threshold to sustain transfer/attention overlap.
template <bool ROW_MERGE, typename Params>
__device__ __forceinline__ uint32_t host_batch_fetch_rows(const Params& params, uint32_t unique) {
  const uint32_t count = params.stage ? min(unique, params.staging_capacity) : unique;
  const uint32_t rounds = params.fetch_ctas <= 8 ? 3 : 1;
  if constexpr (ROW_MERGE) {
    const uint32_t threshold = params.fetch_ctas <= 8
                                   ? rounds * params.fetch_ctas * kHostBatchFetchRows
                                   : params.fetch_ctas * kHostBatchMinFetchRows + 1;
    return count < threshold ? kHostBatchMinFetchRows : kHostBatchFetchRows;
  } else {
    return count < rounds * params.fetch_ctas * kHostBatchFetchRows ? kHostBatchMinFetchRows
                                                                 : kHostBatchFetchRows;
  }
}

// Prepare has finished every resident lookup before the fetch starts. The staging path may
// therefore bind a selected victim here, after releasing the staged bytes to attention, instead
// of putting these dependent map loads/stores on the LRU item's critical path.
template <typename Params>
__device__ __forceinline__ void bind_host_batch_staged_slot(const Params& params, uint32_t staged,
                                                           int32_t victim) {
  if (victim < 0) return;
  const auto page = params.unique_pages[staged];
  const int32_t evicted = params.slot_page[victim];
  if (evicted >= 0) params.resident[evicted - params.host_page_start] = -1;
  params.resident[page - params.host_page_start] = victim;
  params.slot_page[victim] = page;
}

// Copy staged pages Host -> HBM in slot order, `rows` slots per chunk (host_batch_fetch_rows),
// publishing a ready flag per chunk. Slots follow row order (see prepare_host_batch_rows), as do attention
// items. cp.async into shared memory keeps enough PCIe reads in flight: register loads
// (ld.global.cg) reached only ~15 GB/s on sparse 1152-byte rows, cp.async ~26 GB/s. Two buffers
// overlap the next chunk's reads with the copy-out of the current one when shared memory allows.
template <typename KTraits, typename Params>
__device__ __forceinline__ void fetch_host_batch_pages(const Params& params, uint8_t* smem,
                                                       uint32_t epoch, uint32_t unique,
                                                       uint32_t rows) {
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

  const uint32_t chunk_vecs = rows * VECS_PER_ROW;
  __shared__ int32_t cache_rows[kHostBatchFetchRows];
  __shared__ typename Params::IdType page_ids[BUFFERS][kHostBatchFetchRows];
  __shared__ uint32_t deferred[kHostBatchMaxDeferred], deferred_count;
  if (tid == 0) deferred_count = 0;
  FAR_EMU_INIT(2);
  FAR_EMU_SYNC();
  auto issue = [&](uint32_t chunk, uint32_t buf) {
    if constexpr (Params::shared_fetch_ids) {
      if (tid < rows && chunk * rows + tid < count) {
        page_ids[buf][tid] = params.unique_pages[chunk * rows + tid];
      }
      // This synchronizes the ordinary ID stores without draining the other buffer's
      // outstanding KV cp.async group. Every thread then reuses one shared ID per row.
      __syncthreads();
    }
#pragma unroll
    for (uint32_t i = 0; i < CHUNK_VECS / KTraits::NUM_THREADS; ++i) {
      const uint32_t v = i * KTraits::NUM_THREADS + tid;
      const uint32_t slot = chunk * rows + v / VECS_PER_ROW;
      if (v < chunk_vecs && slot < count) {
        const auto page = [&]() {
          if constexpr (Params::shared_fetch_ids)
            return page_ids[buf][v / VECS_PER_ROW];
          else return params.unique_pages[slot];
        }();
        const auto* source = reinterpret_cast<const uint4*>(
            params.ckv + static_cast<int64_t>(page) * ROW);
        cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffer + buf * CHUNK_VECS + v,
                                                       source + v % VECS_PER_ROW);
      }
    }
    FAR_EMU_TILE(2, buf, tid == 0 ? min(rows, count - chunk * rows) : 0, ROW * sizeof(DTypeKV));
    cp_async::commit_group();
  };

  uint32_t chunk = fetch_id;
  if (chunk * rows < count) issue(chunk, 0);
#pragma unroll 1
  for (uint32_t k = 0; chunk * rows < count; ++k, chunk += params.fetch_ctas) {
    const uint32_t next = chunk + params.fetch_ctas;
    if (BUFFERS == 2 && next * rows < count) {
      issue(next, (k + 1) % 2);
      cp_async::wait_group<1>();
    } else {
      cp_async::wait_group<0>();
    }
    FAR_EMU_WAIT(2, BUFFERS == 2 ? k % 2 : 0);
    __syncthreads();
    const uint32_t buf = BUFFERS == 2 ? k % 2 : 0;
    auto* destination = reinterpret_cast<uint4*>(
        params.staging +
        (params.cache_slots + static_cast<int64_t>(chunk) * rows) * ROW);
#pragma unroll
    for (uint32_t i = 0; i < CHUNK_VECS / KTraits::NUM_THREADS; ++i) {
      const uint32_t v = i * KTraits::NUM_THREADS + tid;
      if (params.stage && v < chunk_vecs && chunk * rows + v / VECS_PER_ROW < count) {
        destination[v] = buffer[buf * CHUNK_VECS + v];
      }
    }
    // The resolve step is done, so the dedup entries can be cleared for the next launch.
    if (tid < rows && chunk * rows + tid < count) {
      const auto page = [&]() {
        if constexpr (Params::shared_fetch_ids)
          return page_ids[buf][tid];
        else return params.unique_pages[chunk * rows + tid];
      }();
      params.host_slot[page - params.host_page_start] = 0;
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
    // this chunk were released: one more HBM write per page, and no reread. The victims come from
    // the pools' LRU items, which workers run first (see run_host_batch_items). A staged chunk
    // that lands before its victims are chosen is not waited for: blocking here put the LRU
    // selection (~20 us at 4096 slots) on the PCIe stream's critical path at small batch. Its
    // index is kept and it is copied from staging after the loop. In place there is no staged
    // copy, so the loop waits (nothing in this launch waits on these copies).
    if (host_batch_caching(params)) {
      bool ready = true;
      if (tid < rows && chunk * rows + tid < count) {
        const uint32_t* pool_flag = params.pool_flags + params.slot_pool[chunk * rows + tid];
        ready = host_batch_ld_acquire(pool_flag) == epoch + 1;
      }
      const bool defer = params.stage && !__syncthreads_and(ready) &&
                         deferred_count < kHostBatchMaxDeferred;
      if (defer) {
        __syncthreads();
        if (tid == 0) deferred[deferred_count++] = chunk;
      } else {
        if (tid < rows) {
          const uint32_t slot = chunk * rows + tid;
          int32_t victim = -1;
          if (slot < count) {
            const uint32_t* pool_flag = params.pool_flags + params.slot_pool[slot];
            while (host_batch_ld_acquire(pool_flag) != epoch + 1) __nanosleep(64);
            victim = params.victim_of[slot];
          }
          cache_rows[tid] = victim;
          if constexpr (Params::shared_fetch_ids) {
            if (victim >= 0) {
              const auto page = page_ids[buf][tid];
              const int32_t evicted = params.slot_page[victim];
              if (evicted >= 0) params.resident[evicted - params.host_page_start] = -1;
              params.resident[page - params.host_page_start] = victim;
              params.slot_page[victim] = page;
            }
          } else {
            bind_host_batch_staged_slot(params, slot, victim);
          }
        }
        __syncthreads();
#pragma unroll
        for (uint32_t i = 0; i < CHUNK_VECS / KTraits::NUM_THREADS; ++i) {
          const uint32_t v = i * KTraits::NUM_THREADS + tid;
          const int32_t victim = v < chunk_vecs ? cache_rows[v / VECS_PER_ROW] : -1;
          if (victim >= 0) {
            reinterpret_cast<uint4*>(
                params.staging +
                static_cast<int64_t>(victim) * ROW)[v % VECS_PER_ROW] =
                buffer[buf * CHUNK_VECS + v];
          }
        }
      }
      // The buffer is refilled next.
      __syncthreads();
    }
    if (BUFFERS == 1 && next * rows < count) issue(next, 0);
  }
  // Deferred staged chunks: their victims are chosen by now; copy staging -> cache, HBM to HBM,
  // the whole CTA over each chunk.
  __syncthreads();
#pragma unroll 1
  for (uint32_t d = 0; d < deferred_count; ++d) {
    const uint32_t chunk = deferred[d];
    if (tid < rows) {
      const uint32_t slot = chunk * rows + tid;
      int32_t victim = -1;
      if (slot < count) {
        const uint32_t* pool_flag = params.pool_flags + params.slot_pool[slot];
        while (host_batch_ld_acquire(pool_flag) != epoch + 1) __nanosleep(64);
        victim = params.victim_of[slot];
      }
      cache_rows[tid] = victim;
      bind_host_batch_staged_slot(params, slot, victim);
    }
    __syncthreads();
    const auto* source = reinterpret_cast<const uint4*>(
        params.staging + (params.cache_slots + static_cast<int64_t>(chunk) * rows) * ROW);
#pragma unroll 4
    for (uint32_t v = tid; v < chunk_vecs; v += KTraits::NUM_THREADS) {
      const int32_t victim = cache_rows[v / VECS_PER_ROW];
      if (victim >= 0) {
        reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * ROW)[v % VECS_PER_ROW] =
            source[v];
      }
    }
    __syncthreads();
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
  const int64_t row = staged ? -static_cast<int64_t>(page) - 1 : (page & ~kHostBatchOwnerBit);
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
    // Cached entries (rows below cache_slots) are resident already.
    const uint32_t row = static_cast<uint32_t>(-(page + 1));
    if (page < 0 && row >= params.cache_slots) {
      const uint32_t* flag = host_batch_flags(params) +
                             (row - params.cache_slots) / scratch[kHostBatchScratchFetchRows];
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

// LRU victims of the rows of each pipeline stage (see load_host_batch_kv).
template <typename KTraits>
__device__ __forceinline__ int32_t* host_batch_tile_victims() {
  __shared__ int32_t victims[KTraits::NUM_STAGES * KTraits::CTA_TILE_KV];
  return victims;
}

// load_kv for page size one with staged entries resolved per row.
template <typename KTraits, typename Params>
__device__ __forceinline__ void load_host_batch_kv(typename KTraits::SharedStorage* smem_storage,
                                                   const Params& params,
                                                   const typename KTraits::IdType* pages,
                                                   const uint32_t kv_len, const uint32_t kv_base,
                                                   const uint32_t stage_idx,
                                                   const uint32_t index_base) {
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
    // In place with the cache, a claiming entry's victim travels with its tile, so the writeback
    // reads no global memory after the tile lands.
    if (host_batch_writes_back(params) && lane_idx % 8 == 0) {
      int32_t* victim = host_batch_tile_victims<KTraits>() + stage_idx * KTraits::CTA_TILE_KV + smem_row;
      if (valid && pages[j] >= 0 && (pages[j] & kHostBatchOwnerBit)) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(
                         static_cast<uint32_t>(__cvta_generic_to_shared(victim))),
                     "l"(params.entry_victim + index_base + kv_base + smem_row));
      } else {
        *victim = -1;
      }
    }
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
    if constexpr (!Params::fully_staged) FAR_EMU_TILE(KTraits::NUM_STAGES, stage_idx,
                 lane_idx % 8 == 0 && valid && pages[j] >= 0 &&
                     (pages[j] & ~kHostBatchOwnerBit) >= params.host_page_start,
                 (KTraits::HEAD_DIM_CKV + KTraits::HEAD_DIM_KPE) * sizeof(DTypeKV));
  }
}

// Merge the tail rows' item states after the grid barrier; every CTA takes (row tile, head) units
// so the partial reads spread over all SMs. Loads are batched because each state is an independent
// L2 read. Partial states: HBM item c of tail row tile t at t * num_chunks + c, Host items after
// all HBM states in item order.
// A single query has too few merge heads to fill the grid at several heads per CTA. Give
// each head one CTA, reduce disjoint state subsets in parallel, then combine those float states
// in shared memory. Larger batches use the existing wider merge to avoid extra CTA rounds.
template <typename KTraits, typename Params, bool ROW_MERGE>
__device__ __forceinline__ void merge_host_batch_single_query(const Params& params,
                                                                uint8_t* smem) {
  using DTypeO = typename KTraits::DTypeO;
  constexpr uint32_t VEC = 8;
  constexpr uint32_t BATCH = 4;
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
    wait_host_batch_row<ROW_MERGE>(params, row, states);
    state_t<VEC> local;
#pragma unroll 1
    for (uint32_t base = part; base < states; base += PARTS * BATCH) {
      uint4 raw[BATCH];
      float lse[BATCH];
#pragma unroll
      for (uint32_t i = 0; i < BATCH; ++i) {
        const uint32_t k = min(base + i * PARTS, states - 1);
        const uint32_t partial = (k < hbm_items ? tile * params.num_chunks + k
                                               : host_first + k - hbm_items) * KTraits::CTA_TILE_Q + h;
        raw[i] = __ldcg(reinterpret_cast<const uint4*>(
            params.partial_o + static_cast<int64_t>(partial) * KTraits::HEAD_DIM_CKV + feature));
        lse[i] = __ldcg(params.partial_lse + partial);
      }
#pragma unroll
      for (uint32_t i = 0; i < BATCH; ++i) {
        if (base + i * PARTS < states) {
          vec_t<float, VEC> value;
#pragma unroll
          for (uint32_t j = 0; j < VEC; ++j) value[j] = float(reinterpret_cast<const DTypeO*>(&raw[i])[j]);
          local.merge(value, lse[i], 1.f);
        }
      }
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

template <typename KTraits, uint32_t VEC_SIZE, typename Params, bool ROW_MERGE>
__device__ __forceinline__ void merge_host_batch_chunks(const Params& params) {
  using DTypeO = typename KTraits::DTypeO;
  // Features per thread (see the caller): every pass pays a chain of dependent metadata loads, so
  // fewer, wider passes are faster when there are many units, and narrower ones spread a small
  // merge over more SMs.
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
    const uint32_t length = host_batch_row_len(params, row);
    const uint32_t host = host_batch_split_host(params, row);
    const uint32_t hbm_chunk = round_up(ceil_div(length - host, params.num_chunks), KTraits::CTA_TILE_KV);
    const uint32_t hbm_items = hbm_chunk ? ceil_div(length - host, hbm_chunk) : 0;
    const uint32_t host_items =
        host_batch_host_chunks<KTraits::CTA_TILE_KV>(host, max_items, params.host_item_entries).x;
    const uint32_t host_first = tail_tiles * params.num_chunks +
                                host_batch_host_items(params)[row] + head_tile * host_items;
    const uint32_t states = hbm_items + host_items;
    wait_host_batch_row<ROW_MERGE>(params, row, states);
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

// In place with the cache: once a KV tile has landed in shared memory, copy the rows of missed
// Host pages that this row claimed into their LRU victims (the LRU step chose, stamped and bound
// them before the second grid barrier; their slots came with the tile). Metadata loads here
// stalled the tile loop: ~370 us over 85k misses at 256 rows, even with the far tier in HBM. Each page is read over the link once, by
// the attention that needs it, and written back from on-chip memory, so no SM is set aside to
// fetch: a fetch CTA moves at most ~21 GB/s (about 40 KB of far reads in flight per SM), so
// filling the cache from fetch CTAs took 12+ of 108 SMs to keep up with NVLink and still read
// each miss twice. Mirrors load_host_batch_kv's thread layout, so every thread reads back the
// shared-memory chunks it loaded.
template <typename KTraits, typename Params>
__device__ __forceinline__ void writeback_host_batch_tile(const Params& params,
                                                          typename KTraits::SharedStorage& storage,
                                                          uint32_t index_base, uint32_t kv_len,
                                                          uint32_t kv_base, uint32_t stage_idx) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t ROW = KTraits::HEAD_DIM_CKV + KTraits::HEAD_DIM_KPE;
  constexpr uint32_t UPCAST_STRIDE_CKV = KTraits::UPCAST_STRIDE_CKV;
  constexpr uint32_t UPCAST_STRIDE_KPE = KTraits::UPCAST_STRIDE_KPE;
  if (!host_batch_writes_back(params)) return;
  if (KTraits::NUM_MMA_KV == 1 && threadIdx.z != 0) return;
  const uint32_t lane_idx = threadIdx.x;
  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(storage.ckv_smem[stage_idx]);
  smem_t<KTraits::SWIZZLE_MODE_KPE> kpe_smem(storage.kpe_p_smem[stage_idx]);
#pragma unroll
  for (uint32_t j = 0; j < kHostBatchTileRows<KTraits>; ++j) {
    const uint32_t smem_row = host_batch_tile_row<KTraits>(j);
    const uint32_t kv_idx = kv_base + smem_row;
    if (kv_idx >= kv_len) continue;
    const int32_t victim =
        host_batch_tile_victims<KTraits>()[stage_idx * KTraits::CTA_TILE_KV + smem_row];
    if (victim < 0) continue;
    DTypeKV* destination = params.staging + static_cast<int64_t>(victim) * ROW +
                           (lane_idx % 8) * upcast_size<DTypeKV>();
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 4; ++mma_d) {
      *reinterpret_cast<b128_t*>(destination + mma_d * 8 * upcast_size<DTypeKV>()) =
          ckv_smem.base[ckv_smem.template get_permuted_offset<UPCAST_STRIDE_CKV>(
              smem_row, 8 * mma_d + lane_idx % 8)];
    }
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_KPE / 4; ++mma_d) {
      *reinterpret_cast<b128_t*>(destination + KTraits::HEAD_DIM_CKV +
                                 mma_d * 8 * upcast_size<DTypeKV>()) =
          kpe_smem.base[kpe_smem.template get_permuted_offset<UPCAST_STRIDE_KPE>(
              smem_row, 8 * mma_d + lane_idx % 8)];
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
#ifdef FLASHINFER_FAR_EMU
  ::flashinfer::far_emu::State far_emu_state{0, 0, -1};
  if constexpr (!Params::fully_staged) ::flashinfer::far_emu::init<NUM_STAGES>();
#endif
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
                                  (kv_tile_idx - stage_idx) % NUM_STAGES, index_base);
      cp_async::commit_group();
    }
  }

  // The noncausal mask only depends on the KV position, so no query offset is passed.
#pragma unroll 1
  for (; kv_tile_idx >= mask_tile_idx && kv_tile_idx > 0; --kv_tile_idx) {
    const int next = kv_tile_idx - NUM_STAGES;
    if (next >= 0) read_host_batch_pages<KTraits>(params, index_base, kv_len, next * CTA_TILE_KV, pages);
    cp_async::wait_group<NUM_STAGES - 1>();
    if constexpr (!Params::fully_staged) FAR_EMU_WAIT(NUM_STAGES, kv_tile_idx % NUM_STAGES);
    __syncthreads();
    if constexpr (!Params::fully_staged) FAR_EMU_PREFETCH(NUM_STAGES, (kv_tile_idx + NUM_STAGES - 1) % NUM_STAGES);
    writeback_host_batch_tile<KTraits>(params, storage, index_base, kv_len,
                                       kv_tile_idx * CTA_TILE_KV, kv_tile_idx % NUM_STAGES);
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
                                  next % NUM_STAGES, index_base);
      cp_async::commit_group();
    }
  }

#pragma unroll 1
  for (; kv_tile_idx + 1 > NUM_STAGES; --kv_tile_idx) {
    const int next = kv_tile_idx - NUM_STAGES;
    read_host_batch_pages<KTraits>(params, index_base, kv_len, next * CTA_TILE_KV, pages);
    cp_async::wait_group<NUM_STAGES - 1>();
    if constexpr (!Params::fully_staged) FAR_EMU_WAIT(NUM_STAGES, kv_tile_idx % NUM_STAGES);
    __syncthreads();
    if constexpr (!Params::fully_staged) FAR_EMU_PREFETCH(NUM_STAGES, (kv_tile_idx + NUM_STAGES - 1) % NUM_STAGES);
    writeback_host_batch_tile<KTraits>(params, storage, index_base, kv_len,
                                       kv_tile_idx * CTA_TILE_KV, kv_tile_idx % NUM_STAGES);
    compute_mla_qk<KTraits>(&storage, kv_tile_idx % NUM_STAGES, s_frag);
    update_mdo_states_<KTraits>(&storage, kv_tile_idx % NUM_STAGES, variant, s_frag, o_frag, m, d);
    compute_mla_pv<KTraits>(&storage, kv_tile_idx % NUM_STAGES, s_frag, d, o_frag);
    if (next < host_tiles) {
      wait_host_batch_tile<KTraits>(params, scratch, index_base, kv_len, next * CTA_TILE_KV);
    }
    __syncthreads();
    load_host_batch_kv<KTraits>(&storage, params, pages, kv_len, next * CTA_TILE_KV,
                                next % NUM_STAGES, index_base);
    cp_async::commit_group();
  }
  cp_async::wait_group<0>();
  if constexpr (!Params::fully_staged) FAR_EMU_WAIT_ALL(NUM_STAGES);
  __syncthreads();

#pragma unroll
  for (; kv_tile_idx >= 0; --kv_tile_idx) {
    writeback_host_batch_tile<KTraits>(params, storage, index_base, kv_len,
                                       kv_tile_idx * CTA_TILE_KV, kv_tile_idx % NUM_STAGES);
    compute_mla_qk<KTraits>(&storage, kv_tile_idx % NUM_STAGES, s_frag);
    logits_mask_<KTraits>(0, kv_tile_idx * CTA_TILE_KV, 1, kv_len, kv_len, params.num_heads,
                          s_frag);
    update_mdo_states_<KTraits>(&storage, kv_tile_idx % NUM_STAGES, variant, s_frag, o_frag, m, d);
    compute_mla_pv<KTraits>(&storage, kv_tile_idx % NUM_STAGES, s_frag, d, o_frag);
  }
}

// Exclusive scan of one value per thread over the CTA; returns (prefix, total). Two barriers.
template <uint32_t THREADS>
__device__ __forceinline__ uint2 host_batch_thread_scan(uint32_t value, uint32_t* warp_sums) {
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t lane = tid % 32, warp = tid / 32;
  uint32_t sum = value;
#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    const uint32_t other = __shfl_up_sync(0xffffffff, sum, offset);
    if (lane >= offset) sum += other;
  }
  if (lane == 31) warp_sums[warp] = sum;
  __syncthreads();
  uint32_t prefix = sum - value, total = 0;
#pragma unroll
  for (uint32_t w = 0; w < THREADS / 32; ++w) {
    if (w < warp) prefix += warp_sums[w];
    total += warp_sums[w];
  }
  __syncthreads();
  return make_uint2(prefix, total);
}

// In place: the claiming entry of the pool's staged-th miss, from the row bases of the claims.
template <typename Params>
__device__ __forceinline__ int32_t host_batch_claim_entry(const Params& params, uint32_t pool,
                                                          const uint32_t* row_base,
                                                          uint32_t staged) {
  uint32_t low = params.pool_row_start[pool], high = params.pool_row_start[pool + 1] - 1;
  while (low < high) {
    const uint32_t mid = (low + high + 1) / 2;
    if (row_base[mid] <= staged) low = mid; else high = mid - 1;
  }
  return params.claim_entry[params.kv_indptr[low] + staged - row_base[low]];
}

// LRU item of one pool: give each page the pool's rows missed this launch a cache slot, taking
// the least recently used slots that no entry of this launch hits, and publish the pool's flag
// for the fetch loop, which copies the pages in (see fetch_host_batch_pages). Runs after the
// second grid barrier, when every hit of this launch is stamped and every missed page has its
// slot. Stamps are launch epochs, so this is exact LRU at launch granularity: entries hit in the
// same launch are equally recent, as in a per-step LRU list.
// At small batch this item sits on the critical path (the fetch loop waits for its victims), so
// it is built for latency. The stamps are read once, coalesced, into shared memory; a radix
// select over only the bytes in which the keys differ finds the m-th oldest (stamps are small
// epochs, so usually one pass), with warp-aggregated histogram atomics since most keys share a
// value; the victims are listed in shared memory, and every thread then binds a share of them.
// Victims pair with missed pages in any order. Reading the stamps per thread in contiguous runs,
// and binding victims in the thread that found them, serialized dependent global round trips:
// ~23 us for 4096 slots and 150 victims.
// Not inlined: the tile loop that shares this item loop must keep its registers.
// In place (host_batch_writes_back) it runs between the grid barriers instead, from each CTA's
// own scan of the row claims (`row_base`), and only chooses and stamps victims: the attention
// CTAs that load the pages bind them (see writeback_host_batch_tile), so it never waits for or
// re-reads a page.
template <typename KTraits, typename Params>
__device__ __noinline__ void run_host_batch_lru(const Params& params, uint8_t* smem,
                                                uint32_t pool, uint32_t epoch, uint32_t unique,
                                                const uint32_t* row_base = nullptr) {
  constexpr uint32_t THREADS = KTraits::NUM_THREADS;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t lane = tid % 32;
  const uint32_t stamp = epoch + 1;
  const uint32_t capacity = params.pool_capacity;
  const uint32_t count = params.stage ? min(unique, params.staging_capacity) : unique;
  const auto slot_base = [&](uint32_t row) {
    return row_base ? (row < params.num_rows ? row_base[row] : unique) : params.row_slot_base[row];
  };
  const uint32_t first = min(slot_base(params.pool_row_start[pool]), count);
  const uint32_t last = min(slot_base(params.pool_row_start[pool + 1]), count);
  if (first == last) {
    if (tid == 0 && !row_base) {
      asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(params.pool_flags + pool),
                   "r"(epoch + 1) : "memory");
    }
    return;
  }
  __shared__ uint32_t histogram[256], warp_sums[THREADS / 32], selected[4];
  // In place, the row bases occupy the start of the shared memory.
  uint32_t* keys = reinterpret_cast<uint32_t*>(smem) + (row_base ? round_up(params.num_rows, 32u) : 0u);
  uint32_t* victim_slots = keys + capacity;
  const uint32_t base = static_cast<uint32_t>(params.pool_id[pool]) * capacity;
  const uint32_t* stamps = params.slot_stamp + base;

  // Slots hit this launch sort last and are never taken.
  uint32_t evictable = 0, low = ~0u, high = 0;
#pragma unroll 8
  for (uint32_t i = tid; i < capacity; i += THREADS) {
    const uint32_t value = stamps[i];
    const uint32_t key = value == stamp ? ~0u : value;
    keys[i] = key;
    if (key != ~0u) {
      ++evictable;
      low = min(low, key);
      high = max(high, key);
    }
  }
#pragma unroll
  for (uint32_t offset = 16; offset > 0; offset /= 2) {
    evictable += __shfl_xor_sync(0xffffffff, evictable, offset);
    low = min(low, __shfl_xor_sync(0xffffffff, low, offset));
    high = max(high, __shfl_xor_sync(0xffffffff, high, offset));
  }
  if (tid == 0) {
    selected[0] = 0;
    selected[1] = ~0u;
    selected[2] = 0;
  }
  __syncthreads();
  if (lane == 0) {
    atomicAdd(selected, evictable);
    atomicMin(selected + 1, low);
    atomicMax(selected + 2, high);
  }
  __syncthreads();
  const uint32_t victims = min(last - first, selected[0]);
  low = selected[1];
  high = selected[2];
  __syncthreads();

  if (victims > 0) {
    // The victims-th smallest evictable key: bytes above the highest one in which keys differ
    // are the common prefix already.
    const uint32_t differ = low ^ high;
    int shift = differ == 0 ? -8 : (31 - __clz(differ)) / 8 * 8;
    uint32_t mask = shift >= 24 ? 0u : ~0u << (shift + 8);
    uint32_t prefix = low & mask, rank = victims;
#pragma unroll 1
    for (; shift >= 0; shift -= 8) {
      for (uint32_t b = tid; b < 256; b += THREADS) histogram[b] = 0;
      __syncthreads();
      for (uint32_t base = 0; base < capacity; base += THREADS) {
        const uint32_t i = base + tid;
        const uint32_t key = i < capacity ? keys[i] : ~0u;
        const bool counted = key != ~0u && (key & mask) == prefix;
        const uint32_t bin = counted ? key >> shift & 255 : 256 + lane;
        // Slots filled in the same launch are adjacent, so whole warps often share a bin.
        if (__all_sync(0xffffffff, bin == __shfl_sync(0xffffffff, bin, 0))) {
          if (lane == 0 && counted) atomicAdd(histogram + bin, 32u);
        } else {
          const uint32_t peers = __match_any_sync(0xffffffff, bin);
          if (counted && lane == __ffs(peers) - 1) atomicAdd(histogram + bin, __popc(peers));
        }
      }
      __syncthreads();
      // Warp 0 finds the bin holding the rank-th key: a scan over lanes of eight bins each.
      // A serial scan of the 256 bins by one thread took ~5 us.
      if (tid < 32) {
        uint32_t bins[8], sum = 0;
#pragma unroll
        for (uint32_t b = 0; b < 8; ++b) sum += bins[b] = histogram[lane * 8 + b];
        uint32_t inclusive = sum;
#pragma unroll
        for (uint32_t offset = 1; offset < 32; offset *= 2) {
          const uint32_t other = __shfl_up_sync(0xffffffff, inclusive, offset);
          if (lane >= offset) inclusive += other;
        }
        const uint32_t owner = __ffs(__ballot_sync(0xffffffff, inclusive >= rank)) - 1;
        if (lane == owner) {
          // Rescan from shared memory: a dynamic index into bins[] would put it on the stack.
          uint32_t below = inclusive - sum, b = 0;
          while (below + histogram[lane * 8 + b] < rank) below += histogram[lane * 8 + b++];
          selected[0] = prefix | (lane * 8 + b) << shift;
          selected[1] = rank - below;
        }
      }
      __syncthreads();
      prefix = selected[0];
      rank = selected[1];
      mask |= 255u << shift;
      __syncthreads();
    }
    // List every evictable key below the threshold, then the first `rank` keys equal to it.
    const uint32_t threshold = prefix;
    uint32_t less = 0, equal = 0;
    for (uint32_t i = tid; i < capacity; i += THREADS) {
      less += keys[i] < threshold;
      equal += keys[i] == threshold;
    }
    // Counts in either half stay below 2^16 (pool capacity <= 8192), so one scan gives both
    // prefixes without a carry between them. Equal-key victims consume the first `rank` ties.
    const uint32_t packed = host_batch_thread_scan<THREADS>(less | (equal << 16), warp_sums).x;
    const uint32_t equal_before = packed >> 16;
    uint32_t take_equal = min(equal, rank > equal_before ? rank - equal_before : 0u);
    uint32_t position = (packed & 65535u) + min(equal_before, rank);
    for (uint32_t i = tid; i < capacity; i += THREADS) {
      const uint32_t key = keys[i];
      if (key < threshold || (key == threshold && take_equal && take_equal--)) {
        victim_slots[position++] = i;
      }
    }
    __syncthreads();
#pragma unroll 1
    for (uint32_t r = tid; r < victims; r += THREADS) {
      const uint32_t staged = first + r;
      const int32_t slot = base + victim_slots[r];
      params.slot_stamp[slot] = stamp;
      if (row_base) {
        // In place: bind here, so the attention only copies the bytes. The page's claiming entry
        // is the rank-th claim of the pool row whose slot range holds `staged`.
        const int32_t entry = host_batch_claim_entry(params, pool, row_base, staged);
        const auto page = params.remap[entry] & ~kHostBatchOwnerBit;
        params.entry_victim[entry] = slot;
        params.host_slot[page - params.host_page_start] = 0;
        const int32_t evicted = params.slot_page[slot];
        if (evicted >= 0) params.resident[evicted - params.host_page_start] = -1;
        params.resident[page - params.host_page_start] = slot;
        params.slot_page[slot] = page;
      } else {
        params.victim_of[staged] = slot;
      }
    }
  }
  // Pages beyond the pool's free slots stay uncached.
  for (uint32_t staged = first + victims + tid; staged < last; staged += THREADS) {
    if (row_base) {
      const int32_t entry = host_batch_claim_entry(params, pool, row_base, staged);
      params.entry_victim[entry] = -1;
      params.host_slot[(params.remap[entry] & ~kHostBatchOwnerBit) - params.host_page_start] = 0;
    } else {
      params.victim_of[staged] = -1;
    }
  }
  __threadfence();
  __syncthreads();
  if (tid == 0 && !row_base) {
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
template <typename KTraits, typename Params, bool ROW_MERGE>
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
  if (Params::whole_rows || item < full_items) {
    row = item / params.num_head_tiles;
    head_tile = item % params.num_head_tiles;
    kv_start = 0;
    kv_len = host_batch_row_len(params, row);
    host_tiles = ceil_div(params.state[kHostBatchHostCounts + row], CTA_TILE_KV);
  } else if (item < hbm_end) {
    const uint32_t tile = (item - full_items) / params.num_chunks;
    const uint32_t chunk = (item - full_items) % params.num_chunks;
    row = params.full_rows + tile / params.num_head_tiles;
    head_tile = tile % params.num_head_tiles;
    partial = item - full_items;
    const uint32_t length = host_batch_row_len(params, row);
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
  // A row of only padding (kv_len 0, e.g. a CUDA Graph filler row) writes a zero output from the
  // initial state (lse at its -5e4 start); it must not issue load_q, whose copies join the first
  // KV group.
  if (kv_len > 0) {
    load_q<KTraits>(&storage, params.q_nope + row * params.q_nope_stride_n,
                    params.q_pe + row * params.q_pe_stride_n, params.q_nope_stride_n,
                    params.q_nope_stride_h, params.q_pe_stride_n, params.q_pe_stride_h, 1,
                    head_tile * CTA_TILE_Q, params.num_heads);
    // load_q's copies join the first KV group, as in the native kernel.
    attend_host_batch_tiles<KTraits>(params, storage, variant, scratch,
                                     params.kv_indptr[row] + kv_start, kv_len, host_tiles, s_frag,
                                     o_frag, m, d);
  }
  __syncthreads();
  normalize_d_<KTraits>(&storage, 0, o_frag, m, d);
  finalize_m_<KTraits>(variant, m);

  // Each warp reads back output rows other lanes stored to shared memory.
  if (Params::whole_rows || item < full_items) {
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
  if constexpr (ROW_MERGE) {
    if (item >= full_items) {
      // Publish every thread's partial writes before advancing this row's completed item count.
      __threadfence();
      __syncthreads();
      const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
      if (tid == 0) atomicAdd(host_batch_row_completed(params) + row, 1u);
    }
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
template <typename KTraits, typename Params, bool ROW_MERGE>
__device__ __forceinline__ void run_host_batch_items(const Params& params,
                                                     typename KTraits::SharedStorage& storage,
                                                     uint8_t* smem, uint32_t* scratch,
                                                     bool fetcher, uint32_t unique) {
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  // With no Host page the item list has no Host items, and host_batch_host_items was never
  // written this launch, so it must not be read.
  const int lru_items = host_batch_caching(params) && params.stage && unique ? params.num_pools : 0;
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
    run_host_batch_item<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, item - lru_items, s_frag, o_frag,
                                 m, d);
  }
}

// Optional short-segment prepare and single-warp scans for high-latency tuning.
// Keep the default translation unit identical to the measured baseline.
#ifdef FLASHINFER_HOST_BATCH_SMALL_PREPARE
template <typename KTraits, typename Params>
__device__ __forceinline__ void prepare_host_batch_small_rows(const Params& params, uint32_t stamp) {
  using IdType = typename Params::IdType;
  constexpr uint32_t THREADS = KTraits::NUM_THREADS;
  // Small launches already split each row into short segments. Avoid expanding seven padded
  // entries per thread when a segment fits in one CTA; the original path keeps wider loads.
  constexpr uint32_t PER_THREAD = 1;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t lane = tid % 32;
  const uint32_t lower = (1u << lane) - 1;
  const uint32_t units = params.num_rows * params.prepare_parts;
  uint32_t* host_cursors = params.state + kHostBatchHostCounts;
  uint32_t* local_cursors = host_batch_local_cursors(params);
#pragma unroll 1
  for (uint32_t unit = blockIdx.y; unit < units; unit += gridDim.y) {
    const uint32_t row = unit / params.prepare_parts;
    const IdType begin = params.kv_indptr[row];
    const uint32_t length = params.kv_indptr[row + 1] - begin;
    const uint32_t part_begin = unit % params.prepare_parts * params.prepare_segment;
    const uint32_t part_end = min(part_begin + params.prepare_segment, length);
#pragma unroll 1
    for (uint32_t offset = part_begin; offset < part_end; offset += PER_THREAD * THREADS) {
      IdType pages[PER_THREAD];
      uint32_t claimed[PER_THREAD];
#pragma unroll
      for (uint32_t i = 0; i < PER_THREAD; ++i) {
        const uint32_t j = offset + i * THREADS + tid;
        pages[i] = j < part_end ? params.kv_indices[begin + j] : kHostBatchNoEntry;
      }
      // Negative entries (padding) are dropped; their positions leave a gap between the Host
      // part and the HBM part that resolve_host_batch_slots closes.
      bool dropped = false;
#pragma unroll
      for (uint32_t i = 0; i < PER_THREAD; ++i) {
        if (pages[i] < 0 && pages[i] != kHostBatchNoEntry) {
          pages[i] = kHostBatchNoEntry;
          dropped = true;
        }
      }
      if (__any_sync(0xffffffff, dropped) && lane == 0) params.state[kHostBatchGaps] = 1;
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
      uint32_t fresh_masks[PER_THREAD], host_masks[PER_THREAD], local_masks[PER_THREAD];
      uint32_t fresh_total = 0, host_total = 0, local_total = 0;
#pragma unroll
      for (uint32_t i = 0; i < PER_THREAD; ++i) {
        fresh_masks[i] =
            __ballot_sync(0xffffffff, pages[i] >= params.host_page_start && claimed[i] == 0);
        host_masks[i] = __ballot_sync(0xffffffff, pages[i] >= params.host_page_start);
        // Cached entries are negative; kHostBatchNoEntry pads short passes.
        local_masks[i] = __ballot_sync(
            0xffffffff, pages[i] != kHostBatchNoEntry && pages[i] < params.host_page_start);
        fresh_total += __popc(fresh_masks[i]);
        host_total += __popc(host_masks[i]);
        local_total += __popc(local_masks[i]);
      }
      uint32_t first = 0, host_first = 0, local_first = 0;
      if (lane == 0) {
        if (fresh_total) first = atomicAdd(host_batch_row_claims(params) + row, fresh_total);
        if (host_total) host_first = atomicAdd(host_cursors + row, host_total);
        if (local_total) local_first = atomicAdd(local_cursors + row, local_total);
      }
      first = __shfl_sync(0xffffffff, first, 0);
      host_first = __shfl_sync(0xffffffff, host_first, 0);
      local_first = __shfl_sync(0xffffffff, local_first, 0);
#pragma unroll
      for (uint32_t i = 0; i < PER_THREAD; ++i) {
        if (fresh_masks[i] >> lane & 1) {
          const uint32_t rank = first + __popc(fresh_masks[i] & lower);
          atomicExch(params.host_slot + (pages[i] - params.host_page_start),
                     (row << kHostBatchRankBits | rank) + 2);
        }
        if (pages[i] >= params.host_page_start) {
          params.remap[begin + host_first + __popc(host_masks[i] & lower)] = pages[i];
        } else if (pages[i] != kHostBatchNoEntry) {
          params.remap[begin + length - 1 - local_first - __popc(local_masks[i] & lower)] = pages[i];
        }
        first += __popc(fresh_masks[i]);
        host_first += __popc(host_masks[i]);
        local_first += __popc(local_masks[i]);
      }
    }
  }
}

template <uint32_t THREADS, typename Value>
__device__ __forceinline__ uint32_t host_batch_small_scan(uint32_t n, Value value, uint32_t* out) {
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  __shared__ uint32_t warp_sums[THREADS / 32], carry;
    // The launch dispatch guarantees at most 32 rows. One warp publishes every prefix and
    // the total before the CTA reads them, without the multi-warp carry barriers.
    if (tid < 32) {
      const uint32_t count = tid < n ? value(tid) : 0;
      uint32_t sum = count;
#pragma unroll
      for (uint32_t offset = 1; offset < 32; offset *= 2) {
        const uint32_t other = __shfl_up_sync(0xffffffff, sum, offset);
        if (tid >= offset) sum += other;
      }
      if (tid < n) out[tid] = sum - count;
      if (tid == 31) carry = sum;
    }
    __syncthreads();
    return carry;
}

template <typename KTraits, typename Params>
__device__ __forceinline__ uint32_t resolve_host_batch_small_slots(const Params& params, uint8_t* smem) {
  using IdType = typename Params::IdType;
  constexpr uint32_t THREADS = KTraits::NUM_THREADS;
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  uint32_t* row_base = reinterpret_cast<uint32_t*>(smem);
  const uint32_t* row_claims = host_batch_row_claims(params);
  const uint32_t unique = host_batch_small_scan<THREADS>(
      params.num_rows, [&](uint32_t row) { return row_claims[row]; }, row_base);
  // No Host page anywhere: nothing to rewrite, and no Host items to list. The caller then skips
  // the second grid barrier and the fetch entirely.
  if (unique == 0) return 0;
  // Same (row, segment) units as the prepare step, so a few rows still spread over the grid. In
  // place with the cache, prepare already marked the claiming entries (no page ids to resolve).
  const uint32_t units = host_batch_writes_back(params) ? 0 : params.num_rows * params.prepare_parts;
#pragma unroll 1
  for (uint32_t unit = blockIdx.y; unit < units; unit += gridDim.y) {
    const uint32_t row = unit / params.prepare_parts;
    const IdType begin = params.kv_indptr[row];
    // Host entries fill each row's leading positions, or anywhere when not staging.
    const uint32_t host = params.stage ? params.state[kHostBatchHostCounts + row]
                                       : host_batch_row_len(params, row);
    const uint32_t part_begin = unit % params.prepare_parts * params.prepare_segment;
    const uint32_t part_end = min(part_begin + params.prepare_segment, host);
#pragma unroll 1
    for (uint32_t j = part_begin + tid; j < part_end; j += THREADS) {
      const IdType page = params.remap[begin + j];
      if (!params.stage && page < params.host_page_start) continue;
      const uint32_t claim = params.host_slot[page - params.host_page_start] - 2;
      const uint32_t owner = claim >> kHostBatchRankBits;
      const uint32_t slot = row_base[owner] + (claim & ((1u << kHostBatchRankBits) - 1));
      if (owner == row) {
        params.unique_pages[slot] = page;
        // (In place with the cache prepare already marked the claims; this loop does not run.)
        if (host_batch_caching(params) && slot < params.staging_capacity) {
          params.slot_pool[slot] = params.row_pool[row];
        }
      }
      if (params.stage && slot < params.staging_capacity) {
        params.remap[begin + j] = host_batch_staged_entry(params, slot);
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
    const uint32_t total = host_batch_small_scan<THREADS>(tail_rows, items, host_items);
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

#endif

template <typename KTraits, typename Params, bool ROW_MERGE>
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
  __shared__ uint32_t scratch[3];
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  // The previous launch's last CTA advanced the epoch; ready flags from it no longer match.
  const uint32_t epoch = params.state[kHostBatchEpoch];

  // Not staging and no cache: nothing to prepare; remap is kv_indices (see the launcher) and
  // the host counts stay zero.
  const bool skip_prepare = !params.stage && !host_batch_caching(params);
  uint32_t unique = 0;
  // Grid-uniform: every CTA reads the flag after the same barrier.
  bool gaps = false;
  if (!skip_prepare) {
    if (params.stage) {
#ifdef FLASHINFER_HOST_BATCH_SMALL_PREPARE
      if constexpr (ROW_MERGE) prepare_host_batch_small_rows<KTraits>(params, epoch + 1);
      else prepare_host_batch_rows<KTraits>(params, epoch + 1);
#else
      prepare_host_batch_rows<KTraits>(params, epoch + 1);
#endif
    } else {
      prepare_host_batch_rows_in_place<KTraits>(params, epoch + 1);
    }
    cg::this_grid().sync();
    gaps = params.stage && host_batch_ld_acquire(params.state + kHostBatchGaps) != 0;
#ifdef FLASHINFER_HOST_BATCH_SMALL_PREPARE
    if constexpr (ROW_MERGE) unique = resolve_host_batch_small_slots<KTraits>(params, smem);
    else unique = resolve_host_batch_slots<KTraits>(params, smem);
#else
    unique = resolve_host_batch_slots<KTraits>(params, smem);
#endif
    if (gaps) compact_host_batch_rows<KTraits>(params);
    // In place, victims are chosen now, so they are known when the attention loads the pages.
    if (host_batch_writes_back(params) && unique > 0) {
#pragma unroll 1
      for (uint32_t pool = blockIdx.y; pool < params.num_pools; pool += gridDim.y) {
        run_host_batch_lru<KTraits>(params, smem, pool, epoch, unique,
                                    reinterpret_cast<const uint32_t*>(smem));
        __syncthreads();
      }
    }
    if (blockIdx.y == 0 && tid == 0) params.state[kHostBatchUnique] = unique;
  }
  // unique is the same on every CTA (each runs the same scan), so the branch is grid-uniform and
  // the cooperative barriers below stay matched.
  // Which end of the item list this CTA walks from. This is a partition of the grid and must not
  // depend on `unique`: if the back CTAs also walked from the front they would retrace the same
  // items and contend on the claims (0% Host, 512 rows: 1514 -> 1784 us).
  const bool from_back = blockIdx.y >= gridDim.y - params.fetch_ctas;
  if (unique > 0 || gaps) {
    // Rows resolved or compacted by other CTAs must be complete before any item or fetch reads them, and fetch
    // CTAs clear the dedup slots the resolve step reads.
    cg::this_grid().sync();
    // Claim counts were read by every CTA's scan; reset them for the next launch.
    for (uint32_t row = blockIdx.y * KTraits::NUM_THREADS + tid; row < params.num_rows;
         row += gridDim.y * KTraits::NUM_THREADS) {
      host_batch_row_claims(params)[row] = 0;
    }

  }
  const uint32_t fetch_rows = host_batch_fetch_rows<ROW_MERGE>(params, unique);
  if (tid == 0) {
    scratch[kHostBatchScratchEpoch] = epoch;
    scratch[kHostBatchScratchFetchRows] = fetch_rows;
  }
  __syncthreads();
  // Fetch CTAs take attention items once their copies are issued, so low Host shares and the end
  // of the PCIe stream keep every SM busy; with no Host page there is nothing to fetch.
  if (unique > 0 && from_back && !host_batch_writes_back(params)) fetch_host_batch_pages<KTraits>(params, smem, epoch, unique, fetch_rows);
  run_host_batch_items<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, from_back, unique);
  // Uniform over the grid, so every CTA takes the same branch.
  if (!Params::whole_rows && params.full_rows < params.num_rows) {
    // A small staged merge waits only for the row it reads, overlapping the remaining rows and
    // cache writeback. Every other path retains the full-grid merge barrier.
    if constexpr (!ROW_MERGE) cg::this_grid().sync();
    // 32 features per thread fill the grid only with enough (row tile, head) units; a small batch
    // merged on 32 of 108 SMs at that width and took ~15 us for 8 rows, so it uses 8.
    const uint32_t units = (params.num_rows - params.full_rows) * params.num_head_tiles *
                           KTraits::CTA_TILE_Q;
    if (params.num_rows == 1) {
      merge_host_batch_single_query<KTraits, Params, ROW_MERGE>(params, smem);
    } else if (units >= gridDim.y * (KTraits::NUM_THREADS / (KTraits::HEAD_DIM_CKV / 32))) {
      merge_host_batch_chunks<KTraits, 32, Params, ROW_MERGE>(params);
    } else {
      merge_host_batch_chunks<KTraits, 8, Params, ROW_MERGE>(params);
    }
  }

  // The last CTA to leave resets the shared counters for the next launch: the exit count, the
  // per-row placement cursors (Host counts included), and then the epoch.
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    scratch[kHostBatchScratchClaimed] =
        atomicAdd(params.state + kHostBatchExit, 1u) == gridDim.y - 1;
  }
  __syncthreads();
  if (scratch[kHostBatchScratchClaimed]) {
    if (!skip_prepare) {
      for (uint32_t row = tid; row < params.num_rows; row += KTraits::NUM_THREADS) {
        params.state[kHostBatchHostCounts + row] = 0;
        host_batch_local_cursors(params)[row] = 0;
      }
      if (tid == 0) params.state[kHostBatchGaps] = 0;
    }
    if constexpr (ROW_MERGE) {
      for (uint32_t row = tid; row < params.num_rows; row += KTraits::NUM_THREADS)
        host_batch_row_completed(params)[row] = 0;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      params.state[kHostBatchExit] = 0;
      params.state[kHostBatchEpoch] = epoch + 1;
    }
  }
}

// Host-only schedule estimate for the medium whole-row dispatch. The original tail planner
// charges six KV tiles for Q/partial/merge traffic; direct rows charge four (Q and final O).
// Model the existing two-ended cyclic item assignment with no remote stall. A short last
// wave of whole rows must not replace a faster split plan just because there are two SM waves.
struct HostBatchCostEvent {
  uint64_t finish;
  uint32_t cta;
  int32_t item;
};
struct HostBatchCostLater {
  bool operator()(const HostBatchCostEvent& a, const HostBatchCostEvent& b) const {
    return a.finish > b.finish || (a.finish == b.finish && a.cta > b.cta);
  }
};
static inline bool host_batch_balanced_whole_rows_uncached(uint32_t rows, uint32_t max_kv_len,
                                          uint32_t grid, uint32_t fetch,
                                          uint32_t partial_capacity) {
  constexpr uint32_t TILE = 32;
  if (grid == 0 || fetch == 0 || fetch >= grid) return false;
  const uint32_t workers = grid - fetch;
  const uint32_t per_worker = rows / grid > 0 ? rows / grid - 1 : 0;
  const uint32_t full = min(rows, workers * per_worker);
  const uint32_t tail = rows - full;
  if (2 * tail > partial_capacity || tail == 0) return true;
  const uint32_t length = max(max_kv_len, 1u);
  uint32_t chunks = 1;
  uint64_t best = UINT64_MAX;
  for (uint32_t c = 1; c <= 32; ++c) {
    const uint32_t chunk_len = round_up(ceil_div(length, c), TILE);
    if (c > 1 && tail * (c + 1) > partial_capacity) break;
    if (c > 1 && chunk_len * (c - 1) >= length) continue;
    const uint64_t cost = ceil_div(tail * c, grid) * (chunk_len / TILE + 6u);
    if (cost < best) { best = cost; chunks = c; }
  }
  const uint32_t chunk_len = round_up(ceil_div(length, chunks), TILE);
  auto predict = [&](bool whole) {
    const uint32_t items = whole ? rows : full + tail * chunks;
    std::vector<uint8_t> claimed(items, 0);
    std::priority_queue<HostBatchCostEvent, std::vector<HostBatchCostEvent>,
                        HostBatchCostLater> pending;
    for (uint32_t cta = 0; cta < grid; ++cta) {
      const int32_t first = cta < workers ? static_cast<int32_t>(cta)
          : static_cast<int32_t>(items) - 1 - static_cast<int32_t>(cta - workers);
      pending.push({0, cta, first});
    }
    uint64_t finish = 0;
    while (!pending.empty()) {
      auto event = pending.top(); pending.pop();
      const bool back = event.cta >= workers;
      const int32_t step = back ? -static_cast<int32_t>(fetch)
                               : static_cast<int32_t>(workers);
      while (event.item >= 0 && static_cast<uint32_t>(event.item) < items &&
             claimed[event.item]) {
        if (back) { event.item = -1; break; }
        event.item += step;
      }
      if (event.item < 0 || static_cast<uint32_t>(event.item) >= items) continue;
      claimed[event.item] = 1;
      uint32_t cost = ceil_div(length, TILE) + 4u;
      if (!whole && static_cast<uint32_t>(event.item) >= full) {
        const uint32_t chunk = (static_cast<uint32_t>(event.item) - full) % chunks;
        cost = ceil_div(min(chunk_len, length - chunk * chunk_len), TILE) + 6u;
      }
      event.finish += cost;
      finish = event.finish > finish ? event.finish : finish;
      event.item += step;
      pending.push(event);
    }
    return finish;
  };
  return predict(true) <= predict(false);
}


// One bounded host memo avoids re-running shape planning on a stable decode shape. It
// depends only on the five scalar inputs, never on KV contents, device state or link timing.
struct HostBatchShapeMemo {
  uint32_t rows, length, grid, fetch, capacity;
  bool valid, whole;
};
static inline bool host_batch_balanced_whole_rows(uint32_t rows, uint32_t max_kv_len,
                                                 uint32_t grid, uint32_t fetch,
                                                 uint32_t partial_capacity) {
  static thread_local HostBatchShapeMemo last{};
  if (last.valid && last.rows == rows && last.length == max_kv_len && last.grid == grid &&
      last.fetch == fetch && last.capacity == partial_capacity) return last.whole;
  const bool whole = host_batch_balanced_whole_rows_uncached(rows, max_kv_len, grid, fetch,
                                                            partial_capacity);
  last = {rows, max_kv_len, grid, fetch, partial_capacity, true, whole};
  return whole;
}

// Two different KV tiles share each Q fragment. Each score accumulator keeps the
// native PE-then-CKV MMA order; this performs no duplicate QK work.
template <bool INIT, typename KTraits, uint32_t D, uint32_t Q_STRIDE,
          uint32_t K_STRIDE, SwizzleMode Q_SWIZZLE, SwizzleMode K_SWIZZLE>
__device__ __forceinline__ void host_batch_qk_pair_fragments(
    smem_t<Q_SWIZZLE> q, smem_t<K_SWIZZLE> ka, smem_t<K_SWIZZLE> kb,
    float (*sa)[8], float (*sb)[8]) {
  static_assert(KTraits::QK_SHARD && KTraits::CTA_TILE_KV == 32);
  const uint32_t lane=threadIdx.x, group=threadIdx.z, warp=threadIdx.y;
  alignas(16) uint32_t qf[4],kf[4];
#pragma unroll 4
  for (uint32_t dim=0;dim<D;++dim) {
    const uint32_t qo=q.template get_permuted_offset<Q_STRIDE>(
        warp*16+lane%16,dim*2+lane/16);
    q.ldmatrix_m8n8x4(qo,qf);
    const uint32_t ko=ka.template get_permuted_offset<K_STRIDE>(
        group*16+8*(lane/16)+lane%8,2*dim+(lane%16)/8);
    ka.ldmatrix_m8n8x4(ko,kf);
    if (INIT && dim==0)
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ,MMAMode::kInit>(sa[0],qf,kf);
    else
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sa[0],qf,kf);
    kb.ldmatrix_m8n8x4(ko,kf);
    if (INIT && dim==0)
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ,MMAMode::kInit>(sb[0],qf,kf);
    else
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sb[0],qf,kf);
  }
}

template <typename KTraits>
__device__ __forceinline__ void host_batch_compute_qk_pair(
    typename KTraits::SharedStorage& storage,uint32_t a,uint32_t b,
    float (*sa)[8],float (*sb)[8]) {
  smem_t<KTraits::SWIZZLE_MODE_Q_PE> qp(storage.q_smem_pe);
  smem_t<KTraits::SWIZZLE_MODE_KPE> kpa(storage.kpe_p_smem[a]),kpb(storage.kpe_p_smem[b]);
  host_batch_qk_pair_fragments<true,KTraits,KTraits::NUM_MMA_D_KPE,
      KTraits::UPCAST_STRIDE_Q_PE,KTraits::UPCAST_STRIDE_KPE>(qp,kpa,kpb,sa,sb);
  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> qn(storage.q_smem_nope);
  smem_t<KTraits::SWIZZLE_MODE_CKV> kna(storage.ckv_smem[a]),knb(storage.ckv_smem[b]);
  host_batch_qk_pair_fragments<false,KTraits,KTraits::NUM_MMA_D_CKV,
      KTraits::UPCAST_STRIDE_Q_NOPE,KTraits::UPCAST_STRIDE_CKV>(qn,kna,knb,sa,sb);
}


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
    // The frozen pair helper reuses Q fragments for the two current tiles. It requires
    // both stages ready, trading the control's first-QK/load overlap for fewer Q loads.
    cp_async::wait_group<0>();
    __syncthreads();
    host_batch_compute_qk_pair<KTraits>(storage, hi%2, lo%2, scores, scores+1);
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

template <typename KTraits, typename Params, bool ROW_MERGE>
__device__ __forceinline__ void run_joint_host_item(const Params& params,
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
  if (Params::whole_rows || item < full_items) {
    row = item / params.num_head_tiles;
    head_tile = item % params.num_head_tiles;
    kv_start = 0;
    kv_len = host_batch_row_len(params, row);
    host_tiles = ceil_div(params.state[kHostBatchHostCounts + row], CTA_TILE_KV);
  } else if (item < hbm_end) {
    const uint32_t tile = (item - full_items) / params.num_chunks;
    const uint32_t chunk = (item - full_items) % params.num_chunks;
    row = params.full_rows + tile / params.num_head_tiles;
    head_tile = tile % params.num_head_tiles;
    partial = item - full_items;
    const uint32_t length = host_batch_row_len(params, row);
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
  // A row of only padding (kv_len 0, e.g. a CUDA Graph filler row) writes a zero output from the
  // initial state (lse at its -5e4 start); it must not issue load_q, whose copies join the first
  // KV group.
  if (kv_len > 0) {
    load_q<KTraits>(&storage, params.q_nope + row * params.q_nope_stride_n,
                    params.q_pe + row * params.q_pe_stride_n, params.q_nope_stride_n,
                    params.q_nope_stride_h, params.q_pe_stride_n, params.q_pe_stride_h, 1,
                    head_tile * CTA_TILE_Q, params.num_heads);
    // load_q's copies join the first KV group, as in the native kernel.
    attend_joint_host_tiles<KTraits>(params, storage, variant, scratch,
                                     params.kv_indptr[row] + kv_start, kv_len, host_tiles, s_frag,
                                     o_frag, m, d);
  }
  __syncthreads();
  normalize_d_<KTraits>(&storage, 0, o_frag, m, d);
  finalize_m_<KTraits>(variant, m);

  // Each warp reads back output rows other lanes stored to shared memory.
  if (Params::whole_rows || item < full_items) {
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
  if constexpr (ROW_MERGE) {
    if (item >= full_items) {
      // Publish every thread's partial writes before advancing this row's completed item count.
      __threadfence();
      __syncthreads();
      const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
      if (tid == 0) atomicAdd(host_batch_row_completed(params) + row, 1u);
    }
  }
}
template <typename KTraits, typename Params, bool ROW_MERGE>
__device__ __forceinline__ void run_joint_host_items(const Params& params,
                                                     typename KTraits::SharedStorage& storage,
                                                     uint8_t* smem, uint32_t* scratch,
                                                     bool fetcher, uint32_t unique) {
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  // With no Host page the item list has no Host items, and host_batch_host_items was never
  // written this launch, so it must not be read.
  const int lru_items = host_batch_caching(params) && params.stage && unique ? params.num_pools : 0;
  const int items = lru_items +
                    (params.full_rows + (params.num_rows - params.full_rows) * params.num_chunks) *
                        params.num_head_tiles +
                    (unique ? host_batch_host_items(params)[params.num_rows] : 0);
  const int workers = gridDim.y - params.fetch_ctas;
  const int first = fetcher ? items - 1 - (blockIdx.y - workers) : blockIdx.y;
  const int step = fetcher ? -static_cast<int>(params.fetch_ctas) : workers;

  float s_frag[2][8];
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
    run_joint_host_item<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, item - lru_items, s_frag, o_frag,
                                 m, d);
  }
}
template <typename KTraits, typename Params, bool ROW_MERGE>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLAJointQKKernel(
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
  __shared__ uint32_t scratch[3];
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  // The previous launch's last CTA advanced the epoch; ready flags from it no longer match.
  const uint32_t epoch = params.state[kHostBatchEpoch];

  // Not staging and no cache: nothing to prepare; remap is kv_indices (see the launcher) and
  // the host counts stay zero.
  const bool skip_prepare = !params.stage && !host_batch_caching(params);
  uint32_t unique = 0;
  // Grid-uniform: every CTA reads the flag after the same barrier.
  bool gaps = false;
  if (!skip_prepare) {
    if (params.stage) {
#ifdef FLASHINFER_HOST_BATCH_SMALL_PREPARE
      if constexpr (ROW_MERGE) prepare_host_batch_small_rows<KTraits>(params, epoch + 1);
      else prepare_host_batch_rows<KTraits>(params, epoch + 1);
#else
      prepare_host_batch_rows<KTraits>(params, epoch + 1);
#endif
    } else {
      prepare_host_batch_rows_in_place<KTraits>(params, epoch + 1);
    }
    cg::this_grid().sync();
    gaps = params.stage && host_batch_ld_acquire(params.state + kHostBatchGaps) != 0;
#ifdef FLASHINFER_HOST_BATCH_SMALL_PREPARE
    if constexpr (ROW_MERGE) unique = resolve_host_batch_small_slots<KTraits>(params, smem);
    else unique = resolve_host_batch_slots<KTraits>(params, smem);
#else
    unique = resolve_host_batch_slots<KTraits>(params, smem);
#endif
    if (gaps) compact_host_batch_rows<KTraits>(params);
    // In place, victims are chosen now, so they are known when the attention loads the pages.
    if (host_batch_writes_back(params) && unique > 0) {
#pragma unroll 1
      for (uint32_t pool = blockIdx.y; pool < params.num_pools; pool += gridDim.y) {
        run_host_batch_lru<KTraits>(params, smem, pool, epoch, unique,
                                    reinterpret_cast<const uint32_t*>(smem));
        __syncthreads();
      }
    }
    if (blockIdx.y == 0 && tid == 0) params.state[kHostBatchUnique] = unique;
  }
  // unique is the same on every CTA (each runs the same scan), so the branch is grid-uniform and
  // the cooperative barriers below stay matched.
  // Which end of the item list this CTA walks from. This is a partition of the grid and must not
  // depend on `unique`: if the back CTAs also walked from the front they would retrace the same
  // items and contend on the claims (0% Host, 512 rows: 1514 -> 1784 us).
  const bool from_back = blockIdx.y >= gridDim.y - params.fetch_ctas;
  if (unique > 0 || gaps) {
    // Rows resolved or compacted by other CTAs must be complete before any item or fetch reads them, and fetch
    // CTAs clear the dedup slots the resolve step reads.
    cg::this_grid().sync();
    // Claim counts were read by every CTA's scan; reset them for the next launch.
    for (uint32_t row = blockIdx.y * KTraits::NUM_THREADS + tid; row < params.num_rows;
         row += gridDim.y * KTraits::NUM_THREADS) {
      host_batch_row_claims(params)[row] = 0;
    }

  }
  const uint32_t fetch_rows = host_batch_fetch_rows<ROW_MERGE>(params, unique);
  if (tid == 0) {
    scratch[kHostBatchScratchEpoch] = epoch;
    scratch[kHostBatchScratchFetchRows] = fetch_rows;
  }
  __syncthreads();
  // Fetch CTAs take attention items once their copies are issued, so low Host shares and the end
  // of the PCIe stream keep every SM busy; with no Host page there is nothing to fetch.
  if (unique > 0 && from_back && !host_batch_writes_back(params)) fetch_host_batch_pages<KTraits>(params, smem, epoch, unique, fetch_rows);
  run_joint_host_items<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, from_back, unique);
  // Uniform over the grid, so every CTA takes the same branch.
  if (!Params::whole_rows && params.full_rows < params.num_rows) {
    // A small staged merge waits only for the row it reads, overlapping the remaining rows and
    // cache writeback. Every other path retains the full-grid merge barrier.
    if constexpr (!ROW_MERGE) cg::this_grid().sync();
    // 32 features per thread fill the grid only with enough (row tile, head) units; a small batch
    // merged on 32 of 108 SMs at that width and took ~15 us for 8 rows, so it uses 8.
    const uint32_t units = (params.num_rows - params.full_rows) * params.num_head_tiles *
                           KTraits::CTA_TILE_Q;
    if (params.num_rows == 1) {
      merge_host_batch_single_query<KTraits, Params, ROW_MERGE>(params, smem);
    } else if (units >= gridDim.y * (KTraits::NUM_THREADS / (KTraits::HEAD_DIM_CKV / 32))) {
      merge_host_batch_chunks<KTraits, 32, Params, ROW_MERGE>(params);
    } else {
      merge_host_batch_chunks<KTraits, 8, Params, ROW_MERGE>(params);
    }
  }

  // The last CTA to leave resets the shared counters for the next launch: the exit count, the
  // per-row placement cursors (Host counts included), and then the epoch.
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    scratch[kHostBatchScratchClaimed] =
        atomicAdd(params.state + kHostBatchExit, 1u) == gridDim.y - 1;
  }
  __syncthreads();
  if (scratch[kHostBatchScratchClaimed]) {
    if (!skip_prepare) {
      for (uint32_t row = tid; row < params.num_rows; row += KTraits::NUM_THREADS) {
        params.state[kHostBatchHostCounts + row] = 0;
        host_batch_local_cursors(params)[row] = 0;
      }
      if (tid == 0) params.state[kHostBatchGaps] = 0;
    }
    if constexpr (ROW_MERGE) {
      for (uint32_t row = tid; row < params.num_rows; row += KTraits::NUM_THREADS)
        host_batch_row_completed(params)[row] = 0;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      params.state[kHostBatchExit] = 0;
      params.state[kHostBatchEpoch] = epoch + 1;
    }
  }
}



template <typename Q, typename KV, typename O, typename Id>
struct CachedPETraits : JointQKTraits<Q, KV, O, Id> {};
template <typename Params>
struct CachedPEHostParams : JointQKHostParams<Params> {};

// Two different KV tiles share each Q fragment. Each score accumulator keeps the
// native PE-then-CKV MMA order; this performs no duplicate QK work.
template <bool INIT, typename KTraits, uint32_t D, uint32_t Q_STRIDE,
          uint32_t K_STRIDE, SwizzleMode Q_SWIZZLE, SwizzleMode K_SWIZZLE>
__device__ __forceinline__ void cached_pe_qk_pair_fragments(
    smem_t<Q_SWIZZLE> q, smem_t<K_SWIZZLE> ka, smem_t<K_SWIZZLE> kb,
    float (*sa)[8], float (*sb)[8]) {
  static_assert(KTraits::QK_SHARD && KTraits::CTA_TILE_KV == 32);
  const uint32_t lane=threadIdx.x, group=threadIdx.z, warp=threadIdx.y;
  alignas(16) uint32_t qf[4],kf[4];
#pragma unroll 4
  for (uint32_t dim=0;dim<D;++dim) {
    const uint32_t qo=q.template get_permuted_offset<Q_STRIDE>(
        warp*16+lane%16,dim*2+lane/16);
    q.ldmatrix_m8n8x4(qo,qf);
    const uint32_t ko=ka.template get_permuted_offset<K_STRIDE>(
        group*16+8*(lane/16)+lane%8,2*dim+(lane%16)/8);
    ka.ldmatrix_m8n8x4(ko,kf);
    if (INIT && dim==0)
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ,MMAMode::kInit>(sa[0],qf,kf);
    else
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sa[0],qf,kf);
    kb.ldmatrix_m8n8x4(ko,kf);
    if (INIT && dim==0)
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ,MMAMode::kInit>(sb[0],qf,kf);
    else
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sb[0],qf,kf);
  }
}


// Each lane owns four packed words per 16-feature QPE MMA fragment. The query
// head/feature map is the same map produced by native ldmatrix.x4.
template <typename Params>
__device__ __forceinline__ void cached_pe_load_query(const Params& params, uint32_t row,
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

template <bool PAIR, typename KTraits>
__device__ __forceinline__ void cached_pe_compute_pe(
    typename KTraits::SharedStorage& storage, uint32_t a, uint32_t b,
    uint32_t (&qpe)[4][4], float (*sa)[8], float (*sb)[8]) {
  const uint32_t lane = threadIdx.x, group = threadIdx.z;
  smem_t<KTraits::SWIZZLE_MODE_KPE> ka(storage.kpe_p_smem[a]), kb(storage.kpe_p_smem[b]);
  alignas(16) uint32_t kf[4];
#pragma unroll
  for (uint32_t dim = 0; dim < 4; ++dim) {
    const uint32_t offset = ka.template get_permuted_offset<KTraits::UPCAST_STRIDE_KPE>(
        group * 16 + 8 * (lane / 16) + lane % 8, 2 * dim + (lane % 16) / 8);
    ka.ldmatrix_m8n8x4(offset, kf);
    if (dim == 0)
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(sa[0], qpe[dim], kf);
    else
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sa[0], qpe[dim], kf);
    if constexpr (PAIR) {
      kb.ldmatrix_m8n8x4(offset, kf);
      if (dim == 0)
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(sb[0], qpe[dim], kf);
      else
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sb[0], qpe[dim], kf);
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void cached_pe_compute_one(
    typename KTraits::SharedStorage& storage, uint32_t (&qpe)[4][4], float (*scores)[8]) {
  cached_pe_compute_pe<false, KTraits>(storage, 0, 0, qpe, scores, scores);
  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> q(storage.q_smem_nope);
  smem_t<KTraits::SWIZZLE_MODE_CKV> kv(storage.ckv_smem[0]);
  compute_qk_<false, KTraits, KTraits::NUM_MMA_D_CKV,
      KTraits::UPCAST_STRIDE_Q_NOPE, KTraits::UPCAST_STRIDE_CKV>(q, kv, scores);
}

template <typename KTraits>
__device__ __forceinline__ void cached_pe_compute_qk_pair(
    typename KTraits::SharedStorage& storage,uint32_t a,uint32_t b,
    uint32_t (&qpe)[4][4], float (*sa)[8],float (*sb)[8]) {
  cached_pe_compute_pe<true, KTraits>(storage, a, b, qpe, sa, sb);
  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> qn(storage.q_smem_nope);
  smem_t<KTraits::SWIZZLE_MODE_CKV> kna(storage.ckv_smem[a]),knb(storage.ckv_smem[b]);
  cached_pe_qk_pair_fragments<false,KTraits,KTraits::NUM_MMA_D_CKV,
      KTraits::UPCAST_STRIDE_Q_NOPE,KTraits::UPCAST_STRIDE_CKV>(qn,kna,knb,sa,sb);
}


// Two native32-row score tiles share one maximum, denominator/output rescale, then each
// runs native PV in descending tile order. This changes FP32 reduction grouping, not the
// attention equation. It is distinct from the earlier paired-QK-only schedule.
template <typename KTraits>
__device__ __forceinline__ void cached_pe_update_joint_host_mdo(
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
__device__ __forceinline__ void cached_pe_attend_joint_host_tiles(
    const Params& params, typename KTraits::SharedStorage& storage,
    typename KTraits::AttentionVariant& variant, const uint32_t* scratch, uint32_t index_base,
    uint32_t kv_len, uint32_t row, int host_tiles, float (*scores)[8], float (*o)[8], float* m, float* d) {
  static_assert(KTraits::NUM_STAGES == 2 && KTraits::CTA_TILE_KV == 32 && Params::fully_staged);
  typename KTraits::IdType pages[kHostBatchTileRows<KTraits>];
  alignas(16) uint32_t qpe[4][4];
  cached_pe_load_query(params, row, qpe);
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
    // The frozen pair helper reuses Q fragments for the two current tiles. It requires
    // both stages ready, trading the control's first-QK/load overlap for fewer Q loads.
    cp_async::wait_group<0>();
    __syncthreads();
    cached_pe_compute_qk_pair<KTraits>(storage, hi%2, lo%2, qpe, scores, scores+1);
    if (hi >= mask_tile)
      logits_mask_<KTraits>(0, hi*32, 1, kv_len, kv_len, params.num_heads, scores);
    cached_pe_update_joint_host_mdo<KTraits>(&storage, variant, scores, scores+1, o, m, d);
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
    cached_pe_compute_one<KTraits>(storage, qpe, scores);
    logits_mask_<KTraits>(0, 0, 1, kv_len, kv_len, params.num_heads, scores);
    update_mdo_states_<KTraits>(&storage, 0, variant, scores, o, m, d);
    compute_mla_pv<KTraits>(&storage, 0, scores, d, o);
  }
}

template <typename KTraits, typename Params, bool ROW_MERGE>
__device__ __forceinline__ void cached_pe_run_joint_host_item(const Params& params,
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
  if (Params::whole_rows || item < full_items) {
    row = item / params.num_head_tiles;
    head_tile = item % params.num_head_tiles;
    kv_start = 0;
    kv_len = host_batch_row_len(params, row);
    host_tiles = ceil_div(params.state[kHostBatchHostCounts + row], CTA_TILE_KV);
  } else if (item < hbm_end) {
    const uint32_t tile = (item - full_items) / params.num_chunks;
    const uint32_t chunk = (item - full_items) % params.num_chunks;
    row = params.full_rows + tile / params.num_head_tiles;
    head_tile = tile % params.num_head_tiles;
    partial = item - full_items;
    const uint32_t length = host_batch_row_len(params, row);
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
  // A row of only padding (kv_len 0, e.g. a CUDA Graph filler row) writes a zero output from the
  // initial state (lse at its -5e4 start); it must not issue load_q, whose copies join the first
  // KV group.
  if (kv_len > 0) {
    load_q<KTraits>(&storage, params.q_nope + row * params.q_nope_stride_n,
                    params.q_pe + row * params.q_pe_stride_n, params.q_nope_stride_n,
                    params.q_nope_stride_h, params.q_pe_stride_n, params.q_pe_stride_h, 1,
                    head_tile * CTA_TILE_Q, params.num_heads);
    // load_q's copies join the first KV group, as in the native kernel.
    cached_pe_attend_joint_host_tiles<KTraits>(params, storage, variant, scratch,
                                     params.kv_indptr[row] + kv_start, kv_len, row, host_tiles, s_frag,
                                     o_frag, m, d);
  }
  __syncthreads();
  normalize_d_<KTraits>(&storage, 0, o_frag, m, d);
  finalize_m_<KTraits>(variant, m);

  // Each warp reads back output rows other lanes stored to shared memory.
  if (Params::whole_rows || item < full_items) {
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
  if constexpr (ROW_MERGE) {
    if (item >= full_items) {
      // Publish every thread's partial writes before advancing this row's completed item count.
      __threadfence();
      __syncthreads();
      const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
      if (tid == 0) atomicAdd(host_batch_row_completed(params) + row, 1u);
    }
  }
}
template <typename KTraits, typename Params, bool ROW_MERGE>
__device__ __forceinline__ void cached_pe_run_joint_host_items(const Params& params,
                                                     typename KTraits::SharedStorage& storage,
                                                     uint8_t* smem, uint32_t* scratch,
                                                     bool fetcher, uint32_t unique) {
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  // With no Host page the item list has no Host items, and host_batch_host_items was never
  // written this launch, so it must not be read.
  const int lru_items = host_batch_caching(params) && params.stage && unique ? params.num_pools : 0;
  const int items = lru_items +
                    (params.full_rows + (params.num_rows - params.full_rows) * params.num_chunks) *
                        params.num_head_tiles +
                    (unique ? host_batch_host_items(params)[params.num_rows] : 0);
  const int workers = gridDim.y - params.fetch_ctas;
  const int first = fetcher ? items - 1 - (blockIdx.y - workers) : blockIdx.y;
  const int step = fetcher ? -static_cast<int>(params.fetch_ctas) : workers;

  float s_frag[2][8];
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
    cached_pe_run_joint_host_item<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, item - lru_items, s_frag, o_frag,
                                 m, d);
  }
}
template <typename KTraits, typename Params, bool ROW_MERGE>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLACachedPEKernel(
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
  __shared__ uint32_t scratch[3];
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  // The previous launch's last CTA advanced the epoch; ready flags from it no longer match.
  const uint32_t epoch = params.state[kHostBatchEpoch];

  // Not staging and no cache: nothing to prepare; remap is kv_indices (see the launcher) and
  // the host counts stay zero.
  const bool skip_prepare = !params.stage && !host_batch_caching(params);
  uint32_t unique = 0;
  // Grid-uniform: every CTA reads the flag after the same barrier.
  bool gaps = false;
  if (!skip_prepare) {
    if (params.stage) {
#ifdef FLASHINFER_HOST_BATCH_SMALL_PREPARE
      if constexpr (ROW_MERGE) prepare_host_batch_small_rows<KTraits>(params, epoch + 1);
      else prepare_host_batch_rows<KTraits>(params, epoch + 1);
#else
      prepare_host_batch_rows<KTraits>(params, epoch + 1);
#endif
    } else {
      prepare_host_batch_rows_in_place<KTraits>(params, epoch + 1);
    }
    cg::this_grid().sync();
    gaps = params.stage && host_batch_ld_acquire(params.state + kHostBatchGaps) != 0;
#ifdef FLASHINFER_HOST_BATCH_SMALL_PREPARE
    if constexpr (ROW_MERGE) unique = resolve_host_batch_small_slots<KTraits>(params, smem);
    else unique = resolve_host_batch_slots<KTraits>(params, smem);
#else
    unique = resolve_host_batch_slots<KTraits>(params, smem);
#endif
    if (gaps) compact_host_batch_rows<KTraits>(params);
    // In place, victims are chosen now, so they are known when the attention loads the pages.
    if (host_batch_writes_back(params) && unique > 0) {
#pragma unroll 1
      for (uint32_t pool = blockIdx.y; pool < params.num_pools; pool += gridDim.y) {
        run_host_batch_lru<KTraits>(params, smem, pool, epoch, unique,
                                    reinterpret_cast<const uint32_t*>(smem));
        __syncthreads();
      }
    }
    if (blockIdx.y == 0 && tid == 0) params.state[kHostBatchUnique] = unique;
  }
  // unique is the same on every CTA (each runs the same scan), so the branch is grid-uniform and
  // the cooperative barriers below stay matched.
  // Which end of the item list this CTA walks from. This is a partition of the grid and must not
  // depend on `unique`: if the back CTAs also walked from the front they would retrace the same
  // items and contend on the claims (0% Host, 512 rows: 1514 -> 1784 us).
  const bool from_back = blockIdx.y >= gridDim.y - params.fetch_ctas;
  if (unique > 0 || gaps) {
    // Rows resolved or compacted by other CTAs must be complete before any item or fetch reads them, and fetch
    // CTAs clear the dedup slots the resolve step reads.
    cg::this_grid().sync();
    // Claim counts were read by every CTA's scan; reset them for the next launch.
    for (uint32_t row = blockIdx.y * KTraits::NUM_THREADS + tid; row < params.num_rows;
         row += gridDim.y * KTraits::NUM_THREADS) {
      host_batch_row_claims(params)[row] = 0;
    }

  }
  const uint32_t fetch_rows = host_batch_fetch_rows<ROW_MERGE>(params, unique);
  if (tid == 0) {
    scratch[kHostBatchScratchEpoch] = epoch;
    scratch[kHostBatchScratchFetchRows] = fetch_rows;
  }
  __syncthreads();
  // Fetch CTAs take attention items once their copies are issued, so low Host shares and the end
  // of the PCIe stream keep every SM busy; with no Host page there is nothing to fetch.
  if (unique > 0 && from_back && !host_batch_writes_back(params)) fetch_host_batch_pages<KTraits>(params, smem, epoch, unique, fetch_rows);
  cached_pe_run_joint_host_items<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, from_back, unique);
  // Uniform over the grid, so every CTA takes the same branch.
  if (!Params::whole_rows && params.full_rows < params.num_rows) {
    // A small staged merge waits only for the row it reads, overlapping the remaining rows and
    // cache writeback. Every other path retains the full-grid merge barrier.
    if constexpr (!ROW_MERGE) cg::this_grid().sync();
    // 32 features per thread fill the grid only with enough (row tile, head) units; a small batch
    // merged on 32 of 108 SMs at that width and took ~15 us for 8 rows, so it uses 8.
    const uint32_t units = (params.num_rows - params.full_rows) * params.num_head_tiles *
                           KTraits::CTA_TILE_Q;
    if (params.num_rows == 1) {
      merge_host_batch_single_query<KTraits, Params, ROW_MERGE>(params, smem);
    } else if (units >= gridDim.y * (KTraits::NUM_THREADS / (KTraits::HEAD_DIM_CKV / 32))) {
      merge_host_batch_chunks<KTraits, 32, Params, ROW_MERGE>(params);
    } else {
      merge_host_batch_chunks<KTraits, 8, Params, ROW_MERGE>(params);
    }
  }

  // The last CTA to leave resets the shared counters for the next launch: the exit count, the
  // per-row placement cursors (Host counts included), and then the epoch.
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    scratch[kHostBatchScratchClaimed] =
        atomicAdd(params.state + kHostBatchExit, 1u) == gridDim.y - 1;
  }
  __syncthreads();
  if (scratch[kHostBatchScratchClaimed]) {
    if (!skip_prepare) {
      for (uint32_t row = tid; row < params.num_rows; row += KTraits::NUM_THREADS) {
        params.state[kHostBatchHostCounts + row] = 0;
        host_batch_local_cursors(params)[row] = 0;
      }
      if (tid == 0) params.state[kHostBatchGaps] = 0;
    }
    if constexpr (ROW_MERGE) {
      for (uint32_t row = tid; row < params.num_rows; row += KTraits::NUM_THREADS)
        host_batch_row_completed(params)[row] = 0;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      params.state[kHostBatchExit] = 0;
      params.state[kHostBatchEpoch] = epoch + 1;
    }
  }
}



template <typename KTraits, typename Params>
__device__ __forceinline__ void fetch_stream_source_pages(const Params& params, uint8_t* smem,
                                                       uint32_t epoch, uint32_t unique,
                                                       uint32_t rows) {
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

  const uint32_t chunk_vecs = rows * VECS_PER_ROW;
  __shared__ int32_t cache_rows[kHostBatchFetchRows];
  __shared__ typename Params::IdType page_ids[BUFFERS][kHostBatchFetchRows];
  __shared__ uint32_t deferred[kHostBatchMaxDeferred], deferred_count;
  if (tid == 0) deferred_count = 0;
  uint64_t stream_source_policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
               : "=l"(stream_source_policy));
  FAR_EMU_INIT(2);
  FAR_EMU_SYNC();
  auto issue = [&](uint32_t chunk, uint32_t buf) {
    if constexpr (Params::shared_fetch_ids) {
      if (tid < rows && chunk * rows + tid < count) {
        page_ids[buf][tid] = params.unique_pages[chunk * rows + tid];
      }
      // This synchronizes the ordinary ID stores without draining the other buffer's
      // outstanding KV cp.async group. Every thread then reuses one shared ID per row.
      __syncthreads();
    }
#pragma unroll
    for (uint32_t i = 0; i < CHUNK_VECS / KTraits::NUM_THREADS; ++i) {
      const uint32_t v = i * KTraits::NUM_THREADS + tid;
      const uint32_t slot = chunk * rows + v / VECS_PER_ROW;
      if (v < chunk_vecs && slot < count) {
        const auto page = [&]() {
          if constexpr (Params::shared_fetch_ids)
            return page_ids[buf][v / VECS_PER_ROW];
          else return params.unique_pages[slot];
        }();
        const auto* source = reinterpret_cast<const uint4*>(
            params.ckv + static_cast<int64_t>(page) * ROW);
        const uint32_t destination = static_cast<uint32_t>(
            __cvta_generic_to_shared(buffer + buf * CHUNK_VECS + v));
        asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;"
                     :: "r"(destination), "l"(source + v % VECS_PER_ROW),
                        "l"(stream_source_policy));
      }
    }
    FAR_EMU_TILE(2, buf, tid == 0 ? min(rows, count - chunk * rows) : 0, ROW * sizeof(DTypeKV));
    cp_async::commit_group();
  };

  uint32_t chunk = fetch_id;
  if (chunk * rows < count) issue(chunk, 0);
#pragma unroll 1
  for (uint32_t k = 0; chunk * rows < count; ++k, chunk += params.fetch_ctas) {
    const uint32_t next = chunk + params.fetch_ctas;
    if (BUFFERS == 2 && next * rows < count) {
      issue(next, (k + 1) % 2);
      cp_async::wait_group<1>();
    } else {
      cp_async::wait_group<0>();
    }
    FAR_EMU_WAIT(2, BUFFERS == 2 ? k % 2 : 0);
    __syncthreads();
    const uint32_t buf = BUFFERS == 2 ? k % 2 : 0;
    auto* destination = reinterpret_cast<uint4*>(
        params.staging +
        (params.cache_slots + static_cast<int64_t>(chunk) * rows) * ROW);
#pragma unroll
    for (uint32_t i = 0; i < CHUNK_VECS / KTraits::NUM_THREADS; ++i) {
      const uint32_t v = i * KTraits::NUM_THREADS + tid;
      if (params.stage && v < chunk_vecs && chunk * rows + v / VECS_PER_ROW < count) {
        destination[v] = buffer[buf * CHUNK_VECS + v];
      }
    }
    // The resolve step is done, so the dedup entries can be cleared for the next launch.
    if (tid < rows && chunk * rows + tid < count) {
      const auto page = [&]() {
        if constexpr (Params::shared_fetch_ids)
          return page_ids[buf][tid];
        else return params.unique_pages[chunk * rows + tid];
      }();
      params.host_slot[page - params.host_page_start] = 0;
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
    // this chunk were released: one more HBM write per page, and no reread. The victims come from
    // the pools' LRU items, which workers run first (see run_host_batch_items). A staged chunk
    // that lands before its victims are chosen is not waited for: blocking here put the LRU
    // selection (~20 us at 4096 slots) on the PCIe stream's critical path at small batch. Its
    // index is kept and it is copied from staging after the loop. In place there is no staged
    // copy, so the loop waits (nothing in this launch waits on these copies).
    if (host_batch_caching(params)) {
      bool ready = true;
      if (tid < rows && chunk * rows + tid < count) {
        const uint32_t* pool_flag = params.pool_flags + params.slot_pool[chunk * rows + tid];
        ready = host_batch_ld_acquire(pool_flag) == epoch + 1;
      }
      const bool defer = params.stage && !__syncthreads_and(ready) &&
                         deferred_count < kHostBatchMaxDeferred;
      if (defer) {
        __syncthreads();
        if (tid == 0) deferred[deferred_count++] = chunk;
      } else {
        if (tid < rows) {
          const uint32_t slot = chunk * rows + tid;
          int32_t victim = -1;
          if (slot < count) {
            const uint32_t* pool_flag = params.pool_flags + params.slot_pool[slot];
            while (host_batch_ld_acquire(pool_flag) != epoch + 1) __nanosleep(64);
            victim = params.victim_of[slot];
          }
          cache_rows[tid] = victim;
          if constexpr (Params::shared_fetch_ids) {
            if (victim >= 0) {
              const auto page = page_ids[buf][tid];
              const int32_t evicted = params.slot_page[victim];
              if (evicted >= 0) params.resident[evicted - params.host_page_start] = -1;
              params.resident[page - params.host_page_start] = victim;
              params.slot_page[victim] = page;
            }
          } else {
            bind_host_batch_staged_slot(params, slot, victim);
          }
        }
        __syncthreads();
#pragma unroll
        for (uint32_t i = 0; i < CHUNK_VECS / KTraits::NUM_THREADS; ++i) {
          const uint32_t v = i * KTraits::NUM_THREADS + tid;
          const int32_t victim = v < chunk_vecs ? cache_rows[v / VECS_PER_ROW] : -1;
          if (victim >= 0) {
            reinterpret_cast<uint4*>(
                params.staging +
                static_cast<int64_t>(victim) * ROW)[v % VECS_PER_ROW] =
                buffer[buf * CHUNK_VECS + v];
          }
        }
      }
      // The buffer is refilled next.
      __syncthreads();
    }
    if (BUFFERS == 1 && next * rows < count) issue(next, 0);
  }
  // Deferred staged chunks: their victims are chosen by now; copy staging -> cache, HBM to HBM,
  // the whole CTA over each chunk.
  __syncthreads();
#pragma unroll 1
  for (uint32_t d = 0; d < deferred_count; ++d) {
    const uint32_t chunk = deferred[d];
    if (tid < rows) {
      const uint32_t slot = chunk * rows + tid;
      int32_t victim = -1;
      if (slot < count) {
        const uint32_t* pool_flag = params.pool_flags + params.slot_pool[slot];
        while (host_batch_ld_acquire(pool_flag) != epoch + 1) __nanosleep(64);
        victim = params.victim_of[slot];
      }
      cache_rows[tid] = victim;
      bind_host_batch_staged_slot(params, slot, victim);
    }
    __syncthreads();
    const auto* source = reinterpret_cast<const uint4*>(
        params.staging + (params.cache_slots + static_cast<int64_t>(chunk) * rows) * ROW);
#pragma unroll 4
    for (uint32_t v = tid; v < chunk_vecs; v += KTraits::NUM_THREADS) {
      const int32_t victim = cache_rows[v / VECS_PER_ROW];
      if (victim >= 0) {
        reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * ROW)[v % VECS_PER_ROW] =
            source[v];
      }
    }
    __syncthreads();
  }
  // Overflow pages were never staged; clear their dedup entries too.
#pragma unroll 1
  for (uint32_t slot = count + fetch_id * KTraits::NUM_THREADS + tid; slot < unique;
       slot += params.fetch_ctas * KTraits::NUM_THREADS) {
    params.host_slot[params.unique_pages[slot] - params.host_page_start] = 0;
  }
}

template <typename Q, typename KV, typename O, typename Id>
struct StreamSourceTraits : JointQKTraits<Q, KV, O, Id> {};
template <typename Params>
struct StreamSourceHostParams : JointQKHostParams<Params> {};

// Two different KV tiles share each Q fragment. Each score accumulator keeps the
// native PE-then-CKV MMA order; this performs no duplicate QK work.
template <bool INIT, typename KTraits, uint32_t D, uint32_t Q_STRIDE,
          uint32_t K_STRIDE, SwizzleMode Q_SWIZZLE, SwizzleMode K_SWIZZLE>
__device__ __forceinline__ void stream_source_qk_pair_fragments(
    smem_t<Q_SWIZZLE> q, smem_t<K_SWIZZLE> ka, smem_t<K_SWIZZLE> kb,
    float (*sa)[8], float (*sb)[8]) {
  static_assert(KTraits::QK_SHARD && KTraits::CTA_TILE_KV == 32);
  const uint32_t lane=threadIdx.x, group=threadIdx.z, warp=threadIdx.y;
  alignas(16) uint32_t qf[4],kf[4];
  uint32_t qo=q.template get_permuted_offset<Q_STRIDE>(
      warp*16+lane%16,lane/16);
  uint32_t ko=ka.template get_permuted_offset<K_STRIDE>(
      group*16+8*(lane/16)+lane%8,(lane%16)/8);
#pragma unroll 4
  for (uint32_t dim=0;dim<D;++dim) {
    q.ldmatrix_m8n8x4(qo,qf);
    ka.ldmatrix_m8n8x4(ko,kf);
    if (INIT && dim==0)
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ,MMAMode::kInit>(sa[0],qf,kf);
    else
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sa[0],qf,kf);
    kb.ldmatrix_m8n8x4(ko,kf);
    if (INIT && dim==0)
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ,MMAMode::kInit>(sb[0],qf,kf);
    else
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sb[0],qf,kf);
    qo=q.template advance_offset_by_column<2>(qo,dim);
    ko=ka.template advance_offset_by_column<2>(ko,dim);
  }
}


// Each lane owns four packed words per 16-feature QPE MMA fragment. The query
// head/feature map is the same map produced by native ldmatrix.x4.
template <typename Params>
__device__ __forceinline__ void stream_source_load_query(const Params& params, uint32_t row,
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

template <bool PAIR, typename KTraits>
__device__ __forceinline__ void stream_source_compute_pe(
    typename KTraits::SharedStorage& storage, uint32_t a, uint32_t b,
    uint32_t (&qpe)[4][4], float (*sa)[8], float (*sb)[8]) {
  const uint32_t lane = threadIdx.x, group = threadIdx.z;
  smem_t<KTraits::SWIZZLE_MODE_KPE> ka(storage.kpe_p_smem[a]), kb(storage.kpe_p_smem[b]);
  alignas(16) uint32_t kf[4];
#pragma unroll
  for (uint32_t dim = 0; dim < 4; ++dim) {
    const uint32_t offset = ka.template get_permuted_offset<KTraits::UPCAST_STRIDE_KPE>(
        group * 16 + 8 * (lane / 16) + lane % 8, 2 * dim + (lane % 16) / 8);
    ka.ldmatrix_m8n8x4(offset, kf);
    if (dim == 0)
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(sa[0], qpe[dim], kf);
    else
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sa[0], qpe[dim], kf);
    if constexpr (PAIR) {
      kb.ldmatrix_m8n8x4(offset, kf);
      if (dim == 0)
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(sb[0], qpe[dim], kf);
      else
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sb[0], qpe[dim], kf);
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void stream_source_compute_one(
    typename KTraits::SharedStorage& storage, uint32_t (&qpe)[4][4], float (*scores)[8]) {
  stream_source_compute_pe<false, KTraits>(storage, 0, 0, qpe, scores, scores);
  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> q(storage.q_smem_nope);
  smem_t<KTraits::SWIZZLE_MODE_CKV> kv(storage.ckv_smem[0]);
  compute_qk_<false, KTraits, KTraits::NUM_MMA_D_CKV,
      KTraits::UPCAST_STRIDE_Q_NOPE, KTraits::UPCAST_STRIDE_CKV>(q, kv, scores);
}

template <typename KTraits>
__device__ __forceinline__ void stream_source_compute_qk_pair(
    typename KTraits::SharedStorage& storage,uint32_t a,uint32_t b,
    uint32_t (&qpe)[4][4], float (*sa)[8],float (*sb)[8]) {
  stream_source_compute_pe<true, KTraits>(storage, a, b, qpe, sa, sb);
  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> qn(storage.q_smem_nope);
  smem_t<KTraits::SWIZZLE_MODE_CKV> kna(storage.ckv_smem[a]),knb(storage.ckv_smem[b]);
  stream_source_qk_pair_fragments<false,KTraits,KTraits::NUM_MMA_D_CKV,
      KTraits::UPCAST_STRIDE_Q_NOPE,KTraits::UPCAST_STRIDE_CKV>(qn,kna,knb,sa,sb);
}


// Two native32-row score tiles share one maximum, denominator/output rescale, then each
// runs native PV in descending tile order. This changes FP32 reduction grouping, not the
// attention equation. It is distinct from the earlier paired-QK-only schedule.
template <typename KTraits>
__device__ __forceinline__ void stream_source_update_joint_host_mdo(
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
__device__ __forceinline__ void stream_source_attend_joint_host_tiles(
    const Params& params, typename KTraits::SharedStorage& storage,
    typename KTraits::AttentionVariant& variant, const uint32_t* scratch, uint32_t index_base,
    uint32_t kv_len, uint32_t row, int host_tiles, float (*scores)[8], float (*o)[8], float* m, float* d) {
  static_assert(KTraits::NUM_STAGES == 2 && KTraits::CTA_TILE_KV == 32 && Params::fully_staged);
  typename KTraits::IdType pages[kHostBatchTileRows<KTraits>];
  alignas(16) uint32_t qpe[4][4];
  stream_source_load_query(params, row, qpe);
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
    // The frozen pair helper reuses Q fragments for the two current tiles. It requires
    // both stages ready, trading the control's first-QK/load overlap for fewer Q loads.
    cp_async::wait_group<0>();
    __syncthreads();
    stream_source_compute_qk_pair<KTraits>(storage, hi%2, lo%2, qpe, scores, scores+1);
    if (hi >= mask_tile)
      logits_mask_<KTraits>(0, hi*32, 1, kv_len, kv_len, params.num_heads, scores);
    stream_source_update_joint_host_mdo<KTraits>(&storage, variant, scores, scores+1, o, m, d);
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
    stream_source_compute_one<KTraits>(storage, qpe, scores);
    logits_mask_<KTraits>(0, 0, 1, kv_len, kv_len, params.num_heads, scores);
    update_mdo_states_<KTraits>(&storage, 0, variant, scores, o, m, d);
    compute_mla_pv<KTraits>(&storage, 0, scores, d, o);
  }
}

template <typename KTraits, typename Params, bool ROW_MERGE>
__device__ __forceinline__ void stream_source_run_joint_host_item(const Params& params,
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
  if (Params::whole_rows || item < full_items) {
    row = item / params.num_head_tiles;
    head_tile = item % params.num_head_tiles;
    kv_start = 0;
    kv_len = host_batch_row_len(params, row);
    host_tiles = ceil_div(params.state[kHostBatchHostCounts + row], CTA_TILE_KV);
  } else if (item < hbm_end) {
    const uint32_t tile = (item - full_items) / params.num_chunks;
    const uint32_t chunk = (item - full_items) % params.num_chunks;
    row = params.full_rows + tile / params.num_head_tiles;
    head_tile = tile % params.num_head_tiles;
    partial = item - full_items;
    const uint32_t length = host_batch_row_len(params, row);
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
  // A row of only padding (kv_len 0, e.g. a CUDA Graph filler row) writes a zero output from the
  // initial state (lse at its -5e4 start); it must not issue load_q, whose copies join the first
  // KV group.
  if (kv_len > 0) {
    load_q<KTraits>(&storage, params.q_nope + row * params.q_nope_stride_n,
                    params.q_pe + row * params.q_pe_stride_n, params.q_nope_stride_n,
                    params.q_nope_stride_h, params.q_pe_stride_n, params.q_pe_stride_h, 1,
                    head_tile * CTA_TILE_Q, params.num_heads);
    // load_q's copies join the first KV group, as in the native kernel.
    stream_source_attend_joint_host_tiles<KTraits>(params, storage, variant, scratch,
                                     params.kv_indptr[row] + kv_start, kv_len, row, host_tiles, s_frag,
                                     o_frag, m, d);
  }
  __syncthreads();
  normalize_d_<KTraits>(&storage, 0, o_frag, m, d);
  finalize_m_<KTraits>(variant, m);

  // Each warp reads back output rows other lanes stored to shared memory.
  if (Params::whole_rows || item < full_items) {
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
  if constexpr (ROW_MERGE) {
    if (item >= full_items) {
      // Publish every thread's partial writes before advancing this row's completed item count.
      __threadfence();
      __syncthreads();
      const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
      if (tid == 0) atomicAdd(host_batch_row_completed(params) + row, 1u);
    }
  }
}
template <typename KTraits, typename Params, bool ROW_MERGE>
__device__ __forceinline__ void stream_source_run_joint_host_items(const Params& params,
                                                     typename KTraits::SharedStorage& storage,
                                                     uint8_t* smem, uint32_t* scratch,
                                                     bool fetcher, uint32_t unique) {
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  // With no Host page the item list has no Host items, and host_batch_host_items was never
  // written this launch, so it must not be read.
  const int lru_items = host_batch_caching(params) && params.stage && unique ? params.num_pools : 0;
  const int items = lru_items +
                    (params.full_rows + (params.num_rows - params.full_rows) * params.num_chunks) *
                        params.num_head_tiles +
                    (unique ? host_batch_host_items(params)[params.num_rows] : 0);
  const int workers = gridDim.y - params.fetch_ctas;
  const int first = fetcher ? items - 1 - (blockIdx.y - workers) : blockIdx.y;
  const int step = fetcher ? -static_cast<int>(params.fetch_ctas) : workers;

  float s_frag[2][8];
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
    stream_source_run_joint_host_item<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, item - lru_items, s_frag, o_frag,
                                 m, d);
  }
}
template <typename KTraits, typename Params, bool ROW_MERGE>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLAStreamSourceKernel(
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
  __shared__ uint32_t scratch[3];
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  // The previous launch's last CTA advanced the epoch; ready flags from it no longer match.
  const uint32_t epoch = params.state[kHostBatchEpoch];

  // Not staging and no cache: nothing to prepare; remap is kv_indices (see the launcher) and
  // the host counts stay zero.
  const bool skip_prepare = !params.stage && !host_batch_caching(params);
  uint32_t unique = 0;
  // Grid-uniform: every CTA reads the flag after the same barrier.
  bool gaps = false;
  if (!skip_prepare) {
    if (params.stage) {
#ifdef FLASHINFER_HOST_BATCH_SMALL_PREPARE
      if constexpr (ROW_MERGE) prepare_host_batch_small_rows<KTraits>(params, epoch + 1);
      else prepare_host_batch_rows<KTraits>(params, epoch + 1);
#else
      prepare_host_batch_rows<KTraits>(params, epoch + 1);
#endif
    } else {
      prepare_host_batch_rows_in_place<KTraits>(params, epoch + 1);
    }
    cg::this_grid().sync();
    gaps = params.stage && host_batch_ld_acquire(params.state + kHostBatchGaps) != 0;
#ifdef FLASHINFER_HOST_BATCH_SMALL_PREPARE
    if constexpr (ROW_MERGE) unique = resolve_host_batch_small_slots<KTraits>(params, smem);
    else unique = resolve_host_batch_slots<KTraits>(params, smem);
#else
    unique = resolve_host_batch_slots<KTraits>(params, smem);
#endif
    if (gaps) compact_host_batch_rows<KTraits>(params);
    // In place, victims are chosen now, so they are known when the attention loads the pages.
    if (host_batch_writes_back(params) && unique > 0) {
#pragma unroll 1
      for (uint32_t pool = blockIdx.y; pool < params.num_pools; pool += gridDim.y) {
        run_host_batch_lru<KTraits>(params, smem, pool, epoch, unique,
                                    reinterpret_cast<const uint32_t*>(smem));
        __syncthreads();
      }
    }
    if (blockIdx.y == 0 && tid == 0) params.state[kHostBatchUnique] = unique;
  }
  // unique is the same on every CTA (each runs the same scan), so the branch is grid-uniform and
  // the cooperative barriers below stay matched.
  // Which end of the item list this CTA walks from. This is a partition of the grid and must not
  // depend on `unique`: if the back CTAs also walked from the front they would retrace the same
  // items and contend on the claims (0% Host, 512 rows: 1514 -> 1784 us).
  const bool from_back = blockIdx.y >= gridDim.y - params.fetch_ctas;
  if (unique > 0 || gaps) {
    // Rows resolved or compacted by other CTAs must be complete before any item or fetch reads them, and fetch
    // CTAs clear the dedup slots the resolve step reads.
    cg::this_grid().sync();
    // Claim counts were read by every CTA's scan; reset them for the next launch.
    for (uint32_t row = blockIdx.y * KTraits::NUM_THREADS + tid; row < params.num_rows;
         row += gridDim.y * KTraits::NUM_THREADS) {
      host_batch_row_claims(params)[row] = 0;
    }

  }
  const uint32_t fetch_rows = host_batch_fetch_rows<ROW_MERGE>(params, unique);
  if (tid == 0) {
    scratch[kHostBatchScratchEpoch] = epoch;
    scratch[kHostBatchScratchFetchRows] = fetch_rows;
  }
  __syncthreads();
  // Fetch CTAs take attention items once their copies are issued, so low Host shares and the end
  // of the PCIe stream keep every SM busy; with no Host page there is nothing to fetch.
  if (unique > 0 && from_back && !host_batch_writes_back(params)) fetch_stream_source_pages<KTraits>(params, smem, epoch, unique, fetch_rows);
  stream_source_run_joint_host_items<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, from_back, unique);
  // Uniform over the grid, so every CTA takes the same branch.
  if (!Params::whole_rows && params.full_rows < params.num_rows) {
    // A small staged merge waits only for the row it reads, overlapping the remaining rows and
    // cache writeback. Every other path retains the full-grid merge barrier.
    if constexpr (!ROW_MERGE) cg::this_grid().sync();
    // 32 features per thread fill the grid only with enough (row tile, head) units; a small batch
    // merged on 32 of 108 SMs at that width and took ~15 us for 8 rows, so it uses 8.
    const uint32_t units = (params.num_rows - params.full_rows) * params.num_head_tiles *
                           KTraits::CTA_TILE_Q;
    if (params.num_rows == 1) {
      merge_host_batch_single_query<KTraits, Params, ROW_MERGE>(params, smem);
    } else if (units >= gridDim.y * (KTraits::NUM_THREADS / (KTraits::HEAD_DIM_CKV / 32))) {
      merge_host_batch_chunks<KTraits, 32, Params, ROW_MERGE>(params);
    } else {
      merge_host_batch_chunks<KTraits, 8, Params, ROW_MERGE>(params);
    }
  }

  // The last CTA to leave resets the shared counters for the next launch: the exit count, the
  // per-row placement cursors (Host counts included), and then the epoch.
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    scratch[kHostBatchScratchClaimed] =
        atomicAdd(params.state + kHostBatchExit, 1u) == gridDim.y - 1;
  }
  __syncthreads();
  if (scratch[kHostBatchScratchClaimed]) {
    if (!skip_prepare) {
      for (uint32_t row = tid; row < params.num_rows; row += KTraits::NUM_THREADS) {
        params.state[kHostBatchHostCounts + row] = 0;
        host_batch_local_cursors(params)[row] = 0;
      }
      if (tid == 0) params.state[kHostBatchGaps] = 0;
    }
    if constexpr (ROW_MERGE) {
      for (uint32_t row = tid; row < params.num_rows; row += KTraits::NUM_THREADS)
        host_batch_row_completed(params)[row] = 0;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      params.state[kHostBatchExit] = 0;
      params.state[kHostBatchEpoch] = epoch + 1;
    }
  }
}


template <typename KTraits>
__device__ __forceinline__ void pv_feature_pv(typename KTraits::SharedStorage* smem_storage,
                                               const uint32_t stage_idx,
                                               typename KTraits::DTypeQKAccum (*s_frag)[8],
                                               typename KTraits::DTypeQKAccum* d,
                                               float (*o_frag)[8]) {
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  static_assert(KTraits::QK_SHARD && KTraits::CTA_TILE_KV == 32);
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

    // Keep both probabilities resident, then finish each O feature with both KV
    // halves. Per-output MMA order and the native P cast/rowsum/publish are kept.
    uint32_t p_frag[2][4];
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
      p_smem.ldmatrix_m8n8x4(p_smem_offset_r, p_frag[mma_kv]);
      p_smem_offset_r = p_smem.template advance_offset_by_column<2>(p_smem_offset_r, mma_kv);
    }
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 2; ++mma_d) {
      uint32_t v_frag[4];
      ckv_smem.ldmatrix_m8n8x4_trans(ckv_smem_offset_r, v_frag);
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(o_frag[mma_d], p_frag[0], v_frag);
      const uint32_t second = ckv_smem.template advance_offset_by_row<16, UPCAST_STRIDE_CKV>(ckv_smem_offset_r);
      ckv_smem.ldmatrix_m8n8x4_trans(second, v_frag);
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(o_frag[mma_d], p_frag[1], v_frag);
      ckv_smem_offset_r = ckv_smem.template advance_offset_by_column<2>(ckv_smem_offset_r, mma_d);
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

template <typename Q, typename KV, typename O, typename Id>
struct PVFeatureTraits : JointQKTraits<Q, KV, O, Id> {};
template <typename Params>
struct PVFeatureHostParams : JointQKHostParams<Params> {};

// Two different KV tiles share each Q fragment. Each score accumulator keeps the
// native PE-then-CKV MMA order; this performs no duplicate QK work.
template <bool INIT, typename KTraits, uint32_t D, uint32_t Q_STRIDE,
          uint32_t K_STRIDE, SwizzleMode Q_SWIZZLE, SwizzleMode K_SWIZZLE>
__device__ __forceinline__ void pv_feature_qk_pair_fragments(
    smem_t<Q_SWIZZLE> q, smem_t<K_SWIZZLE> ka, smem_t<K_SWIZZLE> kb,
    float (*sa)[8], float (*sb)[8]) {
  static_assert(KTraits::QK_SHARD && KTraits::CTA_TILE_KV == 32);
  const uint32_t lane=threadIdx.x, group=threadIdx.z, warp=threadIdx.y;
  alignas(16) uint32_t qf[4],kf[4];
  uint32_t qo=q.template get_permuted_offset<Q_STRIDE>(
      warp*16+lane%16,lane/16);
  uint32_t ko=ka.template get_permuted_offset<K_STRIDE>(
      group*16+8*(lane/16)+lane%8,(lane%16)/8);
#pragma unroll 4
  for (uint32_t dim=0;dim<D;++dim) {
    q.ldmatrix_m8n8x4(qo,qf);
    ka.ldmatrix_m8n8x4(ko,kf);
    if (INIT && dim==0)
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ,MMAMode::kInit>(sa[0],qf,kf);
    else
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sa[0],qf,kf);
    kb.ldmatrix_m8n8x4(ko,kf);
    if (INIT && dim==0)
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ,MMAMode::kInit>(sb[0],qf,kf);
    else
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sb[0],qf,kf);
    qo=q.template advance_offset_by_column<2>(qo,dim);
    ko=ka.template advance_offset_by_column<2>(ko,dim);
  }
}


// Each lane owns four packed words per 16-feature QPE MMA fragment. The query
// head/feature map is the same map produced by native ldmatrix.x4.
template <typename Params>
__device__ __forceinline__ void pv_feature_load_query(const Params& params, uint32_t row,
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

template <bool PAIR, typename KTraits>
__device__ __forceinline__ void pv_feature_compute_pe(
    typename KTraits::SharedStorage& storage, uint32_t a, uint32_t b,
    uint32_t (&qpe)[4][4], float (*sa)[8], float (*sb)[8]) {
  const uint32_t lane = threadIdx.x, group = threadIdx.z;
  smem_t<KTraits::SWIZZLE_MODE_KPE> ka(storage.kpe_p_smem[a]), kb(storage.kpe_p_smem[b]);
  alignas(16) uint32_t kf[4];
#pragma unroll
  for (uint32_t dim = 0; dim < 4; ++dim) {
    const uint32_t offset = ka.template get_permuted_offset<KTraits::UPCAST_STRIDE_KPE>(
        group * 16 + 8 * (lane / 16) + lane % 8, 2 * dim + (lane % 16) / 8);
    ka.ldmatrix_m8n8x4(offset, kf);
    if (dim == 0)
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(sa[0], qpe[dim], kf);
    else
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sa[0], qpe[dim], kf);
    if constexpr (PAIR) {
      kb.ldmatrix_m8n8x4(offset, kf);
      if (dim == 0)
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(sb[0], qpe[dim], kf);
      else
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sb[0], qpe[dim], kf);
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void pv_feature_compute_one(
    typename KTraits::SharedStorage& storage, uint32_t (&qpe)[4][4], float (*scores)[8]) {
  pv_feature_compute_pe<false, KTraits>(storage, 0, 0, qpe, scores, scores);
  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> q(storage.q_smem_nope);
  smem_t<KTraits::SWIZZLE_MODE_CKV> kv(storage.ckv_smem[0]);
  compute_qk_<false, KTraits, KTraits::NUM_MMA_D_CKV,
      KTraits::UPCAST_STRIDE_Q_NOPE, KTraits::UPCAST_STRIDE_CKV>(q, kv, scores);
}

template <typename KTraits>
__device__ __forceinline__ void pv_feature_compute_qk_pair(
    typename KTraits::SharedStorage& storage,uint32_t a,uint32_t b,
    uint32_t (&qpe)[4][4], float (*sa)[8],float (*sb)[8]) {
  pv_feature_compute_pe<true, KTraits>(storage, a, b, qpe, sa, sb);
  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> qn(storage.q_smem_nope);
  smem_t<KTraits::SWIZZLE_MODE_CKV> kna(storage.ckv_smem[a]),knb(storage.ckv_smem[b]);
  pv_feature_qk_pair_fragments<false,KTraits,KTraits::NUM_MMA_D_CKV,
      KTraits::UPCAST_STRIDE_Q_NOPE,KTraits::UPCAST_STRIDE_CKV>(qn,kna,knb,sa,sb);
}


// Two native32-row score tiles share one maximum, denominator/output rescale, then each
// runs native PV in descending tile order. This changes FP32 reduction grouping, not the
// attention equation. It is distinct from the earlier paired-QK-only schedule.
template <typename KTraits>
__device__ __forceinline__ void pv_feature_update_joint_host_mdo(
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
__device__ __forceinline__ void pv_feature_attend_joint_host_tiles(
    const Params& params, typename KTraits::SharedStorage& storage,
    typename KTraits::AttentionVariant& variant, const uint32_t* scratch, uint32_t index_base,
    uint32_t kv_len, uint32_t row, int host_tiles, float (*scores)[8], float (*o)[8], float* m, float* d) {
  static_assert(KTraits::NUM_STAGES == 2 && KTraits::CTA_TILE_KV == 32 && Params::fully_staged);
  typename KTraits::IdType pages[kHostBatchTileRows<KTraits>];
  alignas(16) uint32_t qpe[4][4];
  pv_feature_load_query(params, row, qpe);
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
    // The frozen pair helper reuses Q fragments for the two current tiles. It requires
    // both stages ready, trading the control's first-QK/load overlap for fewer Q loads.
    cp_async::wait_group<0>();
    __syncthreads();
    pv_feature_compute_qk_pair<KTraits>(storage, hi%2, lo%2, qpe, scores, scores+1);
    if (hi >= mask_tile)
      logits_mask_<KTraits>(0, hi*32, 1, kv_len, kv_len, params.num_heads, scores);
    pv_feature_update_joint_host_mdo<KTraits>(&storage, variant, scores, scores+1, o, m, d);
    pv_feature_pv<KTraits>(&storage, hi%2, scores, d, o);
    if (next_hi >= 0) {
      if (next_hi < host_tiles)
        wait_host_batch_tile<KTraits>(params, scratch, index_base, kv_len, next_hi*32);
      __syncthreads();
      load_host_batch_kv<KTraits>(&storage, params, pages, kv_len, next_hi*32, next_hi%2, index_base);
      cp_async::commit_group();
    }
    // Refill the first released stage before PV of the second tile, preserving overlap.
    if (next_lo >= 0) read_host_batch_pages<KTraits>(params, index_base, kv_len, next_lo*32, pages);
    pv_feature_pv<KTraits>(&storage, lo%2, scores+1, d, o);
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
    pv_feature_compute_one<KTraits>(storage, qpe, scores);
    logits_mask_<KTraits>(0, 0, 1, kv_len, kv_len, params.num_heads, scores);
    update_mdo_states_<KTraits>(&storage, 0, variant, scores, o, m, d);
    pv_feature_pv<KTraits>(&storage, 0, scores, d, o);
  }
}

template <typename KTraits, typename Params, bool ROW_MERGE>
__device__ __forceinline__ void pv_feature_run_joint_host_item(const Params& params,
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
  if (Params::whole_rows || item < full_items) {
    row = item / params.num_head_tiles;
    head_tile = item % params.num_head_tiles;
    kv_start = 0;
    kv_len = host_batch_row_len(params, row);
    host_tiles = ceil_div(params.state[kHostBatchHostCounts + row], CTA_TILE_KV);
  } else if (item < hbm_end) {
    const uint32_t tile = (item - full_items) / params.num_chunks;
    const uint32_t chunk = (item - full_items) % params.num_chunks;
    row = params.full_rows + tile / params.num_head_tiles;
    head_tile = tile % params.num_head_tiles;
    partial = item - full_items;
    const uint32_t length = host_batch_row_len(params, row);
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
  // A row of only padding (kv_len 0, e.g. a CUDA Graph filler row) writes a zero output from the
  // initial state (lse at its -5e4 start); it must not issue load_q, whose copies join the first
  // KV group.
  if (kv_len > 0) {
    load_q<KTraits>(&storage, params.q_nope + row * params.q_nope_stride_n,
                    params.q_pe + row * params.q_pe_stride_n, params.q_nope_stride_n,
                    params.q_nope_stride_h, params.q_pe_stride_n, params.q_pe_stride_h, 1,
                    head_tile * CTA_TILE_Q, params.num_heads);
    // load_q's copies join the first KV group, as in the native kernel.
    pv_feature_attend_joint_host_tiles<KTraits>(params, storage, variant, scratch,
                                     params.kv_indptr[row] + kv_start, kv_len, row, host_tiles, s_frag,
                                     o_frag, m, d);
  }
  __syncthreads();
  normalize_d_<KTraits>(&storage, 0, o_frag, m, d);
  finalize_m_<KTraits>(variant, m);

  // Each warp reads back output rows other lanes stored to shared memory.
  if (Params::whole_rows || item < full_items) {
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
  if constexpr (ROW_MERGE) {
    if (item >= full_items) {
      // Publish every thread's partial writes before advancing this row's completed item count.
      __threadfence();
      __syncthreads();
      const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
      if (tid == 0) atomicAdd(host_batch_row_completed(params) + row, 1u);
    }
  }
}
template <typename KTraits, typename Params, bool ROW_MERGE>
__device__ __forceinline__ void pv_feature_run_joint_host_items(const Params& params,
                                                     typename KTraits::SharedStorage& storage,
                                                     uint8_t* smem, uint32_t* scratch,
                                                     bool fetcher, uint32_t unique) {
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  // With no Host page the item list has no Host items, and host_batch_host_items was never
  // written this launch, so it must not be read.
  const int lru_items = host_batch_caching(params) && params.stage && unique ? params.num_pools : 0;
  const int items = lru_items +
                    (params.full_rows + (params.num_rows - params.full_rows) * params.num_chunks) *
                        params.num_head_tiles +
                    (unique ? host_batch_host_items(params)[params.num_rows] : 0);
  const int workers = gridDim.y - params.fetch_ctas;
  const int first = fetcher ? items - 1 - (blockIdx.y - workers) : blockIdx.y;
  const int step = fetcher ? -static_cast<int>(params.fetch_ctas) : workers;

  float s_frag[2][8];
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
    pv_feature_run_joint_host_item<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, item - lru_items, s_frag, o_frag,
                                 m, d);
  }
}
template <typename KTraits, typename Params, bool ROW_MERGE>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLAPVFeatureKernel(
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
  __shared__ uint32_t scratch[3];
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  // The previous launch's last CTA advanced the epoch; ready flags from it no longer match.
  const uint32_t epoch = params.state[kHostBatchEpoch];

  // Not staging and no cache: nothing to prepare; remap is kv_indices (see the launcher) and
  // the host counts stay zero.
  const bool skip_prepare = !params.stage && !host_batch_caching(params);
  uint32_t unique = 0;
  // Grid-uniform: every CTA reads the flag after the same barrier.
  bool gaps = false;
  if (!skip_prepare) {
    if (params.stage) {
#ifdef FLASHINFER_HOST_BATCH_SMALL_PREPARE
      if constexpr (ROW_MERGE) prepare_host_batch_small_rows<KTraits>(params, epoch + 1);
      else prepare_host_batch_rows<KTraits>(params, epoch + 1);
#else
      prepare_host_batch_rows<KTraits>(params, epoch + 1);
#endif
    } else {
      prepare_host_batch_rows_in_place<KTraits>(params, epoch + 1);
    }
    cg::this_grid().sync();
    gaps = params.stage && host_batch_ld_acquire(params.state + kHostBatchGaps) != 0;
#ifdef FLASHINFER_HOST_BATCH_SMALL_PREPARE
    if constexpr (ROW_MERGE) unique = resolve_host_batch_small_slots<KTraits>(params, smem);
    else unique = resolve_host_batch_slots<KTraits>(params, smem);
#else
    unique = resolve_host_batch_slots<KTraits>(params, smem);
#endif
    if (gaps) compact_host_batch_rows<KTraits>(params);
    // In place, victims are chosen now, so they are known when the attention loads the pages.
    if (host_batch_writes_back(params) && unique > 0) {
#pragma unroll 1
      for (uint32_t pool = blockIdx.y; pool < params.num_pools; pool += gridDim.y) {
        run_host_batch_lru<KTraits>(params, smem, pool, epoch, unique,
                                    reinterpret_cast<const uint32_t*>(smem));
        __syncthreads();
      }
    }
    if (blockIdx.y == 0 && tid == 0) params.state[kHostBatchUnique] = unique;
  }
  // unique is the same on every CTA (each runs the same scan), so the branch is grid-uniform and
  // the cooperative barriers below stay matched.
  // Which end of the item list this CTA walks from. This is a partition of the grid and must not
  // depend on `unique`: if the back CTAs also walked from the front they would retrace the same
  // items and contend on the claims (0% Host, 512 rows: 1514 -> 1784 us).
  const bool from_back = blockIdx.y >= gridDim.y - params.fetch_ctas;
  if (unique > 0 || gaps) {
    // Rows resolved or compacted by other CTAs must be complete before any item or fetch reads them, and fetch
    // CTAs clear the dedup slots the resolve step reads.
    cg::this_grid().sync();
    // Claim counts were read by every CTA's scan; reset them for the next launch.
    for (uint32_t row = blockIdx.y * KTraits::NUM_THREADS + tid; row < params.num_rows;
         row += gridDim.y * KTraits::NUM_THREADS) {
      host_batch_row_claims(params)[row] = 0;
    }

  }
  const uint32_t fetch_rows = host_batch_fetch_rows<ROW_MERGE>(params, unique);
  if (tid == 0) {
    scratch[kHostBatchScratchEpoch] = epoch;
    scratch[kHostBatchScratchFetchRows] = fetch_rows;
  }
  __syncthreads();
  // Fetch CTAs take attention items once their copies are issued, so low Host shares and the end
  // of the PCIe stream keep every SM busy; with no Host page there is nothing to fetch.
  if (unique > 0 && from_back && !host_batch_writes_back(params)) fetch_stream_source_pages<KTraits>(params, smem, epoch, unique, fetch_rows);
  pv_feature_run_joint_host_items<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, from_back, unique);
  // Uniform over the grid, so every CTA takes the same branch.
  if (!Params::whole_rows && params.full_rows < params.num_rows) {
    // A small staged merge waits only for the row it reads, overlapping the remaining rows and
    // cache writeback. Every other path retains the full-grid merge barrier.
    if constexpr (!ROW_MERGE) cg::this_grid().sync();
    // 32 features per thread fill the grid only with enough (row tile, head) units; a small batch
    // merged on 32 of 108 SMs at that width and took ~15 us for 8 rows, so it uses 8.
    const uint32_t units = (params.num_rows - params.full_rows) * params.num_head_tiles *
                           KTraits::CTA_TILE_Q;
    if (params.num_rows == 1) {
      merge_host_batch_single_query<KTraits, Params, ROW_MERGE>(params, smem);
    } else if (units >= gridDim.y * (KTraits::NUM_THREADS / (KTraits::HEAD_DIM_CKV / 32))) {
      merge_host_batch_chunks<KTraits, 32, Params, ROW_MERGE>(params);
    } else {
      merge_host_batch_chunks<KTraits, 8, Params, ROW_MERGE>(params);
    }
  }

  // The last CTA to leave resets the shared counters for the next launch: the exit count, the
  // per-row placement cursors (Host counts included), and then the epoch.
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    scratch[kHostBatchScratchClaimed] =
        atomicAdd(params.state + kHostBatchExit, 1u) == gridDim.y - 1;
  }
  __syncthreads();
  if (scratch[kHostBatchScratchClaimed]) {
    if (!skip_prepare) {
      for (uint32_t row = tid; row < params.num_rows; row += KTraits::NUM_THREADS) {
        params.state[kHostBatchHostCounts + row] = 0;
        host_batch_local_cursors(params)[row] = 0;
      }
      if (tid == 0) params.state[kHostBatchGaps] = 0;
    }
    if constexpr (ROW_MERGE) {
      for (uint32_t row = tid; row < params.num_rows; row += KTraits::NUM_THREADS)
        host_batch_row_completed(params)[row] = 0;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      params.state[kHostBatchExit] = 0;
      params.state[kHostBatchEpoch] = epoch + 1;
    }
  }
}


#ifndef FLASHINFER_FAR_EMU
// Fast normal-build BF16 inplace math; exact native owner cachewriteback.
template <typename KTraits>
__device__ __forceinline__ void direct_pv_pv(typename KTraits::SharedStorage* smem_storage,
                                               const uint32_t stage_idx,
                                               typename KTraits::DTypeQKAccum (*s_frag)[8],
                                               typename KTraits::DTypeQKAccum* d,
                                               float (*o_frag)[8]) {
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  static_assert(KTraits::QK_SHARD && KTraits::CTA_TILE_KV == 32);
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

    // Keep both probabilities resident, then finish each O feature with both KV
    // halves. Per-output MMA order and the native P cast/rowsum/publish are kept.
    uint32_t p_frag[2][4];
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
      p_smem.ldmatrix_m8n8x4(p_smem_offset_r, p_frag[mma_kv]);
      p_smem_offset_r = p_smem.template advance_offset_by_column<2>(p_smem_offset_r, mma_kv);
    }
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 2; ++mma_d) {
      uint32_t v_frag[4];
      ckv_smem.ldmatrix_m8n8x4_trans(ckv_smem_offset_r, v_frag);
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(o_frag[mma_d], p_frag[0], v_frag);
      const uint32_t second = ckv_smem.template advance_offset_by_row<16, UPCAST_STRIDE_CKV>(ckv_smem_offset_r);
      ckv_smem.ldmatrix_m8n8x4_trans(second, v_frag);
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(o_frag[mma_d], p_frag[1], v_frag);
      ckv_smem_offset_r = ckv_smem.template advance_offset_by_column<2>(ckv_smem_offset_r, mma_d);
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

template <typename Q, typename KV, typename O, typename Id>
struct DirectPVTraits : JointQKTraits<Q, KV, O, Id> {};
template <typename Params>
struct DirectPVHostParams : Params {
 static constexpr bool stage=false,fully_staged=false,whole_rows=true,shared_fetch_ids=true,host_split=false;
 static constexpr uint32_t num_chunks=1;
};

// Two different KV tiles share each Q fragment. Each score accumulator keeps the
// native PE-then-CKV MMA order; this performs no duplicate QK work.
template <bool INIT, typename KTraits, uint32_t D, uint32_t Q_STRIDE,
          uint32_t K_STRIDE, SwizzleMode Q_SWIZZLE, SwizzleMode K_SWIZZLE>
__device__ __forceinline__ void direct_pv_qk_pair_fragments(
    smem_t<Q_SWIZZLE> q, smem_t<K_SWIZZLE> ka, smem_t<K_SWIZZLE> kb,
    float (*sa)[8], float (*sb)[8]) {
  static_assert(KTraits::QK_SHARD && KTraits::CTA_TILE_KV == 32);
  const uint32_t lane=threadIdx.x, group=threadIdx.z, warp=threadIdx.y;
  alignas(16) uint32_t qf[4],kf[4];
  uint32_t qo=q.template get_permuted_offset<Q_STRIDE>(
      warp*16+lane%16,lane/16);
  uint32_t ko=ka.template get_permuted_offset<K_STRIDE>(
      group*16+8*(lane/16)+lane%8,(lane%16)/8);
#pragma unroll 4
  for (uint32_t dim=0;dim<D;++dim) {
    q.ldmatrix_m8n8x4(qo,qf);
    ka.ldmatrix_m8n8x4(ko,kf);
    if (INIT && dim==0)
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ,MMAMode::kInit>(sa[0],qf,kf);
    else
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sa[0],qf,kf);
    kb.ldmatrix_m8n8x4(ko,kf);
    if (INIT && dim==0)
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ,MMAMode::kInit>(sb[0],qf,kf);
    else
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sb[0],qf,kf);
    qo=q.template advance_offset_by_column<2>(qo,dim);
    ko=ka.template advance_offset_by_column<2>(ko,dim);
  }
}


// Each lane owns four packed words per 16-feature QPE MMA fragment. The query
// head/feature map is the same map produced by native ldmatrix.x4.
template <typename Params>
__device__ __forceinline__ void direct_pv_load_query(const Params& params, uint32_t row,
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

template <bool PAIR, typename KTraits>
__device__ __forceinline__ void direct_pv_compute_pe(
    typename KTraits::SharedStorage& storage, uint32_t a, uint32_t b,
    uint32_t (&qpe)[4][4], float (*sa)[8], float (*sb)[8]) {
  const uint32_t lane = threadIdx.x, group = threadIdx.z;
  smem_t<KTraits::SWIZZLE_MODE_KPE> ka(storage.kpe_p_smem[a]), kb(storage.kpe_p_smem[b]);
  alignas(16) uint32_t kf[4];
#pragma unroll
  for (uint32_t dim = 0; dim < 4; ++dim) {
    const uint32_t offset = ka.template get_permuted_offset<KTraits::UPCAST_STRIDE_KPE>(
        group * 16 + 8 * (lane / 16) + lane % 8, 2 * dim + (lane % 16) / 8);
    ka.ldmatrix_m8n8x4(offset, kf);
    if (dim == 0)
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(sa[0], qpe[dim], kf);
    else
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sa[0], qpe[dim], kf);
    if constexpr (PAIR) {
      kb.ldmatrix_m8n8x4(offset, kf);
      if (dim == 0)
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ, MMAMode::kInit>(sb[0], qpe[dim], kf);
      else
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sb[0], qpe[dim], kf);
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void direct_pv_compute_one(
    typename KTraits::SharedStorage& storage, uint32_t (&qpe)[4][4], float (*scores)[8]) {
  direct_pv_compute_pe<false, KTraits>(storage, 0, 0, qpe, scores, scores);
  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> q(storage.q_smem_nope);
  smem_t<KTraits::SWIZZLE_MODE_CKV> kv(storage.ckv_smem[0]);
  compute_qk_<false, KTraits, KTraits::NUM_MMA_D_CKV,
      KTraits::UPCAST_STRIDE_Q_NOPE, KTraits::UPCAST_STRIDE_CKV>(q, kv, scores);
}

template <typename KTraits>
__device__ __forceinline__ void direct_pv_compute_qk_pair(
    typename KTraits::SharedStorage& storage,uint32_t a,uint32_t b,
    uint32_t (&qpe)[4][4], float (*sa)[8],float (*sb)[8]) {
  direct_pv_compute_pe<true, KTraits>(storage, a, b, qpe, sa, sb);
  smem_t<KTraits::SWIZZLE_MODE_Q_NOPE> qn(storage.q_smem_nope);
  smem_t<KTraits::SWIZZLE_MODE_CKV> kna(storage.ckv_smem[a]),knb(storage.ckv_smem[b]);
  direct_pv_qk_pair_fragments<false,KTraits,KTraits::NUM_MMA_D_CKV,
      KTraits::UPCAST_STRIDE_Q_NOPE,KTraits::UPCAST_STRIDE_CKV>(qn,kna,knb,sa,sb);
}


// Two native32-row score tiles share one maximum, denominator/output rescale, then each
// runs native PV in descending tile order. This changes FP32 reduction grouping, not the
// attention equation. It is distinct from the earlier paired-QK-only schedule.
template <typename KTraits>
__device__ __forceinline__ void direct_pv_update_joint_host_mdo(
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
__device__ __forceinline__ void direct_pv_attend_joint_host_tiles(
    const Params& params, typename KTraits::SharedStorage& storage,
    typename KTraits::AttentionVariant& variant, const uint32_t* scratch, uint32_t index_base,
    uint32_t kv_len, uint32_t row, int host_tiles, float (*scores)[8], float (*o)[8], float* m, float* d) {
  static_assert(KTraits::NUM_STAGES == 2 && KTraits::CTA_TILE_KV == 32 && !Params::fully_staged);
  host_tiles=0; // Direct reads have no staging-ready dependency.
  typename KTraits::IdType pages[kHostBatchTileRows<KTraits>];
  alignas(16) uint32_t qpe[4][4];
  direct_pv_load_query(params, row, qpe);
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
    // The frozen pair helper reuses Q fragments for the two current tiles. It requires
    // both stages ready, trading the control's first-QK/load overlap for fewer Q loads.
    cp_async::wait_group<0>();
    __syncthreads();
    writeback_host_batch_tile<KTraits>(params,storage,index_base,kv_len,hi*32,hi%2);
    writeback_host_batch_tile<KTraits>(params,storage,index_base,kv_len,lo*32,lo%2);
    direct_pv_compute_qk_pair<KTraits>(storage, hi%2, lo%2, qpe, scores, scores+1);
    if (hi >= mask_tile)
      logits_mask_<KTraits>(0, hi*32, 1, kv_len, kv_len, params.num_heads, scores);
    direct_pv_update_joint_host_mdo<KTraits>(&storage, variant, scores, scores+1, o, m, d);
    direct_pv_pv<KTraits>(&storage, hi%2, scores, d, o);
    if (next_hi >= 0) {
      if (next_hi < host_tiles)
        wait_host_batch_tile<KTraits>(params, scratch, index_base, kv_len, next_hi*32);
      __syncthreads();
      load_host_batch_kv<KTraits>(&storage, params, pages, kv_len, next_hi*32, next_hi%2, index_base);
      cp_async::commit_group();
    }
    // Refill the first released stage before PV of the second tile, preserving overlap.
    if (next_lo >= 0) read_host_batch_pages<KTraits>(params, index_base, kv_len, next_lo*32, pages);
    direct_pv_pv<KTraits>(&storage, lo%2, scores+1, d, o);
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
    writeback_host_batch_tile<KTraits>(params,storage,index_base,kv_len,0,0);
    direct_pv_compute_one<KTraits>(storage, qpe, scores);
    logits_mask_<KTraits>(0, 0, 1, kv_len, kv_len, params.num_heads, scores);
    update_mdo_states_<KTraits>(&storage, 0, variant, scores, o, m, d);
    direct_pv_pv<KTraits>(&storage, 0, scores, d, o);
  }
}

template <typename KTraits, typename Params, bool ROW_MERGE>
__device__ __forceinline__ void direct_pv_run_joint_host_item(const Params& params,
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
  if (Params::whole_rows || item < full_items) {
    row = item / params.num_head_tiles;
    head_tile = item % params.num_head_tiles;
    kv_start = 0;
    kv_len = host_batch_row_len(params, row);
    host_tiles = ceil_div(params.state[kHostBatchHostCounts + row], CTA_TILE_KV);
  } else if (item < hbm_end) {
    const uint32_t tile = (item - full_items) / params.num_chunks;
    const uint32_t chunk = (item - full_items) % params.num_chunks;
    row = params.full_rows + tile / params.num_head_tiles;
    head_tile = tile % params.num_head_tiles;
    partial = item - full_items;
    const uint32_t length = host_batch_row_len(params, row);
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
  // A row of only padding (kv_len 0, e.g. a CUDA Graph filler row) writes a zero output from the
  // initial state (lse at its -5e4 start); it must not issue load_q, whose copies join the first
  // KV group.
  if (kv_len > 0) {
    load_q<KTraits>(&storage, params.q_nope + row * params.q_nope_stride_n,
                    params.q_pe + row * params.q_pe_stride_n, params.q_nope_stride_n,
                    params.q_nope_stride_h, params.q_pe_stride_n, params.q_pe_stride_h, 1,
                    head_tile * CTA_TILE_Q, params.num_heads);
    // load_q's copies join the first KV group, as in the native kernel.
    direct_pv_attend_joint_host_tiles<KTraits>(params, storage, variant, scratch,
                                     params.kv_indptr[row] + kv_start, kv_len, row, host_tiles, s_frag,
                                     o_frag, m, d);
  }
  __syncthreads();
  normalize_d_<KTraits>(&storage, 0, o_frag, m, d);
  finalize_m_<KTraits>(variant, m);

  // Each warp reads back output rows other lanes stored to shared memory.
  if (Params::whole_rows || item < full_items) {
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
  if constexpr (ROW_MERGE) {
    if (item >= full_items) {
      // Publish every thread's partial writes before advancing this row's completed item count.
      __threadfence();
      __syncthreads();
      const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
      if (tid == 0) atomicAdd(host_batch_row_completed(params) + row, 1u);
    }
  }
}
template <typename KTraits, typename Params, bool ROW_MERGE>
__device__ __forceinline__ void direct_pv_run_joint_host_items(const Params& params,
                                                     typename KTraits::SharedStorage& storage,
                                                     uint8_t* smem, uint32_t* scratch,
                                                     bool fetcher, uint32_t unique) {
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  // With no Host page the item list has no Host items, and host_batch_host_items was never
  // written this launch, so it must not be read.
  const int lru_items = host_batch_caching(params) && params.stage && unique ? params.num_pools : 0;
  const int items = lru_items +
                    (params.full_rows + (params.num_rows - params.full_rows) * params.num_chunks) *
                        params.num_head_tiles +
                    (unique ? host_batch_host_items(params)[params.num_rows] : 0);
  const int workers = gridDim.y - params.fetch_ctas;
  const int first = fetcher ? items - 1 - (blockIdx.y - workers) : blockIdx.y;
  const int step = fetcher ? -static_cast<int>(params.fetch_ctas) : workers;

  float s_frag[2][8];
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
    direct_pv_run_joint_host_item<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, item - lru_items, s_frag, o_frag,
                                 m, d);
  }
}
template <typename KTraits, typename Params, bool ROW_MERGE>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLADirectPVKernel(
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
  __shared__ uint32_t scratch[3];
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  // The previous launch's last CTA advanced the epoch; ready flags from it no longer match.
  const uint32_t epoch = params.state[kHostBatchEpoch];

  // Not staging and no cache: nothing to prepare; remap is kv_indices (see the launcher) and
  // the host counts stay zero.
  const bool skip_prepare = !params.stage && !host_batch_caching(params);
  uint32_t unique = 0;
  // Grid-uniform: every CTA reads the flag after the same barrier.
  bool gaps = false;
  if (!skip_prepare) {
    if (params.stage) {
#ifdef FLASHINFER_HOST_BATCH_SMALL_PREPARE
      if constexpr (ROW_MERGE) prepare_host_batch_small_rows<KTraits>(params, epoch + 1);
      else prepare_host_batch_rows<KTraits>(params, epoch + 1);
#else
      prepare_host_batch_rows<KTraits>(params, epoch + 1);
#endif
    } else {
      prepare_host_batch_rows_in_place<KTraits>(params, epoch + 1);
    }
    cg::this_grid().sync();
    gaps = params.stage && host_batch_ld_acquire(params.state + kHostBatchGaps) != 0;
#ifdef FLASHINFER_HOST_BATCH_SMALL_PREPARE
    if constexpr (ROW_MERGE) unique = resolve_host_batch_small_slots<KTraits>(params, smem);
    else unique = resolve_host_batch_slots<KTraits>(params, smem);
#else
    unique = resolve_host_batch_slots<KTraits>(params, smem);
#endif
    if (gaps) compact_host_batch_rows<KTraits>(params);
    // In place, victims are chosen now, so they are known when the attention loads the pages.
    if (host_batch_writes_back(params) && unique > 0) {
#pragma unroll 1
      for (uint32_t pool = blockIdx.y; pool < params.num_pools; pool += gridDim.y) {
        run_host_batch_lru<KTraits>(params, smem, pool, epoch, unique,
                                    reinterpret_cast<const uint32_t*>(smem));
        __syncthreads();
      }
    }
    if (blockIdx.y == 0 && tid == 0) params.state[kHostBatchUnique] = unique;
  }
  // unique is the same on every CTA (each runs the same scan), so the branch is grid-uniform and
  // the cooperative barriers below stay matched.
  // Which end of the item list this CTA walks from. This is a partition of the grid and must not
  // depend on `unique`: if the back CTAs also walked from the front they would retrace the same
  // items and contend on the claims (0% Host, 512 rows: 1514 -> 1784 us).
  const bool from_back = blockIdx.y >= gridDim.y - params.fetch_ctas;
  if (unique > 0 || gaps) {
    // Rows resolved or compacted by other CTAs must be complete before any item or fetch reads them, and fetch
    // CTAs clear the dedup slots the resolve step reads.
    cg::this_grid().sync();
    // Claim counts were read by every CTA's scan; reset them for the next launch.
    for (uint32_t row = blockIdx.y * KTraits::NUM_THREADS + tid; row < params.num_rows;
         row += gridDim.y * KTraits::NUM_THREADS) {
      host_batch_row_claims(params)[row] = 0;
    }

  }
  const uint32_t fetch_rows = host_batch_fetch_rows<ROW_MERGE>(params, unique);
  if (tid == 0) {
    scratch[kHostBatchScratchEpoch] = epoch;
    scratch[kHostBatchScratchFetchRows] = fetch_rows;
  }
  __syncthreads();
  // Fetch CTAs take attention items once their copies are issued, so low Host shares and the end
  // of the PCIe stream keep every SM busy; with no Host page there is nothing to fetch.
  if (unique > 0 && from_back && !host_batch_writes_back(params)) fetch_stream_source_pages<KTraits>(params, smem, epoch, unique, fetch_rows);
  direct_pv_run_joint_host_items<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, from_back, unique);
  // Uniform over the grid, so every CTA takes the same branch.
  if (!Params::whole_rows && params.full_rows < params.num_rows) {
    // A small staged merge waits only for the row it reads, overlapping the remaining rows and
    // cache writeback. Every other path retains the full-grid merge barrier.
    if constexpr (!ROW_MERGE) cg::this_grid().sync();
    // 32 features per thread fill the grid only with enough (row tile, head) units; a small batch
    // merged on 32 of 108 SMs at that width and took ~15 us for 8 rows, so it uses 8.
    const uint32_t units = (params.num_rows - params.full_rows) * params.num_head_tiles *
                           KTraits::CTA_TILE_Q;
    if (params.num_rows == 1) {
      merge_host_batch_single_query<KTraits, Params, ROW_MERGE>(params, smem);
    } else if (units >= gridDim.y * (KTraits::NUM_THREADS / (KTraits::HEAD_DIM_CKV / 32))) {
      merge_host_batch_chunks<KTraits, 32, Params, ROW_MERGE>(params);
    } else {
      merge_host_batch_chunks<KTraits, 8, Params, ROW_MERGE>(params);
    }
  }

  // The last CTA to leave resets the shared counters for the next launch: the exit count, the
  // per-row placement cursors (Host counts included), and then the epoch.
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    scratch[kHostBatchScratchClaimed] =
        atomicAdd(params.state + kHostBatchExit, 1u) == gridDim.y - 1;
  }
  __syncthreads();
  if (scratch[kHostBatchScratchClaimed]) {
    if (!skip_prepare) {
      for (uint32_t row = tid; row < params.num_rows; row += KTraits::NUM_THREADS) {
        params.state[kHostBatchHostCounts + row] = 0;
        host_batch_local_cursors(params)[row] = 0;
      }
      if (tid == 0) params.state[kHostBatchGaps] = 0;
    }
    if constexpr (ROW_MERGE) {
      for (uint32_t row = tid; row < params.num_rows; row += KTraits::NUM_THREADS)
        host_batch_row_completed(params)[row] = 0;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      params.state[kHostBatchExit] = 0;
      params.state[kHostBatchEpoch] = epoch + 1;
    }
  }
}


#endif  // !FLASHINFER_FAR_EMU

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
#ifndef FLASHINFER_FAR_EMU
    using DirectPVKernelTraits = DirectPVTraits<DTypeQ, DTypeKV, DTypeO, IdType>;
    using DirectPVKernelParams = DirectPVHostParams<Params>;
    constexpr size_t direct_pv_smem = sizeof(typename DirectPVKernelTraits::SharedStorage);
    static_assert(direct_pv_smem == 147968);
    const bool direct_pv = std::is_same<DTypeQ, nv_bfloat16>::value &&
        std::is_same<DTypeKV, nv_bfloat16>::value && std::is_same<DTypeO, nv_bfloat16>::value &&
        !params.stage && !params.host_split && params.num_head_tiles == 1 &&
        params.pool_capacity > 0 && params.num_rows >= 4u * num_sms && params.fetch_ctas >= 1 &&
        params.fetch_ctas < num_sms && smem_limit_per_sm >= direct_pv_smem + 4096 &&
        smem_limit_per_sm < 200000;
    if (direct_pv) {
      auto direct_pv_kernel = BatchMLADirectPVKernel<DirectPVKernelTraits, DirectPVKernelParams, false>;
      FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(direct_pv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, direct_pv_smem));
      int direct_pv_blocks = 0;
      FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&direct_pv_blocks, direct_pv_kernel, 256, direct_pv_smem));
      if (direct_pv_blocks >= 1) {
        const uint32_t direct_pv_grid = num_sms;
        if (params.pool_capacity > kHostBatchMaxPool || 2ull * params.pool_capacity * sizeof(uint32_t) > direct_pv_smem)
          return cudaErrorInvalidConfiguration;
        static_assert(sizeof(DirectPVKernelParams) == sizeof(Params));
        static_assert(alignof(DirectPVKernelParams) == alignof(Params));
        params.full_rows = params.num_rows;
        params.partial_capacity = partial_capacity;
        params.num_chunks = 1;
        const uint32_t length = max(max_kv_len, 1u);
        params.prepare_parts = min(ceil_div(direct_pv_grid, max(params.num_rows, 1u)), ceil_div(length, 256u));
        params.prepare_segment = round_up(ceil_div(length, params.prepare_parts), 256u);
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(cudaLaunchCooperativeKernel((void*)direct_pv_kernel, dim3(1, direct_pv_grid), dim3(32, 4, 2), args, direct_pv_smem, stream));
        return cudaSuccess;
      }
    }
#endif  // !FLASHINFER_FAR_EMU
    using PVFeatureKernelTraits = PVFeatureTraits<DTypeQ, DTypeKV, DTypeO, IdType>;
    using PVFeatureKernelParams = PVFeatureHostParams<Params>;
    constexpr size_t pv_feature_smem = sizeof(typename PVFeatureKernelTraits::SharedStorage);
    static_assert(pv_feature_smem == 147968);
    const bool pv_feature = params.stage && !params.host_split && params.num_head_tiles == 1 &&
        params.pool_capacity > 0 && params.num_rows >= 4u * num_sms && params.fetch_ctas >= 1 &&
        params.fetch_ctas < num_sms && smem_limit_per_sm >= pv_feature_smem + 4096 &&
        smem_limit_per_sm < 200000 && static_cast<uint64_t>(params.staging_capacity) >=
            static_cast<uint64_t>(params.num_rows) * max_kv_len;
    if (pv_feature) {
      auto pv_feature_kernel = BatchMLAPVFeatureKernel<PVFeatureKernelTraits, PVFeatureKernelParams, false>;
      FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(pv_feature_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, pv_feature_smem));
      int pv_feature_blocks = 0;
      FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&pv_feature_blocks, pv_feature_kernel, 256, pv_feature_smem));
      if (pv_feature_blocks >= 1) {
        const uint32_t pv_feature_grid = num_sms;
        if (params.pool_capacity > kHostBatchMaxPool || 2ull * params.pool_capacity * sizeof(uint32_t) > pv_feature_smem)
          return cudaErrorInvalidConfiguration;
        static_assert(sizeof(PVFeatureKernelParams) == sizeof(Params));
        static_assert(alignof(PVFeatureKernelParams) == alignof(Params));
        params.full_rows = params.num_rows;
        params.partial_capacity = partial_capacity;
        params.num_chunks = 1;
        const uint32_t length = max(max_kv_len, 1u);
        params.prepare_parts = min(ceil_div(pv_feature_grid, max(params.num_rows, 1u)), ceil_div(length, 256u));
        params.prepare_segment = round_up(ceil_div(length, params.prepare_parts), 256u);
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(cudaLaunchCooperativeKernel((void*)pv_feature_kernel, dim3(1, pv_feature_grid), dim3(32, 4, 2), args, pv_feature_smem, stream));
        return cudaSuccess;
      }
    }
    using StreamSourceKernelTraits = StreamSourceTraits<DTypeQ, DTypeKV, DTypeO, IdType>;
    using StreamSourceKernelParams = StreamSourceHostParams<Params>;
    constexpr size_t stream_source_smem = sizeof(typename StreamSourceKernelTraits::SharedStorage);
    static_assert(stream_source_smem == 147968);
    const bool stream_source = params.stage && !params.host_split && params.num_head_tiles == 1 &&
        params.pool_capacity > 0 && params.num_rows >= 4u * num_sms && params.fetch_ctas >= 1 &&
        params.fetch_ctas < num_sms && smem_limit_per_sm >= stream_source_smem + 4096 &&
        smem_limit_per_sm < 200000 && static_cast<uint64_t>(params.staging_capacity) >=
            static_cast<uint64_t>(params.num_rows) * max_kv_len;
    if (stream_source) {
      auto stream_source_kernel = BatchMLAStreamSourceKernel<StreamSourceKernelTraits, StreamSourceKernelParams, false>;
      FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(stream_source_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, stream_source_smem));
      int stream_source_blocks = 0;
      FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&stream_source_blocks, stream_source_kernel, 256, stream_source_smem));
      if (stream_source_blocks >= 1) {
        const uint32_t stream_source_grid = num_sms;
        if (params.pool_capacity > kHostBatchMaxPool || 2ull * params.pool_capacity * sizeof(uint32_t) > stream_source_smem)
          return cudaErrorInvalidConfiguration;
        static_assert(sizeof(StreamSourceKernelParams) == sizeof(Params));
        static_assert(alignof(StreamSourceKernelParams) == alignof(Params));
        params.full_rows = params.num_rows;
        params.partial_capacity = partial_capacity;
        params.num_chunks = 1;
        const uint32_t length = max(max_kv_len, 1u);
        params.prepare_parts = min(ceil_div(stream_source_grid, max(params.num_rows, 1u)), ceil_div(length, 256u));
        params.prepare_segment = round_up(ceil_div(length, params.prepare_parts), 256u);
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(cudaLaunchCooperativeKernel((void*)stream_source_kernel, dim3(1, stream_source_grid), dim3(32, 4, 2), args, stream_source_smem, stream));
        return cudaSuccess;
      }
    }
    using CachedPEKernelTraits = CachedPETraits<DTypeQ, DTypeKV, DTypeO, IdType>;
    using CachedPEKernelParams = CachedPEHostParams<Params>;
    constexpr size_t cached_pe_smem = sizeof(typename CachedPEKernelTraits::SharedStorage);
    static_assert(cached_pe_smem == 147968);
    const bool cached_pe = params.stage && !params.host_split && params.num_head_tiles == 1 &&
        params.pool_capacity > 0 && params.num_rows >= 4u * num_sms && params.fetch_ctas >= 1 &&
        params.fetch_ctas < num_sms && smem_limit_per_sm >= cached_pe_smem + 4096 &&
        smem_limit_per_sm < 200000 && static_cast<uint64_t>(params.staging_capacity) >=
            static_cast<uint64_t>(params.num_rows) * max_kv_len;
    if (cached_pe) {
      auto cached_pe_kernel = BatchMLACachedPEKernel<CachedPEKernelTraits, CachedPEKernelParams, false>;
      FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(cached_pe_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, cached_pe_smem));
      int cached_pe_blocks = 0;
      FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&cached_pe_blocks, cached_pe_kernel, 256, cached_pe_smem));
      if (cached_pe_blocks >= 1) {
        const uint32_t cached_pe_grid = num_sms;
        if (params.pool_capacity > kHostBatchMaxPool || 2ull * params.pool_capacity * sizeof(uint32_t) > cached_pe_smem)
          return cudaErrorInvalidConfiguration;
        static_assert(sizeof(CachedPEKernelParams) == sizeof(Params));
        static_assert(alignof(CachedPEKernelParams) == alignof(Params));
        params.full_rows = params.num_rows;
        params.partial_capacity = partial_capacity;
        params.num_chunks = 1;
        const uint32_t length = max(max_kv_len, 1u);
        params.prepare_parts = min(ceil_div(cached_pe_grid, max(params.num_rows, 1u)), ceil_div(length, 256u));
        params.prepare_segment = round_up(ceil_div(length, params.prepare_parts), 256u);
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(cudaLaunchCooperativeKernel((void*)cached_pe_kernel, dim3(1, cached_pe_grid), dim3(32, 4, 2), args, cached_pe_smem, stream));
        return cudaSuccess;
      }
    }
    using KV64Traits = JointQKTraits<DTypeQ, DTypeKV, DTypeO, IdType>;
    using KV64Params = JointQKHostParams<Params>;
    constexpr size_t kv64_smem = sizeof(typename KV64Traits::SharedStorage);
    static_assert(kv64_smem == 147968);
    const bool kv64 = params.stage && !params.host_split && params.num_head_tiles == 1 &&
        params.pool_capacity > 0 && params.num_rows >= 4u * num_sms && params.fetch_ctas >= 1 &&
        params.fetch_ctas < num_sms && smem_limit_per_sm >= kv64_smem + 4096 &&
        smem_limit_per_sm < 200000 && static_cast<uint64_t>(params.staging_capacity) >=
            static_cast<uint64_t>(params.num_rows) * max_kv_len;
    if (kv64) {
      auto kv64_kernel = BatchMLAJointQKKernel<KV64Traits, KV64Params, false>;
      FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(kv64_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kv64_smem));
      int kv64_blocks = 0;
      FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&kv64_blocks, kv64_kernel, 256, kv64_smem));
      if (kv64_blocks >= 1) {
        const uint32_t kv64_grid = num_sms;
        if (params.pool_capacity > kHostBatchMaxPool || 2ull * params.pool_capacity * sizeof(uint32_t) > kv64_smem)
          return cudaErrorInvalidConfiguration;
        static_assert(sizeof(KV64Params) == sizeof(Params));
        static_assert(alignof(KV64Params) == alignof(Params));
        params.full_rows = params.num_rows;
        params.partial_capacity = partial_capacity;
        params.num_chunks = 1;
        const uint32_t length = max(max_kv_len, 1u);
        params.prepare_parts = min(ceil_div(kv64_grid, max(params.num_rows, 1u)), ceil_div(length, 256u));
        params.prepare_segment = round_up(ceil_div(length, params.prepare_parts), 256u);
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(cudaLaunchCooperativeKernel((void*)kv64_kernel, dim3(1, kv64_grid), dim3(32, 4, 2), args, kv64_smem, stream));
        return cudaSuccess;
      }
    }
    DISPATCH_SMEM_CONFIG(smem_limit_per_sm, NUM_STAGES, CTA_TILE_KV, QK_SHARD, {
      using KTraits = KernelTraits<false, NUM_STAGES, QK_SHARD, HEAD_DIM_CKV, HEAD_DIM_KPE,
                                   /*CTA_TILE_Q_=*/64, CTA_TILE_KV, DTypeQ, DTypeKV, DTypeO, IdType>;
      // Keep the original path free of the row-completion protocol, including its register and
      // instruction-scheduling effects. Each launch still runs one cooperative kernel.
      const bool row_merge = params.stage && params.pool_capacity > 0 &&
                             params.num_rows * params.num_head_tiles <= 32;
      using WholeParams = WholeRowHostBatchParams<Params>;
      // Whole rows avoid partial states and a full-grid merge once enough independent rows
      // fill several waves. host_split=false opts into this schedule; small launches and
      // staging overflow retain the original path.
      const bool whole_rows = params.stage && !params.host_split && params.num_head_tiles == 1 &&
          (params.num_rows >= 4u * num_sms ||
           (CTA_TILE_KV == 32 && params.num_rows >= 2u * num_sms &&
            host_batch_balanced_whole_rows(params.num_rows, max_kv_len, num_sms,
                                          params.fetch_ctas, partial_capacity))) &&
          static_cast<uint64_t>(params.staging_capacity) >=
              static_cast<uint64_t>(params.num_rows) * max_kv_len;
      auto kernel = row_merge ? BatchMLAHostBatchKernel<KTraits, Params, true>
                              : BatchMLAHostBatchKernel<KTraits, Params, false>;
      if (whole_rows) {
        static_assert(sizeof(WholeParams) == sizeof(Params));
        static_assert(alignof(WholeParams) == alignof(Params));
        kernel = reinterpret_cast<decltype(kernel)>(BatchMLAHostBatchKernel<KTraits, WholeParams, false>);
        if constexpr (CTA_TILE_KV == 32) {
          if (params.pool_capacity > 0) {
            using SharedParams = CachedSharedIdsHostBatchParams<Params>;
            static_assert(sizeof(SharedParams)==sizeof(Params));
            static_assert(alignof(SharedParams)==alignof(Params));
            kernel = reinterpret_cast<decltype(kernel)>(BatchMLAHostBatchKernel<KTraits, SharedParams, false>);
          }
        }
      }
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
      // LRU victim selection keeps a pool's keys in the dynamic shared memory, in place after
      // the row bases.
      const size_t lru_words = 2 * params.pool_capacity +
                               (params.stage ? 0 : round_up(params.num_rows, 32u));
      if (params.pool_capacity > kHostBatchMaxPool ||
          (params.pool_capacity && lru_words * sizeof(uint32_t) > smem_size)) {
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
      if (whole_rows) {
        params.full_rows=params.num_rows;
        tail_tiles=0;
      }
      params.partial_capacity = partial_capacity;
      const uint32_t kv_len = max(max_kv_len, 1u);
      // Prepare units: split rows only while there are fewer than the grid, one thread per entry
      // at the finest.
      params.prepare_parts =
          min(ceil_div(grid, max(params.num_rows, 1u)), ceil_div(kv_len, KTraits::NUM_THREADS));
      params.prepare_segment = round_up(ceil_div(kv_len, params.prepare_parts), KTraits::NUM_THREADS);
      params.num_chunks = 1;
      // Staged rows also produce Host items after these HBM chunks. Leave part of the first
      // wave for those items and the LRU, rather than sizing the HBM work to fill it alone.
      const uint32_t max_chunks = params.stage && params.host_split && params.pool_capacity > 0 &&
                                          tail_tiles > 0 && tail_tiles < grid
                                      ? min(32u, max(1u, grid * 3 / (tail_tiles * 4)))
                                      : 32u;
      uint64_t best_cost = UINT64_MAX;
      for (uint32_t chunks = 1; chunks <= max_chunks && tail_tiles > 0; ++chunks) {
        const uint32_t chunk_len = round_up(ceil_div(kv_len, chunks), CTA_TILE_KV);
        if (chunks > 1 && tail_tiles * (chunks + 1) > partial_capacity) break;
        // A count whose tile-rounded chunks leave the last one empty repeats a smaller split, but
        // larger counts can still be finer (one row of 2048: 8 chunks, then 13), so skip it.
        if (chunks > 1 && chunk_len * (chunks - 1) >= kv_len) continue;
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
