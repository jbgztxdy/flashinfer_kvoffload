"""Resolve final miss destinations once; keep the measured cached-PE source immutable."""
import hashlib
import json
from pathlib import Path
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
previous = p.parent / 'tune_cached_pe_20261101'
source = previous / 'cached_pe.cuh'
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(source) == 'a43617290e405fdcd54b09140dbc29b357d2923e17a3d99a193caf5f7d592b15'
text = source.read_text()
assert not (p / 'final_destination.cuh').exists(), 'Do not overwrite frozen candidate'
(p / 'base.cuh').write_bytes(source.read_bytes())
math = (previous / 'cached_pe_mla.cuh').read_bytes()
for v in ['base', 'final_destination']:
    (p / (v + '_mla.cuh')).write_bytes(math)

# Private LRU helper preserves exact stamp selection/ties, then retires old mappings
# before any data writes. Existing hits are protected by this launch's stamp.
start = text.rindex('template <typename KTraits, typename Params>', 0, text.index('__device__ __noinline__ void run_host_batch_lru'))
end = text.index('// Claims an item for this CTA;', start)
lru = text[start:end].replace('run_host_batch_lru(', 'run_final_destination_lru(')
old = '        params.victim_of[staged] = slot;'
assert lru.count(old) == 1
lru = lru.replace(old, '''        params.victim_of[staged] = slot;
        const int32_t evicted = params.slot_page[slot];
        if (evicted >= 0) params.resident[evicted - params.host_page_start] = -1;
        // Pending payload. Only the dedicated fetcher publishes a nonnegative page.
        params.slot_page[slot] = -1;''')

helpers = r'''
// The existing 32-bit remap stores the final destination directly. Resident hits
// are outside this miss prefix. Unadmitted misses retain their staging row.
template <typename KTraits, typename Params>
__device__ __forceinline__ void resolve_final_destinations(const Params& params) {
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  for (uint32_t row = blockIdx.y; row < params.num_rows; row += gridDim.y) {
    const auto begin = params.kv_indptr[row];
    const uint32_t count = params.state[kHostBatchHostCounts + row];
    for (uint32_t j = tid; j < count; j += KTraits::NUM_THREADS) {
      const auto page = params.remap[begin + j];
      const uint32_t physical = static_cast<uint32_t>(-(page + 1));
      if (page < 0 && physical >= params.cache_slots) {
        const int32_t victim = params.victim_of[physical - params.cache_slots];
        if (victim >= 0) params.remap[begin + j] = -victim - 1;
      }
    }
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void wait_final_destination_tile(
    const Params& params, const uint32_t* scratch, uint32_t index_base,
    uint32_t kv_len, uint32_t kv_base) {
  static_assert(KTraits::CTA_TILE_KV <= 64);
  if (threadIdx.y != 0 || threadIdx.z != 0) return;
#pragma unroll
  for (uint32_t i = 0; i < KTraits::CTA_TILE_KV; i += 32) {
    const uint32_t j = kv_base + i + threadIdx.x;
    const auto page = j < kv_len ? params.remap[index_base + j] : 0;
    if (page < 0) {
      const uint32_t row = static_cast<uint32_t>(-(page + 1));
      if (row < params.cache_slots) {
        const uint32_t* flag = reinterpret_cast<const uint32_t*>(params.slot_page + row);
        // -1 is pending, a nonnegative page is release-published after all bytes.
        while (static_cast<int32_t>(host_batch_ld_acquire(flag)) < 0) __nanosleep(64);
      } else {
        const uint32_t* flag = host_batch_flags(params) +
            (row - params.cache_slots) / scratch[kHostBatchScratchFetchRows];
        while (host_batch_ld_acquire(flag) != scratch[kHostBatchScratchEpoch] + 1) __nanosleep(64);
      }
    }
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void fetch_final_destination_pages(
    const Params& params, uint8_t* smem, uint32_t epoch, uint32_t unique, uint32_t rows) {
  using DTypeKV = typename Params::DTypeKV;
  constexpr uint32_t ROW = KTraits::HEAD_DIM_CKV + KTraits::HEAD_DIM_KPE;
  constexpr uint32_t VECS_PER_ROW = ROW * sizeof(DTypeKV) / 16;
  constexpr uint32_t CHUNK_VECS = kHostBatchFetchRows * VECS_PER_ROW;
  constexpr uint32_t BUFFERS = 2;
  static_assert(sizeof(typename KTraits::SharedStorage) >= BUFFERS * CHUNK_VECS * 16);
  static_assert(CHUNK_VECS % KTraits::NUM_THREADS == 0);
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t fetch_id = blockIdx.y - (gridDim.y - params.fetch_ctas);
  const uint32_t count = min(unique, params.staging_capacity);
  const uint32_t chunk_vecs = rows * VECS_PER_ROW;
  auto* buffer = reinterpret_cast<uint4*>(smem);
  __shared__ typename Params::IdType page_ids[BUFFERS][kHostBatchFetchRows];
  __shared__ int32_t destinations[kHostBatchFetchRows];
  FAR_EMU_INIT(2);
  FAR_EMU_SYNC();
  auto issue = [&](uint32_t chunk, uint32_t buf) {
    if (tid < rows && chunk * rows + tid < count)
      page_ids[buf][tid] = params.unique_pages[chunk * rows + tid];
    __syncthreads();
#pragma unroll
    for (uint32_t i = 0; i < CHUNK_VECS / KTraits::NUM_THREADS; ++i) {
      const uint32_t v = i * KTraits::NUM_THREADS + tid;
      if (v < chunk_vecs && chunk * rows + v / VECS_PER_ROW < count) {
        const auto page = page_ids[buf][v / VECS_PER_ROW];
        const auto* source = reinterpret_cast<const uint4*>(params.ckv + static_cast<int64_t>(page) * ROW);
        cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffer + buf * CHUNK_VECS + v, source + v % VECS_PER_ROW);
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
    if (next * rows < count) {
      issue(next, (k + 1) % 2);
      cp_async::wait_group<1>();
    } else {
      cp_async::wait_group<0>();
    }
    FAR_EMU_WAIT(2, k % 2);
    __syncthreads();
    const uint32_t buf = k % 2;
    if (tid < rows) {
      const uint32_t slot = chunk * rows + tid;
      const int32_t victim = slot < count ? params.victim_of[slot] : -1;
      destinations[tid] = victim >= 0 ? victim : params.cache_slots + slot;
    }
    __syncthreads();
#pragma unroll
    for (uint32_t i = 0; i < CHUNK_VECS / KTraits::NUM_THREADS; ++i) {
      const uint32_t v = i * KTraits::NUM_THREADS + tid;
      if (v < chunk_vecs && chunk * rows + v / VECS_PER_ROW < count) {
        auto* destination = reinterpret_cast<uint4*>(params.staging +
            static_cast<int64_t>(destinations[v / VECS_PER_ROW]) * ROW);
        destination[v % VECS_PER_ROW] = buffer[buf * CHUNK_VECS + v];
      }
    }
    // Every writer fences before the CTA barrier. Each cache page is published
    // only after all of its vectors; the original chunk flag guards staging.
    __threadfence();
    __syncthreads();
    if (tid < rows && chunk * rows + tid < count) {
      const auto page = page_ids[buf][tid];
      const int32_t destination = destinations[tid];
      params.host_slot[page - params.host_page_start] = 0;
      if (destination < params.cache_slots) {
        params.resident[page - params.host_page_start] = destination;
        asm volatile("st.release.gpu.global.u32 [%0], %1;" ::
                     "l"(params.slot_page + destination), "r"(static_cast<uint32_t>(page)) : "memory");
      }
    }
    if (tid == 0) {
      asm volatile("st.release.gpu.global.u32 [%0], %1;" ::
                   "l"(host_batch_flags(params) + chunk), "r"(epoch + 1) : "memory");
    }
    __syncthreads();
  }
  for (uint32_t slot = count + fetch_id * KTraits::NUM_THREADS + tid; slot < unique;
       slot += params.fetch_ctas * KTraits::NUM_THREADS)
    params.host_slot[params.unique_pages[slot] - params.host_page_start] = 0;
}
'''
start = text.index('template <typename Q, typename KV, typename O, typename Id>\nstruct CachedPETraits')
end = text.index('// partial_capacity bounds', start)
section = text[start:end].replace('CachedPE', 'FinalDestination').replace('cached_pe_', 'final_destination_')
section = section.replace('wait_host_batch_tile<', 'wait_final_destination_tile<')
section = section.replace('fetch_host_batch_pages<', 'fetch_final_destination_pages<')
old = 'const int lru_items = host_batch_caching(params) && params.stage && unique ? params.num_pools : 0;'
assert section.count(old) == 1
section = section.replace(old, 'const int lru_items = 0;  // LRU completed before remap resolution.')
marker = '  const uint32_t fetch_rows = host_batch_fetch_rows<ROW_MERGE>(params, unique);'
assert section.count(marker) == 1
section = section.replace(marker, '''  if (unique > 0) {
    for (uint32_t pool = blockIdx.y; pool < params.num_pools; pool += gridDim.y) {
      run_final_destination_lru<KTraits>(params, smem, pool, epoch, unique);
      __syncthreads();
    }
    cg::this_grid().sync();
    resolve_final_destinations<KTraits>(params);
    cg::this_grid().sync();
  }
''' + marker)

# New typed dispatch precedes CPE; all existing definitions and dispatch remain.
start_dispatch = text.index('    using CachedPEKernelTraits =')
end_dispatch = text.index('    using KV64Traits =', start_dispatch)
dispatch = text[start_dispatch:end_dispatch].replace('CachedPE', 'FinalDestination').replace('cached_pe', 'final_destination')
dispatch = dispatch.replace('    using FinalDestinationKernelTraits', '    using FinalDestinationKernelTraits', 1)
text = text[:end] + '\n' + lru + helpers + '\n' + section + '\n' + text[end:]
start_dispatch = text.index('    using CachedPEKernelTraits =')
text = text[:start_dispatch] + dispatch + text[start_dispatch:]
(p / 'final_destination.cuh').write_text(text)
for v in ['base', 'final_destination']:
    tree = p / ('include_' + v)
    shutil.copytree(root / 'include', tree)
    (tree / 'flashinfer/attention/mla_host_batch.cuh').write_bytes((p / (v + '.cuh')).read_bytes())
    (tree / 'flashinfer/attention/mla.cuh').write_bytes(math)
record = dict(base_sha256=sha(p / 'base.cuh'), candidate_sha256=sha(p / 'final_destination.cuh'),
    math_sha256=sha(p / 'base_mla.cuh'), candidate_adopted=False, goal_complete=False,
    base_status='Frozen CPEa436, pending production adoption when this candidate was generated; compare directly to CPE, not JQ.',
    mechanism='Upfront exact LRU, retire victim mappings and mark slot_page pending; resolve32bit remap once to finalcache or staging. Shared payload written once, release slot_page only after allwriters fence+barrier. Wait only in tile wait helper; no perKV victim lookup.',
    changes='Two extra grid barriers, LRU moved ahead of farfetch/attention; no LRU attention items. Same256thread108CTA/32KV2stage/64row2fetchbuffers/f24; math/FAR/reference unchanged.',
    risks='Extra upfront LRU/remap/barrier may outweigh saved HBM store; new perpage ready polling may regress. Must verify payload publication, original38/76machinewords, actual branch/resources and performance.')
(p / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
