"""Give each pool a bounded sparse staging range and resolve remap in prepare."""
import hashlib
import json
from pathlib import Path
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
base = root / 'include/flashinfer/attention/mla_host_batch.cuh'
assert sha(base) == 'a43617290e405fdcd54b09140dbc29b357d2923e17a3d99a193caf5f7d592b15'
assert not (p / 'sparse_prepare.cuh').exists()
text = base.read_text()
(p / 'base.cuh').write_bytes(base.read_bytes())
math = (root / 'include/flashinfer/attention/mla.cuh').read_bytes()
for v in ['base', 'sparse_prepare']:
    (p / (v + '_mla.cuh')).write_bytes(math)

def function_block(name, next_marker):
    at = text.index(name + '(')
    start = text.rindex('template <', 0, at)
    return text[start:text.index(next_marker, at)]

prepare = function_block('prepare_host_batch_rows', '// prepare_host_batch_rows without staging')
# The first occurrence is a comment in the parameter declaration; use the
# explicit function signature as the extraction anchor.
at = text.index('__device__ __forceinline__ void prepare_host_batch_rows(')
start = text.rindex('template <', 0, at)
prepare = text[start:text.index('// prepare_host_batch_rows without staging', at)]
prepare = prepare.replace('prepare_host_batch_rows(', 'prepare_sparse_pool_rows(')
prepare = prepare.replace('    const IdType begin = params.kv_indptr[row];',
    '    const uint32_t pool = params.row_pool[row];\n'
    '    const uint32_t pool_base = params.pool_row_start[pool] * params.prepare_segment;\n'
    '    const IdType begin = params.kv_indptr[row];')
assert prepare.count('atomicAdd(host_batch_row_claims(params) + row, fresh_total)') == 1
prepare = prepare.replace('atomicAdd(host_batch_row_claims(params) + row, fresh_total)',
                          'atomicAdd(host_batch_row_claims(params) + pool, fresh_total)')
needle = '''      for (uint32_t i = 0; i < PER_THREAD; ++i) {
        if (fresh_masks[i] >> lane & 1) {'''
at = prepare.index(needle)
end = prepare.index('        if (pages[i] >= params.host_page_start) {', at)
prepare = prepare[:at] + '''      for (uint32_t i = 0; i < PER_THREAD; ++i) {
        if (fresh_masks[i] >> lane & 1) {
          const uint32_t slot = pool_base + first + __popc(fresh_masks[i] & lower);
          params.unique_pages[slot] = pages[i];
          params.slot_pool[slot] = pool;
          atomicExch(params.host_slot + (pages[i] - params.host_page_start), slot + 2);
        }
        first += __popc(fresh_masks[i]);
      }
      // Publish EVERY owned page before waiting on ANY other owner. This avoids
      // cross-iteration cycles; all CTAs are resident in the cooperative launch.
      __syncwarp();
#pragma unroll
      for (uint32_t i = 0; i < PER_THREAD; ++i) {
''' + prepare[end:]
prepare = prepare.replace('''          params.remap[begin + host_first + __popc(host_masks[i] & lower)] = pages[i];''',
'''          const uint32_t* address = params.host_slot + (pages[i] - params.host_page_start);
          uint32_t claim;
          do {
            claim = host_batch_ld_acquire(address);
            if (claim == 1) __nanosleep(64);
          } while (claim == 1);
          params.remap[begin + host_first + __popc(host_masks[i] & lower)] =
              host_batch_staged_entry(params, claim - 2);''')
# The publication pass already advanced first; the placement pass needs only
# the host/local cursors. No semantic use of first remains in that pass.
prepare = prepare.replace('''        first += __popc(fresh_masks[i]);
        host_first''', '''        host_first''')

resolve = r'''
// One bounded scan over pools replaces the old pass over every Host reference.
// host_items[pool] holds this launch's count; row_slot_base holds job prefixes.
template <typename KTraits, typename Params>
__device__ __forceinline__ uint32_t resolve_sparse_pool_counts(const Params& params, uint8_t* smem) {
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  auto* prefix = reinterpret_cast<uint32_t*>(smem);
  const auto* counts = host_batch_row_claims(params);
  const uint32_t unique = host_batch_block_scan<KTraits::NUM_THREADS>(
      params.num_pools, [&](uint32_t pool) { return counts[pool]; }, prefix);
  if (!unique) return 0;
  const uint32_t jobs = host_batch_block_scan<KTraits::NUM_THREADS>(
      params.num_pools, [&](uint32_t pool) { return ceil_div(counts[pool], 64u); }, prefix);
  if (blockIdx.y == 0) {
    for (uint32_t pool = tid; pool < params.num_pools; pool += KTraits::NUM_THREADS) {
      params.row_slot_base[pool] = prefix[pool];
      host_batch_host_items(params)[pool] = counts[pool];
    }
    if (tid == 0) {
      params.row_slot_base[params.num_pools] = jobs;
      host_batch_host_items(params)[params.num_rows] = 0;  // No tail rows.
    }
  }
  return unique;
}

template <typename Params>
__device__ __forceinline__ uint2 sparse_pool_fetch_job(const Params& params, uint32_t job) {
  uint32_t first = 0, last = params.num_pools;
  while (first + 1 < last) {
    const uint32_t mid = (first + last) / 2;
    if (params.row_slot_base[mid] <= job) first = mid;
    else last = mid;
  }
  const uint32_t offset = (job - params.row_slot_base[first]) * 64;
  return make_uint2(params.pool_row_start[first] * params.prepare_segment + offset,
                    min(64u, host_batch_host_items(params)[first] - offset));
}
'''
lru = function_block('run_host_batch_lru', '// Claims an item for this CTA;')
lru = lru.replace('run_host_batch_lru(', 'run_sparse_pool_lru(')
old = '''  const uint32_t first = min(slot_base(params.pool_row_start[pool]), count);
  const uint32_t last = min(slot_base(params.pool_row_start[pool + 1]), count);'''
assert old in lru
lru = lru.replace(old, '''  const uint32_t first = params.pool_row_start[pool] * params.prepare_segment;
  const uint32_t last = first + host_batch_host_items(params)[pool];''')

fetch = function_block('fetch_host_batch_pages', '// Requires the combined [page, 576] layout')
fetch = fetch.replace('fetch_host_batch_pages(', 'fetch_sparse_pool_pages(')
fetch = fetch.replace('  const uint32_t count = params.stage ? min(unique, params.staging_capacity) : unique;',
                      '  const uint32_t count = params.staging_capacity;\n'
                      '  const uint32_t jobs = params.row_slot_base[params.num_pools];')
fetch = fetch.replace('''  auto issue = [&](uint32_t chunk, uint32_t buf) {''',
'''  auto issue = [&](uint32_t chunk, uint32_t buf) {
    const uint2 info = sparse_pool_fetch_job(params, chunk);''')
# Every source/staging/victim/slot_pool/host_slot address uses physical sparse
# slots. Job/chunk loop iteration remains dense and load balanced.
fetch = fetch.replace('chunk * rows +', 'info.x +')
fetch = fetch.replace('slot < count', 'slot < info.x + info.y')
fetch = fetch.replace('info.x + tid < count', 'tid < info.y')
fetch = fetch.replace('info.x + v / VECS_PER_ROW < count', 'v / VECS_PER_ROW < info.y')
fetch = fetch.replace('min(rows, count - chunk * rows)', 'info.y')
fetch = fetch.replace('chunk * rows < count', 'chunk < jobs')
fetch = fetch.replace('next * rows < count', 'next < jobs')
fetch = fetch.replace('''    const uint32_t next = chunk + params.fetch_ctas;''',
'''    const uint2 info = sparse_pool_fetch_job(params, chunk);
    const uint32_t next = chunk + params.fetch_ctas;''')
fetch = fetch.replace('static_cast<int64_t>(chunk) * rows', 'static_cast<int64_t>(info.x)')
fetch = fetch.replace('"l"(flags + chunk)', '"l"(flags + info.x / rows)')
fetch = fetch.replace('''    const uint32_t chunk = deferred[d];''',
'''    const uint32_t chunk = deferred[d];
    const uint2 info = sparse_pool_fetch_job(params, chunk);''')
# Deferred copy loops also need the final partial job's bounds.
fetch = fetch.replace('''      const int32_t victim = cache_rows[v / VECS_PER_ROW];''',
'''      const int32_t victim = v / VECS_PER_ROW < info.y ? cache_rows[v / VECS_PER_ROW] : -1;''')
overflow = fetch.index('  // Overflow pages were never staged;')
fetch = fetch[:overflow] + '}\n\n'
assert 'chunk * rows' not in fetch and 'next * rows' not in fetch

start = text.index('template <typename Q, typename KV, typename O, typename Id>\nstruct CachedPETraits')
end = text.index('// partial_capacity bounds', start)
section = text[start:end].replace('CachedPE', 'SparsePrepare').replace('cached_pe_', 'sparse_prepare_')
section = section.replace('prepare_host_batch_rows<', 'prepare_sparse_pool_rows<')
section = section.replace('resolve_host_batch_slots<', 'resolve_sparse_pool_counts<')
section = section.replace('run_host_batch_lru<', 'run_sparse_pool_lru<')
section = section.replace('fetch_host_batch_pages<', 'fetch_sparse_pool_pages<')
start_dispatch = text.index('    using CachedPEKernelTraits =')
end_dispatch = text.index('    using KV64Traits =', start_dispatch)
dispatch = text[start_dispatch:end_dispatch].replace('CachedPE', 'SparsePrepare').replace('cached_pe', 'sparse_prepare')
dispatch = dispatch.replace('const bool sparse_prepare = params.stage',
    'const bool sparse_prepare = max_kv_len > 0 && max_kv_len % 256 == 0 && params.stage')
text = text[:end] + '\n' + prepare + resolve + lru + fetch + section + text[end:]
at = text.index('    using CachedPEKernelTraits =')
text = text[:at] + dispatch + text[at:]
(p / 'sparse_prepare.cuh').write_text(text)
for v in ['base', 'sparse_prepare']:
    tree = p / ('include_' + v)
    shutil.copytree(root / 'include', tree)
    (tree / 'flashinfer/attention/mla_host_batch.cuh').write_bytes((p / (v + '.cuh')).read_bytes())
    (tree / 'flashinfer/attention/mla.cuh').write_bytes(math)
record = dict(base_sha256=sha(p / 'base.cuh'), candidate_sha256=sha(p / 'sparse_prepare.cuh'),
    math_sha256=sha(p / 'base_mla.cuh'), candidate_adopted=False, goal_complete=False,
    mechanism='Bounded pool staging ranges; warp owner rank allocation, publish all own pages beforewaiting, prepare writes final staging remap. Poolcount scan replaces fullreference resolve pass; dense fetchjobs map to sparsephysicalslots. Original LRU/cachecopy overlap remains.',
    scope='Production CPElargecachedfullystaged dispatch plus positive maxK multiple256. All other paths literaloriginal; math/FAR/reference unmodified.',
    invariants='Poolrange begins pool_row_start*maxK, enough space since poolunique<=poolrows*maxK. maxK multiple256 makes everypoolbase64aligned, no readyflag alias. Counts copied before originalclaimreset. All cooperative CTAs resident; all8ownedpages published beforeanyawait.',
    risks='Inline owner waits/newfetchjob binarysearch/poolallocation contention/changedstage locality may outweigh savedresolve. Must check exactLRU/payload, old38/76machinewords, actualnewbranch and timing.')
(p / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
