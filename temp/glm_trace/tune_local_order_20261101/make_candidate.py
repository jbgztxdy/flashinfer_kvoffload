"""Sort only the resident/near suffix after claiming a whole-row attention item."""
import hashlib
import json
from pathlib import Path
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
base = root / 'include/flashinfer/attention/mla_host_batch.cuh'
assert sha(base) == 'a43617290e405fdcd54b09140dbc29b357d2923e17a3d99a193caf5f7d592b15'
assert not (p / 'local_order.cuh').exists(), 'Do not overwrite a frozen candidate'
text = base.read_text()
(p / 'base.cuh').write_bytes(base.read_bytes())
math = (root / 'include/flashinfer/attention/mla.cuh').read_bytes()
for v in ['base', 'local_order']:
    (p / (v + '_mla.cuh')).write_bytes(math)
helper = r'''
// A whole row is claimed by exactly one CTA. The miss prefix and all LRU/fetch
// metadata remain unchanged; only the resident/near suffix is permuted.
template <typename KTraits, typename Params>
__device__ __forceinline__ void sort_local_order_tail(
    const Params& params, uint8_t* smem, uint32_t row, uint32_t valid) {
  static_assert(Params::whole_rows && Params::fully_staged);
  const uint32_t host = params.state[kHostBatchHostCounts + row];
  const uint32_t count = valid - host;
  if (count < 2) return;  // CTA uniform.
  const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t size = 1u << (32 - __clz(count - 1));
  auto* values = reinterpret_cast<typename Params::IdType*>(smem);
  const auto begin = params.kv_indptr[row] + host;
#pragma unroll 1
  for (uint32_t i = tid; i < size; i += KTraits::NUM_THREADS)
    values[i] = i < count ? params.remap[begin + i] : INT32_MAX;
  __syncthreads();
#pragma unroll 1
  for (uint32_t width = 2; width <= size; width <<= 1) {
#pragma unroll 1
    for (uint32_t stride = width >> 1; stride; stride >>= 1) {
#pragma unroll 1
      for (uint32_t i = tid; i < size; i += KTraits::NUM_THREADS) {
        const uint32_t other = i ^ stride;
        if (other > i) {
          const auto a = values[i], b = values[other];
          const bool ascending = (i & width) == 0;
          if (ascending ? a > b : a < b) {
            values[i] = b;
            values[other] = a;
          }
        }
      }
      __syncthreads();
    }
  }
#pragma unroll 1
  for (uint32_t i = tid; i < count; i += KTraits::NUM_THREADS)
    params.remap[begin + i] = values[i];
  // Every thread's subsequent KV index loads must observe the permutation.
  __syncthreads();
}
'''
start = text.index('template <typename Q, typename KV, typename O, typename Id>\nstruct CachedPETraits')
end = text.index('// partial_capacity bounds', start)
section = text[start:end].replace('CachedPE', 'LocalOrder').replace('cached_pe_', 'local_order_')
needle = '  typename KTraits::AttentionVariant variant(params, row, smem);'
assert section.count(needle) == 1
section = section.replace(needle, '  sort_local_order_tail<KTraits>(params, smem, row, kv_len);\n' + needle)
start_dispatch = text.index('    using CachedPEKernelTraits =')
end_dispatch = text.index('    using KV64Traits =', start_dispatch)
dispatch = text[start_dispatch:end_dispatch].replace('CachedPE', 'LocalOrder').replace('cached_pe', 'local_order')
dispatch = dispatch.replace('const bool local_order = params.stage',
    'const bool local_order = max_kv_len > 1 && max_kv_len <= 2048 && params.stage')
text = text[:end] + '\n' + helper + section + text[end:]
at = text.index('    using CachedPEKernelTraits =')
text = text[:at] + dispatch + text[at:]
(p / 'local_order.cuh').write_text(text)
for v in ['base', 'local_order']:
    tree = p / ('include_' + v)
    shutil.copytree(root / 'include', tree)
    (tree / 'flashinfer/attention/mla_host_batch.cuh').write_bytes((p / (v + '.cuh')).read_bytes())
    (tree / 'flashinfer/attention/mla.cuh').write_bytes(math)
record = dict(base_sha256=sha(p / 'base.cuh'), candidate_sha256=sha(p / 'local_order.cuh'),
    math_sha256=sha(p / 'base_mla.cuh'), candidate_adopted=False, goal_complete=False,
    mechanism='The sole claiming whole-row attention CTA bitonic-sorts only resident/near suffix by signed physical row index, using pre-query dynamic shared memory; miss prefix untouched.',
    invariants='Original prepare, resolve, LRU admission/victim selection, farfetch, cache payload, flags, claims and grid synchronization are literal unchanged. Same query and KV multiset; online FP32 grouping changes after permutation.',
    scope='Original large cached fully-staged CPE dispatch plus1<maxK<=2048; same256threads108CTA/32KV2stage/147968B/f24; all original definitions kept.',
    costs='Per-item sorting, up to66 CTA comparison phases at2048 plus load/store barriers, global suffix write/reread. Locality proxy not a speedup; resources, numerical error and measured cost must decide.')
(p / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
