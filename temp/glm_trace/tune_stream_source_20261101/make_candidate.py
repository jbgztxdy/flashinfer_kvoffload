"""Mark the deduplicated far-source reads evict-first without changing any payload work."""
import hashlib
import json
from pathlib import Path
import shutil

p = Path(__file__).resolve().parent
root = p.parents[2]
sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
base = root / 'include/flashinfer/attention/mla_host_batch.cuh'
assert sha(base) == 'a43617290e405fdcd54b09140dbc29b357d2923e17a3d99a193caf5f7d592b15'
assert not (p / 'stream_source.cuh').exists(), 'Do not overwrite frozen source'
text = base.read_text()
(p / 'base.cuh').write_bytes(base.read_bytes())
math = (root / 'include/flashinfer/attention/mla.cuh').read_bytes()
for v in ['base', 'stream_source']:
    (p / (v + '_mla.cuh')).write_bytes(math)
at = text.index('__device__ __forceinline__ void fetch_host_batch_pages(')
start = text.rindex('template <', 0, at)
end = text.index('// Requires the combined [page, 576] layout', at)
fetch = text[start:end].replace('fetch_host_batch_pages(', 'fetch_stream_source_pages(')
needle = '  FAR_EMU_INIT(2);'
assert fetch.count(needle) == 1
fetch = fetch.replace(needle, '''  uint64_t stream_source_policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
               : "=l"(stream_source_policy));
''' + needle)
needle = '''        cp_async::load_128b<PrefetchMode::kNoPrefetch>(buffer + buf * CHUNK_VECS + v,
                                                       source + v % VECS_PER_ROW);'''
assert fetch.count(needle) == 1
fetch = fetch.replace(needle, '''        const uint32_t destination = static_cast<uint32_t>(
            __cvta_generic_to_shared(buffer + buf * CHUNK_VECS + v));
        asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;"
                     :: "r"(destination), "l"(source + v % VECS_PER_ROW),
                        "l"(stream_source_policy));''')
start = text.index('template <typename Q, typename KV, typename O, typename Id>\nstruct CachedPETraits')
end = text.index('// partial_capacity bounds', start)
section = text[start:end].replace('CachedPE', 'StreamSource').replace('cached_pe_', 'stream_source_')
section = section.replace('fetch_host_batch_pages<', 'fetch_stream_source_pages<')
start_dispatch = text.index('    using CachedPEKernelTraits =')
end_dispatch = text.index('    using KV64Traits =', start_dispatch)
dispatch = text[start_dispatch:end_dispatch].replace('CachedPE', 'StreamSource').replace('cached_pe', 'stream_source')
text = text[:end] + '\n' + fetch + section + text[end:]
at = text.index('    using CachedPEKernelTraits =')
text = text[:at] + dispatch + text[at:]
(p / 'stream_source.cuh').write_text(text)
for v in ['base', 'stream_source']:
    tree = p / ('include_' + v)
    shutil.copytree(root / 'include', tree)
    (tree / 'flashinfer/attention/mla_host_batch.cuh').write_bytes((p / (v + '.cuh')).read_bytes())
    (tree / 'flashinfer/attention/mla.cuh').write_bytes(math)
record = dict(base_sha256=sha(p / 'base.cuh'), candidate_sha256=sha(p / 'stream_source.cuh'),
    math_sha256=sha(p / 'base_mla.cuh'), candidate_adopted=False, goal_complete=False,
    mechanism='Private dedicated fetch cp.async uses L2 evict_first cache policy for deduplicated far-source KV lines. No attention, staging/cache-store or metadata load policy changes.',
    invariants='Same64rowdoublebuffer/f24, actual1152bytes/uniquepage, FARpostreadhooks, commit/wait/barriers, LRU/payload/prepare/remap, claims, mathematics and dispatch. Onlyfar-source hardwarecacheevictionhint plus its setup changes.',
    rationale='Source rows read once perlaunch afterdedup; fullystagedattention reads staging/cache instead. Streaming source may displace repeatedlyusedHBM, but cachepolicy is onlyhint and no gain promised.',
    scope='SameproductionCPElargecachedfullystagedSM80dispatch;256thread108CTA32KV2stage147968B. No parameter scans orreference/simulator changes.')
(p / 'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2))
