"""Add per-CTA tile-wait and item budgets to the retained production source.

Diagnostic only: extra timers/scratch can affect scheduling. Fixed128epoch/128CTA
capacity, used only for MTP8/full1024 positions and <=128CTA. No performance claim.
"""
import hashlib
import json
import shutil
from pathlib import Path
p = Path(__file__).resolve().parent
root = p.parents[2]
old = p.parent / 'tune_general_20261031'
base = (root / 'include/flashinfer/attention/mla_host_batch.cuh').read_text()
assert hashlib.sha256(base.encode()).hexdigest() == '52919c286398d4668933a772c646bdccd46273a53aafeb8d5c1b14e2d9bff2c5'
s = base

def once(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)

once('template <typename Params>\n__device__ __forceinline__ bool host_batch_caching', '''__device__ __forceinline__ unsigned long long hb_now() {
  unsigned long long tick;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tick));
  return tick;
}
#define HB_MARK(k) do { if (threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0) { \
 auto hb_addr=reinterpret_cast<uintptr_t>(host_batch_row_completed(params)+params.num_rows); \
 auto hb_out=reinterpret_cast<unsigned long long*>((hb_addr+7)&~uintptr_t(7)); \
 auto hb_at=(((params.state[kHostBatchEpoch]%128)*128+blockIdx.y)*32); \
 hb_out[hb_at+(k)]=hb_now(); hb_out[hb_at+12]=params.state[kHostBatchEpoch]; \
 hb_out[hb_at+13]=gridDim.y; hb_out[hb_at+14]=params.full_rows; hb_out[hb_at+15]=params.num_chunks; \
} } while(0)

template <typename Params>
__device__ __forceinline__ bool host_batch_caching''')
s = s.replace('const uint32_t* scratch', 'uint32_t* scratch')
once('  __shared__ uint32_t scratch[3];', '  __shared__ uint32_t scratch[10];')
once('  const uint32_t epoch = params.state[kHostBatchEpoch];', '''  const uint32_t epoch = params.state[kHostBatchEpoch];
  if (tid == 0) {
    for (uint32_t i=3; i<10; ++i) scratch[i]=0;
    auto addr=reinterpret_cast<uintptr_t>(host_batch_row_completed(params)+params.num_rows);
    auto out=reinterpret_cast<unsigned long long*>((addr+7)&~uintptr_t(7));
    for (uint32_t i=0; i<32; ++i) out[((epoch%128)*128+blockIdx.y)*32+i]=0;
  }
  HB_MARK(0);''')
once('  if (threadIdx.y != 0 || threadIdx.z != 0) return;\n#pragma unroll', '''  if (threadIdx.y != 0 || threadIdx.z != 0) return;
  const uint32_t wait_begin=static_cast<uint32_t>(hb_now());
  bool blocked=false;
#pragma unroll''')
once('      while (host_batch_ld_acquire(flag) != scratch[kHostBatchScratchEpoch] + 1) __nanosleep(64);', '''      while (host_batch_ld_acquire(flag) != scratch[kHostBatchScratchEpoch] + 1) {
        blocked=true;
        __nanosleep(64);
      }''')
once('''  }
}

// Rows of a KV tile one thread loads''', '''  }
  const bool any_blocked=__any_sync(0xffffffff, blocked);
  if (threadIdx.x==0) {
    scratch[3]+=static_cast<uint32_t>(hb_now())-wait_begin;
    scratch[4]+=1;
    scratch[5]+=any_blocked;
  }
}

// Rows of a KV tile one thread loads''')
once('''    if (item < lru_items) {
      run_host_batch_lru<KTraits>''', '''    const uint32_t item_begin=static_cast<uint32_t>(hb_now());
    if (item < lru_items) {
      run_host_batch_lru<KTraits>''')
once('''      // The next item reuses the shared memory the selection scanned into.
      __syncthreads();
      continue;''', '''      // The next item reuses the shared memory the selection scanned into.
      __syncthreads();
      if (threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0) {
        scratch[8]+=static_cast<uint32_t>(hb_now())-item_begin;
        scratch[9]+=1;
      }
      continue;''')
once('''    run_host_batch_item<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, item - lru_items, s_frag, o_frag,
                                 m, d);''', '''    HB_MARK(26);
    run_host_batch_item<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, item - lru_items, s_frag, o_frag,
                                 m, d);
    if (threadIdx.x==0 && threadIdx.y==0 && threadIdx.z==0) {
      scratch[6]+=static_cast<uint32_t>(hb_now())-item_begin;
      scratch[7]+=1;
    }''')
once('''    cg::this_grid().sync();
    gaps =''', '''    HB_MARK(1);
    cg::this_grid().sync();
    HB_MARK(2);
    gaps =''')
once('''    if (gaps) compact_host_batch_rows<KTraits>(params);
    // In place''', '''    if (gaps) compact_host_batch_rows<KTraits>(params);
    HB_MARK(3);
    // In place''')
once('''    cg::this_grid().sync();
    // Claim counts''', '''    HB_MARK(4);
    cg::this_grid().sync();
    HB_MARK(5);
    // Claim counts''')
once('''  if (unique > 0 && from_back && !host_batch_writes_back(params)) fetch_host_batch_pages<KTraits>(params, smem, epoch, unique, fetch_rows);
  run_host_batch_items<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, from_back, unique);''', '''  HB_MARK(6);
  if (unique > 0 && from_back && !host_batch_writes_back(params)) fetch_host_batch_pages<KTraits>(params, smem, epoch, unique, fetch_rows);
  HB_MARK(7);
  run_host_batch_items<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, from_back, unique);
  HB_MARK(8);
  if (tid==0) {
    auto addr=reinterpret_cast<uintptr_t>(host_batch_row_completed(params)+params.num_rows);
    auto out=reinterpret_cast<unsigned long long*>((addr+7)&~uintptr_t(7));
    auto at=((epoch%128)*128+blockIdx.y)*32;
    for (uint32_t i=3; i<10; ++i) out[at+16+i-3]=scratch[i];
  }''')
for v, source in [('base', base), ('profile_wait', s)]:
    assert not (p / (v + '.cuh')).exists()
    (p / (v + '.cuh')).write_text(source)
    shutil.copytree(root / 'include', p / ('include_' + v))
    (p / ('include_' + v) / 'flashinfer/attention/mla_host_batch.cuh').write_text(source)
for name in ['.gitignore', 'resources.py', 'resources_gpu3.py', 'experiment.json', 'plans.cpp', 'compile.py']:
    (p / name).write_bytes((old / name).read_bytes())
bench = (old / 'bench.py').read_text()
bench = bench.replace('("profile", "profile_final")', '("profile_wait",)')
bench = bench.replace('2*128*128*16', '2*128*128*32')
bench = bench.replace('phase_timestamps_ns_flat', 'wait_diagnostic_flat')
# Fixed epoch capacity must never silently wrap a different shape/number of steps.
bench = bench.replace('A = P.parse_args()', '''A = P.parse_args()
if VARIANT == "profile_wait":
    assert A.mtp == 8 and A.positions == 1024 and A.passes == 1 and A.kind == "lru"
''')
(p / 'bench.py').write_text(bench)
(p / 'manifest.json').write_text(json.dumps(dict(
    base_sha256=hashlib.sha256(base.encode()).hexdigest(), diagnostic_sha256=hashlib.sha256(s.encode()).hexdigest(),
    classification='Diagnostic only, not performance optimization. 32 fields per epoch/CTA; capacity128epochs/128CTAs, 1pass MTP8/1024positions.',
    fields={0:'start',1:'prepare_done',2:'prepare_barrier_done',3:'resolve_done',4:'before_fetch_barrier',5:'after_fetch_barrier',6:'before_fetch',7:'after_fetch_before_items',8:'after_items',12:'epoch',13:'grid',14:'full_rows',15:'chunks',16:'wait_call_ns_sum',17:'wait_call_count',18:'blocked_call_count',19:'attention_item_ns_sum',20:'attention_item_count',21:'lru_item_ns_sum',22:'lru_item_count',26:'last_attention_start'}), indent=2) + '\n')
print((p / 'manifest.json').read_text())
