# Purpose: Add globaltimer phase tracing to mla_host_batch.cuh in place (debug only; restore after).
# Usage: python3 temp/hb_regs/add_trace.py include/flashinfer/attention/mla_host_batch.cuh
import sys

p = sys.argv[1]
s = open(p).read()


def rep(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)


rep('''__device__ __forceinline__ uint32_t host_batch_ld_acquire(const uint32_t* address) {''', '''__device__ __forceinline__ uint64_t hb_now() {
  uint64_t t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
__device__ uint64_t hb_wait_ns[1024];

__device__ __forceinline__ uint32_t host_batch_ld_acquire(const uint32_t* address) {''')
rep('''      while (host_batch_ld_acquire(flag) != scratch[kHostBatchScratchEpoch] + 1) __nanosleep(64);
    }
  }
}''', '''      const uint64_t t = hb_now();
      while (host_batch_ld_acquire(flag) != scratch[kHostBatchScratchEpoch] + 1) __nanosleep(64);
      if (threadIdx.x == 0) hb_wait_ns[blockIdx.y] += hb_now() - t;
    }
  }
}''')
rep('''  const uint32_t epoch = params.state[kHostBatchEpoch];
''', '''  const uint32_t epoch = params.state[kHostBatchEpoch];
  uint64_t trace[6];
  trace[0] = hb_now();
  if (tid == 0) hb_wait_ns[blockIdx.y] = 0;
''')
rep('''  if (tid == 0) scratch[kHostBatchScratchEpoch] = epoch;
  __syncthreads();''', '''  if (tid == 0) scratch[kHostBatchScratchEpoch] = epoch;
  __syncthreads();
  trace[1] = hb_now();''')
rep('''#ifndef HB_SKIP_ATTENTION
  run_host_batch_items<KTraits>(params, storage, smem, scratch, fetcher);
#endif''', '''  trace[2] = hb_now();
#ifndef HB_SKIP_ATTENTION
  run_host_batch_items<KTraits>(params, storage, smem, scratch, fetcher);
#endif
  trace[3] = hb_now();''')
rep('''  // The last CTA to leave resets the shared counters for the next launch.
  __syncthreads();''', '''  trace[4] = hb_now();
  __syncthreads();
  if (tid == 0) {
    auto* out = reinterpret_cast<uint64_t*>(params.partial_lse) + 8 * blockIdx.y;
    for (int i = 0; i < 5; ++i) out[i] = trace[i];
    out[5] = hb_wait_ns[blockIdx.y];
    out[6] = fetcher;
  }
  // The last CTA to leave resets the shared counters for the next launch.
  __syncthreads();''')
open(p, 'w').write(s)
