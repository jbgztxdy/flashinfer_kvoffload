"""Native64KV/one-stage tile: same shared budget, fewer online softmax updates."""
import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
base=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_text()
math=(root/'include/flashinfer/attention/mla.cuh').read_text()
assert sha(base)=='b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
assert sha(math)=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
s=base.replace('constexpr uint32_t kHostBatchFetchRows = 64;', '''template <typename Params>
struct KV64HostBatchParams : CachedSharedIdsHostBatchParams<Params> {};

constexpr uint32_t kHostBatchFetchRows = 64;''',1)
key='    DISPATCH_SMEM_CONFIG(smem_limit_per_sm, NUM_STAGES, CTA_TILE_KV, QK_SHARD, {'
assert s.count(key)==1
s=s.replace(key,'''    using KV64Traits = KernelTraits<false, 1, true, 512, 64, 64, 64, DTypeQ, DTypeKV, DTypeO, IdType>;
    using KV64Params = KV64HostBatchParams<Params>;
    constexpr size_t kv64_smem = sizeof(typename KV64Traits::SharedStorage);
    static_assert(kv64_smem == 147968);
    const bool kv64 = params.stage && !params.host_split && params.num_head_tiles == 1 &&
        params.pool_capacity > 0 && params.num_rows >= 4u * num_sms && params.fetch_ctas >= 1 &&
        params.fetch_ctas < num_sms && smem_limit_per_sm >= kv64_smem + 4096 &&
        smem_limit_per_sm < 200000 && static_cast<uint64_t>(params.staging_capacity) >=
            static_cast<uint64_t>(params.num_rows) * max_kv_len;
    if (kv64) {
      auto kv64_kernel = BatchMLAHostBatchKernel<KV64Traits, KV64Params, false>;
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
'''+key,1)
for v,host in [('base',base),('kv64',s)]:
    for name,value in [(v+'.cuh',host),(v+'_mla.cuh',math)]:
        f=p/name
        if f.exists():assert f.read_text()==value
        else:f.write_text(value)
    for suffix in ['', '_normal']:
        d=p/('include_'+v+suffix)
        if not d.exists():
            shutil.copytree(root/'include',d)
            (d/'flashinfer/attention/mla_host_batch.cuh').write_text(host)
        assert sha((d/'flashinfer/attention/mla_host_batch.cuh').read_text())==sha(host)
        assert sha((d/'flashinfer/attention/mla.cuh').read_text())==sha(math)
m=dict(base_sha256=sha(base),candidate_sha256=sha(s),math_sha256=sha(math),
    mechanism='Original64head/256thread attention with native sharded64KV/1stage, same147968B as baseline32KV/2stage. Half as many online MDO updates/barriers, larger native Qfragment reuse, but one-stage load latency and extra score registers may offset savings. Original fetch64row double buffer/f24,metadata,LRU,SMbudget preserved.',
    distinction='Stage1/32KV allocation control was rejected in C; pairedQK grouped two32KV QK calculations but retained two separate MDO/PV updates. This uses one native64KV max/denominator update and PV region. No private math edits; FP32 online-reduction grouping differs, no bitwise claim.',
    candidate_adopted=False,goal_complete=False)
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
