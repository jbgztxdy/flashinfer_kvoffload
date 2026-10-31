"""Generate a spatial four-fetch/four-compute-warp cooperative kernel in isolation."""
import hashlib
import json
from pathlib import Path
import re
import shutil
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
base=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_text()
canonical=(root/'include/flashinfer/attention/mla.cuh').read_text()
assert sha(base)=='b6eea0957bc5f972e5645c1f8e90b14217601ba11ef9e57f3ead51159d66e72a'
assert sha(canonical)=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
math=(p.parent/'tune_heads_20261031/half_heads_mla.cuh').read_text()
assert sha(math)=='e5968e7605bb657de30e5600c4a8fc09b977a3e2e7163f64e46718de519289a3'
helper='''template <typename KTraits, typename = void>
struct MLAWarpRole {
  static constexpr bool enabled = false;
  __device__ __forceinline__ static uint32_t group() { return threadIdx.z; }
  __device__ __forceinline__ static void sync() { __syncthreads(); }
};
template <typename KTraits>
struct MLAWarpRole<KTraits, std::void_t<decltype(KTraits::WARP_GROUP_OFFSET)>> {
  static constexpr bool enabled = true;
  __device__ __forceinline__ static uint32_t group() { return threadIdx.z - KTraits::WARP_GROUP_OFFSET; }
  __device__ __forceinline__ static void sync() { asm volatile("bar.sync 1, 128;" ::: "memory"); }
};

'''
key='template <typename KTraits>\n__device__ __forceinline__ void init_states_'
assert math.count(key)==1
math=math.replace(key,helper+key)
math=math.replace('#include "mla_params.cuh"','#include <type_traits>\n#include "mla_params.cuh"')
# Only these native attention helpers are called by the role's compute subgroup. Leave all
# other device/legacy overlap/planner code intact. The default helper reduces to the original.
names=['load_q','compute_qk_','logits_mask_','update_mdo_states_','compute_mla_qk','compute_mla_pv','normalize_d_','write_o']
def transform_function(s,name,fn):
    match=re.search(r'\bvoid '+re.escape(name)+r'\(',s)
    assert match,name
    first=s.index('{',match.end());depth=1;end=first+1
    while depth:
        depth+=(s[end]=='{')-(s[end]=='}');end+=1
    body=s[first:end];return s[:first]+fn(body)+s[end:]
for name in names:
    math=transform_function(math,name,lambda x:x.replace('threadIdx.z','MLAWarpRole<KTraits>::group()').replace('__syncthreads();','MLAWarpRole<KTraits>::sync();'))
s=base
key='constexpr uint32_t kHostBatchFetchRows = 64;'
s=s.replace(key,'''template <typename Q, typename KV, typename O, typename Id>
struct WarpRoleComputeTraits : KernelTraits<false, 2, true, 512, 64, 32, 32, Q, KV, O, Id> {
  static constexpr uint32_t NUM_THREADS = 128;
  static constexpr uint32_t WARP_GROUP_OFFSET = 2;
};
template <typename Params>
struct WarpRoleHostParams : CachedSharedIdsHostBatchParams<Params> {
  static constexpr uint32_t num_head_tiles = 2;
};

'''+key)
s=s.replace('constexpr uint32_t kHostBatchTileRows = KTraits::NUM_MMA_KV == 1 ? 1 : KTraits::NUM_MMA_KV / 2;',
'''constexpr uint32_t kHostBatchTileRows = MLAWarpRole<KTraits>::enabled
    ? KTraits::CTA_TILE_KV / 16 : (KTraits::NUM_MMA_KV == 1 ? 1 : KTraits::NUM_MMA_KV / 2);''')
s=s.replace('  return KTraits::NUM_MMA_KV == 1 ? row : row + (threadIdx.z + j * 2) * 16;',
'''  if constexpr (MLAWarpRole<KTraits>::enabled)
    return row + (MLAWarpRole<KTraits>::group() + j * 2) * 8;
  else return KTraits::NUM_MMA_KV == 1 ? row : row + (threadIdx.z + j * 2) * 16;''')
# Full-CTA prepare/LRU remain untouched; barriers in the item and tile loop belong to compute.
for name in ['wait_host_batch_tile','load_host_batch_kv','writeback_host_batch_tile','run_host_batch_item','attend_host_batch_tiles']:
    s=transform_function(s,name,lambda x:x.replace('threadIdx.z','MLAWarpRole<KTraits>::group()').replace('__syncthreads();','MLAWarpRole<KTraits>::sync();'))
# Derive the new full-CTA lifecycle directly from the frozen production kernel.
a=base.index('template <typename KTraits, typename Params, bool ROW_MERGE>\n__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLAHostBatchKernel(')
b=base.index('// The last CTA to leave resets',a)
end=base.index('\n// ',b)
kernel=base[a:end]
assert kernel.rstrip().endswith('}')
cut=kernel.index('  const uint32_t fetch_rows =')
tail=kernel.index('  // The last CTA to leave resets')
prefix=kernel[:cut]
prefix=prefix.replace('template <typename KTraits, typename Params, bool ROW_MERGE>','template <typename KTraits, typename ComputeTraits, typename Params>')
prefix=prefix.replace('BatchMLAHostBatchKernel(', 'BatchMLAWarpRoleKernel(')
prefix=prefix.replace('  uint8_t* smem = host_batch_smem;', '  constexpr bool ROW_MERGE = false;\n  uint8_t* smem = host_batch_smem;')
prefix=prefix.replace('  auto& storage = reinterpret_cast<typename KTraits::SharedStorage&>(*smem);\n','')
role='''  if (unique > 0) {
#pragma unroll 1
    for (uint32_t pool = blockIdx.y; pool < params.num_pools; pool += gridDim.y) {
      run_host_batch_lru<KTraits>(params, smem, pool, epoch, unique);
      __syncthreads();
    }
  }
  // All pools are selected and full-CTA scratch reuse has ended before either subgroup runs.
  cg::this_grid().sync();
  if (tid == 0) {
    scratch[kHostBatchScratchEpoch] = epoch;
    scratch[kHostBatchScratchFetchRows] = 16;
  }
  __syncthreads();
  constexpr size_t compute_bytes = sizeof(typename ComputeTraits::SharedStorage);
  static_assert(compute_bytes == 110848);
  static_assert(compute_bytes + 2 * 16 * 576 * 2 <= sizeof(typename KTraits::SharedStorage));
  if (threadIdx.z < 2) {
    if (unique > 0) fetch_warp_role_pages<KTraits>(params, smem + compute_bytes, unique);
  } else {
    auto& storage = *reinterpret_cast<typename ComputeTraits::SharedStorage*>(smem);
    compute_warp_role_items<ComputeTraits>(params, storage, smem, scratch);
  }
  __syncthreads();
  cg::this_grid().sync();

'''
kernel=prefix+role+kernel[tail:]
s=s[:a]+(p/'warp_roles.inc').read_text()+'\n'+kernel+'\n'+s[a:]
key='    DISPATCH_SMEM_CONFIG(smem_limit_per_sm, NUM_STAGES, CTA_TILE_KV, QK_SHARD, {'
assert s.count(key)==1
launch='''    using FullTraits = KernelTraits<false, 2, true, 512, 64, 64, 32, DTypeQ, DTypeKV, DTypeO, IdType>;
    using ComputeTraits = WarpRoleComputeTraits<DTypeQ, DTypeKV, DTypeO, IdType>;
    using RoleParams = WarpRoleHostParams<Params>;
    constexpr size_t role_smem = sizeof(typename FullTraits::SharedStorage);
    static_assert(role_smem == 147968);
    const bool warp_roles = params.stage && !params.host_split && params.num_head_tiles == 1 &&
        params.pool_capacity > 0 && params.num_rows >= 4u * num_sms && params.fetch_ctas >= 1 &&
        params.fetch_ctas < num_sms && smem_limit_per_sm >= role_smem + 4096 &&
        smem_limit_per_sm < 200000 && static_cast<uint64_t>(params.staging_capacity) >=
            static_cast<uint64_t>(params.num_rows) * max_kv_len;
    if (warp_roles) {
      auto role_kernel = BatchMLAWarpRoleKernel<FullTraits, ComputeTraits, RoleParams>;
      FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(role_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, role_smem));
      int role_blocks = 0;
      FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&role_blocks, role_kernel, 256, role_smem));
      if (role_blocks >= 1) {
        const uint32_t role_grid = num_sms;
        if (params.pool_capacity > kHostBatchMaxPool || 2ull * params.pool_capacity * sizeof(uint32_t) > role_smem)
          return cudaErrorInvalidConfiguration;
        static_assert(sizeof(RoleParams) == sizeof(Params));
        static_assert(alignof(RoleParams) == alignof(Params));
        params.full_rows = params.num_rows;
        params.partial_capacity = partial_capacity;
        params.num_chunks = 1;
        const uint32_t length = max(max_kv_len, 1u);
        params.prepare_parts = min(ceil_div(role_grid, max(params.num_rows, 1u)), ceil_div(length, 256u));
        params.prepare_segment = round_up(ceil_div(length, params.prepare_parts), 256u);
        void* args[] = {(void*)&params};
        FLASHINFER_CUDA_CALL(cudaLaunchCooperativeKernel((void*)role_kernel, dim3(1, role_grid), dim3(32, 2, 4), args, role_smem, stream));
        return cudaSuccess;
      }
    }
'''
s=s.replace(key,launch+key)
# Freeze before building. An existing differing snapshot/tree is an error, not an overwrite.
for variant,host,mat in [('base',base,canonical),('warp_roles',s,math)]:
    for name,value in [(variant+'.cuh',host),(variant+'_mla.cuh',mat)]:
        f=p/name
        if f.exists():assert f.read_text()==value,name
        else:f.write_text(value)
    for suffix in ['', '_normal']:
        dest=p/('include_'+variant+suffix)
        if not dest.exists():
            shutil.copytree(root/'include',dest)
            (dest/'flashinfer/attention/mla_host_batch.cuh').write_text(host)
            (dest/'flashinfer/attention/mla.cuh').write_text(mat)
        assert sha((dest/'flashinfer/attention/mla_host_batch.cuh').read_text())==sha(host)
        assert sha((dest/'flashinfer/attention/mla.cuh').read_text())==sha(mat)
manifest=dict(base_sha256=sha(base),canonical_math_sha256=sha(canonical),candidate_sha256=sha(s),candidate_math_sha256=sha(math),
    mechanism='Spatial4fetch/4computewarps in256threadCTA; physicalz0/1fetch,virtualz0/1=physicalz2/3compute; namedbarriers2/1 each128. Full256threadprepare/LRU then grid barrier; full256 finalgrid/exit. Native32head/sharded32KV/2stageattention. Two16rowfetchbuffersafter110848Bcompute; total147712 within147968. OriginalFARwaiter/hooks/bytes unchanged.',
    fetch_policy='Every108CTAfetchsubgroup walks16rowchunks in round-robin;108*16*2=3456 rows in flight versus base24*64*2=3072. No budget sweep. API fetch_ctas remains24 in paired screen; policy is part of algorithm change.',
    candidate_adopted=False,goal_complete=False)
(p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
