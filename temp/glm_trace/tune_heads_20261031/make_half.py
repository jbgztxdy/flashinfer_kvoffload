"""Try two resident128-thread CTAs/SM by splitting64 heads into two32-head items.

Same single cooperative kernel, no head merge: the two CTAs write disjoint heads.
Original64-head traits and helpers fold to their original constants. Independent
include/JIT only; production and the pending cached-ID inputs are never changed.
"""
import hashlib
import json
import shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
base=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_text()
assert hashlib.sha256(base.encode()).hexdigest()=='52919c286398d4668933a772c646bdccd46273a53aafeb8d5c1b14e2d9bff2c5'
mla=(root/'include/flashinfer/attention/mla.cuh').read_text()
new_mla=mla.replace('(warpgroup_idx + mma_q * 2) * 16 + warp_idx_in_wg * 4',
    '(warpgroup_idx + mma_q * 2) * (KTraits::CTA_TILE_Q / 4) + warp_idx_in_wg * 4')
assert new_mla!=mla
assert new_mla.count('32 * mma_q + warpgroup_idx * 16 + warp_idx_in_wg * 4 + lane_idx / 8')==2
new_mla=new_mla.replace('32 * mma_q + warpgroup_idx * 16 + warp_idx_in_wg * 4 + lane_idx / 8',
    '(KTraits::CTA_TILE_Q / 2) * mma_q + warpgroup_idx * (KTraits::CTA_TILE_Q / 4) + warp_idx_in_wg * 4 + lane_idx / 8')
s=base
old='constexpr uint32_t kHostBatchFetchRows = 64;'
assert s.count(old)==1
s=s.replace(old,'''// Large launches only.128 threads cover32 heads with two warp rows in each
// feature/KV group. The query and KV buffers fit two resident CTAs on SM80.
template <typename Q,typename KV,typename O,typename Id>
struct HalfHeadHostTraits : KernelTraits<false,1,true,512,64,32,32,Q,KV,O,Id> {
  static constexpr uint32_t NUM_THREADS=128;
};
template <typename Params>
struct HalfHeadHostParams : WholeRowHostBatchParams<Params> {
  static constexpr uint32_t num_head_tiles=2;
};

'''+old)
old='  constexpr uint32_t CHUNK_VECS = kHostBatchFetchRows * VECS_PER_ROW;'
assert s.count(old)==1
s=s.replace(old,'  constexpr uint32_t CHUNK_VECS = (KTraits::CTA_TILE_Q == 32 ? 32u : kHostBatchFetchRows) * VECS_PER_ROW;')
old='constexpr uint32_t kHostBatchTileRows = KTraits::NUM_MMA_KV == 1 ? 1 : KTraits::NUM_MMA_KV / 2;'
assert s.count(old)==1
s=s.replace(old,'''constexpr uint32_t kHostBatchTileRows = KTraits::CTA_TILE_Q == 32
    ? KTraits::CTA_TILE_KV / 16 : (KTraits::NUM_MMA_KV == 1 ? 1 : KTraits::NUM_MMA_KV / 2);''')
old='''  return KTraits::NUM_MMA_KV == 1 ? row : row + (threadIdx.z + j * 2) * 16;'''
assert s.count(old)==1
s=s.replace(old,'''  if constexpr (KTraits::CTA_TILE_Q == 32) return row + (threadIdx.z + j * 2) * 8;
  else return KTraits::NUM_MMA_KV == 1 ? row : row + (threadIdx.z + j * 2) * 16;''')
old='  const uint32_t fetch_rows = host_batch_fetch_rows<ROW_MERGE>(params, unique);'
assert s.count(old)==1
s=s.replace(old,'''  const uint32_t fetch_rows = KTraits::CTA_TILE_Q == 32
      ? (unique < params.fetch_ctas * 32u ? 16u : 32u)
      : host_batch_fetch_rows<ROW_MERGE>(params, unique);''')
old='    DISPATCH_SMEM_CONFIG(smem_limit_per_sm, NUM_STAGES, CTA_TILE_KV, QK_SHARD, {'
assert s.count(old)==1
s=s.replace(old,'''    using HalfTraits=HalfHeadHostTraits<DTypeQ,DTypeKV,DTypeO,IdType>;
    using HalfParams=HalfHeadHostParams<Params>;
    constexpr size_t half_smem=sizeof(typename HalfTraits::SharedStorage);
    // The original workspace reserves row-completion words even for the whole-row
    // path. Those unused words give its item-claim region room for two head tiles:
    // capacity=2*rows+partial_capacity+num_pools, required=2*rows+num_pools.
    // ROW_MERGE=false and whole_rows=true never access row-completion/partial data.
    const bool half_heads=params.stage && !params.host_split && params.num_head_tiles==1 &&
        params.num_rows>=4u*num_sms && params.fetch_ctas>=1 && params.fetch_ctas<num_sms &&
        smem_limit_per_sm>=2*(half_smem+2048) && smem_limit_per_sm<200000 &&
        static_cast<uint64_t>(params.staging_capacity)>=static_cast<uint64_t>(params.num_rows)*max_kv_len;
    if (half_heads) {
      auto half_kernel=BatchMLAHostBatchKernel<HalfTraits,HalfParams,false>;
      FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(half_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,half_smem));
      int half_blocks=0;
      FLASHINFER_CUDA_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&half_blocks,half_kernel,128,half_smem));
      if (half_blocks>=2) {
        const uint32_t half_grid=half_blocks*num_sms;
        if (params.pool_capacity>kHostBatchMaxPool || 2ull*params.pool_capacity*sizeof(uint32_t)>half_smem)
          return cudaErrorInvalidConfiguration;
        static_assert(sizeof(HalfParams)==sizeof(Params));
        static_assert(alignof(HalfParams)==alignof(Params));
        params.full_rows=params.num_rows;
        params.partial_capacity=partial_capacity;
        params.num_chunks=1;
        const uint32_t kv_len=max(max_kv_len,1u);
        params.prepare_parts=min(ceil_div(half_grid,max(params.num_rows,1u)),ceil_div(kv_len,128u));
        params.prepare_segment=round_up(ceil_div(kv_len,params.prepare_parts),128u);
        void* args[]={(void*)&params};
        FLASHINFER_CUDA_CALL(cudaLaunchCooperativeKernel((void*)half_kernel,dim3(1,half_grid),dim3(32,2,2),args,half_smem,stream));
        return cudaSuccess;
      }
    }
'''+old)
for v,header,math in [('base',base,mla),('half_heads',s,new_mla)]:
    assert not (p/(v+'.cuh')).exists()
    (p/(v+'.cuh')).write_text(header)
    (p/(v+'_mla.cuh')).write_text(math)
    shutil.copytree(root/'include',p/('include_'+v))
    for name,content in [('mla_host_batch.cuh',header),('mla.cuh',math)]:
        (p/('include_'+v)/'flashinfer/attention'/name).write_text(content)
q=p.parent/'tune_wait_20261031'
for name in ['.gitignore','compile.py','resources.py','resources_gpu3.py','experiment.json','plans.cpp','bench.py','bench_gpu3_native.py','test_candidate.py','candidatepaths.py','screen.py']:
    text=(q/name).read_text()
    if name in ['test_candidate.py','candidatepaths.py']:
        text=text.replace("'cached_ids'","'cached_ids','half_heads'")
    if name=='screen.py':text=text.replace("'cached_ids']","'cached_ids', 'half_heads']")
    (p/name).write_text(text)
# Bench source provenance must capture the executed private math header too.
for name in ['bench.py','bench_gpu3_native.py']:
    f=p/name;text=f.read_text()
    key='    files["include/flashinfer/attention/mla_host_batch.cuh"]=digest(VARIANT_INCLUDE/"flashinfer/attention/mla_host_batch.cuh")'
    assert text.count(key)==1
    text=text.replace(key,key+'\n    files["include/flashinfer/attention/mla.cuh"]=digest(VARIANT_INCLUDE/"flashinfer/attention/mla.cuh")')
    text=text.replace('if name=="include/flashinfer/attention/mla_host_batch.cuh" else',
        'if name=="include/flashinfer/attention/mla_host_batch.cuh" else (VARIANT_INCLUDE/name.removeprefix("include/")) if name=="include/flashinfer/attention/mla.cuh" else')
    f.write_text(text)
(p/'manifest.json').write_text(json.dumps(dict(base_sha256=hashlib.sha256(base.encode()).hexdigest(),candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),
    base_mla_sha256=hashlib.sha256(mla.encode()).hexdigest(),candidate_mla_sha256=hashlib.sha256(new_mla.encode()).hexdigest(),
    mechanism='Two32-head items per whole row;128threads/CTA,1stage32KV,cooperative grid from actual occupancy. Try twoCTAs/SM; fetch32rows/double buffer. Same heads/math/cache/far traffic/simulator, no head merge. Original64-head constants unchanged; require compiled-code equivalence.',
    expected_dynamic_shared_bytes=73984,workspace_claim_capacity='2*rows+partial_capacity+num_pools already allocated; whole/ROW_MERGE=false skips completion and partial data',candidate_adopted=False,goal_complete=False),indent=2)+'\n')
print((p/'manifest.json').read_text())
