"""One-time isolated full all-warp candidate, frozen source only."""
from pathlib import Path
import hashlib,json,shutil,difflib
p=Path(__file__).resolve().parent;root=p.parents[2];base=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_text()
assert hashlib.sha256(base.encode()).hexdigest()=='cab508dcb6e83b7a953d94a8e54e30e0aaf60bb1d7ba98480fc66893edcf4384';assert not(p/'candidate.cuh').exists()
s=base.replace('PVFeature','Uniform').replace('pv_feature_','uniform_')
a=s.index('template <typename Q, typename KV, typename O, typename Id>\nstruct UniformTraits');b=s.index('// Two different KV tiles',a)
s=s[:a]+(p/'service.cuh').read_text()+'\n'+s[b:]
# Forward declarations are needed by the copied PV helper only for the later
# optional future scheduling. Service rounds are placed in the paired QK innerloop.
a=s.index('void uniform_qk_pair_fragments(');b=s.index('// Each lane owns',a);t=s[a:b]
# Thread the existing context by reference rather than duplicating service state.
t=t.replace('float (*sa)[8], float (*sb)[8]) {','float (*sa)[8], float (*sb)[8], typename KTraits::SharedStorage& storage, const auto& params) {')
# C++17 doesn't allow abbreviated function templates: explicit Params.
t=t.replace('template <bool INIT, typename KTraits, uint32_t D, uint32_t Q_STRIDE,','template <bool INIT, typename KTraits, typename Params, uint32_t D, uint32_t Q_STRIDE,')
t=t.replace('const auto& params','const Params& params')
old='''    qo=q.template advance_offset_by_column<2>(qo,dim);'''
new='''    if(dim==0)uniform_issue_round<0,KTraits>(params,storage);
    if(dim==12)uniform_issue_round<1,KTraits>(params,storage);
    if(dim==24)uniform_issue_round<2,KTraits>(params,storage);
'''+old
assert old in t;t=t.replace(old,new);s=s[:a]+t+s[b:]
a=s.index('void uniform_compute_qk_pair(');b=s.index('// Two native32-row',a);t=s[a:b]
t=t.replace('float (*sa)[8],float (*sb)[8]) {','float (*sa)[8],float (*sb)[8],const auto& params) {')
t=t.replace('template <typename KTraits>','template <typename KTraits,typename Params>').replace('const auto& params','const Params& params')
t=t.replace('uniform_qk_pair_fragments<false,KTraits,KTraits::NUM_MMA_D_CKV,','uniform_qk_pair_fragments<false,KTraits,Params,KTraits::NUM_MMA_D_CKV,').replace('(qn,kna,knb,sa,sb);','(qn,kna,knb,sa,sb,storage,params);')
assert 'storage,params' in t;s=s[:a]+t+s[b:]
a=s.index('void uniform_attend_joint_host_tiles(');b=s.index('// partial_capacity bounds',a);t=s[a:b]
t=t.replace('wait_host_batch_tile<KTraits>(params, scratch,','uniform_wait_tile<KTraits>(params, storage, scratch,')
old='''    cp_async::wait_group<0>();
    __syncthreads();
    uniform_compute_qk_pair<KTraits>(storage, hi%2, lo%2, qpe, scores, scores+1);'''
new='''    uniform_progress<KTraits>(params,storage);
    uniform_compute_qk_pair<KTraits>(storage, hi%2, lo%2, qpe, scores, scores+1,params);'''
assert t.count(old)==1;t=t.replace(old,new)
old='''    load_q<KTraits>(&storage, params.q_nope + row * params.q_nope_stride_n,
                    params.q_pe + row * params.q_pe_stride_n, params.q_nope_stride_n,
                    params.q_nope_stride_h, params.q_pe_stride_n, params.q_pe_stride_h, 1,
                    head_tile * CTA_TILE_Q, params.num_heads);'''
new='''    uniform_load_q<KTraits>(&storage,params.q_nope+row*params.q_nope_stride_n,params.q_nope_stride_n,params.q_nope_stride_h,params.num_heads);'''
assert old in t;t=t.replace(old,new)
# All108CTAs own exact disjoint rows; LRU completed before service starts.
a0=t.index('template <typename KTraits, typename Params, bool ROW_MERGE>\n__device__ __forceinline__ void uniform_run_joint_host_items');b0=t.index('template <typename KTraits, typename Params, bool ROW_MERGE>\n__global__',a0)
items='''template <typename KTraits, typename Params, bool ROW_MERGE>
__device__ __forceinline__ void uniform_run_joint_host_items(const Params& params,typename KTraits::SharedStorage& storage,uint8_t* smem,uint32_t* scratch,bool,uint32_t) {
 float scores[2][8],output[KTraits::NUM_MMA_D_CKV/2][8],m[2],d[2];
 for(uint32_t row=blockIdx.y;row<params.num_rows;row+=gridDim.y)
   uniform_run_joint_host_item<KTraits,Params,false>(params,storage,smem,scratch,row,scores,output,m,d);
 uniform_drain<KTraits>(params,storage);
 const uint32_t tid=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x;
 if(tid==0){auto* diag=uniform_diag(params);diag[0]=storage.service.produced;diag[1]=storage.service.rounds;diag[2]=storage.service.wait_loops;diag[3]=storage.service.rows;diag[4]=sizeof(typename KTraits::SharedStorage);}
}
'''
t=t[:a0]+items+t[b0:]
old='''  const uint32_t fetch_rows = host_batch_fetch_rows<ROW_MERGE>(params, unique);'''
new='''  // Readyflags occupy existing unused whole-row partial backing, one per10pages.
  for(uint32_t flag=blockIdx.y*KTraits::NUM_THREADS+tid;flag<ceil_div(unique,10u);flag+=gridDim.y*KTraits::NUM_THREADS)uniform_flags(params)[flag]=0;
  if(unique>0){
    for(uint32_t pool=blockIdx.y;pool<params.num_pools;pool+=gridDim.y){run_host_batch_lru<KTraits>(params,smem,pool,epoch,unique);__syncthreads();}
    cg::this_grid().sync();
  }
  const uint32_t fetch_rows=10;'''
assert t.count(old)==1;t=t.replace(old,new)
old='''  if (unique > 0 && from_back && !host_batch_writes_back(params)) fetch_stream_source_pages<KTraits>(params, smem, epoch, unique, fetch_rows);'''
new='''  uniform_initialize<KTraits>(params,storage,unique,epoch);'''
assert t.count(old)==1;t=t.replace(old,new);s=s[:a]+t+s[b:]
s=s.replace('static_assert(uniform_smem == 147968);','static_assert(uniform_smem <= 163000);')
s=s.replace('const bool uniform = params.stage','const bool uniform = num_sms<=128 && params.num_rows<=4096 && max_kv_len<=8192 && params.stage')
s=s.replace('smem_limit_per_sm >= uniform_smem + 4096','smem_limit_per_sm >= uniform_smem')
s=s.replace('const uint32_t uniform_grid = num_sms;','const uint32_t uniform_grid = num_sms;\n        if((ceil_div(params.staging_capacity,10u)+128*16ull)*4 > static_cast<uint64_t>(partial_capacity)*64*512*sizeof(DTypeO))return cudaErrorInvalidConfiguration;')
for v,value in [('base',base),('candidate',s)]:
 tree=p/('include_'+v);shutil.copytree(root/'include',tree);(tree/'flashinfer/attention/mla_host_batch.cuh').write_text(value)
(p/'candidate.cuh').write_text(s);(p/'base.cuh').write_text(base);(p/'candidate.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='production',tofile='uniform',n=0)))
for name in ['compile.py','resources_gpu3.py','experiment.json','plans.cpp']:shutil.copy2(p.parent/'tune_fetch_qk_assist_20261103'/name,p/name)
(p/'.gitignore').write_text('include_*\njit_*\n__pycache__/\nplans.so\n!*.log\n')
(p/'manifest.json').write_text(json.dumps(dict(production_sha256=hashlib.sha256(base.encode()).hexdigest(),candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),math_sha256=hashlib.sha256((root/'include/flashinfer/attention/mla.cuh').read_bytes()).hexdigest(),far_sha256=hashlib.sha256((root/'include/flashinfer/attention/far_link_emu.cuh').read_bytes()).hexdigest(),mechanism='AllCTA8warps2mathstages,compactQPE,2x10rowprivatefarstages,staticuniquechunkownership,distributedcopyroundsamidpairedQK,noQKscorehandoff'),indent=2)+'\n')
