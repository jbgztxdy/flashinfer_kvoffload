"""One-time private candidate, no production mutation."""
from pathlib import Path
import difflib,hashlib,json,shutil
p=Path(__file__).resolve().parent;root=p.parents[2]
base=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_text();assert hashlib.sha256(base.encode()).hexdigest()=='cab508dcb6e83b7a953d94a8e54e30e0aaf60bb1d7ba98480fc66893edcf4384'
assert not(p/'candidate.cuh').exists()
s=base.replace('PVFeature','FetchAssist').replace('pv_feature_','fetch_assist_')
a=s.index('template <typename Q, typename KV, typename O, typename Id>\nstruct FetchAssistTraits')
s=s[:a]+(p/'assist_helpers.cuh').read_text()+'\n'+s[a:]
s=s.replace('struct FetchAssistTraits : JointQKTraits<Q, KV, O, Id> {};','''struct FetchAssistTraits : JointQKTraits<Q, KV, O, Id> {
 struct SharedStorage : JointQKTraits<Q,KV,O,Id>::SharedStorage {alignas(128) uint8_t service_work[12288];};
};''')
# Private128loader version preserves original cache payload operations and epoch.
a=base.index('template <typename KTraits, typename Params>\n__device__ __forceinline__ void fetch_stream_source_pages');b=base.index('template <typename Q, typename KV, typename O, typename Id>\nstruct StreamSourceTraits',a)
f=base[a:b].replace('fetch_stream_source_pages','assist_fetch_pages').replace('const uint32_t tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;','const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;')
f=f.replace('KTraits::NUM_THREADS','128').replace('__syncthreads();','assist_loader_sync();').replace('  FAR_EMU_SYNC();','  assist_loader_sync();')
f=f.replace('  FAR_EMU_INIT(2);','') # initialized once before role split
old='    FAR_EMU_WAIT(2, BUFFERS == 2 ? k % 2 : 0);'
new='''#ifdef FLASHINFER_FAR_EMU
    if (::flashinfer::far_emu::config.enabled) {
      assist_loader_sync();
      if(tid==0) {auto c=::flashinfer::far_emu::cta_slot();auto stage=BUFFERS==2?k%2:0;
        ::flashinfer::far_emu::wait_until(*(volatile unsigned long long*)&::flashinfer::far_emu::pending_ready[c][stage]);
        ::flashinfer::far_emu::pending_ready[c][stage]=0;}
    }
#endif'''
assert old in f;f=f.replace(old,new)
# __syncthreads_and cannot be reached by128 of256threads. Equivalent128lane AND.
f=f.replace('  auto issue =', '  __shared__ uint32_t producer_ready[4];\n  auto issue =')
old='''      const bool defer = params.stage && !__syncthreads_and(ready) &&
                         deferred_count < kHostBatchMaxDeferred;'''
new='''      bool warp_ready=__all_sync(0xffffffffu,ready);
      if(threadIdx.x==0)producer_ready[threadIdx.y]=warp_ready;
      assist_loader_sync();
      const bool all_ready=producer_ready[0]&&producer_ready[1]&&producer_ready[2]&&producer_ready[3];
      const bool defer=params.stage && !all_ready && deferred_count<kHostBatchMaxDeferred;'''
assert old in f;f=f.replace(old,new)
a=s.index('template <typename Q, typename KV, typename O, typename Id>\nstruct FetchAssistTraits');s=s[:a]+f+s[a:]
# Row setup is only leader metadata; original barrier broadcasts normal state.
a=s.index('void fetch_assist_attend_joint_host_tiles(');b=s.index('template <typename KTraits, typename Params, bool ROW_MERGE>\n__device__ __forceinline__ void fetch_assist_run_joint_host_item',a)
t=s[a:b];old='  int hi = ceil_div(kv_len, 32u) - 1;';new=old+'''
  const int initial_hi=hi;
  const uint32_t assist_tid=(threadIdx.z*blockDim.y+threadIdx.y)*blockDim.x+threadIdx.x;
  if(assist_tid==0)assist_begin_row(params,row,index_base,kv_len);''';assert t.count(old)==1;t=t.replace(old,new)
old='''    cp_async::wait_group<0>();
    __syncthreads();
    fetch_assist_compute_qk_pair<KTraits>(storage, hi%2, lo%2, qpe, scores, scores+1);'''
new='''    cp_async::wait_group<0>();
    if(assist_tid==0)assist_take(params,const_cast<uint32_t*>(scratch),row,(initial_hi-hi)/2);
    __syncthreads();
    if(scratch[kHostBatchScratchClaimed]) {
      float* src=assist_scores(params,blockIdx.y,scratch[kHostBatchScratchClaimed]-1);
#pragma unroll
      for(uint32_t j=0;j<8;++j){scores[0][j]=src[assist_tid*8+j];scores[1][j]=src[2048+assist_tid*8+j];}
    }else fetch_assist_compute_qk_pair<KTraits>(storage, hi%2, lo%2, qpe, scores, scores+1);'''
assert old in t;t=t.replace(old,new)
old='''    fetch_assist_update_joint_host_mdo<KTraits>(&storage, variant, scores, scores+1, o, m, d);'''
new=old+'''
    // First jointMDO barrier guarantees all score loads finished before slot reuse.
    if(assist_tid==0 && scratch[kHostBatchScratchClaimed])atomicExch(&assist_controls(params)[blockIdx.y].lease[scratch[kHostBatchScratchClaimed]-1],0u);'''
assert t.count(old)==1;t=t.replace(old,new);s=s[:a]+t+s[b:]
# Kernel initialization, then original prepare barrier publishes controls.
a=s.index('__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLAFetchAssistKernel');b=s.index('// partial_capacity bounds',a);t=s[a:b]
old='  const uint32_t epoch = params.state[kHostBatchEpoch];';new=old+'''
  if(tid==0) {auto* c=assist_controls(params)+blockIdx.y;c->gen=0;c->lease[0]=c->lease[1]=0;c->produced=c->consumed=c->canceled=c->waited=0;
    if(blockIdx.y==0)c->helper_done=c->fetch_done=0;}
''';assert t.count(old)==1;t=t.replace(old,new)
old='''  if (unique > 0 && from_back && !host_batch_writes_back(params)) fetch_stream_source_pages<KTraits>(params, smem, epoch, unique, fetch_rows);'''
new='''  if (unique > 0 && from_back && !host_batch_writes_back(params)) {
    __shared__ volatile uint32_t producer_finished;
    if(tid==0)producer_finished=0;
    FAR_EMU_INIT(2);FAR_EMU_SYNC();
    if(threadIdx.z==0) {
      assist_fetch_pages<KTraits>(params,smem,epoch,unique,fetch_rows);
      if(tid==0){atomicAdd(&assist_controls(params)->fetch_done,1u);__threadfence_block();producer_finished=1;}
    }else assist_math_service<KTraits>(params,smem,&producer_finished);
    __syncthreads();
  }''';assert old in t;t=t.replace(old,new);s=s[:a]+t+s[b:]
s=s.replace('static_assert(fetch_assist_smem == 147968);','static_assert(fetch_assist_smem == 160256);')
s=s.replace('const bool fetch_assist = params.stage', 'const bool fetch_assist = num_sms<=128 && max_kv_len<=8192 && params.num_rows<=65535 && params.stage')
(p/'candidate.cuh').write_text(s)
for v,value in [('base',base),('candidate',s)]:
 tree=p/('include_'+v);shutil.copytree(root/'include',tree);(tree/'flashinfer/attention/mla_host_batch.cuh').write_text(value)
(p/'candidate.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='production',tofile='candidate',n=0)))
for name,origin in [('compile.py','diagnose_fetch_sm_v2_20261103'),('resources_gpu3.py','diagnose_fetch_sm_v2_20261103'),('experiment.json','diagnose_fetch_sm_v2_20261103'),('plans.cpp','diagnose_fetch_sm_v2_20261103')]:shutil.copy2(p.parent/origin/name,p/name)
(p/'.gitignore').write_text('include_*\njit_*\n__pycache__/\nplans.so\n!*.log\n')
(p/'manifest.json').write_text(json.dumps(dict(production_sha256=hashlib.sha256(base.encode()).hexdigest(),candidate_sha256=hashlib.sha256(s.encode()).hexdigest(),math_sha256=hashlib.sha256((root/'include/flashinfer/attention/mla.cuh').read_bytes()).hexdigest(),far_sha256=hashlib.sha256((root/'include/flashinfer/attention/far_link_emu.cuh').read_bytes()).hexdigest(),mechanism='Only fetchCTA4loader+4QK, original worker layout and compute order, pull-based generation-tagged boundedscore exchange',lookahead_pairs=4),indent=2)+'\n')
