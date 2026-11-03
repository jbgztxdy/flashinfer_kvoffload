"""Private diagnostics only: phase/wait trace and same-remap attention replay capacity."""
from pathlib import Path
import difflib,hashlib,json,shutil
p=Path(__file__).resolve().parent;root=p.parents[2]
base=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_text()
assert hashlib.sha256(base.encode()).hexdigest()=='cab508dcb6e83b7a953d94a8e54e30e0aaf60bb1d7ba98480fc66893edcf4384'
assert not(p/'trace.cuh').exists()
helper=r'''
// DIAGNOSTIC ONLY. Whole-row dispatch leaves partial_o idle. Trace occupies
// 32 epochs *128CTAs *40uint64 fields=1.31MiB of its existing56MiB backing.
__device__ __forceinline__ unsigned long long diag_now() {
  unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;
}
template<typename Params>
__device__ __forceinline__ unsigned long long* diag_record(const Params& params) {
  return reinterpret_cast<unsigned long long*>(params.partial_o)+
      ((params.state[kHostBatchEpoch]%32)*128+blockIdx.y)*40;
}
template<typename Params>
__device__ __forceinline__ void diag_mark(const Params& params,int k) {
  if(threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0)diag_record(params)[k]=diag_now();
}
template<typename Params>
__device__ __forceinline__ void diag_add(const Params& params,int k,unsigned long long n) {
  if(threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0)diag_record(params)[k]+=n;
}
'''
s=base.replace('PVFeature','TracePV').replace('pv_feature_','trace_pv_')
at=s.index('template <typename KTraits, typename Params>\n__device__ __forceinline__ void wait_host_batch_tile')
s=s[:at]+helper+s[at:]
# Only the new typed path calls this timed helper. All original math stays intact.
a=s.index('template <typename KTraits, typename Params>\n__device__ __forceinline__ void wait_host_batch_tile');b=s.index('// Rows of a KV tile',a)
wait=s[a:b].replace('void wait_host_batch_tile(','void diag_wait_host_batch_tile(')
wait=wait.replace('  if (threadIdx.y != 0 || threadIdx.z != 0) return;','  if (threadIdx.y != 0 || threadIdx.z != 0) return;\n  const uint32_t begin=static_cast<uint32_t>(diag_now());\n  bool blocked=false;')
wait=wait.replace('while (host_batch_ld_acquire(flag) != scratch[kHostBatchScratchEpoch] + 1) __nanosleep(64);','while (host_batch_ld_acquire(flag) != scratch[kHostBatchScratchEpoch] + 1) {blocked=true;__nanosleep(64);}')
end=wait.rfind('\n}')
wait=wait[:end]+'''
  const bool any=__any_sync(0xffffffff,blocked);
  if(threadIdx.x==0) {
    auto* out=diag_record(params);
    out[17]+=static_cast<uint32_t>(diag_now())-begin;
    out[18]+=1;out[19]+=any;
  }'''+wait[end:]
s=s[:b]+wait+s[b:]
# Clone fetch for the traced type, retaining original payload/cache/FAR operations.
a=s.index('template <typename KTraits, typename Params>\n__device__ __forceinline__ void fetch_stream_source_pages');b=s.index('template <typename Q, typename KV, typename O, typename Id>\nstruct StreamSourceTraits',a)
fetch=s[a:b].replace('fetch_stream_source_pages','diag_fetch_pages')
old='''      cp_async::wait_group<1>();
    } else {
      cp_async::wait_group<0>();
    }
    FAR_EMU_WAIT(2, BUFFERS == 2 ? k % 2 : 0);'''
new='''    }
    diag_mark(params,34);
    if (BUFFERS == 2 && next * rows < count) cp_async::wait_group<1>();
    else cp_async::wait_group<0>();
    if(tid==0) {auto* out=diag_record(params);out[26]+=diag_now()-out[34];}
    diag_mark(params,35);
    FAR_EMU_WAIT(2, BUFFERS == 2 ? k % 2 : 0);
    if(tid==0) {auto* out=diag_record(params);out[27]+=diag_now()-out[35];out[29]+=1;}'''
assert fetch.count(old)==1;fetch=fetch.replace(old,new)
old='''    // Fill the LRU cache from the same shared-memory copy'''
assert fetch.count(old)==1;fetch=fetch.replace(old,'''    if(tid==0) {auto* out=diag_record(params);auto now=diag_now();if(!out[23])out[23]=now;out[24]=now;}
    // Fill the LRU cache from the same shared-memory copy''')
fetch=fetch.replace('''  // Deferred staged chunks:''','''  diag_mark(params,25);
  if(tid==0)diag_record(params)[28]=deferred_count;
  // Deferred staged chunks:''')
at=s.index('template <typename Q, typename KV, typename O, typename Id>\nstruct TracePVTraits')
s=s[:at]+fetch+s[at:]
a=s.index('template <typename KTraits, typename Params>\n__device__ __forceinline__ void trace_pv_attend_joint_host_tiles');b=s.index('// partial_capacity bounds',a)
section=s[a:b].replace('wait_host_batch_tile<','diag_wait_host_batch_tile<').replace('fetch_stream_source_pages<','diag_fetch_pages<')
old='''    if (item < lru_items) {
      run_host_batch_lru<KTraits>'''
new='''    diag_mark(params,36);
    if (item < lru_items) {
      run_host_batch_lru<KTraits>'''
assert section.count(old)==1;section=section.replace(old,new)
old='''      __syncthreads();
      continue;
    }
    trace_pv_run_joint_host_item'''
new='''      __syncthreads();
      if(threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0) {auto* out=diag_record(params);out[14]++;out[15]+=diag_now()-out[36];}
      continue;
    }
    if(threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0) {auto* out=diag_record(params);auto now=diag_now();if(!out[20])out[20]=now;out[21]=now;}
    trace_pv_run_joint_host_item'''
assert section.count(old)==1;section=section.replace(old,new)
old='''    trace_pv_run_joint_host_item<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, item - lru_items, s_frag, o_frag,
                                 m, d);'''
new=old+'''
    if(threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0) {auto* out=diag_record(params);auto now=diag_now();out[13]++;out[16]+=now-out[36];out[22]=now;}'''
assert section.count(old)==1;section=section.replace(old,new)
old='''  const uint32_t epoch = params.state[kHostBatchEpoch];'''
new=old+'''
  if(tid==0) {auto* out=diag_record(params);for(int i=0;i<40;++i)out[i]=0;out[11]=epoch;out[12]=blockIdx.y>=gridDim.y-params.fetch_ctas;unsigned sm;asm volatile("mov.u32 %0, %%smid;":"=r"(sm));out[10]=sm;}
  diag_mark(params,0);'''
assert section.count(old)==1;section=section.replace(old,new)
section=section.replace('''    cg::this_grid().sync();
    gaps =''','''    diag_mark(params,1);
    cg::this_grid().sync();
    diag_mark(params,2);
    gaps =''',1)
section=section.replace('''    if (gaps) compact_host_batch_rows<KTraits>(params);''','''    if (gaps) compact_host_batch_rows<KTraits>(params);
    diag_mark(params,3);''',1)
section=section.replace('''    cg::this_grid().sync();
    // Claim counts''','''    cg::this_grid().sync();
    diag_mark(params,4);
    // Claim counts''',1)
old='''  if (unique > 0 && from_back && !host_batch_writes_back(params)) diag_fetch_pages<KTraits>(params, smem, epoch, unique, fetch_rows);
  trace_pv_run_joint_host_items<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, from_back, unique);'''
new='''  diag_mark(params,5);
  if (unique > 0 && from_back && !host_batch_writes_back(params)) diag_fetch_pages<KTraits>(params, smem, epoch, unique, fetch_rows);
  diag_mark(params,6);diag_mark(params,7);
  trace_pv_run_joint_host_items<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, from_back, unique);
  diag_mark(params,8);'''
assert section.count(old)==1;section=section.replace(old,new)
section=section.replace('''  // The last CTA to leave resets''','''  diag_mark(params,9);
  // The last CTA to leave resets''',1)
s=s[:a]+section+s[b:]
# Replay-only control uses the exact current math and remap from a completed full
# launch; all data is already local and ready. No metadata/cache state is modified.
c=base.replace('PVFeature','CapacityPV').replace('pv_feature_','capacity_pv_')
a=c.index('__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLACapacityPVKernel');b=c.index('// partial_capacity bounds',a)
section=c[a:b]
old='''  const uint32_t epoch = params.state[kHostBatchEpoch];'''
new=old+'''
  if (params.host_item_entries==840 || params.host_item_entries==1080) {
    const uint32_t workers=params.host_item_entries/10;
    if(tid==0) {scratch[kHostBatchScratchEpoch]=epoch;scratch[kHostBatchScratchFetchRows]=64;}
    __syncthreads();
    if(blockIdx.y<workers) {
      float s[2][8],o[KTraits::NUM_MMA_D_CKV/2][8],m[2],d[2];
      for(uint32_t row=blockIdx.y;row<params.num_rows;row+=workers)
        capacity_pv_run_joint_host_item<KTraits,Params,false>(params,storage,smem,scratch,row,s,o,m,d);
    }
    return;
  }'''
assert section.count(old)==1;section=section.replace(old,new);c=c[:a]+section+c[b:]
old=p.parent/'rebench_100gb_v2_20261103'
for name in ['resources_gpu3.py','experiment.json','plans.cpp']:
 shutil.copy2(old/name,p/name)
shutil.copy2(p.parent/'emu_fix_20261103/compile.py',p/'compile.py')
for v,value in [('trace',s),('capacity',c)]:
 (p/(v+'.cuh')).write_text(value);tree=p/('include_'+v);shutil.copytree(root/'include',tree);(tree/'flashinfer/attention/mla_host_batch.cuh').write_text(value)
 (p/(v+'.diff')).write_text(''.join(difflib.unified_diff(base.splitlines(True),value.splitlines(True),fromfile='production.cuh',tofile=v+'.cuh',n=0)))
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
record=dict(production_sha256=hashlib.sha256(base.encode()).hexdigest(),math_sha256=sha(root/'include/flashinfer/attention/mla.cuh'),far_sha256=sha(root/'include/flashinfer/attention/far_link_emu.cuh'),variants={v:sha(p/(v+'.cuh')) for v in ['trace','capacity']},trace_shape=[32,128,40],classification='Diagnostics only; no adoptedcandidate, no formalreference orproductiontiming change.',capacity='Host item scalar840/1080 selects same productionattention helper over completedremap/currentcache,84/108activeCTA inone108CTAgrid;alloutputrows computed,metadata/cache/epochunchanged.')
(p/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
(p/'.gitignore').write_text('include_*\njit_*\n__pycache__/\nplans.so\n')
