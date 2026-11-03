"""One-time minimal normal production timeline, no per-tile instrumentation."""
from pathlib import Path
import hashlib,json,shutil
p=Path(__file__).resolve().parent;root=p.parents[2];assert not(p/'native_diag.cuh').exists()
s=(p/'base.cuh').read_text().replace('PVFeature','NativeDiagPV').replace('pv_feature_','native_diag_pv_')
a=s.index('template <typename KTraits>\n__device__ __forceinline__ void native_diag_pv_pv')
helper='''// Nine uint64 fields per CTA/epoch in unused whole-row partial backing.
__device__ __forceinline__ unsigned long long native_diag_now(){unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;}
template<typename Params>__device__ __forceinline__ unsigned long long* native_diag_record(const Params& p){return reinterpret_cast<unsigned long long*>(p.partial_o)+((p.state[kHostBatchEpoch]%32)*128+blockIdx.y)*9;}
template<typename Params>__device__ __forceinline__ void native_diag_mark(const Params& p,uint32_t k){if(threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0)native_diag_record(p)[k]=native_diag_now();}
'''
s=s[:a]+helper+s[a:]
a=s.index('void native_diag_pv_run_joint_host_items(');b=s.index('// partial_capacity bounds',a);x=s[a:b]
call='''    native_diag_pv_run_joint_host_item<KTraits, Params, ROW_MERGE>(params, storage, smem, scratch, item - lru_items, s_frag, o_frag,
                                 m, d);'''
assert call in x;x=x.replace(call,'''    if(threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0){auto* r=native_diag_record(params);if(!r[3])r[3]=native_diag_now();}
'''+call+'''
    if(threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0){auto* r=native_diag_record(params);r[4]=native_diag_now();r[6]++;}
''')
x=x.replace('  const uint32_t epoch = params.state[kHostBatchEpoch];','''  const uint32_t epoch = params.state[kHostBatchEpoch];
  if(tid==0){auto* r=native_diag_record(params);for(int k=0;k<9;++k)r[k]=0;r[7]=blockIdx.y>=gridDim.y-params.fetch_ctas;r[8]=epoch;}
  native_diag_mark(params,0);''')
call='  if (unique > 0 && from_back && !host_batch_writes_back(params)) fetch_stream_source_pages<KTraits>(params, smem, epoch, unique, fetch_rows);'
assert call in x;x=x.replace(call,'  native_diag_mark(params,1);\n'+call+'\n  native_diag_mark(params,2);')
x=x.replace('  // The last CTA to leave resets','  native_diag_mark(params,5);\n  // The last CTA to leave resets');s=s[:a]+x+s[b:]
(p/'native_diag.cuh').write_text(s);shutil.copytree(root/'include',p/'include_native_diag');(p/'include_native_diag/flashinfer/attention/mla_host_batch.cuh').write_text(s)
m=json.loads((p/'manifest.json').read_text());m['native_diag_sha256']=hashlib.sha256(s.encode()).hexdigest();m['native_diag_mechanism']='Production normal build with ninefields/CTA/epoch inunusedpartialbacking; onlyphase stamps and perrow counter; no tile/fetchloop probes';(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
x=(p/'bench_clean.py').read_text().replace("['base','clean']","['base','clean','native_diag']").replace('latency_ns=0,emulated=True)','latency_ns=0,emulated=False)').replace('service_counters=False,','service_counters=False,diagnostic_only=True,')
old="        result['passes'].append(row);save(target,result)"
new="        row['native_timeline']=w._host_batch_buffers[6].view(torch.int64).reshape(-1)[:32*128*9].reshape(32,128,9)[:,:108].cpu().tolist()\n"+old
assert old in x;x=x.replace(old,new);(p/'bench_native_diag.py').write_text(x)
x=(p/'run_clean.py').read_text();x=x[:x.index("save()\nrun('correctness_clean'")].replace("p/'clean_state.json'","p/'native_diag_state.json'")
x+='''save()
run('native_diag_native','native_diag',['-u',str(p/'bench_native_diag.py'),'--kind','lru','--mtp','8','--requests','128','--positions','256','--passes','1','--settings','native_hbm','--fetch-ctas','24','--host-split','off'])
r['complete']=True;r['end']=time.strftime('%FT%T%z');save()
'''
(p/'run_native_diag.py').write_text(x)
