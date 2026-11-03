from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'claimrow_cached_packet.cuh').read_text()
marker='template<typename Params> __device__ __forceinline__ uint32_t* claimrow_packet_list'
helper='''// Private diagnostic: fetch_ctas23 selects roundrobin,24 selects quota.
// Both launch108 full computeCTAs; this is not fetch-SM tuning.
template<typename Params> __device__ __forceinline__ void claimrow_stamp(const Params& p,uint32_t phase){
 if(threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0){unsigned long long now;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(now));clean_service_diag(p)[phase]=now;}
}
''';assert s.count(marker)==1;s=s.replace(marker,helper+marker)
a=' if(ticket>=s.service.limit)return ceil_div(s.service.count,20u);';assert s.count(a)==1;s=s.replace(a,a+'\n if(p.fetch_ctas==23)return blockIdx.y+ticket*gridDim.y;')
a='if(t==0){uint32_t chunk=s.service.packet;asm volatile("st.release.gpu.global.u32 [%0],%1;"::"l"(clean_service_flags(p)+chunk),"r"(s.service.epoch+1):"memory");}'
b=a[:-1]+'if(ticket+1==s.service.limit)claimrow_stamp(p,6);}';assert s.count(a)==1;s=s.replace(a,b)
a=' if(tid==0){auto* diag=clean_service_diag(params);diag[0]=storage.service.produced;diag[1]=storage.service.issues;diag[2]=storage.service.wait_loops;diag[3]=storage.service.rows;diag[4]=sizeof(typename KTraits::SharedStorage);}'
assert s.count(a)==1;s=s.replace(a,' claimrow_stamp(params,7);')
start=s.index('void BatchMLACleanServiceKernel(');end=s.index('// partial_capacity bounds',start);body=s[start:end]
a='  const uint32_t epoch = params.state[kHostBatchEpoch];';assert body.count(a)==1;body=body.replace(a,a+'\n  claimrow_stamp(params,0);')
a='    cg::this_grid().sync();\n    gaps =';assert body.count(a)==1;body=body.replace(a,'    cg::this_grid().sync();\n    claimrow_stamp(params,1);\n    gaps =')
a='    claimrow_plan<KTraits>(params, smem, unique);';assert body.count(a)==1;body=body.replace(a,'    claimrow_stamp(params,2);\n    if(params.fetch_ctas!=23)claimrow_plan<KTraits>(params, smem, unique);\n    claimrow_stamp(params,3);')
a='  const uint32_t fetch_rows=20;';assert body.count(a)==1;body=body.replace(a,'  claimrow_stamp(params,4);\n'+a)
a='  clean_service_initialize<KTraits>(params,storage,unique,epoch);';assert body.count(a)==1;body=body.replace(a,a+'\n  claimrow_stamp(params,5);\n  if(!storage.service.limit)claimrow_stamp(params,6);')
s=s[:start]+body+s[end:]
assert not(p/'claimrow_timeline.cuh').exists();(p/'claimrow_timeline.cuh').write_text(s)
shutil.copytree(p/'include_clean',p/'include_claimrow_timeline');(p/'include_claimrow_timeline/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_claimrow_timeline.py').write_text((p/f'{n}_claimrow_cached_packet.py').read_text().replace('claimrow_cached_packet','claimrow_timeline'))
f=p/'test_claimrow_timeline.py';t=f.read_text().replace("@pytest.mark.parametrize('length,rows,staged'","@pytest.mark.parametrize('fetch_ctas',[23,24])\n@pytest.mark.parametrize('length,rows,staged'").replace('(length,rows,staged):','(length,rows,staged,fetch_ctas):').replace('w.host_fetch_ctas=24','w.host_fetch_ctas=fetch_ctas');f.write_text(t)
f=p/'bench_claimrow_timeline.py';t=f.read_text()
a='        stats=torch.zeros((STEPS,8),dtype=torch.int64,device=DEV);';assert t.count(a)==1;t=t.replace(a,'        stamps=torch.empty((STEPS,108,8),dtype=torch.int64,device=DEV)\n'+a)
a='                refs[t]=(w._host_batch_buffers[3]<=-(R*CAP+1)).sum()';b=a+'''
                flag_words=(((NNZ+19)//20+1)//2)*2
                stamp_source=w._host_batch_buffers[6].view(torch.int64).flatten()[flag_words//2:flag_words//2+108*8].view(108,8)
                stamps[t].copy_(stamp_source) # afterevent, identical in both policies
''';assert t.count(a)==1;t=t.replace(a,b)
a="        for t,got in expected:";assert t.count(a)==1;t=t.replace(a,"        row['phase_timestamps_ns']=stamps.cpu().tolist()\n"+a)
a="    # Event-external exact inspection of the actual GPU-produced packet list.";start=t.index(a);end=t.index('    resident,sp,stamp,store=',start);block=t[start:end];t=t[:start]+'    if A.fetch_ctas==24:\n'+''.join('    '+line+'\n' for line in block.splitlines())+t[end:]
a='config=dict(normal_build=True,simulator_hooks=False,service_counters=False,';assert t.count(a)==1;t=t.replace(a,"config=dict(timeline_diagnostic=True,producer_policy='roundrobin' if A.fetch_ctas==23 else 'claimrow_quota',normal_build=True,simulator_hooks=False,service_counters=False,")
f.write_text(t)
f=p/'run_claimrow_timeline.py';t=f.read_text();line=next(x for x in t.splitlines() if x.startswith("run('claimrow_timeline_native'"));assert "'--passes','3'" in line
new=line.replace("'--passes','3'","'--passes','1'")
t=t.replace(line,new.replace("'claimrow_timeline_native'","'claimrow_timeline_rr_native'").replace("'--fetch-ctas','24'","'--fetch-ctas','23'")+'\n'+new.replace("'claimrow_timeline_native'","'claimrow_timeline_quota_native'"));f.write_text(t)
(p/'claimrow_timeline_source.json').write_text(json.dumps(dict(sha256=hashlib.sha256(s.encode()).hexdigest(),base_sha256=hashlib.sha256((p/'claimrow_cached_packet.cuh').read_bytes()).hexdigest(),mechanism='samebinary full108CTA kernel; privatefetch_ctas23=roundrobin/24=quota; eightglobaltimer phases, realpayload/LRU/attention retained; afterevent perstepdiagcopy',phase_names=['kernel_start','prepare_grid_done','resolve_done','planner_done','lru_grid_done','service_initialized','last_packet_published','attention_drain_done']),indent=2)+'\n')
