from pathlib import Path
import shutil,hashlib,json
p=Path(__file__).resolve().parent
s=(p/'claimrow_interleave.cuh').read_text()
marker='template<typename Params> __device__ __forceinline__ uint32_t* claimrow_packet_list'
helper='''// Diagnostic only. Event-external trace copy; exact payload path retained.
template<typename Params> __device__ __forceinline__ void interleave_stamp(const Params& p,uint32_t phase){
 if(threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0){unsigned long long now;asm volatile("mov.u64 %0,%%globaltimer;":"=l"(now)::"memory");clean_service_diag(p)[phase]=now;}
}
'''
assert s.count(marker)==1;s=s.replace(marker,helper+marker)
a=' clean_service_drain<KTraits>(params,storage);'
assert s.count(a)==1;s=s.replace(a,a+chr(10)+' interleave_stamp(params,6);')
start=s.index('void BatchMLACleanServiceKernel(');end=s.index('// partial_capacity bounds',start)
body=s[start:end]
marks=[('  const uint32_t epoch = params.state[kHostBatchEpoch];',' 0'),
       ('    cg::this_grid().sync();'+chr(10)+'    gaps =',' 1'),
       ('    claimrow_plan<KTraits>(params, smem, unique);',' 2_before_3_after'),
       ('  const uint32_t fetch_rows=20;',' 4'),
       ('  clean_service_initialize<KTraits>(params,storage,unique,epoch);',' 5')]
a=marks[0][0];assert body.count(a)==1;body=body.replace(a,a+chr(10)+'  interleave_stamp(params,0);')
a=marks[1][0];assert body.count(a)==1;body=body.replace(a,'    cg::this_grid().sync();'+chr(10)+'    interleave_stamp(params,1);'+chr(10)+'    gaps =')
a=marks[2][0];assert body.count(a)==1;body=body.replace(a,'    interleave_stamp(params,2);'+chr(10)+a+chr(10)+'    interleave_stamp(params,3);')
a=marks[3][0];assert body.count(a)==1;body=body.replace(a,'  interleave_stamp(params,4);'+chr(10)+a)
a=marks[4][0];assert body.count(a)==1;body=body.replace(a,a+chr(10)+'  interleave_stamp(params,5);')
s=s[:start]+body+s[end:]
a=' const uint32_t tid=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x;\n if(tid==0){auto* diag=clean_service_diag(params);diag[0]=storage.service.produced;diag[1]=storage.service.issues;diag[2]=storage.service.wait_loops;diag[3]=storage.service.rows;diag[4]=sizeof(typename KTraits::SharedStorage);}'
b=' interleave_stamp(params,7);'
assert s.count(a)==1;s=s.replace(a,b)
name='claimrow_interleave_timeline';assert not(p/(name+'.cuh')).exists()
(p/(name+'.cuh')).write_text(s)
shutil.copytree(p/'include_clean',p/('include_'+name))
(p/('include_'+name)/'flashinfer/attention/mla_host_batch.cuh').write_text(s)
bench=(p/'bench_claimrow_timeline.py').read_text().replace('claimrow_timeline','claimrow_interleave_timeline')
bench=bench.replace("'roundrobin' if A.fetch_ctas==23 else 'claimrow_quota'", "'half_local_interleaved'")
a=bench.index('    if A.fetch_ctas==24:'+chr(10)+'        # Event-external exact inspection')
b=bench.index('    resident,sp,stamp,store=',a)
bench=bench[:a]+bench[b:]
(p/('bench_'+name+'.py')).write_text(bench)
run=(p/'run_claimrow_timeline.py').read_text().replace('claimrow_timeline',name)
run=chr(10).join(line for line in run.splitlines() if not line.startswith("run('correctness_") and not line.startswith("run('claimrow_interleave_timeline_rr_native'"))+chr(10)
(p/('run_'+name+'.py')).write_text(run)
(p/(name+'_source.json')).write_text(json.dumps(dict(sha256=hashlib.sha256(s.encode()).hexdigest(),base_sha256=hashlib.sha256((p/'claimrow_interleave.cuh').read_bytes()).hexdigest(),scope='diagnostic normal GPU3; full kernel unchanged except 7 globaltimer stamps; no100GB/hardware byte conclusion',phase_names=['kernel_start','prepare_grid_done','resolve_done','planner_done','lru_grid_done','service_initialized','compute_and_drain_done']),indent=2)+chr(10))
