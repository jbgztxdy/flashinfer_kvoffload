from pathlib import Path
import shutil,hashlib,json
p=Path(__file__).resolve().parent;s=(p/'clean.cuh').read_text()
a='__syncthreads();if(!s.service.waiting)return;clean_service_step<KTraits>(p,s);__nanosleep(64);'
b='__syncthreads();if(!s.service.waiting)return;uint64_t wait_start=clock64();clean_service_step<KTraits>(p,s);if(t==0){++s.service.wait_loops;s.service.issues+=uint32_t(clock64()-wait_start);}__nanosleep(64);'
assert s.count(a)==1;s=s.replace(a,b)
a='if(t==0){asm volatile("st.release.gpu.global.u32';b='if(t==0){++s.service.produced;asm volatile("st.release.gpu.global.u32';assert s.count(a)==1;s=s.replace(a,b)
a='   if((blockIdx.y+storage.service.done*gridDim.y)*20<storage.service.count)\n';b='   if(threadIdx.x==0&&threadIdx.y==0&&threadIdx.z==0 && (blockIdx.y+storage.service.done*gridDim.y)*20<storage.service.count)++storage.service.rows;\n'+a
assert s.count(a)==1;s=s.replace(a,b)
assert not (p/'waitdiag.cuh').exists();(p/'waitdiag.cuh').write_text(s);shutil.copytree(p/'include_clean',p/'include_waitdiag');(p/'include_waitdiag/flashinfer/attention/mla_host_batch.cuh').write_text(s)
s=(p/'bench_clean.py').read_text().replace('clean','waitdiag').replace("if VARIANT not in ['base','waitdiag']:","if VARIANT != 'base':")
s='\n'.join(x for x in s.splitlines() if not "assert all(x[0]==x[1]" in x)+'\n';s=s.replace('latency_ns=0,emulated=True','latency_ns=0,emulated=False');(p/'bench_waitdiag.py').write_text(s)
s=(p/'run_clean.py').read_text().replace('clean','waitdiag');s='\n'.join(x for x in s.splitlines() if not x.startswith("run('correctness"))+'\n';s=s.replace("'--passes','3'","'--passes','1'");(p/'run_waitdiag.py').write_text(s)
(p/'waitdiag_source.json').write_text(json.dumps({'sha256':hashlib.sha256((p/'waitdiag.cuh').read_bytes()).hexdigest(),'counters':['published_packets','blocked_step_cycles_sum','blocked_step_count','rows_started_with_own_work_remaining','sizeof_storage'],'scope':'diagnostic instrumentation, not optimized timing'},indent=2))
