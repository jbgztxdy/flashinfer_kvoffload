from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'clean.cuh').read_text()
a='  low = selected[1];\n  high = selected[2];\n  __syncthreads();';b=a+'''
  if(tid==0){auto* counts=reinterpret_cast<uint32_t*>(params.partial_o)+round_up(ceil_div(params.staging_capacity,20u),2u)+2048;
   atomicAdd(counts,1u);atomicAdd(counts+1,uint32_t(victims>0&&victims==selected[0]));atomicAdd(counts+2,uint32_t(victims>0&&victims==selected[0]&&low!=high));atomicAdd(counts+3,victims);
  }
''';assert s.count(a)==1;s=s.replace(a,b)
a='  if(unique>0){\n    for(uint32_t pool=blockIdx.y;pool<params.num_pools;pool+=gridDim.y)';b='''  if(blockIdx.y==0&&tid<4)(clean_service_flags(params)+round_up(ceil_div(params.staging_capacity,20u),2u)+2048)[tid]=0;
  cg::this_grid().sync();
'''+a;assert a in s;s=s.replace(a,b)
assert not (p/'lru_frequency.cuh').exists();(p/'lru_frequency.cuh').write_text(s);shutil.copytree(p/'include_clean',p/'include_lru_frequency');(p/'include_lru_frequency/flashinfer/attention/mla_host_batch.cuh').write_text(s)
s=(p/'bench_clean.py').read_text().replace('clean','lru_frequency');a="                if VARIANT not in ['base','lru_frequency']:";b="""                counter_start=((NNZ+19)//20+1)//2*2+2048
                stats[t,:4]=w._host_batch_buffers[6].view(torch.int32).reshape(-1)[counter_start:counter_start+4].to(torch.int64)
"""+a;assert a in s;s=s.replace(a,b).replace('latency_ns=0,emulated=True','latency_ns=0,emulated=False');(p/'bench_lru_frequency.py').write_text(s)
s=(p/'run_clean.py').read_text().replace('clean','lru_frequency');s='\n'.join(x for x in s.splitlines() if not x.startswith("run('correctness"))+'\n';s=s.replace("'--passes','3'","'--passes','1'");(p/'run_lru_frequency.py').write_text(s)
(p/'lru_frequency_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'counters':['pools_with_miss','select_all','select_all_low_ne_high','victims'],'scope':'frequency diagnostic only; extra init gridbarrier and atomics timed, no speed claim'},indent=2))
