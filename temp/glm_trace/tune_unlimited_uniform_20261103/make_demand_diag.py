from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'demand_phase.cuh').read_text();a=' uint32_t total=ceil_div(count,20u);if(wanted<total&&atomicCAS(demand_claims(p)+wanted,0u,1u)==0)return wanted;';b=''' auto* counters=demand_claims(p)-16;
 uint32_t total=ceil_div(count,20u);
 if(wanted<total){if(atomicCAS(demand_claims(p)+wanted,0u,1u)==0){atomicAdd(counters,1u);return wanted;}atomicAdd(counters+1,1u);}else atomicAdd(counters+2,1u);''';assert a in s;s=s.replace(a,b)
a='if(atomicCAS(demand_claims(p)+chunk,0u,1u)==0)return chunk;';b='if(atomicCAS(demand_claims(p)+chunk,0u,1u)==0){atomicAdd(counters+3,1u);return chunk;}atomicAdd(counters+4,1u);';assert a in s;s=s.replace(a,b)
a='  if(blockIdx.y==0&&tid==0)*demand_cursor(params)=0;';b=a+'\n  if(blockIdx.y==0&&tid<5)(demand_claims(params)-16)[tid]=0;';assert a in s;s=s.replace(a,b)
(p/'demand_diag.cuh').write_text(s);shutil.copytree(p/'include_demand_phase',p/'include_demand_diag');(p/'include_demand_diag/flashinfer/attention/mla_host_batch.cuh').write_text(s)
s=(p/'bench_demand_phase.py').read_text().replace('demand_phase','demand_diag');a="                if VARIANT not in ['base','demand_diag']:";b="""                claim_start=((NNZ+19)//20+1)//2*2+2048
                stats[t,:5]=w._host_batch_buffers[6].view(torch.int32).reshape(-1)[claim_start-16:claim_start-11].to(torch.int64)
"""+a;assert a in s;s=s.replace(a,b);(p/'bench_demand_diag.py').write_text(s)
s=(p/'run_demand_phase.py').read_text().replace('demand_phase','demand_diag');s='\n'.join(x for x in s.splitlines() if not x.startswith("run('correctness"))+'\n';s=s.replace("'--passes','3'","'--passes','1'");(p/'run_demand_diag.py').write_text(s)
(p/'demand_diag_source.json').write_text(json.dumps({'sha256':hashlib.sha256((p/'demand_diag.cuh').read_bytes()).hexdigest(),'counters':['hint_success','hint_already_claimed','no_hint','fallback_success','fallback_skipped_claimed'],'scope':'instrumented diagnostic timing not adopted'},indent=2))
