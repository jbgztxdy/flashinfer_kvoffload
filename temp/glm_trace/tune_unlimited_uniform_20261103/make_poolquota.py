from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'poolrows.cuh').read_text()
a=s.index(' __syncthreads();\n if(t==0){uint32_t before=0,jobs=0;',s.index('void clean_service_initialize'));b=s.index('\n FAR_EMU_INIT',a)
s=s[:a]+''' if(t==0){
  const uint32_t total=ceil_div(unique,20u),q=total/gridDim.y,r=total%gridDim.y;
  auto* remain=reinterpret_cast<uint32_t*>(s.q_smem_nope);
  auto* spill_first=remain+gridDim.y;auto* spill_count=spill_first+p.num_pools;
  for(uint32_t c=0;c<gridDim.y;++c)remain[c]=q+(c<r);
  s.service.produced=blockIdx.y*q+min(uint32_t(blockIdx.y),r);s.service.issues=q+(blockIdx.y<r);
  uint32_t output=s.service.produced;
  for(uint32_t pool=0;pool<p.num_pools;++pool){
   uint32_t row0=p.pool_row_start[pool],row1=p.pool_row_start[pool+1];
   uint32_t at=ceil_div(p.row_slot_base[row0],20u),end=ceil_div(p.row_slot_base[row1],20u);
   while(at<end){uint32_t best=gridDim.y,capacity=0;
    for(uint32_t row=row0;row<row1;++row){uint32_t c=row%gridDim.y;if(remain[c]>capacity){capacity=remain[c];best=c;}}
    if(!capacity)break;
    uint32_t take=min(end-at,capacity);remain[best]-=take;
    if(best==blockIdx.y)for(uint32_t j=0;j<take;++j)poolowner_list(p)[output++]=at+j;
    at+=take;
   }
   spill_first[pool]=at;spill_count[pool]=end-at;
  }
  uint32_t c=0;
  for(uint32_t pool=0;pool<p.num_pools;++pool){uint32_t at=spill_first[pool],left=spill_count[pool];
   while(left){while(c<gridDim.y&&!remain[c])++c;uint32_t take=min(left,remain[c]);remain[c]-=take;
    if(c==blockIdx.y)for(uint32_t j=0;j<take;++j)poolowner_list(p)[output++]=at+j;
    at+=take;left-=take;
   }
  }
 }
'''+s[b:]
assert not (p/'poolquota.cuh').exists();(p/'poolquota.cuh').write_text(s);shutil.copytree(p/'include_poolrows',p/'include_poolquota');(p/'include_poolquota/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_poolquota.py').write_text((p/f'{n}_poolrows.py').read_text().replace('poolrows','poolquota'))
(p/'poolquota_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'base_sha256':hashlib.sha256((p/'poolrows.cuh').read_bytes()).hexdigest(),'mechanism':'device-local coarse pool interval greedy, fixed round-robin packet quotas, local-first then spill; all planning and copies timed','scope':'prototype same<=4096row <=128SM gate; scratch fits unused q shared before query loads; duplicated leader planner cost retained'},indent=2))
