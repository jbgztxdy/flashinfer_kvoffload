from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'demand_phase.cuh').read_text()
a='    const uint32_t service_tid=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x;'
b=a+'''
    if constexpr(ACTIVE){
     if(threadIdx.y==0&&threadIdx.z==0){
      int future=min(hi-2,host_tiles-1);uint32_t wanted=ceil_div(storage.service.count,20u);
      bool candidate=false;
      if(future>=0){uint32_t k=uint32_t(future)*32+threadIdx.x;auto entry=k<kv_len?params.remap[index_base+k]:0;uint32_t slot=uint32_t(-(entry+1));
       if(entry<0&&slot>=params.cache_slots){wanted=(slot-params.cache_slots)/20;candidate=atomicAdd(demand_claims(params)+wanted,0u)==0;}
      }
      unsigned mask=__ballot_sync(0xffffffffu,candidate);uint32_t chosen=__shfl_sync(0xffffffffu,wanted,mask?__ffs(mask)-1:0);
      if(threadIdx.x==0)storage.service.rows=mask?chosen:ceil_div(storage.service.count,20u);
     }
    }'''
assert s.count(a)==1;s=s.replace(a,b)
assert not (p/'demand_lookahead.cuh').exists();(p/'demand_lookahead.cuh').write_text(s);shutil.copytree(p/'include_demand_phase',p/'include_demand_lookahead');(p/'include_demand_lookahead/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_demand_lookahead.py').write_text((p/f'{n}_demand_phase.py').read_text().replace('demand_phase','demand_lookahead'))
(p/'demand_lookahead_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'mechanism':'warp0 actual future far tile remap lookup before current QK; existing preQK CTA barrier publishes hint; no early payload/bandwidth reservation'},indent=2))
