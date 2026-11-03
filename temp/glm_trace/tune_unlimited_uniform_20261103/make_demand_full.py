from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'clean.cuh').read_text()
marker='template<typename KTraits,typename Params>\n__device__ __forceinline__ void clean_service_ids'
helper='''template<typename Params> __device__ __forceinline__ uint32_t* demand_claims(const Params& p){return clean_service_flags(p)+round_up(ceil_div(p.staging_capacity,20u),2u)+2048;}
template<typename Params> __device__ __forceinline__ uint32_t* demand_cursor(const Params& p){return demand_claims(p)+ceil_div(p.staging_capacity,20u);}
template<typename Params> __device__ __forceinline__ uint32_t demand_claim(const Params& p,uint32_t count,uint32_t wanted){
 uint32_t total=ceil_div(count,20u);if(wanted<total&&atomicCAS(demand_claims(p)+wanted,0u,1u)==0)return wanted;
 while(true){uint32_t chunk=atomicAdd(demand_cursor(p),1u);if(chunk>=total)return total;if(atomicCAS(demand_claims(p)+chunk,0u,1u)==0)return chunk;}
}
'''
assert marker in s;s=s.replace(marker,helper+marker,1)
s=s.replace('blockIdx.y+ticket*gridDim.y','ticket').replace('blockIdx.y+s.service.done*gridDim.y','s.service.done')
a=' clean_service_ids<KTraits>(p,s,ticket+1);';b=' if(t==0){s.service.done=demand_claim(p,s.service.count,s.service.rows);s.service.rows=ceil_div(s.service.count,20u);}\n __syncthreads();clean_service_ids<KTraits>(p,s,s.service.done);';assert a in s;s=s.replace(a,b)
a=' clean_service_issue<KTraits>(p,s,ticket+1);if(t==0)s.service.done=ticket+1;';b=' clean_service_issue<KTraits>(p,s,s.service.done);';assert a in s;s=s.replace(a,b)
a='bool any=__any_sync(0xffffffffu,wait);if(threadIdx.x==0)s.service.waiting=any;';b='unsigned mask=__ballot_sync(0xffffffffu,wait);unsigned candidate=__shfl_sync(0xffffffffu,(row-p.cache_slots)/20,mask?__ffs(mask)-1:0);if(threadIdx.x==0){s.service.waiting=mask!=0;s.service.rows=mask?candidate:ceil_div(s.service.count,20u);}';assert a in s;s=s.replace(a,b)
a=' FAR_EMU_INIT(2);__syncthreads();clean_service_ids<KTraits>(p,s,0);__syncthreads();clean_service_issue<KTraits>(p,s,0);__syncthreads();'
b=' if(t==0){s.service.rows=ceil_div(unique,20u);s.service.done=demand_claim(p,unique,s.service.rows);}\n FAR_EMU_INIT(2);__syncthreads();clean_service_ids<KTraits>(p,s,s.service.done);__syncthreads();clean_service_issue<KTraits>(p,s,s.service.done);__syncthreads();'
assert a in s;s=s.replace(a,b)
a='''   if((blockIdx.y+storage.service.done*gridDim.y)*20<storage.service.count)
     clean_service_run_joint_host_item<true,KTraits,Params,false>(params,storage,smem,scratch,row,scores,output,m,d);
   else clean_service_run_joint_host_item<false,KTraits,Params,false>(params,storage,smem,scratch,row,scores,output,m,d);'''
b='   clean_service_run_joint_host_item<true,KTraits,Params,false>(params,storage,smem,scratch,row,scores,output,m,d);';assert a in s;s=s.replace(a,b)
# Clear claim array before existing grid barrier in LRU section; always sync including zero misses.
a='  if(unique>0){\n    for(uint32_t pool=blockIdx.y;pool<params.num_pools;pool+=gridDim.y)'
b='  for(uint32_t k=blockIdx.y*KTraits::NUM_THREADS+tid;k<ceil_div(unique,20u);k+=gridDim.y*KTraits::NUM_THREADS)demand_claims(params)[k]=0;\n  if(blockIdx.y==0&&tid==0)*demand_cursor(params)=0;\n  if(unique>0){\n    for(uint32_t pool=blockIdx.y;pool<params.num_pools;pool+=gridDim.y)';assert a in s;s=s.replace(a,b)
s=s.replace('    cg::this_grid().sync();\n  }\n  const uint32_t fetch_rows=20;', '  }\n  cg::this_grid().sync();\n  const uint32_t fetch_rows=20;')
assert not (p/'demand_full.cuh').exists();(p/'demand_full.cuh').write_text(s);shutil.copytree(p/'include_clean',p/'include_demand_full');(p/'include_demand_full/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_demand_full.py').write_text((p/f'{n}_clean.py').read_text().replace('clean','demand_full'))
f=p/'bench_demand_full.py';f.write_text(f.read_text().replace('latency_ns=0,emulated=True','latency_ns=0,emulated=False'))
(p/'demand_full_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'mechanism':'dynamic whole-packet CAS claim, consumer demand hint, global fallback cursor, original real attention/LRU/payload; all active rows to preserve progress','status':'prototype GPU unverified'},indent=2))
