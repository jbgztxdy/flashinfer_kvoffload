from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'clean.cuh').read_text()
marker='template<typename KTraits,typename Params>\n__device__ __forceinline__ void clean_service_ids('
helper='''template<typename Params> __device__ __forceinline__ uint32_t* claimrow_packet_list(const Params& p){return clean_service_flags(p)+round_up(ceil_div(p.staging_capacity,20u),2u)+128*16;}
template<typename KTraits,typename Params>
__device__ __forceinline__ uint32_t clean_service_chunk(const Params& p,const typename KTraits::SharedStorage& s,uint32_t ticket){
 const uint32_t n=ceil_div(s.service.count,20u),q=n/gridDim.y,r=n%gridDim.y,capacity=q+(blockIdx.y<r);
 if(ticket>=capacity)return n;
 return claimrow_packet_list(p)[blockIdx.y*q+min(blockIdx.y,r)+ticket];
}
// Reuse the row prefix that resolve has already produced in this CTA's shared
// memory. Parallel per-owner counts + two scans + inverse ranges; no greedy planner.
template<typename KTraits,typename Params>
__device__ __forceinline__ void claimrow_plan(const Params& p,uint8_t* smem,uint32_t unique){
 if(!unique)return; // resolve does not publish global rowbases in the zero-miss case.
 const uint32_t t=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x,n=ceil_div(unique,20u),q=n/gridDim.y,rem=n%gridDim.y;
 auto* rows=reinterpret_cast<uint32_t*>(smem);auto* counts=rows+round_up(p.num_rows+1,32u);
 auto* excess=counts+round_up(gridDim.y,32u);auto* deficit=excess+round_up(gridDim.y,32u);
 for(uint32_t owner=t;owner<gridDim.y;owner+=KTraits::NUM_THREADS){
  uint32_t count=0;
  for(uint32_t row=owner;row<p.num_rows;row+=gridDim.y){uint32_t end=row+1<p.num_rows?rows[row+1]:unique;count+=ceil_div(end,20u)-ceil_div(rows[row],20u);}
  counts[owner]=count;
 }
 __syncthreads();
 host_batch_block_scan<KTraits::NUM_THREADS>(gridDim.y,[&](uint32_t owner){uint32_t cap=q+(owner<rem);return counts[owner]>cap?counts[owner]-cap:0u;},excess);
 __syncthreads();
 host_batch_block_scan<KTraits::NUM_THREADS>(gridDim.y,[&](uint32_t owner){uint32_t cap=q+(owner<rem);return counts[owner]<cap?cap-counts[owner]:0u;},deficit);
 __syncthreads();
 const uint32_t capacity=q+(blockIdx.y<rem),keep=min(counts[blockIdx.y],capacity),base=blockIdx.y*q+min(blockIdx.y,rem);
 for(uint32_t j=t;j<capacity;j+=KTraits::NUM_THREADS){
  uint32_t source=blockIdx.y,ordinal=j;
  if(j>=keep){
   const uint32_t global=deficit[blockIdx.y]+j-keep;uint32_t lo=0,hi=gridDim.y;
   while(lo+1<hi){uint32_t mid=(lo+hi)/2;if(excess[mid]<=global)lo=mid;else hi=mid;}
   source=lo;ordinal=min(counts[source],q+(source<rem))+global-excess[source];
  }
  uint32_t packet=n;
  for(uint32_t row=source;row<p.num_rows;row+=gridDim.y){
   const uint32_t first=ceil_div(rows[row],20u),end=ceil_div(row+1<p.num_rows?rows[row+1]:unique,20u),length=end-first;
   if(ordinal<length){packet=first+ordinal;break;}ordinal-=length;
  }
  claimrow_packet_list(p)[base+j]=packet;
 }
 __syncthreads();
}
'''
assert s.count(marker)==1;s=s.replace(marker,helper+marker)
start=s.index(marker);end=s.index('template <typename KTraits, typename Params, bool ROW_MERGE>\n__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLACleanServiceKernel(',start)
body=s[start:end];a='blockIdx.y+ticket*gridDim.y';assert body.count(a)==5;body=body.replace(a,'clean_service_chunk<KTraits>(p,s,ticket)')
a='blockIdx.y+s.service.done*gridDim.y';assert body.count(a)==1;body=body.replace(a,'clean_service_chunk<KTraits>(p,s,s.service.done)')
a='blockIdx.y+storage.service.done*gridDim.y';assert body.count(a)==1;body=body.replace(a,'clean_service_chunk<KTraits>(params,storage,storage.service.done)')
s=s[:start]+body+s[end:]
start=s.index('void BatchMLACleanServiceKernel(');end=s.index('// partial_capacity bounds',start);body=s[start:end]
a='    if (gaps) compact_host_batch_rows<KTraits>(params);';assert body.count(a)==1;body=body.replace(a,'    claimrow_plan<KTraits>(params, smem, unique);\n'+a);s=s[:start]+body+s[end:]
a='(ceil_div(params.staging_capacity,20u)+128*16ull)*4';assert s.count(a)==1;s=s.replace(a,'(2ull*round_up(ceil_div(params.staging_capacity,20u),2u)+128*16ull)*4')
a='const bool pv_feature = params.stage && !params.host_split && params.num_head_tiles == 1 &&';assert s.count(a)==1;s=s.replace(a,a+'\n        4ull*(round_up(params.num_rows+1,32u)+3*round_up(static_cast<uint32_t>(num_sms),32u)) <= clean_service_smem &&')
assert not(p/'claimrow_quota.cuh').exists();(p/'claimrow_quota.cuh').write_text(s)
shutil.copytree(p/'include_clean',p/'include_claimrow_quota');(p/'include_claimrow_quota/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_claimrow_quota.py').write_text((p/f'{n}_rescale_skip.py').read_text().replace('rescale_skip','claimrow_quota'))
(p/'claimrow_quota_source.json').write_text(json.dumps(dict(sha256=hashlib.sha256(s.encode()).hexdigest(),base_sha256=hashlib.sha256((p/'clean.cuh').read_bytes()).hexdigest(),mechanism='existing shared rowprefix => parallel owner counts, excess/deficit scans, perpacket inverse ranges => exactbalanced localfirst/spill list; noadditionalgridbarrier; unique source/stage/cache/LRU unchanged'),indent=2)+'\n')
