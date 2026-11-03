from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'clean.cuh').read_text()
a='for(uint32_t v=0;v<72;v+=8){uint4 value=src[v];stage[v]=value;if(victim>=0)cache[v]=value;}'
b='for(uint32_t v=0;v<72;v+=8){uint4 value=src[v];if(victim>=0)cache[v]=value;else stage[v]=value;}'
assert s.count(a)==1;s=s.replace(a,b)
marker='template <bool ACTIVE,typename KTraits, typename Params>\n__device__ __forceinline__ void clean_service_attend_joint_host_tiles'
helper='''template<typename KTraits,typename Params>
__device__ __forceinline__ void single_destination_read_pages(const Params& params,uint32_t index_base,uint32_t kv_len,uint32_t kv_base,typename KTraits::IdType* pages){
 read_host_batch_pages<KTraits>(params,index_base,kv_len,kv_base,pages);
#pragma unroll
 for(uint32_t j=0;j<kHostBatchTileRows<KTraits>;++j){
  auto page=pages[j];uint32_t row=uint32_t(-(page+1));
  if(page<0&&row>=params.cache_slots){int32_t victim=params.victim_of[row-params.cache_slots];if(victim>=0)pages[j]=-victim-1;}
 }
}
'''
assert marker in s;s=s.replace(marker,helper+marker,1)
a=s.index('void clean_service_attend_joint_host_tiles');b=s.index('template <bool ACTIVE,typename KTraits, typename Params, bool ROW_MERGE>',a);part=s[a:b];assert part.count('read_host_batch_pages<KTraits>')==3;part=part.replace('read_host_batch_pages<KTraits>','single_destination_read_pages<KTraits>');s=s[:a]+part+s[b:]
assert not (p/'single_destination.cuh').exists();(p/'single_destination.cuh').write_text(s);shutil.copytree(p/'include_clean',p/'include_single_destination');(p/'include_single_destination/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_single_destination.py').write_text((p/f'{n}_clean.py').read_text().replace('clean','single_destination'))
f=p/'bench_single_destination.py';f.write_text(f.read_text().replace('latency_ns=0,emulated=True','latency_ns=0,emulated=False'))
(p/'single_destination_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'mechanism':'resolve final cache source in existing one-tile-ahead register page prefetch; remap and packet-ready unchanged; producer writes only required final cache or stage; no new grid barrier or page-ready protocol','difference_from_rejected':'old final_destination rewrote global remap with2extra grids and per-page readiness; old dependent loader lookup not prefetched'},indent=2))
