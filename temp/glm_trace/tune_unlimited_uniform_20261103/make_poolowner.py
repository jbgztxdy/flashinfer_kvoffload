from pathlib import Path
import shutil,hashlib,json
p=Path(__file__).resolve().parent;s=(p/'clean.cuh').read_text()
# Use previously unused diagnostic fields for packed-list base/count.
marker='template<typename KTraits,typename Params>\n__device__ __forceinline__ void clean_service_ids'
helper='''template<typename Params> __device__ __forceinline__ uint32_t* poolowner_list(const Params& p){return clean_service_flags(p)+round_up(ceil_div(p.staging_capacity,20u),2u)+1024;}
template<typename KTraits,typename Params> __device__ __forceinline__ uint32_t poolowner_chunk(const Params& p,const typename KTraits::SharedStorage& s,uint32_t ticket){return ticket<s.service.issues?poolowner_list(p)[s.service.produced+ticket]:ceil_div(s.service.count,20u);}
'''
assert marker in s;s=s.replace(marker,helper+marker,1)
for a,b in [('blockIdx.y+ticket*gridDim.y','poolowner_chunk<KTraits>(p,s,ticket)'),('blockIdx.y+s.service.done*gridDim.y','poolowner_chunk<KTraits>(p,s,s.service.done)'),('blockIdx.y+storage.service.done*gridDim.y','poolowner_chunk<KTraits>(params,storage,storage.service.done)')]:
 assert a in s;s=s.replace(a,b)
a=' FAR_EMU_INIT(2);__syncthreads();clean_service_ids<KTraits>(p,s,0);'
b=''' if(t==0){uint32_t before=0,jobs=0,total=ceil_div(unique,20u);for(uint32_t chunk=0;chunk<total;++chunk){uint32_t owner=p.slot_pool[chunk*20]%gridDim.y;before+=owner<blockIdx.y;jobs+=owner==blockIdx.y;}s.service.produced=before;s.service.issues=jobs;uint32_t at=before;for(uint32_t chunk=0;chunk<total;++chunk)if(p.slot_pool[chunk*20]%gridDim.y==blockIdx.y)poolowner_list(p)[at++]=chunk;}
 FAR_EMU_INIT(2);__syncthreads();clean_service_ids<KTraits>(p,s,0);'''
assert a in s;s=s.replace(a,b)
a='for(uint32_t row=blockIdx.y;row<params.num_rows;row+=gridDim.y) {';b='for(uint32_t row=0;row<params.num_rows;++row) {\n   if(params.row_pool[row]%gridDim.y!=blockIdx.y)continue;'
assert a in s;s=s.replace(a,b)
# Original diagnostic stores conflict with list prefix; remove unused diagnostic write.
a=' if(tid==0){auto* diag=clean_service_diag(params);diag[0]=storage.service.produced;diag[1]=storage.service.issues;diag[2]=storage.service.wait_loops;diag[3]=storage.service.rows;diag[4]=sizeof(typename KTraits::SharedStorage);}'
assert a in s;s=s.replace(a,'')
assert not (p/'poolowner.cuh').exists();(p/'poolowner.cuh').write_text(s);shutil.copytree(p/'include_clean',p/'include_poolowner');(p/'include_poolowner/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for name in ['test','bench','run']:(p/f'{name}_poolowner.py').write_text((p/f'{name}_clean.py').read_text().replace('clean','poolowner'))
f=p/'bench_poolowner.py';f.write_text(f.read_text().replace('latency_ns=0,emulated=True','latency_ns=0,emulated=False'))
(p/'poolowner_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'mechanism':'exact slot_pool owner mod grid, packed packet lists in unused whole-row backing, same-pool query rows; GPU unverified'},indent=2))
