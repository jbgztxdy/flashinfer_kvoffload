from pathlib import Path
import shutil,hashlib,json
p=Path(__file__).resolve().parent
s=(p/'clean.cuh').read_text()
helper='''__device__ __forceinline__ uint32_t contiguous_chunk(uint32_t count,uint32_t ticket){
 const uint32_t total=ceil_div(count,20u),begin=uint64_t(total)*blockIdx.y/gridDim.y,end=uint64_t(total)*(blockIdx.y+1)/gridDim.y;
 return ticket<end-begin?begin+ticket:total;
}
'''
s=s.replace('struct CleanService {',helper+'struct CleanService {')
for old,new in [('blockIdx.y+ticket*gridDim.y','contiguous_chunk(s.service.count,ticket)'),('blockIdx.y+s.service.done*gridDim.y','contiguous_chunk(s.service.count,s.service.done)'),('blockIdx.y+storage.service.done*gridDim.y','contiguous_chunk(storage.service.count,storage.service.done)')]:
 assert old in s,old
 s=s.replace(old,new)
assert not (p/'contiguous.cuh').exists()
(p/'contiguous.cuh').write_text(s)
shutil.copytree(p/'include_clean',p/'include_contiguous')
(p/'include_contiguous/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for name in ['test','bench','run']:
 t=(p/f'{name}_clean.py').read_text().replace('clean','contiguous')
 (p/f'{name}_contiguous.py').write_text(t)
(p/'contiguous_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'base':hashlib.sha256((p/'clean.cuh').read_bytes()).hexdigest(),'mechanism':'contiguous balanced unique packet ownership; unchanged row schedule, payload and publication'},indent=2))
