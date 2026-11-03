from pathlib import Path
import shutil,hashlib,json
p=Path(__file__).resolve().parent;s=(p/'poolowner.cuh').read_text();a=s.index(' if(t==0){uint32_t before=0,jobs=0,total=ceil_div(unique,20u);');b=s.index('\n FAR_EMU_INIT',a)
s=s[:a]+''' uint32_t total=ceil_div(unique,20u),begin=uint64_t(total)*t/256,end=uint64_t(total)*(t+1)/256;
 auto* prefix=reinterpret_cast<uint32_t*>(s.q_smem_nope);
 uint32_t before=0,jobs=0;
 for(uint32_t chunk=begin;chunk<end;++chunk){uint32_t owner=p.slot_pool[chunk*20]%gridDim.y;before+=owner<blockIdx.y;jobs+=owner==blockIdx.y;}
 prefix[t]=jobs;prefix[256+t]=before;__syncthreads();
 if(t==0){uint32_t sum=0,base=0;for(uint32_t i=0;i<256;++i){uint32_t n=prefix[i];prefix[i]=sum;sum+=n;base+=prefix[256+i];}s.service.produced=base;s.service.issues=sum;}
 __syncthreads();uint32_t at=s.service.produced+prefix[t];
 for(uint32_t chunk=begin;chunk<end;++chunk)if(p.slot_pool[chunk*20]%gridDim.y==blockIdx.y)poolowner_list(p)[at++]=chunk;
'''+s[b:]
assert not (p/'poolparallel.cuh').exists();(p/'poolparallel.cuh').write_text(s);shutil.copytree(p/'include_poolowner',p/'include_poolparallel');(p/'include_poolparallel/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_poolparallel.py').write_text((p/f'{n}_poolowner.py').read_text().replace('poolowner','poolparallel'))
(p/'poolparallel_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'mechanism':'parallel contiguous metadata scan using dead query shared before query loads; identical stable packed lists'},indent=2))
