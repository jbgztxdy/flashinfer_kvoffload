from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'poolrows.cuh').read_text()
a=s.index(' __syncthreads();\n if(t==0){uint32_t before=0,jobs=0;',s.index('void clean_service_initialize'))
b=s.index('\n FAR_EMU_INIT',a)
s=s[:a]+''' __syncthreads();
 auto* counts=reinterpret_cast<uint32_t*>(s.q_smem_nope);
 // Thread c counts packets assigned to compute row owner c. Pool rows are contiguous.
 if(t<gridDim.y){uint32_t jobs=0;
  for(uint32_t pool=0;pool<p.num_pools;++pool){uint32_t row0=p.pool_row_start[pool],nrows=p.pool_row_start[pool+1]-row0;
   uint32_t first=ceil_div(p.row_slot_base[row0],20u),last=ceil_div(p.row_slot_base[row0+nrows],20u);
   for(uint32_t r=0;r<nrows;++r)if((row0+r)%gridDim.y==t && first+r<last)jobs+=ceil_div(last-first-r,nrows);
  }counts[t]=jobs;
 }
 __syncthreads();
 if(t==0){uint32_t base=0;for(uint32_t c=0;c<blockIdx.y;++c)base+=counts[c];s.service.produced=base;s.service.issues=counts[blockIdx.y];}
 __syncthreads();uint32_t at=s.service.produced;
 for(uint32_t pool=0;pool<p.num_pools;++pool){uint32_t row0=p.pool_row_start[pool],nrows=p.pool_row_start[pool+1]-row0;
  uint32_t first=ceil_div(p.row_slot_base[row0],20u),last=ceil_div(p.row_slot_base[row0+nrows],20u);
  for(uint32_t r=0;r<nrows;++r)if((row0+r)%gridDim.y==blockIdx.y){uint32_t jobs=first+r<last?ceil_div(last-first-r,nrows):0;
   for(uint32_t j=t;j<jobs;j+=256)poolowner_list(p)[at+j]=first+r+j*nrows;
   at+=jobs;
  }
 }
'''+s[b:]
assert not (p/'poolstripe.cuh').exists();(p/'poolstripe.cuh').write_text(s);shutil.copytree(p/'include_poolrows',p/'include_poolstripe');(p/'include_poolstripe/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_poolstripe.py').write_text((p/f'{n}_poolrows.py').read_text().replace('poolrows','poolstripe'))
(p/'poolstripe_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'mechanism':'stripe each actual pool packet interval among its query row owners; balanced row compute; immutable packet list'},indent=2))
