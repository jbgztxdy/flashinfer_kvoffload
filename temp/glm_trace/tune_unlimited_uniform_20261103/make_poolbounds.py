from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'poolparallel.cuh').read_text();a=s.index(' uint32_t total=ceil_div(unique,20u),begin=');b=s.index('\n FAR_EMU_INIT',a)
s=s[:a]+''' __syncthreads();
 if(t==0){uint32_t before=0,jobs=0;
  for(uint32_t pool=0;pool<p.num_pools;++pool){uint32_t first=ceil_div(p.row_slot_base[p.pool_row_start[pool]],20u),last=ceil_div(p.row_slot_base[p.pool_row_start[pool+1]],20u);uint32_t owner=pool%gridDim.y;before+=(owner<blockIdx.y)*(last-first);jobs+=(owner==blockIdx.y)*(last-first);}
  s.service.produced=before;s.service.issues=jobs;
 }
 __syncthreads();uint32_t at=s.service.produced;
 for(uint32_t pool=blockIdx.y;pool<p.num_pools;pool+=gridDim.y){uint32_t first=ceil_div(p.row_slot_base[p.pool_row_start[pool]],20u),last=ceil_div(p.row_slot_base[p.pool_row_start[pool+1]],20u);
  for(uint32_t i=t;i<last-first;i+=256)poolowner_list(p)[at+i]=first+i;
  at+=last-first;
 }
'''+s[b:]
assert not (p/'poolbounds.cuh').exists();(p/'poolbounds.cuh').write_text(s);shutil.copytree(p/'include_poolparallel',p/'include_poolbounds');(p/'include_poolbounds/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_poolbounds.py').write_text((p/f'{n}_poolparallel.py').read_text().replace('poolparallel','poolbounds'))
(p/'poolbounds_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'mechanism':'derive packet intervals from existing row_slot_base at pool_row_start; O(pools) per CTA, parallel list writes; same owners'},indent=2))
