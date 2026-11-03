from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'poolrows.cuh').read_text()
a=s.index(' __syncthreads();\n if(t==0){uint32_t before=0,jobs=0;',s.index('void clean_service_initialize'));b=s.index('\n FAR_EMU_INIT',a)
s=s[:a]+''' const uint32_t total=ceil_div(unique,20u),q=total/gridDim.y,r=total%gridDim.y;
 if(t==0){s.service.produced=blockIdx.y*q+min(uint32_t(blockIdx.y),r);s.service.issues=q+(blockIdx.y<r);}
 __syncthreads();for(uint32_t j=t;j<s.service.issues;j+=256)poolowner_list(p)[s.service.produced+j]=blockIdx.y+j*gridDim.y;
'''+s[b:]
assert not (p/'listcontrol.cuh').exists();(p/'listcontrol.cuh').write_text(s);shutil.copytree(p/'include_poolrows',p/'include_listcontrol');(p/'include_listcontrol/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_listcontrol.py').write_text((p/f'{n}_poolrows.py').read_text().replace('poolrows','listcontrol'))
(p/'listcontrol_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'mechanism':'same balanced rows and packet sequence as clean; indirect list and list-control fields retained; device list construction timed'},indent=2))
