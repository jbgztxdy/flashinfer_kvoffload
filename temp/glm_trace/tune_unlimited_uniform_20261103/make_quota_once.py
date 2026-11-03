from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'poolquota_v2.cuh').read_text();a=s.index(' if(t==0){\n  const uint32_t total=',s.index('void clean_service_initialize'));b=s.index('\n FAR_EMU_INIT',a);part=s[a:b]
part=part.replace(' if(t==0){',' if(t==0&&blockIdx.y==0){',1)
part=part.replace('auto* spill_count=spill_first+p.num_pools;', 'auto* spill_count=spill_first+p.num_pools;auto* cursors=spill_count+p.num_pools;')
part=part.replace('for(uint32_t c=0;c<gridDim.y;++c)remain[c]=q+(c<r);','for(uint32_t c=0;c<gridDim.y;++c){remain[c]=q+(c<r);cursors[c]=c*q+min(c,r);}')
part=part.replace('  s.service.produced=blockIdx.y*q+min(uint32_t(blockIdx.y),r);s.service.issues=q+(blockIdx.y<r);\n  uint32_t output=s.service.produced;','')
part=part.replace('if(best==blockIdx.y)for(uint32_t j=0;j<take;++j)poolowner_list(p)[output++]=at+j;','for(uint32_t j=0;j<take;++j)poolowner_list(p)[cursors[best]++]=at+j;')
part=part.replace('if(c==blockIdx.y)for(uint32_t j=0;j<take;++j)poolowner_list(p)[output++]=at+j;','for(uint32_t j=0;j<take;++j)poolowner_list(p)[cursors[c]++]=at+j;')
assert 'output' not in part
part+='\n cg::this_grid().sync();\n if(t==0){uint32_t total=ceil_div(unique,20u),q=total/gridDim.y,r=total%gridDim.y;s.service.produced=blockIdx.y*q+min(uint32_t(blockIdx.y),r);s.service.issues=q+(blockIdx.y<r);}\n'
s=s[:a]+part+s[b:]
assert not (p/'quota_once.cuh').exists();(p/'quota_once.cuh').write_text(s);shutil.copytree(p/'include_poolquota_v2',p/'include_quota_once');(p/'include_quota_once/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_quota_once.py').write_text((p/f'{n}_poolquota_v2.py').read_text().replace('poolquota_v2','quota_once'))
(p/'quota_once_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'base_sha256':hashlib.sha256((p/'poolquota_v2.cuh').read_bytes()).hexdigest(),'mechanism':'one CTA0 leader plans same strict quotas and writes all stable packet lists; cooperative grid sync before consumption; all work timed'},indent=2))
