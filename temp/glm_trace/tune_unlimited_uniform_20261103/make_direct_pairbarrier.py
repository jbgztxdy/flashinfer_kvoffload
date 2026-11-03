from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'retained.cuh').read_text()
for fn,needle in [('direct_pv_pv','    // wait for p_smem to be filled\n    __syncthreads();'),('direct_pv_update_joint_host_mdo','  __syncthreads();')]:
 a=s.index('void '+fn+'(');brace=s.index('{',a);depth=1;b=brace+1
 while depth:depth+=(s[b]=='{')-(s[b]=='}');b+=1
 part=s[a:b];assert part.count(needle)==1
 part=part.replace(needle,'  asm volatile("bar.sync %0,64;"::"r"(threadIdx.y+1):"memory");');s=s[:a]+part+s[b:]
assert not (p/'direct_pairbarrier.cuh').exists();(p/'direct_pairbarrier.cuh').write_text(s);shutil.copytree(p/'include_retained',p/'include_direct_pairbarrier');(p/'include_direct_pairbarrier/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_direct_pairbarrier.py').write_text((p/f'{n}_retained.py').read_text().replace('retained','direct_pairbarrier'))
(p/'direct_pairbarrier_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'mechanism':'same64thread head-pair barriers in existing direct math; all KPE reuse/load/writeback/exit barriers retained; direct bytes unchanged'},indent=2))
