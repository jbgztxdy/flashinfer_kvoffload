"""Freeze the measured direct math, normal BF16 only. Existing paths byte-preserved."""
from pathlib import Path
import json,hashlib,shutil,difflib
p=Path(__file__).resolve().parent;root=p.parents[2];assert not(p/'retained.cuh').exists()
s=(p/'direct_pv.cuh').read_text();a=s.index('template <typename KTraits>\n__device__ __forceinline__ void direct_pv_pv(');b=s.index('// partial_capacity bounds',a)
s=s[:a]+'#ifndef FLASHINFER_FAR_EMU\n// Fast normal-build BF16 inplace math; exact native owner cachewriteback.\n'+s[a:b]+'#endif  // !FLASHINFER_FAR_EMU\n\n'+s[b:]
a=s.index('    using DirectPVKernelTraits');b=s.index('    using PVFeatureKernelTraits',a);x=s[a:b]
x=x.replace('    const bool direct_pv = !params.stage', '    const bool direct_pv = std::is_same<DTypeQ, nv_bfloat16>::value &&\n        std::is_same<DTypeKV, nv_bfloat16>::value && std::is_same<DTypeO, nv_bfloat16>::value &&\n        !params.stage')
s=s[:a]+'#ifndef FLASHINFER_FAR_EMU\n'+x+'#endif  // !FLASHINFER_FAR_EMU\n'+s[b:]
(p/'retained.cuh').write_text(s);shutil.copytree(root/'include',p/'include_retained');(p/'include_retained/flashinfer/attention/mla_host_batch.cuh').write_text(s)
m=json.loads((p/'manifest.json').read_text());m['retained_sha256']=hashlib.sha256(s.encode()).hexdigest();m['retained_scope']='Only normal BF16 directcached>=4SMrows/one64headtile/whole row path, same existing guardfor SMshared<200000. EMU/newFP16/staged guardfalse. No routingdefault change.';(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(p/'retained.diff').write_text(''.join(difflib.unified_diff((p/'base.cuh').read_text().splitlines(True),s.splitlines(True),fromfile='production_before',tofile='retained',n=2)))
(p/'bench_retained.py').write_text((p/'bench_direct_pv.py').read_text().replace("'direct_pv']","'direct_pv','retained']"))
(p/'test_retained.py').write_text((p/'test_direct_pv.py').read_text().replace('test_direct_pv_output','test_retained_output'))
(p/'run_retained.py').write_text((p/'run_direct_pv.py').read_text().replace('direct_pv','retained'))
f=p/'probe.py';s=f.read_text().replace("'trycache','clean']","'trycache','clean','retained']").replace("'direct_pv':'BatchMLADirectPVKernel'","'direct_pv':'BatchMLADirectPVKernel','retained':'BatchMLADirectPVKernel'");f.write_text(s)
