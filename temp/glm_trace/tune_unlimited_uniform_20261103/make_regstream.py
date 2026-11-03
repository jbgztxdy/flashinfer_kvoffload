"""One-time normal-only direct register streaming, preserving both attention stages."""
from pathlib import Path
import hashlib,json,re,shutil
p=Path(__file__).resolve().parent;root=p.parents[2];assert not(p/'regstream.cuh').exists()
s=(p/'clean.cuh').read_text()
a=s.index('// All256threads');s=s[:a]+re.sub(r'\b20(u?)\b',r'32\1',s[a:])
s=s.replace(' alignas(128) KV remote[1][32*576];\n CleanService service;',' alignas(128) CleanService service;')
a=s.index('__device__ __forceinline__ void clean_service_issue(');b=s.index('// Every lane has already',a)
s=s[:a]+'''__device__ __forceinline__ void clean_service_issue(const Params&,typename KTraits::SharedStorage&,uint32_t){}
'''+s[b:]
a=s.index(' const uint32_t group=t/8,lane=t%8,slot=chunk*32+group;',s.index('void clean_service_stores('));b=s.index('\n if(t<32',a)
s=s[:a]+''' const uint32_t group=t/8,lane=t%8,slot=chunk*32+group;
 if(slot<s.service.count){
  const auto page=s.service.pages[0][group];
  const uint4* src=reinterpret_cast<const uint4*>(p.ckv+static_cast<int64_t>(page)*576)+lane;
  uint4* stage=reinterpret_cast<uint4*>(p.staging+(p.cache_slots+static_cast<int64_t>(slot))*576)+lane;
  const int32_t victim=s.service.victims[0][group];
  uint4* cache=reinterpret_cast<uint4*>(p.staging+static_cast<int64_t>(victim)*576)+lane;
  uint64_t policy;asm volatile("createpolicy.fractional.L2::evict_first.b64 %0,1.0;":"=l"(policy));
#pragma unroll
  for(uint32_t v=0;v<72;v+=8){
   uint4 value;
   asm volatile("ld.global.cg.L2::cache_hint.v4.u32 {%0,%1,%2,%3},[%4],%5;":"=r"(value.x),"=r"(value.y),"=r"(value.z),"=r"(value.w):"l"(src+v),"l"(policy):"memory");
   stage[v]=value;if(victim>=0)cache[v]=value;
  }
 }'''+s[b:]
s=s.replace('CleanService','RegstreamService').replace('clean_service_','regstream_service_').replace('static_assert(regstream_service_smem == 163072);','static_assert(regstream_service_smem == 140160);')
a=s.index('    using UniformKernelTraits = RegstreamServiceTraits');b=s.index('    using StreamSourceKernelTraits',a);s=s[:a]+'#ifndef FLASHINFER_FAR_EMU\n'+s[a:b]+'#endif\n'+s[b:]
(p/'regstream.cuh').write_text(s);shutil.copytree(root/'include',p/'include_regstream');(p/'include_regstream/flashinfer/attention/mla_host_batch.cuh').write_text(s)
m=json.loads((p/'manifest.json').read_text());m['regstream_sha256']=hashlib.sha256(s.encode()).hexdigest();m['regstream_mechanism']='Normal-only 32page packets copied directly through4temporaryregs into staging+cache, no persistentpayload/farshared/asyncread; same8mathwarps and2KVstages; source L2evict_first via ld.cg hint; actualonceperunique16Bread; no prefetch extra bytes';m['regstream_sources']=['https://docs.nvidia.com/cuda/archive/12.6.0/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld'];(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(p/'bench_regstream.py').write_text((p/'bench_clean.py').read_text().replace("service_rows=20 if", "service_rows=32 if VARIANT=='regstream' else 20 if").replace("['base','clean']","['base','clean','regstream']"))
(p/'test_regstream.py').write_text((p/'test_clean.py').read_text().replace('test_clean_output','test_regstream_output'))
(p/'run_regstream.py').write_text((p/'run_clean.py').read_text().replace('clean','regstream'))
