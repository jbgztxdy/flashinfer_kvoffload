from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'clean.cuh').read_text()
start=s.index(' // Eight lanes own one complete1152Bpage:');end=s.index('\n FAR_EMU_TILE(2,b,',start)
s=s[:start]+''' // Flatten the fixed20-page packet across all256threads:1440 uint4s,
 // five full rounds plus160 tail threads. Ownership/bytes/publication unchanged.
 uint64_t policy;asm volatile("createpolicy.fractional.L2::evict_first.b64 %0,1.0;":"=l"(policy));
#pragma unroll
 for(uint32_t v=t;v<20*72;v+=256){
  const uint32_t group=v/72,word=v%72;
  if(chunk*20+group<s.service.count){
   const auto page=s.service.pages[b][group];
   const uint4* src=reinterpret_cast<const uint4*>(p.ckv+static_cast<int64_t>(page)*576)+word;
   const uint32_t dst=static_cast<uint32_t>(__cvta_generic_to_shared(reinterpret_cast<uint4*>(s.remote[b])+v));
   asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0],[%1],16,%2;"::"r"(dst),"l"(src),"l"(policy));
  }
 }'''+s[end:]
start=s.index(' const uint32_t group=t/8,lane=t%8,slot=chunk*20+group;',s.index('void clean_service_stores('));end=s.index('\n if(t<20&&chunk*20+t<s.service.count)',start)
s=s[:start]+'''#pragma unroll
 for(uint32_t v=t;v<20*72;v+=256){
  const uint32_t group=v/72,word=v%72,slot=chunk*20+group;
  if(slot<s.service.count){
   const uint4 value=reinterpret_cast<const uint4*>(s.remote[b])[v];
   reinterpret_cast<uint4*>(p.staging+(p.cache_slots+static_cast<int64_t>(slot))*576)[word]=value;
   const int32_t victim=s.service.victims[b][group];
   if(victim>=0)reinterpret_cast<uint4*>(p.staging+static_cast<int64_t>(victim)*576)[word]=value;
  }
 }'''+s[end:]
assert not(p/'flat_payload.cuh').exists();(p/'flat_payload.cuh').write_text(s)
shutil.copytree(p/'include_clean',p/'include_flat_payload');(p/'include_flat_payload/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_flat_payload.py').write_text((p/f'{n}_rescale_skip.py').read_text().replace('rescale_skip','flat_payload'))
(p/'flat_payload_source.json').write_text(json.dumps(dict(sha256=hashlib.sha256(s.encode()).hexdigest(),base_sha256=hashlib.sha256((p/'clean.cuh').read_bytes()).hexdigest(),mechanism='fixed20page producer cp.async and true stage/cache stores flattened1440uint4 across256threads; metadata and protocol unchanged'),indent=2)+'\n')
