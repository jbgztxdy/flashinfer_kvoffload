from pathlib import Path
import shutil,hashlib,json
p=Path(__file__).resolve().parent;s=(p/'clean.cuh').read_text()
a=s.index(' while(true){if(threadIdx.y==0&&threadIdx.z==0)',s.index('void clean_service_wait_tile'))
b=s.index('\n}\n// No own service',a)
s=s[:a]+''' while(true){
 bool wait=false;
 if(threadIdx.y==0&&threadIdx.z==0){uint32_t k=tile+threadIdx.x;auto page=k<len?p.remap[base+k]:0;uint32_t row=static_cast<uint32_t>(-(page+1));wait=page<0&&row>=p.cache_slots&&host_batch_ld_acquire(clean_service_flags(p)+(row-p.cache_slots)/20)!=scratch[kHostBatchScratchEpoch]+1;}
 if(!__syncthreads_or(wait))return;
 clean_service_step<KTraits>(p,s);__nanosleep(64);
 }'''+s[b:]
assert not (p/'waitvote.cuh').exists();(p/'waitvote.cuh').write_text(s);shutil.copytree(p/'include_clean',p/'include_waitvote');(p/'include_waitvote/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for name in ['test','bench','run']:(p/f'{name}_waitvote.py').write_text((p/f'{name}_clean.py').read_text().replace('clean','waitvote'))
f=p/'bench_waitvote.py';f.write_text(f.read_text().replace('latency_ns=0,emulated=True','latency_ns=0,emulated=False'))
(p/'waitvote_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'mechanism':'CTA bar.red.or replaces warp vote and shared waiting handoff; preserves CTA synchronization and acquire predicates'},indent=2))
