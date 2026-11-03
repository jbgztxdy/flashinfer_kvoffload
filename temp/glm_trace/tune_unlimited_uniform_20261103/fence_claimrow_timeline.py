from pathlib import Path
import json,hashlib
p=Path(__file__).resolve().parent;f=p/'claimrow_timeline.cuh';s=f.read_text();old=p/'claimrow_timeline_unfenced.cuh';assert not old.exists();old.write_bytes(f.read_bytes())
a='asm volatile("mov.u64 %0,%%globaltimer;":"=l"(now));';b='asm volatile("mov.u64 %0,%%globaltimer;":"=l"(now)::"memory");';assert s.count(a)==1;s=s.replace(a,b)
f.write_text(s);(p/'include_claimrow_timeline/flashinfer/attention/mla_host_batch.cuh').write_text(s)
f=p/'claimrow_timeline_source.json';r=json.loads(f.read_text());r['unmeasured_first_header_sha256']=r['sha256'];r['sha256']=hashlib.sha256(s.encode()).hexdigest();r['timer_compiler_memory_clobber']=True;r['first_build_scope']='unfenced header compiled only, neverGPU executed; archived before adding compiler memory clobber';f.write_text(json.dumps(r,indent=2)+'\n')
