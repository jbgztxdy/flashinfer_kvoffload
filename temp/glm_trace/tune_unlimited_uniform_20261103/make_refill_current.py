from pathlib import Path
import shutil,hashlib,json
p=Path(__file__).resolve().parent;s=(p/'clean.cuh').read_text()
a=' clean_service_issue<KTraits>(p,s,ticket+1);if(t==0)s.service.done=ticket+1;'
b=' clean_service_issue<KTraits>(p,s,ticket+1);if(t==0){s.service.done=ticket+1;s.service.current=ticket+1;}'
assert s.count(a)==1;s=s.replace(a,b)
a=''' uint32_t t=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x;if(t==0)s.service.current=s.service.done;
 cp_async::wait_group<0>();__syncthreads();clean_service_stores<KTraits>(p,s);__syncthreads();'''
b=''' // current was published with done at the preceding PV/step barrier. Each
 // lane waits for and reads only its own remote cp.async writes; the following
 // barrier still publishes all real stage/cache stores before ready release.
 cp_async::wait_group<0>();clean_service_stores<KTraits>(p,s);__syncthreads();'''
assert s.count(a)==1;s=s.replace(a,b)
a='''    const uint32_t service_tid=(threadIdx.z*blockDim.y+threadIdx.y)*32+threadIdx.x;
    if constexpr(ACTIVE)if(service_tid==0)storage.service.current=storage.service.done;
    cp_async::wait_group<0>();'''
b='''    cp_async::wait_group<0>();'''
assert s.count(a)==1;s=s.replace(a,b)
name='refill_current';assert not (p/(name+'.cuh')).exists()
(p/(name+'.cuh')).write_text(s)
shutil.copytree(p/'include_clean',p/('include_'+name))
(p/('include_'+name)/'flashinfer/attention/mla_host_batch.cuh').write_text(s)
for kind in ['test','bench','run']:
 (p/(kind+'_'+name+'.py')).write_text((p/(kind+'_clean.py')).read_text().replace('clean',name))
f=p/('bench_'+name+'.py');t=f.read_text();assert "['base','refill_current']" in t;f.write_text(t)
(p/(name+'_source.json')).write_text(json.dumps(dict(sha256=hashlib.sha256(s.encode()).hexdigest(),base_sha256=hashlib.sha256((p/'clean.cuh').read_bytes()).hexdigest(),mechanism='publish service.current together with done at existing refill barrier; remove only first wait-step CTA barrier and redundant per-math-pair current store; same-thread cp.async remote consumption, all other release/acquire/barriers, exactLRU and payload retained; unlike ticketstep which retained first barrier'),indent=2)+chr(10))
print(name,hashlib.sha256(s.encode()).hexdigest())
