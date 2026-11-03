from pathlib import Path
import shutil,hashlib,json
p=Path(__file__).resolve().parent;s=(p/'clean.cuh').read_text()
for fn in ['stores','publish_prepare','refill']:
 start=s.index('void clean_service_'+fn+'(');end=s.index('\n}',start)+2
 part=s[start:end]
 part=part.replace('SharedStorage& s){','SharedStorage& s,uint32_t explicit_ticket=0xffffffffu){')
 assert 'ticket=s.service.current' in part
 part=part.replace('ticket=s.service.current','ticket=explicit_ticket==0xffffffffu?s.service.current:explicit_ticket')
 s=s[:start]+part+s[end:]
a=s.index(' uint32_t t=',s.index('void clean_service_step('));b=s.index('\n}',a)
s=s[:a]+''' const uint32_t ticket=s.service.done;
 cp_async::wait_group<0>();__syncthreads();
 clean_service_stores<KTraits>(p,s,ticket);__syncthreads();
 clean_service_publish_prepare<KTraits>(p,s,ticket);__syncthreads();
 clean_service_refill<KTraits>(p,s,ticket);
'''+s[b:]
a='while((blockIdx.y+s.service.done*gridDim.y)*20<s.service.count)clean_service_step<KTraits>(p,s);'
b='while((blockIdx.y+s.service.done*gridDim.y)*20<s.service.count){clean_service_step<KTraits>(p,s);__syncthreads();}'
assert a in s;s=s.replace(a,b)
assert not (p/'ticketstep.cuh').exists();(p/'ticketstep.cuh').write_text(s);shutil.copytree(p/'include_clean',p/'include_ticketstep');(p/'include_ticketstep/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for name in ['test','bench','run']:(p/f'{name}_ticketstep.py').write_text((p/f'{name}_clean.py').read_text().replace('clean','ticketstep'))
f=p/'bench_ticketstep.py';f.write_text(f.read_text().replace('latency_ns=0,emulated=True','latency_ns=0,emulated=False'))
(p/'ticketstep_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'mechanism':'immutable explicit step ticket; trailing step barrier absorbed by next wait vote; drain keeps barrier'},indent=2))
