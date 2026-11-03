from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'clean.cuh').read_text()
for fn in ['pv','update_joint_host_mdo']:
 a=s.index('void clean_service_'+fn+'(');b=s.index(' {',a);part=s[a:b];assert 'const Params& params)' in part;part=part.replace('const Params& params)','const Params& params,bool service_live=true)');s=s[:a]+part+s[b:]
s=s.replace('if constexpr(SERVICE)clean_service_refill','if constexpr(SERVICE)if(service_live)clean_service_refill').replace('if constexpr(ACTIVE)clean_service_publish_prepare','if constexpr(ACTIVE)if(service_live)clean_service_publish_prepare')
a='    if constexpr(ACTIVE)if(service_tid==0)storage.service.current=storage.service.done;'
b='    bool service_live=false;\n    if constexpr(ACTIVE)service_live=(blockIdx.y+storage.service.done*gridDim.y)*20<storage.service.count;\n    if constexpr(ACTIVE)if(service_live&&service_tid==0)storage.service.current=storage.service.done;'
assert a in s;s=s.replace(a,b)
s=s.replace('if constexpr(ACTIVE)clean_service_stores<KTraits>(params,storage);','if constexpr(ACTIVE)if(service_live)clean_service_stores<KTraits>(params,storage);')
s=s.replace('scores, scores+1, o, m, d,params);','scores, scores+1, o, m, d,params,service_live);').replace('hi%2, scores, d, o,params);','hi%2, scores, d, o,params,service_live);')
assert not (p/'midrow_release.cuh').exists();(p/'midrow_release.cuh').write_text(s);shutil.copytree(p/'include_clean',p/'include_midrow_release');(p/'include_midrow_release/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_midrow_release.py').write_text((p/f'{n}_clean.py').read_text().replace('clean','midrow_release'))
f=p/'bench_midrow_release.py';f.write_text(f.read_text().replace('latency_ns=0,emulated=True','latency_ns=0,emulated=False'))
(p/'midrow_release_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'mechanism':'one live predicate each active pair guards all payload service helpers after own work ends within row; original pure row specialization retained; all barriers and ready waits retained'},indent=2))
