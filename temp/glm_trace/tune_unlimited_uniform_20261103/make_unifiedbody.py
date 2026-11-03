from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'clean.cuh').read_text()
s=s.replace('template <bool SERVICE,typename KTraits,typename Params>','template <typename KTraits,typename Params>')
s=s.replace('template <bool ACTIVE,typename KTraits,typename Params>','template <typename KTraits,typename Params>').replace('template <bool ACTIVE,typename KTraits, typename Params>','template <typename KTraits, typename Params>').replace('template <bool ACTIVE,typename KTraits, typename Params, bool ROW_MERGE>','template <typename KTraits, typename Params, bool ROW_MERGE>')
for fn in ['pv','update_joint_host_mdo','attend_joint_host_tiles','run_joint_host_item']:
 name='clean_service_'+fn
 needle='void '+name+'(';assert s.count(needle)==1;s=s.replace(needle,needle+'bool active,')
s=s.replace('if constexpr(SERVICE)','if(active)').replace('if constexpr(ACTIVE)','if(active)')
s=s.replace('clean_service_update_joint_host_mdo<ACTIVE,KTraits>(', 'clean_service_update_joint_host_mdo<KTraits>(active,')
s=s.replace('clean_service_pv<ACTIVE,KTraits>(', 'clean_service_pv<KTraits>(active,').replace('clean_service_pv<false,KTraits>(', 'clean_service_pv<KTraits>(false,')
s=s.replace('clean_service_attend_joint_host_tiles<ACTIVE,KTraits>(', 'clean_service_attend_joint_host_tiles<KTraits>(active,')
a='''   if((blockIdx.y+storage.service.done*gridDim.y)*20<storage.service.count)
     clean_service_run_joint_host_item<true,KTraits,Params,false>(params,storage,smem,scratch,row,scores,output,m,d);
   else clean_service_run_joint_host_item<false,KTraits,Params,false>(params,storage,smem,scratch,row,scores,output,m,d);'''
b='''   bool active=(blockIdx.y+storage.service.done*gridDim.y)*20<storage.service.count;
   clean_service_run_joint_host_item<KTraits,Params,false>(active,params,storage,smem,scratch,row,scores,output,m,d);'''
assert a in s;s=s.replace(a,b)
assert 'ACTIVE' not in s and 'SERVICE)' not in s
assert not (p/'unifiedbody.cuh').exists();(p/'unifiedbody.cuh').write_text(s);shutil.copytree(p/'include_clean',p/'include_unifiedbody');(p/'include_unifiedbody/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_unifiedbody.py').write_text((p/f'{n}_clean.py').read_text().replace('clean','unifiedbody'))
f=p/'bench_unifiedbody.py';f.write_text(f.read_text().replace('latency_ns=0,emulated=True','latency_ns=0,emulated=False'))
(p/'unifiedbody_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'mechanism':'one runtime-active math body instead of active/pure template duplication; same row-level active decision and all synchronization retained'},indent=2))
