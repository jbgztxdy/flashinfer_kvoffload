"""Compile separate active-service/pure-math row bodies after localfetch completion."""
from pathlib import Path
import hashlib,json,shutil
p=Path(__file__).resolve().parent;root=p.parents[2];assert not(p/'phase.cuh').exists()
s=(p/'release.cuh').read_text()
s=s.replace('template <typename KTraits,typename Params>\n__device__ __forceinline__ void release_service_update_joint_host_mdo','template <bool ACTIVE,typename KTraits,typename Params>\n__device__ __forceinline__ void release_service_update_joint_host_mdo')
s=s.replace('  release_service_publish_prepare<KTraits>(params,*storage);','  if constexpr(ACTIVE)release_service_publish_prepare<KTraits>(params,*storage);')
s=s.replace('template <typename KTraits, typename Params>\n__device__ __forceinline__ void release_service_attend_joint_host_tiles','template <bool ACTIVE,typename KTraits, typename Params>\n__device__ __forceinline__ void release_service_attend_joint_host_tiles')
s=s.replace('if(service_tid==0)storage.service.current=storage.service.done;', 'if constexpr(ACTIVE)if(service_tid==0)storage.service.current=storage.service.done;')
s=s.replace('    release_service_stores<KTraits>(params,storage);','    if constexpr(ACTIVE)release_service_stores<KTraits>(params,storage);')
s=s.replace('release_service_update_joint_host_mdo<KTraits>', 'release_service_update_joint_host_mdo<ACTIVE,KTraits>').replace('release_service_pv<true,KTraits>', 'release_service_pv<ACTIVE,KTraits>')
s=s.replace('template <typename KTraits, typename Params, bool ROW_MERGE>\n__device__ __forceinline__ void release_service_run_joint_host_item(', 'template <bool ACTIVE,typename KTraits, typename Params, bool ROW_MERGE>\n__device__ __forceinline__ void release_service_run_joint_host_item(')
s=s.replace('release_service_attend_joint_host_tiles<KTraits>', 'release_service_attend_joint_host_tiles<ACTIVE,KTraits>')
old=''' for(uint32_t row=blockIdx.y;row<params.num_rows;row+=gridDim.y)
   release_service_run_joint_host_item<KTraits,Params,false>(params,storage,smem,scratch,row,scores,output,m,d);'''
new=''' for(uint32_t row=blockIdx.y;row<params.num_rows;row+=gridDim.y) {
   if((blockIdx.y+storage.service.done*gridDim.y)*20<storage.service.count)
     release_service_run_joint_host_item<true,KTraits,Params,false>(params,storage,smem,scratch,row,scores,output,m,d);
   else release_service_run_joint_host_item<false,KTraits,Params,false>(params,storage,smem,scratch,row,scores,output,m,d);
 }'''
assert old in s;s=s.replace(old,new).replace('ReleaseService','PhaseService').replace('release_service_','phase_service_')
(p/'phase.cuh').write_text(s);shutil.copytree(root/'include',p/'include_phase');(p/'include_phase/flashinfer/attention/mla_host_batch.cuh').write_text(s)
m=json.loads((p/'manifest.json').read_text());m['phase_sha256']=hashlib.sha256(s.encode()).hexdigest();m['phase_mechanism']='AfterthisCTAownedmissesfinished,rowbranchselectscompiledpureattentionwithoutservicehotloop; remainingKVacquiresandallnecessarywritesretained';(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
s=(p/'bench_release.py').read_text().replace("['packet','release']", "['packet','release','phase']");(p/'bench_phase.py').write_text(s)
(p/'test_phase.py').write_text((p/'test_packet.py').read_text().replace('test_packet_output','test_phase_output'))
s=(p/'run_release.py').read_text().replace("p/'release_state.json'","p/'phase_state.json'").replace("'correctness_release','release'","'correctness_phase','phase'").replace("p/'test_release.py'","p/'test_phase.py'").replace("'release_native','release'","'phase_native','phase'").replace("p/'bench_release.py'","p/'bench_phase.py'");(p/'run_phase.py').write_text(s)
