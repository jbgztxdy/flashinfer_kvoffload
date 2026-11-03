"""One-time direct-path math specialization, exact native inplace cache writeback."""
from pathlib import Path
import hashlib,json,shutil
p=Path(__file__).resolve().parent;root=p.parents[2];assert not(p/'direct_pv.cuh').exists()
base=(p/'base.cuh').read_text();a=base.index('template <typename KTraits>\n__device__ __forceinline__ void pv_feature_pv(');b=base.index('// partial_capacity bounds',a)
x=base[a:b].replace('PVFeature','DirectPV').replace('pv_feature_','direct_pv_')
# Native joint traits retained, but direct params must not inherit fully-staged constants.
x=x.replace('struct DirectPVHostParams : JointQKHostParams<Params> {};','''struct DirectPVHostParams : Params {
 static constexpr bool stage=false,fully_staged=false,whole_rows=true,shared_fetch_ids=true,host_split=false;
 static constexpr uint32_t num_chunks=1;
};''')
assert 'struct DirectPVHostParams : Params' in x
x=x.replace('KTraits::CTA_TILE_KV == 32 && Params::fully_staged','KTraits::CTA_TILE_KV == 32 && !Params::fully_staged')
a=x.index('void direct_pv_attend_joint_host_tiles(');at=x.index('  typename KTraits::IdType pages',a);x=x[:at]+'  host_tiles=0; // Direct reads have no staging-ready dependency.\n'+x[at:]
old='''    cp_async::wait_group<0>();
    __syncthreads();
    direct_pv_compute_qk_pair<KTraits>'''
new='''    cp_async::wait_group<0>();
    __syncthreads();
    writeback_host_batch_tile<KTraits>(params,storage,index_base,kv_len,hi*32,hi%2);
    writeback_host_batch_tile<KTraits>(params,storage,index_base,kv_len,lo*32,lo%2);
    direct_pv_compute_qk_pair<KTraits>'''
assert old in x;x=x.replace(old,new)
x=x.replace('''  if (hi == 0) {
    direct_pv_compute_one<KTraits>''','''  if (hi == 0) {
    writeback_host_batch_tile<KTraits>(params,storage,index_base,kv_len,0,0);
    direct_pv_compute_one<KTraits>''')
s=base[:b]+x+base[b:]
# Copy original fully-staged host dispatch, changing only the new direct gate/type.
a=base.index('    using PVFeatureKernelTraits');b=base.index('    using StreamSourceKernelTraits',a);launch=base[a:b].replace('PVFeature','DirectPV').replace('pv_feature','direct_pv').replace('params.stage && !params.host_split','!params.stage && !params.host_split')
at=s.index('    using PVFeatureKernelTraits');s=s[:at]+launch+s[at:]
(p/'direct_pv.cuh').write_text(s);shutil.copytree(root/'include',p/'include_direct_pv');(p/'include_direct_pv/flashinfer/attention/mla_host_batch.cuh').write_text(s)
m=json.loads((p/'manifest.json').read_text());m['direct_pv_sha256']=hashlib.sha256(s.encode()).hexdigest();m['direct_pv_mechanism']='Normal direct inplace native prepare/LRU/owner cachewriteback + new pairedQK/QPEitemreuse/jointMDO/PVfeature math; no dedicated fetch, same8mathwarps/twoKVstages. Far reads per reference, not unique. All existing staged code preserved.';(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
x=(p/'bench_clean.py').read_text().replace("['base','clean']","['base','clean','direct_pv']").replace('latency_ns=0,emulated=True)','latency_ns=0,emulated=False)')
x=x.replace('w.host_stage=True;w.host_split=False','w.host_stage=False;w.host_split=False').replace('fetch_ctas=A.fetch_ctas,staged=True','fetch_ctas=A.fetch_ctas,staged=False')
x=x.replace("refs[t]=(w._host_batch_buffers[3]<=-(R*CAP+1)).sum()", "refs[t]=((w._host_batch_buffers[3]>=0)&((w._host_batch_buffers[3]&~(1<<30))>=near)).sum()")
x=x.replace("row=dict(pass_index=rep,", "row=dict(remote_bytes_kind='direct per reference',pass_index=rep,")
x=x.replace("        result['passes'].append(row);save(target,result)","        row['actual_far_read_bytes']=[n*1152 for n in row['reference_misses']]\n        result['passes'].append(row);save(target,result)")
(p/'bench_direct_pv.py').write_text(x)
x=(p/'test_clean.py').read_text().replace('test_clean_output','test_direct_pv_output').replace('w.host_stage=staged','w.host_stage=False');(p/'test_direct_pv.py').write_text(x)
(p/'run_direct_pv.py').write_text((p/'run_clean.py').read_text().replace('clean','direct_pv'))
