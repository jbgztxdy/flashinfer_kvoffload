"""Remove persistent register payload with the largest three-shared-stage packing."""
import hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
control=(p/'control_qp.cuh').read_text();old=(p/'qks.cuh').read_text();helpers=(p/'qks_generated.inc').read_text()
assert sha(p/'qks.cuh')=='cabe65e10626720653dc8596c3cf5347833b9ff902729ae5a40bc4bd4bc9f4b4'
limits=json.loads((p/'resource_limits.json').read_text());assert limits['packing_review']['largest_three_equal_shared_rows_if_static_same']==7
start=helpers.index('// Shared subjob metadata;');end=helpers.index('template <typename KTraits, typename Params>\n__device__ __forceinline__ void qks_produce(',start)
helpers=helpers[:start]+(p/'service_shared.inc').read_text()+'\n'+helpers[end:]
helpers=helpers.replace('uint4 service_payload[2][576]','uint4 service_payload[3][504]')
key='    qks_all_sync();  // PV-lo has released stage1; stage0 loading overlaps it.'
assert helpers.count(key)==1
helpers=helpers.replace(key,'    if (hi >= 1) qks_service_step<KTraits>(params, storage, metadata, ctx, epoch, count);  // One subjob during PV-lo.\n'+key)
def rename(s):return s.replace('qks_','qkss_').replace('QKS','QKSShared')
helpers=rename(helpers)
def function(source,name):
 pos=source.index('void '+name+'(');start=source.rfind('template <',0,pos);first=source.index('{',pos);depth=1;end=first+1
 while depth:depth+=(source[end]=='{')-(source[end]=='}');end+=1
 return source[start:end]+'\n\n'
kernel=rename(function(old,'BatchMLAQKSServiceKernel'))
insert=control.index('// partial_capacity bounds the item states')
s=control[:insert]+helpers+kernel+'\n'+control[insert:]
a=old.index('    using QKSKernelTraits = QKSTraits<');b=old.index('    using QKPVKernelTraits = QKPVTraits<',a)
dispatch=old[a:b]
new=dispatch.replace('QKS','QKSShared').replace('qks','qks_shared')
new=new.replace('static_assert(qks_shared_smem == 158464);','static_assert(qks_shared_smem == 164224);')
new=new.replace('smem_limit_per_sm >= qks_shared_smem + 4096','smem_limit_per_sm >= qks_shared_smem')
key='      FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(qks_shared_kernel,'
index=new.index(key)
attrs='''      cudaFuncAttributes qks_shared_attributes{};
      int qks_shared_max_optin = 0;
      FLASHINFER_CUDA_CALL(cudaFuncGetAttributes(&qks_shared_attributes, qks_shared_kernel));
      FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&qks_shared_max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
      if (qks_shared_smem + qks_shared_attributes.sharedSizeBytes <= static_cast<size_t>(qks_shared_max_optin)) {
'''
new=new[:index]+attrs+new[index:]
index=new.rfind('    }\n');new=new[:index]+'      }\n'+new[index:]
anchor='    using QKPVKernelTraits = QKPVTraits<'
assert s.count(anchor)==1;s=s.replace(anchor,new+anchor,1)
f=p/'qks_shared.cuh'
if f.exists():assert f.read_text()==s
else:f.write_text(s)
shutil.copy2(p/'base_mla.cuh',p/'qks_shared_mla.cuh')
d=p/'include_qks_shared'
if not d.exists():
 shutil.copytree(root/'include',d);(d/'flashinfer/attention/mla_host_batch.cuh').write_text(s)
assert (d/'flashinfer/attention/mla_host_batch.cuh').read_text()==s
(p/'qks_shared_generated.inc').write_text(helpers)
m=dict(variant='qks_shared',candidate_sha256=sha(f),control_sha256=sha(p/'control_qp.cuh'),math_sha256=sha(p/'base_mla.cuh'),
 parent_service_sha256=sha(p/'qks.cuh'),candidate_adopted=False,goal_complete=False,
 reason='Observed oldserviceREG168/STACK152 and74staticLDL-STLsites. Move thirdpayload from registers to shared memory; driver166912Bblock limit and measured1408Bstatic permit largest equal three-stage packing7rows. Not a parameter sweep.',
 dynamic_shared_bytes=164224,compute_bytes=140032,payload_shared_bytes=3*7*1152,register_payload_bytes=0,
 progress='One bounded subjob per actual PV tile(two per paired tile,one odd),plus host-wait/enddrain. Increased from originaloneperjointpair to keep7row producer throughput with the smaller stage.',
 jobs='Ten subjobs per64rowchunk:nine7row plus one1row;lastsubjobboundedbychunk/count and only publishesafterallsubcopies. All64rowflag semantics/FARpost-readhooks remain.',
 policy='SameupfrontLRU/gridbarrier/staticall108workers as oldservice;nominal21inflightrows/CTA(2268total),lowerwhen1rowtailstagepresent,notold2592ororiginal3072.',
 dispatch='Query actual cudaFuncAttributes.sharedSizeBytes and device maxblockoptin before setting164224B. Old+4096conservative gate would reject a fitting kernel; actualdispatchmustbeprobed.',
 original_codegen_required=dict(EMU=40,normal=80,base='QP1252'))
(p/'shared_manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(m)
