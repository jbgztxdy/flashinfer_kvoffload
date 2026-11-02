"""Advance swizzled Q/K fragment cursors instead of rebuilding row/column offsets."""
import difflib,hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda x:hashlib.sha256(x).hexdigest()
b=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes();assert sha(b)=='48c2a724548c2ac554400b26592a1212e159cd279cd1a48da191b5139cd86f73'
assert not (p/'base.cuh').exists();s=b.decode();math=(root/'include/flashinfer/attention/mla.cuh').read_bytes()
start=s.index('__device__ __forceinline__ void stream_source_qk_pair_fragments(');end=s.index('// Each lane owns four packed words',start)
f=s[start:end]
a='''  alignas(16) uint32_t qf[4],kf[4];
#pragma unroll 4
  for (uint32_t dim=0;dim<D;++dim) {
    const uint32_t qo=q.template get_permuted_offset<Q_STRIDE>(
        warp*16+lane%16,dim*2+lane/16);
    q.ldmatrix_m8n8x4(qo,qf);
    const uint32_t ko=ka.template get_permuted_offset<K_STRIDE>(
        group*16+8*(lane/16)+lane%8,2*dim+(lane%16)/8);'''
z='''  alignas(16) uint32_t qf[4],kf[4];
  uint32_t qo=q.template get_permuted_offset<Q_STRIDE>(
      warp*16+lane%16,lane/16);
  uint32_t ko=ka.template get_permuted_offset<K_STRIDE>(
      group*16+8*(lane/16)+lane%8,(lane%16)/8);
#pragma unroll 4
  for (uint32_t dim=0;dim<D;++dim) {
    q.ldmatrix_m8n8x4(qo,qf);'''
assert f.count(a)==1;f=f.replace(a,z)
a='''      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sb[0],qf,kf);
  }
}'''
z='''      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(sb[0],qf,kf);
    qo=q.template advance_offset_by_column<2>(qo,dim);
    ko=ka.template advance_offset_by_column<2>(ko,dim);
  }
}'''
assert f.count(a)==1;f=f.replace(a,z);s=s[:start]+f+s[end:]
# Exhaustive address oracle for both current query and key row maps.
checks=0
for lane in range(32):
 for group in range(2):
  for warp in range(4):
   for row,col in [(warp*16+lane%16,lane//16),(group*16+8*(lane//16)+lane%8,(lane%16)//8)]:
    offset=row*64+(col^(row%8))
    for dim in range(32):
     assert offset==row*64+((2*dim+col)^(row%8))
     offset=(offset^(2+4*(dim%2==1)))+8*(dim%4==3);checks+=1
for v,data in [('base',b),('qk_cursor',s.encode())]:
 (p/(v+'.cuh')).write_bytes(data);(p/(v+'_mla.cuh')).write_bytes(math)
 tree=p/('include_'+v);shutil.copytree(root/'include',tree);(tree/'flashinfer/attention/mla_host_batch.cuh').write_bytes(data)
old=p.parent/'tune_query_copy_20261102'
for name in ['bench_gpu3_native.py','resources_gpu3.py','experiment.json','plans.cpp','test_cofetch_progress.py','compile.py','audit_performance.py']:
 shutil.copy2(old/name,p/name)
for name in ['candidatepaths.py','run_builds.py','run_focused.py','check_codegen.py']:
 (p/name).write_text((old/name).read_text().replace('query_copy','qk_cursor'))
(p/'candidate.diff').write_text(''.join(difflib.unified_diff(b.decode().splitlines(True),s.splitlines(True),fromfile='base.cuh',tofile='qk_cursor.cuh')))
record=dict(base_sha256=sha(b),candidate_sha256=sha(s.encode()),math_sha256=sha(math),mechanism='QK paired CKV loop computes initial Q/K swizzled addresses once and advances with the existing xor/add cursor. Same unroll4, Q/K load/MMA order, joint MDO/PV, QPE, fetch and LRU.',address_oracle_checks=checks,adopted=False)
(p/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
