import difflib,hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
s=(p/'square_p.cuh').read_text();assert not(p/'square_smem.cuh').exists()
begin=s.index('template <typename Q, typename KV, typename O, typename Id>\nstruct SquarePTraits');end=s.index('// partial_capacity bounds',begin)
section=s[begin:end].replace('SquareP','SquareSmem').replace('square_p_','square_smem_')
old='''#pragma unroll
      for(uint32_t r=0;r<4;++r) {qa[r]=qpe[dim][r];qb[r]=qpe[4+dim][r];}'''
new='''      smem_t<KTraits::SWIZZLE_MODE_Q_PE> query_pe(storage.q_smem_pe);
      const uint32_t offset=query_pe.template get_permuted_offset<KTraits::UPCAST_STRIDE_Q_PE>(heads+lane%16,2*dim+lane/16);
      query_pe.ldmatrix_m8n8x4(offset,qa);
      query_pe.ldmatrix_m8n8x4(offset+16*KTraits::UPCAST_STRIDE_Q_PE,qb);'''
assert section.count(old)==1;section=section.replace(old,new)
assert section.count('  square_smem_load_query(params, row, qpe);')==1
section=section.replace('  square_smem_load_query(params, row, qpe);','  // PE fragments come from the existing shared query copy, not persistent registers.')
s=s[:end]+section+s[end:]
at=s.index('    using SquarePKernelTraits =');dend=s.index('    using SquareKernelTraits =',at)
dispatch=s[at:dend].replace('SquareP','SquareSmem').replace('square_p','square_smem')
dispatch=dispatch.replace('bool square_smem =','bool square_smem_enabled =').replace('if (square_smem)','if (square_smem_enabled)')
s=s[:at]+dispatch+s[at:]
(p/'square_smem.cuh').write_text(s);math=(p/'base_mla.cuh').read_bytes();(p/'square_smem_mla.cuh').write_bytes(math)
tree=p/'include_square_smem';shutil.copytree(root/'include',tree);(tree/'flashinfer/attention/mla_host_batch.cuh').write_text(s)
for source,dest in [('build_direct_p.py','build_smem.py'),('run_direct_p.py','run_smem.py'),('directpaths.py','smempaths.py'),('test_square_p.py','test_square_smem.py')]:
 value=(p/source).read_text().replace('square_p','square_smem')
 if source=='run_direct_p.py':
  value=value.replace("'directpaths'","'smempaths'").replace("'direct_state.json'","'smem_state.json'").replace('direct256_','smem256_').replace('direct_targeted_correctness','smem_targeted_correctness')
 (p/dest).write_text(value)
(p/'smem_pe.diff').write_text(''.join(difflib.unified_diff((p/'square_p.cuh').read_text().splitlines(True),s.splitlines(True),fromfile='square_p.cuh',tofile='square_smem.cuh')))
sha=lambda data:hashlib.sha256(data).hexdigest()
m=dict(base_sha256=sha((p/'base.cuh').read_bytes()),control_sha256=sha((p/'square_p.cuh').read_bytes()),candidate_sha256=sha(s.encode()),math_sha256=sha(math),
 mechanism='Only square-P QPE source changes: load two head fragments per feature from existing sharedQPE instead of holding32packedwords over the entire item. Same squareQK/Ppublish/denominator/PV/barriers/metadata/fetch/FAR.',
 basis='Measured square-P regression andREG255/STACK88 versus squareREG255/STACK16; reduce long-lived32wordQPE state. No unroll/stage/fetch/threshold scan; actualstackandtimingmustdecide.',adopted=False)
(p/'smem_manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
