import difflib,hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];old=p.parent/'tune_elastic_fetch_20261102'
sha=lambda x:hashlib.sha256(x).hexdigest()
b=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes();math=(root/'include/flashinfer/attention/mla.cuh').read_bytes()
assert sha(b)=='754706f8d8fba14a361ce6dcfc64a358446e5df36b225f392fa8a3e7756ded79'
assert not(p/'square.cuh').exists()
s=b.decode();start=s.index('template <typename Q, typename KV, typename O, typename Id>\nstruct StreamSourceTraits');end=s.index('// partial_capacity bounds',start)
section=s[start:end].replace('StreamSource','Square').replace('stream_source_','square_').replace('fetch_square_pages','fetch_stream_source_pages')
section=section.replace('struct SquareTraits : JointQKTraits<Q, KV, O, Id> {};','''struct SquareTraits : JointQKTraits<Q, KV, O, Id> {
  struct SharedStorage : JointQKTraits<Q,KV,O,Id>::SharedStorage {
    alignas(16) float score_exchange[2][64*32];
  };
};''')
begin=section.index('// Two different KV tiles share each Q fragment.')
endhelper=section.index('// Two native32-row score tiles share one maximum',begin)
# Keep native one-tile QK on the odd tail, using the original shared QPE copy.
section=section[:begin]+(p/'square.inc').read_text()+'\n'+section[endhelper:]
section=section.replace('alignas(16) uint32_t qpe[4][4];','alignas(16) uint32_t qpe[8][4];')
section=section.replace('square_compute_one<KTraits>(storage, qpe, scores);','compute_mla_qk<KTraits>(&storage, 0, scores);')
at=s.index('    using StreamSourceKernelTraits =');dend=s.index('    using CachedPEKernelTraits =',at)
dispatch=s[at:dend].replace('StreamSource','Square').replace('stream_source','square')
dispatch=dispatch.replace('static_assert(square_smem == 147968);','static_assert(square_smem == 164352);')
dispatch=dispatch.replace('smem_limit_per_sm >= square_smem + 4096','smem_limit_per_sm >= square_smem + 2304')
needle='''      FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(square_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, square_smem));'''
assert dispatch.count(needle)==1
dispatch=dispatch.replace(needle,'''      cudaFuncAttributes actual;int block_optin;
      FLASHINFER_CUDA_CALL(cudaFuncGetAttributes(&actual,square_kernel));
      FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&block_optin,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
      if(square_smem+actual.sharedSizeBytes>static_cast<size_t>(block_optin))return cudaErrorInvalidConfiguration;
'''+needle)
s=s[:end]+section+s[end:];at=s.index('    using StreamSourceKernelTraits =');s=s[:at]+dispatch+s[at:]
for v,data in [('base',b),('square',s.encode())]:
 (p/(v+'.cuh')).write_bytes(data);(p/(v+'_mla.cuh')).write_bytes(math)
 tree=p/('include_'+v);shutil.copytree(root/'include',tree);(tree/'flashinfer/attention/mla_host_batch.cuh').write_bytes(data)
for name in ['bench_gpu3_native.py','resources_gpu3.py','experiment.json','plans.cpp','compile.py','audit_performance.py']:
 shutil.copy2(old/name,p/name)
for name in ['candidatepaths.py','build_candidate.py']:
 (p/name).write_text((old/name).read_text().replace('elastic_fetch','square'))
runner=(p.parent/'tune_global_query_20261102/run_focused.py').read_text().replace('global_query','square')
(p/'run_focused.py').write_text(runner)
test=(p.parent/'tune_global_query_20261102/test_global_query.py').read_text().replace('global_query','square')
(p/'test_square.py').write_text(test)
(p/'candidate.diff').write_text(''.join(difflib.unified_diff(b.decode().splitlines(True),s.splitlines(True),fromfile='base.cuh',tofile='square.cuh')))
record=dict(base_sha256=sha(b),candidate_sha256=sha(s.encode()),math_sha256=sha(math),
 mechanism='Four of the original8warps do32head x32KV square microtiles across a pair. Both QandK fragments reused twice;16KiBshared FP32score exchange restores native8warp score mapping, then unchanged jointMDO/PV/refill. All8warps still own128FP32output values.',
 invariants='Original PEthenCKV feature order, original32KV/2stage/64heads/256threads, exactLRU/fetch/sourcehint/FAR/bytes; canonicalmath unchanged, singlecooperative kernel. Oddtail uses originalnativeQK.',
 risks='Sharedscore writes/barrier/reads and4activeQKwarps may outweigh LDSM saving; 32QPEpackedregisters/32score accumulators may spill. Dynamicshared164352B plus actualstatic must fit optin.',adopted=False)
(p/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
