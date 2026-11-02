"""Eliminate the dead shared QPE upload after item-wide QPE fragment reuse."""
import difflib,hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda x:hashlib.sha256(x).hexdigest()
b=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes();assert sha(b)=='48c2a724548c2ac554400b26592a1212e159cd279cd1a48da191b5139cd86f73'
assert not (p/'base.cuh').exists();s=b.decode();math=(root/'include/flashinfer/attention/mla.cuh').read_bytes()
start=math.decode().index('template <typename KTraits>\n__device__ __forceinline__ void load_q(');end=math.decode().index('template <typename KTraits>\n__device__ __forceinline__ void load_kv(',start)
helper=math.decode()[start:end].replace('void load_q(','void stream_source_load_q_nope(')
for line in ['  constexpr uint32_t UPCAST_STRIDE_Q_PE = KTraits::UPCAST_STRIDE_Q_PE;\n','  constexpr uint32_t NUM_MMA_D_KPE = KTraits::NUM_MMA_D_KPE;\n','  smem_t<KTraits::SWIZZLE_MODE_Q_PE> q_smem_pe(smem_storage->q_smem_pe);\n', '    DTypeQ* q_pe_ptr =\n        q_pe + q * q_pe_stride_n + r * q_pe_stride_h + (lane_idx % 8) * upcast_size<DTypeQ>();\n']:
    assert helper.count(line)==1;helper=helper.replace(line,'')
start=helper.index('#pragma unroll\n    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_KPE / 4; ++mma_d)')
end=helper.index('\n    }',start)+6
helper=helper[:start]+helper[end:]
assert 'q_smem_pe' not in helper and 'q_pe_ptr' not in helper
start=s.index('__device__ __forceinline__ void stream_source_run_joint_host_item(');end=s.index('template <typename KTraits, typename Params, bool ROW_MERGE>\n__device__ __forceinline__ void stream_source_run_joint_host_items',start)
section=s[start:end];assert section.count('load_q<KTraits>')==1
section=section.replace('load_q<KTraits>','stream_source_load_q_nope<KTraits>')
s=s[:start]+section+s[end:]
at=s.rindex('template <typename KTraits, typename Params, bool ROW_MERGE>',0,start)
s=s[:at]+helper+s[at:]
for v,data in [('base',b),('query_copy',s.encode())]:
    (p/(v+'.cuh')).write_bytes(data);(p/(v+'_mla.cuh')).write_bytes(math)
    tree=p/('include_'+v);shutil.copytree(root/'include',tree);(tree/'flashinfer/attention/mla_host_batch.cuh').write_bytes(data)
old=p.parent/'tune_cache_store_20261102'
for name in ['bench_gpu3_native.py','resources_gpu3.py','experiment.json','plans.cpp','test_cofetch_progress.py','compile.py']:
    shutil.copy2(old/name,p/name)
for name in ['candidatepaths.py','run_builds.py','run_focused.py','check_codegen.py']:
    (p/name).write_text((old/name).read_text().replace('cache_store','query_copy'))
# Include padded head stride because the direct register path supplies all QPE now.
f=(p/'run_focused.py').read_text().replace("long_fetch_queues or sparse_pool_unequal","long_fetch_queues or sparse_pool_unequal or cached_query_pe_head_stride")
(p/'run_focused.py').write_text(f)
(p/'candidate.diff').write_text(''.join(difflib.unified_diff(b.decode().splitlines(True),s.splitlines(True),fromfile='base.cuh',tofile='query_copy.cuh')))
record=dict(base_sha256=sha(b),candidate_sha256=sha(s.encode()),math_sha256=sha(math),mechanism='Only StreamSource item loader omits the unused shared QPE upload; direct item-wide QPE register loads, CKV query upload, shared storage, commit/wait, attention, exact LRU, bytes and FAR unchanged.',adopted=False)
(p/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
