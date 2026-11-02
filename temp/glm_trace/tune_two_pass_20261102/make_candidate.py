"""Whole-item QK then PV, with scores in an enlarged existing partial workspace."""
import difflib,hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2];sha=lambda x:hashlib.sha256(x).hexdigest()
b=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes();assert sha(b)=='754706f8d8fba14a361ce6dcfc64a358446e5df36b225f392fa8a3e7756ded79';assert not(p/'base.cuh').exists()
s=b.decode();math=(root/'include/flashinfer/attention/mla.cuh').read_bytes()
start=s.index('template <typename Q, typename KV, typename O, typename Id>\nstruct StreamSourceTraits');end=s.index('// partial_capacity bounds',start)
section=s[start:end].replace('StreamSource','TwoPass').replace('stream_source_','two_pass_')
startfn=section.index('template <typename KTraits, typename Params>\n__device__ __forceinline__ void two_pass_attend_joint_host_tiles(')
endfn=section.index('template <typename KTraits, typename Params, bool ROW_MERGE>',startfn)
section=section[:startfn]+(p/'two_pass.inc').read_text()+'\n'+section[endfn:]
# Initialize output accumulators only after QK (the score pass never uses O).
a='''  init_states_<KTraits>(o_frag, m, d);
  __syncthreads();'''
assert section.count(a)==1;section=section.replace(a,'  __syncthreads();')
a='''  }
  __syncthreads();
  normalize_d_<KTraits>(&storage, 0, o_frag, m, d);'''
z='''  } else {
    init_states_<KTraits>(o_frag, m, d);
  }
  __syncthreads();
  normalize_d_<KTraits>(&storage, 0, o_frag, m, d);'''
assert section.count(a)==1;section=section.replace(a,z)
# Clone source-fetch helper so original instantiated kernels remain literal.
fstart=s.rindex('template <typename KTraits, typename Params>',0,s.index('__device__ __forceinline__ void fetch_stream_source_pages('));fend=s.index('template <typename Q, typename KV, typename O, typename Id>\nstruct StreamSourceTraits',fstart)
fetch=s[fstart:fend].replace('fetch_stream_source_pages','fetch_two_pass_pages')
section=section.replace('fetch_two_pass_pages','fetch_two_pass_pages')
# The blind name normalization produced fetch_two_pass_pages in the cloned kernel.
assert 'fetch_two_pass_pages<KTraits>' in section
at=s.index('    using StreamSourceKernelTraits =');dend=s.index('    using CachedPEKernelTraits =',at)
dispatch=s[at:dend].replace('StreamSource','TwoPass').replace('stream_source','two_pass')
needle='''    const bool two_pass = params.stage'''
assert needle in dispatch
dispatch=dispatch.replace('''    if (two_pass) {''','''    const uint64_t score_bytes = static_cast<uint64_t>(params.num_rows) * max_kv_len * 64 * sizeof(float);
    const uint64_t workspace_bytes = static_cast<uint64_t>(partial_capacity) * 64 * 512 * sizeof(DTypeO);
    if (two_pass && workspace_bytes >= score_bytes) {''')
s=s[:end]+fetch+section+s[end:];at=s.index('    using StreamSourceKernelTraits =');s=s[:at]+dispatch+s[at:]
for v,data in [('base',b),('two_pass',s.encode())]:
 (p/(v+'.cuh')).write_bytes(data);(p/(v+'_mla.cuh')).write_bytes(math)
 tree=p/('include_'+v);shutil.copytree(root/'include',tree);(tree/'flashinfer/attention/mla_host_batch.cuh').write_bytes(data)
old=p.parent/'tune_qk_cursor_20261102'
for name in ['bench_gpu3_native.py','resources_gpu3.py','experiment.json','plans.cpp','compile.py','audit_performance.py']:
 shutil.copy2(old/name,p/name)
(p/'candidatepaths.py').write_text((old/'candidatepaths.py').read_text().replace('qk_cursor','two_pass'))
for name in ['run_builds.py','run_focused.py']:
 (p/name).write_text((old/name).read_text().replace('qk_cursor','two_pass'))
(p/'candidate.diff').write_text(''.join(difflib.unified_diff(b.decode().splitlines(True),s.splitlines(True),fromfile='base.cuh',tofile='two_pass.cuh')))
record=dict(base_sha256=sha(b),candidate_sha256=sha(s.encode()),math_sha256=sha(math),mechanism='Two passes per whole attention item: QK scores in existing partial_o backing enlarged outside timing; whole-row max once, then fixed-max exp/quantizedPV/denominator. No O live in QK; no per-tile O/d rescale. KV reread and score storage increase HBM traffic.',invariants='One cooperative kernel, same prepare/exactLRU/fetch/sourcehint/FAR/actualremote1152B-perunique, same scores PEthenCKV accumulation and BF16P nativePV; reduction grouping changes. No simulator/reference/timing edits.',adopted=False)
(p/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
