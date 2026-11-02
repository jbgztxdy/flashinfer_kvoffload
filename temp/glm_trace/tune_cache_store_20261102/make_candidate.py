"""Freeze production and test streaming LRU payload stores, preserving live staging stores."""
import difflib, hashlib, json, shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda x:hashlib.sha256(x).hexdigest()
base=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes()
assert sha(base)=='48c2a724548c2ac554400b26592a1212e159cd279cd1a48da191b5139cd86f73'
assert not (p/'base.cuh').exists()
s=base.decode();at=s.index('__device__ __forceinline__ void fetch_stream_source_pages(');end=s.index('template <typename Q, typename KV, typename O, typename Id>\nstruct StreamSourceTraits',at)
f=s[at:end]
a='''            reinterpret_cast<uint4*>(
                params.staging +
                static_cast<int64_t>(victim) * ROW)[v % VECS_PER_ROW] =
                buffer[buf * CHUNK_VECS + v];'''
b='''            auto* cache_destination = reinterpret_cast<uint4*>(
                params.staging + static_cast<int64_t>(victim) * ROW) + v % VECS_PER_ROW;
            const uint4 payload = buffer[buf * CHUNK_VECS + v];
            // The live launch reads the staging copy; cache payload is for future launches.
            asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                         :: "l"(cache_destination), "r"(payload.x), "r"(payload.y),
                            "r"(payload.z), "r"(payload.w) : "memory");'''
assert f.count(a)==1;f=f.replace(a,b)
a='''        reinterpret_cast<uint4*>(params.staging + static_cast<int64_t>(victim) * ROW)[v % VECS_PER_ROW] =
            source[v];'''
b='''        auto* cache_destination = reinterpret_cast<uint4*>(
            params.staging + static_cast<int64_t>(victim) * ROW) + v % VECS_PER_ROW;
        const uint4 payload = source[v];
        asm volatile("st.global.cs.v4.u32 [%0], {%1, %2, %3, %4};"
                     :: "l"(cache_destination), "r"(payload.x), "r"(payload.y),
                        "r"(payload.z), "r"(payload.w) : "memory");'''
assert f.count(a)==1;f=f.replace(a,b)
candidate=s[:at]+f+s[end:]
math=(root/'include/flashinfer/attention/mla.cuh').read_bytes()
for v,data in [('base',base),('cache_store',candidate.encode())]:
    (p/(v+'.cuh')).write_bytes(data);(p/(v+'_mla.cuh')).write_bytes(math)
    tree=p/('include_'+v);shutil.copytree(root/'include',tree)
    (tree/'flashinfer/attention/mla_host_batch.cuh').write_bytes(data)
old=p.parent/'tune_stream_source_20261101'
for name in ['bench_gpu3_native.py','resources_gpu3.py','experiment.json','plans.cpp','test_cofetch_progress.py']:
    shutil.copy2(old/name,p/name)
compile_text=(old/'compile.py').read_text().replace("dtypes=(torch.bfloat16,) if os.environ.get('FLASHINFER_EXTRA_CUDAFLAGS') else (torch.bfloat16,torch.float16)","dtypes=(torch.bfloat16,)")
(p/'compile.py').write_text(compile_text)
(p/'candidatepaths.py').write_text((old/'candidatepaths.py').read_text().replace("('base','stream_source','claim_store')","('base','cache_store')"))
(p/'candidate.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),candidate.splitlines(True),fromfile='base.cuh',tofile='cache_store.cuh')))
record=dict(base_sha256=sha(base),candidate_sha256=sha(candidate.encode()),math_sha256=sha(math),mechanism='Only dedicated fetch LRU payload stores use st.global.cs.v4.u32; live staging stores unchanged.',scope='Existing large SM80 cached fully staged dispatch only.',invariants='Same attention, exact launch LRU, bytes, source hint, FAR hooks, synchronization, fetch geometry and launch.',adopted=False)
(p/'manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
