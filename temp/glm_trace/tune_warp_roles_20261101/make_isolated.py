"""Isolate role helpers after the first build changed unrelated original instruction text."""
import hashlib,json,re,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
base=(p/'base.cuh').read_text();canonical=(p/'base_mla.cuh').read_text()
first=(p/'warp_roles.cuh').read_text();half=(p.parent/'tune_heads_20261031/half_heads_mla.cuh').read_text()
def function(s,name):
    m=re.search(r'\bvoid '+re.escape(name)+r'\(',s)
    if m is None:m=re.search(r'\buint32_t '+re.escape(name)+r'\(',s)
    assert m,name
    begin=s.rfind('template <',0,m.start());a=s.index('{',m.end());depth=1;b=a+1
    while depth:
        depth+=(s[b]=='{')-(s[b]=='}');b+=1
    return s[begin:b]
mathnames=['load_q','compute_qk_','logits_mask_','update_mdo_states_','compute_mla_qk','compute_mla_pv','normalize_d_','write_o']
hostnames=['wait_host_batch_tile','host_batch_tile_row','read_host_batch_pages','load_host_batch_kv','writeback_host_batch_tile','attend_host_batch_tiles','run_host_batch_item']
renames={n:'warp_role_'+n for n in mathnames+hostnames}
def rename(s):
    for a,b in renames.items():s=re.sub(r'\b'+re.escape(a)+r'\b',b,s)
    return s
helper='''// Only the spatial-role clones below use these; every original helper stays literal.
template <typename KTraits>
__device__ __forceinline__ uint32_t warp_role_group() { return threadIdx.z - 2; }
template <typename KTraits>
__device__ __forceinline__ void warp_role_sync() { asm volatile("bar.sync 1, 128;" ::: "memory"); }

'''
clones=[]
for name in mathnames:
    s=function(half,name).replace('threadIdx.z','warp_role_group<KTraits>()').replace('__syncthreads();','warp_role_sync<KTraits>();')
    clones.append(rename(s))
# Define clones inside the same namespace, before its final close, without editing originals.
key='}  // namespace mla'
assert canonical.count(key)==1
math=canonical.replace(key,helper+'\n\n'.join(clones)+'\n\n'+key)
clones=[]
for name in hostnames:
    s=function(base,name).replace('threadIdx.z','warp_role_group<KTraits>()').replace('__syncthreads();','warp_role_sync<KTraits>();')
    s=s.replace('kHostBatchTileRows<KTraits>','warp_role_tile_rows<KTraits>')
    if name=='host_batch_tile_row':
        s=s.replace('return KTraits::NUM_MMA_KV == 1 ? row : row + (warp_role_group<KTraits>() + j * 2) * 16;',
                    'return row + (warp_role_group<KTraits>() + j * 2) * 8;')
    clones.append(rename(s))
rolehelpers='template <typename KTraits>\nconstexpr uint32_t warp_role_tile_rows = KTraits::CTA_TILE_KV / 16;\n\n'+'\n\n'.join(clones)+'\n\n'
# The full-CTA lifecycle was already statically repaired before its CPU build; retain it.
a=first.index('template <typename KTraits, typename ComputeTraits, typename Params>\n__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLAWarpRoleKernel(')
b=first.index('template <typename KTraits, typename Params, bool ROW_MERGE>\n__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLAHostBatchKernel(',a)
kernel=first[a:b]
inc=(p/'warp_roles.inc').read_text().replace('run_host_batch_item<KTraits, Params, false>','warp_role_run_host_batch_item<KTraits, Params, false>')
insert=base.index('template <typename KTraits, typename Params, bool ROW_MERGE>\n__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLAHostBatchKernel(')
s=base[:insert]+rolehelpers+inc+'\n'+kernel+'\n'+base[insert:]
a=first.index('template <typename Q, typename KV, typename O, typename Id>\nstruct WarpRoleComputeTraits')
b=first.index('constexpr uint32_t kHostBatchFetchRows',a)
s=s.replace('constexpr uint32_t kHostBatchFetchRows',first[a:b]+'constexpr uint32_t kHostBatchFetchRows',1)
a=first.index('    using FullTraits = KernelTraits<false, 2, true, 512, 64, 64, 32,')
b=first.index('    DISPATCH_SMEM_CONFIG(',a)
key='    DISPATCH_SMEM_CONFIG(smem_limit_per_sm, NUM_STAGES, CTA_TILE_KV, QK_SHARD, {'
s=s.replace(key,first[a:b]+key,1)
v='warp_roles_isolated'
for filename,value in [(v+'.cuh',s),(v+'_mla.cuh',math)]:
    f=p/filename
    if f.exists():assert f.read_text()==value
    else:f.write_text(value)
for suffix in ['', '_normal']:
    d=p/('include_'+v+suffix)
    if not d.exists():
        shutil.copytree(root/'include',d)
        (d/'flashinfer/attention/mla_host_batch.cuh').write_text(s)
        (d/'flashinfer/attention/mla.cuh').write_text(math)
    assert sha((d/'flashinfer/attention/mla_host_batch.cuh').read_text())==sha(s)
    assert sha((d/'flashinfer/attention/mla.cuh').read_text())==sha(math)
# Original body text must remain unchanged, beyond compiler-generated proofs.
for name in mathnames:assert function(canonical,name)==function(math,name),name
for name in hostnames:assert function(base,name)==function(s,name),name
m=json.loads((p/'manifest.json').read_text())
m.update(variant=v,predecessor_host_sha256=m['candidate_sha256'],predecessor_math_sha256=m['candidate_math_sha256'],
    candidate_sha256=sha(s),candidate_math_sha256=sha(math),
    correction='Original generic34/68 exact instruction-text gate failed in v1 (commuted LOP3 operands/reuse hints observed). Clone role-specific math/host helpers and keep all original definitions literal; do not weaken auditor or overwrite v1 frozen input. No v1 performance launch.')
(p/'isolated_manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
