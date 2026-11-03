"""Interleave the existing O rescale with first-tile PV, without changing arithmetic."""
import difflib, hashlib, json, shutil
from pathlib import Path

p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda b:hashlib.sha256(b).hexdigest()
base=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes()
math=(root/'include/flashinfer/attention/mla.cuh').read_bytes()
assert sha(base)=='754706f8d8fba14a361ce6dcfc64a358446e5df36b225f392fa8a3e7756ded79'
assert sha(math)=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
assert not(p/'pv_rescale.cuh').exists()
s=base.decode();m=math.decode()
start=s.index('template <typename Q, typename KV, typename O, typename Id>\nstruct StreamSourceTraits')
end=s.index('// partial_capacity bounds',start)
section=s[start:end].replace('StreamSource','PVRescale').replace('stream_source_','pv_rescale_')
# Keep the current fetch function and its evict-first source policy, byte-for-byte.
section=section.replace('fetch_pv_rescale_pages<','fetch_stream_source_pages<')
name='__device__ __forceinline__ void pv_rescale_update_joint_host_mdo('
a=section.index(name);b=section.index('template <typename KTraits, typename Params>',a)
update=section[a:b]
old='float (*s0)[8], float (*s1)[8], float (*o)[8], float* m, float* d)'
assert update.count(old)==1
update=update.replace(old,old[:-1]+', float* output_scale)')
old='''#pragma unroll
    for (uint32_t feature = 0; feature < KTraits::NUM_MMA_D_CKV/2; ++feature) {
      o[feature][j*2] *= factor;
      o[feature][j*2+1] *= factor;
      o[feature][j*2+4] *= factor;
      o[feature][j*2+5] *= factor;
    }'''
assert update.count(old)==1
update=update.replace(old,'    output_scale[j] = factor;')
section=section[:a]+update+section[b:]
old='    pv_rescale_update_joint_host_mdo<KTraits>(&storage, variant, scores, scores+1, o, m, d);\n    compute_mla_pv<KTraits>(&storage, hi%2, scores, d, o);'
new='''    float output_scale[2];
    pv_rescale_update_joint_host_mdo<KTraits>(&storage, variant, scores, scores+1, o, m, d, output_scale);
    pv_rescale_first_pv<KTraits>(&storage, hi%2, scores, d, o, output_scale);'''
assert section.count(old)==1;section=section.replace(old,new)
# Copy native PV into the private host header. Canonical math remains untouched.
a=m.index('template <typename KTraits>\n__device__ __forceinline__ void compute_mla_pv(')
b=m.index('template <typename KTraits>',a+30)
pv=m[a:b].replace('void compute_mla_pv(','void pv_rescale_first_pv(')
pv=pv.replace('float (*o_frag)[8]) {','float (*o_frag)[8], const float* output_scale) {')
old='''        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(o_frag[mma_d], p_frag,
                                                                             v_frag);'''
new='''        // Same FMULs as joint MDO, immediately before this feature's first PV MMA.
        // Every output element is scaled exactly once; subsequent KV MMAs stay native.
        if (mma_kv == 0) {
#pragma unroll
          for (uint32_t j = 0; j < 2; ++j) {
            o_frag[mma_d][j*2] *= output_scale[j];
            o_frag[mma_d][j*2+1] *= output_scale[j];
            o_frag[mma_d][j*2+4] *= output_scale[j];
            o_frag[mma_d][j*2+5] *= output_scale[j];
          }
        }
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(o_frag[mma_d], p_frag,
                                                                             v_frag);'''
assert pv.count(old)==1;pv=pv.replace(old,new)
# Only this sharded32-KV specialization is dispatched. Catch accidental future reuse.
pv=pv.replace('  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;','  static_assert(KTraits::QK_SHARD && KTraits::CTA_TILE_KV == 32);\n  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;')
s=s[:end]+pv+section+s[end:]
a=s.index('    using StreamSourceKernelTraits =');b=s.index('    using CachedPEKernelTraits =',a)
dispatch=s[a:b].replace('StreamSource','PVRescale').replace('stream_source','pv_rescale')
s=s[:a]+dispatch+s[a:]
for v,data in [('base',base),('pv_rescale',s.encode())]:
 (p/(v+'.cuh')).write_bytes(data);(p/(v+'_mla.cuh')).write_bytes(math)
 tree=p/('include_'+v);shutil.copytree(root/'include',tree)
 (tree/'flashinfer/attention/mla_host_batch.cuh').write_bytes(data)
old=p.parent/'tune_qk_square_20261103'
for name in ['bench_gpu3_native.py','resources_gpu3.py','experiment.json','plans.cpp','compile.py','audit_performance.py','.gitignore']:
 shutil.copy2(old/name,p/name)
for name,dest in [('build_smem.py','build.py'),('run_smem.py','run.py'),('smempaths.py','candidatepaths.py'),('test_square_smem.py','test_pv_rescale.py')]:
 value=(old/name).read_text().replace('square_smem','pv_rescale')
 if dest=='run.py':
  value=value.replace("'smempaths'","'candidatepaths'").replace("'smem_state.json'","'state.json'").replace('smem256_','focused256_').replace('smem_targeted_correctness','targeted_correctness')
 (p/dest).write_text(value)
# Limit codegen review to the two new instances, preserving the square experiment records.
value=(old/'review_codegen.py').read_text()
value=value.replace("choices=['square','square_p','square_smem'],default=['square','square_p']","choices=['pv_rescale'],default=['pv_rescale']")
value=value.replace("{'square':'Square','square_p':'SquareP','square_smem':'SquareSmem'}","{'pv_rescale':'PVRescale'}")
value=value.replace("'codegen_square_smem.json' if args.variants==['square_smem'] else 'codegen_review.json'","'codegen_review.json'")
(p/'review_codegen.py').write_text(value)
(p/'candidate.diff').write_text(''.join(difflib.unified_diff(base.decode().splitlines(True),s.splitlines(True),fromfile='base.cuh',tofile='pv_rescale.cuh',n=0)))
manifest=dict(base_sha256=sha(base),candidate_sha256=sha(s.encode()),math_sha256=sha(math),
 mechanism='Defer existing joint-MDO O FMULs to each feature of first PV mma_kv0. Same P cast/rowsum/publish, barriers, QK, second PV/refill, fetch/FAR/LRU and per-element arithmetic order. Only large cached whole-row SM80 typed dispatch.',
 hypothesis='Interleave 128 scalar O FMULs per lane with PV tensor-core operations instead of an upfront FMUL batch. No operation reduction, geometry or parameter scan; compiler schedule and paired timing decide.',adopted=False)
(p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
