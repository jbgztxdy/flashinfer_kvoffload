"""PV feature-major loop, independently from retained production."""
import difflib,hashlib,json,shutil
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[2]
sha=lambda b:hashlib.sha256(b).hexdigest()
base=(root/'include/flashinfer/attention/mla_host_batch.cuh').read_bytes()
math=(root/'include/flashinfer/attention/mla.cuh').read_bytes()
assert sha(base)=='754706f8d8fba14a361ce6dcfc64a358446e5df36b225f392fa8a3e7756ded79'
assert sha(math)=='34afad69da6bafe0a62aa1cd93db798fffe4d846a2d697aa97d6eef535cc83b3'
assert not(p/'pv_feature.cuh').exists()
s=base.decode();m=math.decode()
a=s.index('template <typename Q, typename KV, typename O, typename Id>\nstruct StreamSourceTraits');b=s.index('// partial_capacity bounds',a)
section=s[a:b].replace('StreamSource','PVFeature').replace('stream_source_','pv_feature_')
section=section.replace('fetch_pv_feature_pages<','fetch_stream_source_pages<')
assert section.count('compute_mla_pv<KTraits>')==3
section=section.replace('compute_mla_pv<KTraits>','pv_feature_pv<KTraits>')
a=m.index('template <typename KTraits>\n__device__ __forceinline__ void compute_mla_pv(');e=m.index('template <typename KTraits>',a+30)
pv=m[a:e].replace('void compute_mla_pv(','void pv_feature_pv(')
pv=pv.replace('  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;','  static_assert(KTraits::QK_SHARD && KTraits::CTA_TILE_KV == 32);\n  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;')
a=pv.index('#pragma unroll\n    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {')
e=pv.index('  } else {',a)
pv=pv[:a]+'''    // Keep both probabilities resident, then finish each O feature with both KV
    // halves. Per-output MMA order and the native P cast/rowsum/publish are kept.
    uint32_t p_frag[2][4];
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
      p_smem.ldmatrix_m8n8x4(p_smem_offset_r, p_frag[mma_kv]);
      p_smem_offset_r = p_smem.template advance_offset_by_column<2>(p_smem_offset_r, mma_kv);
    }
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 2; ++mma_d) {
      uint32_t v_frag[4];
      ckv_smem.ldmatrix_m8n8x4_trans(ckv_smem_offset_r, v_frag);
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(o_frag[mma_d], p_frag[0], v_frag);
      const uint32_t second = ckv_smem.template advance_offset_by_row<16, UPCAST_STRIDE_CKV>(ckv_smem_offset_r);
      ckv_smem.ldmatrix_m8n8x4_trans(second, v_frag);
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(o_frag[mma_d], p_frag[1], v_frag);
      ckv_smem_offset_r = ckv_smem.template advance_offset_by_column<2>(ckv_smem_offset_r, mma_d);
    }
'''+pv[e:]
s=s[:b]+pv+section+s[b:]
a=s.index('    using StreamSourceKernelTraits =');b=s.index('    using CachedPEKernelTraits =',a)
dispatch=s[a:b].replace('StreamSource','PVFeature').replace('stream_source','pv_feature')
s=s[:a]+dispatch+s[a:]
for v,data in [('base',base),('pv_feature',s.encode())]:
 (p/(v+'.cuh')).write_bytes(data);(p/(v+'_mla.cuh')).write_bytes(math)
 tree=p/('include_'+v);shutil.copytree(root/'include',tree);(tree/'flashinfer/attention/mla_host_batch.cuh').write_bytes(data)
old=p.parent/'tune_pv_rescale_20261103'
for name in ['bench_gpu3_native.py','resources_gpu3.py','experiment.json','plans.cpp','compile.py','audit_performance.py','.gitignore']:
 shutil.copy2(old/name,p/name)
for name in ['build.py','run.py','candidatepaths.py','review_codegen.py']:
 value=(old/name).read_text().replace('pv_rescale','pv_feature').replace('PVRescale','PVFeature')
 (p/name).write_text(value)
(p/'test_pv_feature.py').write_text((old/'test_pv_rescale.py').read_text().replace('pv_rescale','pv_feature'))
# Source review checks all surrounding helpers; the only changed call sites are
# the three PV calls in attend. The softmax update itself is byte-identical.
value=(old/'review_source.py').read_text().replace('pv_rescale','pv_feature').replace('PVRescale','PVFeature')
value=value.replace("'stream_source_compute_qk_pair','stream_source_run_joint_host_item'","'stream_source_compute_qk_pair','stream_source_update_joint_host_mdo','stream_source_run_joint_host_item'")
value=value.replace("'For each O feature: O <- factor*O followed by original first PV MMA(s), then original second PV MMA(s). d and P arithmetic/order unchanged. KPE->P barriers unchanged.'","'For each O feature, the two native KV MMA updates remain in order0then1. Feature iterations are interchanged. d/P, joint MDO and all barriers unchanged.'")
(p/'review_source.py').write_text(value)
(p/'candidate.diff').write_text(''.join(difflib.unified_diff(base.decode().splitlines(True),s.splitlines(True),fromfile='base.cuh',tofile='pv_feature.cuh',n=0)))
# Integer address proof for all lanes/groups/features of native32KV PV.
checks=0
for lane in range(32):
 for group in range(2):
  # k128B swizzle in16B units. Row shift16 leaves the XOR row component equal.
  offset=(lane%16)*64+((group*32+lane//16)^(lane%8))
  for feature in range(16):
   for kvhalf in range(2):
    direct=(lane%16+16*kvhalf)*64+((group*32+2*feature+lane//16)^(lane%8))
    assert offset+16*64*kvhalf==direct
    checks+=1
   offset=(offset^(2+4*(feature%2==1)))+8*(feature%4==3)
r=dict(base_sha256=sha(base),candidate_sha256=sha(s.encode()),math_sha256=sha(math),address_oracle_checks=checks,adopted=False,
 mechanism='Native32KV PV loop interchange: preload both P fragments, then perfeature consume KVhalf0and1 consecutively. Same peroutput MMA order, Pcast/rowsum/publish, softmax rescale, barriers, QK/refill/fetch/LRU/FAR. Only large cached whole-row SM80 typed dispatch.',
 hypothesis='Change output-accumulator scheduling/locality while keeping every arithmetic operation; costs4more live P registers. No parameter or unroll sweep. Independent from rejected pv_rescale.')
(p/'manifest.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
