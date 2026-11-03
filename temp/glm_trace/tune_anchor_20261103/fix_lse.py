"""Keep accurate LSE separately from the quantized-P output denominator."""
import difflib,hashlib,json,shutil,time
from pathlib import Path
p=Path(__file__).resolve().parent;archive=p/'failed_lse';archive.mkdir()
for name in ['anchor.cuh','anchor.inc','candidate.diff','manifest.json','build_anchor.json','compile_anchor_emu.log','compile_anchor_normal.log','targeted_correctness.log','focused_state.json']:
 shutil.copy2(p/name,archive/name)
for name in ['jit_anchor','jit_anchor_normal']:(p/name).rename(archive/name)
s=(p/'anchor.cuh').read_text()
begin=s.index('template <typename Q, typename KV, typename O, typename Id>\nstruct AnchorTraits');end=s.index('// partial_capacity bounds',begin)
section=s[begin:end]
old='float (*s0)[8], float (*s1)[8], float (*o)[8], float* m, float* d) {'
assert section.count(old)==1
section=section.replace(old,'float (*s0)[8], float (*s1)[8], float (*o)[8], float* m, float* d, bool want_lse, float* exact_d) {')
section=section.replace('      d[j]*=factor;','      d[j]*=factor;\n      if(want_lse)exact_d[j]*=factor;')
old='''      s[j*2+5]=math::ptx_exp2(s[j*2+5]*scale-m[j]);'''
section=section.replace(old,old+'''
      if(want_lse)exact_d[j]+=(s[j*2]+s[j*2+1])+(s[j*2+4]+s[j*2+5]);''')
old='uint32_t kv_len, uint32_t row, int host_tiles, float (*scores)[8], float (*o)[8], float* m, float* d) {'
assert section.count(old)==1
section=section.replace(old,'uint32_t kv_len, uint32_t row, int host_tiles, float (*scores)[8], float (*o)[8], float* m, float* d, bool want_lse, float* exact_d) {')
section=section.replace('variant, scores, scores+1, o, m, d);','variant, scores, scores+1, o, m, d, want_lse, exact_d);')
section=section.replace('variant, scores, scores, o, m, d);','variant, scores, scores, o, m, d, want_lse, exact_d);')
old='  init_states_<KTraits>(o_frag, m, d);'
assert section.count(old)==1;section=section.replace(old,old+'\n  float exact_d[2]={0.f,0.f};\n  const bool want_lse=params.final_lse!=nullptr;')
old='                                     o_frag, m, d);'
assert section.count(old)==1;section=section.replace(old,'                                     o_frag, m, d, want_lse, exact_d);')
old='  normalize_d_<KTraits>(&storage, 0, o_frag, m, d);'
section=section.replace(old,old+'''
  if(want_lse && kv_len>0) {
    const uint32_t lane=threadIdx.x,warp=threadIdx.y,group=threadIdx.z;
#pragma unroll
    for(uint32_t j=0;j<2;++j) {
      exact_d[j]+=math::shfl_xor_sync(exact_d[j],2);
      exact_d[j]+=math::shfl_xor_sync(exact_d[j],1);
      if(lane%4==0)storage.d_wg[group][warp*16+j*8+lane/4]=exact_d[j];
    }
    __syncthreads();
#pragma unroll
    for(uint32_t j=0;j<2;++j)
      d[j]=storage.d_wg[0][warp*16+j*8+lane/4]+storage.d_wg[1][warp*16+j*8+lane/4];
  }
''')
s=s[:begin]+section+s[end:]
(p/'anchor.cuh').write_text(s);(p/'include_anchor/flashinfer/attention/mla_host_batch.cuh').write_text(s)
begin=section.index('// The exponent anchor');end=section.index('template <typename KTraits, typename Params>\n__device__ __forceinline__ void anchor_attend_joint_host_tiles',begin)
(p/'anchor.inc').write_text(section[begin:end])
m=json.loads((p/'manifest.json').read_text());m['candidate_sha256']=hashlib.sha256(s.encode()).hexdigest()
m['lse_correction']='Original2tests failed single-element LSE by~0.005 from BF16P rounding with nonmax anchor. For requested LSE maintain FP32 exponential denominator separately; output still normalized using matching quantizedP denominator. No tolerance changes.'
(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(p/'candidate.diff').write_text(''.join(difflib.unified_diff((p/'base.cuh').read_text().splitlines(True),s.splitlines(True),fromfile='base.cuh',tofile='anchor.cuh')))
(archive/'state.json').write_text(json.dumps(dict(timestamp=time.strftime('%FT%T%z'),numerical_failure=True,performance_runs=0,production_unchanged=True),indent=2)+'\n')
print(m['candidate_sha256'])
