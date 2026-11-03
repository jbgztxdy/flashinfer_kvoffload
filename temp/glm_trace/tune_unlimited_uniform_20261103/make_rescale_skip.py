from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent
s=(p/'clean.cuh').read_text()
a='''    d[j] *= factor;
#pragma unroll
    for (uint32_t feature = 0; feature < KTraits::NUM_MMA_D_CKV/2; ++feature) {
      o[feature][j*2] *= factor;
      o[feature][j*2+1] *= factor;
      o[feature][j*2+4] *= factor;
      o[feature][j*2+5] *= factor;
    }'''
b='''    // Exact identity factor: keep all scores/PV, skip only O/d multiply by one.
    if (factor != 1.f) {
      d[j] *= factor;
#pragma unroll
      for (uint32_t feature = 0; feature < KTraits::NUM_MMA_D_CKV/2; ++feature) {
        o[feature][j*2] *= factor;
        o[feature][j*2+1] *= factor;
        o[feature][j*2+4] *= factor;
        o[feature][j*2+5] *= factor;
      }
    }'''
start=s.index('__device__ __forceinline__ void clean_service_update_joint_host_mdo(');head=s[:start];tail=s[start:];assert tail.count(a)==1;s=head+tail.replace(a,b)
assert not (p/'rescale_skip.cuh').exists()
(p/'rescale_skip.cuh').write_text(s)
shutil.copytree(p/'include_clean',p/'include_rescale_skip')
(p/'include_rescale_skip/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_rescale_skip.py').write_text((p/f'{n}_stream_cachewrite.py').read_text().replace('stream_cachewrite','rescale_skip'))
(p/'rescale_skip_source.json').write_text(json.dumps(dict(sha256=hashlib.sha256(s.encode()).hexdigest(),base_sha256=hashlib.sha256((p/'clean.cuh').read_bytes()).hexdigest(),mechanism='skip O/d rescale only if original computed exp factor exactly equals1; no change to score/PV/prepare/service/LRU'),indent=2)+'\n')
