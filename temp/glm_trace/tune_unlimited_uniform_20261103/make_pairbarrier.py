from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent;s=(p/'clean.cuh').read_text();marker='template <bool SERVICE,typename KTraits,typename Params>\n__device__ __forceinline__ void clean_service_pv';a='template <bool SERVICE,typename KTraits,typename Params,bool PAIR_SYNC=false>\n__device__ __forceinline__ void clean_service_pv';assert marker in s;s=s.replace(marker,a)
a='    // wait for p_smem to be filled\n    __syncthreads();';b='''    // P consumers share only this16-head pair. KPE reuse barrier above stays CTA-wide.
    if constexpr(PAIR_SYNC)asm volatile("bar.sync %0,64;"::"r"(threadIdx.y+1):"memory");
    else __syncthreads();''';assert s.count(a)==1;s=s.replace(a,b)
a='  __syncthreads();\n  if constexpr(ACTIVE)clean_service_publish_prepare<KTraits>(params,*storage);';b='''  if constexpr(ACTIVE)__syncthreads();
  else asm volatile("bar.sync %0,64;"::"r"(threadIdx.y+1):"memory");
  if constexpr(ACTIVE)clean_service_publish_prepare<KTraits>(params,*storage);''';assert s.count(a)==1;s=s.replace(a,b)
s=s.replace('clean_service_pv<ACTIVE,KTraits>(', 'clean_service_pv<ACTIVE,KTraits,Params,!ACTIVE>(').replace('clean_service_pv<false,KTraits>(&storage, lo%2', 'clean_service_pv<false,KTraits,Params,!ACTIVE>(&storage, lo%2')
assert not (p/'pairbarrier.cuh').exists();(p/'pairbarrier.cuh').write_text(s);shutil.copytree(p/'include_clean',p/'include_pairbarrier');(p/'include_pairbarrier/flashinfer/attention/mla_host_batch.cuh').write_text(s)
for n in ['test','bench','run']:(p/f'{n}_pairbarrier.py').write_text((p/f'{n}_clean.py').read_text().replace('clean','pairbarrier'))
f=p/'bench_pairbarrier.py';f.write_text(f.read_text().replace('latency_ns=0,emulated=True','latency_ns=0,emulated=False'))
(p/'pairbarrier_source.json').write_text(json.dumps({'sha256':hashlib.sha256(s.encode()).hexdigest(),'mechanism':'pure row m_wg and P publication use64thread head-pair named barriers1..4; KPE read/write reuse and all service barriers retain CTA sync'},indent=2))
