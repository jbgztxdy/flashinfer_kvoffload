from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent
s=(p/'claimrow_interleave.cuh').read_text()
start=s.index('__device__ __forceinline__ void claimrow_plan(')
end=s.index('template<typename KTraits,typename Params>',start+1)
body=s[start:end]
a=' __syncthreads();'+chr(10)+'}'+chr(10)
assert body.endswith(a)
b=''' __syncthreads();
 // Keep ownership and actual bytes unchanged; issue each CTA's assigned pages
 // in ascending global packet order so early consumer tiles are not delayed by
 // the coprime permutation used solely for load balancing.
 if(capacity<=128){
  __shared__ uint32_t sort_a[128],sort_b[128];
  uint32_t width=1;while(width<capacity)width<<=1;
  if(t<width)sort_a[t]=t<capacity?claimrow_packet_list(p)[base+t]:n;
  __syncthreads();
  uint32_t* src=sort_a;uint32_t* dst=sort_b;
  for(uint32_t k=2;k<=width;k<<=1){
   for(uint32_t j=k>>1;j>0;j>>=1){
    if(t<width){
     const uint32_t x=src[t],y=src[t^j];
     const bool take_min=((t&k)==0)==((t&j)==0);
     dst[t]=take_min?min(x,y):max(x,y);
    }
    __syncthreads();
    uint32_t* tmp=src;src=dst;dst=tmp;
   }
  }
  if(t<capacity)claimrow_packet_list(p)[base+t]=src[t];
  __syncthreads();
 }
}
'''
body=body[:-len(a)]+b
s=s[:start]+body+s[end:]
name='claimrow_interleave_sorted';assert not (p/(name+'.cuh')).exists()
(p/(name+'.cuh')).write_text(s)
shutil.copytree(p/'include_clean',p/('include_'+name))
(p/('include_'+name)/'flashinfer/attention/mla_host_batch.cuh').write_text(s)
for kind in ['test','bench','run']:
    (p/(kind+'_'+name+'.py')).write_text((p/(kind+'_claimrow_interleave.py')).read_text().replace('claimrow_interleave',name))
f=p/('bench_'+name+'.py');t=f.read_text()
a='      cursor+=need;expected.extend(part)'
b='      cursor+=need;part=sorted(part);expected.extend(part)'
assert t.count(a)==1;t=t.replace(a,b)
f.write_text(t)
(p/(name+'_source.json')).write_text(json.dumps(dict(sha256=hashlib.sha256(s.encode()).hexdigest(),base_sha256=hashlib.sha256((p/'claimrow_interleave.cuh').read_bytes()).hexdigest(),mechanism='same half-local assignment, true unique remote payload/stage/cache writes and exactLRU; CTA-local bitonic sort of assigned packet IDs before service issue; no new grid barrier; capacity>128 leaves original order'),indent=2)+chr(10))
print(name,hashlib.sha256(s.encode()).hexdigest())
