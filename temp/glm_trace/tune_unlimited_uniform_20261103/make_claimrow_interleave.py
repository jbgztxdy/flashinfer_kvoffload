from pathlib import Path
import shutil,json,hashlib
p=Path(__file__).resolve().parent
s=(p/'claimrow_quota.cuh').read_text()
a='return counts[owner]>cap?counts[owner]-cap:0u;'
b='uint32_t local=cap/2;return counts[owner]>local?counts[owner]-local:0u;'
assert s.count(a)==1;s=s.replace(a,b)
a='return counts[owner]<cap?cap-counts[owner]:0u;'
b='return cap-min(counts[owner],cap/2);'
assert s.count(a)==1;s=s.replace(a,b)
a=' const uint32_t capacity=q+(blockIdx.y<rem),keep=min(counts[blockIdx.y],capacity),base=blockIdx.y*q+min(blockIdx.y,rem);'
b=''' const uint32_t capacity=q+(blockIdx.y<rem),keep=min(counts[blockIdx.y],capacity/2),base=blockIdx.y*q+min(blockIdx.y,rem);
 const uint32_t last=gridDim.y-1,spill_total=excess[last]+counts[last]-min(counts[last],(q+(last<rem))/2);
 __shared__ uint32_t spill_stride;
 if(t==0){
  uint32_t stride=gridDim.y+1;
  if(spill_total>1){
   while(true){uint32_t a=spill_total,b=stride;while(b){uint32_t next=a%b;a=b;b=next;}if(a==1)break;++stride;}
  } else stride=1;
  spill_stride=stride;
 }
 __syncthreads();'''
assert s.count(a)==1;s=s.replace(a,b)
a='''   const uint32_t global=deficit[blockIdx.y]+j-keep;uint32_t lo=0,hi=gridDim.y;
   while(lo+1<hi){uint32_t mid=(lo+hi)/2;if(excess[mid]<=global)lo=mid;else hi=mid;}
   source=lo;ordinal=min(counts[source],q+(source<rem))+global-excess[source];'''
b='''   const uint32_t global=deficit[blockIdx.y]+j-keep;
   const uint32_t shuffled=static_cast<uint32_t>((static_cast<uint64_t>(global)*spill_stride)%spill_total);
   uint32_t lo=0,hi=gridDim.y;
   while(lo+1<hi){uint32_t mid=(lo+hi)/2;if(excess[mid]<=shuffled)lo=mid;else hi=mid;}
   source=lo;ordinal=min(counts[source],(q+(source<rem))/2)+shuffled-excess[source];'''
assert s.count(a)==1;s=s.replace(a,b)
# The actual maximum resident shared consumption remains within the old bound.
name='claimrow_interleave'
assert not (p/(name+'.cuh')).exists()
(p/(name+'.cuh')).write_text(s)
shutil.copytree(p/'include_clean',p/('include_'+name))
(p/('include_'+name)/'flashinfer/attention/mla_host_batch.cuh').write_text(s)
for kind in ['test','bench','run']:
    (p/(kind+'_'+name+'.py')).write_text((p/(kind+'_claimrow_quota.py')).read_text().replace('claimrow_quota',name))
f=p/('bench_'+name+'.py');t=f.read_text()
t=t.replace('import json','import json'+chr(10)+'import math',1)
a='    keep=[min(len(a),b) for a,b in zip(streams,capacities)]'
b='    keep=[min(len(a),b//2) for a,b in zip(streams,capacities)]'
assert t.count(a)==1;t=t.replace(a,b)
a='''    expected=[];cursor=0;owners=torch.empty(n,dtype=torch.int64)
    for t in range(grid):
      need=capacities[t]-keep[t];part=streams[t][:keep[t]]+overflow[cursor:cursor+need];cursor+=need
      expected.extend(part)
      if part:owners[torch.tensor(part,dtype=torch.int64)]=t
    assert expected==actual and cursor==len(overflow)'''
b='''    stride=grid+1
    if len(overflow)>1:
      while math.gcd(stride,len(overflow))!=1:stride+=1
    else:stride=1
    expected=[];cursor=0;owners=torch.empty(n,dtype=torch.int64)
    for t in range(grid):
      need=capacities[t]-keep[t]
      part=streams[t][:keep[t]]+[overflow[((cursor+j)*stride)%len(overflow)] for j in range(need)]
      cursor+=need;expected.extend(part)
      if part:owners[torch.tensor(part,dtype=torch.int64)]=t
    assert expected==actual and cursor==len(overflow)'''
assert t.count(a)==1;t=t.replace(a,b)
a="result['packet_plan_validation']=dict(exact_cpu_match=True"
b="result['packet_plan_validation']=dict(local_cap=capacities[0]//2,spill_stride=stride,exact_cpu_match=True"
assert t.count(a)==1;t=t.replace(a,b)
f.write_text(t)
(p/(name+'_source.json')).write_text(json.dumps(dict(sha256=hashlib.sha256(s.encode()).hexdigest(),base_sha256=hashlib.sha256((p/'claimrow_quota.cuh').read_bytes()).hexdigest(),mechanism='fixed half local packet quota then coprime interleaved excess permutation; exactly same unique source/stage/cache/LRU/attention; existing same CTA scan and exact balanced packet count'),indent=2)+chr(10))
print('created',name,hashlib.sha256(s.encode()).hexdigest())
