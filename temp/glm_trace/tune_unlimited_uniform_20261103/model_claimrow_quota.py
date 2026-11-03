from pathlib import Path
import json,random,bisect,itertools
p=Path(__file__).resolve().parent;rng=random.Random(20261104)
def plan(bases,grid):
 rows=len(bases)-1;n=(bases[-1]+19)//20
 streams=[[] for _ in range(grid)]
 for row in range(rows):streams[row%grid].extend(range((bases[row]+19)//20,(bases[row+1]+19)//20))
 counts=list(map(len,streams));cap=[n//grid+(t<n%grid) for t in range(grid)];keep=[min(a,b) for a,b in zip(counts,cap)]
 extra=[a-b for a,b in zip(counts,keep)];need=[a-b for a,b in zip(cap,keep)]
 ep=[0,*itertools.accumulate(extra)];dp=[0,*itertools.accumulate(need)];assert ep[-1]==dp[-1]
 out=[]
 for t in range(grid):
  packets=[]
  for j in range(cap[t]):
   if j<keep[t]:src=t;k=j
   else:
    g=dp[t]+j-keep[t];src=bisect.bisect_right(ep,g)-1;k=keep[src]+g-ep[src]
   # GPU will invert this stream with existing row boundaries, no full greedy planner.
   ordinal=k;packet=None
   for row in range(src,rows,grid):
    lo=(bases[row]+19)//20;hi=(bases[row+1]+19)//20
    if ordinal<hi-lo:packet=lo+ordinal;break
    ordinal-=hi-lo
   assert packet==streams[src][k];packets.append(packet)
  out.append(packets)
 assert sorted(x for a in out for x in a)==list(range(n))
 assert [len(a) for a in out]==cap
 return dict(packets=out,capacities=cap,kept_packets=sum(keep),spill_packets=sum(extra),total_packets=n)
checked=0
for grid in [2,24,108]:
 for rows in [1,31,128,432,1024]:
  for i in range(12):
   counts=[rng.randrange(4000) if i else 0 for _ in range(rows)];bases=[0,*itertools.accumulate(counts)];plan(bases,grid);checked+=1
raw=json.loads((p/'clean_claimrow_dependency_native_unlimited_f24.json').read_text());actual=plan(raw['claimrow_dependency']['row_slot_base'],108)
(p/'claimrow_quota_model.json').write_text(json.dumps(dict(random_exact_partition_cases=checked,actual=actual,scope='CPU inverse-prefix exact quota/partition proof; actuallaststep rowbases; noGPU performance/byte forecast'),indent=2)+'\n');print(checked,'partitions; actual kept',actual['kept_packets'],'spill',actual['spill_packets'])
