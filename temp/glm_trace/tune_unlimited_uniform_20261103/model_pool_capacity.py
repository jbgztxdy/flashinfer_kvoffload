"""CPU feasibility bound only. Never supplies GPU runtime scheduling decisions."""
from pathlib import Path
from collections import deque
import json,hashlib
p=Path(__file__).resolve().parent;f=p/'clean_pooldependency_native_unlimited_f24.json';raw=json.loads(f.read_text());counts=raw['pool_dependency']['unique_per_pool'];grid=108
prefix=[0]
for n in counts:prefix.append(prefix[-1]+n)
jobs=[(prefix[i+1]+19)//20-(prefix[i]+19)//20 for i in range(len(counts))];total=sum(jobs)
# Exact global-round-robin quotas: no extra worst-owner work.
quota=[total//grid+(c<total%grid) for c in range(grid)]
P=len(jobs);source=P+grid;sink=source+1;g=[[] for _ in range(sink+1)]
def edge(a,b,n):
 g[a].append([b,n,len(g[b]),n]);g[b].append([a,0,len(g[a])-1,0])
for pool,n in enumerate(jobs):
 edge(source,pool,n)
 for c in sorted({(pool*8+r)%grid for r in range(8)}):edge(pool,P+c,n)
for c,n in enumerate(quota):edge(P+c,sink,n)
flow=0
while True:
 level=[-1]*len(g);level[source]=0;q=deque([source])
 while q:
  a=q.popleft()
  for b,n,_,_ in g[a]:
   if n and level[b]<0:level[b]=level[a]+1;q.append(b)
 if level[sink]<0:break
 pos=[0]*len(g)
 def push(a,n):
  if a==sink:return n
  while pos[a]<len(g[a]):
   e=g[a][pos[a]];b,c,rev,_=e
   if c and level[b]==level[a]+1:
    sent=push(b,min(n,c))
    if sent:e[1]-=sent;g[b][rev][1]+=sent;return sent
   pos[a]+=1
  return 0
 while True:
  sent=push(source,total)
  if not sent:break
  flow+=sent
allocation=[[] for _ in range(P)];used=[0]*grid
for pool in range(P):
 for b,n,rev,cap in g[pool]:
  if P<=b<P+grid and cap-n:allocation[pool].append([b-P,cap-n]);used[b-P]+=cap-n
remaining=[jobs[i]-sum(n for _,n in allocation[i]) for i in range(P)]
local=flow
for pool,n in enumerate(remaining):
 for c in range(grid):
  take=min(n,quota[c]-used[c])
  if take:allocation[pool].append([c,take]);used[c]+=take;n-=take
 assert n==0
assert used==quota and all(sum(n for _,n in a)==j for a,j in zip(allocation,jobs))
result=dict(source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),scope='last-step CPU feasibility only; assumes8contiguousqueryrows/pool as actual benchmark; not an implemented GPU planner',packets=total,local_packets_under_exact_quota=local,local_fraction=local/total,nonlocal_packets=total-local,max_packets=max(used),min_packets=min(used),owner_intervals=sum(len(x) for x in allocation),allocation=allocation)
(p/'pool_capacity_model.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='allocation'})
