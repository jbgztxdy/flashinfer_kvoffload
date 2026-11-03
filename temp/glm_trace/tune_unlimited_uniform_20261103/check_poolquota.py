from pathlib import Path
import json,random
p=Path(__file__).resolve().parent;random.seed(20261104);cases=0
for grid in [1,2,24,108]:
 for repeat in range(40):
  n=random.randrange(1,129);rows=[random.randrange(0,17) for _ in range(n)];jobs=[random.randrange(0,200) if nr else 0 for nr in rows];total=sum(jobs);q,r=divmod(total,grid);remain=[q+(c<r) for c in range(grid)];lists=[[] for _ in range(grid)];spill=[];row0=0;start=0
  for nr,njobs in zip(rows,jobs):
   at=start;end=at+njobs
   while at<end:
    best=grid;cap=0
    for row in range(row0,row0+nr):
     c=row%grid
     if remain[c]>cap:best=c;cap=remain[c]
    if not cap:break
    take=min(end-at,cap);remain[best]-=take;lists[best].extend(range(at,at+take));at+=take
   spill.append((at,end));start=end;row0+=nr
  c=0
  for at,end in spill:
   while at<end:
    while c<grid and not remain[c]:c+=1
    assert c<grid
    take=min(end-at,remain[c]);remain[c]-=take;lists[c].extend(range(at,at+take));at+=take
  assert not any(remain)
  assert [len(x) for x in lists]==[q+(c<r) for c in range(grid)]
  assert sorted(x for a in lists for x in a)==list(range(total));cases+=1
(p/'poolquota_partition.json').write_text(json.dumps({'cases':cases,'exact_quotas':True,'each_packet_exactly_once':True,'includes_zero_row_pools':True,'scope':'CPU mirror, not CUDA race/liveness proof'},indent=2)+'\n');print(cases)
