"""Adversarial interleaving model, not a CUDA memory ordering proof."""
from pathlib import Path
import random,json
p=Path(__file__).resolve().parent
cases=0;max_ticks=0
for workers in [1,2,8,108]:
 for packets in [0,1,3,20,109,501]:
  for seed in range(20):
   rng=random.Random(seed+workers*1000+packets);tasks=[]
   for w in range(workers):
    # Skew and cyclic consumer overlap; arbitrary per-worker computation length.
    tasks.append([rng.randrange(packets) for _ in range(rng.randrange(0,40))] if packets else [])
   status=[0]*packets #0unclaimed,1inflight,2payloadready
   copied=[0]*packets;owner=[None]*packets;current=[None]*workers;delay=[0]*workers;pos=[0]*workers;cursor=0;ticks=0
   while not(all(x==2 for x in status) and all(pos[w]==len(tasks[w]) for w in range(workers))):
    # Fair shuffled rounds. Finished consumers remain participants in drain.
    order=list(range(workers));rng.shuffle(order)
    for w in order:
     ticks+=1
     if current[w] is not None:
      delay[w]-=1
      if delay[w]<=0:
       k=current[w];assert status[k]==1 and owner[k]==w
       copied[k]+=1;status[k]=2;current[w]=None
      continue
     if pos[w]<len(tasks[w]):
      k=tasks[w][pos[w]]
      if status[k]==2:pos[w]+=1;continue
      if status[k]==0:status[k]=1;owner[k]=w;current[w]=k;delay[w]=rng.randrange(1,5);continue
     # Never wait passively on another producer: claim an unassigned fallback
     # packet; cursor scans each candidate once. No payload is marked ready early.
     while cursor<packets and status[cursor]!=0:cursor+=1
     if cursor<packets:
      k=cursor;cursor+=1;assert status[k]==0
      status[k]=1;owner[k]=w;current[w]=k;delay[w]=rng.randrange(1,5)
    assert ticks<max(1,packets+sum(map(len,tasks))+workers)*workers*20
   assert copied==[1]*packets and all(current[w] is None for w in range(workers))
   cases+=1;max_ticks=max(max_ticks,ticks)
result=dict(cases=cases,max_ticks=max_ticks,each_payload_copied_once=True,all_consumers_finish=True,scope='fair sequentially consistent randomized interleaving model, no attention math, no CUDA release/acquire or timing proof',requirements=['never abandon an in-flight claimed packet','ready published only after actual full stage/cache payload stores','completed compute CTAs participate until all unique packets drain','failed claim does not block unrelated fallback progress'])
(p/'demand_packet_model.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
