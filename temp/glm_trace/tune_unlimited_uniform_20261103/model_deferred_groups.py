"""Conservative async-group state check; no GPU/math/performance claim."""
import hashlib
import json
import random
from pathlib import Path

p=Path(__file__).resolve().parent
rng=random.Random(2026110409)
cases=0
wait_one=0
service_steps=0
for packets in range(1,65):
 for tiles in range(1,18):
  for trial in range(16):
   queue=[]
   local_ready=set()
   issued=set()
   retired=set()
   ready=set()
   meta=[0,1]
   payload=None
   done=0
   deferred=False
   def wait(n):
    global payload
    while len(queue)>n:
     kind,key=queue.pop(0)
     if kind=='local':local_ready.add(key)
     else:
      assert payload is None,(payload,key)
      payload=key
   def far(key):
    assert key<packets and key not in issued
    assert meta[key%2]==key,(meta,key)
    assert payload is None
    assert not any(kind=='far' for kind,_ in queue)
    issued.add(key)
    queue.append(('far',key))
   def local(key):queue.append(('local',key))
   def flush():
    global deferred
    issue=deferred and done<packets
    if issue:far(done)
    deferred=False
    return issue
   def retire(key):
    global payload
    if key>=packets:return
    assert payload==key and meta[key%2]==key,(payload,key,meta)
    assert key not in retired
    retired.add(key)
    payload=None
    ready.add(key)
    meta[key%2]=key+2
   def step():
    global done,service_steps
    flush()
    wait(0)
    if done<packets:
     retire(done)
     done+=1
     if done<packets:far(done)
    service_steps+=1
   far(0)
   hi=tiles-1
   # Initial tile waits may retire own packets before initial local loads.
   for _ in range(rng.randrange(4)):step()
   for t in (hi,hi-1):
    if t>=0:local(t)
   newest=False
   while hi>=1:
    wait(1 if newest else 0)
    wait_one+=int(newest)
    assert hi in local_ready and hi-1 in local_ready,(tiles,hi,queue,newest)
    # The QK pair runs here, then waits before consuming remote payload.
    wait(0)
    if done<packets:
     retire(done)
     done+=1
     deferred=True
    for nxt in (hi-2,hi-3):
     if nxt>=0:
      # Missing local tile may require any number of producer progress steps.
      for _ in range(rng.randrange(4)):step()
      local(nxt)
    newest=flush() and hi-3>=0
    hi-=2
   wait(0)
   if hi==0:assert 0 in local_ready
   while done<packets:step()
   flush()
   assert issued==retired==ready==set(range(packets))
   assert not queue and payload is None
   cases+=1

out=dict(complete=True,cases=cases,wait_group_one=wait_one,service_steps=service_steps,
         header_sha256=hashlib.sha256((p/'deferred_packet.cuh').read_bytes()).hexdigest(),
         coverage='1..64 packets; 1..17 KV tiles; 16 deterministic randomized progress schedules each; odd tails; FIFO worst-case completion',
         limitation='serial abstract async groups only, not CUDA memory ordering/races, math/LRU or timing')
(p/'deferred_groups_model.json').write_text(json.dumps(out,indent=2)+'\n')
print(out)
