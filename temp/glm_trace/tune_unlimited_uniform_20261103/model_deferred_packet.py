"""Ticket/metadata state model only; not a CUDA synchronization proof."""
import hashlib
import json
import random
from pathlib import Path

p=Path(__file__).resolve().parent
rng=random.Random(20261104)
cases=0
transitions=0
for count in range(130):
    for trial in range(64):
        done=0
        current=0
        deferred=False
        metadata=[0,1]
        issued=set()
        stored=set()
        ready=set()
        def issue(ticket):
            if ticket>=count:return
            assert metadata[ticket%2]==ticket,(metadata,ticket)
            assert ticket not in issued
            issued.add(ticket)
        def flush():
            global deferred,transitions
            if deferred:issue(done)
            deferred=False
            transitions+=1
        def store_publish(ticket):
            assert ticket in issued and ticket not in stored
            assert metadata[ticket%2]==ticket
            stored.add(ticket)
            ready.add(ticket)
            metadata[ticket%2]=ticket+2
        def step():
            global current,done,transitions
            flush()
            current=done
            if current<count:
                store_publish(current)
                issue(current+1)
                done=current+1
            transitions+=1
        issue(0)
        # A math service retires a packet and defers its successor. Each
        # local-tile boundary may request progress zero or multiple times.
        while done<count:
            if rng.randrange(2):
                step()
            else:
                current=done
                store_publish(current)
                done=current+1
                deferred=True
                for boundary in range(2):
                    for _ in range(rng.randrange(4)):
                        if done<count:step()
                flush()
            transitions+=1
        flush()
        assert issued==stored==ready==set(range(count))
        cases+=1
header=p/'deferred_packet.cuh'
out=dict(complete=True,cases=cases,transitions=transitions,
         coverage='serial ticket transitions, two metadata slots, 0..129 packets, two local tile boundaries with 0..3 progress steps',
         limitation='does not model CUDA thread races, memory ordering, attention math or performance',
         header_sha256=hashlib.sha256(header.read_bytes()).hexdigest())
(p/'deferred_packet_model.json').write_text(json.dumps(out,indent=2)+'\n')
print(out)
