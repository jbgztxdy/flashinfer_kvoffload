from pathlib import Path
import json,random
p=Path(__file__).resolve().parent;r=random.Random(20261104);cases=0;selected=0
for _ in range(50000):
    host=r.randrange(2049);start=r.randrange(2049);length=r.randrange(2049-start)
    prefix=max(0,min(host,start+length)-start)
    host_tiles=(prefix+31)//32
    for tile in range((length+31)//32):
        if tile+1<host_tiles:
            assert start+tile*32+31<host
            assert tile*32+31<length
            selected+=1
        cases+=1
(p/'stage_tile_uniform_model.json').write_text(json.dumps(dict(seed=20261104,tile_cases=cases,selected_staged_only_tiles=selected,scope='CPU integer guard proof across random host prefix/kv_start/kv_len; notGPU payload or performance proof'),indent=2)+chr(10))
print(cases,'tiles tested',selected,'selected staged-only')
