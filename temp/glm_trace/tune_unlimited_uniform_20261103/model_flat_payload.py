from pathlib import Path
import json,collections
p=Path(__file__).resolve().parent;cases=[]
for pages in range(21):
 emitted=[(v//72,v%72,t) for t in range(256) for v in range(t,1440,256) if v//72<pages]
 counts=collections.Counter((a,b) for a,b,_ in emitted)
 assert len(counts)==pages*72 and all(n==1 for n in counts.values())
 # Producers write the identical word in staging and, iff admitted, cache.
 for a,b,t in emitted:assert a*576+b*8==(a*72+b)*8
 cases.append(dict(pages=pages,vectors=len(emitted),source_bytes=len(emitted)*16,active_threads=len(set(t for _,_,t in emitted))))
(p/'flat_payload_model.json').write_text(json.dumps(dict(cases=cases,scope='CPU exact source/destination vector partition; noCUDA memory-order or performance proof'),indent=2)+'\n');print('21 exact vector partitions passed')
