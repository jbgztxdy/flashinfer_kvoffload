from pathlib import Path
import random,json,hashlib
p=Path(__file__).resolve().parent;rng=random.Random(20261104);cases=[]
for length in [0,1,31,32,65,255,256,511,512,1024,2047,2048]:
 for repeat in range(32):
  #0padding,1local/cache,2far duplicate,3far owner; every fresh bit is a far bit.
  category=[rng.randrange(4) for _ in range(length)]+[0]*(2048-length)
  masks=[[[sum(1<<lane for lane in range(32) if (category[i*256+w*32+lane] in wanted)) for i in range(8)] for w in range(8)] for wanted in [{3},{2,3},{1}]]
  old=[[],[],[]];new=[[],[],[]];totals=[[],[],[]]
  for kind in range(3):
   counts=[sum(m.bit_count() for m in masks[kind][w]) for w in range(8)]
   prefix=[sum(counts[:w]) for w in range(8)];totals[kind]=counts
   atomic_cursor=0
   for w in range(8):
    oldfirst=atomic_cursor;atomic_cursor+=counts[w];newfirst=prefix[w]
    for i,m in enumerate(masks[kind][w]):
     for lane in range(32):
      if(m>>lane)&1:
       old[kind].append((i*256+w*32+lane,oldfirst+(m&((1<<lane)-1)).bit_count()))
       new[kind].append((i*256+w*32+lane,newfirst+(m&((1<<lane)-1)).bit_count()))
     oldfirst+=m.bit_count();newfirst+=m.bit_count()
   assert old[kind]==new[kind]
   assert sorted(rank for _,rank in new[kind])==list(range(sum(counts)))
  assert sum(totals[1])+sum(totals[2])==sum(c!=0 for c in category)
  cases.append(dict(length=length,counts=[sum(c) for c in totals],mapping_sha256=hashlib.sha256(repr(new).encode()).hexdigest()))
(p/'cta_row_counts_model.json').write_text(json.dumps(dict(cases=cases,scope='CPU exact prefix/rank equivalence to valid old warp0..7atomic serialization given same pageCAS winners; notGPU race/performance proof'),indent=2)+'\n');print(len(cases),'exact row count/rank cases passed')
