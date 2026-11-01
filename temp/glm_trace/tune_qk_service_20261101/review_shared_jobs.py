"""Check chunk ownership, exact far bytes and tail packing against a slot-set oracle."""
import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent;records=[]
for grid in [1,3,108]:
 for count in [0,1,7,8,63,64,65,71,127,128,129,4095,4096,4097,65537]:
  seen=[0]*count;released=[];far_rows=jobs=0
  for cta in range(grid):
   j=0
   while True:
    first=(cta+(j//10)*grid)*64+(j%10)*7
    if first>=count:break
    rows=min(7,64-(j%10)*7,count-first)
    assert 1<=rows<=7 and first//64==(first+rows-1)//64
    indices=[i*128+tid for i in range(4) for tid in range(128) if i*128+tid<rows*72]
    assert sorted(indices)==list(range(rows*72)) and max(indices)<504
    for slot in range(first,first+rows):seen[slot]+=1
    if j%10==9 or first+rows>=count:released.append(first//64)
    far_rows+=rows;jobs+=1;j+=1
  assert all(x==1 for x in seen) and far_rows==count
  assert sorted(released)==list(range((count+63)//64))
  records.append(dict(grid=grid,unique=count,jobs=jobs,model_far_rows=far_rows,model_far_bytes=far_rows*1152,released_chunks=len(released)))
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
(p/'shared_job_review.json').write_text(json.dumps(dict(complete=True,structural_only=True,performance_claim=False,
 source_sha256=sha(Path(__file__)),service_source_sha256=sha(p/'service_shared.inc'),candidate_sha256=sha(p/'qks_shared.cuh'),records=records,
 limitation='Finite independent slot-set/byte-count oracle for irregular packing. Not a compiled CUDA/FAR-runtime,ownership/sanitizer or performance proof.'),indent=2)+'\n')
print('SHARED_JOBS_REVIEWED',len(records))
