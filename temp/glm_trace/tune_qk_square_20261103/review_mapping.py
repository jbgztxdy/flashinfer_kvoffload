import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent
writers={};readers={}
for warp in range(4):
 for lane in range(32):
  for hp in range(2):
   for cp in range(2):
    for j in range(2):
     for c in range(4):
      key=(warp//2,warp%2*32+hp*16+j*8+lane//4,cp*16+lane%4*2+c//2*8+c%2)
      assert key not in writers;writers[key]=(warp,lane,hp*2+cp,j*2+c//2*4+c%2)
for group in range(2):
 for warp in range(4):
  for lane in range(32):
   for tile in range(2):
    for j in range(2):
     for c in range(4):
      key=(tile,warp*16+j*8+lane//4,group*16+lane%4*2+c//2*8+c%2)
      assert key not in readers;readers[key]=(group,warp,lane,j*2+c//2*4+c%2)
assert writers.keys()==readers.keys() and len(writers)==4096
r=dict(complete=True,unique_scores=len(writers),coverage='2tiles x64heads x32KV; one writer/reader per coordinate',source_sha256=hashlib.sha256((p/'square.inc').read_bytes()).hexdigest(),scope='Integer fragment-coordinate check, not GPU numerics or a proof of synchronization.')
(p/'mapping_review.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
