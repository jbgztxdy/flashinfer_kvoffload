"""Check private shared-memory spans/packing, not simulated execution or timing."""
from collections import Counter
import hashlib
import json
from pathlib import Path

p=Path(__file__).resolve().parent
offset=lambda row,col:row*2+(col^((row//4)&1))
words=[]
for warp in range(4):
    for lane in range(32):
        o=offset(warp*16+lane//4,0)
        words += [o*4+lane%4,(o+16)*4+lane%4,(o^1)*4+lane%4,((o^1)+16)*4+lane%4]
assert Counter(words)==Counter(range(512))
bank_groups=[]
for row0,col in [(0,0),(8,0),(0,1),(8,1)]:
    banks=[(offset(row0+row,col)*4+word)%32 for row in range(8) for word in range(4)]
    assert sorted(banks)==list(range(32))
    bank_groups.append(dict(row_start=row0,column=col,word_banks=banks))
ckv=[];pe=[]
for tid in range(256):
    row,vector=divmod(tid,16)
    ckv += [row*64+chunk*16+vector for chunk in range(4)]
    if vector<8:pe.append(row*8+vector)
assert Counter(ckv)==Counter(range(16*64))
assert Counter(pe)==Counter(range(16*8))
result=dict(complete=True,structural_layout_only=True,kernel_performance_claim=False,
    source_sha256=hashlib.sha256((p/'half_ring.inc').read_bytes()).hexdigest(),
    p_half=dict(rows=64,columns=16,bytes=2048,unique_uint32_writes=len(set(words)),no_overlap_or_holes=True,
                offset='uint4_offset=row*2+(column^((row/4)&1))',ldmatrix_8row_word_bank_groups=bank_groups),
    half_loader=dict(threads=256,rows=16,threads_per_row=16,ckv_uint4s=len(ckv),pe_uint4s=len(pe),no_overlap_or_holes=True),
    storage=dict(query_nope=65536,alignment_padding=128,ckv_five_halves=81920,kpe_five_halves=10240,cross_warpgroup_state=512,
        dynamic_bytes=158336,prior_static_bytes=2304,total_if_static_unchanged=160640),
    limits='Address enumeration proves packing and bounds, not actual instruction bank transactions, cp.async ownership or performance. CUDA numerical/cache/Graph tests and sanitizer evidence remain required.')
(p/'layout_review.json').write_text(json.dumps(result,indent=2)+'\n')
print('LAYOUT_SPANS_AND_PACKING_VERIFIED')
