"""Independent CPU oracle for actual baseline cache payload and LRU remapping."""
import ctypes
import json
from pathlib import Path
import numpy as np

DEST=Path(__file__).resolve().parent
fn=ctypes.CDLL(str(DEST/"plans.so")).make_plan
fn.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*7
fn.restype=ctypes.c_int64
rng=np.random.default_rng(20261030)
records=[]
for mtp in (1,8):
    steps,requests,topk,context,near,capacity=31,3,9,64,192,11
    raw=rng.integers(0,context,(steps,requests,mtp,topk),dtype=np.int32)
    for t in range(1,steps):raw[t,:,:,:4]=raw[t-1,:,:,:4]
    far=rng.random(raw.shape)<.8
    owners=np.arange(requests,dtype=np.int32).reshape(1,requests,1,1)
    pages=np.where(far,near+owners*context+raw,owners*context+raw).astype(np.int32)
    remap=np.empty_like(pages)
    src=np.empty(pages.size,np.int64);dst=np.empty_like(src)
    offsets=np.empty(steps+1,np.int64);counts=np.empty((steps,4),np.int64)
    used=fn(*[ctypes.c_void_p(a.ctypes.data) for a in (pages,remap,src,dst,offsets,counts)],
            steps,requests,mtp,topk,context,near,capacity)
    slots=[[-1]*capacity for _ in range(requests)]
    stamps=[[-1]*capacity for _ in range(requests)]
    payload={}
    expected_used=0
    for t in range(steps):
        expected_counts=np.zeros(4,np.int64);expected=[];overflow=0
        for r in range(requests):
            refs=[int(v) for v in pages[t,r].flat if v>=near]
            need=sorted(set(refs));before={p:i for i,p in enumerate(slots[r]) if p>=near}
            missing=[p for p in need if p not in before]
            expected_counts+=np.array([len(refs),len(need),len(missing),sum(p in before for p in refs)])
            for p in need:
                if p in before:stamps[r][before[p]]=t
            victims=sorted((i for i in range(capacity) if stamps[r][i]!=t),key=lambda i:(stamps[r][i],i))
            for j,p in enumerate(missing):
                if j<len(victims):
                    i=victims[j];slots[r][i]=p;stamps[r][i]=t;destination=near+r*capacity+i
                else:
                    destination=near+requests*capacity+overflow;overflow+=1
                expected.append((p-near,destination))
        a,b=map(int,offsets[t:t+2]);assert a==expected_used
        assert list(zip(src[a:b],dst[a:b]))==expected
        assert np.array_equal(counts[t],expected_counts)
        for source,destination in expected:payload[destination]=near+source
        for original,mapped in zip(pages[t].flat,remap[t].flat):
            assert (int(mapped) if original<near else payload[int(mapped)])==original
        expected_used=b
    assert used==expected_used
    records.append(dict(mtp=mtp,steps=steps,requests=requests,capacity=capacity,
                        copied_pages=int(used),verified=True))
result=dict(passed=True,cases=records)
(DEST/"planner_check.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
