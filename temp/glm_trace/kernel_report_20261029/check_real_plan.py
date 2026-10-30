"""Compare optimized and independent serial LRU plans on complete real trace windows."""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch

DEST=Path(__file__).resolve().parent
ROOT=DEST.parents[2]
torch.set_num_threads(4)
for source,target,extra in (("plans.cpp","plans.so",["-fopenmp"]),
                            ("plans_reference.cpp","plans_reference.so",[])):
    subprocess.run(["g++","-O3","-shared","-fPIC","-std=c++17",*extra,
                    str(DEST/source),"-o",str(DEST/target)],check=True)


def run_plan(lib,pages,mtp,near):
    steps,requests,_,topk=pages.shape
    arrays=[np.empty_like(pages),np.empty(pages.size,np.int64),np.empty(pages.size,np.int64),
            np.empty(steps+1,np.int64),np.empty((steps,4),np.int64)]
    fn=ctypes.CDLL(str(DEST/lib)).make_plan
    fn.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*7;fn.restype=ctypes.c_int64
    start=time.perf_counter()
    used=fn(*[ctypes.c_void_p(a.ctypes.data) for a in [pages,*arrays]],
            steps,requests,mtp,topk,65536,near,4096)
    elapsed=time.perf_counter()-start
    arrays[1]=arrays[1][:used];arrays[2]=arrays[2][:used]
    return arrays,elapsed


capture=torch.load(ROOT/"temp/glm_trace/assistant64k/doc1.pt",map_location="cpu",weights_only=False)
base=capture["selections"][:4,0].numpy().astype(np.int32)
starts=[int(capture["group_starts"][0])]*len(base)
records=[]
for mtp in (1,8):
    requests=len(base);near=requests*65536;steps=1024//mtp
    indices=base.reshape(requests,steps,mtp,2048).transpose(1,0,2,3).copy()
    newest=np.array(starts).reshape(1,requests,1,1)+np.arange(1024).reshape(steps,1,mtp,1)
    owner=np.arange(requests,dtype=np.int32).reshape(1,requests,1,1)*65536
    pages=np.where(indices<newest,near+owner+indices,owner+indices).astype(np.int32)
    old,serial_seconds=run_plan("plans_reference.so",pages,mtp,near)
    new,parallel_seconds=run_plan("plans.so",pages,mtp,near)
    equal=[bool(np.array_equal(a,b)) for a,b in zip(old,new)]
    assert all(equal),(mtp,equal)
    # Every baseline remap resolves to its selected page after the actual per-step copies.
    payload=np.full(requests*4096+requests*mtp*2048,-1,np.int32)
    for t in range(steps):
        a,b=new[3][t:t+2]
        payload[new[2][a:b]-near]=new[1][a:b]+near
        original=pages[t].reshape(-1);mapped=new[0][t].reshape(-1)
        far=original>=near
        assert np.array_equal(payload[mapped[far]-near],original[far])
        assert np.array_equal(mapped[~far],original[~far])
    records.append(dict(mtp=mtp,requests=requests,trace_positions=1024,steps=steps,topk=2048,
                        all_arrays_equal=equal,payload_all_steps_verified=True,
                        copied_pages=int(new[3][-1]),serial_seconds=serial_seconds,
                        parallel_seconds=parallel_seconds))
    print(json.dumps(records[-1]),flush=True)
result=dict(passed=True,cases=records,sources={p:hashlib.sha256((DEST/p).read_bytes()).hexdigest()
            for p in ("plans.cpp","plans_reference.cpp","check_real_plan.py")})
(DEST/"planner_real_trace_check.json").write_text(json.dumps(result,indent=2)+"\n")
