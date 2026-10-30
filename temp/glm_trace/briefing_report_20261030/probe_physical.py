"""Post-check physical bandwidth on the same A100 pair; no changes to measured matrix."""
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
DEST=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/"temp/glm_trace/kernel_report_20261029"),str(ROOT/"temp/h800")]
from resources import require_exclusive
import torch
import tiers

lock=(ROOT/"temp/glm_trace/kernel_report_20261029/gpu01.lock").open("w")
fcntl.flock(lock,fcntl.LOCK_EX)
before=require_exclusive()
os.environ.pop("FLASHINFER_FAR_EMU",None)
torch.set_num_threads(8);torch.cuda.set_device(0);torch.manual_seed(42)
result=dict(timestamp=time.strftime("%FT%T%z"),resources_start=before,settings=[])
rows=1<<20
for name,tier in (("nvlink","peer"),("pcie","host")):
    kv=tiers.TieredKV(rows,rows,rows,0,tier,1)
    main,alias=kv.main(),kv.alias()
    for begin in range(0,2*rows,1<<18):main[begin:begin+(1<<18)].normal_()
    src=main[rows:2*rows];target=alias[rows:2*rows]
    ids=torch.randperm(rows,device="cuda",dtype=torch.int64)+rows
    times={}
    for method in ("dma_contiguous","sm_gather_random"):
        def operation():
            if method=="dma_contiguous":target.copy_(src)
            else:torch.index_select(main,0,ids,out=target)
        operation();torch.cuda.synchronize()
        samples=[]
        for _ in range(9):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            torch.cuda._sleep(4_000_000)
            start.record();operation();end.record();end.synchronize()
            samples.append(start.elapsed_time(end)*1000)
        times[method]=dict(samples_us=samples,median_GBps=rows*1152/statistics.median(samples)/1000)
    torch.testing.assert_close(target[:32],main.index_select(0,ids[:32]),rtol=0,atol=0)
    result["settings"].append(dict(name=name,tier=tier,rows=rows,bytes=rows*1152,
        measurements=times,reference_GBps=max(t["median_GBps"] for t in times.values()),
        residency=kv.residency,payload_sample_check=True))
    del main,alias,src,target,ids
    torch.cuda.synchronize();kv.close();torch.cuda.empty_cache()
result["resources_end"]=require_exclusive()
result["hardware"]=subprocess.check_output(["nvidia-smi","--id=0,1",
    "--query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.mem","--format=csv,noheader"],text=True)
result["complete"]=True
(DEST/"physical_bandwidth.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
